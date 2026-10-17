# 11 · Strong Drafts Need Compact Memories：MASW 方法与研究问题

阅读：2026-09-09；已读正文、附录实际启用内容、方法图与 attention mask，未运行代码。按本轮要求压缩背景，保留方法完整公式、实现条件、关键实验及创新筛选依据。

## 1. 来源与核心判断

论文：**Strong Drafts Need Compact Memories: Long-Context Speculative Decoding with Compressed KV Cache**，Tong Yuan、Chengxi Liao、Zeyi Wen；[arXiv:2608.30252v1](https://arxiv.org/abs/2608.30252v1)，2026-08-31，页面标注 EMNLP 2026 Findings。

本地：[论文目录](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Strong Drafts Need Compact Memories- Long-Context Speculative Decoding with Compressed KV Cache>)。`acl_latex.tex` 引入 `src/introduction.tex` 和 `src/appendix.tex`；附录 Qwen2.5 表位于 `comment` 环境，**不作为正式跨模型结果**。

方法名 **Memory-Augmented Sliding-Window（MASW）Drafting**：让较强的独立小模型只读取“sink + 最近原始 KV + 历史记忆 slots”。slots 由可训练的 K/V 投影生成；Drafter 骨干冻结。目标模型保留完整原始 KV，沿用标准验证。

**和 AsymSpec 的根本差别**：AsymSpec 压缩 Verifier 并主动改变输出策略；MASW 压缩 Drafter，在目标验证正确的条件下保持完整 Verifier 的输出分布。这里的 memory 是连续 KV 状态，不是动作 pattern 库、经验文字或工具结果缓存。

标记：**【原文】**指论文方法/数据；**【展开】**指实现所需数学；**【审查】**指推导及适用边界；**【待确认】**不补成作者事实。

## 2. 为什么选择“强草稿 + 短 KV”

【原文】设每轮提出 $L_{\mathrm{draft}}$ 个 token，平均接受 $L_{\mathrm{acc}}$ 个，再输出一个 target 修正/bonus token：

$$
L_{\mathrm{emit}}=L_{\mathrm{acc}}+1,
\qquad
\mathrm{Speedup}=\frac{L_{\mathrm{acc}}+1}
{L_{\mathrm{draft}}\,t_d/t_t+1}.
$$

$t_d,t_t$ 分别是 Drafter/Target 的单 token 延迟；分母的 1 假设一次批量验证约等于一次 target 单 token 前向。它是忽略终止与额外调度的近似，不是所有服务环境的精确计时公式。

【原文】单 token 解码成本近似：

$$
T_{\mathrm{decode}}\approx\alpha d^2+\beta Ld,
$$

$d$ 为 hidden size，$L$ 为前缀长度；第一项代表权重相关成本，第二项代表随历史增长的 KV 访问。$\alpha,\beta$ 吸收架构、层数、精度及硬件因素，并非通用常数。

强 Drafter 往往接受更多 token，但长 KV 使每一步变慢。MASW 不只追求接受长度最大，而是允许接受略降，换更低 drafting 成本。轻量 EAGLE 在作者测试中的不足是经验现象，不能从少数配置推成所有轻量草稿必然无法处理长上下文。

【审查：更完整的收益式】令实际验证时间为 $T_v$，每轮 slot、同步及回滚新增时间为 $T_o$，则：

$$
\mathrm{Speedup}\approx
\frac{(L_{\mathrm{acc}}+1)t_t}
{L_{\mathrm{draft}}t_d+T_v+T_o}.
$$

只有分母小于对应 AR 生成时间才加速。新增 slot 的代价若已包含在测得 $t_d$ 中，不要再次加进 $T_o$。

## 3. 三部分 memory 与压缩边界

### 3.1 缓存结构

【原文】在原始 token 步 $t$：

$$
\mathcal M_t=\mathcal M_{\mathrm{sink}}
\cup\mathcal M_{\mathrm{local},t}
\cup\mathcal M_{\mathrm{slot},t}.
$$

前三类分别是：

$$
\mathcal S=\{1,\ldots,S\},\qquad
\mathcal M_{\mathrm{sink}}=\{(K_i,V_i):i\in\mathcal S\},
$$

$$
\mathcal M_{\mathrm{local},t}
=\{(K_i,V_i):\max(S+1,t-W+1)\leq i\leq t\},
$$

$$
\mathcal M_{\mathrm{slot},t}
=\{(K_{g_m}^{(l)},V_{g_m}^{(l)})\}_{\text{已生成的 }m,\ \text{所有层 }l}.
$$

$S$ 是始终保留的 sink token 数，$W$ 是最近非 sink 原始 token 窗口；slot 是学习得到的多层 KV。最后一式是原文文字的展开。

“Exact local KV”表示不压缩这部分原始 token 的缓存，**不是说它数值上等于 full-KV Drafter 的缓存**：普通 token 从 prefill 起就用压缩可见范围，隐藏状态会与全注意力骨干不同。

### 3.2 每 r 个原始 token 写一个 slot

【原文】用排除 sink 的计数定义边界 $\tau_m=mr$，每 $r$ 个 raw token 插入一个记忆 token $g_m$：

$$
g_m\leftarrow f_\theta\bigl(
\mathcal M_{\mathrm{local},\tau_m},
\mathcal M_{\mathrm{sink}},
\mathcal M_{\mathrm{slot},<m}\bigr).
$$

$f_\theta$ 是通过现有 Transformer 全部层的 masked forward，含冻结骨干及可训练 adaptor。新 slot 能读旧 slots，因此不是独立地对每 $r$ 个 token 做平均池化，而是持续读取旧记忆、写入新的压缩状态。

位置编码规则：**slot 与它后面紧邻的 raw token 使用同一 RoPE position ID**，raw token 原 ID 不移动。两个物理位置可有重复 RoPE ID，不能把“位置 ID 相同”当作 attention 可见性相同。实现还要统一文中非 sink 边界计数与全序列下标。

### 3.3 它不是常数大小的记忆

【原文/计数展开】若已处理 $t$ 个 raw token，近似 slot 数为 $\lfloor(t-S)/r\rfloor$，忽略未满窗口和边界偏移：

$$
N_{\mathrm{cache}}(t)\approx S+W+\left\lfloor\frac{t-S}{r}\right\rfloor,
\qquad
\frac{N_{\mathrm{cache}}(t)}t\longrightarrow\frac1r.
$$

所以 nominal $r\times$ 只指历史 materialization 频率，不等于全缓存恰好缩小 $r$ 倍；rollback buffer 还需额外空间。历史 slots 持续累积，长期仍为 $O(t/r)$，不是无界序列下 $O(1)$。

【展开】若 $n_l$ 层、每层 $n_{kv}$ 个 KV heads、head 维数 $d_h$、每元素 $b_e$ bytes，则纯 KV 容量近似：

$$
B_{\mathrm{KV}}=2n_l n_{kv}d_h b_e N_{\mathrm{cache}}.
$$

这是 KV 字节数，不包含骨干/adaptor 权重、mask、logits 和临时激活；后面的峰值表不能直接当作本式测量。

## 4. Slot 如何产生：投影、mask 与训练

### 4.1 Mirrored K/V projections

【原文全部投影公式】对第 $l$ 层：

$$
\begin{aligned}
K_r^{(l)}&=H_r^{(l-1)}W_{K,r}^{(l)},&
V_r^{(l)}&=H_r^{(l-1)}W_{V,r}^{(l)},\\
K_g^{(l)}&=H_g^{(l-1)}W_{K,g}^{(l)},&
V_g^{(l)}&=H_g^{(l-1)}W_{V,g}^{(l)}.
\end{aligned}
$$

$r/g$ 下标在这里分别表示 raw/slot，和 slot interval $r$ 是不同含义。普通 token 使用原 K/V 投影；slot 使用独立镜像 K/V 投影，初始化为相应骨干权重：

$$
W_{K,g}^{(l)}\gets W_{K,r}^{(l)},\qquad
W_{V,g}^{(l)}\gets W_{V,r}^{(l)}.
$$

只有镜像矩阵训练，原骨干冻结。不是每个 slot 单独训练一套权重：同一层的 slots 共享该层镜像投影。若每层 projection 输出维度为 $d_{kv}^{(l)}$，忽略 bias，adaptor 参数量为：

$$
N_{\mathrm{adaptor}}=2\sum_{l=1}^{n_l}d^{(l)}d_{kv}^{(l)}.
$$

这里的参数计数是展开；原文没有列精确训练参数/GPU-hour，不能把“只训 adaptor”理解为训练免费，仍需穿过冻结骨干传播梯度。

### 4.2 Mask 决定写入的信息

【原文】普通 raw token 只看 sink、已可用 slots、有限 local window；slot 读取旧 slots、它负责的局部单元和邻近边界 raw tokens。目标 verifier 不加入这些 slots。

【展开：统一 masked attention 写法】以 $\mathcal A(u)$ 表示查询位置 $u$ 的合法读取集合：

$$
M_{uv}=\begin{cases}0,&v\in\mathcal A(u),\\-\infty,&\text{其他},\end{cases}
\qquad
\operatorname{Attn}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

slot 的读取集合由正文上述三类来源及附录 mask 图决定。原文没有给完整逐下标的边界谓词、特殊 slot 输入 embedding 的配置与全部实现代码；**不要把图中的邻近 token 随意扩成未来 raw token**，造成训练时可见、在线无法获得的泄漏。

Prefill 把 slots 插入扩展序列，在一趟结构化 masked forward 中同时产生各层 KV；不是先运行 full attention，再事后平均/裁剪完整 KV。实现训练 mask 与在线写入可见范围一致，才有意义。

### 4.3 训练目标

【原文】两阶段：RedPajama 2B token 继续预训练，文档后加 EOS；再用 LongAlpaca、BookSum 做 SFT，训练序列截到 8K。损失为：

$$
\mathcal L=-\sum_t\log p(x_t\mid\mathcal D_t,x_{<t}),
\qquad
\mathcal D_t=(\mathcal M_{\mathrm{sink}},
\mathcal M_{\mathrm{local},t},\mathcal M_{\mathrm{slot},t}).
$$

只对 raw-token 预测计 loss，slots 不作为预测标签。这里 $\mathcal D_t$ 应理解为预测 $x_t$ 时已可用的前缀状态，不能先读 $x_t$ 再预测它；$x_{<t}$ 表示条件历史，不表示暗中又让 attention 访问全部 raw KV。

训练让 slots 保留对后续 token 预测有用的信息，**不要求重建每个被丢弃的 KV，也不是训练自然语言摘要**。原文没有采用明确的 target-logit KL 蒸馏或直接优化接受长度；骨干 next-token 能力与目标分布匹配仍有差别。

【待确认】slot embedding/标签移位、SFT 是否仅对 answer span 计损失、优化器/学习率/总 SFT token、边界采样与 stop-gradient 细节未给全。原文损失足以描述目标，但不足以直接重建同一训练结果。

## 5. 与标准 SD 集成及正确性

### 5.1 Prefill、在线写入和 rollback

【原文】Target prefill 保留完整原始 KV；Drafter prefill 建 compact memory，两者可并发。生成期间每到 slot 边界就在 Drafter forward 中 materialize 新 slot，**不等待 target 验证**；过旧 raw KV 被释放，但保留回滚所需窗口。

拒绝后删除未接受 raw token 的 KV 和最后接受位置之后创建的 slots，恢复 local 上下文。主设置 $W=S=128$，raw-KV rollback window 为16，主实验每轮草稿长度5。

关键实现点是提前 eviction 不能删除回退仍需的 raw KV。应为草稿步记录逻辑位置、slot 写入边界及缓存检查点，回滚后恢复压缩计数。固定16对主配置可用，不自动覆盖更深草稿和所有边界规则。

### 5.2 标准验证公式补全

【补充原理；原文使用标准 SD】令 $q_i(v)$ 为 MASW 实际产生草稿时的条件分布，$p_i(v)$ 为 full-KV target 的条件分布。对 $d_i\sim q_i$：

$$
A_i=\min\left(1,\frac{p_i(d_i)}{q_i(d_i)}\right),\quad
u_i\sim U[0,1],\quad u_i\leq A_i\Rightarrow\mathrm{accept}.
$$

首次拒绝时从：

$$
p_i^{\mathrm{res}}(v)=
\frac{[p_i(v)-q_i(v)]_+}{\sum_w[p_i(w)-q_i(w)]_+}
$$

取修正 token；全部接受则从 target 的下一位置分布取 bonus。原理是每个 token 的保留概率质量与残差相加回到 $p_i$。参考：[标准 SD Algorithm 1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)。

**论文实验均为 greedy**：逐位置核对 $d_i=\arg\max p_i$，只接受连续正确前缀，首次不匹配发出 target argmax。无需在本实验中执行随机 residual sampler；上式说明一般保分布机制的条件。

压缩使 $q$ 变差不会自动破坏正确性，只会更易拒绝。必须保证用的是实际 proposal 的概率、target 始终读正确原始前缀、slots 不进入 target/用户输出、拒绝后缓存一致。原文没有做独立随机采样等价实验，不把理论机制和实现验证混为一谈。

### 5.3 最小实现流程

```text
offline:
    freeze draft backbone
    copy its per-layer K/V projections to trainable slot projections
    construct causal sink/local/slot training masks
    train raw-token next-token loss: pretrain -> SFT

prefill:
    target: full original prompt -> full KV
    draft: raw prompt + scheduled slots -> structured masked KV
    retain sinks, local window, slots and rollback storage

each speculative iteration:
    save draft logical position and cache checkpoint
    propose K raw tokens (main experiment K=5)
        at every r-token boundary: forward slot, append its per-layer KV
        evict raw KV only when not needed for rollback
    target verifies raw candidates using full original-token KV
    emit accepted prefix + target correction/bonus (unless EOS)
    discard rejected raw KV and slots created on the invalid suffix
    recover retained local window and synchronize corrected prefix
    continue with restored slot schedule
```

“Append-only”描述有效 slots 不原地改写，不意味着未验证 slots 永远不能删除；回滚本身是算法必需。为检查 causal mask，先在短序列比较训练模式与逐 token 模式，再测跨 slot 边界的拒绝，最后接长上下文服务，避免从8卡大训练起步。

## 6. 实验：保留有助于选题的数值

### 6.1 设置与基线公平性

- Target：Llama3.1-8B/70B-Instruct；独立 draft：Llama3.2-3B、Llama3.1-8B。单个 **8×H100 80GB 节点，主 latency batch=1**。
- 数据：LongBench-v1 的 GovReport、QMSum、MultiNews 摘要任务；输入8K/16K/24K/32K，最多输出1024或EOS。没有在线工具 Agent 实验。
- 主表所有投机方法草稿长度5；动机图单独用10，不能混用接受长度。
- SWA/SnapKV 是组件基线，draft与target使用相同骨干/权重，窗口1024或保留4096条KV；MASW可用更小独立模型。与full-KV同尺寸draft的比较更能隔离MASW作用；不能把胜过这两个组件等同于胜过其全部优化系统。
- EAGLE/EAGLE-3使用发布权重但采用 chain proposal；论文没有比较所有树形和长上下文特化系统的最佳配置。

### 6.2 主指标只计 decoding

$$
\mathrm{tok/s}=\frac{N_{\mathrm{output}}}{T_{\mathrm{decode}}},\qquad
\mathrm{Speedup}=\frac{\mathrm{tok/s}_{\mathrm{method}}}{\mathrm{tok/s}_{AR}}.
$$

**分母排除两边 prefill**；Tok./Iter 是正式发出的 token 数，包含 correction/bonus，不是单纯接受草稿数。

| Target及配置 | 8K加速 | 16K | 24K | 32K |
|---|---:|---:|---:|---:|
| 8B，full-KV SD 3B | 1.18× | 0.98× | 0.91× | 0.98× |
| 8B，MASW 3B，4× | 1.24× | 1.46× | 1.85× | **2.08×** |
| 8B，MASW 3B，8× | 0.90× | 1.20× | 1.51× | 1.48× |
| 70B，full-KV SD 8B | 2.28× | 2.11× | 1.89× | 2.08× |
| 70B，MASW 8B，4× | 2.89× | **3.33×** | 2.91× | 2.76× |
| 70B，MASW 8B，8× | 3.03× | 2.77× | 2.70× | 2.82× |

最高值对应不同配置/长度，不是所有设置都取得2.08×/3.33×。8B target下3B的8×压缩在8K甚至慢于AR。强压缩与更大draft都不是单调更好，需在接受和耗时之间实测。

### 6.3 “超过70%内存下降”的分母

70B target、32K前缀，表中 Extra Peak **排除冻结骨干权重，包含 adaptor、KV、mask、logits和临时缓冲**：

| Draft | 骨干权重 | Extra Peak | Tok./Iter | Draft latency |
|---|---:|---:|---:|---:|
| 3B full | 5.98 GB | 17.21 GB | 4.05 | 41.34 ms |
| 3B MASW 4× | 同上 | 4.42 GB | 3.65 | 13.94 ms |
| 3B MASW 8× | 同上 | 3.98 GB | 3.43 | 12.39 ms |
| 8B full | 14.95 GB | 18.02 GB | 4.82 | 54.92 ms |
| 8B MASW 4× | 同上 | 5.05 GB | 3.78 | 15.37 ms |
| 8B MASW 8× | 同上 | 4.55 GB | 3.77 | 14.02 ms |

例如8B full→MASW8×，Extra Peak降约74.8%，但包含骨干后的draft总峰值由32.97到19.50GB，降约40.9%，还没包括target。不能写成“整个Agent系统省了75%显存”。接受长度有损而速度显著改善，正好体现方法的收益来源。

### 6.4 Prefill是否真的免费

【原文】8B memory draft prefill在8K/16K/32K为751.6/2350.2/7992.5ms；未压缩8B为306.5/711.2/1835.2ms；70B target为2212.9/5480.3/12316.4ms。

作者给出：

$$
T_{\mathrm{prefill}}^{\mathrm{MASW}}=\max(P_d,P_t)=P_t\quad(P_d<P_t).
$$

这依赖两路确实并发且没有严重争用；仅“独立模型”不自动保证充分资源隔离。表支持其70B设置下存在隐藏空间，不能直接推广到8B target、单GPU或繁忙服务。完整任务比较应使用：

$$
T_{\mathrm{total}}=T_{\mathrm{prefill}}+T_{\mathrm{decode}}+T_{\mathrm{other}}.
$$

### 6.5 最重要的消融

- **窗口**：W从32到512，但同步令S=W；8B draft/70B target/32K/8×下128的Tok./Iter=3.77最高。因同时改变sink与window，不能单独归因于local window。
- **初始化**：镜像K/V复制原权重比随机初始化loss更低、梯度更稳。支持copy初始化，不证明所有KV几何被严格保持。
- **训练阶段**：16K下，3B/4×的PT/SFT/PT+SFT为3.70/3.23/4.15 Tok./Iter；8B/8×为3.87/3.01/5.34。两阶段在所测组合最好，但数据量/任务分布不完全等价，不能只归因于训练顺序。
- **训练长度**：固定2B-token预算，8K训练的8B/8×在32K评测为3.77，32K训练为3.58。说明该设置可外推；不是长训练上下文普遍无益，同token预算下样本数不同。

## 7. 与既有文献和候选创新的关系

| 对照 | 已经覆盖 | 不能直接声称的新意 |
|---|---|---|
| AsymSpec | 非对称上下文与小大模型协作 | MASW保持full target，与改变target策略应分开评价 |
| AgentSpec | 历史token检索和动态批预算 | MASW学习连续KV，不是原文检索；主实验也非大batch |
| Speculate with Memory / PASTE / SMC | 经验、动作模式和多步执行 | “memory”同名不代表同一机制，不能以一个概念替代另一层实现 |
| StreamingLLM / SnapKV | sink/window和选择性保留 | 只加窗口或丢旧KV已经有工作 |
| Activation Beacon / ICAE / AutoCompressor | 可学习的连续压缩表示 | slots学习压缩本身非空白；应核查移到draft端后的新增约束 |

MASW已明确把**自适应S/W/r、扩大训练范围、训练更多骨干参数、扩展到self-speculation**列为未来工作。因此简单提出“动态压缩比例”“加LoRA”“压到自投机”不足以构成已确认创新。

保留三个具体诊断候选，不现在决定选题：

1. **压缩损失是否集中于会改变接受长度的远程信息？** 固定target和prompt，比较同尺寸full draft、纯window、MASW，按远程精确字段/局部复述统计首拒绝位置、接受长度和真实draft耗时。先测next-token loss是否能预测接受收益；若简单训练/比例调优已够，放弃复杂目标。目标对齐/蒸馏仍须查相关长上下文SD工作。
2. **长轨迹中累积slots与真实资源预算何时成为新瓶颈？** 先扩长日志观察slot数、KV读成本和接受，不直接建层级memory。固定r本就O(t/r)，自适应是作者已提出方向；只有测得具体失效、简单r/window扫描无法解决才继续。
3. **收益在prefill和并发计入后还剩多少？** 固定硬件比较完整请求延迟及不同输出长度，再扫少量batch；同时保留同尺寸full draft和简单window基线。若额外prefill/竞争吞掉收益，先考虑简单启停阈值，而不是马上设计大调度器。

## 8. 复现起点与最终判断

当前材料未提供可直接运行的项目/训练权重链接；本轮不做全网仓库排查。缺失的主要信息是完整mask边界、slot初始化及label映射、训练超参数、基线GPU/并发划分、样本筛选数量、重复实验与误差条。先确认这些，再决定是否值得训练2B-token adaptor。

它的贡献是**把可学习压缩记忆放进独立Drafter，并处理增量写入和回滚，使较强Drafter的接受能力与较小KV读取成本兼得**。证据主要来自单请求长文摘要解码；严格保证来自保留标准full-target验证，既不是slot无损压缩，也不是完成了Agent动作级加速。

下一篇是第十二篇 **Speculative Uncertainty / How to Speculate about Uncertainty in Agentic Coding? A Draft-Model Gate Method**。读完后再集中筛选跨论文研究问题。

本地证据：[正文方法与实验](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Strong Drafts Need Compact Memories- Long-Context Speculative Decoding with Compressed KV Cache/src/introduction.tex>)、[附录mask与计时/基线设置](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Strong Drafts Need Compact Memories- Long-Context Speculative Decoding with Compressed KV Cache/src/appendix.tex>)、[memory写入图](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Strong Drafts Need Compact Memories- Long-Context Speculative Decoding with Compressed KV Cache/fig/EMNLP-Method.drawio.pdf>)。
