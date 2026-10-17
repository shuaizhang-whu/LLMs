# 12 · Speculative Uncertainty：方法、公式与研究问题

阅读日期：2026-09-09。论文：**How to Speculate about Uncertainty in Agentic Coding? A Draft-Model Gate Method**，Konstantin Grotov、Valentin Malykh，[arXiv:2609.05274v1](https://arxiv.org/abs/2609.05274v1)。

依据：本地 [LaTeX 正文及附录](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/How to Speculate about Uncertainty in Agentic Coding? A Draft-Model Gate Method/acl_latex.tex>)、[部署实验表](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/How to Speculate about Uncertainty in Agentic Coding? A Draft-Model Gate Method/table-deployment.tex>)，并查看了熵曲线原图。下文以章节标题和表格标签定位证据。**“展开”表示为实现补全数学形式；“审查”与“候选问题”不是作者结论。**

## 1. 先把本文放对位置

本文让小模型阅读大模型**已经生成、尚未执行**的动作及近期文本，预测这次工具调用是否会报错；风险高就拦截并要求重规划。小模型不生成未来动作，也不做 token 草稿接受/拒绝验证。

例如 Actor 已生成 `python fix.py`。SU 先评分，认为可能报错则提示 Actor 重写；它没有提前执行脚本，也没有证明脚本正确。训练标签是工具执行无错误、退出码为 0，不能直接等同于修复任务成功、没有幻觉或环境安全。

**最有用的结论：小模型轨迹评分有失败排序能力，但用它拦截正式动作会改变 Agent 策略。工具报错减少与任务成功率提高不是同一目标。** 本文不提供 lossless 保证。

## 2. 方法与公式：从文本到执行 gate

### 2.1 输入、窗口、标签

【原文：Method / Problem Setup】设大模型为 \(p\)，可读取权重的小模型为 \(q\)，任务初始上下文为 \(x\)：

\[
\mathcal T=(r_1,a_1,\ldots,r_N,a_N),\qquad
\tau_t\in\{\mathrm R,\mathrm A\}.
\]

\(r_i\) 是 assistant 的自由文本，\(a_i\) 是工具名称和序列化参数；\(\tau_t\) 来自消息结构。这里的 reasoning 是 harness 中可见的文本，不要求获得闭源模型的内部隐藏推理。

默认只提取最近 \(k=3\) 步的特征：

\[
\mathcal W_i=(r_{\max(1,i-k+1)},a_{\max(1,i-k+1)},\ldots,r_i,a_i).
\]

当前动作已经生成，当前工具返回尚不存在。二元标签及预测目标为：

\[
Y_i=\Omega(a_i,x)\in\{0,1\},\qquad
\hat u_i=\mathcal F(\mathcal W_i)\in[0,1],
\]
\[
\mathbb E[Y_i\mid\mathcal F(\mathcal W_i)=c]\approx c.
\]

附录把 \(Y_i=1\) 定义为工具调用正常执行、exit code 0。为避免原文一处称 failure-likelihood、另一处却定义 success probability 的混用，本笔记统一：\(\hat u\) 是成功分数，\(\hat f=1-\hat u\) 是失败分数。

【审查】动作结果实际依赖当前环境状态，\(x\) 作为“初始上下文”不足以完整表达 oracle 的输入。工程上必须保存执行前状态标识及历史 observation；不能把成功标签理解为仅由动作文本决定。正文窗口、全轨迹条件前缀以及附录 trajectory-so-far 三种说法也没有统一：**统计最近三步，不必然等于模型只读取三步。**

### 2.2 小模型一次 teacher forcing，提取三种信号

【原文：Speculative Cross-Likelihoods】把 Actor 文本按 \(q\) 的 tokenizer 编码为 \(y_1,\ldots,y_T\)。给定已经实现的前缀，一次因果前向计算各位置分布：

\[
q_t(v)=q(v\mid x,y_{<t}),\qquad
\tilde y_t=\arg\max_{v\in V}q_t(v).
\]
\[
\boxed{s_t=-\log q_t(y_t)}
\]
\[
\boxed{g_t=\log q_t(\tilde y_t)-\log q_t(y_t)}
\]
\[
\boxed{H_t=-\sum_{v\in V}q_t(v)\log q_t(v)}.
\]

- \(s_t\)：小模型对 Actor 实际 token 有多意外；低频 token 也可能完全正确。
- \(g_t\)：小模型最偏好的 token 相比 Actor 实际 token 有多大优势；\(g_t\ge0\)。
- \(H_t\)：小模型完整下一 token 分布的离散程度，与实际选中哪个 token 不同。

以上全部是 **\(q\) 的量**。只要拿到 Actor 输出文本，就能计算，不需要推理时读取 \(p\) 的 logits。单次 teacher forcing 也不等于无计算开销：需要处理输入、输出词表分布和熵归约。

【实现展开】若 logits 为 \(z_{t,v}\)，则令 \(\ell_{t,v}=z_{t,v}-\operatorname{logsumexp}_{v'}z_{t,v'}\)，用 \(-\ell_{t,y_t}\)、\(\max_v\ell_{t,v}-\ell_{t,y_t}\)、\(-\sum_v e^{\ell_{t,v}}\ell_{t,v}\) 计算三信号。必须把预测位置与下一个实际 token 对齐，不能用看到 \(y_t\) 后的 logits 给 \(y_t\) 打分。可分块归约，避免持有整条长轨迹的全部词表 logits。

### 2.3 50 维特征：八种统计量写全

【原文：Component-Aware Feature Extraction；附录 Feature Taxonomy】分别取窗口中的两类 token：

\[
\mathcal I_i^P=\{t\in\mathcal W_i:\tau_t=P\},\qquad P\in\{\mathrm R,\mathrm A\}.
\]

对每个阶段的每个信号 \(s,g,H\)，按时间顺序得到数列 \(z_1,\ldots,z_n\)，计算八种统计量。以下补全原文列出的统计名称；**原文明示 trend 是首尾均值差，不是拟合直线斜率。**

\[
\mu=\frac1n\sum_{j=1}^n z_j,\qquad
v=\frac1n\sum_{j=1}^n(z_j-\mu)^2,
\]
\[
z_{\max}=\max_j z_j,\qquad z_{\min}=\min_j z_j,
\]
\[
\operatorname{skew}=\frac{\frac1n\sum_{j=1}^n(z_j-\mu)^3}{(\sqrt v+\epsilon)^3},
\]
\[
m=\max(1,\lceil0.1n\rceil),\qquad
\mu_{\rm first}=\frac1m\sum_{j=1}^{m}z_j,\qquad
\mu_{\rm last}=\frac1m\sum_{j=n-m+1}^{n}z_j,
\]
\[
\Delta=\mu_{\rm last}-\mu_{\rm first}.
\]

于是：

\[
\operatorname{stats}(z)=(\mu,v,z_{\max},z_{\min},\operatorname{skew},\Delta,
\mu_{\rm first},\mu_{\rm last}),
\]
\[
\boxed{\phi_i=\operatorname{concat}_{P\in\{\mathrm R,\mathrm A\},\,h\in\{s,g,H\}}
\operatorname{stats}(h|_{\mathcal I_i^P})\;\Vert\;
(|\mathcal I_i^{\mathrm R}|,|\mathcal I_i^{\mathrm A}|)\in\mathbb R^{50}}.
\]

维数为 \(2\times3\times8+2=50\)。方差是否用无偏估计、偏度修正、10% 的取整、\(\epsilon\)、空阶段填充值，原文均未指定；上式是**可实现约定**，不冒充作者代码。空阶段可填零并由长度标识，但必须记录这个选择。

原文 trend 等于两个已包含特征之差，所以这三维存在精确线性关系；50 维不等于 50 个独立信息源。L1 下单个权重的选择会受这种冗余和标准化影响，不能直接把权重当因果解释。

### 2.4 为什么分 reasoning/action？证据到哪一步？

【原文：附录 The Phase Structure；熵曲线图】500 条训练轨迹，成功/失败各 250 条。作者观察到：失败步骤在 reasoning 阶段熵较高，在 action 阶段大部分位置熵较低，末段又可能抬升。混合平均可能抵消两类差异，分阶段和首尾特征可保留它们。

这支持特征设计，但尚不是“失败必然导致这种熵形状”的机理证明。阶段也改变语法约束、token 类型和长度；相关性未与这些因素完全解耦。SU-SFT 对 Uniform 的 AUC 增量为 0.05，是方法有效的消融证据，不能证明唯一原因就是符号抵消。

【图文口径待核查】实际图右轴标注 **Entropy of top-5 logprobs**，标题为 Qwen3-Coder-480B；正文定义却是小模型 \(q\) 的全词表熵。top-5 是否归一化、来自 Actor 还是 Drafter，图文没有交代清楚。图左侧也能看到每步边界跳变，不能照抄“边界完全不可见”。因此可以记录相位现象，不能把这幅图当作全词表 Drafter 熵公式的无歧义验证。

### 2.5 Logistic regression 与训练目标

【原文：Calibrator；附录 Calibrator Fitting】每维特征使用**训练集**均值和标准差标准化。展开为：

\[
\bar\phi_{i,d}=\frac{\phi_{i,d}-\mu_d^{\rm train}}{\sigma_d^{\rm train}+\epsilon},\qquad
\sigma(z)=\frac1{1+e^{-z}},\qquad
\hat u_i=\sigma(w^\top\bar\phi_i+b).
\]
\[
\boxed{\min_{w,b}\sum_i\left[-Y_i\log\hat u_i-(1-Y_i)\log(1-\hat u_i)\right]
+\lambda\sum_{d=1}^{50}|w_d|}.
\]

这里展开了原文 BCE 记号。\(\lambda\) 在 SWE-rebench 训练集进行五折交叉验证，以验证 AUROC 选取，再冻结至所有测试集；原文搜索网格记为 \(\{10^{-3},\dots,10^1\}\)，未完整列出中间点。复现时需固定 loss 用求和还是均值、库中 C 与 \(\lambda\) 的换算，以及按任务分组的交叉验证方式。

### 2.6 小模型如何对齐 Actor：SFT 与 TF

【原文：附录 Draft-Model Training】比较原始 Qwen3-4B、SFT、teacher-forced 分布蒸馏。**teacher forcing 评分与 TF 蒸馏训练不是同一概念**：所有版本都能 teacher forcing 评分。

原文说明 SFT 用 assistant token 的 next-token cross-entropy，展开为：

\[
\mathcal L_{\rm SFT}(\theta)=
-\frac1{\sum_{j,t}M_{j,t}}\sum_{j,t}M_{j,t}
\log q_\theta(y_{j,t}\mid x_j,y_{j,<t}),
\]

\(M_{j,t}=1\) 表示该位置属于 assistant，其他角色不计训练 loss。历史工具 observation 是否作为输入保留，需要与数据模板核对，不能由 loss masking 推断已删除输入。

TF 额外对齐 Actor 的完整分布，原文**没有给出确切混合目标、权重或温度**。一种标准实现展开为：

\[
p_t^{(T)}(v)=\frac{e^{z^p_{t,v}/T}}{\sum_{v'}e^{z^p_{t,v'}/T}},\qquad
q_t^{(T)}(v)=\frac{e^{z^q_{t,v}/T}}{\sum_{v'}e^{z^q_{t,v'}/T}},
\]
\[
\mathcal L_{\rm KD}=\frac{T^2}{\sum M_{j,t}}
\sum_{j,t}M_{j,t}\sum_v p_{j,t}^{(T)}(v)
\log\frac{p_{j,t}^{(T)}(v)}{q_{j,t}^{(T)}(v)},
\]
\[
\mathcal L=(1-\eta)\mathcal L_{\rm SFT}+\eta\mathcal L_{\rm KD},\qquad0\le\eta\le1.
\]

这组式子仅说明理论上怎样实现分布蒸馏，**不是原文已公布配置**。还需确认两模型词表对齐。Qwen3-Coder-480B 开放权重，作者称训练时重算教师 logits；Claude 无 logits，只做输出文本 SFT。因此本文的 black-box 承诺针对部署，不代表 TF 训练也不访问教师内部。

### 2.7 Gate 时序与伪代码

【原文：Algorithm SU Veto-Gate Policy】设阈值 \(\tau\)，同一步最大重规划次数 \(K\)，当前次数 \(c\)：

\[
\pi_{\rm gate}(\hat u,c)=
\begin{cases}
\mathrm{execute}(a_i),&\hat u\ge\tau,\\
\mathrm{replan},&\hat u<\tau\ \land\ c<K,\\
\mathrm{abstain},&\hat u<\tau\ \land\ c\ge K.
\end{cases}
\]

```text
Actor 根据实际历史生成当前 reasoning 和完整 action
在 action 尚未执行时截取可用历史，标注 R/A token
q 因果前向 → 三类信号 → 最近三步 50 维统计 → 固定标准化
logistic regression 给出成功分数 u
若 u >= tau：执行 action，保存真实结果，再进入下一步
否则若重规划次数未用完：加入“预测有风险”的提示，重新调用 Actor
否则：返回 low-confidence，停止或交给外部流程
```

重规划没有得到真实报错，提示不能伪造已经观察到的 exception。论文未提供提示模板、\(K\)、最终 \(\tau\)、阈值选择准则和重试窗口处理细节。新的 Actor 调用并非免费，也不能默认一定比执行后重试便宜。

## 3. 正确性与理论解释审查

### 3.1 与 speculative sampling 的类比存在逻辑缺口

【原文：Why a black-box proxy works】作者引用标准接受概率：

\[
\alpha_t=\min\left(1,\frac{p(y_t\mid\cdot)}{q(y_t\mid\cdot)}\right),
\]

再把“小模型 surprisal 高”关联到“接受率低”。该推导不成立：若 \(p=q\)，任何正概率 token 都有 \(\alpha_t=1\)，即使 \(q(y_t)\) 极小、\(s_t\) 极大。接受率看 **比值**，不是 \(q(y_t)\) 的绝对大小；SU 评分的 token 还来自 Actor，而非标准采样中的 Drafter proposal。

因此本文建立的是经验失败预测器，没有从接受公式证明恢复了 Actor 的不确定性。这个问题削弱的是机理解释，不会自动否定 AUC 的实验增益。

### 3.2 校准、执行结果、策略保持必须分开

1. 输出在 \([0,1]\) 不代表校准正确。作者在 Calibration 节承认应称 score，未做后校准。
2. exit code 0 不保证语义正确；非零返回也可能是有信息价值的测试。例如修复前跑测试获得失败栈，是合理行动，不能仅因预测会报错而认定应阻止。
3. 通过阈值不保证下一次动作一定成功；拒绝又可能阻断本来有价值的调用。
4. SU 直接改变正式 Actor 的动作轨迹，无轨迹一致或分布保持保证，也没有环境快照验证、rollback、commit 协议。

校准变换后 gate “不变”还要求同步映射/重选阈值：若 \(h\) 严格递增，\(\hat u\ge\tau\iff h(\hat u)\ge h(\tau)\)。不能变换分数后保留原数值阈值，还宣称决策必然不变；OOD 上也不能由 ID 排序好推出固定阈值可靠。

## 4. 实验、公式与证据边界

### 4.1 数据和计算配置

【附录 Data / Training / Compute】

| 项目 | 论文配置 |
|---|---|
| Actor / scorer | Qwen3-Coder-480B、Claude 3.5 Sonnet / Qwen3-4B |
| 训练数据 | SWE-rebench OpenHands：10,000 轨迹，327,098 调用 |
| ID 测试 | 1,000 轨迹，32,585 调用；训练/测试正常执行率均约 84.7% |
| OOD | SWE-Bench Verified 500 轨迹；DA-Code 500 轨迹 |
| 轨迹规模 | 平均约 65 turns、54k context tokens；训练合计约 0.54B tokens |
| 小模型训练 | full FT，2 epochs，batch 32，长度 32768，YaRN，bf16，assistant-only loss |
| 优化 | AdamW，β=(0.9,0.95)，lr=1e-5，cosine，5% warmup，WD=0.1，clip=1.0 |
| 硬件与成本 | SFT/TF：4×A100，约 50 GPU-h；calibrator：64 CPU，不到 1 分钟 |
| 部署新增评分 | 4B 在单 A100 每步约 100ms |

单次评分 100ms 是作者测量值，不应写成零成本；输入长度、缓存方式和并发负载未列全，无法直接外推到任意长度。50 GPU-h 是否包含重算 480B 教师 logits 也需核查。

附录称同一 scaffold 下仅任务不同、工具接口不变；主表 caption 又说 DA-Code 改变 domain 和 tool interface。可以同为 OpenHands 但接入不同工具，然而文字口径需澄清，不能断言已隔离所有 harness 影响。

### 4.2 失败预测结果

【表 main_results】每格是 **AUROC / P@R80**；部署主 Actor 为 Qwen3-Coder-480B。

| 方法 | SWE-rebench | SWE-Bench Verified | DA-Code |
|---|---|---|---|
| Verbalized confidence | .58 / .50 | .57 / .49 | .59 / .51 |
| Last-TP（Actor 最后块平均 log-prob） | .70 / .61 | .68 / .59 | .69 / .60 |
| Global-TP（Actor 全轨迹平均 log-prob） | .67 / .57 | .65 / .55 | .66 / .56 |
| HTC（Actor 内部概率） | .85 / .77 | .82 / .74 | .83 / .75 |
| SU 未训练 | .63 / .54 | .61 / .52 | .63 / .54 |
| SU-Uniform SFT | .74 / .65 | .71 / .62 | .72 / .63 |
| SU-SFT | .79 / .70 | .76 / .67 | .77 / .68 |
| SU-TF | .80 / .71 | .77 / .68 | .78 / .69 |

SFT 对原始 scorer 增益明显；阶段分离相对 Uniform 增加 .05 AUC；TF 相比 SFT 只多 .01。**没有多个 seed 或置信区间**，不能凭 1 个点断言蒸馏稳定更强。HTC 是白盒参照，不是已证明的性能上界。

【指标展开】以失败为正类，\(Z=1-Y\)、失败分数 \(\hat f=1-\hat u\)：

\[
\mathrm{AUROC}=\Pr(\hat f^+>\hat f^-)+\tfrac12\Pr(\hat f^+=\hat f^-),
\]
\[
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},\qquad
\mathrm{P@R80}=\mathrm{Precision}\big|_{\mathrm{Recall}\approx0.8}.
\]

这是失败检测的常用口径；原文没写清 P@R80 的正类、插值和跨任务聚合方式，复现需先确认。AUROC 与 P@R80 衡量区分/排序，不是概率校准指标。

### 4.3 校准结果

【表 calibration】

| 方法 | ECE↓ | MCE↓ | Brier↓ | NLL↓ | BSS↑ |
|---|---:|---:|---:|---:|---:|
| 常数 base-rate | .008 | .008 | .129 | .426 | .000 |
| SU 未训练 | .015 | .054 | .127 | .422 | .019 |
| SU-SFT | .024 | .060 | .101 | .364 | .215 |

【标准定义展开】给定 \(n\) 个成功标签和成功分数，按分数划分非空 bins \(B_m\)，令

\[
\operatorname{acc}_m=\frac1{|B_m|}\sum_{i\in B_m}Y_i,\qquad
\operatorname{conf}_m=\frac1{|B_m|}\sum_{i\in B_m}\hat u_i,
\]
\[
\mathrm{ECE}=\sum_m\frac{|B_m|}{n}|\operatorname{acc}_m-\operatorname{conf}_m|,
\quad\mathrm{MCE}=\max_m|\operatorname{acc}_m-\operatorname{conf}_m|,
\]
\[
\mathrm{Brier}=\frac1n\sum_i(\hat u_i-Y_i)^2,
\quad\mathrm{NLL}=-\frac1n\sum_i[Y_i\log\hat u_i+(1-Y_i)\log(1-\hat u_i)],
\]
\[
\mathrm{BSS}=1-\frac{\mathrm{Brier}_{\rm SU}}{\mathrm{Brier}_{\rm constant}}.
\]

bin 数及分箱策略原文未给。Brier 改善但 ECE/MCE 反而变差，不应统称“校准全面改善”。常数预测器 ECE 低却不能区分样本，说明必须同时看排序、校准和决策结果。也不能把 ECE=.024 转述为每个动作的概率误差最多 2.4%。

### 4.4 Gate 部署：节省与质量损失同表看

【表 impact；源码宏已展开】

| 指标 | SWE-Bench Verified | DA-Code |
|---|---:|---:|
| 无 gate → 有 gate 的每调用错误率 | 21% → 15% | 14% → 6% |
| 错误率绝对下降 | 6 个百分点 | 8 个百分点 |
| 平均 tokens/task | −14% | −19% |
| 任务成功率 | **49% → 44%** | 64% → 63% |
| true-veto recall | .72 | .69 |
| false-veto rate | .14 | .16 |

成功率变化分别是 **−5、−1 个百分点**；表内写成 −5%、−1% 容易误读为相对比例。主部署表仅列 Qwen，不能将摘要对 Claude 的概括理解为已经给出了 Claude 同规格部署表。

【指标展开】若 \(V=1\) 表示拦截，通常定义：

\[
\mathrm{TrueVetoRecall}=\Pr(V=1\mid Y=0),\qquad
\mathrm{FalseVetoRate}=\Pr(V=1\mid Y=1).
\]

原文未写清 false-veto 分母；若采用“所有拦截中错误拦截的比例”，则是另一指标 \(\Pr(Y=1\mid V=1)\)，不能混用。线上未执行动作的 \(Y\) 属于反事实结果，要说明是否用隔离回放获得。附录预告 gate simulation，却没有补全模拟/重新 rollout 协议。

**不能只以错误率与 token 节省认定系统更好：**

- gate 改变执行集合及后续轨迹，错误率分母也可能变化；需报告通过率、abstain、重规划次数和完整任务结果。
- 较早终止失败任务也能减少 tokens/task；至少同时报告成功率—成本曲线。
- tokens/task 减少不是实测端到端时延减少。需计入评分 100ms、Actor 重规划、输入/输出计价和工具等待。
- 所有结果是单次点估计，作者 Limitations 已明确说明；质量下降不能忽略，也不能据此断言其在所有运行中必然下降。

### 4.5 跨 Actor 迁移

【表 cross_agent，SWE-Bench Verified AUC】Qwen-SFT scorer 在 Qwen 为 .76，在 Claude 为 .69；Claude-SFT 在 Claude 为 .76；混合 SFT 在 Claude 为 .75；未训练在 Claude 为 .60。

所以“开放模型数据可迁移到闭源 Actor”有证据，但 .69 与 .75/.76 仍有 6–7 个点差距。作者将收益解释为领域适配，不是已隔离领域、Actor 风格、数据量的因果实验。Claude 数据规模、混合比例和 calibrator 是否随 scorer 重拟合未交代充分。

## 5. 理论上怎样实现；目前不能声称精确复现

先做离线失败预测，暂不启动大规模训练和在线 gate：

1. 读取轨迹并保留 task/repository、step、角色、工具名称/参数、执行前状态、真实退出码。当前标签不得进入当前评分输入。
2. 按任务划分 train/validation/test，三步重叠窗口不能跨集合；五折也按任务分组。保留失败步骤，不能只收集成功任务。
3. 用未训练 4B 对已生成文本因果评分，提取 R/A 50 维特征，固定标准化与 logistic regression。先确认指标和标签语义。
4. 再比较 SFT、Uniform、reasoning-only、action-only、长度/工具类型等简单基线。TF 需要 480B 教师概率，成本大且本文增益小，不适合作为从零实现的第一步。
5. 只有离线收益稳定，才在可回放环境做 shadow scoring 与同任务配对 gate 实验，测真实开销和最终任务成功率。记录被拦截动作的反事实标签来源。

**原理实现可行，但材料不足以精确复现论文数值。** 本次本地正文/附录未给出 SU 专属代码或训练 checkpoint 链接；不据此声称作者永久未开源。主要缺口：具体轨迹 ID/分组、输入上下文与三步窗口关系、TF 目标、图中熵口径、特征边界约定、阈值与 K、重规划提示、反事实评估及线上成本记录。

这一步不需要修改用户已有四个 baseline 项目；代码阶段另行进行。

## 6. 与动作投机主线的关系

| 方法 | 预测/处理对象 | 最终接受依据 | 对研究判断的影响 |
|---|---|---|---|
| Speculative Actions | 未来动作/返回与预执行 | 正式调用匹配及相应缓存/环境条件 | 预测失败会浪费计算，不等于正式动作执行报错 |
| SPORK | Actor 未完成推理时探测工具调用 | 正式完整调用验证；部分失败草稿 token 复用 | 已覆盖自探测和两层投机结合 |
| AOSpec | 动作与 observation 双投机 | target action、origin state、真实 observation 等验证 | 分数不能替代状态和输入一致性条件 |
| SMC | macro 余链提前执行并跳过部分 Actor 决策 | anchor 加统计/运行时规则，明确近似 | 质量取舍必须在最终任务指标上评价 |
| SU | 已生成动作的执行失败风险 | 学习分数与阈值，风险高重规划/退出 | 改正式策略，无严格 lossless；属于风险控制分支 |

若将 SU 信号用于“是否提前执行”，而保留正式 Actor 调用和既有严格验证，理论上可把它当资源分配信号；但这时应预测**匹配、状态有效及可隐藏时间共同决定的收益**，不能直接用“工具不会报错”代替。失败的测试命令也可能被 Actor 正式调用，投机复用它的报错 observation 同样可能节省时间。

## 7. 为后续创新分析保留什么

### 7.1 已覆盖，不再直接当新 idea

- 小模型读取大模型轨迹，以 likelihood/entropy 预测动作失败。
- 分 reasoning/action、利用短期历史、线性风险模型和执行前 veto。
- 用开放 Actor 数据帮助闭源 Actor、SFT 与蒸馏对齐 scorer。
- 根据风险路由更强模型、人类或额外采样：作者已经提出。
- token 生成过程中 streaming veto：作者已列未来方向。

### 7.2 三个可证伪问题，不是已确认创新

| 待检验问题 | 最小实验与必要 baseline | 什么结果让我们放弃/收窄 |
|---|---|---|
| 执行失败标签是否错罚了有价值的诊断动作？ | 对相同任务回放，区分预期失败测试、命令错误和无价值重复；先比较无 gate、SU、简单工具类别豁免和相同通过率 gate | 简单工具豁免就解决，或预期失败调用占比很低；不要立即训练复杂“价值模型” |
| failure score 能否预测动作投机的真实净收益？ | 在 SA 日志分别标注正式调用命中、状态有效、报错、隐藏时间和额外成本；比较失败分数、已有匹配置信度、简单时间阈值 | 控制匹配概率/时延后，失败分数没有增量，或简单规则同样好 |
| 分阶段熵的增益是否超出工具语法和长度？ | 相同 scorer 与数据，比较 50 维、Uniform、action-only、工具类型/长度；按任务分组验证及等模型容量对照 | 增益由格式/长度完全解释，或跨工具消失；结论应是适用边界，不硬造普适机理 |

第二个问题的诊断目标可以明确写为下面的**自定义记账式**，不是原文算法：

\[
B_i=\mathbf1[\text{正式动作匹配且状态验证通过}]\,T_{i,\rm hidden}
-T_{i,\rm added},
\]

其中所有项均按同一 wall-clock 关键路径口径计量；GPU/API 费用另报，不能直接与毫秒相减。是否执行成功并未出现在复用充分条件中，这正是需要验证标签与目标关系的原因。PASTE/AOSpec 已有收益调度工作，确认增量前不能把这个记账式当贡献。

## 8. 本篇结论

核心贡献落在**技术层**：将闭源 Actor 的可见轨迹转成小模型可计算的阶段特征，得到可用于干预的失败分数。主要证据是 SFT 和分阶段消融、跨任务/Actor 排序增益；不是新的正确性理论。

核心局限是执行退出码与长期任务价值不一致、经验分数没有严格校准或 lossless 保证、gate 质量成本取舍与复现协议未充分展开。对当前动作投机研究最直接的启发，是先区分“会不会匹配”“会不会报错”“是否值得提前执行”三个标签，再检验它们之间的关系，而不是直接增加一个 confidence gate。

**十二篇第一轮阅读完成。** 后续用跨论文台账排除重复 idea，再结合四个已有代码项目做最小诊断实验；本笔记中的候选问题均未经过更广近邻检索和复现实验，不能当作已经达到顶会新颖性标准。
