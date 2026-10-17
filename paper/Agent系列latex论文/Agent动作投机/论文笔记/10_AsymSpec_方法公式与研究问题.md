# 10 · AsymSpec：方法公式、实现原理与研究问题

阅读日期：2026-09-09。已读正文、附录全部实际启用内容、Algorithm 1、表格和系统图；本轮不运行项目代码。延续此前要求：完整记录方法公式与理论实现，区分作者陈述、补充推导和待核实问题。

## 1. 来源与关键定位

- **AsymSpec: Context-Asymmetric Speculative Decoding for Agentic LLMs**，Sheng Liang 等；[arXiv:2608.26004v1](https://arxiv.org/abs/2608.26004v1)，2026-08-26 提交。本次页面仅列 v1，Comments 为 EMNLP Main Conference 2026。
- [本地论文目录](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AsymSpec- Context-Asymmetric Speculative Decoding for Agentic LLMs>)：主体和附录均在 `main.tex`，图为 `figures/pipeline_0526_v4.png`。不把注释草稿作为方法事实。
- [用户已下载的代码项目](/mnt/data/zhangshuai/LLMs/Projects/AsymSpec)确实存在。本轮只确认目录，不做代码审计或运行，不将“本地有代码”写成“已复现”。

**最重要的结论：AsymSpec 不是保持原目标分布的严格 speculative decoding。** 正文 §3.3 明确说它是针对 greedy emission 的 speculative-style steering：小模型读取完整输入，用草稿接受和 logits 差值引导只读压缩输入的大模型。它既不保证等于完整上下文大模型，也不保证等于压缩上下文大模型。

**它解决的是质量—成本取舍**：大模型少读上下文，小模型尝试补回压缩丢失的信息。与 SMC 相似，它接受改变原策略的结果；与 SMC 不同，近似发生在 token 生成层，而非工具动作链提交层。

标记：**【作者陈述】**为原文规则/数值；**【本文解释】**为例子；**【补充原理】**为标准数学；**【推导审查】**为新增推导或反例；**【待确认】**为源码或日志尚需回答的问题。

## 2. 一个具体例子：API 参数格式被压缩掉

【作者案例；Case study / RecordHealthData】完整 API 文档要求时间字段形如 `2021-09-17 10:30:00`，健康记录按指定列表结构表达；压缩输入只保留 API 名称和函数签名。大模型可能输出少了秒的时间或自然语言健康记录。

AsymSpec 做三路计算：

1. 小模型看完整文档，偏向补出 `:00` 和结构化字段。
2. 同一个小模型只看压缩文档，作为“少读信息时会怎样”的对照。
3. 大模型只看压缩文档，提供主干推理和输出分数。

若完整文档小模型的草稿通过接受门槛，直接保留；若不通过，用两种小模型视图的 logits 差值修正大模型，发出一个替代 token。

这是传递**具体上下文偏好**，不是提前运行 API，也不是环境状态验证。所谓“Verifier”在此不代表它逐 token 执行了传统 SD 的精确分布校验。

## 3. 完整变量、三路计算与时序

### 3.1 输入与模型

【作者设定】$L$ 是大模型，$S$ 是较小 Drafter；压缩器 $C$ 给出：

$$
x_{\mathrm{comp}}=C(x_{\mathrm{full}}).
$$

两个参照是：

$$
P_L(\cdot\mid x_{\mathrm{full}})\quad\text{与}\quad
P_L(\cdot\mid x_{\mathrm{comp}}).
$$

原文称前者 Ceiling、后者 Floor。它们是实验基线名称，**不是数学上下界**：压缩有时能减小干扰，Floor 也可能高于名为 Ceiling 的参照；GAIA 还有混合参照，见实验部分。

令 $y$ 为当前正式生成前缀，$d_{1:K}$ 为这一轮草稿；对位置 $i$，三路 logits 是：

$$
\begin{aligned}
a_i&=\operatorname{logits}_S(x_{\mathrm{full}}\oplus y\oplus d_{<i}),\\
b_i&=\operatorname{logits}_S(x_{\mathrm{comp}}\oplus y\oplus d_{<i}),\\
t_i&=\operatorname{logits}_L(x_{\mathrm{comp}}\oplus y\oplus d_{<i}).
\end{aligned}
$$

上式显式补出原文省略的共同生成前缀；$a_i,b_i,t_i\in\mathbb R^{|\mathcal V|}$。三个概率向量记作：

$$
p_i^a=\operatorname{softmax}(a_i),\quad
p_i^b=\operatorname{softmax}(b_i),\quad
p_i^t=\operatorname{softmax}(t_i),
\quad
\operatorname{softmax}(z)_v=\frac{e^{z_v}}{\sum_{w\in\mathcal V}e^{z_w}}.
$$

### 3.2 一轮执行

【作者 Algorithm 1】

1. 完整上下文小模型自回归生成 $K$ 个草稿 token，保存对应 logits $a$。
2. 压缩上下文小模型对相同草稿路径计算 $b_{1:K+1}$；大模型对相同路径计算 $t_{1:K+1}$。
3. 对每个草稿位置计算 $\delta_i$、上下文分歧 $D_i$ 和接受门槛。
4. 从第一个位置开始顺序接受；遇到第一次拒绝，就发出该位置的融合 token，丢弃后续草稿并进入下一轮。
5. 若 $K$ 个全部接受，再发出 **$\arg\max t_{K+1}$**，不是融合后的 bonus token。

正文说“三次 forward”是三条计算路径的概括，不是每轮只调用小模型一次：完整视图要做 $K$ 步自回归 drafting，另有压缩视图打分和 verifier 批量打分。生成 $a_{K+1}$ 的缓存/前向安排也需代码核实；主循环实际只在拒绝位置用 $\delta_i$，bonus 不用 $\delta_{K+1}$。

默认所有实验 greedy（$\tau=0$）。伪代码中的 sample 因而应理解为按部署采样策略选 token，此处为 argmax；计算 JSD/接受门槛时仍需 logits 的 softmax，不能把这些概率全替成 greedy 的 one-hot 分布。

## 4. Contrastive δ-fusion：公式与原理

### 4.1 原文公式

【作者公式；§3.2】

$$
\delta_i=a_i-b_i,
$$

$$
d'_i=\arg\max_{v\in\mathcal V}\left[t_i(v)+\beta\delta_i(v)\right].
$$

$\beta$ 控制融合强度，主配置 $\beta=1$。方法和算法 Require 写 $\beta\in[0,1]$，附录又测 $\beta=2$：应分别记录“方法给出的范围”和“实际额外探索值”，不要默默改成同一范围。

**融合发生于首次拒绝的位置**。不是每个输出都取融合 argmax，也不是 $\gamma_{\mathrm{eff}}$ 直接乘在 $\delta$ 上：接受门槛改变哪些位置进入融合分支，融合权重本身仍为 $\beta$。

### 4.2 为什么相减有一定合理性

【作者解释】同一个 Drafter、相同权重，仅改变上下文，差值希望取消模型自身的固定偏好，突出完整输入带来的信息。

【推导审查】如果 logits 能分解为 $a(v)=u(v)+c_{\mathrm{full}}(v)$、$b(v)=u(v)+c_{\mathrm{comp}}(v)$，共享加性部分 $u$ 确实相消。但神经网络不保证“模型偏好 + 上下文贡献”完全可加；差值也可能包含错误理解、注意力变化、噪声和长度效应。论文没有证明它等于完整上下文大模型需要的修正。

### 4.3 概率比视角：它到底在放大什么

【补充推导】定义仅用于理解的融合分布：

$$
\widetilde p_i(v)=\operatorname{softmax}(t_i+\beta(a_i-b_i))_v.
$$

因为各 softmax 归一化常数在 token 间相同：

$$
\widetilde p_i(v)
=\frac{p_i^t(v)\left[p_i^a(v)/p_i^b(v)\right]^\beta}
{\sum_{w\in\mathcal V}p_i^t(w)\left[p_i^a(w)/p_i^b(w)\right]^\beta}.
$$

所以它放大“在小模型完整输入下相对压缩输入更可能”的 token，而不是直接把大模型替换成小模型。实现使用 logits 差或 log-prob 差，避免显式概率比在很小分母处数值不稳。

注意 $\widetilde p$ 只描述融合分支，整个 AsymSpec 还包含接受草稿及未融合 bonus，**不能称整个算法在从 $\widetilde p$ 精确抽样**。

若想让融合恰好复原完整上下文大模型的分布，至少需要如下很强的逐位置关系：

$$
\ell_i^{L,\mathrm{full}}-\ell_i^{L,\mathrm{comp}}
=\beta(a_i-b_i)+c_i\mathbf1,
$$

其中 $c_i$ 是与 token 无关的常数。这是充分条件的表达，不是论文建立的事实。它提示后续可测“大模型真正的上下文修正方向”与小模型差值是否一致，而不是只用最终分数配一个信息恢复故事。

## 5. CDA 门槛：全部公式、推导与保证边界

### 5.1 JSD 完整定义

【作者采用的散度；标准定义展开】令 $m_i=(p_i^a+p_i^b)/2$：

$$
D_i=\operatorname{JSD}(p_i^a\|p_i^b)
=\frac12\operatorname{KL}(p_i^a\|m_i)
+\frac12\operatorname{KL}(p_i^b\|m_i),
$$

$$
\operatorname{KL}(p\|q)=\sum_{v\in\mathcal V}p(v)\ln\frac{p(v)}{q(v)}.
$$

这里采用自然对数，$0\ln0$ 按极限约定为 0，JSD 有界：

$$
0\leq D_i\leq\ln2.
$$

【作者附录解释】令上下文来源 $Z\sim\operatorname{Bernoulli}(1/2)$，条件 token 分布分别为 $p^a,p^b$，则：

$$
D_i=I(X;Z)\leq H(Z)=\ln2.
$$

这是“从下一个 token 能辨别使用了哪种上下文”的互信息；**不是 token 与正确答案的互信息**，不能直接推出 $D$ 大就更值得相信完整上下文 Drafter。

### 5.2 原文门槛与接受条件

【作者公式；§3.3】

$$
\gamma_{\mathrm{eff}}(i)=\gamma e^{-D_i},
\qquad \gamma\in(0,1],
$$

$$
\boxed{\ p_i^t(d_i)>\gamma_{\mathrm{eff}}(i)\,p_i^b(d_i)\ }.
$$

接受分母是 **压缩视图小模型概率 $p^b$**，不是产生草稿的完整视图概率 $p^a$，也不是融合分布。首个失败位置才使用 $\arg\max(t_i+\beta\delta_i)$。

JSD 范围给出：

$$
\frac\gamma2\leq\gamma_{\mathrm{eff}}(i)\leq\gamma.
$$

默认 $\gamma=0.5$，有效门槛为 $[0.25,0.5]$。分歧越大，右侧越低，越容易接受完整视图的小模型草稿。

数值实现可等价比较：

$$
\log p_i^t(d_i)-\log p_i^b(d_i)>\log\gamma-D_i.
$$

论文用严格 $>$，相等时拒绝；实现不要自行改成另一个 tie policy 而不记录。

### 5.3 指数形式如何推出来

【作者附录公理】无分歧时门槛为 $\gamma$；随 $D$ 连续且不增；假定独立分歧可乘法合成：

$$
\gamma_{\mathrm{eff}}(D_1+D_2)
=\frac{\gamma_{\mathrm{eff}}(D_1)\gamma_{\mathrm{eff}}(D_2)}\gamma.
$$

令 $f(D)=\gamma_{\mathrm{eff}}(D)/\gamma$，则 $f(D_1+D_2)=f(D_1)f(D_2)$。连续正解为指数形式，作者写：

$$
f(D)=e^{-D/T},\qquad
\gamma_{\mathrm{eff}}(D)=\gamma e^{-D/T},\quad T>0.
$$

随后取 $T=1$，结合自然对数的 JSD 上界，得到前面的 $[\gamma/2,\gamma]$。

【推导审查】这些假设固定的是指数**函数族**，没有唯一推出 $T=1$；连续不增还允许常数解作为零衰减极限。把 JSD 定标后选 $T=1$ 是免去额外调参的设计选择，不是 JSD 上界强迫的唯一结果。实际不同上下文变化的 JSD 也未证明满足加法合成，这是设计公理，不是测得的物理规律。

“Parameter-free”准确理解为**没有额外散度尺度参数需要调节**。整个算法仍有 $K,\beta,\gamma$、模型选择和压缩策略，不是无超参数。

### 5.4 原文参考变体公式

为了完整记录消融，还需保留：

$$
\gamma_{\mathrm{eff}}=\gamma\qquad\text{（fixed-}\gamma\text{，无散度门控）},
$$

$$
\gamma_{\mathrm{eff}}=\gamma e^{-\lambda\Delta},
\qquad\Delta=\log\operatorname{softmax}(a_i)-\log\operatorname{softmax}(b_i).
$$

后者是作者 tuned-$\lambda$ 参照，默认 $\lambda=0.1$。原文给的 $\Delta$ 是词表向量，却未在这条文字定义中说明取候选 token 分量还是某种归约；要实现单一接受门槛需核查代码，不能擅自补成作者事实。

SCD 参照在压缩输入上用专家 logits $Y_e$、小模型 logits $Y_a$，作者的复现写为：

$$
\mathcal V_{\mathrm{plaus}}=\{v:Y_e(v)>\ln\alpha+\max_wY_e(w)\},
$$

$$
v^*=\arg\max_{v\in\mathcal V_{\mathrm{plaus}}}
\big[(1+\beta)Y_e(v)-\beta Y_a(v)\big],
\quad\alpha=0.5,\quad\beta=1.
$$

它区别于同一小模型双视图差值，不能把这两种 subtraction 当作完全相同的方法。

## 6. 正确性：最需要纠正的两个前提

### 6.1 标准 SD 并不在数学上要求相同输入

【补充原理】标准 speculative sampling 允许任意合适的 proposal $q$，只要验证对目标 $p$ 使用正确的接受和残差修正：

$$
P(\mathrm{accept}\mid d)=\min\left(1,\frac{p(d)}{q(d)}\right),
\qquad
p_{\mathrm{res}}(v)=\frac{[p(v)-q(v)]_+}{\sum_w[p(w)-q(w)]_+}.
$$

Drafter 可以看更短、不同甚至更丰富的上下文；只要共同生成 token 能对齐且 proposal 定义正确，仍可保留指定目标分布。参照：[标准 SD 原论文 §2.3 / Algorithm 1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)。

所以 AsymSpec 文中“标准 SD 强制完全相同输入”的概括过强。**真正的限制是：若坚持保留压缩 verifier 的分布，就不能同时要求算法系统性改变它以修复压缩损失。** AsymSpec 的实质是非对称输入加有意改变策略的 steering，而非仅打破一个概率论上不允许打破的输入约束。

### 6.2 设 β=0、γ=1 也不自动退化成精确 SD

【作者原句审查】正文称此时退化到 compressed input 的 standard verification。但 CDA 的 $e^{-D}$ 仍存在，接受用的是 $p^b$，且没有标准随机接受与 residual；greedy 模式也不是核对 target argmax。

即使 $D=0$，反例仍成立。取三 token：

$$
p^a=p^b=(0.36,0.35,0.29),\qquad
p^t=(0.40,0.45,0.15).
$$

Drafter greedy 选第一个 token，$0.40>1\times0.36$ 因而接受；但 target greedy 本应选第二个。故上述参数不能保证 target greedy 等价。

论文也明确承认不保持严格目标分布，因此这里是**退化条件表述不准确**，不是揭露一项作者承诺过的普遍 lossless 定理。

此外，$\beta=0$ 的 CDA-only 消融已经改善 LongBench，表明**接受完整上下文草稿本身就在改变输出策略**。不能把 accepted token 一律说成“只加速而不传信息”，再把全部质量恢复归给拒绝位置的 $\delta$。

### 6.3 什么是已经保证的，什么不是

- 已有数学支持：JSD 有界；给定所选形式，有效阈值处于 $[\gamma/2,\gamma]$。
- 没有由此保证：完整上下文质量恢复、草稿一定更正确、接受率下界、任务成功不下降、与任一原模型分布相同。
- 高接受率可能只是门槛更松；需要同时看错误接受和最终任务质量。
- 跨模型“上下文增益”的传递依赖小模型能力、校准与词表对齐；正文没有证明小模型 logits 差就是大模型所缺信息的无偏估计。

## 7. 理论实现：三份 KV、同步和多模态

### 7.1 同模型权重共享，KV 不能共享

同一小模型权重可复用，但它在 full/comp 上的上下文不同，必须维护独立 KV。大模型还有第三份 KV。三路对齐的是**相同已提交输出 $y$**，不是相同绝对 token position：full/comp prompt 长度不同。

拒绝发生在第 $i$ 个草稿时，提交前 $i-1$ 个草稿和修正 token；三路 cache 应保留有效前缀，删除后续草稿状态，并让修正 token 进入后续正确的 cache 状态。新的 Agent 轮次若重新压缩历史，不可无条件复用上一轮 comp KV。

### 7.2 可转成代码的算法

```text
full, comp = full_input, compressor(full_input)
initialize S_full_KV, S_comp_KV, L_comp_KV
y = []

while not stopped:
    draft, a = S.greedy_draft(full, y, K, S_full_KV)
    b = S.score(comp, y, draft, S_comp_KV)
    t = L.score(comp, y, draft, L_comp_KV)

    accepted_all = True
    for i in 1..K:
        pa, pb, pt = softmax(a[i]), softmax(b[i]), softmax(t[i])
        mixture = (pa + pb) / 2
        D = 0.5 * KL(pa, mixture) + 0.5 * KL(pb, mixture)
        gate = gamma * exp(-D)
        if pt[draft[i]] > gate * pb[draft[i]]:
            append draft[i] to y
        else:
            append argmax(t[i] + beta * (a[i] - b[i])) to y
            accepted_all = False
            break
        if EOS or output budget reached:
            stop at the committed prefix

    if accepted_all and not stopped:
        append argmax(t[K+1]) to y       # 原文 bonus 不做 delta fusion
    synchronize_all_three_KV_to_committed_output(y)
```

应补齐首次拒绝就是 EOS、全部接受中的 EOS、输出长度上限、缓存回滚等边界；原文伪代码没有逐项写出。按 log-softmax 比较门槛、JSD 用稳定精度归约，是实现选择，不改变纸面逻辑。

### 7.3 成本不能只说“三路里两路很便宜”

小模型 full drafting 仍是串行步骤，comp 路径仍读取 KV；维护缓存只能避免完整重复 prefill，不能使每 token 对长 KV 的 attention 成本消失。

JSD 需要词表向量的 softmax/熵或 KL 归约，通常是 $O(|\mathcal V|)$ 数值工作。附录说 CDA 只多一个 scalar division，不能完整描述纸面 JSD 的所有运算；有无融合 kernel、复用概率和传输开销应由代码/计时确认。也不能仅凭一句“measurement variance”解释不同门控的明显吞吐差距。

### 7.4 多模态与跨词表

【作者附录 vLLM patches】面向 vLLM v0.19.0：按请求缓存 `pixel_values`/`image_grid_thw`；vision tower 与 embedding merge 每请求一次；为 aug prompt 构建 Qwen3-VL 3D M-RoPE；经 `mm_embed_inputs` 真正把视觉 embedding 送进 Drafter prefill；移除仅允许 full token 数大于 comp 的替换门槛。

这使 VL 小模型读图、大模型只读 caption/OCR。视觉编码固定成本摊到输出 token 上会下降，但视觉 token 的 KV attention 成本仍存在；不能把所有跨模态额外开销都说成随输出变长消失。

Qwen–Llama 异构实验把生成限制为 **109,566 个字符串相同 token 加配对特殊 token**，将小模型 $\delta$ 映射到大模型词表。相同字符串匹配不等于通用 tokenizer 序列对齐；限制输出词表本身改变解码可选空间。需要核查 Floor/Ceiling 是否也施加同一约束，才能公平归因。

代码阶段应优先核对：三路位置对齐、拒绝后 KV、接受式是否与论文一致、JSD 对数底和精度、API-Bank 的实际 Drafter/样本、不同词表 mask、计时范围。已有本地代码是优势，但暂不越过阅读阶段开始运行。

## 8. 实验设置与指标：先明确分母

### 8.1 六个任务与压缩方式

【作者设置】主 Verifier 为 Qwen3-32B，文本 Drafter 默认 Qwen3-4B，API-Bank 附录/吞吐表使用 1.7B；视觉使用 Qwen3-VL-2B。bf16、greedy、关闭 Qwen thinking；文本 $K=2,\beta=1,\gamma=0.5$，MathVista 主结果 $K=4$。论文没有给清晰的所有硬件/重复计时配置清单，吞吐表称单 accelerator、eager mode、无 graph capture。

| 任务 | 样本与目标 | 压缩方式 | 输出/上下文上限 |
|---|---|---|---|
| LongBench | 600，三个多跳子集各 200；F1 | 多文档 summary 代替全文，约 8.1× token 比 | 1024 / 24576 |
| MultiChallenge | 271；LLM judge | 保留最新 turn，较早历史总结，约 7.6× | 2000 / 8192 |
| API-Bank Method A | 正文称 200；单调用 exact API/参数匹配 | 完整 API 文档变为名称和签名，约 7.4× | 256 / 16384 |
| MathVista | 587；accuracy | raw image 变为 Bard caption + EasyOCR | 1024 / 8192 |
| GAIA | 165，其中 web 127、file 38 | web 在线 LLMLingua-2；file 截取 2000 token | 1024 / 32768 |
| SimpleQA | 随机 500；官方 judge | 在线 LLMLingua-2 | 1024 / 32768 |

GAIA-web 和 SimpleQA 使用 smolagents CodeAgent，工具仅 cached DuckDuckGo search 与 visit-webpage，步上限分别 8/6。较早历史（包括 system prompt）压缩目标比例 0.3，最近两 turn 保留，所以实际输入比不等于简单 1/0.3。GAIA-file 是 **single-shot file-QA**，不是完整多步工具循环。

### 8.2 “达到 full 的90%”与“恢复损失90%”不同

【指标展开】令 Floor 分数为 $F$，full-context 参照为 $C$，方法分数为 $A$：

$$
\mathrm{RelativePerformance}=\frac AC,
\qquad
\mathrm{Recovery}=\frac{A-F}{C-F}\quad(C\ne F).
$$

LongBench 整体约 $F=45.0,C=65.5,A=59.7$，因此 $A/C\approx91.1\%$，但恢复压缩损失只有约 $71.7\%$。MultiChallenge 的主表 $23.4\to23.5$，相对 $26.4$ 是约 89%，恢复损失却只有 3.3%。不能把摘要的“约90% full accuracy”解读成“修复90%压缩损失”。

### 8.3 吞吐、FLOPs 与完整 Agent 时间

【作者指标】

$$
\mathrm{Throughput}=\frac{\text{emitted tokens}}{T_{\mathrm{LLM}}},
\qquad
\mathrm{Speedup}=\frac{\mathrm{Throughput}_{\mathrm{method}}}
{\mathrm{Throughput}_{\mathrm{reference}}}.
$$

GAIA/SimpleQA 表注明速度和 FLOPs 是 **LLM-only**。完整 Agent 时间还包括压缩、工具、网络、调度；不能写成整任务 wall-clock 同比例加速，也不能忽略不同方法生成长度与步骤数变化。

【原文 FLOPs 口径的数学整理】使用架构和测得输入长估算每步 prefill：

$$
R_{\mathrm{prefill}}=
\frac{F_L(n_c)+F_S(n_f)+F_S(n_c)}{F_L(n_f)},
\qquad n_f=|x_{\mathrm{full}}|,\quad n_c=|x_{\mathrm{comp}}|.
$$

双 Drafter 路径在此保守计为完整 prefill，即使实际跨步会用 KV。若仅作参数矩阵乘法的第一阶近似，每 token 每参数一次乘加记 2 FLOPs，$F_M(n)\approx2N_Mn$，则：

$$
R_{\mathrm{prefill}}\approx\frac{n_c}{n_f}
+\frac{N_S}{N_L}\left(1+\frac{n_c}{n_f}\right).
$$

这个展开忽略 attention 随序列长度变化等项，是理解比例的近似，不是声称准确复原作者完整架构计算器。它不等于总 decode FLOPs、GPU 占用、实际能耗或费用；论文将 FLOPs 与 energy cost 并提，并没有提供焦耳测量。

原文每步 prefill 比例：LongBench 0.28×、MultiChallenge 0.28×、API-Bank 0.19×、GAIA 0.78×、SimpleQA 0.80×。所以摘要的 0.2–0.3× 不能推广到在线 Agent；API-Bank 的 1.7B Drafter 选择也影响成本。

## 9. 主结果与必须保留的基线

### 9.1 文本能力主表

| 方法 | HotpotQA | 2WikiMQA | MuSiQue | MultiChallenge | API-Bank | 平均速度/Full | Prefill FLOPs/Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Floor | 49.4 | 52.8 | 32.7 | 23.4 | 57.7 | 1.17× | 0.11× |
| Ceiling | 64.9 | 76.5 | 55.0 | 26.4 | 66.1 | 1.00× | 1.00× |
| SD-full | 65.5 | 76.6 | 55.1 | 26.7 | 66.1 | 1.73× | 1.04× |
| SCD-comp | 46.6 | 52.7 | 32.0 | 22.6 | 56.7 | 1.04× | 0.12× |
| RAPID | 63.2 | 75.3 | 52.5 | 25.8 | 64.3 | 1.38× | 1.01× |
| AsymSpec | 64.0 | 66.8 | 48.4 | 23.5 | 63.5 | 1.45× | 0.23× |

成立的结论是：相对 Floor，多跳 QA 和 API 调用有恢复，MultiChallenge 几乎无提升；相对 full-context，仍有质量差距。SD-full 在主表更快且质量更高，但 prefill 计算量更大，因此 AsymSpec 不是对 SD 的全面支配，而是另一种质量—计算量取舍。

**小模型单独运行不能省略**：4B full Drafter 在 LongBench 为 54.6，低于 AsymSpec 59.7；API-Bank 却为 64.0，高于主表 AsymSpec 63.5，也高于 4B+32B 同配对的 60.7；MultiChallenge 4B-alone 为 25.8，主表 AsymSpec 23.5、单次附录 22.5。不是每种能力都能证明组合有价值。

这些表格可能有运行/样本差别，需配对重测；但它们足以要求后续基线必须含 full-context small model，不能只和损失较大的 compressed large model 比。

### 9.2 实测吞吐表与主表不能随意拼接

| 方法 | LongBench | MultiChallenge | API-Bank | MathVista |
|---|---:|---:|---:|---:|
| Floor | 50.0 | 52.3 | 51.2 | 49.2 |
| Full reference | 37.5 | 51.5 | 49.5 | 81.7（VL小模型） |
| SD | 50.9 | 77.5 | 70.0 | 65.6 |
| Fixed-$\gamma$ | 57.8 | 68.6 | 67.2 | 44.4 |
| Tuned-$\lambda$ | 88.2 | 96.0 | 66.6 | 46.9 |
| CDA | 63.3 | 66.6 | 66.1 | 48.0 |

单位 tokens/s。CDA 对 Full 文本参照为 1.69×/1.29×/1.34×，但相对 Floor 分别约 1.27×/1.27×/1.29×。长上下文的总体收益同时含压缩和 speculative-style 生成贡献，应分别比较。

作者说存在 eager 模式运行方差，主表进行了 consolidated/accuracy-equivalent 汇总，但没有完整公开转换流程。比如上表 SD 三文本比值平均约 1.43×，不是主表的 1.73×；tuned-$\lambda$ 与 CDA 在前两项差距也较大。需要运行配置和重复计时日志，不能直接把所有差异解释为方差，更不能将不同表的最佳质量和最佳速度组合成不存在的一个配置。

### 9.3 多模态与 GAIA 参照

MathVista 总体：VL Drafter-alone 60.5%，Floor 44.5%，SD-comp 43.8%，AsymSpec 53.9%。它改善只看 caption 的大模型，但没有超过读原图的小模型；吞吐 48.0 也低于 VL-alone 81.7。此处价值在固定 text-only verifier 的能力扩展，不能说实现视觉任务的质量/速度最优。

GAIA Full：Floor 19.4%，名为 Ceiling 的参照 20.0%，AsymSpec-4B 24.2%，AsymSpec-VL2B 23.0%。4B 方案 LLM 速度为参照 1.41×、prefill 比例 0.78×。SimpleQA 为 63.0/66.0/65.0%，AsymSpec 速度 1.38×、prefill 0.80×。

**GAIA Ceiling 是拼接参照**：web 的 18.9% 来自 32B full；file 的 23.7% 没有对应 32B full-file 运行。表注说 file 参照是 VL-2B-alone，附录文字却说 Qwen3-4B full-file alone，两者冲突。不能据总表宣称“超过同一完整上下文32B Agent”，更不能忽略 file 部分是 single-shot。

## 10. 消融与机理证据

### 10.1 两条信息传递路径

LongBench $K=2,\gamma=0.5$：

| 变体 | F1 |
|---|---:|
| Floor | 45.0 |
| CDA-only，$\beta=0$ | 52.8 |
| CDA + raw $a$ 融合 | 56.9 |
| CDA + SCD-style $(t-b)$ 融合 | 48.0 |
| CDA + $(a-b)$ 融合 | 59.7 |
| Full | 65.5 |

它支持同模型双视图差值优于这里测试的两种替代，且接受和融合都能带来收益；不单独证明“容量偏差被完全取消”。原文缺少对每个位置的正确修正/错误修正计数，也没有把固定门槛、CDA、融合的所有组合完全分离。

### 10.2 深度影响质量，不只影响吞吐

CDA 的 $K=2/4/6$：LongBench 59.7/61.1/58.7；MultiChallenge 22.5/19.6/18.5；MathVista 52.6/53.9/52.1。$K$ 增大改变草稿接受、拒绝融合和未融合 bonus 的出现比例，也改变生成策略，因此质量变化不是传统保分布 SD 中单纯的性能变化。

接受诊断：LongBench $K=2$ 的 AR=0.86、MAL=2.72；API-Bank AR=0.92、MAL=2.84；GAIA AR=0.90、MAL=2.81。这里 MAL 数值超过 $K$，显然包含修正/bonus 的输出进度；不能叫成“每轮接受了超过草稿总数的草稿 token”。

【计数展开】若每轮固定提出 $K$ 个候选，接受前缀长为 $R$，不考虑 EOS：

$$
N_{\mathrm{emit}}=R+1,\qquad
\mathbb E[N_{\mathrm{emit}}]=1+\mathbb E[R],
\qquad \mathrm{AR}=\frac{\mathbb E[R]}K.
$$

例如 $K=2,\mathrm{AR}=0.86$ 得 2.72，与表一致。高 AR 不保证接受 token 与 full-context 大模型一致。

### 10.3 压缩、散度、超参数和小模型容量

- LongBench 截断预算 500/1500/3000/6000/12000：Floor 为 25.8/32.6/39.5/50.6/63.1，AsymSpec 为 52.5/53.7/55.5/59.2/63.9。越强压缩，当前实验的恢复绝对增益越大；轻压缩收益小，不证明在所有真实数据上单调。
- Compressor probe 的 summary/LLMLingua/truncate/question-only 恢复率为 66%/63%/64%/70%。Question-only 丢失很多信息却仍改善，说明 full Drafter 是有效的信息路径，但不能据此断言所有压缩器都同样适合。
- KL vs JSD：LongBench 在 $K=2/4$ 分别 58.6/59.6 对 59.7/61.1；API-Bank 两者相同。支持选择有界散度作为稳定设计，未证明只有 JSD 才有效。
- $\gamma=0.4/0.5/0.6/0.7$ 的 LongBench 为 56.8/59.7/58.8/59.1；不是数值完全不敏感，但没有突变。truncate-1500 的 $\beta=0.5/1/2$、$K=2$ 为 50.3/53.7/53.9，支持该设置下 1 到 2 的变化较小。
- LongBench 小模型 0.6/1.7/4B、$K=2$ CDA 为 47.8/50.8/59.7，表现随容量提高；但 0.6B 仍高于 Floor 45.0。原文“至少1.7B”的建议是这些配置的经验，不能视为理论阈值。小模型两行还使用了作者所谓 accuracy-equivalent KL 版本，未全部采用同一种 JSD 实现。
- 跨族：Llama3B→Llama70B 从 50.6 到 54.2；Qwen4B→Llama70B 到 58.4；Llama3B→Qwen32B 从 45.0 到 47.1；Qwen4B→Qwen32B 到 59.7。说明可迁移但幅度差很大，不能概括为无损跨族通用。

### 10.4 复现时要核查的口径差异

1. API-Bank 正文写 $n=200$，KL/JSD 表注却给 339/534、334/534；主表63.5也与1.7B设置对应，不能假定全部用了默认4B同一数据集。
2. MultiChallenge 主表23.5、$\gamma$表23.3、K/模型表22.5；附录称部分单次 judge、主结果三次 judge，不能简单跨表当严格配对消融。
3. Summary compressor probe 为58.6，主表/机制消融为59.7，设置说明不足以唯一对齐；多模态 patch 文段写恢复53.0，而主表为53.9。
4. GAIA file Ceiling 的4B/VL2B来源冲突；总体“full-context参考”不可直接解释为统一32B。
5. “轻压缩下干预小，因此不会增加 hallucination”的文字推断没有对应幻觉专项测量；greedy 是本实验选择，不是所有 Agent 可靠运行的必要条件。

这些问题用于决定哪些日志与配置是复现必需品，不把找错本身包装成研究贡献。

## 11. 与前九篇以及最近邻的关系

| 对比 | 相同或相关点 | 本质区别 |
|---|---|---|
| AgentSpec | 在 token 层改变 drafting | AgentSpec 的历史 proposal 应由正确目标 verifier 验证；AsymSpec 刻意改变输出策略 |
| SMC | 均允许偏离原 Actor | SMC 跳过工具动作决策；AsymSpec 接受/融合 token，没有宏提交 |
| AOSpec | 两种信息来源共同推进 | AOSpec 预测 observation 后验证真实字节、检查 origin state；AsymSpec 的 CDA 不检查工具状态 |
| SA / SPORK | 小—大模型或自投机并行 | 不能把 AsymSpec relaxed gate 当成工具动作可安全提交的条件 |
| Speculate with Memory | 完整历史可能帮助预测 | 本篇让小模型读取 full history，并改大模型输出；不是只给预测器增加经验检索 |
| SWYR / DSP | 预测质量、长度和成本耦合 | 本篇无 RL 或 TD 深度学习，$K$ 主要固定扫描 |

原文已经列出 **CoSteer、SpecSteer、RAPID、Speculative RAG、SCD** 等近邻：涉及上下文引导、本地—云端非对称、检索分配或对比分数。不能仅查这12篇就认定“完整上下文小模型帮助云端大模型”是新想法；本轮只登记引用，不声称已精读这些额外论文。

下一篇 **Strong Drafts Need Compact Memories** 将与这里比较压缩位置：如果压缩 Drafter 的 KV、Verifier 保留完整输入，保分布目标与本篇完全不同，不能只按“都是压缩+投机”归为一种方法。

## 12. 候选研究问题与最小验证

### A1：上下文分歧究竟表示有用信息，还是不可靠变化

- **依据**：$D=I(X;Z)$ 衡量上下文来源可辨识度，不是正确性；规则默认分歧越大越放宽接受。
- **最小实验**：固定生成前缀，分别删除关键 API 格式、删除无关内容、加入误导/矛盾内容，测 $D$、接受决策和目标正确 token。先比较固定门槛、CDA、简单概率差/裕量门槛。
- **排除条件**：D 已足够区分、误差极少或简单阈值已解决；已有 steering 校准工作直接覆盖。不能泛泛提出“加 confidence gate”，本文已有 gate。

### A2：δ 是否近似了大模型真正的上下文修正

- **依据**：同权重相减不能自动证明大小模型的上下文反应可迁移。
- **最小实验**：只在少量离线样本额外计算大模型 full/comp logits，将小模型 $a-b$ 与大模型对应差值去除 token 常数偏移后比较；测修正方向、正确/错误 token 排名变化及任务类别差异。
- **简单参照**：full Drafter-alone、直接融合 $a$、差值融合、不同固定 $\beta$；必要时用一个线性尺度校准，不先训练大适配器。
- **排除条件**：误差无法预测、只在单模板上有效、或标量缩放已足够。离线诊断要避免把 full 大模型计算又引入在线成本。

### A3：质量增益来自接受路径，还是拒绝融合路径

- **依据**：$\beta=0$ 也改善；$K$ 改变质量；全接受 bonus 不融合。
- **最小实验**：固定前缀和草稿，计数接受、拒绝融合、bonus 三种位置；分别与 compressed/full 大模型 token 对照。做固定/CDA门槛 × 无融合/有融合的清晰消融，再进行完整 rollout。
- **排除条件**：全部增益已由单一更简单分支解释；仅调整 bonus 规则没带来稳定跨任务改善。不直接把“每个位置都融合”当新颖主张，需检索已有 contrastive decoding。

### A4：在什么条件下值得组合两个模型

- **依据**：API-Bank、MultiChallenge、MathVista 都有强 Drafter-alone 参照；报告计算主要是 prefill。
- **最小实验**：从已有 AsymSpec 项目选最小可复现任务，统一数据、精度、prompt、输出上限，比较 S-full、L-comp、L-full、SD-full、AsymSpec 的质量、TTFT/生成时间、峰值内存与实际完整任务时间。
- **排除条件**：S-full 已在质量和成本两方面占优，或压缩器本身已避免关键损失；此时无需增加复杂协作。单纯选择更适合的 baseline 不是独立创新，但能防止误判研究空间。

## 13. 总体评价与后续顺序

AsymSpec 的贡献主要在**推理方法层**：把同一小模型的 full/comp 对比信号用于压缩大模型生成，结合有界散度门槛给出可实现的质量—prefill成本取舍。数学上有支撑的是散度与门槛范围；质量恢复依赖经验，既没有保目标分布，也没有证明上下文偏差被严格消除。

本篇对选题的最重要提醒是：**先问准备保持哪个模型、哪种上下文下的行为，再讨论投机“正确性”**；同时把 full small model、真实运行成本和完整配对配置纳入 baseline。

阅读到第十篇结束。下一篇按路线是 **Strong Drafts Need Compact Memories: Long-Context Speculative Decoding with Compressed KV Cache**。尚未开始项目实现。

## 14. 证据入口

- [本地正文与完整附录](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AsymSpec- Context-Asymmetric Speculative Decoding for Agentic LLMs/main.tex>)：§3 方法、Algorithm 1、附录 CDA 推导、实验设置与 sweep。
- [三路计算及接受/融合图](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AsymSpec- Context-Asymmetric Speculative Decoding for Agentic LLMs/figures/pipeline_0526_v4.png>)。
- [本地待复现项目](/mnt/data/zhangshuai/LLMs/Projects/AsymSpec)：这里只确认存在，以上实现判断仍以论文为依据；代码一致性留待下一阶段。
