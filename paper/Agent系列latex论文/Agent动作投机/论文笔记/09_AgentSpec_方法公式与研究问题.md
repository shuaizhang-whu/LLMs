# 09 · AgentSpec：方法公式、实现原理与研究问题

阅读日期：2026-09-09。论文：**AgentSpec: Speculative Decoding for Batch Inference of LLM Agents**。已读正文、附录、两段算法、实际引用的表格及关键图；未运行代码。本篇继续以方法、完整公式和理论实现为主，不套用第五篇的五步 V 型模板。

## 1. 来源、定位和关键判断

- Xin Wang 等；[arXiv:2608.24004v1](https://arxiv.org/abs/2608.24004v1)，2026-08-25 提交；本次页面仅列 v1，Comments 为 EMNLP 2026。
- [本地论文目录](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AgentSpec- Speculative Decoding for Batch Inference of LLM Agents>)；`main.tex` 引用正文与 `Sections/Apprendix.tex`，`Prefix.tex` 引入 `Table.tex` 中的表宏。只定义但未调用的 `adaTable` 不作为正式附录实验结果。
- 标记：**【作者陈述】**为论文方法/数据；**【本文解释】**为例子；**【补充原理】**为标准算法；**【推导审查】**为数学检查和研究诊断；**【待确认】**为原文缺失的代码细节。

**定位**：这是面向 Agent 批量推理的 **token-level speculative decoding**，不是工具预执行。它不需要独立 Drafter 模型，而是从相同任务、相同语义块的历史 token 中检索草稿，再用目标模型验证；用历史续写的一致程度分配各请求的草稿长度。

**最值得保留的贡献**：把 Agent 的结构信息传入推理引擎，让 token 检索和批预算利用这类信息。**最重要的边界**：缓存隔离、可靠性代理和预算公式不能替代目标模型验证；正文与算法存在公式不一致，预算约束也没有完整写成可直接执行的协议。

## 2. 用一个例子理解它在加速哪里

【本文解释】一个 Reflexion 编程 Agent 会生成解题思路、Python 代码，运行测试，再根据错误修改代码。同一个问题的多轮代码常重复导入、函数定义、输入解析；而分析文字和代码之间的 token 规律不同。

当前生成到 `def solve():` 后，与其在所有任务、所有文本里检索续写，不如优先从**这个任务之前的 Python 块**中检索相同前缀，复制其后的 token 作为草稿。目标模型一次前向计算核验多个位置，接受正确前缀，在首次拒绝处修正。

一个批次中，任务 A 的历史代码出现过多次相同续写，任务 B 的分析文字候选分歧很大。AgentSpec 倾向给 A 更多草稿 token，给 B 较少，从而减少 GPU 为无效草稿付出的验证成本。

这里没有提前运行 `python` 或提交工具状态。代码生成仍通过目标模型验证，完整工具调用正式生成后才进入通常的 Agent 执行循环。历史里曾出现过错误代码也不自动污染最终输出：它只是 proposal，拒绝与否由 verifier 决定。

## 3. 大 batch 为什么会使投机收益下降

### 3.1 原文时间模型，完整保留

【作者公式；Analysis / Overall Efficiency】

$$
\begin{aligned}
\Delta T(b)
&=T_{\mathrm{base}}(b)-T_{\mathrm{spec}}(b)\\
&\approx D(b)(1-\rho(b))t_A-D(b)\bigl(t_D(b)+t_V(b)\bigr)\\
&=D(b)\left[(1-\rho(b))t_A-t_D(b)-t_V(b)\right]\\
&\approx D(b)\left[(1-\rho(b))t_A-t_V(b)\right].
\end{aligned}
$$

| 符号 | 原文含义 |
|---|---|
| $b$ | 当前 batch size |
| $D(b)$ | 这一轮提出的草稿 token 数 |
| $\rho(b)$ | 被拒绝 token / 提出 token 的比例 |
| $t_A$ | 接受一个草稿 token 对应的时间节省 |
| $t_D(b)$ | 每 token 草拟成本 |
| $t_V(b)$ | 每 token 验证成本 |

最后一行忽略 draft 成本。作者将低拒绝率和 $D(b)(t_A-t_V(b))$ 描述为有利因素。直觉是：小 batch 时算力可能有余量，一次多验证若干 token 能摊薄模型调用；batch 已很大时，额外验证更可能直接增加计算时间。

【推导审查】这是近似解释，不是完整 GPU 调度模型：batch 内 token 不按独立任务顺序串行相加，attention、FFN、padding、KV 长度和 prefill 混入都会影响一轮耗时。$t_A$ 也未必不随 batch 改变。对模型式 Drafter，$t_D$ 不总是可忽略；本论文分解表中 EAGLE-3 的 drafting 占 5.72/39.96≈14.3%。

如果 $D$ 是全批草稿总数，$\rho$ 应相应使用 token 加权拒绝率：

$$
\rho_{\mathrm{token}}
=\frac{\sum_i R_i}{\sum_i D_i},
\qquad
\rho_{\mathrm{request}}=\frac1{|I|}\sum_{i\in I}\frac{R_i}{D_i},
\quad I=\{i:D_i>0\}.
$$

这两种平均一般不同。原文表格是 average per-request rejection rate；不能未经转换把该平均值代入全批 $D(1-\rho)$ 当作实际接受 token 数。

### 3.2 原文预算与极端情况

【作者定义；Analysis / Token Budget】

$$
M(b)=AI-b.
$$

原文将 $AI$ 称为部署硬件的 arithmetic intensity，并用 A100、Qwen-3-8B 的 FFN/Attention profiling 讨论 $b>256$ 后 FFN 趋近线性增长。若 $t_V(b)\approx t_A$，其公式进一步写为：

$$
\begin{aligned}
\Delta T(b)
&\approx D(b)\left[(1-\rho(b))t_A-t_V(b)\right]\\
&\approx-D(b)\rho(b)t_A<0.
\end{aligned}
$$

严格小于零还需 $D>0,\rho>0,t_A>0$；零拒绝时在这个忽略 drafting 的极限模型里只是零，不是负值。

【推导审查】Arithmetic intensity 通常是运算量/数据搬运量；硬件的计算/带宽比与 workload 的 intensity 也不是同一对象。它不能不经模型尺寸、精度和算子映射就直接减去请求数。若把 $AI$ 解释为**测出的“等效 token/请求阈值”**，可作为经验预算代理，但论文没有完整给出这个单位转换。

此外 $b>AI$ 时 $M$ 为负，实际预算必须回退/截断；实际验证每个请求处理的 token 数不一定为 1，不能只看请求数。不要把这里的 256 宣称为 A100 上所有模型、上下文和精度的普遍常数。

图 `moti_token_number.pdf` 的 $M$ 在 $b=256$ 时到零，AgentSpec 草稿也趋零；主实验配置为**最大** batch 256，并不代表每个实际解码批次都是 256。因此默认配置下仍加速不直接构成矛盾，后续应记录实际 batch 分布。

## 4. Structure-Isolated Drafting：如何按结构检索草稿

### 4.1 应用提供的结构标识

【作者公式；Method】

$$
S_i=\{a_i,q_i,B_i\},
\qquad B_i=[b_i^1,b_i^2,\ldots,b_i^n].
$$

$a_i$ 是 agent application ID，$q_i$ 是触发当前请求的用户 query ID；$B_i$ 描述语义块，由开始/结束字符串标签定义，例如 Python 代码块、工具调用块、数学表达式。这里的 $a_i$ **不是前八篇笔记中的工具动作变量**。

服务器按三元组定位缓存：

$$
\mathcal H(a_i,q_i,b_i^k).
$$

只从当前应用、当前 query、当前语义块对应的历史片段检索。结构 metadata 是给检索器使用的标签，不是把所有历史都加到目标模型 prompt，也不是推理模型自己做语义理解后分段。

### 4.2 Token 到块边界的在线映射

【作者陈述】字符串边界可能横跨多个 subword token。AgentSpec 缓存 token-to-string 映射 $M$，增量把生成 token 映射为字符，交给轻量字符串下推自动机 PDA $P(a_i,q_i)$，识别当前块。

【实现解释】PDA 维护当前状态、嵌套栈和未完成 delimiter 的前缀；收到代码开始标签入栈，收到结束标签退栈。比如标签的两个部分分别由两个 token 输出，不能只对当前 token 的字符串做完整等值比较。

实现时还要处理 byte fallback、Unicode、多 delimiter 共前缀、标签字面文本和跨块 token。直接拼接单 token 的任意 `decode()` 结果不一定等价于 tokenizer 的完整输出；论文只给设计方向，没有全部 tokenizer 适配代码。

缓存可按 $(agent,query,block)$ 共享，但同时进行的请求必须各有解析进度，不能因 query 相同就共享一个可变 PDA 状态。只把**正式接受的 token**写入可复用历史；若先为草稿更新解析状态，则必须在拒绝后恢复，否则错误草稿会污染后续分块。

### 4.3 查找历史续写

【作者算法 1】更新 PDA → 找当前块 → 定位该块历史 → 历史为空则不投机 → 从历史提取候选续写。

【补充实现形式化】设当前已接受前缀为 $x_{1:t}$，匹配后缀长 $k$，历史块 $h$ 的位置 $u$ 满足：

$$
h_{u-k+1:u}=x_{t-k+1:t},
\qquad h\in\mathcal H(a_i,q_i,b_i^k).
$$

候选可取其后连续 token：

$$
c=h_{u+1:u+L}.
$$

这里用 $k$ 表示匹配长度，与块的上标只是记号重用；实现应分开命名。原文没有规定具体 suffix/ngram 匹配长度、匹配失败回退、块结束处截断以及多候选选择的全部规则，上述是可实现的通用检索表达，不冒充作者精确索引算法。

候选的“出现次数”需要保存。算法 1 用集合并集 $CT_i\cup\{c\}$，算法 2 又用历史候选数算频率；若真的对相同完整续写去重，会丢失支持数。因此实现应明确保存出现位置/计数的 multiset，或说明作者采用何种去重口径。

## 5. Redundancy-Aware Budget Allocation：全部公式与疑点

### 5.1 冗余评分

【作者正文公式】

$$
g(c,n)=\frac cn\,p(n),
\qquad
p_{\mathrm{text}}(n)=g_{\min}+(1-g_{\min})(1-e^{1-n}).
$$

$n$ 是同一 query/块内候选续写数，$c$ 是同意最常见续写前缀的候选数。$c/n$ 表示共识比例，$p(n)$ 对支持不足的情形降权。$g_{\min}$ 是低支持情况下的最低置信系数，通常需在 $[0,1]$ 范围内理解。

正文式在 $n=1$ 时 $p(1)=g_{\min}$，$n\to\infty$ 时 $p(n)\to1$。

**【原文不一致】附录算法 2 实际写成：**

$$
p_{\mathrm{algo}}(n)=g_{\min}+(1-g_{\min})(1-e^{-n}),
\qquad g_i=\frac{c_i}{n_i}p_{\mathrm{algo}}(n_i).
$$

两者在 $n=1$ 已不同：

$$
p_{\mathrm{algo}}(1)=1-(1-g_{\min})e^{-1},
\qquad
p_{\mathrm{algo}}(n)-p_{\mathrm{text}}(n)
=(1-g_{\min})(e-1)e^{-n}.
$$

例如示意取 $g_{\min}=0.2$，单一候选会得到 0.2 或约 0.706，可能改变低支持请求的预算。原文未给出用于实验的明确版本；本笔记保留两式，不能静默挑一个并称复现完成。

【待确认】“最常见前缀”如何兼顾长度与频次？如果任意长度都可比，空前缀或很短前缀最容易获高频，必须另有长度/阈值规则。本文没有给全，也没有给完整 $\alpha,g_{\min}$ 配置。

这是启发式分数，不是校准后的真实接受概率；和 SMC 的 Beta 可信下界不是同一种量。重复历史可能来自相邻轮次的高度相关文本，多次出现不等于多份独立证据。

### 5.2 总预算与逐请求长度

【作者正文及算法共有公式】

$$
B_t=\left\lfloor\frac{\alpha}{bz}\right\rfloor,
\qquad G=\sum_jg_j,
$$

$$
L_i=\max\left(|CT_i^*|,\ B_t\frac{g_i}{G}\right).
$$

$bz$ 是实际 batch size，$\alpha$ 控制预算，$CT_i^*$ 是该请求选定的高频续写前缀。正文写 $CT_i$，附录区分候选集 $CT_i$ 和前缀 $CT_i^*$；笔记采用后者以免集合大小与 token 长度混淆。

文中把 $B_t$ 叫 total/batch-level speculative budget，规则意在按相对冗余分配，同时至少保留可靠前缀。

### 5.3 公式没有自动满足预算守恒

【推导审查】当 $G>0$：

$$
\sum_i B_t\frac{g_i}{G}=B_t,
\quad\text{但}\quad
\sum_i\max\left(|CT_i^*|,B_t\frac{g_i}{G}\right)\geq B_t.
$$

例如 $B_t=8$，两个请求评分相同，各有 6-token 前缀，公式分配 $6+6=12>8$。这是式子本身的反例，不是声称作者运行代码一定越界；引擎可能还有未写出的截断/准入。

还有五个必须补齐的接口：

1. $n_i=0$ 时算法令 $g_i=0$，但没有定义该请求的 $CT_i^*$ 和最终 $L_i$。
2. 全部 $g_i=0$ 时 $G=0$，比例除零；应明确退回普通解码。
3. $L_i$ 可能不是整数，取整后怎样分配余数未说明。
4. 比例预算可能大于可用候选长度；没有独立 Drafter 时，不能凭空延长 token。
5. $B_t=\lfloor\alpha/bz\rfloor$ 与 $M(b)=AI-b$ 是两种不同规则，原文没有写出二者的约束关系、$b\geq AI$ 的处理或额外 scheduler cap。

因此主文“所有草稿数均低于 $M(b)$”是所示图上的观察，不能直接从分配公式推出。

### 5.4 可执行的边界处理，不当作新方法

【实现补全示例；不是原文算法】给定引擎提供的非负硬预算 $B_{\mathrm{hard}}$，可令每请求无候选时 $L_i=0$；有候选时先按一个明确版本评分，再将需求整数化、限制在可用候选长度内，最后裁到全批上限：

$$
0\leq L_i\leq |\widehat c_i|,\quad L_i\in\mathbb Z,
\qquad \sum_iL_i\leq B_{\mathrm{hard}}.
$$

若所有“至少保留前缀”的长度之和已经超过硬预算，就无法同时满足全部下限，必须选择请求或截断前缀。用按权重分配、最大余数取整和简单截断便可做一个参照实现；这是补齐协议的选择，不是论文原式或已确认创新。

## 6. Token 如何验证，为什么可以不改变目标分布

AgentSpec 详细描述 proposal 与 budget，但没有在两段算法中展开 verifier。不能因此省略理解它所必需的验证原理，也不能擅称作者用了某个具体 sampler。

### 6.1 Greedy 情况

【补充原理】给定已接受前缀 $x$，历史检索草稿为 $d_{1:L}$。目标模型在一个因果前向中计算：

$$
p_i(v)=P_\theta(v\mid x,d_{<i}),\quad i=1,\ldots,L+1.
$$

接受到第一个不匹配位置之前：

$$
r=\max\left\{k\in\{0,\ldots,L\}:\forall i\leq k,
d_i=\arg\max_vp_i(v)\right\}.
$$

若 $r<L$，输出 $d_{1:r}$ 及目标模型在 $r+1$ 位置的正确 greedy token，丢弃后面的草稿计算；全匹配时再输出目标模型的第 $L+1$ 个 token。忽略 EOS 截断，一次前向可得到 $r+1$ 个正式 token。前缀之后的错误草稿绝不能越过拒绝位置被接受。

### 6.2 非零温度需要匹配正确的抽样协议

【补充原理；依据标准 speculative sampling 的 Algorithm 1，非 AgentSpec 独有公式】设 proposal 分布为 $q_i$、目标分布为 $p_i$，均已包含应有的 temperature/top-p 等变换。对 $d_i\sim q_i$：

$$
A_i=\min\left(1,\frac{p_i(d_i)}{q_i(d_i)}\right).
$$

采样 $u_i\sim U[0,1]$，当 $u_i\leq A_i$ 时接受；首次拒绝时从残差分布采修正 token：

$$
p_i^{\mathrm{res}}(v)
=\frac{[p_i(v)-q_i(v)]_+}{\sum_{w\in\mathcal V}[p_i(w)-q_i(w)]_+},
\quad[z]_+=\max(z,0).
$$

全接受时从 $p_{L+1}$ 采额外 token。拒绝概率为零时不进入残差分支。其正确性在单位置上来自：

$$
P(\mathrm{output}=v)
=\min(p_i(v),q_i(v))+[p_i(v)-q_i(v)]_+=p_i(v).
$$

来源：[Leviathan et al., Fast Inference from Transformers via Speculative Decoding，§2.3 / Algorithm 1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)。

确定性历史草稿可形式化为 $q_i(v)=\mathbf1[v=d_i]$，此时接受概率是 $p_i(d_i)$，拒绝时应从排除 $d_i$ 后归一化的目标分布修正。也可以用等价的目标抽样比对机制；**不能把 greedy 的 argmax 比对直接用于有温度采样并声称保分布**。实际 vLLM 的历史 proposal sampler 需要代码确认，不能从本笔记推导反推其实现。

### 6.3 结构选择影响效率，不应擅自约束 verifier

只有 proposal 来源/长度改变、目标模型的输入和采样规则不变且验证正确，才能继承上述保证。任意频率评分不准只会降低效率，不能授权跳过验证。

系统图含 CFG/PDA、Logits 和剪切标记，但正文算法仅明确说明跟踪块和过滤历史。若实际修改目标 logits 施加额外 grammar mask，参照分布就会变化；必须查代码确认，不能仅因类似 XGrammar 就自行推断它修改或不修改了 logits。

Actor 输入里的工具结果仍依赖真实环境。模型 token 分布不变，不等于整个含时间、网络和非确定性工具的 Agent 逐次轨迹必然完全一样。

## 7. 理论实现流程与复杂度

### 7.1 最小模块

| 模块 | 实现所需状态 | 关键边界 |
|---|---|---|
| Agent 请求适配 | agent/query ID、结构 delimiter 描述 | 同一 query 跨多次 LLM 请求保持身份 |
| 增量解析 | token 字节/字符映射、PDA、未完成标签 | 各请求独立进度；拒绝后恢复 |
| 历史索引 | 接受过的 token，按 query/块分区的匹配位置 | 保留频次；控制缓存清理和冷启动 |
| 草稿检索与评分 | 候选续写、前缀计数、评分版本 | 最常见前缀长度规则需明确 |
| 批预算器 | 当前实际 batch、引擎硬预算、整数长度 | 零分数、候选不足、超限处理 |
| Verifier | 目标模型输出概率、接受前缀、修正 token | 正确采样；同步 KV 与 PDA 状态 |

### 7.2 可转成代码的主流程

```text
on agent request:
    receive prompt + agent_id + query_id + structure
    create request-local parser; attach shared history partitions

on each serving iteration:
    for request in active batch:
        update parser from newly accepted tokens
        identify current semantic block
        candidates = retrieve occurrences matching current suffix
                     from history[agent_id, query_id, block]
        if candidates empty:
            draft = []; score = 0
        else:
            prefix, support, count = select_frequent_prefix(candidates)
            score = (support / count) * support_penalty(count)

    nominal_budget = floor(alpha / actual_batch_size)
    proposed_lengths = paper_allocation_rule(prefixes, scores, nominal_budget)
    lengths = enforce_explicit_integer_and_engine_limits(proposed_lengths)
              # 此处是缺失实现协议，需与作者代码核对

    run target forward on selected draft prefixes
    verify using the deployment sampling policy
    append accepted draft prefix and correction/bonus token only
    discard invalid suffix KV / restore provisional parser state if needed
    update history using accepted tokens
    return completed LLM outputs to ordinary agent tool loop
```

需要引擎内部接入；给普通远端聊天 API 多传几个 metadata 字段，并不会自动获得这些能力。结构字段只是应用到改造过的 serving 后端的接口。

### 7.3 原文复杂度及假设

【作者附录公式】以 $K$ 为候选数、$L$ 为草稿长，检索/扩展为 $O(KL)$，PDA 每 token 摊还 $O(1)$，过滤与评分为 $O(K)$：

$$
T=O(KL)+O(K).
$$

历史规模 $H$、语义块索引规模 $B$、请求数 $R$：

$$
S=O(H+B+R)=O(H).
$$

最后等号依赖 $B,R$ 不超过历史量级，以及解析栈和 schema 规模受限。一般 PDA 的栈随嵌套深度增长，并非天然常数空间；索引查找“常数时间”也依赖具体索引与匹配规则。原文没有给完整数据结构和 worst-case 证明，属于有实现假设的复杂度说明。

### 7.4 当前可复现性

已知 vLLM v0.12.0 V1、FlashAttention-2、A100 80GB，默认最大 batch 256；metadata 为 `structure`、`query_id`、`agent_id`。示例 API 给 temperature=0.6，但不等于所有实验都用此采样参数。

本地材料未提供 AgentSpec 的源码项目或项目链接；本轮未做全网仓库穷尽搜索，不将“未提供”写成永久不开源。缺失项包括评分版本、$\alpha,g_{\min}$、预算硬上限、频繁前缀算法、缓存生命周期、各 workflow 的 schema、数据子集、采样/请求到达配置与目标模型及 EAGLE checkpoint。

附录的 API Listing 是示意：多行结构字符串的引号/转义和工具结束标签不能直接复制运行。源码 `main.tex` 中 `0_abstract` 与实际 `0_Abstract.tex` 大小写也需在 Linux 编译时处理。这些只记作复现备注，不把排版和代码示例错误当作研究创新。

## 8. 实验设置、指标与主要结果

### 8.1 覆盖范围

【作者陈述；Experiments / Appendix】

| Workload | 框架/数据 | 类型 |
|---|---|---|
| Code Generation | Reflexion；USACO Bronze/Silver/Gold/Platinum | 多轮工作流 |
| Deep Research | LangChain DeepResearch；DeepResearch-Bench | 多轮工作流 |
| SWE-Bench-Lite | OpenHands | 工具 Agent |
| GAIA | OpenHands | 工具 Agent |
| Spec-Bench | 非 Agent 的多类生成任务 | 泛化对照 |

所以摘要的五种 workload 不等于五个 Agent benchmark，而是四个 Agent workload 加一个非 Agent benchmark。

主表覆盖 Qwen-3-8B、GPT-OSS-20B、DeepSeek-R1-Distill-Llama-8B；另用 MiMo-7B（附录写 MiMo-7B-RL MTP 模块）比较 MTP。四模型并没有全部在每一个 benchmark/每一种 baseline 上做完整笛卡尔积。

基线为普通 autoregressive、EAGLE-3、NGram、SuffixDecoding；MTP 在 MiMo 上单独比较。附录说基线使用 vLLM 官方实现与默认配置；明确列出的 NGram 是 draft length=5、ngram lookup max=4，SuffixDecoding length=32。所有方法固定相同最大 batch/serving 设置，默认最大 batch 为 256。

数值通常是五次不同随机种子的测量平均。当前表格未报告对应标准差或置信区间，也没有完整任务质量、token 一致性或采样分布实测表。**五次测量提升可信度，但不能替代误差报告或正确性实现检查。**

### 8.2 Goodput 与延迟不能混为一谈

【作者指标的形式化】

$$
\mathrm{Goodput}=\frac{N_{\mathrm{generated\ tokens}}}{T_{\mathrm{entire\ workload}}},
\qquad
\mathrm{Speedup}_{X/AR}=\frac{\mathrm{Goodput}_X}{\mathrm{Goodput}_{AR}}.
$$

论文将此称为 goodput，但定义并没有额外加入“满足 SLO 的 token”筛选。应说明计数的是正式输出 token，不能把拒绝草稿计为有用输出；原文没有展开所有日志计数细节。

单 query latency 则从用户请求进入 Agent 环境开始，直到最终答复完成，包含多轮生成及工具等待。**吞吐提高 2× 不自动等于每个任务延迟减半**；它还受生成长度、并发、工具占比和到达模式影响。

### 8.3 四个 Agent workload 的完整方法加速比

下表按主表对 autoregressive 的比值，不混入作者括号内的“相对最佳投机基线提高百分比”：

| 模型 | Deep Research | Code Generation | GAIA | SWE-Bench |
|---|---:|---:|---:|---:|
| Qwen-3-8B | 1.07× | 1.30× | 1.07× | 1.31× |
| GPT-OSS-20B | 1.48× | 2.02× | 1.19× | 1.69× |
| DeepSeek Distill 8B | 1.31× | 1.36× | 1.12× | 1.59× |

主表支持在这些配置下全部超过 AR 和列出的投机基线。很多默认投机基线在大最大 batch 设置下慢于 AR，但不能据此声称所有调优过的 speculative decoding 都不适合大 batch。

**【数值核对】**表中部分绿色增益标注不能由本行 tokens/s 重建。例如 GPT-OSS CodeGen 的 AgentSpec=584.43，最强投机基线 SuffixDecoding=295.85，增益约 $584.43/295.85-1=97.5\%$，不是标出的 104%；Qwen SWE 的 854.69 对 556.64 约为 53.5%，不是标出的 42%。正文的“最高 104%”应记为作者报告，不能直接当作已被表内数值验证。以原始 tokens/s 和明确分母重新计算。

### 8.4 泛化、MTP 和尾延迟

- USACO 难度子集：正文报告最高约 2.2×；这与主表“全 CodeGen 的最高 2.02×”是不同统计范围。
- MiMo CodeGen：完整方法全数据 3,318.99 tokens/s、相对 AR 1.29×；MTP 为 1,875.18、0.73×。分难度 AgentSpec 为 1.20×/1.21×/1.16×/1.34×，MTP 为 0.70×/0.65×/0.80×/1.18×。
- H100 上 GPT-OSS 的 CodeGen/DeepResearch/SWE/GAIA 加速为 2.28×/1.67×/1.92×/1.54×；对应 A100 为 2.02×/1.48×/1.69×/1.19×。这支持该模型的跨 GPU 实验，不证明 $AI$ 预算无需重新测量即可泛化。
- CodeGen 尾延迟：作者报告 P90 和 P99 分别有最高 1.47×、1.39× 的缩短因子，换算约 32.0%、28.1% 减少；不是降低 147% 或 139%。
- Qwen 非 thinking 模式的图/正文报告超过 2.5×，是另一模式设置，不能取代主表数字。

Spec-Bench 全集：AR=697.42，AgentSpec=796.01 tokens/s，约 1.14×。但“每个子集都胜过所有基线”的文字不受表格支持：

| 子集 | AgentSpec | 更高的基线（tokens/s） |
|---|---:|---|
| Summarization | 1,481.90 | EAGLE-3：1,552.62 |
| Translation | 2,294.12 | EAGLE-3：2,319.01 |
| Math Reasoning | 1,075.43 | SuffixDecoding：1,121.48 |

结论应是 **Spec-Bench 全集有收益、各子集均快于 AR，但不是每个子集都最佳**。不照搬误标的粗体和全称叙述。

## 9. 消融与机理闭环

### 9.1 拒绝率与执行分解

Reflexion workload 平均每请求拒绝率：EAGLE-3 53.3%、NGram 88.7%、SuffixDecoding 65.9%、AgentSpec 26.4%。这说明当前候选选择/预算组合显著减少浪费，但不能仅凭该表把差异全部归到“语义隔离”，因为草稿长度变化本身也影响拒绝率。

执行分解（分钟）：

| 部分 | EAGLE-3 | NGram | SuffixDecoding | AgentSpec |
|---|---:|---:|---:|---:|
| Draft | 5.72 | 0.68 | 1.41 | 1.74 |
| Verification | 34.24 | 39.82 | 33.12 | 26.11 |
| Overall | 39.96 | 40.50 | 34.53 | 27.85 |

AgentSpec 相对 SuffixDecoding drafting 多约 0.33 分钟，verification 少约 7.01 分钟，总时间少约 19.3%。这是与“多一点检索处理，换取少做无效验证”一致的证据。

正文一处写它的 overall drafting cost 低于所有基线，但表中 1.74 明显高于 0.68/1.41，应以表为准。作者另称新增处理开销低于 2 ms，但没有完整说明该值的统计单位/分布，不能将它误读为整个 workload 只增加 2 ms。

### 9.2 S/R/N 三个变体

【作者定义】S 仅使用 structure-isolated drafting；R 使用 SuffixDecoding 加 redundancy budget；N 不提供显式 semantic structure；Full 为完整方法。

Qwen-3-8B：

| 方法 | Deep Research tokens/s（相对 AR） | CodeGen tokens/s（相对 AR） |
|---|---:|---:|
| AR | 617.41（1.00×） | 636.09（1.00×） |
| SuffixDecoding | 598.74（0.97×） | 579.21（0.91×） |
| AgentSpec S | 647.12（1.05×） | 795.54（1.25×） |
| AgentSpec R | 623.33（1.01×） | 702.28（1.10×） |
| AgentSpec N | 661.08（1.07×） | 809.12（1.27×） |
| Full | 661.43（1.07×） | 828.10（1.30×） |

可以说 Full 比单独 S/R 好，且在这两项上 S 比 R 好。但 N 几乎追平 Full：完整显式结构相对 N 只增加约 **0.053%**（DeepResearch）和 **2.35%**（CodeGen）吞吐。

这不否定结构方法，也不证明“结构毫无作用”；N 保留了哪些 query 隔离、评分与回退细节未完全写清。它说明我们需要区分**同 query 限定、显式块标签、预算规则**三者的贡献，不能直接把 Full 的全部优势归因于“理解语义结构”。

### 9.3 重复性和接受率只是相关证据

原文在 USACO/Qwen 上统计：历史重复主要集中于同 query、同 block，且高冗余与较高接受率相关。

重复率定义可完整形式化为：令 $\mathcal I_k$ 为历史中有至少 $k$ 个连续 token 匹配的生成位置区间，将重叠区间合并，则：

$$
\mathrm{RepeatedRatio}(k)
=\frac{\left|\bigcup_{I\in\mathcal I_k}I\right|}{N_{\mathrm{generation}}}.
$$

分组按同/不同 query、同/不同 block 统计。原文没有充分规定一个区间同时匹配多个分组时的唯一归属规则；该图用于展示趋势，不是“某个标签因果地提高接受率”的证明。

候选评分与真实接受长度的进一步区别是：若草稿接受前缀长为 $R$，则：

$$
\mathbb E[R]=\sum_{\ell=1}^{L}P(R\geq\ell).
$$

长度 $L$ 的收益取决于每个位置的生存概率，不只是“整个前缀有多少历史候选同意”。用单一 $g_i$ 比例分配是启发式，未证明等价于最大化全批输出进度或最小 wall time。

### 9.4 闭环完成到什么程度

有实验链条支持：重复有局部性 → 结构/任务条件检索减少拒绝 → 验证总时间下降 → 多 workload 吞吐提高。

还未充分闭环的部分：语义块本身的独立贡献、不同历史支持的概率校准、分配公式怎样严格遵守引擎预算、$AI$ 到 token 容量的映射、默认静态长度基线充分调参后的差距、缓存内存/解析尾开销，以及质量/分布保持的实现检查。

归类上属于系统方法与经验分析贡献。它解释了具体性能瓶颈，但没有据此证明一个适用于所有大 batch Agent 服务的普遍规律；哲学层未触及，不强行拔高。

## 10. 与前八篇的关系

| 论文/分支 | 历史或投机的对象 | 验证/提交对象 | 与 AgentSpec 的关系 |
|---|---|---|---|
| SA、PASTE | API/工具动作与依赖模式 | 工具调用及可复用结果 | 外层动作循环；不能用 tokens/s 直接比较工具加速 |
| DSP | Agent 规划草稿长度 | 规划匹配前缀 | 都控制“深度”，但决策单位、开销和反馈不同 |
| SPORK D3 | 失败工具草稿中的 token | 目标模型 token 验证 | 已有动作与 token 复用结合；AgentSpec 新增的是结构历史和批预算 |
| SWYR | 训练下一调用预测 | 工具调用命中评估 | AgentSpec 无此训练，不改变模型参数 |
| Speculate with Memory | 预测器侧转移表、经验和失败记录 | 动作/observation 预测 | AgentSpec memory 是检索原始 token 续写的索引，不是给模型读经验提示 |
| AOSpec | 返回预测与隔离目标动作 | observation 字节、目标动作及 origin state | 不解决相同层次；可以成为外层组合，但组合收益须测资源竞争 |
| SMC | 历史 macro 与已执行动作链 | anchor 后近似跳过决策 | AgentSpec 保留 token verifier，不依据历史频率直接省略目标策略验证 |

后续 AsymSpec 会改变 Drafter/Verifier 的上下文关系；AgentSpec 这里主要改变 proposal 来源与长度，不明确压缩目标模型的语义上下文。不要因为二者都有历史信息处理就混为同一种保证。

**已有工作覆盖的 idea**：同 query/语义块隔离检索；给 serving 传 Agent 结构标签；用候选共识及支持数指导 token 预算；不训练独立 Drafter 的历史续写投机；无显式结构的回退变体。不能简单把“Agent-aware memory + 动态 token 预算”当空白。

## 11. 留给后续的可检验问题

### T1：收益来自语义结构，还是更简单的历史隔离

- **证据**：N 接近 Full；现有 S/R 消融不能完整分离 query、块和预算。
- **最小实验**：固定同一历史、同一 token 总预算，做 global/query-only/block-only/query+block 的条件对照；再加随机分区或等容量分区作为控制，区分语义标签与单纯缩小搜索空间。
- **指标**：接受长度、拒绝 token 总量、检索延迟和真正输出吞吐，不只报接受率。
- **排除条件**：query-only 已获得几乎全部收益，或随机/容量匹配划分一样有效；此时不先设计更复杂“语义 memory”。这些实验是归因诊断，不自动构成新论文。

### T2：单一冗余分数是否足以决定草稿长度

- **证据**：$g_i$ 是历史共识代理，$\mathbb E[R]$ 依赖逐位置接受概率；正文/算法低支持惩罚不同。
- **最小实验**：固定 proposal，对前缀长度记录 $P(R\geq\ell)$、真实验证增量成本；比较调优固定长度、原比例规则、简单按历史接受长度分配以及按每新增 token 边际收益分配。
- **候选目标形式**：在硬预算 $\sum_iL_i\leq B$ 下比较 $\sum_i\mathbb E[R_i(L_i)]$，进一步才考虑批耗时 $C(\mathbf L,\mathbf s)$，其中 $\mathbf s$ 为请求上下文长度。
- **排除条件**：简单长度截断已经足够，收益只来自修复未定义边界，或 AdaSpec/SPIRe/相关预算分配工作已有完整方案。不能把修正 `max` 直接包装成算法贡献。

### T3：预算是否应取决于真实验证负载而不只是 batch size

- **证据**：$M=AI-b$ 的单位映射和模型依赖未说明；同样 $b$ 可以有不同上下文长度、候选长度和 prefill 混合。
- **最小实验**：先用少量微基准在固定 $b$ 下改变上述因素，记录多验证一个 token 的延迟增量；比较常数阈值、按上下文分桶查表、现有动态预算规则。
- **排除条件**：一个调好的简易阈值足够，或目标部署负载变化太小。这是既有 batch scheduling 领域的问题，需最近邻查证后才可主张贡献。

### T4：与动作投机组合时，收益是否被 LLM 分支负载抵消

- **证据**：AOSpec/Memory 的 provisional Actor 可增加请求数与 KV；AgentSpec 的收益依赖可用验证空间，PASTE 已研究外层工具—LLM 联合调度。
- **最小实验**：未来已有动作 baseline 跑通后，只做一个受控 2×2：动作投机开/关 × 历史 token 投机开/关，固定硬件、任务和候选预算，记录批分布、LLM/工具时间、额外分支和 E2E。
- **排除条件**：收益只是两个独立优化的简单叠加，或者简单限流已消除干扰。暂不构建新的全栈调度系统，也不把“把两种投机结合”本身当创新。

## 12. 本篇结论与下一篇

AgentSpec 将 Agent 结构信息用于 **token 草稿检索与批预算**，其主要实验支持在所选大最大 batch 配置下提高吞吐。低拒绝率、较少验证时间与吞吐收益相互呼应；但结构标签的独立增益、严格预算实现和部分表格全称结论需要收窄。

对我们最有价值的是两个区分：**历史模式用于提出草稿，和历史模式用于直接提交动作，是不同的正确性问题；分数更高、接受率更高，也不自动意味着全系统净收益更高。**

阅读进度到第九篇结束。下一篇 **AsymSpec: Context-Asymmetric Speculative Decoding for Agentic LLMs**，重点审查 verifier 使用压缩上下文后的参照分布、保真条件和可复现代码。

## 13. 本地证据入口

- [时间模型、预算动机](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AgentSpec- Speculative Decoding for Batch Inference of LLM Agents/Sections/3_Analysis.tex>)。
- [结构隔离与评分、预算公式](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AgentSpec- Speculative Decoding for Batch Inference of LLM Agents/Sections/4_Method.tex>)。
- [附录算法、vLLM 配置、复杂度](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AgentSpec- Speculative Decoding for Batch Inference of LLM Agents/Sections/Apprendix.tex>)。
- [实验论述](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AgentSpec- Speculative Decoding for Batch Inference of LLM Agents/Sections/5_Experiment.tex>)与[表格定义](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AgentSpec- Speculative Decoding for Batch Inference of LLM Agents/Table.tex>)；`adaTable` 未被实际调用，不能当正式附录结果。
- [系统图](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AgentSpec- Speculative Decoding for Batch Inference of LLM Agents/Figures/AgentSpec.pdf>)、[预算图](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AgentSpec- Speculative Decoding for Batch Inference of LLM Agents/Figures/moti_token_number.pdf>)。
