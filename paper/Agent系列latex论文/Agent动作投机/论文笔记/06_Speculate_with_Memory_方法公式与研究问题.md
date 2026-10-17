# 06 · Speculate with Memory：方法原理、公式与实现路径

阅读日期：2026-09-09。论文：**Speculate with Memory: Lossless Acceleration for LLM Agents**。[官方版本](https://arxiv.org/abs/2607.12236)：v1，2026-07-14。

已读本地正文、附录、表格和方法图。本地源目录名称末尾有空格，文末链接保留真实路径。本文不套用第五篇的五步 V 型格式，重点记录方法与实现；本轮未运行代码或调用付费模型。

标记：**【原文】**为论文明确内容；**【数学展开】**为其文字定义的数学化；**【实现补充/推导】**为理解或实现所需的补充；**【待确认】**表示原文不足以确定。补充公式不能冒充作者证明。

## 1. 一句话说明方法及其定位

**保持 Actor 不变，把历史动作统计、相似任务片段和投机错误记录写进 Speculator 的提示词，让预测随经验改善。** 它不训练模型权重，也不是直接把历史整条动作链提交给环境。

三个组成部分：

| 组件 | 存什么 | 解决什么问题 |
|---|---|---|
| Contrastive Transition Table | 动作类型转移频次、任务成功/失败计数、参数键集合 | 某动作后通常跟什么动作 |
| Episodic Memory | 情境、动作及参数、observation、参数来源、结果、经验；另存预测失配片段 | 同一动作在不同情境下的具体参数与后续内容 |
| Confusion Tracker | 预测类型与实际类型的混淆计数 | 提醒 Speculator 避免反复发生的预测错误 |

最明确的研究增量是**为预测器积累在线经验**。相似轨迹检索、失败记忆、转移表、memory 组合及选择性路由已经被本文覆盖或明确提出，不能再笼统当作新 idea。

## 2. 三类投机：目标、预执行对象和验证不同

### 2.1 统一符号

【数学展开】令 Actor 在状态 \(s_t\) 上生成动作，环境返回 observation：

\[
a_t=(v_t,\sigma_t)\sim\pi_A(\cdot\mid s_t),\qquad
o_t\sim P_E(\cdot\mid s_t,a_t),\qquad
s_{t+1}=U(s_t,a_t,o_t).
\tag{1}
\]

\(v_t\) 是动作类型，\(\sigma_t\) 是参数；\(s_t\) 包含历史与环境反馈。记：

- \(L_A=\ell_{\mathrm{LLM}}\)：Actor 调用时长；
- \(L_E=\ell_{\mathrm{env}}\)：环境返回时长；
- \(L_S=\ell_{\mathrm{spec}}\)：Speculator 推理时长；
- \(p\)：从可用窗口起点到具体预测可派发的实际耗时；
- b：候选分支数；r：检索条数。原文两处都用 k，本笔记将其分开。

### 2.2 Type 1：预测动作，提前执行工具

Actor 正在思考时，Speculator 预测其动作；合法且安全的调用先执行。正式 Actor 发出动作后，比较类型和全部参数，命中才能复用工具结果。

例：Actor 还在生成 `fetch(url)`，Speculator 已根据当前状态预测 URL 并开始抓取。匹配后只等剩余抓取时间，若已完成才可立即返回。

【原文】命中节省：

\[
\Delta_1=\min(L_E,L_A-L_S),\qquad \Delta_1\leq L_E.
\tag{2：要求预测在 Actor 完成前可用}
\]

【实现补充】加入实际派发耗时并在过晚时停止投机：

\[
\Delta_1^{\mathrm{usable}}=\min(L_E,(L_A-p)_+),
\quad (x)_+=\max(x,0).
\tag{3}
\]

### 2.3 Type 2：预测 observation，提前运行下一次 Actor

环境还没返回时，预测 \(\hat o_t\)，构造假设状态 \(\hat s_{t+1}=U(s_t,a_t,\hat o_t)\)，提前运行**真正的 Actor 模型**。真实 \(o_t\) 到达后验证输入一致，才可复用提前生成的 Actor 结果。

例：物体移动尚未完成，Speculator 预测环境会返回“移动成功”，Actor 在该假设下提前计划下一步。此阶段可只提前运行 LLM，不提前执行它输出的物理动作。

【原文】命中节省：

\[
\Delta_2=\min(L_A,L_E-L_S),\qquad \Delta_2\leq L_A.
\tag{4：要求预测在环境返回前可用}
\]

【实现补充】其无竞争时序是：

\[
T_{\mathrm{base}}=L_E+L_A,
\quad T_{\mathrm{hit}}=\max(L_E,p+L_A),
\quad \Delta_2^{\mathrm{usable}}=\min(L_A,(L_E-p)_+).
\tag{5}
\]

若环境只有 0.1 s 而 Speculator 要 2 s，不能靠把它称为“空闲时间计算”得到有效重叠。

### 2.4 Type 3：预测 observation 和后续动作

在环境等待期间预测 \((\hat o_t,\hat a_{t+1})\)，尝试同时提前运行后续 Actor 与安全工具。它组合前两类机会，但不是免验证 macro commit。

【原文】完全命中时，节省上界为：

\[
0\leq\Delta_3\leq L_{A,t+1}+L_{E,t+1}.
\tag{6}
\]

**原文没有给出一个涵盖所有依赖和部分失配分支的完整 Type 3 时序公式。** 成本附录又采用 \(\ell_{\mathrm{hit}}=L_A-L_S\) 做代表性估算；不能把它当作式 (6) 的通用实现。

【实现补充：一个保守、可审查的串联版本】先在 p 时刻拿到预测 observation，预运行下一 Actor，随后才执行其安全工具。假设状态兼容、全部验证通过、无资源竞争：

\[
T_{\mathrm{base},3}=L_{E,t}+L_{A,t+1}+L_{E,t+1},
\]
\[
T_{\mathrm{hit},3}=\max\{L_{E,t},p+L_{A,t+1}+L_{E,t+1}\},
\]
\[
\Delta_{3,\mathrm{serial\ branch}}
=\min\{L_{A,t+1}+L_{E,t+1},(L_{E,t}-p)_+\}.
\tag{7}
\]

式 (7) **不是作者实际调度的重建**，只展示一个安全串联分支的收益边界；若另行预测动作并更早执行，还需加入其独立派发时刻和状态依赖。不能把式 (3)、(5) 在同一个窗口直接相加，重复计算隐藏时间。

### 2.5 匹配指标与实际提交条件

【数学展开】第 t 步、第 j 个候选的两种命中为：

\[
I^a_{tj}=\mathbf1[\hat v_{tj}=v_t\land\hat\sigma_{tj}=\sigma_t],
\qquad I^o_{tj}=\mathbf1[\hat o_{tj}=o_t].
\tag{8}
\]

Type 1 测安全动作完整匹配；Type 2 测 observation exact match；Type 3 **主指标是后续动作完整匹配**，不等于同时验证 observation 和动作的整链命中率。

安全的复用必须区分：

| 情况 | 可以复用什么 |
|---|---|
| 预测 observation 与真实值一致，Actor 的全部输入也一致 | 提前运行的 Actor 结果；仍需处理随机性和模型配置一致性 |
| observation 不一致，但最终正式动作恰好与预执行工具匹配 | 不能复用错误 observation 下的 Actor 推理；工具结果是否可复用需单独验证动作与环境状态 |
| observation 正确，但预测后续动作不同 | 可考虑保留合法 Actor 计算，错误工具分支作废 |
| 动作名/参数正确，但执行时环境版本不同 | 仅动作匹配不够，需状态一致性条件 |

这也是下一篇 AOSpec 要重点回答的问题，不能在这里提前认定已经解决。

## 3. 三个 memory 组件：数据结构、更新和使用

### 3.1 Contrastive Transition Table

【原文】一条完整任务轨迹为：

\[
\tau=[(v^{(1)},\sigma^{(1)}),\ldots,(v^{(n)},\sigma^{(n)})],
\quad y\in\{+,-\}.
\tag{9}
\]

y 是任务整体成功/失败标签。对相邻动作更新：

\[
\forall t\in\{1,\ldots,n-1\}:\quad
n^{y}_{v^{(t)},v^{(t+1)}}\leftarrow n^{y}_{v^{(t)},v^{(t+1)}}+1,
\]
\[
\mathrm{ArgSig}(v^{(t)},v^{(t+1)})\leftarrow
\mathrm{ArgSig}(v^{(t)},v^{(t+1)})
\cup\{\mathrm{sorted\_keys}(\sigma^{(t+1)})\}.
\tag{10}
\]

【原文】查询动作 i 的后续候选 j：

\[
\mathrm{Confidence}(i\to j)=
\frac{n^+_{ij}+n^-_{ij}}{\sum_u(n^+_{iu}+n^-_{iu})},
\]
\[
\mathrm{SuccessRate}(i\to j)=
\frac{n^+_{ij}}{n^+_{ij}+n^-_{ij}}.
\tag{11}
\]

Confidence 是经验转移频率，不是全参数命中概率；SuccessRate 是任务标签下的关联统计，不证明这一步导致成功。整条失败任务里的正确查找步骤也会增加失败计数。

实现用字典 `(from_type,to_type) → {n_pos,n_neg,arg_signatures}`。无记录时不提供该统计，避免除零；平滑先验、衰减和最小样本数原文未规定。当前动作条件下按频次列出后续类型、成功率和常见参数键，作为提示，不直接强制 Actor 选择。

**参数来源边界**：ArgSig 只存键名，不保存当前值的生成规则。提示样例显示参数签名频率，但式 (10) 的集合本身不能计算频率；若复现该样例，还需额外签名计数器，具体计数协议待确认。ALFWorld 的 turn-level 设置会提供整张表，而非只查一个前驱动作。

### 3.2 Episodic Memory

【原文】每条历史片段：

\[
e=(\mathrm{context},\mathrm{actions},\mathrm{observation},
\mathrm{arg\_sources},\mathrm{outcome},\mathrm{lesson}).
\tag{12}
\]

字段含义：

- context：当前情境的自然语言概括；
- actions：历史动作及完整参数；
- observation：历史环境响应；
- arg_sources：通过与先前 observation 的子串匹配追踪参数来源；
- outcome：任务结果；lesson：模板规则生成的经验。

作者说这些字段使用规则构造，不额外调用 LLM。不是模型自动进行无约束反思。**规则不调用 LLM，不代表更新零 CPU、零存储或零 embedding 成本。**

【数学展开】当前文本 q 和片段文本 z(e) 经 embedding 函数 f 编码：

\[
u=f(q),\quad w_e=f(z(e)),\quad
\mathrm{sim}(q,e)=\frac{u^\top w_e}{\|u\|_2\|w_e\|_2},
\]
\[
\mathcal R_r(q)=\operatorname{TopR}_{e\in\mathcal E}\mathrm{sim}(q,e).
\tag{13}
\]

原文使用 `text-embedding-3-small`、1536 维，报告查询 embedding 少于 50 ms。**这是作者测量，不是当前服务时延保证。** 具体 r、embedding 输入字段拼接、历史向量何时建立、空库策略没有完整给出。

附录有一个容易遗漏的实现细节：**observation 用于检索，不直接写入检索结果 prompt**；arg_sources 则折入动作展示。提供给 Speculator 的是情境、历史动作、参数来源、结果和 lesson。不能把“存完整 observation”误写成“原文将完整历史 observation 全塞给预测器”。

示意（本笔记自构造）：

```text
Situation: 用户询问订单状态。
Agent actions: get_order_details(order_id=旧订单号)
  order_id: 来源为用户消息
Outcome: SUCCEEDED
Takeaway: 先使用本轮用户提供的订单号查询。
```

旧参数是例子，实际预测仍需使用当前用户的订单号。本文的子串来源记录不同于 PASTE 的显式参数 mapper，不能假设完全避免旧值复制。

**依赖时序**：若一次 Speculator 调用必须使用刚检索出的片段，则“embedding → 检索 → prompt → Speculator”是串行依赖，不能全部声称彼此并行。可与正在进行的 Actor/环境工作重叠，但有效 p 应包含它们：

\[
p=L_{\mathrm{embed}}+L_{\mathrm{retrieve}}+L_{\mathrm{format}}
+L_{\mathrm{queue}}+L_S,
\tag{14}
\]

式 (14) 是一种同步检索实现的时间分解；缓存和预取可改变它，作者未给完整流水线。

### 3.3 Speculation-miss episodes

【原文】在 episodic memory 中加入“预测是什么，实际是什么”的失败片段，供后续相似情境检索。例如过去预测 `click(7)` 而实际为 `click(12)`。

【实现补充】最小记录结构：

\[
e_{\mathrm{miss}}=(\mathrm{context},\mathrm{predicted\ target},
\mathrm{actual\ target},\mathrm{target\ type},\mathrm{task\ id}).
\tag{15}
\]

target type 应区分 action 和 observation，以支持三类任务。**原文主要给出 action miss 的格式，但最大提升发生在 observation 预测；Type 2 的实际 miss 序列化方式尚需确认。** 式 (15) 不是已披露的原版字段列表。

### 3.4 Confusion Tracker

【原文文字的数学展开】预测动作类型 i、实际类型 j 的计数：

\[
C_{ij}\leftarrow C_{ij}+\mathbf1[\hat v=i\land v=j],
\quad i\ne j,
\qquad \mathcal H=\{(i,j):C_{ij}\geq3\}.
\tag{16}
\]

达到阈值后，在 Speculator prompt 中写出“此前多次把 j 预测成 i，在这类上下文避免 i”。作者称为 hard constraint，但实际描述是**提示词约束**，不是保证概率为零的 constrained decoding。

没有给出严谨的 context key、计数分母、过期机制或恢复条件。三次错误可能来自三次机会，也可能来自一千次机会，不能据此推导同样的错误概率；跨上下文全局禁止也可能误伤正确候选。Type 2 observation 预测如何映射到这一动作类型计数，同样是待确认细节。

### 3.5 何时更新？

原文一处说每次验证后记录 confusion，另一处说三个系统在任务完成后更新。合理的保守实现是**turn 内暂存事件，任务结束后统一写入用于后续任务的 memory**；具体原版是否支持任务内即时更新需确认。

重要约束：当前任务的未来真实结果和最终 outcome 不得提前进入本任务预测输入。回放时也必须按事件时间处理，不能先把完整测试轨迹入库再预测它自己。

## 4. 组合、候选与未采用的 pruning

### 4.1 七个实验设置

| 设置 | Confusion | Table | Episodic | Miss episodes |
|---|---|---|---|---|
| Stateless | — | — | — | — |
| Confusion | ✓ | — | — | — |
| Table | ✓ | ✓ | — | — |
| Episodic | ✓ | — | ✓ | — |
| Table+Episodic | ✓ | ✓ | ✓ | — |
| Episodic+Miss | ✓ | — | ✓ | ✓ |
| Full | ✓ | ✓ | ✓ | ✓ |

**Episodic 不是纯检索 baseline，它已经包含 Confusion。** 这组是递增/组合消融，不是所有组件的全因子实验，不能从单次差值排除交互效应。

【数学展开】预测器的输入与输出：

\[
P_t=\mathrm{Format}(s_t,\mathcal T_t,\mathcal R_r(s_t),\mathcal H_t),
\quad \hat z_{tj}\sim\pi_S(\cdot\mid P_t),\ j=1,\ldots,b.
\tag{17}
\]

Actor 仍只接收正式状态 \(s_t\)，不接收这些 memory。没有新的训练 loss 或 policy gradient；主要算法就是统计、检索、模板提示与调用匹配。

### 4.2 Best-of-b 与 aggregate accuracy

【数学展开】对可评测事件集合 \(\mathcal V\)：

\[
I_t^{(b)}=\mathbf1[\exists j\leq b:\hat z_{tj}=z_t],
\qquad \mathrm{Acc}_b=\frac{\sum_{t\in\mathcal V}I_t^{(b)}}{|\mathcal V|}.
\tag{18}
\]

这是**所有任务按步骤汇总**，不是先算每任务准确率再平均。\(\tau^2\)-bench 因初始身份未知，排除 turn 0；其他设置的首次/末次调用在收益分析中也必须检查是否真的有后续工作可预启动。

每设置运行三个同 prompt、同 temperature 的预测器；主表报告 b=1，并另报 b=2/3。多个候选未必独立，不能未经检验套用 \(1-(1-p)^b\)。例如 ALFWorld Full 从 40.0% 到 43.8%，增幅有限，不能仅凭扩大候选后有收益就声称多样性问题已解决。

### 4.3 Pruning 的完整规则（最终未启用）

【原文文字的数学展开】过取 3r 个候选；若历史任务失败且该次预测也错，相似度乘 0.5；若主要动作至少混淆三次，乘 0.7：

\[
s'(e)=\mathrm{sim}(q,e)\cdot0.5^{I_{\mathrm{failed\ task\ and\ miss}}(e)}
\cdot0.7^{I_{\mathrm{frequent\ confusion}}(e)}.
\tag{19}
\]

再做相似度去重：若候选与已选片段的 cosine 大于 0.92，则去掉冗余候选，最后保留 r 条。精确排序/过滤执行顺序原文没有细化。

作者发现 pruning 没有明显收益：ALFWorld 40.0%→40.4%，因此正式结果关闭它。原文也已提出按任务/turn 做选择性 memory routing；“加一个 memory router”只能作为后续实验起点，不能直接算新颖贡献。

## 5. 理论实现流程：先复现预测，再单独验证加速

### 5.1 最小数据与模块

```text
TransitionTable: (from_type,to_type) → 成功/失败计数、参数签名
EpisodeStore: 历史片段、向量、来源任务/时间、正常或 miss 标记
ConfusionCounts: (predicted_type,actual_type) → count
Speculator: 组装 prompt → 生成候选 → parser
Verifier: 动作/observation 完整匹配
Executor: 正式/投机 future、状态快照标识、安全动作清单
Logger: 候选内容、memory 来源、各阶段时间、费用、最终复用结果
```

这些是实现职责划分，不是作者公开代码中的类名。

### 5.2 忠实于回放主实验的流程

```text
预先固定 Actor 轨迹与任务顺序
每个 memory setting 使用独立的 memory 状态

for task in ordered_tasks:
    snapshot = 仅来自先前已完成任务的 memory
    pending_records = []
    for event in task 的真实时间顺序:
        只暴露此时已知的历史
        根据 Type 1/2/3 决定预测目标
        查询转移表，embedding 检索相似片段，查询混淆提示
        构建 prompt，运行最多 3 个候选
        真实目标到达后，分别记录单候选和 best-of-b 命中
        将预测/真实结果暂存，不让未来标签回流
    取得任务最终 outcome
    更新转移计数、episode、miss episode、confusion
    写入/更新检索向量；进入下一任务
```

**输入含 memory 的候选生成并非历史 action 的直接回放**：LLM 仍根据当前状态生成完整目标。实现需保存每次使用的 memory 条目 ID，才能查出旧参数复制或跨任务信息泄漏。

### 5.3 真正并行执行时额外需要什么？

在 Actor/环境等待窗口开始后做预测，安全候选进入隔离的 future；正式路径继续运行，不等 Speculator。真实目标到达时：

- 命中且结果已完成：验证状态后复用；
- 命中但未完成：等待剩余时间；
- 未命中：正常执行正式请求，丢弃/取消无用分支；
- Type 2：验证真实 observation 后才接纳提前运行的 Actor；
- Type 3：分别验证 Actor 输入和安全动作，不把动作命中当整链验证。

白名单需按**具体调用语义**确定。网页 `click` 可能仅导航，也可能提交订单；只有动作类型字符串相同，不能保证只读。即使 page fetch 不写远端，它也可能改变本地浏览器会话，仍需隔离。

【待确认】检索 r、memory 长度上限、冷启动初值、过期/去重策略、生成 temperature、完整 prompt、Type 2 错误记录结构、Type 3 分支调度/校验、白名单、取消计费均未给成完整可运行规范。可以据论文实现可审查的版本，但不能声称已经精确复现原系统。

## 6. 成本与延迟公式：保留原文，也注明适用范围

### 6.1 原文的预算估算

【原文】每 100 步的预计节省：

\[
\widehat{\mathrm{Saved}}_{100}=100\times\mathrm{Acc}_b\times\ell_{\mathrm{hit}},
\quad \widehat{T}_{\mathrm{spec},100}=T_{100}-\widehat{\mathrm{Saved}}_{100}.
\tag{20}
\]

成本附录分别使用：

\[
\ell_{\mathrm{hit}}=
\begin{cases}
L_E,&\mathrm{Type\ 1}\text{ 的理想完整隐藏},\\
\min(L_A,L_E-L_S),&\mathrm{Type\ 2},\\
L_A-L_S,&\mathrm{Type\ 3}\text{ 的代表性估算}.
\end{cases}
\tag{21}
\]

它们不是每一步都成立的统一 E2E 公式：Type 1 还受式 (3) 限制，Type 2 需正窗口且有下一 Actor，Type 3 需要完整调度定义。Type 2 成本表给 ALFWorld 的 save/hit=7.1 s，同时文字称环境约 7 s；有正 Speculator 时长时，这不能直接由式 (4) 得到，因此应保留不同场景/口径差异，不能照抄为可保证收益。

【实现补充】更适合日志统计的是：

\[
\mathrm{Saved}=\sum_t I_t^{\mathrm{valid}}\Delta_t-H_{\mathrm{critical}},
\qquad S=\frac{T_{\mathrm{base}}}{T_{\mathrm{base}}-\mathrm{Saved}}.
\tag{22}
\]

\(I_t^{\mathrm{valid}}\) 同时要求严格匹配、安全、有效状态和可预启动的后续工作；\(\Delta_t\) 取逐步真实窗口，H 记录队列、内存检索和资源竞争的额外关键路径开销。各段须避免重叠时间重复计数。

### 6.2 Actor 预运行也要付费

【原文】Type 2/3 的净额外费用写为：

\[
\Delta C=bC_S+(1-\mathrm{Acc})C_A.
\tag{23}
\]

命中 Actor 预调用代替原本下一调用；未命中预调用是额外浪费。**式 (23) 对 b 个 Speculator 却只计一个预运行 Actor 的策略可成立，但不能直接用于每条分支都调用 Actor 的实现。**

【推导】若去重后实际提前启动 B 个、每个成本近似为 \(C_A\) 的 Actor 分支，最多一个被复用，\(p_{\mathrm{reuse}}\) 是有效复用概率：

\[
\mathbb E[\Delta C]
=bC_S+(\mathbb E[B]-p_{\mathrm{reuse}})C_A
+C_{\mathrm{embedding/index}}+C_{\mathrm{wasted\ tools}}.
\tag{24}
\]

式 (24) 假设分支费用相同，具体实现应按各请求账单求和。若 B=b，就有 \((b-p_{\mathrm{reuse}})C_A\)，不是只在失败时多付一个 Actor。

因此，本文“Speculator 便宜、准确率超过约 5% 就有净收益”的说法缺少统一的时间—费用效用定义。**加速可以值得额外花费，但不能由价格比推出净省钱或普遍 break-even 阈值。**

## 7. 实验结论：哪些稳，哪些要限定？

### 7.1 主表准确率（b=1）

| Benchmark | 类型 | 任务数 | Stateless | 最好 memory | 最好配置 |
|---|---|---:|---:|---:|---|
| WebArena | 动作 | 812 | 19.8% | 23.7% | Episodic+Miss |
| VWA | 动作 | 910 | 12.5% | 17.4% | Episodic |
| ALFWorld | observation | 134 | 16.3% | 40.0% | Full |
| PDDL | observation | 60 | 17.6% | 33.8% | Full |
| τ²-bench | chained，动作指标 | 164 | 12.7% | 19.9% | Full |
| HotpotQA | chained，动作指标 | 300 | 20.5% | 27.5% | Full |

Actor 轨迹固定回放：WebArena/VWA 使用 GPT-5-mini，其余使用 GPT-5.4；主 Speculator 为 GPT-4.1-mini。表中是预测率，不是任务成功率，也不是加速倍数；ALFWorld 的约 2.5× 是匹配率提升。

关键消融：ALFWorld Episodic→Epi+Miss 为 23.8%→38.6%；PDDL 为 19.0%→33.0%。失败记录在这两个重复结构场景中贡献很大。相反 VWA Full=16.2%，低于 Episodic=17.4%，不能认为组件越多越好。

### 7.2 在线测量与真实部署证据

- HotpotQA：作者报告 Actor/Speculator 在线时延、610 次预测、40.0% action hit；892 s 节省和 1.05× **明确标为 analytical speedup**，不能写成已完整并行执行测得的 E2E 对照。
- ALFWorld：原生文本模拟器近零环境时延。附录用测得 Actor/Speculator 时长，对环境等待 2–20 s 进行分析；加速从 2 s 时的 1.003×，到 10 s 时约 1.128×，再降至 20 s 时 1.095×。绝对省时与相对加速比不是同一趋势。
- 正文图标为 “Measured” 并声称 live validation，但附录又明确写分析计算及仍需完整 live deployment。**证据口径不完全一致；保守结论是有在线时延测量支撑的估算，尚不足以确认所有报告收益均来自真实并行系统。**
- 开头超过 50% 的图是滑动窗口估计，假设命中可隐藏整个 Actor 调用，不可代替上述有限窗口结果。

### 7.3 统计与 memory 增长

作者报告逐步骤 McNemar 检验 p<0.001，并分析三个预测器或三种任务顺序。步骤来自同一任务且 memory 随时间更新，不完全独立；未来复现宜按任务分组 bootstrap，而不是只依赖逐步显著性。

累计曲线上升不自动证明对固定分布持续学习：任务顺序和组成也在变。三种排序检验提供部分支持；若研究 memory 增长机制，应固定一组测试任务，仅改变此前 memory 内容/数量。

## 8. 与前作的差别，以及不要重复提出的 idea

| 前作 | 与本文的主要区别 |
|---|---|
| Speculative Actions | 本文延续预执行与匹配复用框架，为 Speculator 增加跨任务在线经验 |
| DSP | DSP 学习规划深度控制；本文不训练深度模型，而是用非参数 memory 改善预测内容 |
| PASTE | PASTE 有事件模式与当前参数 mapper、工具/LLM 联合调度；本文强调 embedding 检索完整历史片段及自身错误，不提供同等完整联合 serving 调度 |
| SPORK | SPORK 依赖同模型早期 probe 和 token 恢复；本文使用外部 Speculator + memory，没有 token-level 验证恢复 |
| SWYR | SWYR 联合训练且改变共享模型；本文冻结 Actor，把变化限制在预测器侧提示和 memory |

论文 related work 将 PASTE 概括成 DAG 模式，不能因此覆盖我们对 PASTE v3 的阅读结论：其参数绑定和安全调度已经明确存在。比较应以方法实际组件为准。

已覆盖或已明确提出：历史轨迹检索、失败记忆、混淆抑制、统计+episodic 组合、memory pruning、选择性 memory routing。它们不再是独立的新 idea。

## 9. 保留三个可检验问题，不提前定方向

### M1：记住“任务成功”与预测“Actor 实际会做什么”是否冲突？

**依据**：SuccessRate 用任务结局标注全部转移，prompt 会劝避低成功率动作，但 Speculator 的目标是匹配 Actor，即使 Actor 正在走失败轨迹。

**最小实验**：固定 Actor 与检索片段，只移除/保留 outcome、lesson 和 confusion；在成功/失败任务上分别测全调用匹配及真实可复用时间。比较纯频次、仅 contextual retrieval 与完整系统，统一 token 预算。

**放弃条件**：成功标签在目标任务上没有造成实际预测损失，或简单去掉该字段就足够。仅指出相关不等于因果，或把 success reward 换成 hit reward，本身不足以成文。

### M2：链式动作命中究竟对应多少可安全复用的工作？

**依据**：Type 3 主指标只看动作，而复用 Actor 依赖 observation/输入一致；成本又受实际预运行 Actor 分支数影响。

**最小实验**：在一个可控只读环境上记录 observation hit、action hit、联合 hit、Actor 分支数和真实节省；对比仅 Type 1、仅 Type 2 与链式策略。先核查 AOSpec、SMC 的验证规则，不先造通用回滚系统。

**放弃条件**：后续论文已有同等验证及成本闭环，或链式额外收益太小。优先做最小诊断，难度低到中。

### M3：相似历史记录是不是最有用的投机记录？

**依据**：cosine 检索不直接衡量当前参数可迁移性和延迟收益；Full 可能干扰，pruning/选择性 routing 作者已经讨论。

**最小实验**：固定候选 memory 池，对比 cosine、同工具类型过滤、参数来源检查等简单规则；改变 memory 大小与上下文分布，观察命中增益是否超过额外 prompt/检索耗时。先做不训练 router 的基线。

**放弃条件**：简单过滤已解决；问题只出现在人工构造分布；或最近邻 memory 工作已覆盖。不能以“增加智能 memory 路由”作为未经检索的新贡献。

## 10. 当前结论与下一篇

**本文最有用的结论是：不改变 Actor，也能通过记录经验和预测错误改善 Speculator；但改善多少、该组合哪些 memory，与任务结构有关。** 匹配率收益较清楚，严格状态等价、多分支费用和完整 E2E 加速仍要分开验证。

下一篇 **AOSpec**：重点对照 origin state、action/observation 的联合验证、投机结果提交条件及实际收益模型。前六篇已经为这些问题提供了具体比较对象。

## 来源定位

- [正文方法、公式与附录](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculate with Memory- Lossless Acceleration for LLM Agents /main.tex>)：方法约 113–161 行；评测 178–228 行；成本 318–320 行；状态/在线边界 376–399 行；prompt 404–459 行。
- [主结果](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculate with Memory- Lossless Acceleration for LLM Agents /tables/main_results.tex>)
- [成本表](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculate with Memory- Lossless Acceleration for LLM Agents /tables/cost_benefit.tex>)
- [任务顺序对照](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculate with Memory- Lossless Acceleration for LLM Agents /tables/ordering_effects.tex>)
