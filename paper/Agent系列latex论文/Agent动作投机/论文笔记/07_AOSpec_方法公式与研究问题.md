# 07 · AOSpec：方法公式、实现原理与研究问题

阅读日期：2026-09-09。定位：动作与 observation 联合投机；本篇不套用第五篇的五步 V 型模板，重点是方法、公式、可实现条件和选题约束。已读正文全部章节、Algorithm 1、实验图表和本地 reproducibility checklist；未运行代码。

## 1. 来源与先记住的判断

- 论文：**AOSpec: Action and Observation Co-Speculation for Low-Latency Agent Serving**，Hao (Mark) Chen 等；[arXiv:2608.00881v1](https://arxiv.org/abs/2608.00881v1)，2026-08-01 提交，本次核查页面仅列 v1。
- [本地源码目录](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AOSpec- Action and Observation Co-Speculation for Low-Latency Agent Serving >)的目录名末尾有空格。以 `main.tex` 实际引用的六个章节和 `figs/diagram_v4.pdf` 为准，不把注释草稿、未引用旧图当最终结论。
- 使用 AAAI 2027 模板不等于已被该会议录用。Checklist 未被 `main.tex` 引入，只作为单独的作者自述材料。
- 本文的证据标记：**【作者陈述】**为原文方法/结果；**【本文解释】**为帮助理解的例子；**【推导审查】**为补充公式、条件或反例；**【待确认】**为原文未交代的复现细节。

**核心价值**：将“能否复用一段预测正确的动作链”，改成“能否复用从正确起始状态执行的目标动作”；同时用 observation 预测提前启动后续 Actor，并按概率与工具耗时的乘积排序。

**核心边界**：JASV 的实现覆盖 CoW 文件系统状态，明确排除非文件系统输入和效果；实验主要是固定 Actor 轨迹回放。不能把论文结果解读为通用工具环境的严格等价证明，或已经测得的真实在线并发收益。

## 2. 问题与具体例子

### 2.1 为什么单步动作投机不够

【作者陈述；Motivation】Actor 每步生成动作耗时为 $D_i$，工具执行耗时为 $T_i$。串行延迟写为：

$$
L_{\mathrm{serial}}=\sum_{i=1}^{n}(D_i+T_i).
$$

这里按论文工具交互步计时；如果末尾还有独立最终答复生成，应再加其生成时间。不能计算投机方法时计入最终答复、串行时却漏掉。

当下一动作在该步 Actor 开始时就被完美预测，单步最多隐藏：

$$
G_i^{\mathrm{one\text{-}step}}\leq\min(D_i,T_i).
$$

若 Actor 只生成 2 秒，而安装工具运行 60 秒，单步提前执行最多隐藏 2 秒。Actor 越快，这个窗口越短。

作者分析 1,921 条训练轨迹：耗时至少 1 秒的调用只占 17%，却占 97% 的工具时间。这支持“命中数量不是收益”的问题定义；**97% 是工具时间占比，不代表只要命中这些调用就能实现 97% 端到端加速**，仍受重叠窗口、验证和资源限制。

### 2.2 一个能展示 JASV 的例子

【本文解释；非论文实验数据】Agent 在文件系统中依次：

1. `cat requirements.txt`；
2. `grep ... pyproject.toml`；
3. 从本地 wheel 仓库执行一次耗时安装；
4. 根据实际结果生成最终答复。

假设前两步对**纳入版本管理的状态**没有写入，第三步的全部依赖已在快照内，安装无网络、时钟等未建模输入。Drafter 在最初边界猜到第三步的完整调用，就在初始状态的隔离分支中提前安装。它不必猜对前两次读取的具体顺序。

Actor 真正生成安装调用时，系统检查：① 工具和所有参数相同；② 分支执行前的文件系统版本与现在的正式版本相同。两者通过，就复用安装结果及其隔离写入；尚未完成则等待剩余部分。

若中间先修改了 `requirements.txt`，即使安装命令字符串完全相同，也不能复用旧分支。若只改了一个与安装无关的文件，**全文件系统版本检查也可能拒绝**：这是保守验证，不是理解了目标动作的精确依赖。

另一个方向是：某个正在运行的工具常返回完全相同的固定消息，预测器可先猜消息，让 Actor 提前思考下一步；真实消息回来后逐字节验证，匹配才保留这段生成。这就是 observation speculation。

## 3. 两种投机与完整时序

### 3.1 名词必须分开

| 名词 | 对象与含义 | 验证什么 |
|---|---|---|
| Action speculation | 预测未来工具调用，在隔离环境中实际执行 | 完整调用 + 执行前的环境版本 |
| Observation speculation | 预测当前工具的返回内容，先启动下一步 Actor | 预测内容与真实返回逐字节相同 |
| Origin state $S_f$ | 某个投机分支**执行目标动作之前**的状态 | 与目标动作正式发生前的状态比较 |
| Target action $a_i$ | Actor 后来正式生成的动作，可能不在下一步 | 不是 Drafter 单方面宣布的动作 |
| Commit | 将已验证执行路径的环境结果变成正式状态 | 不是把所有预测动作批量批准 |
| Provisional Actor | 在猜测 observation 上运行的暂定续写 | 生成完成不等于可以对外提交 |

Action speculation 中的 observation 是**真的执行出来的**，不是预测器编造的；但真实执行也可能发生在错误状态上，因此还需要 JASV。

### 3.2 按 Algorithm 1 还原流程

【作者陈述；Methodology / Algorithm 1】

1. 从已验证历史 $H$ 和已提交状态 $S$ 开始，启动正式 Actor，同时生成并启动隔离动作分支。
2. Actor 生成正式动作 $a$。从分支池查找满足 JASV 的分支；没有则从正式状态执行 $a$。
3. 如果该执行已经完成，直接取得真实 observation 和执行后状态；无需再启动该窗口的 observation 预测。
4. 如果执行尚未完成，observation 预测与它竞速。每个先完成的候选 $\hat o$ 都可启动 $\operatorname{Actor}(H\circ(a,\hat o))$，受分支预算约束。
5. 真实执行返回 $(o,S^+)$ 后，只保留 $\hat o=o$ 的 Actor 分支；没有匹配则用真实 $o$ 重新启动下一步 Actor。
6. 取消未完成的预测，丢弃其余暂定分支；更新正式历史并提交 $S^+$。
7. 清理与新状态不相容的动作分支，从新边界启动新候选，继续循环。

**三个实现细节**：候选生成循环必须与工具完成事件并发，不能等待所有候选生成完才接收工具结果；保留下来的 Actor 可能仍在运行；它若早已生成后续动作，也不能在 observation 验证之前直接产生正式副作用。

原文一处写提交后丢弃 outstanding forks，但 Algorithm 1 用的是 `PruneInvalid`，后文又允许分支跨不改变状态的调用存活。与完整算法一致的解释是：**丢弃失效分支，仍验证每次复用，不机械地在每个只读步骤后清空所有分支**。这一点复现时应核对代码。

## 4. Observation 模型、EVD 与完整公式

### 4.1 可选监督训练

【作者公式；Methodology / Optional Observation-Model Fine-Tuning】训练数据为真实轨迹三元组 $(H_t,a_t,o_t)$，模型参数为 $\theta$：

$$
\mathcal L_{\mathrm{obs}}(\theta)
=-\sum_{(H_t,a_t,o_t)\in\mathcal D}
\log p_\theta(o_t\mid H_t,a_t).
$$

原文右侧写 $o$，这里按该训练样本明确为 $o_t$，不改变含义。$H_t$ 是此前历史，$a_t$ 是当前动作，$o_t$ 是对应真实返回。

【数学展开；用于实现】若 observation token 序列为 $(o_{t,1},\ldots,o_{t,m_t})$，则：

$$
p_\theta(o_t\mid H_t,a_t)
=\prod_{\ell=1}^{m_t}
p_\theta(o_{t,\ell}\mid H_t,a_t,o_{t,<\ell}),
$$

$$
\mathcal L_{\mathrm{obs}}(\theta)
=-\sum_{t\in\mathcal D}\sum_{\ell=1}^{m_t}
\log p_\theta(o_{t,\ell}\mid H_t,a_t,o_{t,<\ell}).
$$

实现时明确结束符是否包含在序列中，并只对目标输出计算监督损失。作者还对昂贵窗口过采样，但未给完整采样比例和训练超参数；若改写成加权损失，权重只能作为自己的实现选择，不能编成论文公式。

主实验用按 harness 微调的 Qwen3-0.6B；动作 Drafter 则是未经额外训练的 Qwen3.6-35B-A3B-FP8。Observation 模型最终使用当前动作和此前交互历史；输入消融表明任务描述、reasoning 文本并非都必需，具体序列化模板待确认。

### 4.2 估计某种 observation 对应多长工具时间

【作者公式；Expected Value Decoding】历史执行记录为 $\{(o_j,T_j)\}$，候选为 $c\in\mathcal C(H_t,a_t)$：

$$
\widehat T(c)
=\frac{\sum_jK(c,o_j)T_j}{\sum_jK(c,o_j)}.
$$

$K(c,o_j)$ 表示输出相似度；分母是相似权重总和，分子是按相似性加权的工具耗时。它估计“出现这一类输出的执行通常多慢”，不是推理模型自身生成 $c$ 要多久。

两种实现：Pattern-EVD 按挖掘出的正则模式归组；Embedding-EVD 按表示空间的邻近性归组。主实验采用前者。若采用同模式为 1、否则为 0 的 $K$，上式退化为组内平均时长，这是公式的一种直接实现，原文未给所有模式挖掘细节。

【待确认】无匹配导致分母为零时怎样回退；历史记录是否按工具、任务和上下文分桶；模式由什么算法挖掘；相似度与邻居阈值是什么。不能默默以测试集未来时长补齐这些统计。

### 4.3 Expected Value Decoding 排序

【作者公式】

$$
V_o(c)=p_\theta(c\mid H_t,a_t)\widehat T(c).
$$

按 $V_o$ 从高到低选择预算内的候选。$p_\theta$ 是 observation 的模型概率，$\widehat T$ 是估计工具时长。只选高概率会偏向常见快输出；只选长时间会偏向罕见且不可信的输出；乘积兼顾二者。

【实现展开】对完整候选可累计 token log-probability，用：

$$
\log V_o(c)
=\sum_{\ell=1}^{|c|}\log p_\theta(c_\ell\mid H_t,a_t,c_{<\ell})
+\log\widehat T(c)
$$

在概率、时长均为正时排序以避免下溢。**长度归一化会改变评分定义**，不能未经说明把平均 token 概率替换成原文的整序列概率。论文未完整说明候选 proposal、打分与实际 decoding 的先后实现，相关代价必须计入时间线。

### 4.4 EVD 是收益代理，不是完整的净收益公式

【推导审查；非作者公式】以正式动作发出为时间零，尚需执行 $W$ 秒；候选 $c$ 在 $d_c$ 秒后完成，随后 Actor 生成要 $D_{t+1}$ 秒。若 observation 匹配且无争用、验证开销，提前生成的收益为：

$$
G_o(c)=\mathbf1[c=o_t]\min\{D_{t+1},(W-d_c)_+\},
\qquad(x)_+=\max(x,0).
$$

对应单候选期望为：

$$
\mathbb E[G_o(c)\mid H_t,a_t]
=P(o_t=c\mid H_t,a_t)
\mathbb E\!\left[\min\{D_{t+1},(W-d_c)_+\}\mid o_t=c,H_t,a_t\right].
$$

若当前动作已经提前执行了一大半，$W$ 是**残余工具时间**，不是完整 $T_t$。若下一 Actor 只需 2 秒，预测一个 100 秒工具的返回也不会单靠这一次续写隐藏 100 秒。下一 Actor 未验证就发出的额外工具路径不在这个简化公式中。

因此 $p\widehat T$ 可能在不同剩余时间、候选生成延迟和 Actor 长度下排序失准；论文也称其为 proxy。额外分支计算、排队和资源争用还未包含在上式中。这里识别的是需要实测的决策差距，**不是宣布 EVD 无效，更不是已确认新颖性**。

### 4.5 Observation 验证

【作者规则的公式化】

$$
\mathrm{ValidObs}(c,o_t)=\mathbf1[\operatorname{bytes}(c)=\operatorname{bytes}(o_t)].
$$

只有通过它的暂定 Actor 可以续用。语义相似、JSON 大致相同、返回“成功”但日志不同，都不自动满足条件。

作者提到可规范化 process UUID 等字段，这是潜在改进，不是默认的近似验证协议。若规范化改变 Actor 所见文本，要想继续比较严格串行语义，串行基线也必须使用相同规范化规则；“业务上无关”不能直接推出“模型生成分布不变”。

## 5. 动作预测、状态验证与长距离复用

### 5.1 JASV 公式

【作者公式；Joint Action–State Verification】分支 $f$ 保存投机动作 $\hat a_f$ 及其 origin state 的版本 $\nu(S_f)$，正式动作 $a_i$ 将从状态 $S_i$ 执行：

$$
\mathrm{ValidAct}(f,a_i,S_i)
=[\hat a_f=a_i]\land[\nu(S_f)=\nu(S_i)].
$$

方括号是布尔条件。动作相等包括工具及全部参数；比较的是**执行前的状态**。投机安装改变了文件系统，不能拿安装后的状态与正式执行前的状态比较，否则会拒绝本来正确的写入。

运行时将 CoW 文件系统图像的 root hash 作为版本标识。比较已有版本标识不需要每次递归扫描文件；但这不说明生成、维护 root hash、快照与提交本身免费。

### 5.2 为什么不要求中间动作链一致

【作者公式；State-Verified Action Lookahead】在边界 $j$ 提前启动第 $i$ 步目标动作，串行参照中的可提前执行时间为：

$$
R_{j,i}=\sum_{k=j}^{i}D_k+\sum_{k=j}^{i-1}T_k.
$$

因此该动作理想上最多隐藏：

$$
G_{j,i}^{\max}=\min(T_i,R_{j,i}).
$$

当 $j=i$ 时退化为 $\min(T_i,D_i)$。当 $j<i$，中间模型生成和工具执行都提供时间，但候选必须最终同时满足调用与状态验证。

【推导审查】公式没有显式扣除候选生成和 fork 启动延迟，且联合投机已经压缩的前序时间不能再完整算一次。真实实现应记录目标需求时刻 $t_i^{\mathrm{need}}$、分支执行开始时刻 $t_f^{\mathrm{start}}$：

$$
r_{f,i}=\max(0,t_i^{\mathrm{need}}-t_f^{\mathrm{start}}),
\qquad
G_a(f,i)\leq\mathbf1[\mathrm{ValidAct}(f,a_i,S_i)]\min(T_i,r_{f,i}).
$$

这里假设 fork 与串行工具执行速度相同且没有提交开销；否则应从实际完成事件重建关键路径。不能把所有 action/observation 的独立收益直接相加。

【概率拆解；非作者定理】有效命中的概率为：

$$
P(\mathrm{ValidAct})
=P(\hat a_f=a_i)\,
P\!\left(\nu(S_f)=\nu(S_i)\mid\hat a_f=a_i\right).
$$

完整链匹配则一般写为条件概率之积：

$$
P\!\left(\bigcap_{k=j}^{i}\{\hat a_k=a_k\}\right)
=\prod_{k=j}^{i}P\!\left(\hat a_k=a_k\mid\bigcap_{r=j}^{k-1}\{\hat a_r=a_r\}\right).
$$

无需假设各步独立，也不能直接用固定 $p^{i-j+1}$ 代替实际数据。JASV 消除了“必须预测整链”的要求，**没有消除远距离目标更难预测、状态更可能改变的限制**。

## 6. “Lossless”成立需要什么

### 6.1 受限环境内的充分条件

【推导审查；帮助理解的证明骨架，不是论文给出的定理】令工具是确定性状态转移：

$$
F(S,a)=(o,S^+).
$$

要从 JASV 推出与串行等价，至少需要：

1. $S$ 包含工具所依赖的全部输入状态；版本相等足以表示这些状态相等，忽略 hash 碰撞。
2. 工具动作完整相同，相同状态和动作产生相同 observation 与状态转移。
3. 投机写入只在分支内可见，没有泄漏到网络、共享服务或其他正式路径。
4. 验证到提交之间没有外部并发修改；commit 原子地采用已验证分支的结果。
5. Actor 输入的消息、工具输出和序列化形式一致；未验证分支不影响正式 Actor 状态或外部副作用。
6. 若要求逐次运行同一条轨迹，还需控制 Actor 随机性；仅同一输入通常只能保证同一条件输出分布。

由 $\hat a_f=a_i$ 和 $S_f=S_i$ 可得：

$$
F(S_f,\hat a_f)=F(S_i,a_i)=(o_i,S_{i+1}).
$$

若 observation 分支同时满足 $\hat o_i=o_i$，其 Actor 输入为：

$$
H_i\circ(a_i,\hat o_i)=H_i\circ(a_i,o_i)=H_{i+1}.
$$

以初始历史和状态相等为起点，逐步归纳，即可保留串行的状态转移和 Actor 条件分布；固定相同随机流时才能进一步谈逐轨迹相等。

### 6.2 论文实际保证的范围

【作者陈述；Runtime Support】实现使用文件系统 CoW 版本，明确排除 non-filesystem inputs and effects。故下列情况不在上述“文件系统版本相同即可”的直接保证内：时间、随机数、进程内存、网络响应、外部数据库、不可撤销服务调用等，除非另有隔离或固定输入协议。

一个调用即便“不写文件”，如果读取时间或网络，也未必能凭 root hash 复用。同样，“安装”如果需要实时联网，不能直接套用本笔记的离线安装例子。

论文提供系统契约和验证规则；Checklist 也明确说不做理论贡献。正文没有覆盖一般随机环境的正式等价定理，也没有通过真实在线逐状态对照实验证明所有边界。**条件清晰的系统贡献仍有价值，但不能扩大 lossless 的量词。**

错误分支的处理主要是丢弃隔离结果，而不是已经污染正式世界后再撤销；不能把 `discard` 等同于任意外部副作用都可 rollback。

## 7. 理论上如何实现

### 7.1 必要组件

| 组件 | 最少需要维护什么 | 容易漏掉什么 |
|---|---|---|
| 正式状态管理器 | 已验证历史、当前版本、串行提交锁 | Actor 文本历史与环境状态是两个对象 |
| 动作预测器 | 完整候选工具调用、候选生成完成时间 | 远期目标 proposal 与预算策略未完整公开 |
| Fork 管理器 | origin version、隔离执行句柄、真实输出、执行后状态 | 写入必须留在分支，不能先写正式目录 |
| Observation 预测器 | 候选、概率、EVD 统计、候选完成时间 | 候选提出和概率计算并非零开销 |
| Actor 分支管理器 | 按完整 observation 索引的暂定续写 | 相同候选去重，未验证动作不能正式提交 |
| 事件与计时模块 | 生成、执行、取消、验证、提交事件 | cancel 发出与资源真正释放可能不同步 |

### 7.2 可转成程序的流程

以下是 Algorithm 1 的事件式改写；保留原文逻辑，补上并发与清理表达。不同分支的实际选择策略、异常协议仍需实现者明确。

```text
H, S = initial_verified_history, initial_committed_snapshot
B = actor.start(H)
F = launch_action_candidates(H, S)       # 预测完成后才能实际 fork.execute

loop:
    x = await B.output()
    if x is final_answer:
        cancel_and_discard_all(F)
        return x

    a = parse_complete_tool_call(x)
    f = find F where f.action == a and f.origin_version == S.version
    if f does not exist:
        f = execute_from_committed_snapshot(S, a)

    P = {}                              # 完整 observation -> 暂定 Actor
    if not f.finished:
        stream = start_EVD_candidates(H, a, branch_budget)
        while not f.finished:
            event = await first(tool_finished(f), candidate_ready(stream))
            if event is tool_finished:
                break
            c = event.candidate
            if c not in P and budget_allows:
                P[c] = actor.start(H + [(a, c)])

    o, S_plus = await f.result()
    B_next = P[o] if exact_bytes_key(o) exists else actor.start(H + [(a, o)])
    cancel_unfinished_candidates_and_unselected_actors()

    atomically:
        ensure_authoritative_state_has_not_changed()
        S = commit(S_plus)
        H = H + [(a, o)]

    F = prune_incompatible_origins(F minus f, S.version)
    F += launch_action_candidates(H, S)
    B = B_next
```

超时、工具异常、Actor 提前完成、取消失效分支、资源回收都需要显式状态机。原文伪代码未完整规定这些情况，不能声称上面已是作者开源实现。

### 7.3 可复现性现状

已知设置：vLLM；Drafter 用 2 张 NVIDIA H200；工具在 Azure Standard_D4s_v5（4 vCPU、16 GiB RAM）运行；默认 5 个 observation 分支、8 个动作 fork。

本地材料没有提供可直接运行的代码项目。Checklist 表示代码在发表时公开，CoW runtime 来自当时匿名评审中的并行工作，正文未给公开引用。**这是一项未来发布承诺，不是现成可复现 baseline**；本次未做额外全网仓库排查，不把“材料未提供”扩大成永久未开源。

关键缺失项包括：训练/测试任务具体列表与样本量、SFT 超参数及昂贵窗口过采样比例、动作远期候选提示词与候选分配、EVD proposal/正则挖掘、零相似回退、采样参数、runtime 构建与状态版本日志、回放事件规则和计时开销归属。

Checklist 对 seeds、统计检验、完整超参数等多项回答 yes，但当前主文未展示足够细节；不能用自报勾选替代实际可获得的方法描述。

若后续要最小复现，先以固定轨迹和确定性本地工具实现 JASV 两条件及观测竞速，再接模型；无需一开始部署 2×H200 系统。本轮仍只阅读，不改动现有四个代码项目。

## 8. 实验设置、结果与证据强度

### 8.1 数据与延迟口径

【作者陈述；Experiments】Terminal-Bench 2.0 按任务划分训练/测试，并让每个任务类别出现在测试集。测试包含九个 harness–Actor 组合：

| Harness | Actor |
|---|---|
| mini-swe-agent（MSA） | Sonnet 4.5、Gemini 2.5 Pro、GPT-5 mini |
| OpenHands | Sonnet 4.5、Gemini 2.5 Pro、GPT-5 mini |
| Gemini CLI | Gemini 2.5 Pro |
| Judy | Opus 4.6、Gemini 3.1 Pro |

迁移实验使用 SWE-bench Verified 官方 Bash Only leaderboard 的 MSA 轨迹，observation 模型只在 Terminal-Bench 训练。

**评测方式是 trace replay**。各方法固定相同 Actor 轨迹，根据记录的输出 token 数计算 Actor 时间，并扫 TPOT 为 $\{20,15,10,5,1\}$ ms/token。其主要意义是控制轨迹、比较调度窗口，不是实时让 Agent 自由运行重新完成任务。

【口径展开】若一步有 $N_i$ 个输出 token、TPOT 为 $\tau$，其回放生成时间近似为：

$$
D_i=N_i\tau.
$$

正文没有给独立 TTFT、prefill、请求排队项的完整公式；不能假设已经覆盖真实 API 延迟全部组成，也不能把 TPOT 扫描当成作者实际部署了五种 Actor 服务。

【指标定义展开】端到端节省率与加速倍数不同：

$$
\mathrm{Saving}=1-\frac{L_{\mathrm{method}}}{L_{\mathrm{serial}}},
\qquad
\mathrm{Speedup}=\frac{L_{\mathrm{serial}}}{L_{\mathrm{method}}}
=\frac1{1-\mathrm{Saving}}.
$$

例如单个统计量若节省 32.5%，对应约 1.48×；**先跨配置平均节省率再换算，不等于跨配置平均加速倍数**。作者主结果是对配置等权平均。

工具时间隐藏比例与端到端节省率的分母不同；prediction exact hit、目标动作匹配、状态有效和最终 commit 也要分别记录。论文未提供所有这些分层指标的完整口径。

### 8.2 Baseline 公平性

| Baseline | 本文实验中的定义 | 解读边界 |
|---|---|---|
| Serial | 原始串行轨迹 | 比较参照 |
| Spec-Actions | 用 Qwen3.6-35B-A3B-FP8 预测并启动下一动作 | 这是本文配置，不能代表 SA 所有 breadth/depth 变体 |
| SpecHop-cache | 复用历史 observation | 简单 observation 参照 |
| SpecHop-LLM | 同一 35B-A3B 模型预测 observation | 与 AOSpec 的微调 0.6B 架构/训练条件不同 |
| Oracle-action | 直接给出正确的下一动作 | 只有单步 action oracle，不是任意联合投机的全局上界 |

没有直接比较 PASTE、SPORK、Speculate with Memory。因而不能据此断言优于全部动作投机或 memory 方法。动作/观测预算、训练与 GPU 成本也需要放在一起核算。

### 8.3 主结果

- TPOT 从 20 ms 降到 1 ms，AOSpec 平均节省从 **11.8% 到 32.5%**；最强 practical baseline 对应 **4.3% 和 12.2%**。
- 在 45 个配置×TPOT 设置中，AOSpec 都优于论文所选 practical baselines；41/45 超过单步 action oracle。后者来自能力集合更大，不是突破数学上界。
- 在 SWE-bench、TPOT=1 ms，节省 **18.9%**，最强 baseline **4.2%**。这是该回放设定下的跨 benchmark 迁移证据。

10 ms TPOT 的 trial 延迟分位数（秒）：

| 方法 | p10 | p30 | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Serial | 56 | 129 | 229 | 1038 | 1842 | 7780 |
| Spec-Actions | 56 | 129 | 226 | 1036 | 1812 | 7765 |
| SpecHop-cache | 55 | 129 | 223 | 1003 | 1624 | 5052 |
| SpecHop-LLM | 55 | 128 | 223 | 1008 | 1779 | 7449 |
| Oracle-action | 53 | 124 | 220 | 971 | 1669 | 7750 |
| AOSpec | 49 | 114 | 204 | 929 | 1285 | 4449 |

作者报告 p50/p90/p95/p99 减少 10.7%/10.5%/30.3%/42.8%。这些是**分位数本身的变化**，不是“最慢 1% 每个任务都加速 42.8%”。表中秒数四舍五入，不能要求由显示整数反算完全一致。

### 8.4 消融支持什么

| 消融 | 主要数字 | 能支持的结论与限制 |
|---|---|---|
| EVD，两个配置、width=1、10 ms | 时间排序/概率排序/乘积：隐藏 39.4/47.6/60.5 秒；exact accuracy 13.1%/28.1%/27.8% | 命中略低仍可能隐藏更多时间；只在所选两个配置测此消融 |
| 慢调用命中 | >60 秒调用命中率 18.4%→21.0% | EVD 把预测预算向慢调用移动 |
| 相似度 | Pattern vs Embedding：MSA 40.9 vs 22.3 秒；Judy 80.1 vs 61.2 秒 | 当前模式统计优于当前 embedding 实现，不能推出普遍规律 |
| 双组件，九配置平均、10 ms | Obs-only 12.0%，Action-only 7.1%，联合 17.0% | 两者互补但收益不直接相加；不能假设 19.1% |
| Observation 微调 | width=1：2.4%→29.1%；width=5：28.1%→32.9% | 微调在低预算下显著；多候选缩小准确率差距，但不证明等成本等延迟 |
| Action 候选宽度 | 1→32，平均 trace match 11.0%→23.7% | 更多候选提高覆盖，未独立说明每个新增 fork 的净收益 |
| Action 距离，width=32 | 距离 1→10，匹配 23.7%→7.7% | 不必猜整链仍能猜远期目标；这不是通过状态验证后的 commit 率 |

输入消融（MSA×Sonnet 4.5）：

| 去掉什么 | 工具时间隐藏比例 | Exact hit |
|---|---:|---:|
| 无，完整上下文 | 27.1% | 33.3% |
| Task query | 28.7% | 33.6% |
| Reasoning | 27.3% | 33.6% |
| Prior observations | 26.5% | 30.1% |
| 全部历史步骤 | 27.2% | 29.9% |

去掉历史后准确率下降、隐藏时间近似不变，进一步说明不能只用命中率评估 memory 价值。但这只是单配置、同时改变训练和推理输入的消融，不能直接推出所有 observation 预测都不需要历史。

### 8.5 实验闭环哪里还不完整

有证据的链条是：慢调用占主导 → 概率×耗时排序能改善回放隐藏时间；双组件 → 所选配置下联合收益更大；远期单目标 → 比整链预测保留更多命中机会。

还缺的直接证据包括：真实在线逐状态一致性；动作命中到 origin-state 有效再到 commit 的漏斗；完整 fork/commit/hash/取消开销；并发 CPU/I/O/GPU 争用与总费用；真实运行的任务质量；对资源匹配的近期 baseline 的比较。原文 hardware 信息和回放结果不能替代这些测量。

## 9. 与前六篇建立关系

| 维度 | Speculative Actions | PASTE | Speculate with Memory | AOSpec |
|---|---|---|---|---|
| 主要入口 | 通用 API 的预测、预执行、匹配复用 | 历史事件模式、参数绑定、调度 | 为 Speculator 检索经验、失败与混淆记录 | 动作与返回预测的联合循环 |
| Observation 路径 | 框架可预测中间输出并驱动后续调用 | 不以小 observation 模型为核心 | Type 2/3 已有 observation/后续动作预测 | 小模型预测，真实字节校验后保留 Actor |
| 长距离动作 | 已讨论 depth；具体可用性依赖协议 | 模式和动作依赖 | Type 3 链式预测 | 不猜中间整链，只验证远期目标及 origin state |
| 状态/副作用 | 不能仅从 API key 推出任意环境等价 | 已有隔离和安全变体 | 严格链提交条件须另外检查 | CoW 文件系统 + JASV 明确前状态条件 |
| 价值选择 | 已有置信度及成本分析 | 预期隐藏时间、资源压力和联合调度 | memory 改善预测，路由属未来工作 | observation 概率×历史工具时长 |
| 证据 | 多场景，需按各自口径看 | serving 与调度测量 | 主要预测回放，部分计时分析 | 多配置固定轨迹回放 |

DSP 的核心是动态规划长度，不是 JASV；SPORK 是从部分 Actor 前缀做自投机并复用失败草稿 token；SWYR 训练同模型匹配自身下一动作。AOSpec 不提供上述训练或 token 接受机制，因此不能把“联合”理解为同时解决了全部层次。

**已经覆盖、不能直接当新 idea 的内容**：action+observation 联合投机；为慢调用分配更多预测预算；使用历史输出模式估计价值；对目标动作检查 origin state；在隔离快照内执行有文件写入的动作；不猜整条轨迹的远期目标预执行。

## 10. 留给后续的可检验问题

以下是诊断候选，不是已经确定的新研究方向。先读 SMC，并核查相关事务执行、预取和收益调度工作，再判断新颖性。

### A1：全状态验证是否拒绝了大量本来可复用的目标？

- **依据**：JASV 用整个 managed filesystem 的版本相等，远期命中率未拆出状态失效率。
- **最小实验**：确定性文件工具中构造中间只读、无关写、相关写三组轨迹；固定相同动作预测，记录调用命中→状态通过→实际节省。先做诊断，不先设计新的验证器。
- **理论参照**：若已知动作完整读取依赖集合为 $R(a)$，可讨论 $S_f|_{R(a)}=S_i|_{R(a)}$ 的充分性；但还需目录枚举、文件不存在性、元数据及提交写集与冲突处理，不能仅比较几个已打开文件就宣称安全。
- **排除条件**：目标场景几乎没有无关写，或严格版本检查已足够；已有事务/依赖验证方法直接解决问题。此时放弃把“更细状态验证”当主贡献。

### A2：EVD 是否因剩余窗口和生成成本而排错候选？

- **依据**：$p\widehat T$ 与实际 $p\min(D_{t+1},(W-d)_+)$ 不同；联合 action speculation 会改变 $W$。
- **最小实验**：固定候选及预测概率，从时间日志重放不同 $W,D,d$，比较概率排序、EVD、简单截断窗口排序及可实现的净收益近似；统一候选数和资源预算。
- **观测指标**：实际关键路径节省、错误排序频率、额外 Actor tokens、等待与取消成本；不能只比较预测命中。
- **排除条件**：简单阈值/窗口截断已经消除差距，或绝大多数窗口处于同一饱和区；PASTE 等已有方法在相同约束下已解决。

### A3：长距离目标命中能转化成多少有效提交？

- **依据**：width=32、distance=10 的 7.7% 是目标动作 trace match，不能直接代表 state-valid reuse。
- **最小实验**：对距离分别测 $P(\text{action match})$、$P(\text{state valid}\mid\text{match})$、剩余重叠和 fork 成本；以按时间加权的真实 commit 收益作结论。
- **排除条件**：与单步或简单时间阈值相比收益很小；瓶颈只是具体 benchmark 状态变化分布，尚无可迁移问题。A1/A3 可合并，不为凑创新点重复包装。

### A4：回放排序在真实并发下是否保持？

- **依据**：默认 8 个动作 fork、5 个 observation 分支，工具 VM 只有 4 vCPU；多分支资源压力可能改变工具完成时间，但不能据此直接断定已经退化。
- **最小实验**：少量确定性长工具，扫描并发数，对比回放预测与真实 wall clock；先用调优后的简单限流作强基线。
- **排除条件**：在目标工作负载上偏差很小，或简单限流足够。这是评测与机制问题，单纯发现模拟偏差不自动构成论文。

## 11. 本篇留下的结论与下一篇问题

AOSpec 的贡献主要在系统方法层：把有限状态隔离、目标动作验证和 observation 提前续写组织成一个循环，并展示价值排序的回放收益。它对机理的解释来自关键路径与状态等价条件，没有提出新的普遍数学定律；哲学层未触及，不强行拔高。

对后续选题最重要的修正是：**“加状态校验”“联合 action/observation”“按工具延迟加权”“跳过中间链直接猜远期动作”都已有直接工作覆盖**。剩余值得诊断的是条件之外的具体差距，而不是重复这些概念。

第八篇 SMC 要带着本篇标准检查：仅首动作匹配是否足以推出后续每步输入与状态等价；macro 是否限制环境和 observation；是否有逐步提交、隔离与失效协议；任务成功率下降是否说明采用了不同于 AOSpec 的正确性目标。

## 12. 本地证据入口

- [方法、全部五个编号公式与 Algorithm 1](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AOSpec- Action and Observation Co-Speculation for Low-Latency Agent Serving /sections/methodology.tex>)：训练损失、时长估计、EVD、JASV、runway；其余展开公式均已在笔记中区分。
- [动机与串行时间分析](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AOSpec- Action and Observation Co-Speculation for Low-Latency Agent Serving /sections/motivation.tex>)。
- [实验设置、数值与消融](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AOSpec- Action and Observation Co-Speculation for Low-Latency Agent Serving /sections/experiments.tex>)。
- [系统时序图](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AOSpec- Action and Observation Co-Speculation for Low-Latency Agent Serving /figs/diagram_v4.pdf>)。
- [独立可复现性自述](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/AOSpec- Action and Observation Co-Speculation for Low-Latency Agent Serving /ReproducibilityChecklist.tex>)。
