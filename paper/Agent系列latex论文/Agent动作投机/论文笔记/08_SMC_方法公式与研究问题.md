# 08 · SMC：方法公式、实现原理与研究问题

阅读日期：2026-09-09。论文：**Speculative Macro Commit for Faster Tool-Using Agents**。本篇以算法、完整公式和理论实现为主，服务后续创新点筛选；未运行代码，未复现实验。

## 1. 来源与最重要的修正

- Zeyu Liu、Souvik Kundu、Peter A. Beerel；[arXiv:2609.03236v1](https://arxiv.org/abs/2609.03236v1)，2026-09-03 提交，本次页面仅列 v1，Comments 标明 Accepted in MLSP2026。
- [本地论文目录](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculative Macro Commit for Faster Tool-Using Agents>)。`00README.json` 指定 `mlsp_camera_ready.tex` 为主文件；已读其实际引用的 abstract、introduction、background、method、results、ablations、conclusion，全部公式、Algorithm 1 与表格。没有另附方法附录或图像文件，不采用注释掉的草稿。
- [作者给出的代码链接](https://github.com/zeyuliu1037/speculative-macro-commit)本次访问仍返回 404。只能说论文声称公开、当前无法访问，不能当成已可运行 baseline。

**先纠正一个关键疑问：作者没有声称 SMC 严格 lossless。** Introduction、Related Work 和 Conclusion 都明确把它定义为 approximate：首动作经 Actor 确认，后续动作不逐步经 Actor 确认，而是依据历史模式、Drafter 已执行结果和在线检查直接进入正式轨迹。

因此我们要问的不是“它的 lossless 证明为什么有漏洞”，而是：**它用哪些条件控制近似误差；这些条件与任务质量、状态安全、延迟收益之间的关系是否被充分测量？**

标记约定：**【作者陈述】**为原文方法与数据；**【本文解释】**为例子或概念整理；**【推导审查】**为补充数学、实现约束和证据检查；**【待确认】**为原文未公开的细节。原文只有一个独立编号数学公式，即 Beta 后验下分位数；下文将它完整展开，其余公式均注明为补充形式化，不能反称为作者已提出的定理。

## 2. 本质上省掉了什么

### 2.1 SA 与 SMC 的区别

【作者陈述；Method / Executor Roles】执行器 executor 管理模型—工具循环，决定哪些动作和返回内容进入正式历史。SMC 中：

- **Actor**：大模型，生成正常情况下的正式动作。
- **Drafter**：较快小模型，从已提交历史向前预测，并在隔离状态中实际执行短动作链。
- **SA-only**：Actor 每次确认第一个草稿动作，复用其执行；后续仍逐次调用 Actor。
- **SMC**：如果草稿与已保留 macro 对齐，Actor 确认首动作后，可以直接提交后续若干草稿动作及返回，跳过对应 Actor 调用。

SMC 省掉的不只是重复工具等待，更直接地省掉了**后续大模型决策**。代价是正式轨迹的一部分由 Drafter 和执行器选择；“authoritative actor”这个角色名称不表示每个正式动作都经它批准。

与 token-level speculative decoding 不同，SMC 没有对跳过部分计算完整目标模型验证或分布校正。不要把“也有 draft 和 verify”理解为相同的保分布保证。

### 2.2 示例：同一个 macro 能跳过不同数量的步骤

【本文解释；自造 API，不是 benchmark 原例】假设训练中反复出现：

```text
u1: get_customer(customer_id=$C)
u2: list_orders(customer_id=$C)
u3: get_order(order_id=$O)
u4: get_delivery(order_id=$O)
```

若当前历史末尾已经做过 `u1`，Drafter 在隔离环境完成了 `u2,u3,u4`，而 Actor 下一调用与实际 `u2` 完整一致，那么 `u2` 是 anchor。SMC 可在检查后连带提交 `u3,u4`，跳过两个 Actor 决策。

若历史已经包含 `u1,u2`，草稿只覆盖 `u3,u4`，同一个四步 macro 只能跳过一个 Actor 决策。因此最小收益门槛应看**anchor 后还剩多少步**，不能只看 macro 总长度。

`$C`、`$O` 表示跨调用的槽位绑定关系，而不只是任意字符串通配。比如第二步返回多个订单时，Actor 可能选另一订单；首调用相同不能证明后续选择相同。原文称保存 argument structure，但没有完整给出跨 observation 取值和绑定的规则。

## 3. 状态、历史与草稿链

【作者定义的形式化】正式历史和草稿链分别为：

$$
H_t=((a_1,o_1),\ldots,(a_t,o_t)),
\qquad Q=(q_1,\ldots,q_d),\quad d\leq D,
$$

$$
q_i=(\widehat a_i,\widehat o_i).
$$

$D$ 是目标草稿深度，$d$ 是某时刻已经可用的长度；原文用固定 $D$ 写链，实际异步实现必须处理链未填满。$E_t$ 是执行 $H_t$ 后的 live task state；隔离草稿状态记为 $E^i$。

这里 $\widehat o_i$ 的“帽子”表示属于投机路径，**它是执行 $\widehat a_i$ 得到的真实返回，不是语言模型猜的 observation 文本**。

【实现展开】设工具状态转移为 $F$，则草稿链按序生成：

$$
E^i_0=\operatorname{Fork}(E_t),\qquad H^i_0=H_t,
$$

$$
\widehat a_k=A_{\mathrm{draft}}(H^i_{k-1}),
\qquad(\widehat o_k,E^i_k)=F(E^i_{k-1},\widehat a_k),
$$

$$
H^i_k=H^i_{k-1}\circ(\widehat a_k,\widehat o_k).
$$

为了提交前缀而保留剩余草稿，程序至少要知道每步的前状态、后状态和完成情况；不能只有末尾快照。否则仅想提交 $q_1$ 时可能误把后续草稿写入一起带进正式环境。

Actor 在宏提交后看到的是包含这些额外动作—返回对的新历史，再决定下一步。它没有回头验证自己本来会不会选择被跳过的动作。

## 4. 离线 Macro Mining：完整概率公式

### 4.1 候选提取与标签

【作者陈述；Method / Macro Mining】

1. 从成功训练轨迹提取重复多动作模式。
2. 保留动作顺序及参数结构，把用户 ID、文档 ID、时间戳等任务相关值替换成槽位。
3. 从采样训练状态运行同部署设置的 Drafter，预测未来短链。
4. 将草稿与该状态的参考未来轨迹比较；任务特定参数归一化之后的调用序列相同，则标为正确。
5. 同时要求足够支持数和足够保守的可靠性估计，形成 macro library $\mathcal M$。

【本文形式化】设归一化函数为 $N$，一次候选机会的标签为：

$$
y_{m,s}=\mathbf1\left[
N(\widehat a_{s,1:K})=N(a^{\mathrm{ref}}_{s,1:K})
\right],
\qquad k_m=\sum_{s=1}^{n_m}y_{m,s}.
$$

$m$ 为候选 macro，$n_m$ 为被标注的机会数，$k_m$ 为正确草稿数。原文没有给出具体序列挖掘算法、长度范围、机会采样规则、参考轨迹生成方式，以及 $N$ 的完整定义。

**标签只明确描述归一化后的动作序列匹配。** 不应擅自扩展为精确参数、observation、环境状态和最终成功都一致。离线归一化正确与在线完整调用正确是两种标准。

### 4.2 原文 Beta 后验下分位数

【作者唯一编号公式】

$$
\underline p_m
=F^{-1}_{\operatorname{Beta}(k_m+1,\ n_m-k_m+1)}(\delta).
$$

保留条件：

$$
m\in\mathcal M
\quad\Longleftrightarrow\quad
n_m\geq n_{\min}\ \land\ \underline p_m\geq\tau
$$

上式是对文中筛选规则的形式化，限定在已提取候选集合中。$\delta$ 为下尾概率，$\tau$ 为可靠性门槛，$n_{\min}$ 为最小样本支持数。作者在 held-out tasks 上选择 $n_{\min}$ 和 $\tau$；未在当前正文公开全部具体数值和搜索过程。

### 4.3 这个式子怎样得到

【补充数学原理；公式隐含的共轭模型】把相同 macro 的标签暂按条件独立 Bernoulli 处理：

$$
y_{m,s}\mid p_m\sim\operatorname{Bernoulli}(p_m),
\qquad p_m\sim\operatorname{Beta}(1,1).
$$

似然与后验：

$$
P(y_{m,1:n_m}\mid p_m)
\propto p_m^{k_m}(1-p_m)^{n_m-k_m},
$$

$$
p_m\mid y_{m,1:n_m}
\sim\operatorname{Beta}(\alpha_m,\beta_m),
\quad\alpha_m=k_m+1,\quad\beta_m=n_m-k_m+1.
$$

Beta 密度、归一化常数、CDF 和分位数定义完整写为：

$$
f(p;\alpha,\beta)
=\frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)},
\qquad 0<p<1,
$$

$$
B(\alpha,\beta)=\int_0^1u^{\alpha-1}(1-u)^{\beta-1}\,du,
$$

$$
F_{\alpha,\beta}(z)
=\frac{1}{B(\alpha,\beta)}\int_0^z u^{\alpha-1}(1-u)^{\beta-1}\,du,
$$

$$
F^{-1}_{\alpha,\beta}(\delta)
=\inf\{z\in[0,1]:F_{\alpha,\beta}(z)\geq\delta\}.
$$

在该后验模型下，$P(p_m\geq\underline p_m\mid\text{labels})=1-\delta$。它是**对标签生成概率的贝叶斯可信下界**，不是“每次 runtime macro 都有严格 $1-\delta$ 正确率”，也不是任务成功保证。

【数值帮助】若 $k_m=n_m$，没有观察到失败，则后验为 $\operatorname{Beta}(n_m+1,1)$，所以：

$$
\underline p_m=\delta^{1/(n_m+1)},
\qquad n_m\geq\frac{\log\delta}{\log\tau}-1
$$

才有 $\underline p_m\geq\tau$（$0<\delta,\tau<1$）。假设 $\delta=0.05,\tau=0.95$，至少需要 58 次全成功机会，再满足 $n_{\min}$。这是演示参数，不是论文设置。

实际实现可调用统计库的 Beta inverse CDF，或对单调 CDF 二分求根；不能用后验均值 $(k_m+1)/(n_m+2)$ 替代这个保守下分位数。

### 4.4 可靠性估计的适用边界

同一任务内的重叠窗口可能相关；macro 是从成功轨迹挑出的，部署状态却可能含失败恢复；归一化可能合并会影响决策的参数；从很多候选里筛选也不自动提供整个库的同时保证。**这些因素使“训练标签可靠性”与“在线被选中提交的可靠性”不同**。原文公式没有完成分布偏移、相关样本或多重选择校正。

## 5. 在线对齐、anchor 与提交规则

### 5.1 对齐在哪里发生

【作者规则的形式化】macro 为 $m=(u_1,\ldots,u_K)$；$j$ 个前缀动作匹配正式历史后缀，余下 $\ell=K-j$ 个动作匹配草稿前缀。若 $H^-$ 表示 anchor 提交前的历史，可写：

$$
\operatorname{Align}(m,j,H^-,Q)
=\left[N(\operatorname{suffix}_j(\operatorname{actions}(H^-)))=m_{1:j}\right]
\land\left[N(\operatorname{actions}(Q_{1:\ell}))=m_{j+1:K}\right].
$$

同时需要 $1\leq\ell\leq d$，且待提交草稿均已执行完毕。槽位必须在整个对齐中一致绑定；$j$ 的具体可取范围、无历史前缀时是否接受、多个对齐的优先级未完整给出。

Actor 确认首调用的条件应按完整调用理解：

$$
\operatorname{AnchorMatch}
=[a_{\mathrm{actor}}=\widehat a_1],
\qquad a=(\mathrm{tool\ name},\mathrm{arguments}).
$$

原文写“same call”和 `FirstActionMatch`，但没有给具体 JSON 规范化、参数默认值、顺序和类型相等实现。不能把 macro 的槽位匹配直接用来替代 anchor 的完整参数确认。

### 5.2 最小跳过长度

【作者条件】

$$
L_{\mathrm{skip}}=\ell-1=K-j-1,
\qquad L_{\mathrm{skip}}\geq L_{\min}.
$$

anchor 的 Actor 调用已经发生，不算被跳过。审计表明确使用 $L_{\min}=1$，即至少再提交一步。全配置是否均用同一值、调参范围是什么，正文没有完整配置清单。

### 5.3 完整候选接受条件

【本文对文字规则的汇总，不是作者额外公式】

$$
\begin{aligned}
\operatorname{AcceptMacro}
={}&[m\in\mathcal M]\land\operatorname{Align}(m,j,H^-,Q)\\
&\land\operatorname{AnchorMatch}\land[\ell-1\geq L_{\min}]\\
&\land\operatorname{Executed}(q_{1:\ell})
\land\operatorname{OnlineChecks}(m,H^-,Q,E,x).
\end{aligned}
$$

$x$ 为任务。`OnlineChecks` 原文描述了清除过期草稿、状态及参数检查和 benchmark 特定安全策略，但没有给充分的形式定义。AppWorld 使用人工 API 名称规则拒绝不可逆或未知调用；对可 fork 的修改要求 matching live-state replay。

【待确认】这个 replay 比较的是 observation、状态差异、返回成功标记还是全部？失败前若已经部分修改 live state，如何恢复？是否原子提交整段？原文未提供通用 rollback 协议，也没有 AOSpec 那样明确的 origin-version 等式。

**参数/环境检查通过，只说明该执行路径满足指定运行约束，不能证明 Actor 本来会选择这条路径。**

### 5.4 提交与回退

【作者陈述；Algorithm 1】

- Actor 不匹配 $q_1$：从正式环境执行 Actor 动作，写入真实返回，丢弃旧草稿并重启 Drafter。
- Actor 匹配 $q_1$，但没有合格 macro 或检查失败：只提交 anchor，退化为 SA-only；Drafter 从剩余草稿继续延伸。
- macro 合格：提交 $q_1,\ldots,q_\ell$ 的动作与返回，推进正式状态，弹出这个前缀；Drafter 继续剩余链。

【状态展开】若成功提交 $r$ 步，则应满足：

$$
H^+=H^-\circ q_1\circ\cdots\circ q_r,
\qquad E^+=F_{\widehat a_{1:r}}(E^-),
\qquad Q^+=Q_{r+1:d}.
$$

$F_{\widehat a_{1:r}}$ 表示这些动作依次执行的环境转移；返回值必须与写入历史的 observation 一致。若环境合并/replay 后状态与草稿前缀后状态不一致，不能仍把余链当作有效；应丢弃或重新验证。

算法文字以 $H^-$ 对齐，伪代码却先 Commit anchor 再 `SelectMacro(M,H,Q,...)`，此时 $H$ 已含 $q_1$、$Q$ 还没弹出它。复现必须保存 anchor 前边界，或显式调整索引，避免把 anchor 算两遍。原文中的 $r,r',\ell$ 都是在当前草稿前缀上的数量，不应与完整 macro 长度 $K$ 混用。

## 6. 为什么首动作验证不等于整链正确

### 6.1 需要区分的三个事件

【推导审查】令：

$$
A_1=\{a_{\mathrm{actor}}=\widehat a_1\},
\qquad C=\{\text{被跳过动作与 Actor 反事实续写逐步一致}\},
$$

$$
S=\{\text{提交满足所定义的环境状态/副作用约束}\},
\qquad Y=\{\text{最终任务结果保持}\}.
$$

一般不存在 $A_1\Rightarrow C$，也不存在 $S\Rightarrow C$；任务结果保持 $Y$ 也不要求轨迹完全一致。SMC 用经验和检查争取较高的 $P(Y\mid\text{accepted})$，没有逐步验证到 $P(C\mid\text{accepted})=1$。

若 $V$ 代表观察到的 anchor、历史、状态与在线条件，反事实整链一致概率按条件链式规则为：

$$
P(C\mid V)
=\prod_{i=2}^{\ell}
P\left(a_i^{A}=\widehat a_i\mid V,
a_{2:i-1}^{A}=\widehat a_{2:i-1},\widehat o_{1:i-1}\right).
$$

没有测量这些后续条件概率，首步相等无法把乘积变成 1。离线归一化匹配的 $\underline p_m$ 也不是这里具体部署状态的逐步概率。

### 6.2 最小反例

【本文解释】Actor 和 Drafter 都先 `get_order(order_id=7)`，真实返回显示包裹已经送达。Drafter 仍沿历史高频链查询物流，Actor 则直接答复用户。首动作、参数、初始状态甚至首 observation 都一致，后续 Actor 决策仍可不同。

这个只读例子就足以否定“首动作确认推出整链相同”，不需要极端副作用。若后续存在修改，则还要额外控制操作风险。

论文已经承认这种近似，因此反例只解释保证对象，**不是发现作者未意识到的漏洞**。

### 6.3 与 AOSpec 的实质区别

AOSpec 在目标动作真正由 Actor 生成时验证该动作及 origin state；observation 续写也要验证真实字节。它主要把工作提前，保留后续验证。SMC 则用宏规则直接省掉后续 Actor 决策。

所以不能简单提“给 SMC 加 AOSpec 状态 hash 就成为 lossless”。状态 hash 可以帮助验证工具执行是否来自合适状态，不能补回未发生的 Actor 决策；若逐步调用 Actor 验证，又会吃掉 SMC 的主要收益来源。

## 7. 延迟与成本怎样理解

【推导审查；非原文公式】对于同一条固定轨迹，串行延迟可写为：

$$
L_{\mathrm{seq}}=\sum_{i=1}^{n}(A_i+T_i+O_i),
$$

$A_i$ 是 Actor 调用时长，$T_i$ 是工具时长，$O_i$ 为执行器等开销。SMC 会改变轨迹，因此这个分解只能作局部反事实比较，不能从主表反推出每项节省。

在 anchor 之后，对齐剩余 $\ell-1$ 步。与同运行时 SA 相比，若其对应 Actor 时间为 $A_i$，SA 仍需等待的工具时间为 $W_i^{SA}$，宏新增提交/验证/replay 时间为 $C_m$，竞争及路径改变带来的额外时间为 $R_m$，局部净收益近似为：

$$
G_m\approx\sum_{i=2}^{\ell}(A_i+W_i^{SA})-C_m-R_m.
$$

**不能把完整 $T_i$ 再算一次**：SA 可能已经复用提前完成的工具，SMC 相对 SA 的核心额外收益是少调用 Actor。AppWorld live replay 的时间也要进入 $C_m$。

在给定近似误差预算的策略分析中，可另定义：

$$
U_m=p_m^{\mathrm{keep}}G_m^{\mathrm{good}}
-(1-p_m^{\mathrm{keep}})C_m^{\mathrm{bad}}-C_m^{\mathrm{extra}}.
$$

这里 $p_m^{\mathrm{keep}}$ 是任务结果保持概率，$G_m^{\mathrm{good}}$ 是保持结果时的毛节省，$C_m^{\mathrm{bad}}$ 是坏提交的恢复/失败代价，$C_m^{\mathrm{extra}}$ 是其他额外资源成本；必须统一量纲，或把质量风险与延迟分开做约束优化。**这是候选分析模型，不是论文采用的算法，不能用离线模式准确率直接代替 $p_m^{\mathrm{keep}}$。**

长度门槛 $L_{\min}$ 是收益的粗代理；相同两步可能各需 0.1 秒或 10 秒，且提交开销不同。消融支持“原始命中数不够”，尚未证明这个门槛是最优调度。

## 8. 理论实现流程与缺失接口

### 8.1 离线构建

```text
split tasks into mining / tuning / final evaluation sets
extract recurring action sequences from successful mining traces
normalize task-specific values while recording shared argument slots
for each candidate macro:
    sample training opportunities
    run the deployment drafter from each sampled state
    compare normalized proposals with reference continuations
    count n and k
    lower = beta_inverse_cdf(delta, k + 1, n - k + 1)
    retain if n >= n_min and lower >= tau
select n_min and tau using tuning tasks and the intended executor
freeze library and settings before final evaluation
```

任务级隔离是防止相邻窗口泄漏的实现要求。原文只说明训练和 held-out 调参，没有给出足以完全复刻的所有 split 文件。

### 8.2 在线主循环

下面按原文控制流改写，显式保存 anchor 前历史；`validate_staleness`、`safe_commit` 和 replay 接口的细节仍是待补齐项，不冒充作者源码。

```text
H, E = initial_history, initial_live_state
start drafter on isolated snapshot, maintaining executed chain Q
actor_future = submit_actor(H, E)

while task not finished and budget remains:
    a = await actor_future
    Q = freeze_consistent_view_of_completed_draft_prefix()
    H_before, E_before = H, E

    if Q is empty or stale or full_call(a) != full_call(Q[0].action):
        o = execute_on_live(a, E)
        commit_authoritative_pair(a, o)
        discard_old_draft_and_restart(H, E)
    else:
        safe_commit(Q[0])                       # anchor，经 Actor 确认
        r = 1
        c = select_alignment(M, H_before, Q, L_min)
        if c exists and online_checks(c, H, E, task):
            safe_commit_completed_tail(c)      # 未逐步调用 Actor 确认
            r = c.draft_prefix_length
        pop Q[0:r]
        ensure_remainder_origin_matches_committed_prefix()
        continue_drafter_or_restart_if_invalid()

    actor_future = submit_actor(H, E)
```

原文只明确 Actor 失配时重启；这里把检测到 stale/合并不一致时重启写为实现防护，不能宣称是论文完整规定。部分宏提交失败时必须从实际已提交前缀恢复，不能在日志里假设整段成功。

### 8.3 最少数据结构与复现障碍

| 对象 | 最少记录 | 原文缺失或应核查 |
|---|---|---|
| Macro | 工具序列、槽位绑定、$n,k,\underline p$ | 挖掘算法、库大小、长度分布 |
| Draft step | 完整动作、真实返回、起止时间、前后状态/事务信息 | 每步 checkpoint、草稿版本与取消同步 |
| Alignment | macro ID、$j,\ell$、槽位替换 | 多候选排序、历史后缀处理 |
| Safety policy | API 白名单/拒绝表、可逆状态分类 | AppWorld 具体规则、Telecom 规则 |
| Commit log | anchor、跳过步数、replay 和提交耗时 | 原子性、失败恢复、状态/返回比较标准 |
| Evaluation log | 任务 ID、成功、实际步数、预算、Actor 请求和 GPU 时间 | 数据 split、重复次数、每任务配对结果 |

先用确定性内存/文件工具验证“只提交前缀、不泄漏余链”的状态机，再接已有 speculative-action 项目，是可控的实现顺序。当前缺作者实现，不宜声称仅按伪代码就能复现主表；本轮不启动代码开发。

## 9. 实验设置与主结果

### 9.1 环境与公平比较对象

【作者陈述；Results / Setup】Actor 为 Qwen3.5-27B INT4，Drafter 为 Qwen3.5-4B，均 greedy decoding。串行基线用 1 张 GPU；SA 和 SMC 用 3 张：Actor、服务投机请求的 Actor 同级副本、Drafter。副本用于避免投机请求排在正式请求之后。

**SA 与 SMC 是同运行时、同硬件的对照，仅切换 macro commit 规则**，适合隔离该规则的增量效果；相对单 GPU 串行的收益则同时包含额外硬件与动作投机。原文没有解释所有请求为何及何时路由到 Actor 副本，也未给 GPU 型号、完整 serving 配置和 token/GPU 成本。

这里的 SA 是作者执行器中的 **SA-only baseline**，不是已经验证等价于 Speculative Actions 官方代码全部设置的复现。不能把它与 AOSpec 的单步回放 SA 数字直接横比。

作者运行完整工具任务、测平均每任务实际 wall time，与 AOSpec 固定轨迹回放不同；但 greedy decoding 也不保证不同异步调度、环境过程或模型版本下逐次运行完全相同。

### 9.2 指标公式

【对文中指标的数学展开】$N$ 个任务、任务成功标记 $y_b$、耗时 $t_b$：

$$
\mathrm{Accuracy}=\frac1N\sum_{b=1}^{N}y_b,
\qquad \overline T=\frac1N\sum_{b=1}^{N}t_b.
$$

Telecom 使用二值任务正确率；AppWorld 使用 task-goal completion（TGC），主表按完成任务数呈现。延迟下降率为：

$$
\mathrm{Saving}_{X\mid B}=1-\frac{\overline T_X}{\overline T_B},
\qquad
\mathrm{Speedup}_{X\mid B}=\frac{\overline T_B}{\overline T_X}.
$$

主表 $\Delta$ 用负号表达相对串行的下降。质量变化需按百分点与任务个数分别写，不能与延迟百分比混淆。

### 9.3 主表与配对结果

| Benchmark | 方法 | 正确率 / TGC | 平均秒/任务 | 相对串行延迟变化 |
|---|---|---:|---:|---:|
| $\tau^2$ Telecom | Serial | 99.52% | 27.60 | — |
| | SA-only | 99.47% | 25.03 | −9.31% |
| | SMC | 99.52% | 22.47 | −18.59% |
| AppWorld | Serial | 41.67% | 355.7 | — |
| | SA-only | 41.67% | 212.1 | −40.37% |
| | SMC | 40.48% | 195.9 | −44.93% |

- Telecom：作者报告 SMC 与串行逐任务结果相同，均 2,274 成功、11 失败，即 2,285 个评测实例；相对 SA 修好 1 个、无回退。SMC 比 SA 快 **10.23%**。
- AppWorld：成功从 **70/168 到 68/168**，净少 2 个，约 **1.19 个百分点**。SMC 比 SA 快约 **7.64%**，摘要四舍五入为 7.7%。不能写成“只有两个任务轨迹改变”，因为可能同时有修好和回退。
- 2,285 个 Telecom 实例如何由任务、变体或重复生成，具体 split 与评测配置没有充分说明；不能把这个数直接当成标准官方独立任务数或默认配置。

没有多次运行误差、完整置信区间或任务级公开日志。主结果可支持该运行配置的观测收益，不能支持普遍不降质，尤其 AppWorld 已出现质量取舍。

## 10. 覆盖率与消融：逐项区分证据

### 10.1 Macro 覆盖

| Benchmark | 至少提交一次 macro 的任务占比 | Macro commits | 跳过 Actor 步数 | Skip density |
|---|---:|---:|---:|---:|
| AppWorld | 62.0% | 219 | 512 | 3.81% |
| Telecom | 86.2% | 3,352 | 7,154 | 3.91% |

正文笼统描述 skip density 为 skipped/total steps，但表注明确分母是 `tasks × max_steps`。应按表注写为：

$$
\mathrm{SkipDensity}_{\mathrm{budget}}
=\frac{\sum_b\mathrm{Skipped}_b}{N\cdot\mathrm{max\_steps}}.
$$

这不等于实际执行步数占比：

$$
\mathrm{SkipFraction}_{\mathrm{actual}}
=\frac{\sum_b\mathrm{Skipped}_b}{\sum_b\mathrm{ActualSteps}_b}.
$$

用主表任务数和共同步上限 80，可重建 $512/(168\times80)=3.81\%$ 与 $7154/(2285\times80)=3.91\%$；**80 是从这些数值反推的一致值**，不是 Setup 完整配置表明示。实际步骤数和预算如何计入跳过动作仍需原始日志。

两 benchmark 的预算密度接近，不能单独推出“实际省掉了同等比例计算量”。平均每次 macro 跳过约 2.34 步（AppWorld）和 2.13 步（Telecom），也没有按这些步骤的时长加权。

### 10.2 AWO-like 与 passive commit

【作者陈述；Ablations / Interface】单 GPU 的 Telecom 对照：

| 方法 | 正确率 | 秒/任务 | 相对串行 |
|---|---:|---:|---:|
| Serial | 99.52% | 27.60 | — |
| 注册 macro 为工具（AWO-like） | 99.34% | 27.89 | +1.05% |
| 无完整在线检查的 passive commit | 96.48% | 24.47 | −11.34% |

作者观察该 Actor 很少选择新 meta-tool，因而暴露接口没有省掉足够决策；直接隐式提交则更快但降质。它支持“在此配置下隐藏 runtime 复用值得研究”，不能证明所有显式 meta-tool 方法都失败：这里只是 AWO-like 实现，也没有控制所有提示和训练选择。

单 GPU 对照与完整 SMC 的三 GPU 主表不能作为仅改变一个检查项的严格消融。它展示两种替代方案的失败方式，但没有单独估计每个在线 guard 的因果增益。

### 10.3 提交筛选审计：不能把最后一行叫精确率 100%

【作者陈述；Commit Precision】100 个 held-out Telecom 任务：

| 条件 | 事件数 | 正确 / 保持结果 | 检查对象 |
|---|---:|---:|---|
| 仅库匹配 | 1,968 | 681/1,968 = 34.6% | 反事实后续步骤 exact match |
| + Drafter 已执行 | 885 | 625/885 = 70.6% | 同上 |
| + Anchor 确认 | 711 | 625/711 = 87.9% | 同上 |
| + $L_{\min}=1$ | 343 | 310/343 = 90.4% | 同上 |
| 实际触发提交 | 158 | 158/158 = 100% | **最终任务结果保持** |

前四行暂不提交，让 Actor 自己继续以获得事件级反事实；最后一行实际提交并看任务结果。因而不能概括成“经过安全检查，动作精确率由 90.4% 提高到 100%”。另外 macro mining 标签是归一化调用序列，审计则写 exact future steps，二者也需核对比较实现。

343 个候选为何最终只触发 158 次、每个 guard 拒绝多少、结果不变但动作不同占多少，正文未完整拆解。同一任务中多个事件并非独立任务样本，158/158 不构成严格零风险保证。

### 10.4 跳得更多为什么反而慢

| 方法 | 正确率 | 秒/任务 | Macro commits | 跳过步数 | 相对 SA 延迟 |
|---|---:|---:|---:|---:|---:|
| SA | 99.47% | 25.03 | 0 | 0 | — |
| Legacy macro | 99.56% | 25.44 | 6,410 | 10,528 | +1.64% |
| Final SMC | 99.52% | 22.47 | 3,352 | 7,154 | −10.23% |

作者归因于 legacy 提交太浅、与已执行草稿对齐差、同步开销抵消收益；最终实现压掉 6,281 个 depth-one opportunities，并要求足够长度和已执行工作。

这支持“命中数/跳过总步数不是充分目标”。但 legacy 和 final 改了不止一个因素，缺少逐 guard、固定轨迹和计时分解，不能据此把全部差值归因于 $L_{\min}$。文中 depth-one 与 after-anchor skipped length 的术语也应在源码里核对，不能简单当成恰好跳过一步。

### 10.5 AppWorld 切片与一个数值一致性疑问

| 切片 | 任务数 | SA 总秒 | SMC 总秒 | 减少 |
|---|---:|---:|---:|---:|
| Same-accuracy | 143 | 28,920 | 25,014 | 13.5% |
| 两边均无 no-tool-call（NTC）步骤 | 85 | 16,375 | 14,625 | 10.7% |

作者解释：结果改变、恢复过程、没有工具调用却消耗预算的步骤，稀释了全任务平均收益。SMC 的 NTC 为 31 个任务、68 步，SA 为 78 个任务、105 步。但**同成功/失败不等于同轨迹**，按运行后的结果筛选也不是随机对照，这些切片是描述性解释，不能直接当作“纯 macro 效果”的无偏估计。

【推导审查；待作者数据核实】若 143 是主表同一批 168 个任务中“双方二值结果相同”的数量，则有 25 个 discordant tasks。令 $b$ 为 SA 成功而 SMC 失败，$c$ 为反向改善，根据主表 70 与 68 应有：

$$
b+c=168-143=25,\qquad b-c=70-68=2.
$$

解得 $b=13.5,c=11.5$，不可能是整数。因此主表、切片数量或“same accuracy”的集合定义至少需要额外解释，可能来自不同运行/筛选或文字错误。**不据此判定造假，也不擅自修正为某个数字；后续复现必须索取配对任务日志。**

AppWorld 为什么降质，正文没有给出逐失败案例或足以定位原因的参数/状态/replay 分类；可以提出机制假设，不能直接归因于某种特定失配。

## 11. 与已有论文建立关系

| 比较对象 | 已有工作已覆盖什么 | SMC 的实质区别 |
|---|---|---|
| Speculative Actions | Actor/Drafter、提前执行、调用匹配、深度扩展 | 从逐步确认变为首步 anchor 后的近似多步提交 |
| DSP | 根据上下文动态决定草稿长度 | SMC 的 $L_{\min}$ 是提交候选的最低跳过数，不是 DSP 的未来长度预测器 |
| PASTE | 历史模式、参数绑定、工具预执行、隔离和延迟/资源调度 | SMC 把 macro 隐藏在 executor，并直接跳过后续 Actor 决策；原文无与 PASTE 的实验对比 |
| SPORK | 部分前缀自投机、共享 KV、失败草稿 token 验证复用 | SMC 不提供 token-level 接受机制，新增的是多步 commit 规则 |
| Speculate While You Reason | 训练同模型预测自己的下一工具调用 | SMC 不训练 Actor 选择 macro，使用库筛选与运行时近似 |
| Speculate with Memory | 检索经验/失败轨迹改善预测，含动作和返回预测 | SMC 的库直接参与是否省掉 Actor 调用；不只是提升预测器 Hit@1 |
| AOSpec | 隔离目标动作预执行、origin-state 验证、observation 字节校验 | SMC 没有逐步 Actor 确认，语义近似；AOSpec 的受限状态验证不能替它证明整链策略一致 |

还应登记最近邻 **AWO / Agent Workflow Optimization**：原文引用其从轨迹挖掘 meta-tools，但本轮未精读原始论文。不能把 SMC 的 AWO-like baseline 当作完整复现 AWO，也不能因两者界面不同就忽略已覆盖的 workflow compression 思想。

**当前已不能作为独立新意的表述**：历史动作 skeleton 库；ID 参数槽位化；按置信下界过滤 pattern；首动作 anchor 后提交多步；对隐式 macro 加在线状态/参数检查；用最小跳过长度抑制小收益提交；Actor 失配才重启、命中则继续使用草稿余链。

## 12. 候选研究问题：先诊断，后决定

### S1：离线“归一化链可靠”与在线“任务结果保持”之间差多少

- **依据**：Beta 标签、审计 exact match、实际任务结果是三个不同对象；AppWorld 有净质量下降。
- **最小实验**：固定小样本任务与草稿，影子运行不提交，分别记录 normalized match、完整参数 match、observation 条件下的 Actor continuation match；再用相同候选比较提交后的任务结果。按训练成功/失败恢复状态分组。
- **简单参照**：原始 $k/n$、Beta 下界、按任务分组的保守统计、增加状态条件的简单分桶；不要一开始训练风险大模型。
- **放弃条件**：几种标签在目标场景几乎一致、样本数太少无法稳定判断，或简单分桶已解决；相关 selective prediction / workflow 文献已直接覆盖。

### S2：最小步数门槛是否能被简单时间门槛替代

- **依据**：legacy 跳更多却更慢；$\ell-1$ 不表达实际 Actor 时长、SA 剩余等待和 replay 成本。AOSpec EVD、PASTE 已有价值加权，泛泛“加入延迟”不新。
- **最小实验**：固定同一候选和质量约束，比较调优 $L_{\min}$、按历史中位 Actor 时间估算净收益、固定提交开销阈值；记录真实关键路径而不是总 skip。
- **放弃条件**：调优长度门槛已够好，或可测收益只来自额外硬件和改了质量阈值。复杂模型必须击败这些简单参照才有继续价值。

### S3：能否明确划分“安全执行”与“可以省略决策”的条件

- **依据**：首动作确认与状态/replay 检查解决不同问题；加 hash 不能恢复未验证的策略选择。
- **最小实验**：确定性小环境中构造首调用相同、返回触发不同后续选择的例子；对比 SA 逐步确认、SMC 近似提交、人工指定的确定性子流程。测轨迹差异、环境差异、任务差异和成本。
- **研究限制**：如果引入已知确定性流程或契约，必须明确这是额外假设；如果恢复逐步 Actor 验证，应如实计算丢失的加速。不能靠更换“正确性”定义宣称同时获得零风险和全部跳步收益。
- **放弃条件**：目标任务没有可认证流程，或已有事务/工作流方法直接适用；仅写一个反例并不足以形成论文。

### S4：质量—延迟比较是否被配对和预算口径影响

- **依据**：skip density 分母是步预算；AppWorld 同结果切片有计数疑问；串行和投机硬件不同。
- **最小工作**：先做每任务日志核对，报告同一任务集上的改善/退化矩阵、实际步数、GPU 占用或预留成本、全部任务与配对切片的结果。必要时在固定质量阈值下比较 Pareto 曲线。
- **边界**：这首先是复现与评测修正；除非发现稳定、可迁移且现有方法未解决的机制，不把查表或报更全指标单独当作创新。

## 13. 总体判断与下一篇

SMC 的主要贡献在**执行器方法与实证层**：不用 Actor 学会调用新 macro，而由执行器识别并提交已执行的草稿尾部，直接省掉若干大模型决策。它的主要代价已经明确写在论文里——近似执行；统计筛选和环境检查缓解风险，不把它变成严格 lossless。

相对 AOSpec，它提供实际 wall-clock 的同硬件 SA 增量比较，但保证更弱、复现接口和配对数据还不完整。对我们的选题最重要的约束是：**“memory 匹配动作链后多步提交”已被直接做过，下一步只能围绕清晰的条件、测得的差距和可复现 baseline 收窄。**

下一篇按路线读 **AgentSpec: Speculative Decoding for Batch Inference of LLM Agents**，进入 token-level 分支；重点区分目标分布验证与宏近似提交，以及批处理下的预算和资源竞争。本轮到第八篇结束，不提前开展系统实现。

## 14. 本地证据入口

- [Macro 挖掘、Beta 公式、提交规则与 Algorithm 1](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculative Macro Commit for Faster Tool-Using Agents/emnlp_sections/method.tex>)。
- [明确声明 approximate 的 Related Work](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculative Macro Commit for Faster Tool-Using Agents/emnlp_sections/background.tex>)。
- [主结果、硬件、覆盖率和 AppWorld 切片](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculative Macro Commit for Faster Tool-Using Agents/emnlp_sections/results.tex>)。
- [接口替代、提交审计与 Legacy 消融](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculative Macro Commit for Faster Tool-Using Agents/emnlp_sections/ablations.tex>)。
