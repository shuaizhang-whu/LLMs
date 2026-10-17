# R01 批注：对「根据投机候选数量训练动作预测器」的核查

核查对象：`GPT输出记录/创新点探讨.md`
核查所用材料：Speculative Actions **LaTeX 源码**（非笔记）、本地已复现代码 `Projects/speculative-action/hotpotqa`
日期：2026-09-11

---

## 结论

这份提案的**数学不是错的**——GPT-6 关于「K 改变最优概率分配」的直觉，在总体目标层面可以严格推导出来（见第二节）。但它有四个更硬的问题，其中第一条是致命的：

1. **核心公式 `p(k) = 1-(1-p)^k` 就是 Speculative Actions 原文的公式**，且 SA 已有它的异质置信度推广版，还已有基于它的宽度选择定理和实现。
2. 该训练目标在**真实工具调用的概率量级下退化**，「K 改变分配」这一机制不可观测。
3. 即使不退化，同一分布族可以用**温度重标定从标准 NLL 模型免费得到**——存在零训练强基线。
4. 它优化的是**命中率**，而 SA 自己的定理已经说明命中率不足以决定目标。

---

## 一、致命项：公式与机制都已在 baseline 论文里

GPT-6 方案的第 213 行：

```
P_hit = 1 - (1 - q)^K
```

同一条公式，Speculative Actions 的 `sections/revision_addition.tex` 第 6–10 行（**非注释，正文**）：

```latex
\subsection{Breadth-focused speculation details}
Algorithm 1 launches $k$ parallel speculative branches...
We assume each branch independently produces the correct next call with probability $p$. Let
\[
p(k) := \Pr(\text{at least one branch is correct}) = 1 - (1-p)^k.
\]
```

而且 SA 走得更远。`sections/cost-latency-tradeoff.tex` 第 88–92 行给出了**每个分支置信度不同**的一般化：

```latex
If the top $m$ branches are launched, the probability of a cached step is
$
q(m;\mathbf p) = 1 - \prod_{j=1}^{m}(1-p^{(j)}),
$
where $p^{(1)} \ge \dots \ge p^{(k)}$ are the sorted confidences.
```

GPT-6 的式子是同质特例 `p^(j) ≡ p`。**即：GPT-6 独立重新推导了 SA 已有的公式，而且是最弱的那个版本。**

SA 还已经覆盖了 GPT-6 方案叙事里的另外两件事：

| GPT-6 的论述 | SA 原文位置 | 状态 |
|---|---|---|
| 「K 越大边际收益递减，应据 K 决定分配」 | `cost-latency-tradeoff.tex:51`：「given an estimation of $p(k)$, a user can directly use these expressions to **tune $k$ as a knob**」 | 已有定理，且明确以 k 为调优旋钮 |
| 「生成多个候选后**去重**」 | `cost-latency-tradeoff.tex:40-41`：$\tilde k$ 表示「distinct actions produced across the $k$ branches」，重复的投机进程被 kill | 已有，并已进入成本公式 |
| 「用置信度决定接受哪些分支」 | `cost-latency-tradeoff.tex:76-90`：观察到 accuracy vector 后「choose how many of the top branches to launch」 | 已有动态选择性投机 |
| top-1 vs top-3 命中率对比 | `appendix.tex:182-198`：$k \in \{1,3\}$，HotpotQA top-3 命中 46% | **已是 SA 自己的评估指标** |
| 简单阈值规则 | `cost-latency-tradeoff.tex:183`：chess 环境实现 50% 阈值启发式 | 已实现并跑过 |

另外 `main.tex:108` 有一行**被注释掉**的草稿（注意：是注释行，不能当作已发表主张）：

```latex
% ... improved through stronger guessing models, top-$k$ action prediction,
% multi-step speculation, and uncertainty-aware optimization ...
```

作者自己已经把「更强的预测器 + top-k 动作预测」列为后续方向。GPT-6 的方案正好落在这两项上。

---

## 二、数学诊断：机制在真实区间退化

先给出 fairness：GPT-6 的直觉**在总体目标层面是对的**。考虑

    J_K(q) = Σ_a p_a [1 - (1 - q_a)^K]

对 K>1，`1-(1-q)^K` 关于 q 是凹函数，在单纯形上最大化会得到**内点解**，即概率确实被摊开；K=1 时是线性目标，解在顶点（集中猜 argmax）。所以「K 影响最优分配」在总体意义上成立。

**问题出在它写下的训练损失只看到标签处的概率**：

    L(θ) = -(1/N) Σ [1 - (1 - q_θ(a*|x,K))^K]

对 q 求导：`∂L/∂q = -K(1-q)^{K-1}`。
对照标准 NLL：`∂(-log q)/∂q = -1/q`。

两者的梯度之比是：

    K·q·(1-q)^{K-1}

**当 q ≪ 1 时（工具调用的完整动作序列，q 是整串 token 的联合概率，量级约 1e-6 到 1e-15）：**

- `(1-q)^{K-1} ≈ 1 - (K-1)q ≈ 1`
- 梯度之比 ≈ `K·q`

也就是说：**损失退化为「以 q 为权重的 NLL」**。K 只作为整体尺度出现，几乎不改变更新方向；真正主导每样本权重的是 q 本身，而不是 K。等价地：这是一个**置信度加权的 NLL**（focal-loss 那一类重加权技巧），不是新的分配机制。

更严格的表述：在内点解条件 `p_b (1-ν(b))^{K-1} = const` 中，当 `ν(b) ≪ 1` 时 `(1-ν(b))^{K-1} → 1`，条件退化为 `p_b ≈ const`，**无内点解 → 解退到顶点**。即 K 依赖性是「q 不太小」才有的现象，而真实工具调用恰好永远不在那个区间。

GPT-6 在第 274 行自己写：「完整动作很长时，联合概率可能很小，这是后续需要解决的训练问题」——它把这个当成工程小麻烦，实际上它**删掉了整个机制**。

---

## 三、必须跑的零训练证伪基线

就算把 q 不小的理想情形造出来（例如把动作离散成短标签），第二节的推导也说明：最优解是一族按 K 索引的分布，在 argmax（K=1）与真实分布 p（K→∞）之间插值。

**而标准 NLL 训练的动作预测器本身就在估计 p(a|x)。** 因此这族分布可以**后验地**从单个标准模型得到：

    对标准 NLL 模型输出做温度重标定 / top-K 重整，扫描 T 得到不同「集中度」

**零训练、零新参数的强基线。** 如果它在相同 K 下达到或超过 GPT-6 的方案，整个贡献归零。

这是这个提案最该先做的实验，成本极低：不需要训练、不需要新代码，只要拿现成预测器的 logits 做温度扫描 + top-K 采样，测「至少一个命中」。

---

## 四、目标错位

方案优化的是**给定 K 的命中率**，但：

- 论文目标是**延迟**，命中率只是代理指标。GPT-6 第 311 行自己承认：「命中概率提高也不自动意味着加速」。
- 更关键：**SA 的定理已经说明命中率不足**。`p(k)` 同时出现在延迟增益和成本增益的表达式中，SA 明确指出必须同时看两者才能选 k。在固定 K 下最大化命中率，是在优化一个 SA 已证明不能单独决定目标的中间量。
- 这正好是你笔记里反复记录的那个错误模式（Speculate with Memory / SU）：**命中、observation 正确、状态有效、可提交不是同一个指标**。GPT-6 的方案重复了这个模式，而且是在 baseline 已经给出正确决策变量的地方重复。

---

## 五、还活着的部分（是诊断入口，不是创新点）

> **勘误（2026-09-11 修订）**：本节初稿曾断言「speculator 用低温度采样，`constants.py:44 guess_temperature = 0.1`」。**该断言错误，已更正。**
> `guess_temperature / guess_top_p` 属于 **observation speculator**，见 `constants.py:42` 的注释「Observation Guess LLM 设置：该模型预测 Wikipedia 风格的页面内容，**而非直接猜 action**」，构造于 `reproduction_runner.py:141-147`。
> action 候选由 `candidate_llm` 产出，与 Actor **同模型、同温度**（`reproduction_runner.py:134-140`，`temperature=self.temperature`）。

核查中**仍有一处裂纹，但理由与初稿不同，且更站得住**：

- SA 定理假设各分支**独立**命中，概率 p（`sections/revision_addition.tex:7-10`）。
- 但实现里 K 个候选**不是 K 次独立采样**，而是**一次生成调用里连续吐出**：`candidate_llm` 调用 `ACTION_GUESS_PROMPT`，prompt 要求一次返回恰好 K 个候选（`prompts.py:12-19`：「return {num_guesses} candidates … return precisely {num_guesses} valid actions - no more, no less」）。第 i+1 个候选是在已生成前 i 个的条件下产出的。
- 代码不在 action 层强制去重（`runner.py:200` 注释：「解析后不会在代码层强制截断或去重」），而 SA 附录的成本项却用 $\tilde k$ 假定重复已被 kill。

后果：`p(k) = 1-(1-p)^k` 的**独立性是理想化，实现并不满足**；理论预测的「增大 k 的收益」与实测可能系统性偏离。

**为什么现在还不能当创新点**：
1. 这是 SA **明确声明过的假设**（"We assume each branch independently..."），不是遗漏。攻击一个已声明假设，必须证明后果是量级级别的；
2. 单独「修正公式」按规则 7 不构成贡献；
3. 偏差幅度尚未量测。

**廉价探针（数据你已有）**：复现目录里已有 `paired_top1` / `paired_top3` 与 `top1_top3_action_accuracy.png`。从 top-1 命中率拟合 p，再与实测 top-3 命中率对比 `1-(1-p)^3` 的预测值。偏差不随 k 放大 → 作废；偏差过半 → 「何时不该加宽」才可能支撑边界性结果。

---

## 六、给 GPT-6 下一轮的约束建议

如果继续用它发散，建议把以下四条写进提示词，避免重复已经排除的方向：

1. `p(k) = 1-(1-p)^k` 及其异质推广 `1-∏(1-p^(j))` 是 SA 已有公式，不能作为贡献；
2. SA 已有：以 k 为旋钮的成本–延迟定理、去重（$\tilde k$）、置信度向量驱动的 top-m 选择、50% 阈值启发式实现、top-k 命中率评估；
3. 任何「K 影响最优分配」的机制必须说明在 q ≪ 1 时如何不退化为置信度加权 NLL；
4. 「提升命中率」类目标必须附带说明它如何转化为端到端延迟，且需与温度重标定基线对比。

---

## 附：本批注中三类陈述的区分

- **已是论文结论（有源码出处）**：第一节全部引文、SA 的定理与实现。
- **我们的推导（可复核）**：第二节的梯度之比 `K·q·(1-q)^{K-1}` 与退化结论、第三节的温度重标定基线。
- **待确认假设**：第五节裂纹的量级（未实测）；「温度重标定基线能否匹配」未实验。
