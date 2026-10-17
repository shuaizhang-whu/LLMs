# 05 · Speculate While You Reason：V 型分析、方法公式与实现路径

阅读日期：2026-09-09。完整标题：**Speculate While You Reason: Teaching Agents to Predict Their Next Tool Call via Joint Agent–Speculator RL**，下文简称 SWYR。

已读本地全部正文、附录与表格，核对示例图、奖励消融图；[官方论文](https://arxiv.org/abs/2607.25816)，首版 2026-07-28。本轮只完成阅读，不训练模型、不运行项目。

标记约定：**【原文】**为作者明确给出的内容；**【补充原理】**为解释引用算法所需的标准公式；**【分析/推导】**为本笔记判断；**【待确认】**为论文不足以确定的实现细节。原文未给出的公式或超参数，不能为了“写全”而编造。

## 第一部分：【第一步】V 型结构拆解

| 层次 | 判断（每层 1–3 句） | 具体依据 |
|---|---|---|
| 表象层 | Agent 等工具结果耗时，提前运行工具有望减少等待，但预测错了就白做工作。作者想让模型更准确地提前报出自己的下一调用，同时保持任务完成能力。 | Introduction；Abstract |
| 技术层 | 外部 drafter 或历史调用不一定匹配当前 Agent 的具体选择；共享模型虽减少这种差异，却引入长任务轨迹与短调用预测之间的训练干扰。 | §2 外部/自身预测对照；§3 双模式训练；§4 消融 |
| 机理层 | 作者给出两个解释：预测者与 Agent 的行为分布不一致，以及两种目标通过共享参数、优化器历史互相干扰。前者有匹配率对照，后者有训练消融支持，但未测量条件分布差异、梯度冲突或动量的直接因果效应，所以主要是合理机制假设，未成为严格的新规律。 | §2 Findings；§3 Stabilizing Dual-Mode RL；奖励消融图 |
| 哲学层 | **未触及/不强行拔高。** “能预测自己的下一步”在这里是给定提示与轨迹的行为匹配任务，不能据此推断自我意识、元认知或已经理解自身决策过程。 | 全文指标为调用匹配与任务成功，没有对应的认知实验 |

### 从机理到方法、再到评测的闭环

`预测行为不一致 → 同模型共享参数 + 当前轨迹标签 → 全调用 Hit@1 提升`：有实验支撑，但共享参数不消除两种提示/上下文的差别。

`两模式训练干扰 → SFT + 分块交替 + 重置优化器 → 奖励曲线更稳`：有消融支撑，但尚未充分隔离优化器状态、更新频率和训练预算的贡献。

`Hit@1 提升 → 工具更早完成 → 端到端更快`：**闭环未完成**。主评测只执行 Agent 正式调用，未实测训练后完整并行预执行系统的 E2E 收益。

## 第二部分：【第二步】创新层次判定

- [ ] 表象层创新：加速工具调用是已有需求，未提出新场景。
- [x] 技术层创新：将自身轨迹监督、双模式 SFT、交替 RL 和优化器重置组成训练流程，并提供实证。
- [ ] 机理层创新：有机制解释，但没有证明新的信息边界或优化规律。
- [ ] 哲学层创新：未触及，不强行拔高。

**是修补还是重构？** 相对“训练一个外部预测器”，它调整了预测者与执行者的组织方式；但同模型 self-probe 已在我们读过的 SPORK 中出现，因此其区别应落在**如何训练共享模型并维持任务能力**，而不是“首次让模型预测自己”。

**是否从机理自然推导？** 共享参数和新鲜标签与行为对齐问题有关；4:8 调度、SFT warmup 和切换时重置主要是经验配方，没有从分析推导出必需的比例或唯一最优方案。它们可以有技术价值，不需要包装成已发现底层规律。

## 第三部分：【第三步】本质提炼

这篇论文本质上在做的事情是：让一个做事的人，在继续思考前先准确报出自己接下来会做什么，并通过练习同时保持做事能力。它试图回答的根本问题是：能否让“提前说出来”变得更准确，而不让“真正把事情做好”变差？

## 第四部分：方法、完整公式与理论实现

### 4.1 符号与两个工作模式

| 符号 | 含义 |
|---|---|
| \(x,y,y^*\) | 用户问题、最终答案、任务参考答案 |
| \(r_t,a_t,o_t\) | 第 t 次工具调用前的 reasoning、调用、真实 observation |
| \(a_t=(n_t,\sigma_t)\) | 工具名与参数字典；\(\sigma_t\) 不是标准差 |
| \(h_t,s_{\mathrm{sp}}\) | 调用预测可见的轨迹前缀、固定 speculation suffix |
| \(\pi_\theta\) | 一套共享参数的模型；不同输入切换工作模式 |
| \(G_{\mathrm{ag}},G_{\mathrm{sp}}\) | 同一问题的 Agent 样本数；同一前缀的 Speculator 候选数 |
| \(k,m(k)\) | RL 迭代编号、当前更新模式；不是投机深度 |

【原文】轨迹与调用表示为：

\[
\tau=(x,r_1,a_1,o_1,\ldots,r_T,a_T,o_T,y),
\qquad a_t=(n_t,\sigma_t).
\tag{1}
\]

同一策略在 speculator 模式生成：

\[
\hat a_t\sim\pi_\theta(\cdot\mid h_t\oplus s_{\mathrm{sp}}),
\qquad \text{target}=a_t\text{ from the current rollout}.
\tag{2}
\]

Suffix 原文如下，在外部预测、SFT、RL 和评测时保持一致：

```text
<think> Okay, let's see. The user provided what I need.
I'll look it up. The next step is to make the tool call.
</think>
```

**前缀边界必须确认。** §3 写的 \(h_t=(x,r_1,a_1,o_1,\ldots,a_{t-1})\) 省略了上一 observation，与“调用之前的前缀”文字及示例图并不完全对应；示例图确实在前缀中提供了上一工具 observation。正文也没有给出通用的 token 截断位置或剩余 reasoning 时长。

【补充实现记号，不是原文精确定义】为避免未来复现含糊，可将前缀显式写成：

\[
h_t^{(u)}=(x,r_1,a_1,o_1,\ldots,r_{t-1},a_{t-1},o_{t-1},r_t^{\leq u}).
\tag{3}
\]

这里 u 是已经观察到的本轮 reasoning token 数，u=0 表示本轮 reasoning 尚未开始。**作者实际各实验用了哪个 u，仍需代码或补充说明确认**；不能把完整动作 token 或未来 observation 放入预测输入。

### 4.2 评测指标：完整结构相等

【原文定义的数学展开】对 N 个工具调用事件：

\[
\mathrm{Hit@1}=\frac1N\sum_{t=1}^{N}
\mathbf1[\hat n_t=n_t\land\hat\sigma_t=\sigma_t].
\tag{4}
\]

比较的是**解析后的完整字典**，不是原始 JSON 字符串；键顺序通常不影响字典相等，但字符串值不同仍然失配。原文没有完整规定数值类型、默认值、浮点数等规范化细节。

任务成功率由环境评测，不等同于调用 Hit@1。外部小模型对应的任务成功率，是它所预测的固定 4B Agent 的成绩，不是小模型独立解决任务的成绩。

### 4.3 Speculator 奖励：原文公式全部展开

【原文】工具名得分：

\[
R_{\mathrm{name}}=\mathbf1[\hat n_t=n_t].
\tag{5}
\]

设 \(K_t=\mathrm{keys}(\sigma_t)\)，在名称正确时：

\[
R_{\mathrm{args}}=
\begin{cases}
\displaystyle\frac1{|K_t|}\sum_{q\in K_t}
\mathrm{F1}(\hat\sigma_t[q],\sigma_t[q]),&|K_t|>0,\\
1,&|K_t|=0.
\end{cases}
\qquad
R_{\mathrm{sp}}=R_{\mathrm{name}}R_{\mathrm{args}}.
\tag{6}
\]

解析失败得 0；缺失目标参数得 0；额外预测键不获得奖励，也不能补偿缺失键。每个目标键等权，避免很长的 query 淹没短但重要的 ID。

【补充原理】常见的 multiset token-F1 展开如下。将参数值规范化为字符串再分词，预测 token 多重集为 P、目标为 G：

\[
c(P,G)=\sum_v\min(\mathrm{count}_P(v),\mathrm{count}_G(v)),
\quad p=\frac{c}{|P|},\quad r=\frac{c}{|G|},
\]
\[
\mathrm{F1}(P,G)=\frac{2pr}{p+r}
=\frac{2c}{|P|+|G|}\quad (|P|+|G|>0).
\tag{7}
\]

常见实现对双方为空设 1、仅一方为空设 0。**SWYR 未公开到这一层的分词、大小写、标点、空值约定**；式 (7) 解释通常意义，不代表已确认作者 reward 代码。

**关键审查：奖励不是严格匹配的等价表达。** 若按词袋 F1，`red blue` 与 `blue red` 可得 1，却不是相同 query；按式 (6)，目标键完全正确但多出一个可解析的可选参数，也可能 reward=1、Hit@1=0。作者承认用稠密奖励提供部分分，因此应称为 exact match 的代理目标，而不是保证可复用的判据。

### 4.4 Agent 奖励：哪里无法写成作者的完整公式？

【原文】\(R_{\mathrm{ag}}(\tau)\) 由最终任务结果评估，且包含重复、无意义工具调用的惩罚。**论文没有给出任务评分函数、重复判据、惩罚强度或组合公式。**

理论实现可以预留接口：

\[
R_{\mathrm{ag}}(\tau)
=R_{\mathrm{task}}(y,y^*,\text{environment})
-\lambda_{\mathrm{rep}}C_{\mathrm{rep}}(\tau).
\tag{8：仅为实现占位，非作者已给公式}
\]

这里 \(\lambda_{\mathrm{rep}}\)、\(C_{\mathrm{rep}}\) 都待确认；不能擅自用“重复相同调用一次扣 1”代替作者方法，也不能断言所有领域都用最终答案 exact match 做任务 reward。

### 4.5 双模式 SFT warmup

【原文】先用成功 Agent 轨迹及其前缀—下一调用样本 warmup。**附录给出一个例外：ToolScale warmup 只有 5000 条 speculation 行，Agent 行为 0**，所以不能把“双模式混合 SFT”当成所有实验都完全相同的配置。

【补充原理】标准 teacher-forcing 目标可写为：

\[
\mathcal L_{\mathrm{SFT}}(\theta)=
-\mathbb E_{(c,z)\sim\mathcal D_{\mathrm{SFT}}}
\left[\frac1{\sum_\ell M_\ell}
\sum_\ell M_\ell\log\pi_\theta(z_\ell\mid c,z_{<\ell})\right].
\tag{9}
\]

M 标识需要监督的生成 token；外部 observation 和输入 prompt 应与模型输出区分。作者没有给出 SFT 的精确 loss reduction、mask、混合权重和完整超参数，式 (9) 是可实现的标准解释。

### 4.6 DAPO：把“使用 DAPO 更新”展开

SWYR 仅引用 DAPO，没有展开 loss。下面依据 [DAPO 原论文 §3、式 (8)–(12)](https://arxiv.org/html/2503.14476v1#S3)，补全核心更新；它不是 SWYR 新提出的算法。

对输入 c，旧策略采样 \(z_1,\ldots,z_G\)，得到奖励 \(R_1,\ldots,R_G\)：

\[
\bar R=\frac1G\sum_iR_i,
\qquad s_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},
\qquad \hat A_i=\frac{R_i-\bar R}{s_R+\epsilon}.
\tag{10}
\]

\(\epsilon\) 是数值保护；std 具体采用哪种估计须与实现一致。Agent 的分组单位是同一 x，Speculator 的分组单位是同一 \((h_t,a_t)\)，不能跨不同前缀混算相对优势。

逐 token 概率比为：

\[
\rho_{i,\ell}(\theta)=
\frac{\pi_\theta(z_{i,\ell}\mid c,z_{i,<\ell})}
{\pi_{\theta_{\mathrm{old}}}(z_{i,\ell}\mid c,z_{i,<\ell})}.
\tag{11}
\]

DAPO 的核心 token 平均裁剪目标为：

\[
\mathcal J_{\mathrm{DAPO}}(\theta)=
\mathbb E\left[
\frac{1}{\sum_i|z_i|}
\sum_i\sum_{\ell=1}^{|z_i|}
\min\left(
\rho_{i,\ell}\hat A_i,
\operatorname{clip}(\rho_{i,\ell},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})\hat A_i
\right)\right].
\tag{12}
\]

训练最小化 \(-\mathcal J\)。工具交互轨迹中的 \(|z_i|\) 应对应参与训练的生成 token；环境插入文本需单独 mask。

DAPO 原版还动态过滤无有效组内差异的样本。二元奖励时条件可写为：

\[
0<\sum_{i=1}^{G}\mathbf1[\text{response }i\text{ correct}]<G.
\tag{13}
\]

**适配边界**：SWYR 的 \(R_{\mathrm{sp}}\) 是连续 F1；若要以“有非零优势”选组，应检查组内奖励方差，而非直接照搬二元正确率条件。SWYR 没说明实际过滤规则，也没明确其 DAPO-style 是否启用全部原版组件、KL 项、overlong shaping 或使用什么 clip 数值；不能把 DAPO 数学任务的默认配置当成 SWYR 配置。

### 4.7 交替更新与优化器重置

【原文 4:8 的数学展开】令 \(j=(k-1)\bmod12\)：

\[
m(k)=\begin{cases}\mathrm{ag},&0\leq j<4,\\
\mathrm{sp},&4\leq j<12,
\end{cases}
\quad
\mathcal L_k=
\begin{cases}-\mathcal J_{\mathrm{ag}},&m(k)=\mathrm{ag},\\
-\mathcal J_{\mathrm{sp}},&m(k)=\mathrm{sp}.
\end{cases}
\tag{14}
\]

不是每步同时优化两个加权 loss，也不是训练两套大模型。一次更新只采用当前模式的目标；共享权重仍可能同时影响两种行为。200 迭代按这个周期、从 Agent 开始，对应 68 次 Agent 和 132 次 Speculator 迭代；这不等于优化器实际 minibatch step 数，后者论文未详述。

【补充原理】以 Adam 类优化器为例，记一阶、二阶状态为 b、v：

\[
g_q=\nabla_\theta\mathcal L_k,
\quad b_q=\beta_1b_{q-1}+(1-\beta_1)g_q,
\quad v_q=\beta_2v_{q-1}+(1-\beta_2)g_q^{\odot2},
\]
\[
\hat b_q=\frac{b_q}{1-\beta_1^q},\quad
\hat v_q=\frac{v_q}{1-\beta_2^q},\quad
\theta\leftarrow\theta-\eta\frac{\hat b_q}{\sqrt{\hat v_q}+\epsilon}.
\tag{15}
\]

模式切换重置的是优化器状态（例如 b、v、bias-correction step q），**不重置模型权重**。论文未列完整 optimizer 配置；式 (15) 用于理解为何残留统计可能影响新目标，不声称作者逐项采用此配置。学习率 scheduler 是否重启也需确认。

### 4.8 理论上怎样实现整个训练流程？

以下保留 Algorithm 1 的流程，不将未披露细节伪装成可直接复现的完整代码：

```text
先准备：统一工具 parser、准确的轨迹截断器、task reward、spec reward
按数据域构建 warmup 样本，完成 SFT

for k in 1..200:
    从对应训练集抽取问题批次 Q
    保存本轮 rollout 策略 theta_old 和输出 token 的 old logprobs
    每个问题生成 8 条 Agent 轨迹；仅正式 Agent 调用环境
    按问题分组计算 Agent reward/advantages

    对轨迹中每个工具 turn：
        抽取可见前缀 h、实际下一调用 a，标签不参与反向传播
        输入 h + 固定 suffix，生成 8 个 speculation 候选
        解析候选，按 name × argument macro-F1 计算 reward
        对同一个 (h,a) 的候选组计算 advantages

    如果本轮模式与上轮不同：清空 optimizer state，保留 theta
    根据 4:8 schedule 选择 Agent 或 Speculator 的样本与目标
    用选定的 DAPO-style loss 更新同一个 theta
    下一轮重新从更新后的 Agent 采轨迹，不长期固定旧标签
```

**成本不能忽略**：若每批 B 个问题、每条 Agent 轨迹平均 d 个工具 turn，按伪代码每轮生成 \(8B\) 条 Agent 轨迹和约 \(64Bd\) 个短候选。200 steps 不意味着训练便宜；Algorithm 1 甚至每轮都构建两种样本，无论更新哪种模式。

**“On-policy”的准确含义**：轨迹和目标来自当前采样策略，不是外部旧模型；它不代表求导穿过离散 Agent 轨迹，也不代表一个 minibatch 更新后标签仍精确对应新参数。下一轮刷新只能缓解这种移动目标问题。

### 4.9 推理部署：原文评测与潜在实现分开

**论文实际评测**：先取得 Agent 轨迹，针对各工具边界生成候选并比较；环境只执行正式调用。早期 off-the-shelf 比较在单 H100 上用 SGLang 做显式模型切换，是串行交替的计时/内存协议。

**部署时可以怎样实现（需要另做系统验证）**：共享已完成 prefix 的 KV，Agent 继续推理，同时生成短候选；仅对安全工具预执行；待正式动作到达，完整字典匹配后复用，否则执行正式动作。正确候选未完成时仍需等待。KV 共享省的是共同前缀存储/prefill，不是免费生成 speculation 分支。

【分析】若主调用到达时间为 \(T_A\)，候选派发时刻为 \(T_P\)，工具时长为 \(T_U\)，无竞争、同一工具时长下的命中节省为：

\[
\Delta T_{\mathrm{hit}}=
T_A+T_U-\max(T_A,T_P+T_U)
=\min(T_U,T_A-T_P),\quad T_P\leq T_A.
\tag{16}
\]

所以总体净收益应检查：

\[
\mathbb E[\Delta T]
\approx\mathbb E[\mathbf1_{\mathrm{hit}}\min(T_U,(T_A-T_P)_+)]
-\mathbb E[C_{\mathrm{critical}}].
\tag{17}
\]

这里候选过晚则不部署，\(C_{\mathrm{critical}}\) 是额外关键路径延迟，不是全部 GPU 花费。式 (16)–(17) 是帮助设计后续计时实验的推导，**不是 SWYR 已给定理或实测结果**；仅提升 Hit@1 不足以确定这两个式子的收益。

## 第五部分：实验证据与复现边界

### 5.1 主结果应怎样读？

| 模型 | 阶段 | Avg. Hit@1 | Avg. task success |
|---|---|---:|---:|
| Qwen3-4B | Base self | 29.1 | 26.4 |
| 同上 | SFT | 44.1 | 26.6 |
| 同上 | 联合 RL | 61.2 | 27.7 |
| Qwen3.5-4B | Base self | 33.2 | 49.0 |
| 同上 | SFT | 48.9 | 49.2 |
| 同上 | 联合 RL | 66.3 | 50.6 |

摘要 44.1→61.2 和 48.9→66.3 的起点是 **SFT checkpoint**，不是未经训练的模型。支持结论：在这些配置下，训练后的自身调用更易被 speculator 模式准确预测，主表平均任务成功率未下降。不能推出原始 Agent 行为逐次不变、训练后必然加速或任意领域都保持能力。

### 5.2 数据和算力

| 项目 | 原文配置 |
|---|---|
| SearchQA 训练 | FlashQA；16K 长度上限；8 H100 |
| 对话工具训练 | ToolScale；32K 长度上限；8 H200 |
| RL | 每组 8/8 rollouts；学习率 3e-6；每设置 200 steps；4:8 |
| 测试 | HotpotQA、MuSiQue 各 500 subset；BCP 150；Airline 50；Retail 114 |
| SearchQA SFT | Qwen3 / Qwen3.5：Agent 1658 / 1351，Speculator 同数量 |
| ToolScale SFT | Agent 0；Speculator 5000 |

缺失/待确认：精确样本 ID、所有领域的 checkpoint 对应关系、trajectory 截断位置、采样参数、reward/parser 细则、SFT 配置、clip/动态采样配置、重复惩罚和优化器细节。依靠论文可以实现一个合理版本，但不能承诺数字复现。

### 5.3 三项重要的证据限制

1. **跨域成功率下降。** SearchQA 训练的 Qwen3-4B 在 Airline/Retail 的 Hit@1 从 35.6/39.7 升至 45.6/50.1，成功率从 21.6/36.4 降至 17.6/29.8。说明更可预测不等于更会完成任务；“保持能力”须限定训练/评测匹配条件。
2. **交替策略有混杂。** 1:1、2:2、4:4 同时改变块长度和 optimizer reset 频率；4:4→4:8 还改变两目标的更新比例。同样 200 iterations 不等于同样 token 或 FLOPs。奖励曲线显示消融有效，但未唯一证明作者解释。
3. **外部 drafter 的时间对照依赖部署方式。** 单 GPU 参数切换与串行交替不是优化过的双模型并发 serving。8.3 s vs 27.4 s 等差异不能推广为任何部署下 self 都比小 drafter 快。

另一个复现注意点：主结果表和跨域表中，一些相同 Hit@1 对应不同任务成功率；需要保留各表的 checkpoint/训练域口径，不把它们合并成一套一致配置，也不凭表格猜哪个数字正确。

## 第六部分：【第四步】问题与局限诊断

| 维度 | 判断与证据 |
|---|---|
| 机理缺位 | “训练更稳定”的证据主要是 reward 曲线，尚缺梯度相似度、动量状态和更新前后两模式 loss 的直接测量；见 §4 Ablation。 |
| 机理错位 | 将共享权重理解为消除 gap 太强：输入 suffix、可见信息和继续推理过程不同仍会造成差异。将 F1 reward 理解为 exact-match reuse 的等价目标也不成立，见式 (6) 的反例。 |
| 重构不足 | 配方能工作，但 4:8、重置频率与成功率约束之间没有明确推导；它是经验方案，而不是由已知机理唯一推出。 |
| 哲学盲区 | 未发现需要升到哲学层才能解释的证据，不强行诊断作者“视野不足”。 |
| 闭环缺失 | 附录明确只执行正式调用；缺少训练后并行 serving 的 E2E、P95、额外计算、早期不同 cutpoint 和状态一致性评估。无 agent-only 等算力 RL 对照，也难分离一般训练收益与联合训练收益。 |
| 可迁移性 | 跨域出现 Hit@1 升而 success 降；只验证两个 4B backbone；写工具需要另加隔离，不能由训练式预测自动处理。 |

还应区分两个参照：训练改变了 Agent 参数，论文没有声称与**训练前** Agent 轨迹完全相同；即使部署严格匹配，也只能讨论相对**同一训练后模型**的执行语义。将它作为训练前 Actor 的严格 lossless 替代是错误口径。

## 第七部分：【第五步】如果我来做这个研究

### 7.1 机理问题一：学会预测，还是把自己变得更容易预测？

【分析】同一权重承担两种模式，训练 speculator 也可能改变 Agent 的动作选择。当前 diagonal self Hit@1 提升不能区分两个来源。

在固定前缀 h 上，设 Agent 下一调用分布是 \(p(a\mid h)\)，Speculator 是 \(q(a\mid h)\)。若两边在给定 h 后独立采样：

\[
P_{\mathrm{match}}(h)=\sum_a p(a\mid h)q(a\mid h).
\tag{18}
\]

即使 \(q=p\)，也只有：

\[
P_{\mathrm{match}}(h)=\sum_a p(a\mid h)^2,
\qquad \max_qP_{\mathrm{match}}(h)=\max_ap(a\mid h).
\tag{19}
\]

这说明在该假设下，变得更集中可以提高自匹配，却不保证选得更好。**这是本笔记推导，不是论文证明发生了行为坍缩**；确定性解码下 p 退化时也不能直接用随机分布解释所有差距。

**最小实验**：固定同一批上下文，把 base/SFT/RL 的 Actor 与 Speculator 交叉配对；比较旧 Actor—新 Speculator 与新 Actor—旧 Speculator。再测任务成功率、动作多样性和轨迹长度，而不是只比新—新。

**baseline 与排除条件**：加入 frozen-Actor speculator-only 训练和等算力 agent-only RL；若提升主要是正常的预测改善且无能力代价，就不再追“自我变简单”的假说。改变 Actor 来提升可预测性的相关文献也需另查，不能仅凭推导主张新颖。

### 7.2 机理问题二：为什么重置优化器有用？

【分析】记两模式梯度为 \(g_{\mathrm{ag}},g_{\mathrm{sp}}\)。一次小步 speculator 更新对 Agent loss 的一阶影响为：

\[
\Delta\mathcal L_{\mathrm{ag}}
\approx-\eta\,g_{\mathrm{ag}}^\top g_{\mathrm{sp}}.
\tag{20}
\]

若内积为负，当前两个目标即使没有动量也可能冲突；若当前梯度并不冲突，但历史 optimizer 状态改变了实际更新方向，则重置可能有效。**重置历史统计不能自动消除共享参数的当前梯度冲突。**

**最小实验**：先在少量固定样本上记录梯度余弦与实际更新方向，比较共享状态不重置、切换重置、两模式分别保存 optimizer state。固定各模式更新次数、token 预算和学习率，再比较块长。

**排除条件**：简单独立 optimizer state 已足够；观测支持不了梯度/动量假说；或多任务优化已有同类结论。只有调 4:8 为其他比例，不足以构成新贡献。

### 7.3 机理问题三：提高预测分数，为什么可能没有提高等待时间收益？

训练优化参数 F1，评测看严格 Hit@1，部署关心式 (17)。这是三个不同目标；后两个之间还隔着前缀时机和工具时长。

**最小实验**：用固定 Actor 轨迹按早/中/晚前缀比较 F1、严格 Hit@1 和实际可隐藏秒数；加入原始 self-prompt、SFT-only、SPORK 式免训练 probe，并统一候选预算。小实验先证明差距，再考虑以净收益为目标的训练。

**排除条件**：简单阈值/截断点调优消除差距；没有可用权重和训练资源支持公平对照；或 AOSpec 等已完整覆盖。同样，泛泛“将延迟加入 reward”不是新颖性结论。

### 7.4 怎样提高研究层次，而不强行提高哲学层次？

先把“更容易预测”拆成可测量来源，把“更准”与“更早且划算”分开，把优化器经验配方变成可证伪解释。若形成跨模型、跨任务可重复的规律，才能讨论更强的机理贡献；现阶段不需要加入认知或自我意识叙事。

对目前从零实现能力而言，优先级是**固定日志上的交叉评测与奖励审查 → 少量梯度诊断 → 有证据后再做联合 RL**。论文的 8 H100/H200 训练不是现在必须复制的起点，也不能未经实验承诺小算力复现同样结果。

## 第八部分：总体评价与文献关系更新

**这是一篇以技术层训练流程为主要贡献的论文。** 核心价值是展示共享模型可以通过自身轨迹监督与交替训练，提高全参数调用匹配，同时在主设置中维持任务能力；核心局限是收益来源与训练机理未被充分分离，且调用匹配到真实 E2E 加速的系统闭环尚未完成，跨域保持能力也不成立。

相对 SPORK，它推进的是**训练后的自身行为对齐**，不是首次 self-speculation，也没有实现 SPORK 的 D3 token recovery。它已部分覆盖第四篇留下的“只预测工具名不够，需要改善参数一致性”问题；剩余研究应更具体地验证奖励、采样前缀和决策收益之间的差距。

下一篇 **Speculate with Memory**：检查它复用的是调用、轨迹还是预测经验；是否能在不改 Actor 参数的情况下改善一致性；与训练式自投机相比，如何衡量新任务、状态变化和历史失效。

## 原文定位

- [方法与奖励公式](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculate While You Reason- Teaching Agents to Predict Their Next Tool Call via Joint Agent-Speculator RL/sections/03_method_rewrite.tex>)
- [训练 Algorithm 1](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculate While You Reason- Teaching Agents to Predict Their Next Tool Call via Joint Agent-Speculator RL/tables/training_recipe_algorithm.tex>)
- [外部与自身预测评测协议](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculate While You Reason- Teaching Agents to Predict Their Next Tool Call via Joint Agent-Speculator RL/sections/92_empiricaloffshelf.tex>)
- [主实验与消融](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculate While You Reason- Teaching Agents to Predict Their Next Tool Call via Joint Agent-Speculator RL/sections/04_experiment.tex>)
- [附录：真实执行范围、配置与 SFT 例外](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/Speculate While You Reason- Teaching Agents to Predict Their Next Tool Call via Joint Agent-Speculator RL/sections/appendix.tex>)
