# 04 · SPORK：自投机、置信度门控与失败草稿复用

阅读日期：2026-09-09。论文：**SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference**。

已读本地全部正文和附录，核对关键消融与 CoT 曲线。[官方版本](https://arxiv.org/abs/2607.03333)为 v1，2026-07-03。本文记录研究判断，不逐段翻译；候选问题尚未经新颖性确认。本轮未检查或运行项目实现。

## 1. 对后续选题的直接影响

**SPORK 已把动作级投机和 token-level speculative decoding 接起来：预测完整调用命中就复用工具结果，预测失配仍可将 probe token 作为草稿交给主模型验证。**

以下不能再单独作为创新：

- 用正在运行的同一个模型预测自己的下一工具调用，无需训练额外 drafter。
- 共享 prefix KV cache 降低 probe 开销。
- 随 reasoning 增长重复 probe，通过置信度控制提前执行。
- 对失配 probe 做 token 前缀验证和复用。
- 后续更好的 probe 替换已经启动的投机：附录已测试 supersession 开关。

**最值得继续核查的差距**：门控主要预测“工具名是否一致”，实际复用却要求“工具名和全部参数一致”；提高门控指标是否真正提高有用重叠，尚需更针对性的分析。

## 2. 三个机制与时序

### D1：从同一模型 fork 一个短 probe

主请求正常生成 reasoning。收到第一个 streaming token 后，控制器用相同前缀和已观察的 CoT，加上强制工具调用前缀，启动一个短 completion 请求，让模型先输出工具调用。

这不是中断主 reasoning，也不是读取隐藏状态中的一个现成动作；仍然要运行一次短生成。等待首 token 是为了让主请求先建立可复用的 prefix cache，避免两个请求同时重复 prefill。

作者在一个 32B、15K 上下文设置中报告 probe 从约 1.6 s 降至 0.35 s；这是特定配置结果，不能当作所有后端都保证的开销。

### D2：置信度控制何时投机

probe 产生合法调用后，以**工具名 token span 的最低 top-1 概率**作为置信度，论文取阈值 0.90。不够自信时，随着主请求继续生成，每隔一段 CoT token 再 probe，受重试预算限制；默认第一个过阈值的合法调用触发工具，随后停止重试。

要分开两个 gate：

| 阶段 | 判断 | 能保证什么 |
|---|---|---|
| 提前执行前 | 工具名置信度是否过阈值 | 筛选可能有用的候选，不保证全部参数匹配 |
| 主请求完成调用后 | 工具名和序列化参数是否严格一致 | 决定能否复用投机工具结果 |

论文使用的“commit probe”有时指**启动投机工具**，不等于已经向正式对话提交 observation。

### D3：失配的 probe 仍可做 token 草稿

假设 probe 是 `search(query="A B")`，主请求最终要的是 `search(query="A C")`。两者不能复用同一工具结果，但 token 序列可能有公共前缀。

主模型到达工具调用起始标记时，引擎把 probe 的 body 作为 draft，用主模型验证并接受最长匹配前缀，再从失配处继续生成。**必须在主调用尚未全部生成时介入；事后计算两段文本的公共前缀不能节省已经发生的 decode。**

GAIA 的离线失配样本共享 token 前缀中位数为 18；BrowseComp 的另一组分析中位数为 27，不能混成同一数据集。

### 一轮如何结束

1. 主 reasoning 与 probe 并发；合法、自信且允许投机的工具提前执行。
2. 引擎模式可在主调用生成过程中使用 D3；HTTP 模式没有 D3。
3. 主调用完整到达后严格匹配。命中就复用结果，工具未完成时仍须等待；未命中就正常执行主调用。
4. 只有正式确认的结果进入对话历史。

附录另测“启动后继续 probe，取消旧候选并换新候选”。30 个 tau2 任务上有小幅改善，作者将其保留为可选项，没有作为主要机制。

## 3. 实验中真正有启发的现象

### 工具名容易预测，不代表动作参数已经确定

Qwen3-32B 在五个 benchmark 上的早期工具名准确率为 74.6%–99.6%，参数精确匹配只有 7.6%–83.9%。GAIA 的 CoT 回放实验更直观：

| probe 看到的 CoT 比例 | 参数精确匹配率 |
|---|---:|
| 0% | 7.6% |
| 40% | 39.0% |
| 80% | 61.0% |
| 100% | 97.5% |

**判断**：参数逐渐稳定，但越晚 probe，留给工具的重叠窗口越小。不能等价为“模型从一开始就知道完整动作”。而且这些百分比使用完整轨迹回放，在线控制器事先不知道最终 CoT 总长度。

工具名结果还需结合工具集合大小：论文配置中 HotpotQA 只有 1 个工具、GAIA 有 2 个。因此 99% 工具名准确率不能单独证明复杂动作预测能力。

### Confidence 的标签与最终收益不一致

0.90 阈值在 GAIA probe 数据上得到 precision 88%、recall 100%、F1 0.937，但这里的“正确”按**工具名匹配**定义。它不是完整调用匹配的校准概率，也不表示提前执行有 90% 的安全保证。

D2 的 probe 总数在一组 GAIA 对比中从 434 增至 2124。筛掉工具调用并不意味着少做 LLM 工作：必须分别计量 probe 次数、工具派发次数、最终严格命中和被隐藏的时间。

### 失败草稿复用不是总能加速

GAIA 消融图中，叠加在 ngram baseline 上的各配置 P95 约为：

| 配置 | P95，约 s |
|---|---:|
| Ngram baseline | 132 |
| +D1 | 113 |
| +D1+D3 | 130 |
| +D1+D2 | 126 |
| +D1+D2+D3 | 108 |

完整配置尾延迟最好，但单加 D3 或 D2 未必胜过 D1。作者也报告 D1 的 mean 表现更好。**因此不能从“有公共 token 前缀”直接推导 D3 的端到端净收益，更不能声称所有指标单调改善。** 需要考虑验证成本、重试策略及真实工具噪声。

## 4. Baseline、结果和复现口径

| 项目 | 需要保留的信息 |
|---|---|
| 主设置 | Qwen3-32B、Qwen3-4B、Qwen3.5-35B-A3B；主实验 vLLM 0.19.1，H20-3e，bf16，TP=1 |
| 解码 | temperature 0、seed 42、thinking mode；不是任意随机采样条件的验证 |
| 真实工具 | GAIA 搜索/浏览，HotpotQA Wikipedia API |
| tau2 | 确定性 canned DB 结果，加 0.5–5 s 延迟下限；主要是时序实验，不是完整业务交互能力评测 |
| 32B GAIA | N=165，P95 131.9→108.1 s，约减少 18%；对照已启用 ngram speculative decoding |
| 32B HotpotQA | N=200，P95 约 1.06×；EM +1.5 个百分点，不据此推断投机提高推理能力 |
| 4B | GAIA/HotpotQA 接近中性；tau2 的 1.15× 是 mean 口径，不是其 P95 比率 |
| 模型差异 | 32B/4B 引擎模式对照 ngram；MoE HTTP 模式对照串行，使用 D1+D2，GAIA 例外仅 D1；4B tau2 也走 HTTP |
| 其他数据 | BrowseComp 附录使用不同版本和 TP=4；不可假设全篇实验完全同配置 |

D1/D2 需要 streaming、raw completion、logprobs 和跨请求 prefix-cache 支持；D3 需要接入引擎的 speculative proposer。**“HTTP 控制器容易接入”不等于全部 D1+D2+D3 都能无修改地经 API 使用。**

本地项目：[spork](/mnt/data/zhangshuai/LLMs/Projects/spork)。代码阶段先确认 release 实际支持的模式、工具后端与 benchmark harness，再决定复现范围；已有下载不等于完整论文结果已可复现。

## 5. 正确性与适用边界

- 实现仅投机 manifest 中标记为只读的工具，写入和非幂等工具走串行路径。**SPORK 当前没有实现通用 rollback。**
- 同名同参数的只读工具也可能随时间返回不同结果。作者明确承认真实搜索 API 使轨迹不完全一致，使用 EM/F1 比较；这是质量近似保持的实验证据，不是外部状态逐次等价证明。
- D3 在本文 greedy 设置中逐 token 验证主模型输出。不能仅凭“使用 speculative decoding”就将当前具体实现推广为任意采样分布保证。
- no-think 模式窗口过短，tau2 报告 0.79×，即出现减速。4B 的长 CoT token 数也未必提供足够长的真实时间窗口。
- MoE 原生 XML 格式的 probe 可解析率达 97%，工具名匹配却仅 4.5%，往往预测最终目标工具而非下一中间工具。JSON 提示缓解此问题；说明这是预测语义问题，不只是 parser 出错。
- 作者将目标部署限定为有空闲容量的低延迟 serving；重 batching 下 probe 与主请求竞争，不能用单个设置的低 TPOT 开销保证所有负载。

### 成本模型只保留一个实用判断

用统一平均口径记录：

`净节省 ≈ 严格命中带来的实际重叠 + D3 净节省 − probe/验证/竞争造成的额外延迟`

正文与附录对 D3 和 overhead 使用了不同摊销口径，复现时应从日志分别统计命中、失配与每轮平均，不直接混填公式；平均每轮模型也不能直接预测 P95。

**一个相关的时序修正**：附录将 `工具时长 < probe 时长` 的部分描述为零重叠，这不是一般条件。若主 decode 要 10 s，probe 在 1 s 完成，即使工具只要 0.2 s，仍可提前完成并隐藏这 0.2 s；是否净赚还取决于开销。真正窗口取决于 probe 完成时距主调用还有多久。这个修正用于正确计时，不作为独立创新。

## 6. 与前作的关系和已覆盖 idea

| 对照 | 关键差别 |
|---|---|
| Speculative Actions | SPORK 固定使用同一运行模型做短 probe，具体利用 CoT 和 prefix-cache；不是一般外部预测器框架 |
| DSP | DSP 控制规划链深度；SPORK 控制下一调用的探测/执行时机，并复用调用 token |
| PASTE | PASTE 从历史模式绑定当前参数并联合调度；SPORK 无需历史轨迹，利用同模型早期意图，但承认重负载竞争边界 |
| 小 drafter | 论文已做相同 32B 主模型、self vs cross-4B 的对照；小 probe 更快但参数一致性更低，不可再以“换小模型”作为充分创新 |

“动作级 + token-level 投机”“置信度动态 probe”“失败前缀复用”“后续候选替换”均已覆盖。memory 与 self-probe 的组合是否有价值，也必须先证明互补性，而非直接拼接两篇方法。

## 7. 两个候选问题：先诊断，再决定是否做方法

### S1：工具名置信度能否有效筛选完整调用收益？

**证据**：高工具名准确率与低参数匹配并存；阈值按 name F1 选择，正式复用按全调用匹配。

**最小实验**：固定一批主轨迹和候选 probe，分别按工具名、全参数匹配标注；比较原 name min-prob、参数 span 的简单置信度、固定延迟/固定次数基线。在**相同 probe 和工具预算**下比较严格匹配、实际隐藏时间、无效工作，不先训练大模型。

**放弃条件**：name 信号已足够；简单阈值调参即可解决；参数概率受长度/格式影响而无法稳定泛化；或下一篇训练式自投机已完整覆盖。仅“把 name gate 改成 args gate”不够形成论文贡献。

**难度**：日志分析低，接 HTTP controller 做在线对照中。优先接着读 Speculate While You Reason，再判断训练是否必要。

### S2：何时值得再 probe，而不是等待或停止？

**证据**：D2 重试增加推理量，first-confident 默认提前停止；supersession 已有小实验。D1 的 mean 好、完整系统的 P95 好，说明控制目标有差别。

**最小实验**：明确只选一个优化目标，固定底层模型，对比原 D2、固定间隔、指数退避和简单预算上限；记录每次 probe 的增量命中收益、剩余 decode 时间、工具时长与主请求干扰。先回放找规律，再做低并发在线验证。

**放弃条件**：原策略或简单退避已足够；收益只来自多花算力；或测量噪声大于改进。通用“动态选择投机时机”已被 SPORK 覆盖，只有稳定、可解释的剩余决策缺口才值得继续。

**难度**：中；先不做多 GPU 全局调度，也不急于接 D3。该问题并入台账 R1/R3，而非再扩张一个大方向。

## 8. 后续文献线索与下一篇

SPORK 的 related work 引用 **Atomix: Timely, Transactional Tool Use for Reliable Agentic Workflows**（arXiv:2602.14849），将其描述为投机副作用的事务语义工作。此处仅从本地参考文献记录线索，尚未精读；研究 rollback/commit 前必须核查，不能默认空白。

下一篇 **Speculate While You Reason**：它训练什么、是否直接改善全参数一致性、能否减少 SPORK 的重复 probe，以及相同预算下比免训练 self-probe 多获得什么。

## 原文定位

- [自投机现象与准确率](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/SPORK- Self-Speculative Forking to Accelerate Agentic LLM Inference/sections/3_self_speculation.tex>)
- [D1–D3 设计](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/SPORK- Self-Speculative Forking to Accelerate Agentic LLM Inference/sections/4_design.tex>)
- [控制器伪代码与只读限制](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/SPORK- Self-Speculative Forking to Accelerate Agentic LLM Inference/sections/5_implementation.tex>)
- [实验与消融](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/SPORK- Self-Speculative Forking to Accelerate Agentic LLM Inference/sections/6_evaluation.tex>)
- [附录：配置、成本模型和 supersession](</mnt/data/zhangshuai/LLMs/paper/Agent系列latex论文/Agent动作投机/SPORK- Self-Speculative Forking to Accelerate Agentic LLM Inference/sections/A_appendix.tex>)
