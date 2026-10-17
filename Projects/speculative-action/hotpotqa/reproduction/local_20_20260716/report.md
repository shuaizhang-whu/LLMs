# HotPotQA 本地 Qwen 20 样本复现报告

## 结论

固定 20 个 HotPotQA dev 样本已完成 sequential、paired Top-1、paired Top-3
三组完整 8-step 实验。每个输出目录都包含逐样本原始轨迹、实际问题编号、任务
结果、严格动作指标、token/调用统计、环境和样本 wall-clock、权威动作哈希。

本阶段复现的是本地模型下的 baseline 工程链路和动作可预测性，不等于论文
Gemini/GPT 模型主图，也不声称 paired 串行采集实现了在线加速。

## 本地 20 样本结果

| 模式 | EM | F1 | 主动完成率 | 平均墙钟/样本 | 总 tokens |
|---|---:|---:|---:|---:|---:|
| sequential | 20.00% | 0.2900 | 55.00% | 11.89s | 232,018 |
| paired Top-1 | 20.00% | 0.2900 | 60.00% | 27.62s | 499,815 |
| paired Top-3 | 15.00% | 0.2067 | 50.00% | 28.20s | 481,528 |

paired 运行是独立的实时模型/实时 Wikipedia 轨迹，不能把三行 EM/F1 的差异解释为
投机执行改变了语义。严格 lossless 比较使用另行保存的 sequential 权威轨迹重放。

### 论文兼容动作口径（跳过首步）

| 运行 | Top-1 宏平均 | Top-1 微平均 | Top-3 宏平均 | Top-3 微平均 |
|---|---:|---:|---:|---:|
| paired Top-1 | 19.17% | 14.61% | — | — |
| paired Top-3 | 22.92% | 18.82% | 25.42% | 22.35% |

paired Top-3 的分类宏平均：Search 27.31%、Lookup 16.67%、Finish 20.00%。
新指标严格截断前三项，且完整动作的动作类型与参数必须一致。

## 在线 smoke 结论

样本 7107、5619 已运行保守 online Top-3：

- 权威 Actor 动作来自 sequential 轨迹重放。
- 只为严格匹配的 Search 候选预取真实 Wikipedia。
- 最终答案、EM/F1、样本编号和权威动作序列均通过 lossless 对照。
- 本地 Qwen 预测 observation 并生成下一动作的速度慢于 Wikipedia 返回；8 个
  Search 机会中候选均未在下一步决策前就绪，实际 hit rate 和隐藏等待时间为 0。

这说明在线链路正确运行，但 Qwen2.5-7B 作为该配置的本地 speculator 没有产生加速。

## 付费论文主图预算

依据本地 20 样本的实际调用和 token 统计，完整矩阵基础估计为：

- 1,749 次请求；加 20% 重试预留约 2,100 次。
- Gemini Actor/候选生成：约 2,799,696 input、72,888 output tokens。
- 每个 observation speculator：约 9,343 input、14,472 output tokens。
- 按 2026-07-16 官方标准价格估算约 1.08 USD；20% 预留约 1.29 USD。
- 考虑 Gemini thinking token 与跨 tokenizer 偏差，建议完整批次审批硬上限 3 USD。

首个两样本、GPT-4.1-nano speculator、k=1/3 试跑按本地日志约 63 次请求，估计
0.038 USD；建议审批硬上限 0.15 USD。当前官方目录已把 GPT-4.1-nano 标为
deprecated，因此配置 Key 后先做模型可用性检查；不可用时停止，不替换模型。

## 限制

- Wikipedia 是实时网页服务，同一问题在不同运行时 observation 可能变化。
- 本地模型的 token 数只能用于费用量级估计，不能替代 Gemini/OpenAI 实际账单。
- paired 的环境与样本耗时是串行采集成本，不是端到端 speculative speedup。
- 论文 API 主图尚未执行；没有 API Key 也没有产生任何付费调用。

价格核验来源：

- https://ai.google.dev/gemini-api/docs/pricing
- https://developers.openai.com/api/docs/models/gpt-5-nano
- https://developers.openai.com/api/docs/models/gpt-4.1-nano
