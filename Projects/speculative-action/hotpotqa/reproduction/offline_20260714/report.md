# HotPotQA 无 API 离线 baseline 复现报告

## 结论

仓库附带轨迹的无 API 后处理流程已实际跑通：指标汇总、Top-1/Top-3 列表重算和两类官方 README 绘图命令均以退出码 0 完成，生成 3 个聚合数据文件和 6 张可解码 PNG。这个结论仅适用于离线后处理，不代表完整 Speculative Actions 系统、在线并发或论文的 lossless 结论已经复现。

## 修改与根因

- 原 `run.py` 在解释 `--norun` 前无条件构造 `HotPotQARun`，继而 eager 初始化三类模型客户端，并因缺少 `constants.gemini_api_key` 抛出 `AttributeError`。
- 现在只在真实实验分支延迟导入并构造 Runner；指标函数使用与 Runner 一致的纯路径规则。CLI 参数、固定模型矩阵、指标公式和输出文件名均未改变。
- `grapher.py` 只增加保存前创建 `run_metrics/images/` 的处理，没有修改图表计算与布局。

## 验证结果

| 检查项 | 实际结果 |
|---|---:|
| unittest | 11/11 通过 |
| 固定模型组合 | 15 |
| 有附带 Top-3 数据的组合 | 12 |
| 按原行为跳过的组合 | 3 |
| 完整 Top-3 样本 | 48（每组 4 个） |
| 重算与保存 `metrics.json` 不一致 | 0 |
| 逐样本 Top-1 > Top-3 | 0 |
| 聚合 JSON / CSV 记录 | 12 / 12 |
| 可解码 PNG | 6/6 |
| strace 网络系统调用 | 0 |
| 原始轨迹哈希变化 | 0/376 |

缺失的三组均为 GPT-5 Actor 与 `gpt-5-nano`、`gpt-3.5-turbo`、`gemini-2.5-flash` Speculator 的组合。

## General 动作命中率

以下 Top-1 和 Top-3 均由附带的 `agent_*_top3` 轨迹重算；Top-1 取每步候选列表首项。

| Actor | Speculator | Top-1 | Top-3 | 样本数 |
|---|---|---:|---:|---:|
| Gemini 2.5 Flash | GPT-5 Nano | 23.33% | 48.33% | 4 |
| Gemini 2.5 Flash | GPT-4.1 Nano | 43.75% | 81.25% | 4 |
| Gemini 2.5 Flash | GPT-3.5 Turbo | 18.75% | 68.75% | 4 |
| Gemini 2.5 Flash | Gemini 2.5 Flash | 31.25% | 81.25% | 4 |
| Gemini 2.5 Flash | Gemini 2.5 Flash Lite | 45.00% | 70.00% | 4 |
| GPT-4 | GPT-5 Nano | 8.33% | 20.83% | 4 |
| GPT-4 | GPT-4.1 Nano | 62.50% | 75.00% | 4 |
| GPT-4 | GPT-3.5 Turbo | 25.00% | 64.58% | 4 |
| GPT-4 | Gemini 2.5 Flash | 29.17% | 50.00% | 4 |
| GPT-4 | Gemini 2.5 Flash Lite | 25.00% | 54.17% | 4 |
| GPT-5 | GPT-4.1 Nano | 25.00% | 87.50% | 4 |
| GPT-5 | Gemini 2.5 Flash Lite | 25.00% | 75.00% | 4 |

## 已确认的 baseline 限制

- HotPotQA 代码按顺序调用 Actor、候选生成、真实环境和模拟环境，没有在线异步流水线；`--graph` 使用的只是轨迹中记录的 Search step 时间，不是端到端 wall-clock latency。
- `Metrics.get_action_metrics(..., k=3)` 不会显式截断前三项，而是检查保存列表中的全部候选。183 个候选列表的长度分布为：1 项 16 次、2 项 1 次、3 项 165 次、4 项 1 次；因此这里应称“仓库 Top-3 口径”，不能泛化为严格 Top-3。
- Top-1 列表由 Top-3 轨迹首候选派生，并未使用仓库单独的 `agent_*_top1` 目录。
- 每个有效模型组合只有 4 个样本，不能代表论文完整实验规模。
- Search 时间图保留了官方代码两组柱使用同一横坐标的行为。两张含 5 个 Speculator 的准确率图还存在横轴标签重叠/越界，Gemini Actor 图的图例会遮挡部分数值标签；本阶段只记录、不改图表呈现。
- `--graph2` 依赖仓库中不存在且没有生成入口的 `comparision_top1_top3.json`，因此未执行。

## 证据与后续

- 原始日志、命令、环境、网络跟踪和逐文件哈希均保存在本目录。
- 聚合结果位于 `run_metrics/`，图片位于 `run_metrics/images/`。
- 下一阶段才需要创建独立 `agent-action-vllm` 环境、选定 Instruct 模型并运行 1～2 个样本；本阶段未创建任何其他环境。
