# Speculative Actions 官方代码中文导读

> 审计基线：`dc938b9ef7474caf07fe4ad16549c1fa8c7d268c`  
> 本文描述仓库当前代码的真实行为，不以论文描述替代代码核验。

## 1. 仓库全景

```text
speculative-action/
├── chess-game/      # TextArena 国际象棋；含顺序与在线投机 runner
├── e-commerce/      # τ-bench retail/airline；论文脚本主要重放历史轨迹
├── hotpotqa/        # ReAct + Wikipedia；当前优先复现对象
└── os-tuning/       # sysbench + Linux 调度参数；有系统级副作用的 lossy 扩展
```

四个目录是相对独立的实验项目，依赖、入口、指标口径和“投机执行”的实现程度都不同，
不能把其中一个环境的结论直接套到另一个环境。

| 环境 | 当前代码中的主要入口 | 实际并发语义 | 结果性质 |
| --- | --- | --- | --- |
| HotPotQA | `hotpotqa/run.py` | Actor、Wikipedia、Guess LLM 串行 | 轨迹采集后离线算 action match |
| Chess | `speculative_workflow/Speculative_Chess.py` | 当前 Actor 与投机任务并发，Top-K 分支并行 | 在线复用分支，但保存时间是组合估计 |
| E-commerce | `tau-bench/exp_static.sh` | task 之间并发，单 task 静态预测串行 | 固定 Actor 轨迹上的离线预测分析 |
| OS Tuning | `barebones_optimizer.main` | benchmark、Actor、Speculator 在线并发 | 直接覆盖系统参数的 lossy 实验 |

## 2. HotPotQA：当前优先复现对象

### 2.1 文件职责

```text
hotpotqa/
├── run.py                 # CLI：运行、指标汇总、绘图
├── requirements.txt       # 未锁版本的最小依赖列表
├── data/                  # HotPotQA train/dev/test 简化数据
├── prompts/               # ReAct few-shot 示例
├── trajs/                 # LoggingWrapper 的原始混合轨迹
├── run_metrics/           # normal/sim 分离轨迹和逐样本指标
└── src/
    ├── constants.py       # 样本数、步数、Top-K、模型生成参数
    ├── prompts.py         # ReAct/action/observation guess 模板
    ├── llm_client.py      # Gemini/OpenAI/OpenRouter 同步客户端
    ├── environment.py     # WikiEnv：Search/Lookup/Finish
    ├── wrappers.py        # 数据集、奖励、日志和历史包装器
    ├── runner.py          # HotPotQARun 主循环
    ├── metrics.py         # action exact-match 与 Search 时间
    ├── grapher.py         # 聚合结果绘图
    └── utils.py           # JSON、文本、目录清理和 CSV 转换
```

### 2.2 对象装配

```text
run.py
└── HotPotQARun
    ├── self.llm: Actor LLM
    └── HistoryWrapper
        └── LoggingWrapper
            └── HotPotQAWrapper
                └── WikiEnv
                    └── guess_llm: observation Guess LLM
```

- `self.llm` 同时生成真实历史上的 Top-1 action 和模拟历史上的 Top-K action。
- `guess_llm` 不直接预测 action；它根据 `Search[entity]` 生成拟 Wikipedia 页面。
- `WikiEnv` 支持 `search[...]`、`lookup[...]`、`finish[...]`，代码还接受 `think[...]`。

### 2.3 单步真实顺序

当 `simulate=True` 时，每轮代码按以下顺序同步执行：

```text
1. Actor(real prompt) -> authoritative action
2. Actor(sim prompt)  -> Top-K candidate actions
3. WikiEnv(wiki)      -> 真实 Wikipedia observation
4. WikiEnv(simulate)  -> Guess LLM 生成模拟 observation
5. 分别追加 normalobs / simobs
```

代码没有 `asyncio`、线程池、future 或候选工具分支预执行。因此 HotPotQA 当前实现是
“串行采集两套轨迹，再分析潜在命中”，不是端到端在线并发流水线。

### 2.4 两种轨迹

`normalobs.json` 与 `simobs.json` 都包含：

```text
prompt
observations[]
thoughts[]
actions[]
time_taken[]
```

- normal `actions[i]` 是一个字符串。
- sim `actions[i]` 是候选字符串列表。
- normal time 只包围真实 `env.step()`。
- sim time 只包围模拟 `env.step()`；Search 时主要是 Guess LLM 延迟。
- 两次 Actor 推理时间都没有写入这两个时间列表。

### 2.5 指标口径

- `--getmetric` 读取已有逐样本 `metrics.json` 后做模型组合宏平均。
- `--getmetric2` 从 normal/sim actions 重新计算 Top-1 与任一候选命中列表。
- 第一个 action pair 被跳过，因为第一步之前没有模拟 observation。
- 默认是忽略大小写后的完整 action 字符串相等，不是语义匹配。
- `get_avg_time_taken` 只统计 Search action。
- 现有图比较真实 Search 与模拟 Search 的平均耗时，不是 E2E speedup。

### 2.6 复现前必须记录的代码风险

1. `HistoryWrapper.reset()` 没有向下传递 `idx`，目录样本号可能不对应实际问题。
2. 真实和模拟调用共享一个 `WikiEnv`；模拟页面可能覆盖真实 `page`。
3. Lookup 分支不区分 `step_type`，共享游标可能被真实/模拟调用共同推进。
4. `extract_actions` 按 Search、Lookup、Finish 三类分别收集，会改变模型原始候选顺序。
5. 纯指标 CLI 仍先初始化 LLM 客户端，不能视为完全独立的离线入口。
6. 当前客户端只返回文本，没有保存 token usage 或货币成本。

这些是 baseline 审计项。复现附带结果和修正实现应分开记录，不能混成同一组结果。

## 3. Chess

### 3.1 主链

```text
TextArena Chess-v0
├── regular_workflow/regular_chess.py
│   └── 当前玩家 Actor -> 环境 step -> 下一玩家 Actor
└── speculative_workflow/Speculative_Chess.py
    ├── 当前玩家 Actor
    └── speculation task
        ├── Speculator 猜 Top-K 当前动作
        └── K 个复制棋盘并行生成对手下一动作
```

投机版下一轮会检查真实动作是否出现在上一轮候选中；命中时复用对应的预生成对手响应。
这一复用是在线发生的，但程序会等待整个投机任务完成，保存的时间字段却只选择部分分项
组合，因此不能直接当作进程级 wall-clock。

### 3.2 结果分析链

```text
stepsinfo.json
→ guess_over_regular_traj.py / guess_over_spec_traj.py
→ confidence-prediction.py（可选）
→ guess_analysis.py
→ plot.py
```

需要留意顺序/投机 runner 的动作解析不一致、候选合法性、失败分支索引对齐，及 confidence
文件名不包含 Top-K 导致的跨实验污染。`textarena/` 内大部分文件是无关游戏环境，不属于
本 baseline 的主要注释范围。

## 4. E-commerce

### 4.1 论文静态实验链

```text
exp_static.sh
→ 顶层 run.py
→ tau_bench/run.py
→ ToolCallingStaticAgent
→ historical_trajectories 中的固定 GPT-4o trial-0 轨迹
→ Guesser 猜下一条用户消息
→ Decider 预测最多 5 步工具 action
→ 本地工具执行预测分支
→ analysis_static_combine.ipynb 离线匹配
```

静态 agent 中原本的线程提交被注释，当前 Guesser、Decider、工具执行严格串行。
`max_concurrency` 并行的是不同 benchmark task。最终 reward/info 直接来自固定历史轨迹，
预测分支不参与最终任务奖励，因此附带结果本身不能证明在线 lossless。

### 4.2 动态路径

`ToolCallingReduceAgent` 会在等待真实用户响应时提交后台预测，是更接近在线延迟隐藏的路径，
但它不是 `exp_static.sh` 的论文样例结果路径，并且前后台共享环境状态，需要单独审计。

### 4.3 本地模型注意点

主实验 agent 使用 LiteLLM 调用；`model_utils/model/vllm_chat.py` 没有接入该主链。
因此存在 vLLM 文件不代表静态实验已经支持本地 OpenAI-compatible 服务。

## 5. OS Tuning

### 5.1 主链

```text
JSON config
→ ReplicationConfig
→ SchedulerParameterManager
→ SysbenchCpuBenchmark
→ Gemini Actor / Speculator futures
→ ReplicationOptimizer
→ history.json / reaction replay / plots
```

benchmark 在后台线程运行，主线程轮询 Actor/Speculator future，并可能在一个 sysbench window
尚未结束时直接写入 `/sys/kernel/debug/sched/*`。Dual 模式中两类 LLM 都可修改 live 参数，
后返回者覆盖先返回者，没有 lossless 的命中验证。

### 5.2 为什么当前不优先运行

- 需要 `sysbench`、debugfs、特定 scheduler knob 和无交互 sudo。
- 会改变整机 Linux 调度行为，GPU 独占不等于允许修改整机 scheduler。
- 强杀、断连或首次部分写入失败可能绕过恢复路径。
- 当前若干 shell 脚本把 monorepo 根目录误当作 `os-tuning` 根目录。

因此本阶段只保留代码审计，不执行 setup 或 live tuning。

## 6. 推荐复现顺序

1. 在 Python 3.11 隔离环境安装并锁定 HotPotQA 依赖。
2. 不调用 API地核验附带 normal/sim 轨迹、Top-1/Top-3 与绘图输入。
3. 记录原始 CLI 的初始化失败，再做纯指标入口的最小修复。
4. 接入本地 OpenAI-compatible vLLM，先跑 1 个样本和少量步骤。
5. 分开记录顺序 baseline、官方串行模拟路径以及后续真正在线并发实现。
6. HotPotQA 稳定后再转向 Chess；E-commerce 最后处理，OS Tuning 暂缓。

## 7. 推荐的 Python 环境边界

不建议把四个子项目和 vLLM 服务端全部安装到同一个环境。仓库中的依赖声明彼此独立，
Chess 还在 `pyproject.toml` 中明确要求 Python 3.12 及以上。

| 建议环境 | Python | 用途 |
| --- | --- | --- |
| `agent-action` | 3.11 | 当前 HotPotQA runner、指标与绘图 |
| `agent-action-vllm` | 以当前 vLLM 官方安装要求为准 | 独立启动 OpenAI-compatible 推理服务 |
| `agent-action-chess` | 3.12 | Chess 与最小 TextArena 调用链 |
| `agent-action-ecom` | 3.11 | τ-bench、LiteLLM 和多 provider 依赖 |
| `agent-action-os` | 3.11 | 仅在获准运行 OS Tuning 后创建 |

当前阶段只需要完成 `agent-action`。把 vLLM 服务端隔离的原因不是业务代码冲突，而是
vLLM、PyTorch、CUDA wheel 和 NCCL 对版本组合更敏感；客户端环境只需安装 OpenAI SDK，
通过 HTTP 请求本地服务即可。每个环境安装完成后都应保存 `python --version`、
`pip freeze` 与 `pip check` 输出，作为实验元数据。
