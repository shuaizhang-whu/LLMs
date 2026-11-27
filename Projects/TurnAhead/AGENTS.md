# 实验执行约定

- 本目录 `Projects/TurnAhead` 是 TurnAhead 论文的主项目目录。E2b/E5 实验链于 2026-11-25 完成，目录于 2026-11-26 从 `Projects/spork-codex` 改名。
- 同级 `spork-ds` 即原 `spork-claude`（2026-11-22 已改名），是另一条实验线；本项目不使用其代码和结果。
- 所有 GPU 实验及其他耗时实验由用户在 tmux 中启动；助手负责准备命令、编辑代码、只读分析和轻量 CPU 检查。
- vLLM 在 tmux 的 `vllm` 会话启动，评测在 `train` 会话启动。
- 新的实验使用本目录的代码、τ² 副本与任务划分；模型权重和 conda 环境共用。
- 使用 `scripts/codex_env.sh` 设置路径。全量评测入口为 `scripts/run_codex_full_eval.sh`，默认仅准备，`--run` 才执行。
- 历史结果、冻结源码、manifest 和 seal 保持原样，不批量替换其中的旧路径。迁移证据位于 `research_runs/migration_20261025`。
- GPU 互斥继续兼容现有按端口的跨项目文件锁；锁不代表可以编辑其他项目代码。
