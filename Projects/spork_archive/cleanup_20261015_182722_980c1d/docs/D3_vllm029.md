# D3 最小验证（vLLM 0.29 / Qwen3-32B）

本次目标：确认探针草稿进入 vLLM 的 draft proposal 和验证，并读到其实际接受数。不是全量复现或性能比较。GPU 操作全部由用户在 tmux 执行。

已修正：

- 旧 launcher 使用的 API 入口和参数解析函数在 0.29 已迁移；现在包装实际使用的 `entry.build_app`。
- 用 `--worker-extension-cls` 在各 worker 注册 RPC，不依赖父进程 monkey-patch 经过 fork 继承。
- 0.29 的 ngram proposer 首参数是草稿预算，已适配其签名；保留原 ngram proposer 与 rejection sampler。
- 草稿绑定唯一主请求 ID，兼容 vLLM 添加的请求 ID 前后缀，不再固定选择批次第 0 行。
- 用完整已生成工具正文定位草稿剩余部分；正文不匹配时回到 ngram，不在错误偏移处注入。
- 计数区分成功注册、提出草稿、调度并验证、接受 token；接受数取同步验证返回的 token 数减去 bonus/correction，并核对调度草稿和接受前缀。

本适配仅支持 0.29.x、Qwen3 completions、V1 runner、同步 scheduling。只运行一个评测客户端，主请求与自身探针可以并发；清理接口是服务级的。旧 `spork_proposer.py` 保留，但此 launcher 使用新的 `spork_core/vllm029_d3.py`。

已通过：9 项 CPU 单元测试、Python/shell 语法检查、launcher `--help` 导入检查。尚未加载 GPU 模型或实测 RPC / D3 链路；GPU 上的验证由以下命令完成。

## 启动

先在旧 plain 服务的 tmux 窗口 Ctrl+C 停止服务、释放四张卡，再执行：

```bash
cd /mnt/data/zhangshuai/LLMs/Projects/spork
bash scripts/serve_qwen32_d3_smoke.sh
```

端口 8014；4×GPU / TP=4 / BF16 / 40960 上下文；ngram 草稿预算 20。最小验证启用 eager、同步调度和较小并发，以先核对功能。所有比较臂使用同一服务；不能将此处时间直接与此前 plain、非 eager 的结果混比。

出现 `Application startup complete` 后，另一个 tmux 窗口执行：

```bash
cd /mnt/data/zhangshuai/LLMs/Projects/spork
bash scripts/run_d3_smoke.sh
```

固定 seed 42 的第一个 τ² 任务，运行 baseline、D1+D2、D1+D2+D3，每组最多 3 轮。baseline 在此服务上包含 ngram；3 轮截断的质量分数不解释为完整任务成功率。

## 查看结果

脚本打印唯一结果目录 `results/d3_smoke_<时间>_<随机后缀>`，主要看 `d3_smoke_report.json`：

- `set_count`：成功绑定并注册草稿。
- `inject_count`：主请求确实提出 D3 草稿。
- `verified_injections`：该草稿出现在后续调度并通过验证流程。
- `accepted_draft_tokens`：真实接受数量，不含 ngram 草稿及 bonus/correction token。
- `accounting_errors`：应为 0。

`stage=tokens_accepted` 表示观察到正接受数；`verified_but_zero_accepted` 表示验证链路工作，但本样本草稿全部未被接受；`no_injection` 不能当作 D3 已跑通，需要连同 `run.log` 和对应 `server_d3_smoke_*.log` 检查。不自动扩大任务或修改阈值。
