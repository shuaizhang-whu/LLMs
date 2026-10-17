# HotPotQA 本地在线 smoke test 报告

## 结论

本地 OpenAI-compatible vLLM 服务和 HotPotQA 在线轨迹采集链路已经调通。成功命令
退出码为 0，并生成四类完整样本文件。本结果只证明 1 个样本、最多 2 个 action 的
工程 smoke test 成功；不代表论文 HotPotQA 全量 baseline、在线并发加速、任务准确率
或 lossless 结论已经复现。

## 成功产物

目录：

`run_metrics/agent_local-qwen2.5-7b-instruct_top3/trajs_local-qwen2.5-7b-instruct/7107/`

- `log.txt`：2934 bytes
- `normalobs.json`：8226 bytes，2 个 action
- `simobs.json`：8929 bytes，2 组候选，每组 3 个 action
- `metrics.json`：`{"general": 0.0, "Search": 0.0}`

本次两步真实 Wikipedia 环境耗时为 1.1500s、0.9968s；两步模拟 observation
生成耗时为 2.0930s、1.9718s。样本总墙钟时间约 12.4s，最终 EM=False、F1=0。
这组数值不用于论文结论，只用于证明真实 Wikipedia 请求、Actor 生成、Top-3 候选、
Guess LLM、轨迹落盘和指标计算全部实际执行。

## 为本地 smoke test 所做的最小修改

- `src/llm_client.py`：新增 `local/<served-model-name>` 路由；仅初始化实际使用的
  provider；从环境变量读取本地 base URL，不要求云 API Key。
- `run.py`：新增 `--num-samples` 和 `--num-steps`，避免初次测试直接跑 20 个样本。
- `src/wrappers.py`、`src/runner.py`：把 Gymnasium 1.x 已移除的隐式 wrapper
  属性转发改成公开 `get_wrapper_attr` 调用。
- 新增本地客户端和 Gymnasium 兼容测试，并把离线导入测试改为独立子进程检查。

共 13 项 `unittest` 已通过，`agent-action` 与 `agent-action-vllm` 的 `pip check`
均无冲突。

## 实际遇到并定位的错误

1. vLLM 首次启动失败：FlashInfer sampler warmup 找不到 `/usr/local/cuda/nvcc`。
   环境内虽有 pip CUDA 工具链，但 FlashInfer 未找到系统 CUDA。使用 vLLM 支持的
   `VLLM_USE_FLASHINFER_SAMPLER=0` 后服务成功；注意力仍使用 FlashAttention 2。
2. 首次项目运行失败：Gymnasium 1.3.0 不再通过 `Wrapper.__getattr__` 转发
   `steps`、`answer`、`sim_obs`、`write`。已用公开 wrapper API 最小修复。
3. 第二次项目运行收到 HTTP 502：服务器全局 `HTTP_PROXY` 把本机 8000 请求送进
   代理。已用 `NO_PROXY=127.0.0.1,localhost` 验证解决。

## 尚未修复的 baseline 缺陷

`HistoryWrapper.reset` 当前丢弃 runner 传入的 `idx`。因此本次输出目录编号是固定洗牌
得到的 7107，但实际问题是随机抽取的问题，不能把 7107 当作真实 HotPotQA 数据索引。
这会影响严格可复现性，下一步应先补测试并修复 idx 透传，再跑 1～2 个完整 8-step
样本。当前 runner 仍按串行顺序执行 Actor、候选生成、真实环境调用和模拟调用，不能
把本次 12.4s 解释为论文中的在线并发加速。
