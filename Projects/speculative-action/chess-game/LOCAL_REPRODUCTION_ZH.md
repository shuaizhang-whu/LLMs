# Chess 本地免费模型复现指南

本指南对应新增的可审计本地路径；官方 `regular_workflow/` 和
`speculative_workflow/` 保持不变。离线样例结果见
`reproduction/offline_20260718/`。

## 环境

Chess 客户端与模型服务分离：

```bash
source /mnt/data/zhangshuai/venvs/agent-action-chess/bin/activate
cd /mnt/data/zhangshuai/LLMs/Projects/speculative-action/chess-game
```

- 当前环境运行棋盘、调度、指标和测试。
- vLLM 由启动器固定使用
  `/mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/python`，不需要在
  Chess 环境重复安装 CUDA/PyTorch/vLLM。
- 当前服务器没有 `nvcc`。启动器固定设置
  `VLLM_USE_FLASHINFER_SAMPLER=0`，使用 vLLM 内置 sampler，避免 FlashInfer
  首次 JIT 失败；该覆盖会写入实验 manifest。
- 服务器存在全局 HTTP proxy。启动器健康检查显式直连 localhost，Chess 子进程
  固定设置 `NO_PROXY=127.0.0.1,localhost,::1`，避免本地请求被代理截获。

## 先检查模型

三个模型均固定到 `Tool/model_manager.py` 中记录的官方 commit：

```bash
cd /mnt/data/zhangshuai/LLMs
MM_PY=/mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/python
HF=/mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/hf
$MM_PY Tool/model_manager.py --hf-executable "$HF" status
$MM_PY Tool/model_manager.py --hf-executable "$HF" verify --model qwen3.5-4b
$MM_PY Tool/model_manager.py --hf-executable "$HF" verify --model qwen3.5-9b
$MM_PY Tool/model_manager.py --hf-executable "$HF" verify --model qwen3.6-27b-fp8
```

`STATIC_VERIFIED` 只表示磁盘文件完整；只有启动器实际完成 vLLM 加载和一次生成
后，对应实验 manifest 才会记录动态 `READY`。

## 每个批次必须显式选择 GPU 数量

只读 preflight 示例：

```bash
python scripts/run_local_experiment.py \
  --gpu-count 2 \
  --environment chess \
  --experiment smoke \
  --preflight-only
```

也可以加 `--gpu-ids 0,2`。启动器连续采样三次；存在 compute 进程、显存超过
1GiB 或任一次利用率超过 5% 的卡都不会被选择。它不会终止或抢占已有进程。

分配规则：

- 2 卡：9B Actor（1 卡）+ 4B Speculator（1 卡），仅限不超过 10 步 smoke。
- 3 卡：27B-FP8 Actor（TP=2）+ 4B/9B Speculator（TP=1）。
- 4 卡：27B-FP8 Actor（TP=2）+ 4B/9B Speculator（TP=2）。

## 运行

2 卡顺序与真异步 smoke 示例（两个 `run-id` 必须不同）：

```bash
python scripts/run_local_experiment.py \
  --gpu-count 2 --environment chess --experiment smoke \
  --execution-mode sequential --num-guesses 1 --stop-after 4 \
  --seed 248 --run-id qwen35-sequential-s248

python scripts/run_local_experiment.py \
  --gpu-count 2 --environment chess --experiment smoke \
  --execution-mode speculative --num-guesses 1 --stop-after 4 \
  --seed 248 --run-id qwen35-spec-k1-s248
```

3 卡 27B 示例：

```bash
python scripts/run_local_experiment.py \
  --gpu-count 3 --environment chess --experiment main \
  --execution-mode speculative --speculator-model qwen3.5-4b \
  --num-guesses 3 --stop-after 30 --seed 248 \
  --run-id qwen36-27b-qwen35-4b-k3-s248
```

启动器会再次核对同一批物理卡、启动两个隔离的 vLLM 服务、执行小生成、运行
Chess，并在 `finally` 中只关闭自己创建的进程组。服务异常、OOM、Chess 非零或
显存未确认释放都会得到非零退出码。

退出后的 `utilization.gpu` 是时间窗口统计，首帧可能残留高利用率。释放验收严格
要求没有 compute 进程且显存回到阈值内；残留利用率继续写入 manifest，但不会在
进程和显存已经释放时制造假失败。启动前的空闲判定仍要求三次利用率都不超过 5%。

## Lossless 重放比较

对相同 seed、步数和模型的顺序/投机目录执行：

```bash
python scripts/compare_chess_runs.py \
  --sequential-run /path/to/sequential/run \
  --speculative-run /path/to/speculative/run \
  --output /path/to/lossless_comparison.json
```

它严格比较动作序列、动作哈希、最终 FEN 和完成步数。通过只说明该次确定性
重放等价，不能单独外推为论文全部环境的 “lossless” 结论。

## 结果口径

每次运行按 GPU 数量、物理卡和型号拓扑分目录保存：GPU/模型 revision、命令、
依赖、seed、服务日志、流式 TTFT、token、命中率、分支数、真实 wall-clock、
后台 drain 和官方风格理想估算。服务器内部排队时间无法由 OpenAI-compatible
接口观测，因此明确记为 `null`，不会用客户端耗时伪造。

本地多个分支共享同一个 Actor 服务，可能出现零加速或负加速；这与论文中可横向
扩展的付费 API 条件不同，必须在每次报告中单独说明。
