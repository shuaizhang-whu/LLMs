# Chess 四卡本地 smoke 报告

## 结论

2026-07-18 使用 GPU 0、1、2、3 完成了一组 4-step、seed=248、k=1 的本地
sequential/speculative 对照实验。两条最终命令均退出码为 0，模型动态 READY、
棋局、产物写入和 GPU 释放均通过。

本次只能标记为“代码路径已跑通”。样本只有一个初始棋局和四个动作，不能当作
论文主实验复现结果，也不能外推论文全部环境的 lossless 结论。

## 固定配置

- Baseline commit：`dc938b9ef7474caf07fe4ad16549c1fa8c7d268c`
- Workspace HEAD：`657730558c34c5b79e9a2144165d814e679959b9`
- GPU：4 × RTX 3090；Actor 使用 0、1，Speculator 使用 2、3。
- Actor：Qwen3.6-27B-FP8，TP=2，固定 revision
  `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`。
- Speculator：Qwen3.5-4B，TP=2，固定 revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- vLLM 0.25.1、PyTorch 2.11.0+cu130；最大上下文 8192。
- temperature=0；Actor thinking 开、最大输出 2048；Speculator thinking 关、最大输出 256。
- 服务器没有 nvcc，因此按照已安装 vLLM 0.25.1 的源码开关设置
  `VLLM_USE_FLASHINFER_SAMPLER=0`，使用内置 sampler。该配置已写入每次 manifest。

## 成功命令

```bash
python scripts/run_local_experiment.py \
  --gpu-count 4 --gpu-ids 0,1,2,3 \
  --environment chess --experiment smoke \
  --execution-mode sequential --speculator-model qwen3.5-4b \
  --num-guesses 1 --stop-after 4 --seed 248 \
  --run-id chess-4gpu-sequential-k1-s248-20260718-r4

python scripts/run_local_experiment.py \
  --gpu-count 4 --gpu-ids 0,1,2,3 \
  --environment chess --experiment smoke \
  --execution-mode speculative --speculator-model qwen3.5-4b \
  --num-guesses 1 --stop-after 4 --seed 248 \
  --run-id chess-4gpu-speculative-k1-s248-20260718
```

## 结果

| 指标 | Sequential | Speculative k=1 |
|---|---:|---:|
| 游戏关键路径 wall-clock | 45.5125 s | 32.7478 s |
| 后台分支 drain | 0.00004 s | 104.1533 s |
| 含 drain 的总 wall-clock | 45.5125 s | 136.9011 s |
| Actor 调用 | 4 | 7 |
| Speculator 调用 | 0 | 3 |
| 总 token | 3,778 | 11,644 |
| 投机机会 / 命中 | 0 / 0 | 3 / 1 |
| 命中率 | 0% | 33.33% |
| 额外分支 | 0 | 2 |

- 游戏关键路径缩短 28.05%。
- 官方风格 Actor-only 理想公式估算节省 20.62%。
- 未使用分支继续生成导致总 wall-clock 增加 200.80%。
- 总 token 增加 208.21%，请求数从 4 次增加到 10 次。
- 三段 Actor/Speculator 请求重叠区间合计 1.0501 秒。这个简单求和不是独占时间，
  不能直接等同于隐藏延迟。

因此，本次结果同时显示了“用户可见关键路径有加速”和“本地总资源成本大幅增加”。
不能只报告前者。

## 确定性等价检查

严格比较全部通过：

- 动作：`[e2e4] [e7e5] [g1f3] [b8c6]`
- 动作哈希：`8fb3ba15d1cb2f9b7d1ef7288be0835f091dc67116b5d8d7b46ab7d86f9cc048`
- 最终 FEN：
  `r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3`
- 完成步数：4 vs 4。

这只证明本次确定性重放等价。

## 与论文付费 API 条件的差异

1. 论文可把不同分支横向分发到独立 API；本地所有 Actor 分支竞争同一个 27B
   vLLM 服务。命中可缩短游戏关键路径，但未命中分支仍占用同一服务。
2. RTX 3090 没有原生 FP8 计算，vLLM 日志表明 27B FP8 使用 Marlin
   weight-only 路径，计算密集场景可能更慢。
3. 当前机器的 GPU P2P 测试未通过，自定义 all-reduce 被禁用并回退到 NCCL；
   这是本地 TP 延迟的一部分。
4. 本次只跑四步、一个 seed、k=1；论文是多局、多次和 k=1/2/3 的聚合结果。
5. 本地实验不产生 API 费用，但 token 和调用数仍作为计算成本代理完整记录。

## 失败尝试与修复记录

失败目录全部保留，没有覆盖或删除：

- 初次：vLLM 0.25.1 不接受旧参数 `--disable-log-requests`，在 CUDA 前退出。
- `r1`：FlashInfer sampler 首次 JIT 需要 nvcc；机器只有 CUDA runtime/driver。
- `r2`：全局 HTTP proxy 截获 localhost 健康检查，人工中断后安全清理。
- `r3`：棋局成功，但退出后的首帧利用率统计仍为 100%，旧释放口径误报并返回 5。
- `r4`：上述问题均修复，sequential 全流程退出码 0。

修复没有删除检查、伪造结果、改变模型或降低 TP；每次失败的 manifest 和日志均保留。
