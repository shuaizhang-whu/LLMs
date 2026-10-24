# SPORK v2：稳定候选刷新（stable candidate refresh）交接说明

日期：2026-10-21
工作目录：`/mnt/data/zhangshuai/LLMs/Projects/spork-claude`

## 1. 一句话结论

在 SPORK 的动作投机基础上，把 ETA 的「重复探针检测动作稳定 → 截断推理」改造成
「稳定候选与已 dispatch 候选不一致时，刷新投机执行」，从而让**首探针预测错误**的回合
也能提前退出。两个独立冒烟显示：

- 首探针错误的 airline:2：185.1 s → **36.6 s（5.06×）**；
- 首探针正确的 retail:0：65.5 s → **47.4 s（27.6%）**；
- 两个任务的动作、工具结果、环境状态、最终答案均与 SPORK 一致。

## 2. 机制

对每个回合：

1. 主模型输出第一个 token 时，按 SPORK D1/D2 发起强制探针，预测并 dispatch 工具调用；
2. 之后每隔 50 个 CoT token（吸附到句界，最多 15 次）做一次检查：
   - 自由探针：在 `</think>` 边界生成 1 个 token，读 `p(<tool_call>)`；
   - 强制探针：续写完整工具调用；
   - 候选资格：`p ≥ 0.9`、强制探针置信度 ≥ 0.9、调用可解析；
3. 连续 3 次资格候选完全相同：
   - 若与当前 dispatch 的调用相同 → 满足 ETA 的稳定条件；
   - 若不同，且旧投机执行已完成 → 取消旧候选，dispatch 新候选（默认最多刷新 1 次）；
4. 工具就绪 + 到达句界后截断主模型推理，只追加 `</think>`；
5. 模型自己生成最终动作；只有与当前 dispatch 调用 exact match 才复用投机结果，
   否则串行重执行；
6. 质量护栏（影子回滚）：截断后不立即取消主流，让它在后台继续作为完整推理的影子；
   续写动作与当前 dispatch 候选匹配则取消影子，不匹配则直接用影子已经生成的完整推理，
   不再发起第二次解码。回滚原因记为 `no_call` / `truncated` / `action_mismatch`。
   推理被截断本身是不可逆的近似，**不是 lossless**，但最坏情况只比 SPORK 慢约 1%。

## 3. 已验证的数字

同一 8017 服务、Qwen3-32B、TP=4、batch-invariant、2 s 工具下限、固定 seed 42。

| 任务 | 类型 | SPORK | ETA | refresh | 省时 | 质量 |
|---|---|---:|---:|---:|---:|---|
| airline:2 | 首探针错 | 185.0 s | 188.4 s | 36.8 s | 148.2 s（80.1%） | 一致 |
| airline:3 | 首探针对 | 29.9 s | 25.5 s | 25.5 s | 4.4 s（14.8%） | 一致 |
| airline:4 | 截断后动作分叉 | 76.3 s | 38.0 s（分叉） | 77.0 s（影子回滚） | −0.75 s（−1.0%） | 一致 |
| airline:6 | 首探针对 | 52.4 s | 25.4 s | 25.5 s | 26.9 s（51.4%） | 一致 |
| retail:0 | 首探针对 | 65.5 s | 47.4 s | 47.4 s | 18.0 s（27.6%） | 一致 |
| retail:3 | 首探针错 | 103.0 s | — | 52.5 s | 50.6 s（49.1%） | 一致 |

airline:4 是质量护栏的关键案例：不加护栏时 ETA/refresh 在 token 266 截断后生成了
`get_flight_status`，而完整推理其实直接回答、不调用工具，导致最终答案为空。
影子回滚把主流保留为完整推理的备份，检测到动作不一致时直接使用影子输出，
代价只有约 1%。最坏情况不再改变动作。

按回合类型汇总（最终 6 任务冒烟 `results/final_smoke_20261021`，24 次运行）：

| 类型 | n | SPORK 均值 | refresh 均值 | 平均省时 | 质量 |
|---|---:|---:|---:|---:|---|
| tool_matched | 3 | 49.2 s | 33.1 s | 16.1 s（32.7%） | 3/3 |
| tool_mismatched | 2 | 143.9 s | 44.7 s | 99.2 s（68.9%） | 2/2 |
| no_tool（影子回滚） | 1 | 76.2 s | 76.9 s | −0.7 s（−0.9%） | 1/1 |

总体：SPORK 85.3 s → refresh 44.2 s（**1.93×**）；ETA 78.5 s（1.09×，两个未命中任务
无法退出，反而略慢）；off 87.3 s。无错误、无质量回归。
注意这 6 个任务是为机制验证**有意选的**（2 个未命中 + 1 个 no-tool），
不能把 1.93× 当作 155 任务的总体加速；无偏数字以完整验证为准。

`off`（完整探针、禁止截断）的轨迹分析：

- airline:2 的首次 3 连稳定调用 = `get_user_details`，与整轮最终动作一致；
  首探针 `search_direct_flight` 与最终动作不一致——正是 refresh 解锁的回合。
- airline:3 的首次稳定调用也与最终动作一致。

现有 930 次运行数据的事后统计：

- 工具回合 614 个，投机命中 478 个（77.9%），未命中 136 个（22.1%）；
- 命中回合的工具在 `main_end` 前中位数 48.1 s 就已完成，slack 占全部解码时间 64.2%；
- 有未命中工具回合的任务占 83/143（58.0%）。

因此 refresh 影响的不是边角案例，而是超过一半任务至少一次。

## 4. 完整验证命令（由用户在 tmux 启动）

前置条件：

- 8017 服务空闲（`/spork/status` 的 `active_requests == 0`）；
- 服务为 batch-invariant 且 `has_drafter == true`；
- `spork-claude` 下 `tau2-bench`、`prefix_data` 软链接有效。

```bash
cd /mnt/data/zhangshuai/LLMs/Projects/spork-claude
PYTHONPATH=. /mnt/data/zhangshuai/anaconda3/envs/spork/bin/python \
  scripts/run_refresh_full_eval.py \
  --output results/refresh_full_20261021 \
  --configs spork,eta,refresh,off \
  --scope all --max-turns 8 --repetitions 1 \
  2>&1 | tee results/refresh_full_20261021.launch.log
```

- 规模：155 任务 × 4 配置 × 1 次 = 620 次运行。
- 按旧实验速度估计约 10–15 小时；脚本每完成一次运行就写 `summary.json`/`summary.md`。
- 断点续跑：重新执行同一命令会跳过已完成的记录；失败任务会重跑。
- 脚本会冻结 manifest 与代码快照（`spork_core/evaluation_artifacts.py`）；
  运行期间修改源码后再次续跑会被拒绝，确保所有记录对应同一实现。
- 若 D3/server 状态不干净，脚本会保存结果后停止，不会带脏状态继续计时。
- 小批量冒烟可用 `--task-ids airline:2,airline:4` 精确选任务。

先做一次不占 GPU 的检查：

```bash
cd /mnt/data/zhangshuai/LLMs/Projects/spork-claude
PYTHONPATH=. /mnt/data/zhangshuai/anaconda3/envs/spork/bin/python \
  scripts/run_refresh_full_eval.py --output results/_dryrun --dry-run --scope all
rm -rf results/_dryrun
```

## 5. 结果解读的边界

- 质量指标是固定用户消息流程下的 τ² ACTION reward，**不是官方交互式 τ² 成功率**。
  若论文需要任务成功率，应在官方 user simulator 下另跑一轮。
- 冒烟样本只有 2–8 个任务，不能据此声称质量非劣；只能排除灾难性分叉。
- 已经观测到一次真实分叉：airline:4 上 SPORK 完整推理直接回答，ETA/refresh 在 token 266
  截断后生成了 `get_flight_status`，最终答案为空。`off` 轨迹显示模型直到 token 650
  才放弃工具调用。这证明「早期稳定」不等于「完整推理会做同一决定」。
  已加入 `rollback_on_action_mismatch` 护栏；完整验证必须报告回滚率，
  并按「是否回滚」分组报告质量。
- 截断改变了 Actor 的推理策略，方法不是 lossless。必须在完整验证中按
  「首探针命中 / 未命中」分组报告质量，不能只报平均值。
- 探针与主解码重叠良好（未命中回合 overhead 约 1.3%）；不要把它包装成
  「探针成本是主要瓶颈」。D3 复用/分层探针在当前硬件上不是主要卖点。
- `spork_archive` 本次没有改动，因此没有创建 `spork_archive-claude`。

## 6. 代码位置

- `spork_core/stable_refresh_exit.py`：主实现（ETA 守卫 + 稳定候选刷新）。
- `scripts/run_stable_refresh_smoke.py`：小批量在线冒烟（支持 spork/eta/refresh/off）。
- `scripts/run_refresh_full_eval.py`：155 任务完整验证（断点续跑、按划分汇总）。
- `scripts/analyze_refresh_smoke.py`：按初始命中/未命中分组的配对分析。
- `scripts/analyze_off_stability.py`：从 `off` 轨迹估计提前退出的动作一致率。
- `spork_core/early_exit.py`：上一轮的 ETA/reuse/redo 实现，保留作对照；
  正式冒烟以 `stable_refresh_exit.py` 为准。
- `spork_core/evaluation_artifacts.py`：manifest/代码快照冻结、记录封存、
  可见最终答案比较（完整验证用）。
- `research_runs/ideas_20261021/GRAFT_MATRIX.md`：7 篇论文的候选嫁接矩阵与证据。
