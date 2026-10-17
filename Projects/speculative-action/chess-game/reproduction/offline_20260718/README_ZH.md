# Chess 官方样例离线重算（2026-07-18）

本目录仅复算仓库附带的 5 条 Chess 轨迹，不调用模型、网络或付费 API。为避免
污染 baseline，所有分析均在 `sample_trajectories/` 工作副本中生成；官方源目录
`trajectories/sample_trajectories/` 未被修改。

执行入口如下（工作目录为 `chess-game/`）：

```bash
python guess_analysis.py --base-dir reproduction/offline_20260718/sample_trajectories
python guess_analysis.py --base-dir reproduction/offline_20260718/sample_trajectories --confidence
python plot.py bars --root-path reproduction/offline_20260718/sample_trajectories \
  --target-steps 30 --step-accuracy \
  --output reproduction/offline_20260718/combined_bar_plots_step_accuracy.pdf
python plot.py bars --root-path reproduction/offline_20260718/sample_trajectories \
  --target-steps 30 --speculative-accuracy \
  --output reproduction/offline_20260718/combined_bar_plots_window_accuracy.pdf
python plot.py time-token --base-dir reproduction/offline_20260718/sample_trajectories
python plot.py time-token --base-dir reproduction/offline_20260718/sample_trajectories --confidence
python reproduction/offline_20260718/audit_offline.py
```

30 步聚合结果与官方绘图流程一致：严格按总步数计算的命中率在 k=1/2/3 时为
31.3±8.6%、41.3±6.3%、54.7±11.2%；论文兼容的投机窗口命中率为
34.7±13.9%、43.3±7.5%、62.9±21.3%；估算节省时间为
11.8±4.8%、15.0±3.8%、19.5±9.0%（均为均值±95% t 置信区间半宽，n=5）。

重要边界：这里的“节省时间”由已保存请求耗时套用官方理想公式得到，不是真正的
在线端到端 wall-clock 加速。真实并发、GPU 竞争、任务正确性和 lossless 等价性
必须由后续本地在线实验单独验证。
