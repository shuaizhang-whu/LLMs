#!/usr/bin/env python3
"""Plot existing controller metrics on CPU; never load or train a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


VARIANTS = {
    "full": ("完整输入", "#2563EB"),
    "masked_reasoning": ("屏蔽思考内容", "#E87924"),
}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.run.resolve()
    output = (args.output or root / "figures").resolve()
    names = ["bert_training_200epochs.png", "bert_training_200epochs.svg",
             "metrics_by_epoch.csv", "training_plot_summary.json"]
    if not args.overwrite and any((output / name).exists() for name in names):
        raise FileExistsError("Plot artifacts already exist; choose another output or use --overwrite")
    manifest, summary = read(root / "manifest.json"), read(root / "summary.json")
    seeds, epochs = manifest["seeds"], manifest["epochs_per_fit"]
    expected_epochs = list(range(1, epochs + 1))
    fixed_wait = summary["baselines"]["fixed_wait"]["task_mean_recorded_gain_s"]
    oracle = summary["baselines"]["fixed_wait"]["empirical_oracle_task_gain_s"]
    runs, sources, selected = {}, {}, {}
    flat_rows = []
    for variant in VARIANTS:
        runs[variant], selected[variant] = [], []
        for seed in seeds:
            path = root / f"{variant}_seed_{seed}" / "metrics.jsonl"
            rows = [json.loads(line) for line in path.open() if line.strip()]
            report = read(path.parent / "training_report.json")
            if [r["epoch"] for r in rows] != expected_epochs or report["epochs_completed"] != epochs:
                raise ValueError(f"Incomplete epoch sequence: {path}")
            if any(r["rows"] != 46 or r["tasks"] != 23 for r in rows):
                raise ValueError(f"Unexpected validation population: {path}")
            runs[variant].append(rows)
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            chosen = report["selected_validation_metrics"]
            if rows[chosen["epoch"] - 1]["task_mean_recorded_gain_s"] != chosen["task_mean_recorded_gain_s"]:
                raise ValueError(f"Selected checkpoint does not match epoch metrics: {path}")
            selected[variant].append({"seed": seed, "epoch": chosen["epoch"],
                                      "checkpoint": report["best_checkpoint"],
                                      "gain_s": chosen["task_mean_recorded_gain_s"],
                                      "gain_over_fixed_wait_ms": 1000 * (chosen["task_mean_recorded_gain_s"] - fixed_wait)})
            for row in rows:
                flat_rows.append({
                    "variant": variant, "seed": seed, "epoch": row["epoch"],
                    "training_normalized_mse": row["training_normalized_mse"],
                    "training_eval_mse_s2": row["training_evaluation"]["mse_seconds_squared"],
                    "validation_mse_s2": row["mse_seconds_squared"],
                    "validation_mae_now_s": row["mae_now_s"],
                    "validation_mae_wait_s": row["mae_wait_s"],
                    "validation_mae_mean_s": (row["mae_now_s"] + row["mae_wait_s"]) / 2,
                    "validation_task_gain_s": row["task_mean_recorded_gain_s"],
                    "gain_over_fixed_wait_ms": 1000 * (row["task_mean_recorded_gain_s"] - fixed_wait),
                    **{f"validation_{policy}_count": row["selected_policy_counts"][policy]
                       for policy in ("off", "now", "wait")},
                })

    def values(variant, get):
        array = np.array([[get(row) for row in run] for run in runs[variant]], dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite metrics in {variant}")
        return array

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font_path.exists():
        raise FileNotFoundError("A CJK font is required for the Chinese figure labels")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        "font.family": [font_name, "DejaVu Sans"], "axes.unicode_minus": False,
        "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#94A3B8", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "path",
    })
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.subplots_adjust(left=.075, right=.975, bottom=.135, top=.825, hspace=.52, wspace=.23)
    fig.suptitle(f"BERT 启动时机控制器：{epochs} 轮训练与验证", fontsize=21, y=.98)
    fig.text(.5, .929, "BERT-Tiny  ·  214 个训练前缀 / 46 个验证前缀  ·  随机种子 42、43、44  ·  GPU 训练",
             ha="center", fontsize=11, color="#475569")
    handles = [Line2D([0], [0], color=color, lw=2.6, label=label) for label, color in VARIANTS.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .904), ncol=2,
               frameon=False, handlelength=3, columnspacing=3)
    x = np.arange(1, epochs + 1)

    def line(ax, variant, get, *, style="-", shade=True):
        array = values(variant, get)
        mean = array.mean(axis=0)
        sd = array.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
        color = VARIANTS[variant][1]
        if shade:
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.11, linewidth=0)
        ax.plot(x, mean, style, color=color, lw=1.65, alpha=.98)

    for ax in axes.flat:
        ax.set_xlim(1, epochs)
        ax.set_xticks([1, 50, 100, 150, 200] if epochs == 200 else np.linspace(1, epochs, 5, dtype=int))
        ax.set_xlabel("训练轮次（epoch）")
        ax.grid(axis="y", color="#E2E8F0", linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)

    a, b, c, d = axes.flat
    a.set_title("A  训练优化损失", loc="left", pad=12)
    a.set_ylabel("标准化 MSE（无量纲）")
    for variant in VARIANTS:
        line(a, variant, lambda r: r["training_normalized_mse"])
    a.set_ylim(bottom=0)

    b.set_title("B  训练集与验证集误差", loc="left", pad=12)
    b.set_ylabel("MSE（秒²）")
    for variant in VARIANTS:
        line(b, variant, lambda r: r["mse_seconds_squared"])
        line(b, variant, lambda r: r["training_evaluation"]["mse_seconds_squared"], style="--", shade=False)
    b.set_ylim(bottom=0)
    b.legend(handles=[Line2D([0], [0], color="#475569", lw=1.6, label="验证集（实线）"),
                      Line2D([0], [0], color="#475569", lw=1.6, ls="--", label="训练集（虚线）")],
             loc="upper right", fontsize=9, frameon=True, framealpha=.92)

    c.set_title("C  验证集策略收益", loc="left", pad=12)
    c.set_ylabel("相对固定 wait 的增益（毫秒 / 任务）")
    for variant in VARIANTS:
        line(c, variant, lambda r: 1000 * (r["task_mean_recorded_gain_s"] - fixed_wait))
    zero = c.axhline(0, color="#475569", lw=1.2, ls="--", label="固定 wait 基线")
    reference = c.axhline(1000 * (oracle - fixed_wait), color="#64748B", lw=1.2, ls=":", label="标签逐前缀最优参考")
    c.legend(handles=[zero, reference], loc="lower right", fontsize=9, frameon=True, framealpha=.92)

    d.set_title("D  验证集绝对预测误差", loc="left", pad=12)
    d.set_ylabel("平均 MAE（秒）")
    for variant in VARIANTS:
        line(d, variant, lambda r: (r["mae_now_s"] + r["mae_wait_s"]) / 2)
    d.set_ylim(bottom=0)

    fig.text(.075, .065, "曲线：3 个随机种子的均值；阴影：种子间 ±1 标准差；未做平滑。D 图对 now、wait 两个输出的 MAE 取平均。",
             fontsize=10, color="#475569")
    fig.text(.075, .032, "收益来自离线标签，不含控制器耗时；验证集用于选检查点。上述曲线不代表在线端到端加速。",
             fontsize=10, color="#475569")
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"bert_training_200epochs.{suffix}", dpi=args.dpi, facecolor="white")
    plt.close(fig)
    with (output / "metrics_by_epoch.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    details = {}
    for variant in VARIANTS:
        details[variant] = {"selected_checkpoints": selected[variant], "epoch_1": {}, "epoch_last": {}}
        getters = {
            "training_normalized_mse": lambda r: r["training_normalized_mse"],
            "training_eval_mse_s2": lambda r: r["training_evaluation"]["mse_seconds_squared"],
            "validation_mse_s2": lambda r: r["mse_seconds_squared"],
            "validation_mae_mean_s": lambda r: (r["mae_now_s"] + r["mae_wait_s"]) / 2,
            "task_gain_s": lambda r: r["task_mean_recorded_gain_s"],
        }
        for name, get in getters.items():
            array = values(variant, get)
            for key, index in (("epoch_1", 0), ("epoch_last", -1)):
                details[variant][key][name] = float(array[:, index].mean())
        details[variant]["loss_reduction_fraction"] = 1 - details[variant]["epoch_last"]["training_normalized_mse"] / details[variant]["epoch_1"]["training_normalized_mse"]
        details[variant]["selected_mean_task_gain_s"] = summary["variants"][variant]["mean_task_gain_s"]
        details[variant]["selected_gain_over_fixed_wait"] = summary["variants"][variant]["paired_differences_vs_baselines_s"]["fixed_wait"]
    audit = {"run": str(root), "epochs": epochs, "seeds": seeds, "fits": len(seeds) * len(VARIANTS),
             "training_metric_records": len(flat_rows), "source_metrics_sha256": sources,
             "loss": "mean(((predicted_gain_s - observed_gain_s) / train_only_scale_s)**2)",
             "loss_scale": "Per-output training standard deviation, floored at one second; fixed after fitting on training data.",
             "curve_aggregation": "Same-epoch mean across all three predeclared seeds, no smoothing.",
             "shading": "One sample standard deviation across seeds, not a confidence interval.",
             "gain_metric": "Offline recorded gain, averaged within task then equally across validation tasks.",
             "fixed_wait_gain_s": fixed_wait, "label_oracle_gain_s": oracle, "variants": details,
             "full_minus_masked_reasoning_s": summary["full_minus_masked_reasoning_s"],
             "scope": "Validation results; checkpoint selection bias and label timing noise are not removed. Excludes controller runtime cost. No test evaluation, training or GPU work is performed by this plotting script."}
    (output / "training_plot_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "records": len(flat_rows), "epochs": epochs,
                      "selected_gain_s": {v: details[v]["selected_mean_task_gain_s"] for v in VARIANTS}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
