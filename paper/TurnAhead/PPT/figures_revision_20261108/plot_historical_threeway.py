import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def main():
    output = Path(__file__).resolve().parent
    root = output.parents[3]
    baseline = root / "Projects/spork/results/controller_full_20261017_021939_Wid7sh"
    successor = root / "Projects/spork-codex/results/successor_full_20261025_codex"
    baseline_review = baseline / "reports/final_review_20261023_024853"
    baseline_csv = baseline_review / "paired_task_latencies.csv"
    paired_csv = (
        root
        / "Projects/spork-codex/research_runs/full_eval_review_20261026/paired_tasks.csv"
    )
    with baseline_csv.open(encoding="utf-8-sig", newline="") as stream:
        baseline_rows = [
            row for row in csv.DictReader(stream) if row["config"] == "timing_off"
        ]
    with paired_csv.open(encoding="utf-8", newline="") as stream:
        paired_rows = [
            row for row in csv.DictReader(stream) if row["valid"] == "True"
        ]
    baseline_stats = json.loads(
        (baseline_review / "final_results_review.json").read_text(encoding="utf-8")
    )["statistics"]["all"]["timing_off"]
    if len(baseline_rows) != baseline_stats["tasks"] or len(paired_rows) != 148:
        raise ValueError("Historical report sample counts have changed.")
    if not all(
        row["same_completed_turn_tokens"] == "True"
        and row["same_completed_actions_results_state"] == "True"
        for row in paired_rows
    ):
        raise ValueError("SPORK/TurnAhead pairs failed the recorded equivalence checks.")
    overall = [
        statistics.mean(float(row["wall_s"]) for row in baseline_rows),
        statistics.mean(float(row["spork_s"]) for row in paired_rows),
        statistics.mean(float(row["successor_s"]) for row in paired_rows),
    ]
    if not math.isclose(overall[0], baseline_stats["mean_s"], rel_tol=1e-12):
        raise ValueError("Baseline CSV does not reproduce the archived mean.")

    record_paths = [
        baseline / "records/repeat_000/retail_0/timing_off.json",
        successor / "records/retail_0_r0_spork.json",
        successor / "records/retail_0_r0_raw.json",
    ]
    records = []
    for index, path in enumerate(record_paths):
        seal_path = (
            path.with_suffix(".complete.json")
            if index == 0
            else path.with_name(path.name + ".seal")
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != json.loads(seal_path.read_text(encoding="utf-8"))["sha256"]:
            raise ValueError(f"Record checksum mismatch: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["group_id"] != "retail:0" or record["errors"]:
            raise ValueError(f"Invalid single-task record: {path}")
        records.append(record)
    if any(
        turn["spec_dispatched"] or turn["num_probes"]
        for turn in records[0]["turns"]
    ):
        raise ValueError("The baseline record includes speculative tool dispatch.")
    turns = [record["turns"] for record in records]
    if not all(len(arm) == 6 for arm in turns):
        raise ValueError("The retail:0 example no longer contains six turns per arm.")
    same_actions = all(
        [(turn["baseline_tool_name"], turn["baseline_tool_args"]) for turn in arm]
        == [
            (turn["baseline_tool_name"], turn["baseline_tool_args"])
            for turn in turns[0]
        ]
        for arm in turns[1:]
    )
    same_state = all(
        arm[-1]["environment_after"] == turns[0][-1]["environment_after"]
        for arm in turns[1:]
    )
    same_tokens = [
        all(
            left["main_token_ids"] == right["main_token_ids"]
            for left, right in zip(turns[0], arm)
        )
        for arm in turns[1:]
    ]
    if not same_actions or not same_state:
        raise ValueError("The retail:0 example has different actions or final states.")
    single = [record["wall_s"] for record in records]
    if not all(math.isfinite(value) and value > 0 for value in overall + single):
        raise ValueError("All plotted durations must be finite and positive.")

    font_path = subprocess.check_output(
        ["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True
    ).strip()
    fm.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": fm.FontProperties(fname=font_path).get_name(),
            "font.size": 18,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(17, 7.2))
    figure.suptitle(
        "Qwen3-32B｜无动作投机、SPORK 与 TurnAhead",
        fontsize=26,
        y=0.965,
    )
    panels = [
        (
            axes[0],
            "整体平均｜历史实验记录",
            overall,
            ["无动作投机\nN=143", "SPORK\nN=148", "TurnAhead\nN=148"],
        ),
        (
            axes[1],
            "单任务示例｜retail:0 · 6 轮",
            single,
            ["无动作投机", "SPORK", "TurnAhead"],
        ),
    ]
    reductions = []
    for axis, title, values, labels in panels:
        axis.barh(
            range(3), values, color=["#394B69", "#77818C", "#A60000"], height=0.53
        )
        axis.set_yticks(range(3), labels)
        axis.set_ylim(2.5, -0.5)
        axis.set_xlim(0, 460)
        axis.set_xticks([0, 100, 200, 300, 400])
        axis.set_xlabel("任务耗时（秒）", fontsize=20)
        axis.tick_params(axis="y", length=0, pad=10, labelsize=19)
        axis.xaxis.grid(True, color="#E4E7EB")
        axis.set_axisbelow(True)
        axis.set_title(title, fontsize=22, loc="left", pad=54)
        reduction = 100 * (1 - values[2] / values[1])
        reductions.append(reduction)
        axis.text(
            0,
            1.035,
            f"TurnAhead 较 SPORK 降低 {reduction:.2f}%",
            transform=axis.transAxes,
            fontsize=18,
            color="#A60000",
            va="bottom",
        )
        for position, value in enumerate(values):
            axis.text(value + 7, position, f"{value:.2f}", va="center", fontsize=21)
    figure.subplots_adjust(
        left=0.12, right=0.98, bottom=0.24, top=0.73, wspace=0.58
    )
    figure.text(
        0.045,
        0.095,
        "无动作投机：未提前投递工具调用，不代表关闭所有推理优化。",
        fontsize=13,
        color="#647080",
    )
    figure.text(
        0.045,
        0.045,
        "跨批次历史耗时对照：整体样本数不同；retail:0 动作与最终状态一致，基线生成 token 不同。",
        fontsize=13,
        color="#647080",
    )
    name = "实验结果-三组对比-整体与单任务"
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=240)
    plt.close(figure)

    provenance = {
        "model": "Qwen3-32B",
        "comparison_type": "historical_cross_batch_not_a_controlled_three_arm_trial",
        "method_order": ["no_early_tool_dispatch", "spork", "turnahead_raw"],
        "baseline_definition": (
            "timing_off disables early tool dispatch; engine-level optimizations "
            "are not claimed to be disabled."
        ),
        "source_runs": {"baseline": str(baseline), "spork_turnahead": str(successor)},
        "overall": {
            "mean_task_s": overall,
            "sample_counts": [len(baseline_rows), len(paired_rows), len(paired_rows)],
            "baseline_scope": "Shared successful task set of the six-policy controller evaluation.",
            "spork_turnahead_scope": "Shared successful task set of the two-policy successor evaluation.",
            "turnahead_vs_spork_reduction_percent": reductions[0],
        },
        "retail_0": {
            "wall_s": single,
            "turns": [len(arm) for arm in turns],
            "decode_tokens": [
                sum(turn["total_decode_tokens"] for turn in arm) for arm in turns
            ],
            "same_action_trajectory": same_actions,
            "same_final_state": same_state,
            "baseline_same_tokens_as_spork_and_turnahead": same_tokens,
            "turnahead_vs_spork_reduction_percent": reductions[1],
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [baseline_csv, paired_csv] + record_paths
        },
        "outputs": [f"{name}.png", f"{name}.svg"],
    }
    (output / "threeway-data-and-scope.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(provenance["retail_0"], ensure_ascii=False, indent=2))
    print(f"Saved PNG, SVG and provenance under {output}")


if __name__ == "__main__":
    main()
