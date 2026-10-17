"""新 HotPotQA 实验目录的汇总、绘图与 lossless 对照。"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import Utils


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_run(run_root):
    """读取一个 ReproductionRunner 输出根目录。"""
    root = Path(run_root).expanduser().resolve()
    return root, _read_json(root / "summary.json")


def _sample_dirs(root):
    return sorted(
        (
            path
            for path in Path(root).iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )


def validate_lossless_pair(sequential_root, online_root):
    """逐样本比较权威动作、最终答案与 EM/F1，返回可落盘审计结果。"""
    sequential_root = Path(sequential_root).expanduser().resolve()
    online_root = Path(online_root).expanduser().resolve()
    sequential_ids = {path.name for path in _sample_dirs(sequential_root)}
    online_ids = {path.name for path in _sample_dirs(online_root)}
    all_ids = sorted(sequential_ids | online_ids, key=int)
    samples = []
    passed = True

    for sample_id in all_ids:
        record = {"sample_id": int(sample_id), "passed": False, "mismatches": []}
        if sample_id not in sequential_ids or sample_id not in online_ids:
            record["mismatches"].append("sample_missing_from_one_run")
            samples.append(record)
            passed = False
            continue

        sequential_trace = _read_json(
            sequential_root / sample_id / "authority_trace.json"
        )
        online_trace = _read_json(online_root / sample_id / "authority_trace.json")
        sequential_metrics = _read_json(
            sequential_root / sample_id / "sample_metrics.json"
        )
        online_metrics = _read_json(online_root / sample_id / "sample_metrics.json")
        sequential_actions = [
            item["action"] for item in sequential_trace["steps"]
        ]
        online_actions = [item["action"] for item in online_trace["steps"]]
        if sequential_actions != online_actions:
            record["mismatches"].append("authority_actions")

        for field in ("answer", "ground_truth", "em", "f1"):
            if (
                sequential_metrics["task"].get(field)
                != online_metrics["task"].get(field)
            ):
                record["mismatches"].append(f"task.{field}")
        if (
            sequential_metrics["actual_question_idx"]
            != online_metrics["actual_question_idx"]
        ):
            record["mismatches"].append("actual_question_idx")
        record["passed"] = not record["mismatches"]
        passed = passed and record["passed"]
        samples.append(record)

    return {
        "passed": passed,
        "sequential_root": str(sequential_root),
        "online_root": str(online_root),
        "n_samples": len(samples),
        "samples": samples,
    }


def _save_bar(path, title, labels, values, ylabel):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.5), 4.5))
    positions = range(len(labels))
    axis.bar(positions, [0 if value is None else value for value in values])
    axis.set_xticks(list(positions), labels, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate_comparison(run_roots, output_dir):
    """从多个 summary.json 生成动作、任务、token 与延迟对照图。"""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for run_root in run_roots:
        root, summary = load_run(run_root)
        label = f"{summary['execution_mode']}-k{summary['top_k']}"
        runs.append({"root": str(root), "label": label, "summary": summary})

    labels = [item["label"] for item in runs]
    _save_bar(
        output_dir / "task_em.png",
        "HotPotQA Exact Match",
        labels,
        [item["summary"]["task"]["em_macro"] for item in runs],
        "EM",
    )
    _save_bar(
        output_dir / "sample_wall_clock.png",
        "End-to-end sample wall clock",
        labels,
        [
            item["summary"]["latency"]["sample_wall_seconds_macro"]
            for item in runs
        ],
        "seconds / sample",
    )
    _save_bar(
        output_dir / "total_tokens.png",
        "Total LLM tokens by run",
        labels,
        [
            sum(
                role["total_tokens"]
                for role in item["summary"]["usage"].values()
            )
            for item in runs
        ],
        "tokens",
    )

    prediction_runs = [
        item for item in runs if item["summary"].get("prediction") is not None
    ]
    if prediction_runs:
        prediction_labels = [item["label"] for item in prediction_runs]
        top1 = [
            item["summary"]["prediction"]["paper"]["top1"]["general"]["macro"]
            for item in prediction_runs
        ]
        top3 = []
        for item in prediction_runs:
            prediction = item["summary"]["prediction"]["paper"]
            top3.append(
                prediction.get("top3", prediction["top1"])["general"]["macro"]
            )

        figure, axis = plt.subplots(figsize=(max(7, len(prediction_labels) * 1.7), 4.5))
        positions = list(range(len(prediction_labels)))
        width = 0.36
        axis.bar([value - width / 2 for value in positions], top1, width, label="Top-1")
        axis.bar([value + width / 2 for value in positions], top3, width, label="Top-3")
        axis.set_xticks(positions, prediction_labels, rotation=20, ha="right")
        axis.set_ylabel("strict action accuracy")
        axis.set_title("Speculative next-action accuracy (first step skipped)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "top1_top3_action_accuracy.png", dpi=180)
        plt.close(figure)

        selected = prediction_runs[-1]
        prediction = selected["summary"]["prediction"]["paper"]
        selected_key = "top3" if "top3" in prediction else "top1"
        action_names = ["Search", "Lookup", "Finish"]
        _save_bar(
            output_dir / "action_type_accuracy.png",
            f"Action-type strict accuracy ({selected['label']}, {selected_key})",
            action_names,
            [
                prediction[selected_key][action_name]["macro"]
                for action_name in action_names
            ],
            "accuracy",
        )

    comparison = {"runs": runs}
    Utils.save_json(comparison, str(output_dir / "comparison.json"))
    return comparison
