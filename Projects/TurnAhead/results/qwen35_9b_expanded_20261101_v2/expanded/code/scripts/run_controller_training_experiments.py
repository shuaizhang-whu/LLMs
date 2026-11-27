#!/usr/bin/env python3
"""Offline training comparison on the completed, frozen stage-3 dataset.

Default: full input vs masked current reasoning, 3 seeds, 20 epochs per fit.
No Actor loading, vLLM requests, data collection, or test-set evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_timing_controller import run as train, write_json  # noqa: E402
from spork_core.controller_data import load_controller_rows, partition_controller_rows  # noqa: E402
from spork_core.controller_evaluation import (  # noqa: E402
    bootstrap_task_difference, describe_labels, fit_simple_baselines, score_predictions,
)

VARIANTS = {"full": "none", "masked_reasoning": "mask_reasoning"}
CODE_FILES = ["scripts/run_controller_training_experiments.py",
              "scripts/run_controller_training_experiments.sh", "scripts/train_timing_controller.py",
              "spork_core/controller_data.py", "spork_core/timing_controller.py",
              "spork_core/controller_evaluation.py"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(dataset, encoder, max_length):
    """Check the existing frozen exports; never replay or rewrite collection."""
    dataset, encoder = Path(dataset).resolve(), Path(encoder).resolve()
    manifest, gate = read(dataset / "manifest.json"), read(dataset / "training_data_gate.json")
    if gate.get("passed") is not True or not gate.get("checks") or not all(gate["checks"].values()):
        raise ValueError("The dataset's declared training gate did not pass")
    split_path = dataset / "task_split.json"
    if sha(split_path) != manifest["task_split_sha256"]:
        raise ValueError("The frozen task split changed")
    assignment = read(split_path)
    owners = {t["group_id"]: t["split"] for t in assignment["tasks"]}
    if len(owners) != len(assignment["tasks"]):
        raise ValueError("Duplicate groups in the frozen task split")
    reports, chunks = [], []
    for i in range(len(manifest["batches"])):
        batch = dataset / f"batch_{i:03d}"
        report = read(batch / "collection_report.json")
        raw = (batch / "training_rows.jsonl").read_bytes()
        if not report["collection_complete"] or hashlib.sha256(raw).hexdigest() != report["training_rows_sha256"]:
            raise ValueError(f"Incomplete/changed frozen batch {i}")
        chunks.append(raw)
        reports.append(report)
    data = dataset / "training_rows.jsonl"
    raw = data.read_bytes()
    if not chunks or raw != b"".join(chunks):
        raise ValueError("Merged training rows disagree with the completed batch exports")
    raw_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    for row in raw_rows:
        if row.get("split") not in ("train", "validation") or owners.get(row["group_id"]) != row["split"]:
            raise ValueError("Unexpected/unassigned/test row in development data")
    rows, data_report = load_controller_rows([data], "matched")
    train_rows, val_rows, test_rows, split_report = partition_controller_rows(rows)
    if test_rows or len(rows) != gate["usable_prefixes"] or len(raw_rows) != gate["recorded_prefixes"]:
        raise ValueError("Training gate row counts disagree with the current exports")
    groups = {s: len({r["group_id"] for r in rs}) for s, rs in (("train", train_rows), ("validation", val_rows))}
    if groups != gate["usable_groups"]:
        raise ValueError("Training gate group counts disagree with the current exports")
    for name in ("config.json", "model.safetensors", "vocab.txt"):
        if not (encoder / name).is_file():
            raise ValueError(f"Missing local encoder file: {encoder / name}")
    config = read(encoder / "config.json")
    if config.get("model_type") not in ("bert", "distilbert") or not 32 <= max_length <= config["max_position_embeddings"]:
        raise ValueError("Unsupported local encoder or input length")
    audits = [a for r in reports for a in r["per_prefix_audits"]]
    descriptions = {"train": describe_labels(train_rows), "validation": describe_labels(val_rows)}
    for s in descriptions:
        good = [a for a in audits if a["label_usable"] and a["split"] == s]
        intervals = [a["nominal_gain_intervals_s"]["now_minus_wait"] for a in good]
        descriptions[s]["nominal_now_wait_intervals_excluding_zero"] = sum(
            v["lower"] > 0 or v["upper"] < 0 for v in intervals)
        descriptions[s]["labels_meeting_collection_precision_criterion"] = sum(
            r["metadata"].get("precision_passed") is True for r in (train_rows if s == "train" else val_rows))
    # The loader only checks schema/quality. Stability came from the prior
    # collector; record that separately rather than relabeling the loader check.
    data_report["prior_collection_training_gate"] = gate
    data_report["label_stability_source"] = "completed three-repeat collector reports; raw runs not reexecuted"
    data_report["label_descriptions"] = descriptions
    data_report["reserved_test_tasks"] = sum(s == "test" for s in owners.values())
    data_report["collector_settings"] = {k: manifest.get(k) for k in (
        "model", "tool_floor_s", "prefix_tokens", "repetitions", "max_turns_per_task", "replay_config", "collection_scope")}
    return train_rows, val_rows, data_report, split_report


def comparison_summary(val_rows, baselines, runs, seeds):
    summary = {"scope": "Development validation comparison, used for checkpoint selection; not final test or online speedup.",
               "seed_policy": "Report every predeclared seed, never choose the best seed or ensemble its predictions.",
               "primary_metric": "mean recorded gain per task, with equal weight for each validation task",
               "baselines": baselines["metrics"], "variants": {}, "runs": runs,
               "test_evaluated": False, "online_performance_evaluated": False,
               "controller_overhead_included": False, "deployment_checkpoint_selected": False}
    for variant in VARIANTS:
        subset = [r for r in runs if r["variant"] == variant]
        if [r["seed"] for r in subset] != list(seeds):
            raise ValueError("Missing/reordered seeds in the comparison")
        values = [r["metrics"]["task_mean_recorded_gain_s"] for r in subset]
        task_keys = sorted(subset[0]["metrics"]["task_gains_s"])
        task_average = {g: statistics.mean(r["metrics"]["task_gains_s"][g] for r in subset) for g in task_keys}
        contrasts = {}
        for name, baseline in baselines["metrics"].items():
            contrasts[name] = bootstrap_task_difference({g: task_average[g] - baseline["task_gains_s"][g]
                                                        for g in task_keys})
        summary["variants"][variant] = {
            "seeds": list(seeds), "mean_task_gain_s": statistics.mean(values),
            "std_task_gain_across_seeds_s": statistics.stdev(values) if len(values) > 1 else 0.,
            "mean_mse_seconds_squared": statistics.mean(r["metrics"]["mse_seconds_squared"] for r in subset),
            "last_epoch_mean_task_gain_s": statistics.mean(r["last_epoch_task_gain_s"] for r in subset),
            "paired_differences_vs_baselines_s": contrasts, "seed_averaged_task_gains_s": task_average,
            "selected_policy_counts_per_seed": {str(r["seed"]): r["metrics"]["selected_policy_counts"] for r in subset},
        }
    full, masked = (summary["variants"][v]["seed_averaged_task_gains_s"] for v in VARIANTS)
    summary["full_minus_masked_reasoning_s"] = bootstrap_task_difference({g: full[g] - masked[g] for g in full})
    summary["validation_label_distribution"] = describe_labels(val_rows)
    return summary


def markdown_summary(summary, data_report):
    d = data_report["label_descriptions"]
    lines = ["# 启动时机控制器训练对照", "",
             f"训练：{d['train']['rows']} 个前缀 / {d['train']['tasks']} 个任务；"
             f"验证：{d['validation']['rows']} 个前缀 / {d['validation']['tasks']} 个任务。", "",
             "以下是用于选模型的验证集结果，收益来自已采集标签，不含控制器耗时，也不是在线加速结果。", "",
             "| 方法 | 每任务平均收益（秒） | 相对固定等待（秒） | 随机种子间标准差 |",
             "|---|---:|---:|---:|"]
    fixed_wait = summary["baselines"]["fixed_wait"]["task_mean_recorded_gain_s"]
    names = {"fixed_off": "固定不启动", "fixed_now": "固定现在启动", "fixed_wait": "固定等待",
             "train_best_constant": "训练集选择的固定策略", "visible_history_bucket": "工具集合 + 是否已有工具结果"}
    for name, metric in summary["baselines"].items():
        gain = metric["task_mean_recorded_gain_s"]
        lines.append(f"| {names[name]} | {gain:.6f} | {gain - fixed_wait:+.6f} | — |")
    for name, metric in summary["variants"].items():
        lines.append(f"| {'完整输入' if name == 'full' else '屏蔽当前思考内容'} | {metric['mean_task_gain_s']:.6f} | "
                     f"{metric['mean_task_gain_s'] - fixed_wait:+.6f} | {metric['std_task_gain_across_seeds_s']:.6f} |")
    gap = summary["full_minus_masked_reasoning_s"]
    lines += ["", f"完整输入相对屏蔽思考：{gap['mean_s']:+.6f} 秒；按任务配对重采样区间 "
              f"[{gap['lower_s']:+.6f}, {gap['upper_s']:+.6f}]。",
              "该区间仅描述当前验证任务；没有校正选 epoch 的偏差，也没有消除原始计时噪声。", "",
              "屏蔽思考的对照保留原 token 位置、长度和其他字段预算，检查的是思考文本内容的增量。",
              "各轮指标、各随机种子的决策数量、最后一轮结果、逐前缀预测与检查点见 summary.json 和各训练子目录。",
              "没有选择表现最好的随机种子，也没有读取保留测试任务。"]
    if d["validation"]["best_recorded_policy_counts"]["off"] == 0:
        lines += ["", "当前验证集没有“不启动”为最优的标签，不能据此评价控制器识别该类状态的能力。"]
    return "\n".join(lines) + "\n"


def run(args):
    train_rows, val_rows, data_report, split_report = preflight(args.dataset, args.encoder, args.max_length)
    if args.check_only:
        print(json.dumps({"passed": True, "training_started": False,
                          "labels": data_report["label_descriptions"],
                          "reserved_test_tasks": data_report["reserved_test_tasks"]}, ensure_ascii=False, indent=2))
        return None
    if args.output is None:
        raise ValueError("--output is required for training")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "data_report.json", data_report)
    write_json(output / "split.json", split_report)
    manifest = {"purpose": "offline_controller_training_comparison",
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "variants": VARIANTS, "seeds": args.seeds, "epochs_per_fit": args.epochs,
                "checkpoint_selection": "max validation task-mean gain, then min validation MSE, then earlier epoch",
                "data_sha256": sha(args.dataset / "training_rows.jsonl"),
                "task_split_sha256": sha(args.dataset / "task_split.json"),
                "encoder_sha256": {n: sha(args.encoder / n) for n in ("config.json", "model.safetensors", "vocab.txt")},
                "code_sha256": {n: sha(ROOT / n) for n in CODE_FILES},
                "packages": {n: version(n) for n in ("torch", "transformers", "tokenizers", "safetensors")},
                "vllm_used": False, "actor_loaded_or_modified": False, "test_tasks_used": False,
                "validation_previously_examined": True,
                "prior_fit_path": (args.dataset / "controller_training_path.txt").read_text().strip()
                if (args.dataset / "controller_training_path.txt").exists() else None}
    write_json(output / "manifest.json", manifest)
    for name in CODE_FILES:
        target = output / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    baselines, _ = fit_simple_baselines(train_rows, val_rows)
    write_json(output / "baselines.json", baselines)
    print(f"训练 {len(train_rows)} / 验证 {len(val_rows)}，固定任务划分；输出 {output}", flush=True)
    runs = []
    for variant, ablation in VARIANTS.items():
        for seed in args.seeds:
            if sha(args.dataset / "training_rows.jsonl") != manifest["data_sha256"]:
                raise ValueError("Dataset changed during the comparison; prior results are preserved")
            name = f"{variant}_seed_{seed}"
            print(f"开始 {name}，{args.epochs} epochs，device={args.device}", flush=True)
            cfg = argparse.Namespace(mode="train", data=[args.dataset / "training_rows.jsonl"],
                                     output=output / name, encoder=args.encoder, device=args.device,
                                     quality_policy="matched", validation_fraction=.2, seed=seed,
                                     max_length=args.max_length, head_hidden_size=128, dropout=.1,
                                     input_ablation=ablation, checkpoint_selection="gain", report_training_metrics=True,
                                     epochs=args.epochs, batch_size=args.batch_size, encoder_lr=2e-5,
                                     head_lr=1e-3, weight_decay=.01, cpu_threads=args.cpu_threads)
            report = train(cfg)
            trained_source = read(cfg.output / "data_report.json")["sources"][0]["sha256"]
            if trained_source != manifest["data_sha256"]:
                raise ValueError("A fit used a different data snapshot; results cannot be pooled")
            records = [json.loads(line) for line in (cfg.output / report["selected_predictions"]).read_text().splitlines()]
            if [r["id"] for r in records] != [r["id"] for r in val_rows]:
                raise ValueError("Saved validation prediction IDs disagree with frozen rows")
            metrics = score_predictions(val_rows, [[r["q_now_s"], r["q_wait_s"]] for r in records])
            runs.append({"variant": variant, "seed": seed, "directory": name,
                         "checkpoint": str(Path(name) / report["best_checkpoint"]),
                         "selected_epoch": report["selected_validation_metrics"]["epoch"],
                         "metrics": metrics,
                         "last_epoch_task_gain_s": report["last_validation_metrics"]["task_mean_recorded_gain_s"]})
            write_json(output / "completed_runs.json", runs)
    summary = comparison_summary(val_rows, baselines, runs, args.seeds)
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(markdown_summary(summary, data_report))
    print(f"全部训练结束：{output / 'summary.md'}", flush=True)
    return summary


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--check-only", action="store_true", help="Short CPU preflight; does not initialize or train any model")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()
    args.dataset, args.encoder = args.dataset.resolve(), args.encoder.resolve()
    if len(set(args.seeds)) != len(args.seeds) or min(args.epochs, args.batch_size, args.cpu_threads) <= 0:
        ap.error("Seeds must be distinct and training sizes positive")
    if args.output is None and not args.check_only:
        ap.error("Specify --output or --check-only")
    return args


if __name__ == "__main__":
    run(parse_args())
