#!/usr/bin/env python3
"""Audit all formal outcomes, then summarize paired training-task latency."""
import argparse
import collections
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.evaluation_artifacts import planned_records, digest, visible_final_answers, write_json


def signature(record):
    fields = ("main_token_ids", "baseline_text", "prompt_token_ids", "generation_settings",
              "baseline_has_tool_call", "baseline_tool_name", "baseline_tool_args",
              "tool_result_sha256", "environment_after")
    turns = [{k: t.get(k) for k in fields} for t in record["turns"]]
    failed = record.get("failed_generation")
    if failed:
        failed = {k: failed.get(k) for k in ("turn", "prompt_token_ids", "token_ids", "text",
                                             "finish_reason", "generation_settings")}
    return (turns, failed, record["errors"], visible_final_answers(record),
            bool(record.get("turn_limit_reached")))


def analyze(output):
    manifest = json.loads((output / "manifest.json").read_text())
    stage = json.loads((output / "stage_check.json").read_text())
    if not stage["passed"] or stage["stage"] != "ablation" or not manifest["require_identical_trajectories"]:
        raise ValueError("A completed strict ablation gate is required")
    records, unsealed = planned_records(output, manifest)
    if len(records) != len(manifest["schedule"]) or unsealed:
        raise ValueError("Incomplete or unsealed formal records")
    for rel, expected in manifest["code_sha256"].items():
        if digest(output / "code" / rel) != expected:
            raise ValueError(f"Frozen source mismatch: {rel}")
    for rel, expected in manifest.get("external_parser_source_sha256", {}).items():
        if digest(output / "external_code/vllm" / rel) != expected:
            raise ValueError(f"Frozen parser mismatch: {rel}")
    groups, repeated = collections.defaultdict(dict), collections.defaultdict(list)
    for row in records:
        if row["config"] == "raw_no_probe":
            for pair in row["prefetch"]["pairs"]:
                if pair.get("child_started") and (
                    pair.get("child_prediction_policy") != "off"
                    or pair.get("child_prediction_requests")):
                    raise ValueError("The no-probe child executed an additional prediction")
        for turn in row["turns"]:
            if not turn.get("prompt_token_ids") or not turn.get("generation_settings"):
                raise ValueError("Missing successful-turn input/generation provenance")
        if row["errors"] and not row.get("failed_generation"):
            raise ValueError("Cannot verify failed outcome without its generation trace")
        groups[(row["group_id"], row["repeat"])][row["config"]] = row
        repeated[(row["group_id"], row["config"])].append(row)
    for key, group in groups.items():
        if set(group) != set(manifest["configs"]):
            raise ValueError(f"Missing method in {key}")
        if any(signature(r) != signature(group["spork"]) for r in group.values()):
            raise ValueError(f"Outcome or request-provenance divergence in {key}")
    for key, rows in repeated.items():
        if any(signature(r) != signature(rows[0]) for r in rows[1:]):
            raise ValueError(f"Repeated outcome changed: {key}")
    good = [g for g in groups.values() if all(not r["errors"] for r in g.values())]
    per_task = collections.defaultdict(list)
    for group in good:
        per_task[group["spork"]["group_id"]].append(group)
    task_means = [{"group_id": task, "repeats": len(gs), **{
        c: statistics.mean(g[c]["wall_s"] for g in gs) for c in manifest["configs"]}}
        for task, gs in sorted(per_task.items())]
    result = {"verified_all_outcomes_and_requests": True, "records": len(records),
        "common_error_free_pairs": len(good), "common_tasks": len(task_means),
        "scope": "Training-only bounded fixed-user trajectories; script completion is not official task success. Eager reference-GDN runtime; no cross-profile speedup claim.",
        "modes": {}, "task_means": task_means}
    for mode in manifest["configs"]:
        chosen = [r for r in records if r["config"] == mode]
        times = [g[mode]["wall_s"] for g in good]
        pairs = [p for r in chosen for p in r["prefetch"]["pairs"]]
        result["modes"][mode] = {
            "runs": len(chosen), "script_failures": sum(bool(r["errors"]) for r in chosen),
            "mean_common_s": statistics.mean(times) if times else None,
            "p95_common_s": float(np.percentile(times, 95)) if times else None,
            "mean_all_attempts_s": statistics.mean(r["wall_s"] for r in chosen),
            "turn_limits": sum(bool(r.get("turn_limit_reached")) for r in chosen),
            "early_children": sum(p.get("child_launch_phase") == "before_parent_completion" for p in pairs),
            "adopted_children": sum(bool(p.get("adopted")) for p in pairs),
            "child_prediction_requests": sum(len(p.get("child_prediction_requests", [])) for p in pairs),
            "child_prediction_output_tokens": sum(
                (q.get("usage") or {}).get("completion_tokens", 0)
                for p in pairs for q in p.get("child_prediction_requests", [])),
            "rejections": dict(collections.Counter(p.get("rejection") for p in pairs if not p.get("adopted"))),
            "observed_wasted_child_tokens_lower_bound": sum(p.get("observed_wasted_child_tokens", 0) for p in pairs),
        }
    contrasts = {}
    for baseline, method in [("spork", "serial"), ("spork", "actor_only"),
                             ("spork", "raw"), ("serial", "actor_only"), ("actor_only", "raw"),
                             ("spork", "raw_no_probe"), ("raw", "raw_no_probe")]:
        if baseline not in manifest["configs"] or method not in manifest["configs"] or not task_means:
            continue
        a = np.array([r[baseline] for r in task_means])
        b = np.array([r[method] for r in task_means])
        item = {"saved_s_per_task": float((a - b).mean()),
                "latency_reduction_percent": float(100 * (1 - b.mean() / a.mean())),
                "speedup": float(a.mean() / b.mean())}
        if len(a) > 1:
            indices = np.random.default_rng(42).integers(0, len(a), size=(5000, len(a)))
            reductions = 100 * (1 - b[indices].mean(1) / a[indices].mean(1))
            item["task_bootstrap_95_percentile_interval"] = np.percentile(reductions, [2.5, 97.5]).tolist()
        contrasts[f"{method}_vs_{baseline}"] = item
    result["paired_contrasts"] = contrasts
    result["uncertainty_scope"] = "Bootstrap clusters by task and averages repeats; tiny selected training cohort, not a generalization or publication-level significance claim."
    write_json(output / "formal_analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
