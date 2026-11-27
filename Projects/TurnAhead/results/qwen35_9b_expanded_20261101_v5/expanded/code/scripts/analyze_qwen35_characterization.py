#!/usr/bin/env python3
"""Describe all sealed trials without treating trajectory drift as acceleration."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.evaluation_artifacts import planned_records, visible_final_answers, write_json


def trajectory(record):
    return ([t["main_token_ids"] for t in record["turns"]],
            [{k: t.get(k) for k in ("baseline_has_tool_call", "baseline_tool_name",
              "baseline_tool_args", "tool_result_sha256", "environment_after")}
             for t in record["turns"]],
            visible_final_answers(record), bool(record.get("turn_limit_reached")))


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def analyze(output):
    manifest = json.loads((output / "manifest.json").read_text())
    records, unsealed = planned_records(output, manifest)
    if unsealed or len(records) != len(manifest["schedule"]):
        raise ValueError(f"Incomplete characterization: {len(records)}/{len(manifest['schedule'])}")
    groups = collections.defaultdict(dict)
    repeats = collections.defaultdict(list)
    for r in records:
        groups[(r["group_id"], r["repeat"])][r["config"]] = r
        repeats[(r["group_id"], r["config"])].append(r)
    common = [g for g in groups.values() if set(g) == set(manifest["configs"])
              and all(not r["errors"] for r in g.values())]
    exact_common = [g for g in common if all(trajectory(r) == trajectory(g["spork"])
                                           for r in g.values())]
    modes = {}
    for mode in manifest["configs"]:
        chosen = [r for r in records if r["config"] == mode]
        good = [r for r in chosen if not r["errors"]]
        modes[mode] = {
            "runs": len(chosen), "failed_runs": len(chosen) - len(good),
            "mean_all_attempts_s": mean(r["wall_s"] for r in chosen),
            "mean_common_error_free_s": mean(g[mode]["wall_s"] for g in common),
            "mean_exact_common_s": mean(g[mode]["wall_s"] for g in exact_common),
            "adopted_children": sum(r["prefetch"].get("adopted", 0) for r in chosen),
            "early_children": sum(p.get("child_launch_phase") == "before_parent_completion"
                                  for r in chosen for p in r["prefetch"].get("pairs", [])),
            "observed_wasted_child_tokens": sum(r["prefetch"].get("observed_wasted_child_tokens", 0)
                                                for r in chosen),
            "turn_limits": sum(bool(r.get("turn_limit_reached")) for r in good),
        }
    comparisons = []
    for (group, repeat), g in sorted(groups.items()):
        if "spork" not in g:
            continue
        base = g["spork"]
        for mode, r in g.items():
            if mode == "spork":
                continue
            good = not base["errors"] and not r["errors"]
            a, b = trajectory(base), trajectory(r)
            comparisons.append({"group_id": group, "repeat": repeat, "config": mode,
                "both_error_free": good,
                "same_actor_tokens": a[0] == b[0] if good else None,
                "same_actions_results_state": a[1] == b[1] if good else None,
                "same_visible_answer": a[2] == b[2] if good else None,
                "same_turn_limit": a[3] == b[3] if good else None,
                "same_full_trajectory": a == b if good else None,
                "baseline_s": base["wall_s"], "method_s": r["wall_s"],
                "observed_wall_difference_s": base["wall_s"] - r["wall_s"],
                "baseline_completed_tokens": sum(map(len, a[0])),
                "method_completed_tokens": sum(map(len, b[0]))})
    repeated = []
    for (group, mode), rs in sorted(repeats.items()):
        rs.sort(key=lambda r: r["repeat"])
        if len(rs) < 2:
            continue
        good = all(not r["errors"] for r in rs)
        repeated.append({"group_id": group, "config": mode, "runs": len(rs),
            "all_error_free": good,
            "same_full_trajectory": all(trajectory(r) == trajectory(rs[0]) for r in rs[1:]) if good else None})
    counts = {mode: {"error_free_pairs": len(good := [r for r in comparisons
                if r["config"] == mode and r["both_error_free"]]),
                **{key + "_count": sum(r[key] for r in good) for key in (
                    "same_actor_tokens", "same_actions_results_state", "same_visible_answer", "same_full_trajectory")}}
              for mode in manifest["configs"] if mode != "spork"}
    result = {"status": "diagnostic_only", "recorded": len(records),
        "expected": len(manifest["schedule"]), "common_error_free_groups": len(common),
        "exact_common_groups": len(exact_common), "modes": modes,
        "comparison_counts": counts, "comparisons": comparisons, "same_seed_repeats": repeated,
        "errors": [{"group_id": r["group_id"], "repeat": r["repeat"], "config": r["config"],
                    "errors": r["errors"]} for r in records if r["errors"]],
        "d3_error_counters": {k: sum(r["d3_counters"].get(k, 0) for r in records)
                              for k in ("accounting_errors", "dropped_injections")},
        "scope": "All training trials retained. Wall differences on changed trajectories are not equivalent-work speedups. No official interactive tau2 success claim.",
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output / "characterization.json", result)
    lines = ["# Qwen3.5 diagnostic characterization", "", result["scope"], "",
             f"Completed: {len(records)}/{len(manifest['schedule'])}; error-free common groups: {len(common)}; exact common groups: {len(exact_common)}.", "",
             "| Config | Runs | Errors | Common error-free mean seconds | Adopted children |",
             "|---|---:|---:|---:|---:|"]
    for mode, r in modes.items():
        value = r['mean_common_error_free_s']
        lines.append(f"| {mode} | {r['runs']} | {r['failed_runs']} | {value:.3f} | {r['adopted_children']} |"
                     if value is not None else f"| {mode} | {r['runs']} | {r['failed_runs']} | n/a | {r['adopted_children']} |")
    lines += ["", "A passing strict preflight is still required for the formal ablation. This report does not override that gate."]
    (output / "characterization.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.output)
    print(json.dumps({k: report[k] for k in ("status", "recorded", "common_error_free_groups",
          "exact_common_groups", "modes", "comparison_counts")}, indent=2))
