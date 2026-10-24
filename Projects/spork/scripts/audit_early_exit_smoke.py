#!/usr/bin/env python3
"""Read-only audit of an early-exit run, including incomplete experiments.

Outputs a separate report. Never repairs, deletes, or selects source records.
Action differences are diagnostics, not automatically task-quality regressions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def visible_final_answers(record):
    """Compare delivered answer content separately from hidden reasoning text."""
    answers = []
    for turn in record.get("turns", []):
        if not turn.get("baseline_has_tool_call") and turn.get("final_answer") is not None:
            text = turn.get("baseline_text", turn["final_answer"])
            answers.append(text.split("</think>")[-1].strip())
    return answers


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    records = []
    for path in sorted((directory / "records").glob("*.json")):
        record = json.loads(path.read_text())
        records.append((path, record))
    findings, rows = [], []
    grouped = defaultdict(dict)
    for path, record in records:
        grouped[(record["group_id"], record["repeat"])][record["method"]] = record
        for turn in record.get("turns", []):
            identity = f"{path.name}/turn_{turn['turn']}"
            cut = turn.get("exit", {})
            checks = turn.get("checks", [])
            local_findings = []
            if cut.get("fired"):
                if cut.get("stable_count", 0) < turn["exit_config"]["stable_checks"]:
                    local_findings.append("exit_without_required_stability")
                if turn.get("spec_end_ms") is None or turn["spec_end_ms"] > cut["cut_ms"] + .01:
                    local_findings.append("exit_before_tool_ready")
                if not checks or not checks[-1].get("agrees"):
                    local_findings.append("exit_without_last_confirmation")
            if any(b["tokens_at_dispatch"] < a["tokens_at_dispatch"] for a, b in zip(checks, checks[1:])):
                local_findings.append("non_monotone_prefix_positions")
            initial = checks[0].get("call") if checks else None
            current = initial
            redispatches = turn.get("redispatch_log", [])
            if len(redispatches) != turn.get("redispatch_count", 0):
                local_findings.append("redispatch_count_disagrees_with_events")
            for event in redispatches:
                if event.get("previous_call") != current:
                    local_findings.append("redispatch_chain_disagrees_with_previous_call")
                current = event.get("call")
            jobs = turn.get("speculations", [])
            if jobs:
                current = jobs[-1]["call"]
                if len(jobs) - 1 != cut.get("refresh_count", 0):
                    local_findings.append("refresh_count_disagrees_with_executions")
                if sum(bool(j.get("committed")) for j in jobs) > 1:
                    local_findings.append("multiple_speculative_states_committed")
                for index, job in enumerate(jobs):
                    if job.get("superseded") and job.get("committed"):
                        local_findings.append("superseded_result_committed")
                    if job.get("end_ms") is None or job["end_ms"] > turn["turn_end_ms"]:
                        local_findings.append("tool_worker_outlives_turn")
                    if index and jobs[index-1].get("end_ms", float("inf")) > job["dispatch_ms"]:
                        local_findings.append("refresh_started_before_old_tool_finished")
                if bool(turn.get("spec_used")) != bool(jobs[-1].get("committed")):
                    local_findings.append("commit_flag_disagrees_with_reuse")
            emitted = {"name": turn.get("baseline_tool_name"), "arguments": turn.get("baseline_tool_args")}
            if turn.get("spec_used") and current is not None and emitted != current:
                local_findings.append("speculative_result_used_for_different_call")
            for key in ("accounting_errors", "dropped_injections"):
                if (turn.get("d3_status") or {}).get(key, 0) < 0:
                    local_findings.append("invalid_counter")
            findings.extend({"where": identity, "finding": value} for value in local_findings)
            check_details = []
            for c in checks:
                prediction = c.get("prediction") or {}
                check_details.append({"tokens": c["tokens_at_dispatch"], "call": c.get("call"),
                                      "tool_probability": c.get("tool_probability", (c.get("free") or {}).get("tool_probability")),
                                      "stable_count": c.get("stable_count"),
                                      "duration_ms": c.get("end_ms", 0) - c.get("dispatch_ms", c.get("end_ms", 0)),
                                      "d3_registered": prediction.get("d3_registered", False)})
            rows.append({"where": identity, "method": record["method"],
                         "task_wall_s": record["wall_s"], "turn_s": turn.get("turn_end_ms", 0)/1000,
                         "call": emitted, "initial_candidate": initial,
                         "current_candidate": current, "redispatches": redispatches,
                         "speculations": jobs,
                         "speculative_dispatches": len(jobs) if jobs else int(bool(turn.get("spec_dispatched"))) + len(redispatches),
                         "exit": cut, "cot_tokens": turn.get("cot_tokens"),
                         "spec_used": turn.get("spec_used"), "checks": check_details})
    comparisons = []
    for group, modes in grouped.items():
        baseline = modes.get("spork")
        if baseline is None:
            continue
        sig = lambda r: [(t.get("baseline_tool_name"), t.get("baseline_tool_args"),
                          t.get("tool_result_sha256"), t.get("environment_after")) for t in r.get("turns", [])]
        for method, record in modes.items():
            if method == "spork":
                continue
            good = not baseline.get("errors") and not record.get("errors")
            base_answers, other_answers = visible_final_answers(baseline), visible_final_answers(record)
            comparisons.append({"group_id": group[0], "repeat": group[1], "method": method,
                                "valid": good, "baseline_s": baseline["wall_s"], "method_s": record["wall_s"],
                                "speedup": baseline["wall_s"] / record["wall_s"] if good and record["wall_s"] else None,
                                "same_action_result_state": good and sig(baseline) == sig(record),
                                "same_final_answer": [t.get("final_answer") for t in baseline.get("turns", [])]
                                                     == [t.get("final_answer") for t in record.get("turns", [])],
                                "final_answer_observed": bool(base_answers or other_answers),
                                "same_visible_final_answer": (base_answers == other_answers)
                                    if base_answers or other_answers else None,
                                "same_actor_token_ids": [t.get("main_token_ids") for t in baseline.get("turns", [])]
                                                       == [t.get("main_token_ids") for t in record.get("turns", [])]})
    return {"source": str(directory.resolve()), "expected_runs": len(manifest["schedule"]),
            "recorded_runs": len(records), "complete": len(records) == len(manifest["schedule"]),
            "source_sha256": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in records},
            "source_failures": [{"group_id": r["group_id"], "method": r["method"], "errors": r["errors"]}
                                for _, r in records if r.get("errors")],
            "integrity_findings": findings, "turns": rows, "comparisons": comparisons,
            "limitations": ["Partial/capped fixed-user trajectories cannot establish official tau2 success.",
                            "Changed actions require task-level evaluation; exact agreement is not a necessary condition for approximate-method quality.",
                            "Probe microbenchmarks used a candidate generated at the same prefix: they test verification cost, not future full-Actor accuracy.",
                            "For treatment turns main_end_ms includes waiting for checks; use total wall time, not this field, for speed claims."]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("recorded_runs", "expected_runs", "complete", "source_failures", "integrity_findings", "comparisons")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
