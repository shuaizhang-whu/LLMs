#!/usr/bin/env python3
"""Read-only audit of sealed early-exit records, including incomplete experiments.

Outputs a separate report. Never repairs, deletes, or selects source records.
Action differences are diagnostics, not automatically task-quality regressions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def cap_status(record):
    explicit = record.get("turn_limit_reached")
    if isinstance(explicit, bool):
        return explicit
    reason = record.get("termination_reason")
    if reason in {"user_stop", "agent_stop"}:
        return False
    if reason == "max_steps":
        return True
    return None


def public_trajectory(record):
    simulation = record.get("simulation")
    if not isinstance(simulation, dict) or not isinstance(simulation.get("messages"), list):
        return None
    fields = ("role", "content", "tool_calls", "id", "requestor", "error")
    return [{key: message.get(key) for key in fields} for message in simulation["messages"]]


def turn_signature(turn):
    return (turn.get("baseline_tool_name"), turn.get("baseline_tool_args"),
            turn.get("tool_result_sha256"), turn.get("environment_after"))


def user_echoes(record):
    """Flag exact adjacent echoes for review without changing quality scores."""
    messages = (record.get("simulation") or {}).get("messages", [])
    echoes = []
    for index, message in enumerate(messages):
        if (index == 0 or message.get("role") != "user"
                or messages[index - 1].get("role") != "assistant"
                or not message.get("content")
                or message["content"] != messages[index - 1].get("content")):
            continue
        choices = (message.get("raw_data") or {}).get("choices") or []
        choice = choices[0] if choices else None
        echoes.append({
            "message_index_zero_based": index,
            "characters": len(message["content"]),
            "raw_model_content_matches": ((choice.get("message") or {}).get("content") == message["content"]
                                          if choice is not None else None),
            "finish_reason": choice.get("finish_reason") if choice is not None else None,
        })
    return echoes


def audit(directory, reference="spork"):
    manifest_bytes = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != json.loads(
            (directory / "manifest_sha256.json").read_text())["sha256"]:
        raise ValueError("Manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    record_key = lambda r: (r["group_id"], r["repeat"], r["method"])
    scheduled = {record_key(r) for r in manifest["schedule"]}
    if len(scheduled) != len(manifest["schedule"]):
        raise ValueError("Duplicate entries in experiment schedule")
    observed, source_hashes, unsealed = set(), {}, []
    records = []
    for path in sorted((directory / "records").glob("*.json")):
        if path.name.endswith(".complete.json"):
            continue
        seal = path.with_suffix(".complete.json")
        if not seal.is_file():
            unsealed.append(str(path.relative_to(directory)))
            continue
        raw = path.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != json.loads(seal.read_text())["sha256"]:
            raise ValueError(f"Completed result hash mismatch: {path}")
        record = json.loads(raw)
        key = record_key(record)
        if key in observed or key not in scheduled:
            raise ValueError(f"Duplicate or unscheduled completed record: {path}")
        observed.add(key)
        source_hashes[str(path.relative_to(directory))] = actual_hash
        records.append((path, record))
    findings, rows = [], []
    grouped = defaultdict(dict)
    for path, record in records:
        grouped[(record["group_id"], record["repeat"])][record["method"]] = record
        public = public_trajectory(record) or []
        tool_errors = {message.get("id"): message.get("content") for message in public
                       if message["role"] == "tool" and message.get("error") is True}
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
                if turn["exit_config"].get("defer_forced_checks"):
                    k = turn["exit_config"]["stable_checks"]
                    window = checks[-k:]
                    if len(window) != k or any(
                            ("prediction" not in c and "proposal_index" not in c)
                            or c.get("call") is None
                            or c["call"] != window[-1].get("call") for c in window):
                        local_findings.append("deferred_exit_without_verified_consecutive_window")
            if any(b["tokens_at_dispatch"] < a["tokens_at_dispatch"] for a, b in zip(checks, checks[1:])):
                local_findings.append("non_monotone_prefix_positions")
            initial = checks[0].get("call") if checks else None
            emitted = {"name": turn.get("baseline_tool_name"), "arguments": turn.get("baseline_tool_args")}
            executions = turn.get("speculative_executions", [])
            active = executions[-1].get("call") if executions else initial
            if turn.get("spec_used") and active is not None and emitted != active:
                local_findings.append("speculative_result_used_for_different_call")
            if executions:
                committed = [e for e in executions if e.get("committed")]
                if len(committed) != int(bool(turn.get("spec_used"))):
                    local_findings.append("incorrect_speculative_commit_count")
                if any(e["call"] != emitted for e in committed):
                    local_findings.append("committed_obsolete_tool")
                if len(executions) > 2 or any(e.get("end_ms") is None for e in executions):
                    local_findings.append("unbounded_or_undrained_speculative_tools")
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
                         "exit": cut, "cot_tokens": turn.get("cot_tokens"),
                         "spec_used": turn.get("spec_used"), "checks": check_details,
                         "active_candidate": active,
                         # ETA permits a different free continuation. Record this
                         # as changed behavior, not as an invalid cache commit.
                         "continuation_changed_candidate": bool(cut.get("fired")
                             and not cut.get("rolled_back") and emitted["name"]
                             and active is not None and emitted != active),
                         "canonical_tool_error": tool_errors.get(f"spork-turn-{turn['turn']}")})
    comparisons = []
    for group, modes in grouped.items():
        baseline = modes.get(reference)
        if baseline is None:
            continue
        sig = lambda r: [turn_signature(t) for t in r.get("turns", [])]
        for method, record in modes.items():
            if method == reference:
                continue
            good = not baseline.get("errors") and not record.get("errors")
            bt, mt = baseline.get("turns", []), record.get("turns", [])
            bp, mp = public_trajectory(baseline), public_trajectory(record)
            prefix = None
            if good and bt and mt and all(t.get("input_prompt_sha256") for t in bt + mt):
                prefix = 0
                for before, after in zip(bt, mt):
                    if (before["input_prompt_sha256"] != after["input_prompt_sha256"]
                            or turn_signature(before) != turn_signature(after)):
                        break
                    prefix += 1
            comparisons.append({"group_id": group[0], "repeat": group[1], "method": method,
                                "valid": good, "baseline_method": reference,
                                "baseline_s": baseline["wall_s"], "method_s": record["wall_s"],
                                "baseline_capped": cap_status(baseline),
                                "method_capped": cap_status(record),
                                "speedup": baseline["wall_s"] / record["wall_s"] if good and record["wall_s"] else None,
                                "same_action_result_state": good and sig(baseline) == sig(record),
                                "same_public_trajectory": bp == mp if bp is not None and mp is not None else None,
                                "same_input_action_result_state_prefix_turns": prefix,
                                "matched_prefix_baseline_s": sum(t["turn_end_ms"] for t in bt[:prefix])/1000 if prefix is not None else None,
                                "matched_prefix_method_s": sum(t["turn_end_ms"] for t in mt[:prefix])/1000 if prefix is not None else None,
                                "same_final_answer": [t.get("final_answer") for t in baseline.get("turns", [])]
                                                     == [t.get("final_answer") for t in record.get("turns", [])]})
    methods = list(dict.fromkeys(r["method"] for r in manifest["schedule"]))
    common = {key: modes for key, modes in grouped.items()
              if all(m in modes and not modes[m].get("errors") for m in methods)}
    # Unknown cap status is not evidence of a naturally ended trajectory.
    uncapped = {key: modes for key, modes in common.items()
                if all(cap_status(modes[m]) is False for m in methods)}

    echo_diagnostics = [{"record": path.name, "method": record["method"],
                         "group_id": record["group_id"], "echoes": echoes}
                        for path, record in records if (echoes := user_echoes(record))]
    quality_summary = {}
    for method in methods:
        selected = [record for _, record in records if record["method"] == method]
        evaluated = [record["quality"]["reward_info"] for record in selected
                     if record.get("quality", {}).get("reward_info") is not None
                     and not record["quality"].get("error")]
        assertions = [record.get("quality", {}).get("assertion_diagnostic", {}).get("all_met")
                      for record in selected]
        counters = [record["d3_counters"] for record in selected if "d3_counters" in record]
        quality_summary[method] = {
            "runs": len(selected), "runtime_failures": sum(bool(r.get("errors")) for r in selected),
            "quality_evaluated": len(evaluated),
            "local_reward_passes": sum(value.get("reward") == 1 for value in evaluated),
            "assertions_evaluated": sum(value is not None for value in assertions),
            "all_assertions_passes": sum(value is True for value in assertions),
            "quality_errors": [r["quality"]["error"] for r in selected if r.get("quality", {}).get("error")],
            "canonical_tool_errors": sum(message["role"] == "tool" and message.get("error") is True
                                         for r in selected for message in public_trajectory(r) or []),
            "continuation_candidate_mismatches": sum(row["continuation_changed_candidate"]
                                                     for row in rows if row["method"] == method),
            "user_echo_records": sum(row["method"] == method for row in echo_diagnostics),
            "user_echo_messages": sum(len(row["echoes"]) for row in echo_diagnostics
                                      if row["method"] == method),
            "d3_counter_records": len(counters),
            "d3_counters": {key: sum(value.get(key, 0) for value in counters) for key in
                            ("verified_injections", "verified_draft_tokens", "accepted_draft_tokens",
                             "dropped_injections", "prefix_mismatches", "accounting_errors")}}

    def cohort(selected):
        return {"n_task_repeats": len(selected),
                "n_unique_tasks": len({key[0] for key in selected}),
                "groups": [{"group_id": key[0], "repeat": key[1]} for key in sorted(selected)],
                "mean_wall_s": {m: statistics.mean(v[m]["wall_s"] for v in selected.values())
                                if selected else None for m in methods}}

    if (directory / "code").is_dir():
        for name, expected in manifest["code_sha256"].items():
            path = directory / "code" / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                findings.append({"where": name, "finding": "source_snapshot_hash_mismatch"})
    if manifest.get("code_archive_sha256"):
        archive = directory / "code_snapshot.zip"
        if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["code_archive_sha256"]:
            findings.append({"where": archive.name, "finding": "source_archive_hash_mismatch"})
        else:
            try:
                with zipfile.ZipFile(archive) as source:
                    for name, expected in manifest["code_sha256"].items():
                        if name not in source.namelist() or hashlib.sha256(source.read(name)).hexdigest() != expected:
                            findings.append({"where": name, "finding": "source_snapshot_hash_mismatch"})
            except zipfile.BadZipFile:
                findings.append({"where": archive.name, "finding": "invalid_source_archive"})
    return {"source": str(directory.resolve()), "reference": reference,
            "expected_runs": len(manifest["schedule"]),
            "recorded_runs": len(records), "complete": observed == scheduled,
            "source_sha256": source_hashes, "unsealed_records_excluded": unsealed,
            "quality_summary": quality_summary,
            "user_echo_diagnostics": echo_diagnostics,
            "user_echo_scope": "Exact adjacent assistant-to-user copies are flagged for transcript review, not automatically classified as simulator failures or removed from cohorts. Short acknowledgments may be legitimate. Raw-content matching distinguishes generated echoes from later transcript copying when raw responses are available.",
            "quality_scope": "Local reward, assertion grading, canonical tool errors, and runtime errors are separate diagnostics. A local reward pass is not proof that every policy assertion passed. Prefix mismatches can represent safely skipped D3 drafts; they are not automatically invalid commits.",
            "runtime_cohorts": {"common_error_free": cohort(common),
                                "common_error_free_uncapped": cohort(uncapped),
                                "scope": "Both cohorts require every scheduled method. The first includes capped trajectories; the second requires explicit uncapped status or a normal interactive termination. Neither implies task success, and differing actions remain included."},
            "source_failures": [{"group_id": r["group_id"], "method": r["method"], "errors": r["errors"]}
                                for _, r in records if r.get("errors")],
            "integrity_findings": findings, "turns": rows, "comparisons": comparisons,
            "limitations": ["Neither fixed-user runs nor this diagnostic with a local user/judge establish official tau2 success.",
                            "Changed actions require task-level evaluation; exact agreement is not a necessary condition for approximate-method quality.",
                            "Probe microbenchmarks used a candidate generated at the same prefix: they test verification cost, not future full-Actor accuracy.",
                            "For treatment turns main_end_ms includes waiting for checks; use total wall time, not this field, for speed claims."]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", default="spork", help="Named method to compare against, e.g. eta")
    args = p.parse_args()
    report = audit(args.results, reference=args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("recorded_runs", "expected_runs", "complete", "source_failures", "integrity_findings", "runtime_cohorts", "quality_summary", "comparisons")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
