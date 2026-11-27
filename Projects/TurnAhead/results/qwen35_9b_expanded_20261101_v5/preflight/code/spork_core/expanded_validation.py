"""Frozen task selection and conservative audits for interactive development."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import statistics

from .evaluation_artifacts import digest, planned_records, write_json

CONFIGS = ("spork", "raw", "raw_no_probe", "raw_one_probe")
OLD_TASKS = ("airline:2", "airline:3", "airline:4", "airline:6",
             "retail:0", "retail:1", "retail:3", "retail:4")
NATURAL_STOPS = {"user_stop", "agent_stop"}


def select_tasks(split, seed=20261101):
    by_id = {r["group_id"]: r for r in split["tasks"]}
    if any(by_id[k]["split"] != "train" for k in OLD_TASKS):
        raise ValueError("A previous development task is no longer in train")
    chosen = []
    for domain in ("airline", "retail"):
        old = [by_id[k] for k in OLD_TASKS if by_id[k]["domain"] == domain]
        pool = [r for r in by_id.values() if r["domain"] == domain
                and r["split"] == "train" and r["group_id"] not in OLD_TASKS]
        pool.sort(key=lambda r: hashlib.sha256(f'{seed}:{r["group_id"]}'.encode()).hexdigest())
        if len(pool) < 12:
            raise ValueError("Not enough training tasks for the fixed 16-per-domain plan")
        chosen.extend({**r, "cohort": "previous8"} for r in old)
        chosen.extend({**r, "cohort": "new24"} for r in pool[:12])
    return sorted(chosen, key=lambda r: (r["domain"], int(r["task_id"])))


def normalized_messages(messages):
    return [{"role": m["role"], "content": m.get("content"),
             "error": m.get("error"), "tool_calls": [
                 {k: c.get(k) for k in ("name", "arguments", "requestor")}
                 for c in (m.get("tool_calls") or [])]} for m in messages]


def signature(record):
    fields = ("prompt_token_ids", "generation_settings", "main_token_ids", "baseline_text",
              "baseline_has_tool_call", "baseline_tool_name", "baseline_tool_args",
              "tool_result_sha256", "environment_after")
    users = [{k: c.get(k) for k in ("prompt_token_ids", "generation_settings", "token_ids",
              "text", "finish_reason")} for c in record.get("user_calls", [])]
    reward = record.get("reward_info") or {}
    return {"turns": [{k: t.get(k) for k in fields} for t in record["turns"]],
        "users": users, "messages": record.get("normalized_messages"),
        "final_environment": record.get("final_environment"),
        "termination": record.get("termination_reason"),
        "reward": reward.get("reward"), "reward_breakdown": reward.get("reward_breakdown"),
        "errors": record["errors"], "failed_generation": record.get("failed_generation")}


def verify_frozen(output, manifest, root, *, current=True):
    for rel, h in manifest["code_sha256"].items():
        if digest(output / "code" / rel) != h or (current and digest(root / rel) != h):
            raise ValueError(f"Frozen source changed: {rel}")
    for f, h in manifest["input_sha256"].items():
        if digest(f) != h:
            raise ValueError(f"Frozen input changed: {f}")
    for rel, h in manifest["external_parser_source_sha256"].items():
        if (digest(output / "external_code/vllm" / rel) != h
                or digest(Path(manifest["external_parser_root"]) / rel) != h):
            raise ValueError(f"Parser changed: {rel}")
    for rel, h in manifest["model_config_sha256"].items():
        if digest(Path(manifest["model"]) / rel) != h:
            raise ValueError(f"Model metadata changed: {rel}")
    for f in manifest["absent_optional_model_files"]:
        if (Path(manifest["model"]) / f).exists():
            raise ValueError(f"Optional model metadata appeared: {f}")


def analyze(output, manifest):
    rows, unsealed = planned_records(output, manifest)
    errors = []
    groups = collections.defaultdict(dict)
    repeats = collections.defaultdict(list)
    for r in rows:
        groups[(r["group_id"], r["repeat"])][r["config"]] = r
        repeats[(r["group_id"], r["config"])].append(r)
        if r["errors"]:
            errors.append(f'{r["group_id"]}/r{r["repeat"]}/{r["config"]}: recorded failure')
        for t in r["turns"]:
            if not t.get("prompt_token_ids") or not t.get("generation_settings"):
                errors.append("missing Actor input provenance")
        for c in r.get("user_calls", []):
            if not c.get("prompt_token_ids") or not c.get("token_ids") or c.get("finish_reason") != "stop":
                errors.append("missing or incomplete user provenance")
        if r.get("trajectory_sha256") and digest(output / r["trajectory_path"]) != r["trajectory_sha256"]:
            errors.append("trajectory hash mismatch")
        if not r["errors"]:
            trajectory = json.loads((output / r["trajectory_path"]).read_text())
            if normalized_messages(trajectory["messages"]) != r["normalized_messages"]:
                errors.append("normalized trajectory does not match the saved simulation")
            actors = [m for m in trajectory["messages"] if m["role"] == "assistant"]
            tools = [m for m in trajectory["messages"] if m["role"] == "tool"]
            users = [m["content"] for m in trajectory["messages"] if m["role"] == "user"]
            if len(actors) != len(r["turns"]) + 1:
                errors.append("native Actor message count differs from committed turns")
            for message, turn in zip(actors[1:], r["turns"]):
                if turn["baseline_has_tool_call"]:
                    calls = message.get("tool_calls") or []
                    if (len(calls) != 1 or calls[0]["name"] != turn["baseline_tool_name"]
                            or calls[0]["arguments"] != turn["baseline_tool_args"]):
                        errors.append("native action differs from generated action")
                    elif not any(t["id"] == calls[0]["id"] and hashlib.sha256(t["content"].encode()).hexdigest()
                                 == turn["tool_result_sha256"] for t in tools):
                        errors.append("native tool result differs from adopted observation")
                elif message.get("content") != turn["baseline_text"].split("</think>")[-1].strip():
                    errors.append("native visible answer differs from full Actor output")
            if users != [c["text"].split("</think>")[-1].strip() for c in r["user_calls"]]:
                errors.append("native user messages differ from simulated outputs")
        for pair in r.get("prefetch", {}).get("pairs", []):
            n = len(pair.get("child_prediction_requests", []))
            if r["config"] == "raw_no_probe" and n:
                errors.append("no-probe arm made a child prediction")
            if r["config"] == "raw_one_probe" and n > 1:
                errors.append("one-probe arm exceeded its child prediction budget")
        if any(r.get("d3_counters", {}).get(k, 0) for k in
               ("accounting_errors", "prefix_mismatches", "dropped_injections")):
            errors.append("D3 accounting, prefix, or cleanup error")
    for key, g in groups.items():
        if "spork" in g:
            for mode, r in g.items():
                if signature(r) != signature(g["spork"]):
                    errors.append(f"{key}/{mode}: outcome or request divergence")
    for key, rr in repeats.items():
        if any(signature(r) != signature(rr[0]) for r in rr[1:]):
            errors.append(f"{key}: same-seed repeat divergence")
    complete = len(rows) == len(manifest["schedule"]) and not unsealed
    quality_common = [g for g in groups.values() if set(g) == set(CONFIGS)
                      and all(not r["errors"] for r in g.values())]
    cross_session = [g for g in quality_common if len({r["session_id"] for r in g.values()}) != 1]
    common = [g for g in quality_common if len({r["session_id"] for r in g.values()}) == 1]
    natural = [g for g in common if all(r["termination_reason"] in NATURAL_STOPS for r in g.values())]
    result = {"expected": len(manifest["schedule"]), "recorded": len(rows),
        "complete": complete, "unsealed": unsealed, "gate_errors": sorted(set(errors)),
        "common_pairs": len(common), "common_natural_pairs": len(natural),
        "cross_session_pairs_not_used_for_latency": [
            [g["spork"]["group_id"], g["spork"]["repeat"]] for g in cross_session],
        "scope": manifest["scope"], "methods": {}, "cohorts": {}}
    for mode in CONFIGS:
        rr = [r for r in rows if r["config"] == mode]
        pairs = [p for r in rr for p in r.get("prefetch", {}).get("pairs", [])]
        scored = [r for r in rr if r.get("reward_info") is not None]
        result["methods"][mode] = {
            "recorded": len(rr), "script_failures": sum(bool(r["errors"]) for r in rr),
            "terminations": dict(collections.Counter(r.get("termination_reason") for r in rr)),
            "scored": len(scored),
            "mean_native_reward_scored": statistics.mean(r["reward_info"]["reward"] for r in scored) if scored else None,
            "mean_agent_pipeline_s": statistics.mean(g[mode]["agent_pipeline_s"] for g in common) if common else None,
            "mean_interaction_s": statistics.mean(g[mode]["wall_s"] for g in common) if common else None,
            "mean_user_s": statistics.mean(g[mode]["user_wall_s"] for g in common) if common else None,
            "mean_agent_pipeline_natural_s": statistics.mean(g[mode]["agent_pipeline_s"] for g in natural) if natural else None,
            "early_children": sum(p.get("child_launch_phase") == "before_parent_completion" for p in pairs),
            "delivered_adopted_children": sum(bool(p.get("adopted")) and p.get("native_delivery_count", 0) == 2 for p in pairs),
            "child_predictions": sum(len(p.get("child_prediction_requests", [])) for p in pairs),
            "rejections": dict(collections.Counter(p.get("rejection") for p in pairs if p.get("child_started") and not p.get("adopted"))),
            "received_wasted_child_tokens_lower_bound": sum(p.get("observed_wasted_child_tokens", 0) for p in pairs)
                + sum(e["received_tokens"] for r in rr for e in r.get("abandoned_entries", [])),
        }
    if common:
        import numpy as np
        for cohort in ("all", "previous8", "new24", "airline", "retail"):
            chosen = [g for g in common if cohort == "all" or g["spork"].get("cohort") == cohort
                      or g["spork"]["domain"] == cohort]
            by_task = collections.defaultdict(list)
            for g in chosen:
                by_task[g["spork"]["group_id"]].append(g)
            means = [{"group_id": k, "repeats": len(gs), **{c: statistics.mean(g[c]["agent_pipeline_s"] for g in gs)
                      for c in CONFIGS}} for k, gs in sorted(by_task.items())]
            contrasts = {}
            for base, mode in [("spork", m) for m in CONFIGS if m != "spork"] + [
                    ("raw", "raw_no_probe"), ("raw", "raw_one_probe")]:
                if not means:
                    continue
                x = np.array([r[base] for r in means]); y = np.array([r[mode] for r in means])
                ix = np.random.default_rng(42).integers(0, len(x), (3000, len(x)))
                contrasts[f"{mode}_vs_{base}"] = {
                    "saved_s": float((x-y).mean()), "reduction_percent": float(100*(1-y.mean()/x.mean())),
                    "task_bootstrap_95_interval": np.percentile(100*(1-y[ix].mean(1)/x[ix].mean(1)), [2.5,97.5]).tolist()}
            result["cohorts"][cohort] = {"tasks": len(means), "task_means": means, "contrasts": contrasts}
    write_json(output / "summary.json", result)
    return result


def check_results(output, root):
    manifest = json.loads((output / "manifest.json").read_text())
    verify_frozen(output, manifest, root)
    report = analyze(output, manifest)
    if not report["complete"] or report["gate_errors"]:
        raise ValueError(f'Incomplete or invalid results: {report["recorded"]}/{report["expected"]}; {report["gate_errors"]}')
    if manifest["stage"] == "preflight":
        rows, _ = planned_records(output, manifest)
        if any(r["termination_reason"] not in NATURAL_STOPS or len(r["user_calls"]) < 2 for r in rows):
            raise ValueError("Preflight needs natural completion and continuing user interaction in every run")
        if not report["methods"]["raw"]["delivered_adopted_children"]:
            raise ValueError("Preflight did not exercise successor adoption")
        if not any(r.get("judge_calls") for r in rows):
            raise ValueError("Preflight did not exercise local NL scoring")
    result = {"passed": True, "stage": manifest["stage"], "recorded": report["recorded"],
              "complete_interaction_harness": True, "published_benchmark_setup": False}
    write_json(output / "stage_check.json", result)
    return result
