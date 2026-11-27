"""Planning, validation and label aggregation for staged controller data."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import random
import statistics

from .prefix_replay import COUNTERS, paired_labels
from .prefix_stability import POLICIES, QUALITY_KEYS, mean_interval


def task_batches(tasks, batch_size=16, seed=43):
    """Preassigned task groups; interleave domains and use three train per val."""
    groups = [t["group_id"] for t in tasks]
    if len(set(groups)) != len(groups) or batch_size < 4:
        raise ValueError("Duplicate groups or invalid batch size")
    rng = random.Random(seed)
    pools = {}
    for split in ("train", "validation"):
        domains = {d: sorted((dict(t) for t in tasks if t["split"] == split and t["domain"] == d),
                             key=lambda t: t["group_id"]) for d in ("airline", "retail")}
        for values in domains.values():
            rng.shuffle(values)
        pools[split] = [domains[d][i] for i in range(max(map(len, domains.values()), default=0))
                        for d in domains if i < len(domains[d])]
    ordered = []
    while any(pools.values()):
        for split in ("train", "train", "train", "validation"):
            if pools[split]:
                ordered.append(pools[split].pop(0))
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


def policy_orders(sample_id, seed=43):
    seed_value = int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:16], 16)
    first = list(POLICIES)
    random.Random(seed_value).shuffle(first)
    # Three repeats balance each policy's position, but not all six permutations.
    return [first[i:] + first[:i] for i in range(3)]


def branch_errors(policy, out, inp, cfg):
    """Length termination is recorded data; infrastructure/protocol failures stop."""
    errors = []
    if out.get("error") or out.get("finish_reason") not in ("stop", "length"):
        errors.append("transport_or_completion_error")
    if out.get("completion_valid") is not (out.get("finish_reason") == "stop"):
        errors.append("inconsistent_completion_flag")
    if out.get("policy") != policy or out.get("config") != cfg:
        errors.append("changed_policy_or_config")
    enabled = policy != "off"
    if out.get("interventions") != {"enable_d3": enabled, "enable_speculative_tools": enabled}:
        errors.append("changed_interventions")
    if out.get("environment_before") != inp["environment_hash"]:
        errors.append("changed_initial_state")
    if out.get("main_text") != inp["prefix_text"] + out.get("continuation_text", ""):
        errors.append("changed_prefix")
    if not out.get("main_request_sha256"):
        errors.append("missing_request_identity")
    wall, main = out.get("wall_s"), out.get("main_finished_s")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (wall, main)) or not wall >= main >= 0:
        errors.append("invalid_timing")
    if out.get("cleanup_tool_error"):
        errors.append("tool_cleanup_error")
    clear = out.get("clear_after", {})
    if any(clear.get(k) != 0 for k in ("active_requests", "pending_verifications")):
        errors.append("residual_or_missing_server_state")
    counters = out.get("d3_counters", {})
    if any(counters.get(k) != 0 for k in ("accounting_errors", "prefix_mismatches", "dropped_injections")):
        errors.append("d3_integrity_error")
    probes = out.get("probes", [])
    if policy == "off" and (probes or out.get("spec_used") or any(counters.get(k, 0) for k in COUNTERS)):
        errors.append("off_branch_speculated")
    for p in probes:
        if p.get("error") or ("d3_set" in p and p["d3_set"].get("status") != "ok"):
            errors.append("prediction_request_error")
        if not out.get("main_text", "").startswith(p["visible_prefix_text"]):
            errors.append("prediction_saw_a_different_continuation")
        if p["total_visible_generated_tokens"] != len(inp["prefix_token_ids"]) + p["new_tokens_at_dispatch"]:
            errors.append("wrong_visible_token_count")
    if probes:
        first = probes[0]
        if policy == "now" and (first["new_tokens_at_dispatch"] != 0 or first["visible_prefix_text"] != inp["prefix_text"]):
            errors.append("now_saw_future_tokens")
        if policy == "wait" and first["new_tokens_at_dispatch"] < cfg["delay_tokens"]:
            errors.append("wait_started_early")
    events = out.get("token_events", [])
    if not events or events[-1]["end_token"] != len(out.get("continuation_token_ids", [])):
        errors.append("invalid_token_trace")
    return errors


def aggregate_prefix(inp, repetitions, manifest):
    all_out = [o for row in repetitions for o in row.values()]
    complete = len(repetitions) == 3 and all(set(row) == set(POLICIES) for row in repetitions)
    valid_completion = complete and all(o.get("completion_valid") and not o.get("error") for o in all_out)
    same_tokens = bool(all_out) and all(o.get("continuation_token_ids") == all_out[0].get("continuation_token_ids") for o in all_out)
    same_requests = bool(all_out) and len({o.get("main_request_sha256") for o in all_out}) == 1
    same_outcomes = bool(all_out) and all(
        all(o.get(k) == all_out[0].get(k) for k in ("formal_call", "tool_result", "environment_after"))
        for o in all_out)
    labels = [paired_labels(row) for row in repetitions]
    flags = {key: False for key in QUALITY_KEYS}
    stats, targets = {}, {"gain_now_s": None, "gain_wait_s": None}
    if valid_completion:
        for policy in ("now", "wait"):
            for label, row in zip(labels, repetitions):
                label[policy + "_same_state_as_off"] = row[policy]["environment_after"] == row["off"]["environment_after"]
        flags = {key: all(label.get(key) is True for label in labels) for key in QUALITY_KEYS}
        for p in ("now", "wait"):
            stats[p] = mean_interval([r["off"]["wall_s"] - r[p]["wall_s"] for r in repetitions])
        stats["now_minus_wait"] = mean_interval([r["wait"]["wall_s"] - r["now"]["wall_s"] for r in repetitions])
    usable = valid_completion and same_tokens and same_requests and same_outcomes and all(flags.values())
    if usable:
        targets = {f"gain_{p}_s": stats[p]["mean"] for p in ("now", "wait")}
    tolerance = None
    precise = False
    if stats:
        tolerance = max(.5, .05 * statistics.mean(r["off"]["wall_s"] for r in repetitions))
        precise = all(s["half_width"] is not None and s["half_width"] <= tolerance for s in stats.values())
    audit = {
        "sample_id": inp["sample_id"], "group_id": inp["group_id"], "split": inp["split"],
        "all_branches_recorded": complete, "all_completions_valid": valid_completion,
        "all_observed_tokens_equal": same_tokens, "same_actor_requests": same_requests,
        "same_outcomes_across_repetitions": same_outcomes,
        "label_usable": usable, "recorded_branch_runs": len(all_out),
        "truncated_branch_runs": sum(o.get("finish_reason") == "length" for o in all_out),
        "nominal_gain_intervals_s": stats, "precision_tolerance_s": tolerance,
        "meets_initial_precision_criterion": precise,
        "scope": "Three repeats, nominal per-contrast t intervals; not the earlier six-repeat stability gate. "
                 "No future outcome enters input. All failures, ties, negative gains and noisy labels remain recorded.",
    }
    row = {
        "id": manifest["dataset_id"] + "/" + inp["sample_id"], "group_id": inp["group_id"], "split": inp["split"],
        "input": {**{k: inp[k] for k in ("messages", "tools", "prefix_text", "prompt_token_ids", "prefix_token_ids",
                                        "visible_generated_tokens")}, "wait_tokens": manifest["replay_config"]["delay_tokens"]},
        "targets": targets, "label_valid": usable,
        "quality_flags": {**flags, "task_success_evaluated": False, "all_continuation_tokens_match": same_tokens},
        "metadata": {"domain": inp["domain"], "task_id": inp["task_id"], "turn": inp["turn"],
                     "model": manifest["model"], "measurement": "warmed_prefix_replay_batch_invariant_v1",
                     "tool_floor_s": manifest["tool_floor_s"], "raw_sample": inp["sample_id"],
                     "repetitions": 3, "collected_repetitions": len(repetitions),
                     "precision_checked": True, "precision_passed": precise,
                     "batch_invariant_requested": True},
    }
    return row, audit
