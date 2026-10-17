"""Offline comparisons for the two-prefix generation-fork diagnostic."""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import random
import statistics

from .prefix_replay import same_call


MODES = {
    "main_only": {"policy": "off", "enable_d3": False, "enable_speculative_tools": False},
    "prediction_only": {"policy": "now", "enable_d3": False, "enable_speculative_tools": False},
    "spork_full": {"policy": "now", "enable_d3": True, "enable_speculative_tools": True},
}


def schedule_modes(sample_ids, repetitions=3, seed=43):
    if repetitions not in (3, 6):
        raise ValueError("Diagnostic repetitions must be 3 or 6")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate prefix ID")
    rng = random.Random(seed)
    orders = {}
    for sid in sorted(sample_ids):
        modes = list(MODES)
        rng.shuffle(modes)
        if repetitions == 3:
            orders[sid] = [modes[i:] + modes[:i] for i in range(3)]
        else:
            orders[sid] = [list(p) for p in itertools.permutations(modes)]
            rng.shuffle(orders[sid])
    schedule = []
    for rep in range(repetitions):
        ids = sorted(sample_ids)
        rng.shuffle(ids)
        schedule.extend({"sample_id": sid, "repetition": rep, "order": orders[sid][rep]} for sid in ids)
    return schedule


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare_outputs(left, right, tokenizer, prefix_tokens):
    a, b = left["continuation_token_ids"], right["continuation_token_ids"]
    equal = a == b
    index = None if equal else next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    end_think = tokenizer.convert_tokens_to_ids("</think>")
    tool_open = tokenizer.convert_tokens_to_ids("<tool_call>")

    def position(tokens, token):
        return tokens.index(token) if token in tokens else None

    def side(out, ids):
        end = position(ids, end_think)
        opener = position(ids, tool_open)
        received = next((e["received_s"] for e in out.get("token_events", [])
                         if index is not None and e["end_token"] > index), None)
        return {
            "continuation_tokens": len(ids), "reasoning_end_token_0based": end,
            "tool_opener_token_0based": opener, "divergence_chunk_received_s": received,
            "context_around_divergence": tokenizer.decode(ids[max(0, index - 12):index + 24],
                                                          skip_special_tokens=False) if index is not None else None,
            "first_prediction_dispatch_s": out["probes"][0]["dispatch_s"] if out["probes"] else None,
            "first_prediction_end_s": out["probes"][0].get("end_s") if out["probes"] else None,
        }

    ls, rs = side(left, a), side(right, b)
    a_call, b_call = left.get("formal_call"), right.get("formal_call")
    return {
        "same_tokens": equal, "first_difference_token_0based": index,
        "first_difference_including_saved_prefix_0based": index + prefix_tokens if index is not None else None,
        "before_both_reasoning_ends": (index < ls["reasoning_end_token_0based"]
                                       and index < rs["reasoning_end_token_0based"])
                                      if index is not None and ls["reasoning_end_token_0based"] is not None
                                      and rs["reasoning_end_token_0based"] is not None else None,
        "same_formal_action": (a_call is None and b_call is None) or same_call(a_call, b_call),
        "same_tool_result": left.get("tool_result") == right.get("tool_result"),
        "same_environment_after": left.get("environment_after") == right.get("environment_after"),
        "same_main_request": left.get("main_request_sha256") == right.get("main_request_sha256"),
        "left": ls, "right": rs,
    }


def check_branch(mode, out, snapshot, cfg):
    errors = []
    setting = MODES[mode]
    if not out.get("completion_valid") or out.get("error") or out.get("finish_reason") != "stop":
        errors.append("failed_or_truncated_completion")
    if out.get("policy") != setting["policy"] or out.get("config") != cfg:
        errors.append("changed_policy_or_config")
    expected = {k: setting[k] for k in ("enable_d3", "enable_speculative_tools")}
    if out.get("interventions") != expected:
        errors.append("incorrect_intervention_flags")
    if out.get("environment_before") != snapshot["environment_hash"]:
        errors.append("initial_environment_mismatch")
    if out.get("main_text") != snapshot["prefix_text"] + out.get("continuation_text", ""):
        errors.append("changed_saved_prefix")
    if not out.get("main_request_sha256"):
        errors.append("missing_main_request_identity")
    counters = out.get("d3_counters", {})
    for key in ("dropped_injections", "prefix_mismatches", "accounting_errors"):
        if key not in counters or counters[key]:
            errors.append("d3_" + key)
    cleared = out.get("clear_after", {})
    if ("active_requests" not in cleared or "pending_verifications" not in cleared
            or cleared.get("active_requests") or cleared.get("pending_verifications")):
        errors.append("residual_or_missing_d3_state")
    if out.get("cleanup_tool_error"):
        errors.append("tool_cleanup_error")
    probes = out.get("probes", [])
    if mode == "main_only" and probes:
        errors.append("unexpected_prediction")
    if mode != "main_only" and not probes:
        errors.append("concurrent_prediction_not_exercised")
    if mode != "spork_full":
        if any(counters.values()) or any("d3_set" in p for p in probes):
            errors.append("d3_was_not_disabled")
        if out.get("spec_used") or out.get("spec_wasted") or any(p.get("dispatched") for p in probes):
            errors.append("speculative_tools_were_not_disabled")
    if probes and (probes[0]["new_tokens_at_dispatch"] != 0
                   or probes[0]["visible_prefix_text"] != snapshot["prefix_text"]):
        errors.append("first_prediction_saw_future_tokens")
    for p in probes:
        if p.get("error"):
            errors.append("prediction_request_error")
        if "d3_set" in p and p["d3_set"].get("status") != "ok":
            errors.append("action_draft_registration_failed")
        if not out.get("main_text", "").startswith(p["visible_prefix_text"]):
            errors.append("prediction_not_on_actor_continuation")
    if probes and probes[0]["dispatch_s"] >= out.get("main_finished_s", 0):
        errors.append("first_prediction_not_concurrent_with_actor")
    events = out.get("token_events", [])
    ends = [e["end_token"] for e in events]
    times = [e["received_s"] for e in events]
    if (not ends or ends[-1] != len(out.get("continuation_token_ids", []))
            or any(b <= a for a, b in zip(ends, ends[1:]))
            or any(b < a for a, b in zip(times, times[1:])) or any(t < 0 for t in times)):
        errors.append("invalid_token_arrival_trace")
    return errors


def summarize(samples, snapshots, manifest, tokenizer):
    errors, details = [], []
    totals = {mode: Counter() for mode in MODES}
    comparisons = (("main_only", "prediction_only"), ("main_only", "spork_full"),
                   ("prediction_only", "spork_full"))
    for sid in manifest["sample_ids"]:
        rows = sorted(samples.get(sid, []), key=lambda r: r["repetition"])
        if [r["repetition"] for r in rows] != list(range(manifest["repetitions"])):
            errors.append(sid + ":incomplete_repetitions")
        per_repeat, variants = [], {}
        for row in rows:
            outcomes = row["outcomes"]
            if set(outcomes) != set(MODES):
                errors.append(sid + ":missing_modes")
                continue
            for mode, out in outcomes.items():
                errors.extend(f'{sid}/r{row["repetition"]}/{mode}:{e}'
                              for e in check_branch(mode, out, snapshots[sid], manifest["replay_config"]))
                totals[mode].update(out.get("d3_counters", {}))
            if len({out.get("main_request_sha256") for out in outcomes.values()}) != 1:
                errors.append(sid + ":different_actor_requests")
            per_repeat.append({"repetition": row["repetition"], "pairs": {
                left + "_vs_" + right: compare_outputs(outcomes[left], outcomes[right], tokenizer,
                                                       len(snapshots[sid]["prefix_token_ids"]))
                for left, right in comparisons}})
        for mode in MODES:
            values = [r["outcomes"][mode] for r in rows if mode in r["outcomes"]]
            if not values:
                continue
            variants[mode] = {
                "runs": len(values), "unique_token_sequences": len({digest_value(v["continuation_token_ids"]) for v in values}),
                "unique_formal_actions": len({digest_value(v["formal_call"]) for v in values}),
                "formal_action_names": dict(Counter(v["formal_call"]["name"] if v["formal_call"] else "(no tool)" for v in values)),
                "continuation_lengths": [len(v["continuation_token_ids"]) for v in values],
                "mean_wall_s": statistics.mean(v["wall_s"] for v in values),
                "mean_actor_finished_s": statistics.mean(v["main_finished_s"] for v in values),
                "prediction_counts": [len(v["probes"]) for v in values],
            }
        diff = {left + "_vs_" + right: sum(not r["pairs"][left + "_vs_" + right]["same_tokens"] for r in per_repeat)
                for left, right in comparisons}
        reference_stable = variants.get("main_only", {}).get("unique_token_sequences") == 1
        if not reference_stable:
            observation = "reference_itself_varies__cannot_attribute_from_this_comparison_alone"
        elif diff["main_only_vs_prediction_only"]:
            observation = "divergence_observed_without_action_draft_registration_or_early_tool_execution"
        elif diff["main_only_vs_spork_full"]:
            observation = "divergence_only_observed_with_additional_spork_effects_in_these_runs"
        else:
            observation = "prior_divergence_not_reproduced_in_these_runs"
        details.append({"sample_id": sid, "reference_tokens_stable": reference_stable,
                        "observation": observation, "token_differences_by_pair": diff,
                        "variants": variants, "comparisons": per_repeat})
    exercised = totals["spork_full"].get("verified_injections", 0) > 0
    if not exercised:
        errors.append("full_spork_d3_not_exercised")
    return {
        "schema_version": 1, "pipeline_passed": not errors, "errors": errors,
        "completed_branch_runs": sum(len(r["outcomes"]) for rows in samples.values() for r in rows),
        "planned_branch_runs": len(manifest["sample_ids"]) * manifest["repetitions"] * len(MODES),
        "d3_counters_by_mode": {k: dict(v) for k, v in totals.items()}, "per_prefix": details,
        "scope": "Generation-fork diagnostic, not controller training data or a speedup estimate. "
                 "Same ngram-enabled server in every mode; only SPORK action-draft registration is disabled "
                 "in prediction_only. Formal tools still execute after the Actor. Client chunk receipt times "
                 "do not identify exact GPU execution times. A fork without D3 narrows the cause to the concurrent "
                 "prediction path; it does not by itself distinguish numerical batching, caching or another serving issue.",
    }
