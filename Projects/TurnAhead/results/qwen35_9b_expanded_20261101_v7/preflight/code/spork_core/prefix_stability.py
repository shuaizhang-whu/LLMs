"""Offline repeatability checks for paired launch-timing labels (no inference).

Thresholds are engineering go/no-go criteria fixed before collection, not a
statistical proof of policy benefit. Repetitions of a prefix are not independent
tasks, and six repeats only provide an initial estimate of measurement noise.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import itertools
import math
import random
import statistics


POLICIES = ("off", "now", "wait")
QUALITY_KEYS = tuple(f"{p}_same_{kind}_as_off"
                     for p in ("now", "wait") for kind in ("action", "result", "state"))


@dataclass(frozen=True)
class StabilityCriteria:
    min_prefixes: int = 10
    min_repetitions: int = 6
    min_outcome_match_fraction: float = .95
    min_precise_prefix_fraction: float = .80
    absolute_precision_s: float = .5
    relative_precision: float = .05


def balanced_schedule(sample_ids, repetitions=6, seed=42):
    """All six permutations once per six rounds, shuffled across prefixes.

Every round visits all prefixes before the next repetition. Saved continuations
and their seeds remain fixed; only execution order is randomized.
"""
    if repetitions < 6 or repetitions % 6:
        raise ValueError("Use a positive multiple of six repetitions for balanced orders")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs")
    rng = random.Random(seed)
    orders = {}
    for sid in sorted(sample_ids):
        orders[sid] = []
        for _ in range(repetitions // 6):
            block = list(itertools.permutations(POLICIES))
            rng.shuffle(block)
            orders[sid].extend(block)
    schedule = []
    for rep in range(repetitions):
        ids = sorted(sample_ids)
        rng.shuffle(ids)
        schedule.extend({"sample_id": sid, "repetition": rep,
                         "order": list(orders[sid][rep])} for sid in ids)
    return schedule


def mean_interval(values):
    """Nominal 95% Student-t interval, with conservative rounded-up constants.

No new scipy dependency. For df > 30, using the df=30 constant is conservative.
The approximation presumes independent repeats and is diagnostic at small n.
"""
    x = [float(v) for v in values]
    if not x or not all(math.isfinite(v) for v in x):
        raise ValueError("Expected nonempty finite measurements")
    mean = statistics.mean(x)
    if len(x) == 1:
        return {"n": 1, "mean": mean, "sd": None, "half_width": None,
                "lower": None, "upper": None}
    constants = (12.707, 4.303, 3.183, 2.777, 2.571, 2.447, 2.365, 2.307,
                 2.263, 2.229, 2.201, 2.179, 2.161, 2.145, 2.132, 2.120,
                 2.110, 2.101, 2.094, 2.087, 2.080, 2.074, 2.069, 2.064,
                 2.060, 2.056, 2.052, 2.049, 2.046, 2.043)
    sd = statistics.stdev(x)
    half = constants[min(len(x) - 2, 29)] * sd / math.sqrt(len(x))
    return {"n": len(x), "mean": mean, "sd": sd, "half_width": half,
            "lower": mean - half, "upper": mean + half}


def branch_integrity_errors(outcomes):
    errors = []
    if set(outcomes) != set(POLICIES):
        return ["incomplete_policy_triple"]
    for p, out in outcomes.items():
        if out.get("policy") != p or not out.get("completion_valid") or out.get("error"):
            errors.append(f"{p}:failed_or_truncated")
        t = out.get("wall_s")
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            errors.append(f"{p}:invalid_wall_time")
        if out.get("cleanup_tool_error"):
            errors.append(f"{p}:tool_cleanup_error")
        cleared = out.get("clear_after", {})
        if ("active_requests" not in cleared or "pending_verifications" not in cleared
                or cleared.get("active_requests") or cleared.get("pending_verifications")):
            errors.append(f"{p}:missing_or_nonempty_cleanup_status")
        counters = out.get("d3_counters", {})
        for k in ("dropped_injections", "prefix_mismatches", "accounting_errors"):
            if k not in counters or counters[k]:
                errors.append(f"{p}:{k}")
    return errors


def analyze_repetitions(samples, expected_ids, repetitions, criteria=None):
    """Use all valid repeats, including outcome disagreements, for diagnostics.

samples maps prefix ID to [{repetition, labels, outcomes}, ...]. No noisy rows
are removed before estimating dispersion or the quality-disagreement rate.
"""
    criteria = criteria or StabilityCriteria()
    details, errors = [], []
    matched, valid, total, verified = 0, 0, 0, 0
    for sid in expected_ids:
        records = samples.get(sid, [])
        ids = [r["repetition"] for r in records]
        complete = sorted(ids) == list(range(repetitions))
        if not complete:
            errors.append(f"{sid}:missing_or_duplicate_repetitions")
        clean = []
        for row in records:
            total += 1
            problems = branch_integrity_errors(row["outcomes"])
            if not row["labels"].get("valid"):
                problems.append("invalid_label")
            for out in row["outcomes"].values():
                verified += out.get("d3_counters", {}).get("verified_injections", 0)
            if problems:
                errors.extend(f'{sid}/r{row["repetition"]}:{e}' for e in problems)
                continue
            clean.append(row)
            valid += 1
            matched += all(row["labels"].get(k) is True for k in QUALITY_KEYS)
        item = {"sample_id": sid, "repetitions": len(records), "valid_repetitions": len(clean),
                "complete": complete, "precise": False, "all_outcomes_match": False}
        if clean:
            gains = {p: [r["outcomes"]["off"]["wall_s"] - r["outcomes"][p]["wall_s"]
                          for r in clean] for p in ("now", "wait")}
            for r, gn, gw in zip(clean, gains["now"], gains["wait"]):
                if (not math.isclose(r["labels"].get("gain_now_s", math.nan), gn, abs_tol=1e-9)
                        or not math.isclose(r["labels"].get("gain_wait_s", math.nan), gw, abs_tol=1e-9)):
                    errors.append(f'{sid}/r{r["repetition"]}:label_recomputation_failed')
            walls = {p: mean_interval([r["outcomes"][p]["wall_s"] for r in clean]) for p in POLICIES}
            stats = {p: mean_interval(gains[p]) for p in ("now", "wait")}
            stats["now_minus_wait"] = mean_interval([a - b for a, b in zip(gains["now"], gains["wait"])])
            tolerance = max(criteria.absolute_precision_s, criteria.relative_precision * walls["off"]["mean"])
            precise = (complete and len(clean) >= criteria.min_repetitions
                       and all(s["half_width"] is not None and s["half_width"] <= tolerance
                               for s in stats.values()))
            means = {"off": 0., **{p: stats[p]["mean"] for p in ("now", "wait")}}
            winner = max(POLICIES, key=means.get)
            observed = [max(POLICIES, key=lambda p: -r["outcomes"][p]["wall_s"]) for r in clean]
            pair_gaps = {}
            for other in POLICIES:
                if other != winner:
                    pair_gaps[other] = mean_interval([
                        r["outcomes"][other]["wall_s"] - r["outcomes"][winner]["wall_s"] for r in clean])
            # Near ties need not have an identical argmax in every repeat.
            clearly_best = all(s["lower"] is not None and s["lower"] > tolerance
                               for s in pair_gaps.values())
            item.update(
                precise=precise, precision_tolerance_s=tolerance, gains_s=stats, wall_s=walls,
                mean_best_policy=winner, clearly_best=clearly_best,
                per_repeat_best_counts=dict(Counter(observed)),
                all_outcomes_match=all(r["labels"].get(k) is True for r in clean for k in QUALITY_KEYS),
                outcome_matched_repetitions=sum(all(r["labels"].get(k) is True for k in QUALITY_KEYS) for r in clean),
                continuation_lengths={p: [len(r["outcomes"][p].get("continuation_token_ids", []))
                                          for r in clean] for p in POLICIES},
                first_half_mean_gains={p: statistics.mean(gains[p][:max(1, len(clean) // 2)]) for p in gains},
                second_half_mean_gains={p: statistics.mean(gains[p][len(clean) // 2:]) for p in gains},
            )
        details.append(item)
    if set(samples) - set(expected_ids):
        errors.append("unexpected_sample_ids")
    precise_count = sum(d["precise"] for d in details)
    match_fraction = matched / total if total else 0.
    precise_fraction = precise_count / len(expected_ids) if expected_ids else 0.
    checks = {
        "enough_prefixes": len(expected_ids) >= criteria.min_prefixes,
        "enough_repetitions": repetitions >= criteria.min_repetitions,
        "complete_valid_measurements": not errors and total == len(expected_ids) * repetitions,
        "d3_exercised": verified > 0,
        "outcome_match_rate": match_fraction >= criteria.min_outcome_match_fraction,
        "label_precision_rate": precise_fraction >= criteria.min_precise_prefix_fraction,
    }
    return {
        "schema_version": 1, "passed": all(checks.values()), "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v], "criteria": asdict(criteria),
        "prefixes": len(expected_ids), "planned_repetitions_per_prefix": repetitions,
        "observed_triples": total, "valid_triples": valid, "outcome_matched_triples": matched,
        "outcome_match_fraction": match_fraction, "precise_prefixes": precise_count,
        "precise_prefix_fraction": precise_fraction, "verified_d3_injections": verified,
        "mean_best_policy_counts": dict(Counter(d["mean_best_policy"] for d in details if "mean_best_policy" in d)),
        "clearly_best_policy_counts": dict(Counter(d["mean_best_policy"] for d in details if d.get("clearly_best"))),
        "errors": errors, "per_prefix": details,
        "scope": "Pilot repeatability only. Criteria fixed before collection. Nominal per-contrast 95% t intervals; "
                 "no multiplicity correction or small-sample reliability guarantee. Not official tau2 success, "
                 "not proof of learnability or online speedup. Negative gains and disagreements retained.",
    }
