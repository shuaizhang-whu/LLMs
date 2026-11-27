"""Offline policy comparisons. Recorded gains are not measured online speedups.

Task means give each task equal weight; bootstrap resamples whole tasks, never
individual turns. Intervals on checkpoint-selection validation data are only
descriptive and do not correct model-selection bias or timing-label noise.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import statistics


POLICIES = ("off", "now", "wait")
TARGETS = ("gain_now_s", "gain_wait_s")
SCOPE = ("Offline recorded gains on usable prefixes; excludes controller latency. "
         "Validation is used for model selection, not an untouched test set.")


def task_values(rows, values):
    if len(rows) != len(values) or not rows:
        raise ValueError("Expected one value per nonempty evaluation row")
    groups = defaultdict(list)
    for row, value in zip(rows, values):
        if not math.isfinite(value):
            raise ValueError("Non-finite evaluation value")
        groups[row["group_id"]].append(float(value))
    return {key: statistics.mean(v) for key, v in sorted(groups.items())}


def task_mean(rows, values):
    return statistics.mean(task_values(rows, values).values())


def label_gains(rows):
    return [[0., *(float(r["targets"][k]) for k in TARGETS)] for r in rows]


def choose_policy(values):
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected two finite predicted gains")
    return max(range(3), key=lambda i: [0., *values][i])


def prediction_records(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("Prediction/row length mismatch")
    return [{"id": r["id"], "group_id": r["group_id"],
             "q_now_s": float(p[0]), "q_wait_s": float(p[1]),
             "policy": POLICIES[choose_policy(p)]}
            for r, p in zip(rows, predictions)]


def score_policy(rows, picks):
    if len(rows) != len(picks) or any(type(p) is not int or p not in range(3) for p in picks):
        raise ValueError("Expected a valid policy ID per evaluation row")
    gains = label_gains(rows)
    realized = [v[p] for v, p in zip(gains, picks)]
    fixed = {p: task_mean(rows, [v[i] for v in gains]) for i, p in enumerate(POLICIES)}
    oracle = task_mean(rows, [max(v) for v in gains])
    selected = task_mean(rows, realized)
    return {
        "rows": len(rows), "tasks": len({r["group_id"] for r in rows}),
        "selected_policy_counts": {p: picks.count(i) for i, p in enumerate(POLICIES)},
        "offline_selected_recorded_gain_s": statistics.mean(realized),
        "task_mean_recorded_gain_s": selected,
        "fixed_policy_recorded_gains_s": {p: statistics.mean(v[i] for v in gains)
                                          for i, p in enumerate(POLICIES)},
        "task_mean_fixed_policy_gains_s": fixed,
        "task_gain_over_fixed_s": {p: selected - v for p, v in fixed.items()},
        "empirical_oracle_task_gain_s": oracle,
        "task_regret_to_empirical_oracle_s": oracle - selected,
        "selected_negative_gain_rows": sum(v < 0 for v in realized),
        "task_gains_s": task_values(rows, realized),
        "scope": SCOPE,
    }


def score_predictions(rows, predictions):
    picks = [choose_policy(p) for p in predictions]
    result = score_policy(rows, picks)
    errors = [[p[i] - r["targets"][k] for i, k in enumerate(TARGETS)]
              for r, p in zip(rows, predictions)]
    result.update(mse_seconds_squared=statistics.mean(e * e for pair in errors for e in pair),
                  mae_now_s=statistics.mean(abs(e[0]) for e in errors),
                  mae_wait_s=statistics.mean(abs(e[1]) for e in errors))
    return result


def visible_bucket(inp):
    """A cheap baseline sees the tool vocabulary and whether a tool ran before.

    No task ID, future action, outcome, domain metadata or timing label is input.
    """
    names = sorted(t.get("function", t).get("name", "") for t in inp.get("tools", []))
    return (tuple(names), any(m.get("role") == "tool" for m in inp["messages"]))


def fit_simple_baselines(train_rows, evaluation_rows):
    # Targets are consulted exclusively on training tasks when fitting policies.
    global_means = [task_mean(train_rows, [r["targets"][k] for r in train_rows]) for k in TARGETS]
    constant = choose_policy(global_means)
    buckets = defaultdict(list)
    for row in train_rows:
        buckets[visible_bucket(row["input"])].append(row)
    table = {key: [task_mean(rs, [r["targets"][k] for r in rs]) for k in TARGETS]
             for key, rs in buckets.items()}
    estimates = [table.get(visible_bucket(r["input"]), global_means) for r in evaluation_rows]
    picks = {"fixed_" + p: [i] * len(evaluation_rows) for i, p in enumerate(POLICIES)}
    picks["train_best_constant"] = [constant] * len(evaluation_rows)
    picks["visible_history_bucket"] = [choose_policy(v) for v in estimates]
    return {"train_best_constant_policy": POLICIES[constant],
            "global_training_mean_gains_s": global_means,
            "bucket_definition": "sorted available tool names + presence of a past tool message",
            "unseen_bucket_fallback": "global training means",
            "training_buckets": [{"tool_names": list(k[0]), "has_past_tool_message": k[1],
                                  "training_rows": len(buckets[k]), "q_means_s": v}
                                 for k, v in sorted(table.items())],
            "policies": picks,
            "metrics": {name: score_policy(evaluation_rows, value) for name, value in picks.items()}}, estimates


def bootstrap_task_difference(task_differences, repetitions=2000, seed=20261010):
    values = list(task_differences.values())
    if not values or repetitions < 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("Invalid task bootstrap inputs")
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return {"mean_s": statistics.mean(values),
            "lower_s": samples[int(.025 * (repetitions - 1))],
            "upper_s": samples[int(.975 * (repetitions - 1))],
            "tasks": len(values), "resamples": repetitions, "seed": seed,
            "scope": "Descriptive paired task bootstrap, not corrected for validation selection or label noise."}


def describe_labels(rows):
    gains = label_gains(rows)
    best = [max(range(3), key=lambda i: v[i]) for v in gains]
    fixed = {p: task_mean(rows, [v[i] for v in gains]) for i, p in enumerate(POLICIES)}
    return {"rows": len(rows), "tasks": len({r["group_id"] for r in rows}),
            "domains": dict(Counter(r["metadata"]["domain"] for r in rows)),
            "best_recorded_policy_counts": {p: best.count(i) for i, p in enumerate(POLICIES)},
            "both_speculation_gains_negative_rows": sum(max(v[1:]) < 0 for v in gains),
            "median_absolute_now_wait_gap_s": statistics.median(abs(v[1] - v[2]) for v in gains),
            "task_mean_fixed_policy_gains_s": fixed,
            "empirical_oracle_headroom_over_best_fixed_s": task_mean(rows, [max(v) for v in gains]) - max(fixed.values()),
            "label_noise_caveat": "The empirical per-row maximum also selects timing noise; it is not a noise-free ceiling."}
