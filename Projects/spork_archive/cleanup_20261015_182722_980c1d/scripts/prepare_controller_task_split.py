#!/usr/bin/env python3
"""Freeze the full supported tau2 task split before expanded collection.

Offline manifest preparation only. Existing development/pilot tasks are forced
into training; validation and test tasks are disjoint within each domain.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def assign_splits(tasks, development_groups, seed=43):
    keys = [f'{t["domain"]}:{t["task_id"]}' for t in tasks]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate task group")
    if not set(development_groups) <= set(keys):
        raise ValueError("Development tasks are missing from this task set")
    rng = random.Random(seed)
    result = []
    for domain in sorted({t["domain"] for t in tasks}):
        pool = sorted((t for t in tasks if t["domain"] == domain), key=lambda t: str(t["task_id"]))
        holdout = [t for t in pool if f'{domain}:{t["task_id"]}' not in development_groups]
        rng.shuffle(holdout)
        count = max(1, round(len(pool) * .15))
        if len(holdout) < 2 * count or len(pool) <= 2 * count:
            raise ValueError(f"Insufficient independent tasks to split {domain}")
        val = {str(t["task_id"]) for t in holdout[:count]}
        test = {str(t["task_id"]) for t in holdout[count:2 * count]}
        for task in pool:
            sid = str(task["task_id"])
            result.append({**task, "group_id": f"{domain}:{sid}",
                           "split": "validation" if sid in val else "test" if sid in test else "train"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing split: {args.output}")
    development_groups = set()
    for path in args.pilot_data:
        development_groups.update(json.loads(line)["group_id"] for line in path.read_text().splitlines() if line.strip())
    tasks, sources = [], []
    for domain in ("airline", "retail"):
        path = ROOT / "tau2-bench/data/tau2/domains" / domain / "tasks.json"
        raw = path.read_bytes()
        source_tasks = json.loads(raw)
        eligible = [t for t in source_tasks if (t.get("evaluation_criteria") or {}).get("actions")]
        sources.append({"domain": domain, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                        "all_tasks": len(source_tasks), "supported_tasks": len(eligible),
                        "excluded_no_evaluation_actions": len(source_tasks) - len(eligible)})
        for t in eligible:
            if t.get("initial_state"):
                raise ValueError("Task requires an unsupported initial-state override")
            tasks.append({"domain": domain, "task_id": str(t.get("id", t.get("task_id")))})
    tasks = assign_splits(tasks, development_groups, args.seed)
    plan = {
        "schema_version": 1, "purpose": "preassigned_full_supported_tau2_controller_task_split",
        "seed": args.seed, "task_count": len(tasks), "sources": sources,
        "task_counts": dict(Counter(t["split"] for t in tasks)),
        "domain_counts": dict(Counter(t["domain"] for t in tasks)),
        "development_groups_forced_to_training": sorted(development_groups), "tasks": tasks,
        "collection_scope": "SPORK's existing fixed-user-message airline/retail harness. "
                            "All supported tasks, up to eight turns each; this is not the full official interactive tau2 benchmark.",
        "planned_max_turns_per_task": 8, "planned_prefix_tokens": 32, "planned_wait_tokens": 64,
        "start_condition": "A separate six-repeat pilot stability_report.json must pass before expanded inference.",
        "test_policy": "Do not fit input vocabulary, target scaling, model weights or checkpoint selection on test tasks.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tasks": len(tasks), "splits": plan["task_counts"],
                      "domains": plan["domain_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
