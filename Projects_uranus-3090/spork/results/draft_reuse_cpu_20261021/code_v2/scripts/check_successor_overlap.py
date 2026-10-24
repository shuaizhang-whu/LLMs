#!/usr/bin/env python3
"""Check a necessary condition for overlapping the *next* Actor turn.

SPORK supplies an actual early tool result, while ObsSpec/AOSpec motivate
overlapping subsequent reasoning. This is an availability/prompt check only:
it neither executes a successor nor estimates shared-GPU speedup. It uses the
existing history rule verbatim (completed reasoning removed, other text kept).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replay_draft_reuse import sealed_result
from spork_core.qwen import FORK_PREFIX


def history_content(text):
    # Mirrors turn_runner._append_tool_messages; do not canonicalize whitespace
    # or silently remove post-think prose, since either changes the next input.
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing results")
    counts, leads, records = Counter(), [], []
    for line in (args.dataset / "training_rows.jsonl").open():
        row = json.loads(line)
        if row["split"] != args.split or not row["label_valid"]:
            continue
        meta = row["metadata"]
        folder = args.dataset / "tasks" / f'{meta["domain"]}_{meta["task_id"]}' / f'turn_{int(meta["turn"]):03d}'
        for policy in ("now", "wait"):
            result = sealed_result(folder / "measurements/repeat_000" / policy)
            counts["traces"] += 1
            if not result["spec_used"] or not result["completion_valid"]:
                counts["unmatched_or_incomplete"] += 1
                continue
            counts["matched_tool"] += 1
            probe, = [p for p in result["probes"] if p.get("dispatched")]
            predicted = FORK_PREFIX + probe["prediction"]["text"]
            identical = history_content(predicted) == history_content(result["main_text"])
            counts["identical_history" if identical else "different_history"] += 1
            # Start timestamp precedes worker scheduling; wall time is the tool
            # duration. Their sum is only an approximate availability time.
            lead = max(0.0, result["main_finished_s"] - probe["tool_dispatch_s"] - result["tool_wall_s"])
            if identical:
                leads.append(lead)
            records.append({"id": row["id"], "policy": policy,
                            "identical_history": identical, "approx_lead_s": lead})
    report = {"kind": "necessary_condition_only_not_successor_execution",
              "split": args.split, "counts": dict(counts),
              "approx_lead_mean_s_given_identical_history": statistics.mean(leads) if leads else None,
              "approx_lead_median_s_given_identical_history": statistics.median(leads) if leads else None,
              "limitations": ["No successor Actor was run; no speedup or quality claim.",
                              "The tool is a measured sandbox execution with a 2 s floor.",
                              "Lead omits thread scheduling delay and concurrent Actor contention.",
                              "Future use must verify full rendered token IDs, budgets, source/destination state and action before adoption."],
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
