#!/usr/bin/env python3
"""Retrospective train-only bound for one speculative successor Actor.

No latency simulator is presented as a measurement. Fixed disjoint adjacent
turn pairs avoid crediting the same Actor twice. Concurrency interference,
wrong-branch work, request overhead, and live prefix compatibility still need
real experiments. Only the already observed correct-branch window is bounded.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def candidate_message(turn):
    if not turn.get("committed_probe_name"):
        return None
    call = {"name": turn["committed_probe_name"], "arguments": turn["committed_probe_args"]}
    return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n"


def analyze(root):
    rows = []
    total_s = 0.
    tasks = 0
    excluded = Counter()
    for path in sorted(root.glob("*/timing_spork.json")):
        record = json.loads(path.read_text())
        if record["split"] != "train":
            continue
        if record.get("errors"):
            excluded["failed_train_task"] += 1
            continue
        tasks += 1
        turns = record.get("turns", [])
        total_s += sum(t.get("turn_end_ms", 0) for t in turns) / 1000
        # Fixed phase, chosen without inspecting gains: (turn 1,2), (3,4), ...
        for parent_index in range(0, len(turns) - 1, 2):
            parent, child = turns[parent_index:parent_index+2]
            actual_message = re.sub(r"^\s*<think>.*?</think>\s*", "", parent["baseline_text"],
                                    count=1, flags=re.DOTALL)
            predicted_message = candidate_message(parent)
            matched = bool(parent.get("spec_used"))
            same_message = matched and predicted_message == actual_message
            ready = parent.get("spec_end_ms")
            window = max(0., (parent["main_end_ms"] - ready) / 1000) if ready is not None else 0.
            child_actor = child["main_end_ms"] / 1000
            bound = min(window, child_actor) if same_message else 0.
            rows.append({"group_id": record["group_id"], "parent_turn": parent["turn"],
                         "speculative_tool_dispatched": bool(parent.get("spec_dispatched")),
                         "action_match": matched, "canonical_message_exact": same_message,
                         "parent_remaining_after_tool_s": window,
                         "child_actor_s": child_actor,
                         "ideal_correct_branch_overlap_s": bound,
                         "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    total_bound = sum(r["ideal_correct_branch_overlap_s"] for r in rows)
    return {"tasks": tasks, "pairs": len(rows), "selection": "existing train split; fixed pairs (1,2),(3,4),...",
            "excluded": dict(excluded), "observed_total_turn_s": total_s,
            "action_matched_pairs": sum(r["action_match"] for r in rows),
            "canonical_message_exact_pairs": sum(r["canonical_message_exact"] for r in rows),
            "ideal_correct_branch_overlap_s": total_bound,
            "ideal_correct_branch_fraction": total_bound / total_s if total_s else None,
            "per_pair": rows,
            "not_estimated": ["GPU contention from concurrent Actor requests", "cost of incorrect successor branches",
                              "online causal admission accuracy", "exact whole-prompt token equivalence at adoption",
                              "cost of successor tools and cancellation", "interactive task success"],
            "scope": "Retrospective opportunity bound, not an online speedup or a complete schedule simulation."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = analyze(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_pair"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
