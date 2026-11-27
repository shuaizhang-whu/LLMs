#!/usr/bin/env python3
"""Read-only summary of an early-exit smoke run.

Reports every run, every turn, and every paired comparison. No selection by
positive speedup. Quality is action/state agreement with the SPORK reference on
the same task, not official tau2 task success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(root: Path):
    records = []
    for path in sorted(root.glob("records/*.json")):
        records.append(json.loads(path.read_text()))
    return records


def turn_row(method, record, turn):
    return {
        "method": method,
        "group": record["group_id"],
        "repeat": record["repeat"],
        "turn": turn.get("turn"),
        "wall_s": round(record["wall_s"], 3),
        "main_end_s": round(turn.get("main_end_ms", 0) / 1000, 3),
        "spec_end_s": round((turn.get("spec_end_ms") or 0) / 1000, 3),
        "spec_used": turn.get("spec_used"),
        "exit_fired": turn.get("exit", {}).get("fired"),
        "cut_tokens": turn.get("exit", {}).get("cut_tokens"),
        "rolled_back": turn.get("exit", {}).get("rolled_back"),
        "stable_count": turn.get("exit", {}).get("stable_count"),
        "checks": len(turn.get("checks") or []),
        "probes": len(turn.get("probes") or []),
        "tool": turn.get("baseline_tool_name"),
        "has_call": turn.get("baseline_has_tool_call"),
        "finish": turn.get("main_finish_reason"),
        "d3_set": turn.get("d3_set_count"),
        "errors": record["errors"],
    }


def signature(record):
    return [
        {k: t.get(k) for k in (
            "baseline_has_tool_call", "baseline_tool_name", "baseline_tool_args",
            "tool_result_sha256", "environment_after")}
        for t in record["turns"]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    records = load_records(args.root)
    rows = []
    for record in records:
        for turn in record["turns"]:
            rows.append(turn_row(record["method"], record, turn))

    print(f"runs={len(records)} turns={len(rows)} root={args.root}")
    print()
    header = ("method", "group", "rep", "turn", "wall_s", "main_end", "spec_end",
              "used", "exit", "cut_tok", "rollback", "stable", "checks", "probes", "tool")
    print("  ".join(f"{h:>9}" if h not in ("method", "group", "tool") else f"{h:<9}" for h in header))
    for r in rows:
        print("  ".join(str(r[h]).rjust(9) if h not in ("method", "group", "tool") else str(r[h]).ljust(9)
                        for h in header))

    print("\nPaired comparisons (SPORK reference):")
    groups = {}
    for record in records:
        groups.setdefault((record["group_id"], record["repeat"]), {})[record["method"]] = record
    for key, by_method in sorted(groups.items()):
        ref = by_method.get("spork")
        if ref is None:
            continue
        for method, other in sorted(by_method.items()):
            if method == "spork":
                continue
            print(f"  {key[0]} r{key[1]} spork->{method}: "
                  f"saved={ref['wall_s'] - other['wall_s']:+.3f}s "
                  f"same_action_state={signature(ref) == signature(other)} "
                  f"same_final_answer={[t.get('final_answer') for t in ref['turns']] == [t.get('final_answer') for t in other['turns']]} "
                  f"errors={other['errors']}")

    print("\nExit check trajectories:")
    for record in records:
        if record["method"] == "spork":
            continue
        for turn in record["turns"]:
            checks = turn.get("checks") or []
            if not checks:
                continue
            print(f"  {record['group_id']} r{record['repeat']} {record['method']} turn {turn.get('turn')}:")
            for c in checks:
                pred = c.get("prediction") or {}
                print(f"    idx={c.get('index')} tok={c.get('tokens_at_dispatch')} "
                      f"p={c.get('tool_probability') if c.get('tool_probability') is not None else pred.get('tool_probability')} "
                      f"agrees={c.get('agrees')} stable={c.get('stable_count')} "
                      f"call={(c.get('call') or {}).get('name') if isinstance(c.get('call'), dict) else None} "
                      f"wall={round((pred.get('wall_s') or 0), 3)}")


if __name__ == "__main__":
    main()
