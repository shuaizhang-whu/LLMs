#!/usr/bin/env python3
"""From `off` records, estimate what an early exit would have committed.

`off` runs the full ETA probe cadence with truncation disabled, so each turn
contains the complete check trajectory AND the full-reasoning final action.
For every turn we find the first k consecutive identical probe calls and compare
that stable call with the final action. This is an offline proxy for the
quality risk of an early exit, not a substitute for task success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def args_equal(a, b):
    return a == b


def same_call(a, b):
    return bool(a and b and a.get("name") == b.get("name")
                and args_equal(a.get("arguments"), b.get("arguments")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("-k", "--stable-checks", type=int, default=3)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("records/*_off.json")):
        record = json.loads(path.read_text())
        for turn in record["turns"]:
            sequence = []
            if turn.get("probes"):
                sequence.append(turn["probes"][0].get("call") or {
                    "name": turn["probes"][0].get("probe_tool_name"),
                    "arguments": turn["probes"][0].get("probe_tool_args")})
            for check in turn.get("checks") or []:
                sequence.append(check.get("call"))
            stable_at = None
            stable_call = None
            for index in range(args.stable_checks - 1, len(sequence)):
                window = sequence[index - args.stable_checks + 1:index + 1]
                if window[0] is not None and all(same_call(c, window[0]) for c in window[1:]):
                    stable_at, stable_call = index, window[0]
                    break
            final = ({"name": turn.get("baseline_tool_name"),
                      "arguments": turn.get("baseline_tool_args")}
                     if turn.get("baseline_has_tool_call") else None)
            initial = sequence[0] if sequence else None
            rows.append({
                "group": record["group_id"], "repeat": record["repeat"], "turn": turn.get("turn"),
                "initial_call": initial, "stable_at": stable_at, "stable_call": stable_call,
                "final_call": final,
                "stable_matches_final": same_call(stable_call, final),
                "initial_matches_final": same_call(initial, final),
                "stable_tokens": (turn["checks"][stable_at - 1].get("tokens_at_dispatch")
                                  if stable_at and stable_at >= 1 and turn.get("checks") else None),
                "checks": len(turn.get("checks") or []),
            })
    print(f"turns={len(rows)} source={args.root}")
    for r in rows:
        print(f"{r['group']:<16} t{r['turn']} stable_at={str(r['stable_at']):>4} "
              f"stable={((r['stable_call'] or {}).get('name')):<28} "
              f"final={((r['final_call'] or {}).get('name')):<28} "
              f"ok={r['stable_matches_final']} initial_ok={r['initial_matches_final']}")
    stable = [r for r in rows if r["stable_at"] is not None]
    print(f"\nfound stable k={args.stable_checks}: {len(stable)}/{len(rows)}")
    if stable:
        print(f"stable matches final: {sum(r['stable_matches_final'] for r in stable)}/{len(stable)}")
        print(f"initial matches final: {sum(r['initial_matches_final'] for r in stable)}/{len(stable)}")
    no_stable = [r for r in rows if r["stable_at"] is None]
    if no_stable:
        print("no stable call:", [(r["group"], r["turn"]) for r in no_stable])


if __name__ == "__main__":
    main()
