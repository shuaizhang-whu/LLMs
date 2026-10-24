#!/usr/bin/env python3
"""Paired analysis of a stable-refresh smoke, split by SPORK's initial match.

Every comparison is reported, including regressions. Quality here is
action/tool-result/environment agreement with the SPORK reference on the same
fixed-user-message task; it is not official interactive tau2 success.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def signature(record):
    return [{k: t.get(k) for k in (
        "baseline_has_tool_call", "baseline_tool_name", "baseline_tool_args",
        "tool_result_sha256", "environment_after")} for t in record["turns"]]


def final_answers(record):
    return [t.get("final_answer") for t in record["turns"]]


def load(root: Path):
    return [json.loads(p.read_text()) for p in sorted(root.glob("records/*.json"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    records = load(args.root)
    groups = defaultdict(dict)
    for record in records:
        groups[(record["group_id"], record["repeat"])][record["method"]] = record

    rows = []
    for (group_id, repeat), by_method in sorted(groups.items()):
        spork = by_method.get("spork")
        if spork is None:
            continue
        initial_match = bool(spork["turns"] and spork["turns"][0].get("spec_used"))
        refresh = by_method.get("refresh")
        eta = by_method.get("eta")
        off = by_method.get("off")
        rows.append({
            "group": group_id,
            "repeat": repeat,
            "initial_match": initial_match,
            "spork_s": spork["wall_s"],
            "eta_s": eta["wall_s"] if eta else None,
            "refresh_s": refresh["wall_s"] if refresh else None,
            "off_s": off["wall_s"] if off else None,
            "refresh_saved_s": (spork["wall_s"] - refresh["wall_s"]) if refresh else None,
            "eta_saved_s": (spork["wall_s"] - eta["wall_s"]) if eta else None,
            "refresh_count": sum(t.get("exit", {}).get("refresh_count", 0)
                                 for t in refresh["turns"]) if refresh else None,
            "refresh_exits": sum(t.get("exit", {}).get("fired", False)
                                 for t in refresh["turns"]) if refresh else None,
            "refresh_rollbacks": sum(t.get("exit", {}).get("rolled_back", False)
                                     for t in refresh["turns"]) if refresh else None,
            "refresh_same_action_state": (signature(spork) == signature(refresh)) if refresh else None,
            "eta_same_action_state": (signature(spork) == signature(eta)) if eta else None,
            "refresh_same_final_answer": (final_answers(spork) == final_answers(refresh)) if refresh else None,
            "errors": {m: r["errors"] for m, r in by_method.items() if r["errors"]},
        })

    print(f"root={args.root} groups={len(rows)}")
    print("group                 match  spork_s   eta_s  refresh_s   off_s  r_saved  eta_saved  r#  r_exit  r_ok")
    for r in rows:
        print(f"{r['group']:<20} {str(r['initial_match']):<6} "
              f"{r['spork_s']:7.1f} {r['eta_s'] if r['eta_s'] is not None else -1:7.1f} "
              f"{r['refresh_s'] if r['refresh_s'] is not None else -1:9.1f} "
              f"{r['off_s'] if r['off_s'] is not None else -1:7.1f} "
              f"{r['refresh_saved_s'] if r['refresh_saved_s'] is not None else 0:8.1f} "
              f"{r['eta_saved_s'] if r['eta_saved_s'] is not None else 0:9.1f} "
              f"{r['refresh_count'] if r['refresh_count'] is not None else -1:3} "
              f"{str(r['refresh_exits']):>7} {str(r['refresh_same_action_state']):>6}")

    print("\nBy SPORK first-turn match:")
    for match in (True, False):
        subset = [r for r in rows if r["initial_match"] is match and r["refresh_s"] is not None]
        if not subset:
            continue
        print(f"  initial_match={match}: n={len(subset)} "
              f"mean_spork={statistics.mean(r['spork_s'] for r in subset):.1f}s "
              f"mean_refresh={statistics.mean(r['refresh_s'] for r in subset):.1f}s "
              f"mean_saved={statistics.mean(r['refresh_saved_s'] for r in subset):.1f}s "
              f"quality_ok={sum(bool(r['refresh_same_action_state']) for r in subset)}/{len(subset)} "
              f"refreshes={sum(r['refresh_count'] or 0 for r in subset)}")

    print("\nAll refresh errors:", {r["group"]: r["errors"] for r in rows if r["errors"]})
    print("Regressions (refresh vs spork):",
          [(r["group"], r["refresh_saved_s"]) for r in rows
           if r["refresh_same_action_state"] is False])


if __name__ == "__main__":
    main()
