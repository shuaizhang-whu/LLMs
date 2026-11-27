#!/usr/bin/env python3
"""Report a bounded failure-recovery check; never a strict acceleration gate."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_qwen35_characterization import analyze
from spork_core.evaluation_artifacts import planned_records, write_json


def report(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if set(manifest["configs"]) != {"spork", "raw", "raw_fallback"}:
        raise ValueError("This report requires the three recovery-control arms")
    result = analyze(output)
    records, _ = planned_records(output, manifest)
    details = []
    for mode, row in result["modes"].items():
        pairs = [p for r in records if r["config"] == mode for p in r["prefetch"]["pairs"]]
        row["recovery_attempts"] = sum(bool(p.get("recovery_attempted")) for p in pairs)
        row["recovery_completed"] = sum(bool(p.get("recovery_succeeded")) for p in pairs)
    for r in records:
        for p in r["prefetch"]["pairs"]:
            if p.get("recovery_attempted"):
                details.append({"group_id": r["group_id"], "repeat": r["repeat"],
                    "reason": p.get("recovery_reason"), "completed": p.get("recovery_succeeded"),
                    "task_errors": r["errors"], "whole_task_s": r["wall_s"],
                    "observed_discarded_tokens": p.get("observed_wasted_child_tokens", 0)})
    result["recovery_details"] = details
    result["scope"] = ("Training-only recovery stress test. A recovery adds one canonical Target attempt; "
        "all wasted computation and cleanup remain timed. Neither a backend invariance repair nor "
        "a stochastic distribution-preservation or official task-success claim.")
    write_json(output / "recovery_check.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = report(args.output)
    print(json.dumps({k: result[k] for k in ("recorded", "modes", "comparison_counts",
                     "recovery_details", "scope")}, ensure_ascii=False, indent=2))
