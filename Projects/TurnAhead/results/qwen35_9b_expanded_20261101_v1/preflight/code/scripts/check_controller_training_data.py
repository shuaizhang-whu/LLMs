#!/usr/bin/env python3
"""CPU-only frozen quality gate and merge. Invalid rows remain in the export."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_controller_dataset import dump, export_batch, read  # noqa: E402


def assess(reports, thresholds, first_batch=False):
    prefix = "first_batch" if first_batch else "final"
    audits = [a for r in reports for a in r["per_prefix_audits"]]
    good = [a for a in audits if a["label_usable"]]
    groups = {s: {a["group_id"] for a in good if a["split"] == s} for s in ("train", "validation")}
    checks = {
        "collection_complete": bool(reports) and all(r["collection_complete"] for r in reports),
        "enough_usable_labels": len(good) >= thresholds[prefix + "_min_usable_prefixes"],
        "enough_train_groups": len(groups["train"]) >= thresholds[prefix + "_min_train_groups"],
        "enough_validation_groups": len(groups["validation"]) >= thresholds[prefix + "_min_validation_groups"],
        "usable_label_coverage": bool(audits) and len(good) / len(audits) >= thresholds["min_usable_fraction"],
        "label_precision": bool(good) and sum(a["meets_initial_precision_criterion"] for a in good) / len(good)
                           >= thresholds["min_precise_fraction"],
        "no_observed_token_forks": sum(r["token_disagreement_prefixes"] for r in reports)
                                   <= thresholds["max_token_disagreement_prefixes"],
        "d3_exercised": sum(r["d3_counter_totals"].get("verified_injections", 0) for r in reports) > 0,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "failed_checks": [k for k, v in checks.items() if not v],
            "usable_prefixes": len(good), "recorded_prefixes": len(audits),
            "usable_groups": {s: len(g) for s, g in groups.items()},
            "scope": "Predeclared three-repeat engineering gate for fitting the controller, not a success/speedup guarantee."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--first-batch", action="store_true")
    args = ap.parse_args()
    root = args.dataset.resolve()
    m = read(root / "manifest.json")
    indices = [0] if args.first_batch else range(len(m["batches"]))
    reports = [export_batch(root, m, i) for i in indices]
    report = assess(reports, m["training_gate"], args.first_batch)
    dump(root / ("first_batch_gate.json" if args.first_batch else "training_data_gate.json"), report)
    if not args.first_batch:
        temp = root / "training_rows.jsonl.tmp"
        temp.write_bytes(b"".join((root / f"batch_{i:03d}/training_rows.jsonl").read_bytes() for i in indices))
        temp.replace(root / "training_rows.jsonl")
    print(f"Gate passed: {report['passed']}; usable prefixes: {report['usable_prefixes']}; "
          f"failed checks: {report['failed_checks']}", flush=True)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
