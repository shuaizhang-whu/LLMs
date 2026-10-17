#!/usr/bin/env python3
"""CPU-only migration: retain prior evidence in a new, provenance-linked run.

Only orchestration/reporting code may change. Requests, model, intervention,
environment and timing code must still match the original manifest. No model
server is contacted, and source measurements are never edited.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_prefix_forks import (  # noqa: E402
    MODES, completed_triplet, digest, dump, inspect_outcome, prepare, read,
    seal_branch, seal_triplet, snapshot_for,
)

ORCHESTRATION_FILES = {
    "scripts/diagnose_prefix_forks.py", "spork_core/fork_diagnostic.py",
    "scripts/prepare_prefix_invariance_continuation.py", "scripts/continue_prefix_invariance.sh",
}


def migrate(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Continuation must be a separate directory, outside the source run")
    # Refuse to copy a running collection; preserve a coherent snapshot.
    with (source / "run.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read(source / "manifest.json")
        if previous.get("purpose") != "two_prefix_generation_fork_diagnostic" or not previous.get("require_invariance"):
            raise ValueError("Expected a batch-invariance diagnostic source")
        for name, checksum in previous["code_sha256"].items():
            if digest(source / "code" / name) != checksum:
                raise ValueError(f"Archived source code changed: {name}")
            if name not in ORCHESTRATION_FILES and digest(ROOT / name) != checksum:
                raise ValueError(f"Measurement code changed; saved branches cannot be reused: {name}")
        preflights = sorted(source.glob("preflight_*.json"))
        if not preflights or any(read(p).get("d3_status", {}).get("server_config", {}).get(
                "batch_invariant_requested") is not True for p in preflights):
            raise ValueError("Source preflight does not confirm the batch-invariant setting")
        args = argparse.Namespace(output=output, source_run=source, resume=False, analyze_only=False,
                                  repetitions=previous["repetitions"], order_seed=previous["order_seed"],
                                  model_url=previous["model_url"], require_invariance=True, continue_on_truncation=True)
        manifest = prepare(args)
        for key in ("schedule", "sample_ids", "model", "model_url", "timeout_s", "tool_floor_s",
                    "replay_config", "modes", "packages"):
            if manifest[key] != previous[key]:
                raise ValueError(f"Cannot reuse measurements with changed {key}")
        imports = []
        for item in manifest["schedule"]:
            sid, rep = item["sample_id"], item["repetition"]
            src = source / "measurements" / sid / f"repeat_{rep:03d}"
            if not src.exists():
                continue
            complete = completed_triplet(src)  # Validate sealed triplet hashes, if present.
            if complete is not None:
                attempt = src / read(src / "completed.json")["attempt"]
            else:
                attempts = sorted(src.glob("attempt_*"))
                if not attempts:
                    continue
                attempt = attempts[-1]
            if read(attempt / "order.json") != item:
                raise ValueError(f"Source order differs: {attempt}")
            existing = [mode for mode in item["order"] if (attempt / f"{mode}.json").exists()]
            if existing != item["order"][:len(existing)]:
                raise ValueError(f"Source has nonsequential branch measurements: {attempt}")
            snapshot, _ = snapshot_for(source, previous, sid)
            dest = output / "measurements" / sid / f"repeat_{rep:03d}"
            dst_attempt = dest / "attempt_000"
            dst_attempt.mkdir(parents=True)
            shutil.copyfile(attempt / "order.json", dst_attempt / "order.json")
            for mode in existing:
                raw = attempt / f"{mode}.json"
                result = read(raw)
                integrity, fatal = inspect_outcome(mode, result, snapshot, manifest)
                if fatal:
                    raise ValueError(f"Cannot import an invalid measurement: {raw}: {integrity['errors']}")
                shutil.copyfile(raw, dst_attempt / raw.name)
                dump(dst_attempt / f"{mode}_integrity.json", integrity)
                seal_branch(dst_attempt, mode, {"path": str(raw), "sha256": digest(raw)})
                imports.append({"sample_id": sid, "repetition": rep, "mode": mode,
                                "source": str(raw), "sha256": digest(raw), "finish_reason": result["finish_reason"]})
            if len(existing) == len(MODES):
                seal_triplet(dest, dst_attempt, item)
        manifest["continuation"] = {
            "source_run": str(source), "source_manifest_sha256": digest(source / "manifest.json"),
            "prepared_at": datetime.now().isoformat(), "imported_branch_runs": len(imports),
            "remaining_branch_runs": manifest["planned_branch_runs"] - len(imports),
            "scope": "Saved and new timings span an interruption; this is a diagnostic continuation, not training data.",
        }
        provenance = output / "source_evidence"
        provenance.mkdir()
        for path in [source / "manifest.json"] + preflights + sorted(source.glob("failure_*.json")):
            shutil.copyfile(path, provenance / path.name)
        dump(output / "imported_measurements.json", imports)
        dump(output / "manifest.json", manifest)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = migrate(args.source_run, args.output)
    print(f"Prepared: {args.output.resolve()}")
    print(f"Reused {manifest['continuation']['imported_branch_runs']} branches; "
          f"remaining {manifest['continuation']['remaining_branch_runs']}. No inference was started.")


if __name__ == "__main__":
    main()
