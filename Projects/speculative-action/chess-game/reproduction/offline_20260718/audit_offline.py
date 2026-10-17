#!/usr/bin/env python3
"""生成 Chess 官方样例离线重算的可核验 manifest。

只读取官方样例目录和本次工作副本；不会修改官方轨迹。输出均写入本脚本
所在的 ``offline_20260718`` 目录。
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from scipy import stats


OUT = Path(__file__).resolve().parent
CHESS_ROOT = OUT.parents[1]
WORKSPACE = CHESS_ROOT.parents[2]
SOURCE = CHESS_ROOT / "trajectories" / "sample_trajectories"
COPY = OUT / "sample_trajectories"
BASELINE_COMMIT = "dc938b9ef7474caf07fe4ad16549c1fa8c7d268c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {
        path.relative_to(SOURCE).as_posix(): sha256(path)
        for path in sorted(SOURCE.rglob("*"))
        if path.is_file()
    }


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def git(*args: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["git", *args], cwd=WORKSPACE, text=True, capture_output=True, check=False
    )
    return completed.returncode, completed.stdout.strip()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("chess", "matplotlib", "numpy", "openai", "scipy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def read_step_30() -> dict[int, list[dict[str, float]]]:
    grouped: dict[int, list[dict[str, float]]] = {1: [], 2: [], 3: []}
    for csv_path in sorted(COPY.glob("*/analysis_results.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["target_steps"]) != 30:
                    continue
                k = int(row["num_predictions"])
                grouped[k].append(
                    {
                        "step_accuracy": 100.0 * float(row["accuracy"]),
                        "window_accuracy": 100.0 * float(row["speculative_window_accuracy"]),
                        "time_saved": float(row["time_saved_percentage"]),
                    }
                )
    return grouped


def mean_sem(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    ci95 = float(stats.t.ppf(0.975, len(values) - 1) * sem) if len(values) > 1 else 0.0
    return {
        "mean_percent": mean,
        "sem_percent": sem,
        "ci95_half_width_percent": ci95,
    }


def aggregate() -> dict[str, object]:
    result: dict[str, object] = {}
    for k, rows in read_step_30().items():
        result[str(k)] = {
            "n_games": len(rows),
            "step_accuracy": mean_sem([row["step_accuracy"] for row in rows]),
            "window_accuracy": mean_sem([row["window_accuracy"] for row in rows]),
            "estimated_time_saved": mean_sem([row["time_saved"] for row in rows]),
        }
    return result


def main() -> int:
    hashes = source_hashes()
    missing = [name for name in hashes if not (COPY / name).is_file()]
    mismatched = [
        name
        for name, digest in hashes.items()
        if (COPY / name).is_file() and sha256(COPY / name) != digest
    ]
    pdfs = sorted(OUT.rglob("*.pdf"))
    invalid_pdfs = [
        path.relative_to(OUT).as_posix()
        for path in pdfs
        if path.stat().st_size == 0 or path.read_bytes()[:5] != b"%PDF-"
    ]
    _, head = git("rev-parse", "HEAD")
    source_diff_rc, _ = git(
        "diff", "--quiet", "--", "Projects/speculative-action/chess-game/trajectories/sample_trajectories"
    )

    manifest = {
        "schema_version": "1.0",
        "scope": "Chess official five-sample offline metric/plot recomputation",
        "baseline_commit": BASELINE_COMMIT,
        "workspace_head": head,
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "gpu_used": False,
        "network_or_api_used": False,
        "source_directory": str(SOURCE),
        "working_copy": str(COPY),
        "source_file_count": len(hashes),
        "source_copy_missing": missing,
        "source_copy_sha256_mismatches": mismatched,
        "tracked_source_diff_exit_code": source_diff_rc,
        "trajectory_count": len([path for path in COPY.iterdir() if path.is_dir()]),
        "pdf_count": len(pdfs),
        "invalid_pdfs": invalid_pdfs,
        "step_30_summary": aggregate(),
        "interpretation": {
            "time_saved_is": "offline ideal-formula estimate from saved request timings",
            "time_saved_is_not": "measured online end-to-end wall-clock speedup",
        },
    }
    hash_lines = "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    write_atomic(OUT / "source_sha256.txt", hash_lines)
    write_atomic(OUT / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    ok = not missing and not mismatched and not invalid_pdfs and source_diff_rc == 0
    print(json.dumps({"ok": ok, **manifest}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
