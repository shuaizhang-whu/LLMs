#!/usr/bin/env python3
"""CLI for the local deterministic Chess baseline runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Optional, Sequence

from chess_runner import ChessExperimentRunner, sanitize_json, trace_to_dict
from local_runtime import ChessLocalRuntime, EndpointConfig


RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
BASELINE_COMMIT = "dc938b9ef7474caf07fe4ad16549c1fa8c7d268c"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for package in ("chess", "openai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            sanitize_json(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_gpu_ids(raw: Optional[str], count: int) -> Optional[list[int]]:
    if raw is None:
        return None
    pieces = [piece.strip() for piece in raw.split(",")]
    if any(not piece or not piece.isdigit() for piece in pieces):
        raise ValueError("--gpu-ids 必须是逗号分隔的非负整数")
    ids = [int(piece) for piece in pieces]
    if len(ids) != count:
        raise ValueError("--gpu-ids 数量必须与 --gpu-count 一致")
    if len(ids) != len(set(ids)):
        raise ValueError("--gpu-ids 不允许重复")
    return ids


def _artifact_reference(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"launcher 产物不存在或不是文件: {resolved}")
    digest = sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(resolved), "sha256": digest.hexdigest()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local sequential or truly concurrent speculative Chess."
    )
    parser.add_argument(
        "--execution-mode", choices=("sequential", "speculative"), required=True
    )
    parser.add_argument("--actor-base-url", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--speculator-base-url", required=True)
    parser.add_argument("--speculator-model", required=True)
    parser.add_argument("--num-guesses", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--seed", type=int, default=248)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stop-after", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("reproduction/local_runs"))
    parser.add_argument(
        "--gpu-count",
        type=int,
        choices=(2, 3, 4),
        required=True,
        help="必须显式记录本次已由统一 launcher 分配的 GPU 数量",
    )
    parser.add_argument(
        "--gpu-ids",
        help="可选：统一 launcher 实际分配的物理 GPU，例如 0,2,3",
    )
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        help="可选：统一 launcher 的 experiment_manifest.json，只读记录路径和哈希",
    )
    parser.add_argument(
        "--topology-path",
        type=Path,
        help="可选：统一 launcher 的 gpu_topology.json，只读记录路径和哈希",
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    return parser


def _initial_manifest(
    args: argparse.Namespace,
    gpu_ids: Optional[list[int]],
    launcher_artifacts: dict[str, dict[str, str]],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "baseline_commit": BASELINE_COMMIT,
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "command": [str(item) for item in sys.argv],
        "configuration": {
            "execution_mode": args.execution_mode,
            "actor": {"base_url": args.actor_base_url, "model": args.actor_model},
            "speculator": {
                "base_url": args.speculator_base_url,
                "model": args.speculator_model,
            },
            "num_guesses": args.num_guesses,
            "seed": args.seed,
            "stop_after": args.stop_after,
            "gpu_count": args.gpu_count,
            "gpu_ids": gpu_ids,
        },
        "notes": {
            "queue_time": (
                "Local OpenAI-compatible streaming does not expose server queue time; "
                "trace.queue_seconds remains null rather than being inferred."
            ),
            "gpu_preflight": (
                "GPU availability and UUID/topology belong to the unified launcher "
                "preflight manifest; this runner only records the allocated count/ids."
            ),
        },
    }
    if launcher_artifacts:
        manifest["launcher_artifacts"] = launcher_artifacts
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stop_after <= 0:
        parser.error("--stop-after 必须大于 0")
    if args.request_timeout <= 0:
        parser.error("--request-timeout 必须大于 0")
    if not RUN_ID_RE.fullmatch(args.run_id):
        parser.error("--run-id 仅允许字母、数字、点、下划线和连字符，且长度不超过 128")
    try:
        gpu_ids = _parse_gpu_ids(args.gpu_ids, args.gpu_count)
    except ValueError as exc:
        parser.error(str(exc))
    launcher_artifacts: dict[str, dict[str, str]] = {}
    try:
        if args.experiment_manifest is not None:
            launcher_artifacts["experiment_manifest"] = _artifact_reference(
                args.experiment_manifest
            )
        if args.topology_path is not None:
            launcher_artifacts["gpu_topology"] = _artifact_reference(args.topology_path)
    except ValueError as exc:
        parser.error(str(exc))

    run_dir = args.output_dir.resolve() / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"运行目录已存在且非空，拒绝覆盖: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _initial_manifest(args, gpu_ids, launcher_artifacts)
    manifest["run_directory"] = str(run_dir)
    _atomic_json(run_dir / "manifest.json", manifest)

    runtime: Optional[ChessLocalRuntime] = None
    try:
        runtime = ChessLocalRuntime(
            EndpointConfig(
                args.actor_base_url,
                args.actor_model,
                timeout_seconds=args.request_timeout,
            ),
            EndpointConfig(
                args.speculator_base_url,
                args.speculator_model,
                timeout_seconds=args.request_timeout,
            ),
        )
        runner = ChessExperimentRunner(
            runtime,
            actor_model=args.actor_model,
            speculator_model=args.speculator_model,
            execution_mode=args.execution_mode,
            num_guesses=args.num_guesses,
            run_seed=args.seed,
            stop_after=args.stop_after,
        )
        result = runner.run()
        _atomic_json(run_dir / "steps.json", [asdict(step) for step in result.steps])
        _atomic_json(run_dir / "traces.json", [trace_to_dict(trace) for trace in result.traces])
        _atomic_json(run_dir / "metrics.json", result.metrics)
        manifest.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "artifacts": {
                    "steps": "steps.json",
                    "traces": "traces.json",
                    "metrics": "metrics.json",
                },
                "summary": {
                    "steps_completed": len(result.steps),
                    "termination_reason": result.termination_reason,
                    "final_fen": result.final_fen,
                    "action_sequence_sha256": result.action_sequence_sha256,
                    "game_wall_clock_seconds": result.game_wall_clock_seconds,
                    "background_drain_wall_clock_seconds": (
                        result.background_drain_wall_clock_seconds
                    ),
                },
            }
        )
        _atomic_json(run_dir / "manifest.json", manifest)
        print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        if runtime is not None:
            traces = runtime.recorder.snapshot()
            _atomic_json(run_dir / "traces.json", [trace_to_dict(trace) for trace in traces])
            manifest["artifacts"] = {"traces": "traces.json"}
        _atomic_json(run_dir / "manifest.json", manifest)
        print(f"Chess experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
