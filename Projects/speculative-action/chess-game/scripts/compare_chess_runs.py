#!/usr/bin/env python3
"""比较同一 seed 的 Chess sequential/speculative 轨迹是否严格等价。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence


def _metrics_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    candidates = sorted(resolved.rglob("metrics.json")) if resolved.is_dir() else []
    if len(candidates) != 1:
        raise ValueError(
            f"{resolved} 下应恰有一个 metrics.json，实际找到 {len(candidates)} 个"
        )
    return candidates[0]


def _load(path: Path) -> tuple[Path, dict[str, Any]]:
    metrics_path = _metrics_path(path)
    with metrics_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
        raise ValueError(f"指标文件格式错误: {metrics_path}")
    return metrics_path, payload


def compare_runs(sequential: Path, speculative: Path) -> dict[str, Any]:
    sequential_path, seq = _load(sequential)
    speculative_path, spec = _load(speculative)
    if seq.get("execution_mode") != "sequential":
        raise ValueError("第一个输入不是 sequential 运行")
    if spec.get("execution_mode") != "speculative":
        raise ValueError("第二个输入不是 speculative 运行")

    seq_actions = seq["actions"]
    spec_actions = spec["actions"]
    first_mismatch = None
    for index in range(max(len(seq_actions), len(spec_actions))):
        left = seq_actions[index] if index < len(seq_actions) else None
        right = spec_actions[index] if index < len(spec_actions) else None
        if left != right:
            first_mismatch = {
                "step": index,
                "sequential_action": left,
                "speculative_action": right,
            }
            break

    checks = {
        "actions_equal": seq_actions == spec_actions,
        "action_sequence_sha256_equal": (
            seq.get("action_sequence_sha256") == spec.get("action_sequence_sha256")
        ),
        "final_fen_equal": seq.get("final_fen") == spec.get("final_fen"),
        "steps_completed_equal": seq.get("steps_completed") == spec.get("steps_completed"),
    }
    return {
        "schema_version": "1.0",
        "sequential_metrics": str(sequential_path),
        "speculative_metrics": str(speculative_path),
        "checks": checks,
        "first_mismatch": first_mismatch,
        "lossless_pass": all(checks.values()),
        "scope": (
            "Deterministic action/FEN replay equivalence; this does not by itself "
            "establish the paper-wide lossless claim."
        ),
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-run", type=Path, required=True)
    parser.add_argument("--speculative-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compare_runs(args.sequential_run, args.speculative_run)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    if args.output is not None:
        _write_atomic(args.output.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["lossless_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
