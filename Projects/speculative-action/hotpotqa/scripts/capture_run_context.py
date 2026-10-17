"""保存复现实验所需的版本、diff、GPU、依赖和产物哈希。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def run_command(args, cwd=ROOT):
    completed = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_hashes(roots):
    records = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            records.append(
                {
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                    "path": str(path),
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-roots", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_roots = [
        item.strip() for item in args.artifact_roots.split(",") if item.strip()
    ]
    git_root_result = run_command(["git", "rev-parse", "--show-toplevel"])
    git_root = Path(git_root_result["stdout"].strip())
    hotpotqa_relative = str(ROOT.relative_to(git_root))

    git_context = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(git_root),
        "head": run_command(["git", "rev-parse", "HEAD"], git_root),
        "branch": run_command(["git", "branch", "--show-current"], git_root),
        "remote": run_command(["git", "remote", "-v"], git_root),
        "status": run_command(["git", "status", "--short"], git_root),
        "hotpotqa_diff": run_command(
            ["git", "diff", "--", hotpotqa_relative],
            git_root,
        ),
    }
    (output_dir / "git_context.json").write_text(
        json.dumps(git_context, indent=2),
        encoding="utf-8",
    )
    (output_dir / "hotpotqa.diff").write_text(
        git_context["hotpotqa_diff"]["stdout"],
        encoding="utf-8",
    )

    packages = {}
    for package in (
        "beautifulsoup4",
        "google-genai",
        "gymnasium",
        "matplotlib",
        "numpy",
        "openai",
        "pandas",
        "requests",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    environment_context = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "pip_check": run_command([sys.executable, "-m", "pip", "check"]),
        "gpu": run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        "safe_environment_flags": {
            "LOCAL_OPENAI_BASE_URL_set": bool(os.environ.get("LOCAL_OPENAI_BASE_URL")),
            "OPENAI_API_KEY_set": bool(os.environ.get("OPENAI_API_KEY")),
            "GEMINI_API_KEY_set": bool(os.environ.get("GEMINI_API_KEY")),
            "OPENROUTER_API_KEY_set": bool(os.environ.get("OPENROUTER_API_KEY")),
        },
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment_context, indent=2),
        encoding="utf-8",
    )
    (output_dir / "artifact_sha256.json").write_text(
        json.dumps(collect_hashes(artifact_roots), indent=2),
        encoding="utf-8",
    )

    official_hash_file = ROOT / "reproduction" / "offline_20260714" / "source_sha256_after.txt"
    official_check = run_command(
        ["sha256sum", "--check", str(official_hash_file)],
        ROOT,
    )
    (output_dir / "official_trajectory_hash_check.json").write_text(
        json.dumps(official_check, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
