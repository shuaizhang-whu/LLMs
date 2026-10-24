"""Shared filesystem paths for SPORK.

All roots default to locations relative to the repository and can be
overridden via environment variables so the release is portable:

  SPORK_REPO_ROOT   repository root (default: parent of this package)
  SPORK_MODEL_ROOT  weights directory (../../Models in the LLMs workspace;
                    models/ in a standalone checkout)
  SPORK_TAU2_ROOT   tau2-bench checkout (default: $SPORK_REPO_ROOT/tau2-bench)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get("SPORK_REPO_ROOT", Path(__file__).resolve().parent.parent)
).expanduser().resolve()
CODE_ROOT = REPO_ROOT.parent
RESULTS_ROOT = REPO_ROOT / "results"
LOG_ROOT = REPO_ROOT / "memory" / "logs"
DEFAULT_MODEL_ROOT = (REPO_ROOT.parent.parent / "Models"
                      if REPO_ROOT.parent.name == "Projects" else REPO_ROOT / "models")
MODEL_ROOT = Path(os.environ.get("SPORK_MODEL_ROOT", DEFAULT_MODEL_ROOT)).expanduser()
if not MODEL_ROOT.is_absolute():
    MODEL_ROOT = REPO_ROOT / MODEL_ROOT
MODEL_ROOT = MODEL_ROOT.resolve()
QWEN3_32B_MODEL = MODEL_ROOT / "Qwen3-32B"
TAU2_ROOT = Path(os.environ.get("SPORK_TAU2_ROOT", REPO_ROOT / "tau2-bench")).expanduser()
if not TAU2_ROOT.is_absolute():
    TAU2_ROOT = REPO_ROOT / TAU2_ROOT
TAU2_ROOT = TAU2_ROOT.resolve()
TAU2_SRC = TAU2_ROOT / "src"


def resolve_artifact_path(value: str | Path) -> Path:
    """Locate a copied manifest's local asset without changing its provenance.

    Relative references are project-relative. Historical absolute references
    under Projects/spork or Models are rebased onto this checkout. Other
    absolute paths are left explicit. Callers must still verify frozen hashes;
    relocation is not permission to resume an experiment with changed code.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        return (REPO_ROOT / path).resolve()
    if path.is_relative_to(REPO_ROOT) or path.is_relative_to(MODEL_ROOT):
        return path.resolve()
    parts = path.parts
    for i in range(len(parts) - 1):
        if parts[i:i + 2] == ("Projects", "spork"):
            return (REPO_ROOT.joinpath(*parts[i + 2:])).resolve()
    if "Models" in parts:
        return (MODEL_ROOT.joinpath(*parts[parts.index("Models") + 1:])).resolve()
    return path


def add_runtime_paths() -> None:
    """Make coder scripts and tau2 source imports work."""
    for path in (REPO_ROOT / "coder" / "scripts", TAU2_SRC):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
