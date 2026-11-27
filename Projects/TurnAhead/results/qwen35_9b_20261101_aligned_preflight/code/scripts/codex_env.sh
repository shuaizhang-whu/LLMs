#!/usr/bin/env bash
# Source this file to use this checkout's code and benchmark data.
SPORK_CODEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SPORK_CODEX_ENV="${SPORK_CODEX_ENV:-/mnt/data/zhangshuai/anaconda3/envs/spork}"
SPORK_CODEX_MODEL="${SPORK_CODEX_MODEL:-$(dirname -- "$(dirname -- "$SPORK_CODEX_ROOT")")/Models/Qwen3-32B}"
SPORK_CHECK_PORT="${SPORK_CHECK_PORT:-8017}"
export SPORK_CODEX_ROOT SPORK_CODEX_ENV SPORK_CODEX_MODEL SPORK_CHECK_PORT
export SPORK_REPO_ROOT="$SPORK_CODEX_ROOT"
export SPORK_TAU2_ROOT="$SPORK_CODEX_ROOT/tau2-bench"
export SPORK_MODEL_ROOT="$(dirname -- "$SPORK_CODEX_MODEL")"
export TAU2_DATA_DIR="$SPORK_TAU2_ROOT/data"
export PYTHONPATH="$SPORK_CODEX_ROOT:$SPORK_TAU2_ROOT/src"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LITELLM_LOCAL_MODEL_COST_MAP=True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
