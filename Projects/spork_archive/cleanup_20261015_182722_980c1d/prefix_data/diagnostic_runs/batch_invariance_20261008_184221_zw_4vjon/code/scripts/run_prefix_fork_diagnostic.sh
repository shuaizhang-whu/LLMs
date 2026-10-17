#!/usr/bin/env bash
# Start manually in tmux after serve_qwen32_prefix_collection.sh is ready.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
SPORK_FORK_PYTHON=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if (( $# > 1 )); then
  printf 'Usage: bash %s [prepared-run-directory]\n' "$0" >&2
  exit 1
fi
if (( $# == 1 )); then
  SPORK_FORK_RUN=$1
  if [[ ! -f "$SPORK_FORK_RUN/manifest.json" ]]; then
    printf 'Prepared diagnostic manifest not found: %s\n' "$SPORK_FORK_RUN" >&2
    exit 1
  fi
  SPORK_FORK_ARGS=(--resume)
else
  mkdir -p "$PWD/prefix_data/diagnostic_runs"
  SPORK_FORK_RUN=$(mktemp -d "$PWD/prefix_data/diagnostic_runs/generation_fork_$(date +%Y%m%d_%H%M%S)_XXXXXX")
  SPORK_FORK_ARGS=()
  if [[ -n "${SPORK_PREFIX_URL:-}" ]]; then
    SPORK_FORK_ARGS+=(--model-url "$SPORK_PREFIX_URL")
  fi
fi
# Prepare before tee creates run.log in an otherwise empty new directory.
"$SPORK_FORK_PYTHON" scripts/diagnose_prefix_forks.py \
  --output "$SPORK_FORK_RUN" --repetitions 3 --prepare-only "${SPORK_FORK_ARGS[@]}"
exec > >(tee -a "$SPORK_FORK_RUN/run.log") 2>&1
"$SPORK_FORK_PYTHON" scripts/diagnose_prefix_forks.py \
  --output "$SPORK_FORK_RUN" --repetitions 3 --resume
