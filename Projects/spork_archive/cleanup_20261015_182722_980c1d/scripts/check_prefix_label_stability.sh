#!/usr/bin/env bash
# Run in a user-owned tmux session, against the dedicated existing D3 server.
# Optional argument: a prepared/interrupted run directory to resume.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
SPORK_STABILITY_PYTHON=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python
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
  SPORK_STABILITY_RUN=$1
  if [[ ! -f "$SPORK_STABILITY_RUN/manifest.json" ]]; then
    printf 'Prepared run manifest not found: %s\n' "$SPORK_STABILITY_RUN" >&2
    exit 1
  fi
  SPORK_STABILITY_ARGS=(--resume)
else
  mkdir -p "$PWD/prefix_data/stability_runs"
  SPORK_STABILITY_RUN=$(mktemp -d "$PWD/prefix_data/stability_runs/stability_$(date +%Y%m%d_%H%M%S)_XXXXXX")
  SPORK_STABILITY_ARGS=()
  if [[ -n "${SPORK_PREFIX_URL:-}" ]]; then
    SPORK_STABILITY_ARGS+=(--model-url "$SPORK_PREFIX_URL")
  fi
fi
exec > >(tee -a "$SPORK_STABILITY_RUN/run.log") 2>&1
printf 'Stability collection: %s\n' "$SPORK_STABILITY_RUN"
"$SPORK_STABILITY_PYTHON" scripts/repeat_prefix_pairs.py \
  --output "$SPORK_STABILITY_RUN" --repetitions 6 "${SPORK_STABILITY_ARGS[@]}"
