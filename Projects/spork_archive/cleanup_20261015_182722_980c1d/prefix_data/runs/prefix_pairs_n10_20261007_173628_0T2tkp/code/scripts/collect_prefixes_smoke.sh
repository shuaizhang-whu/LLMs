#!/usr/bin/env bash
# External vLLM server must already be running. No training, ten prefixes only.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
SPORK_PYTHON=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=''
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
SPORK_PREFIX_DATA_ROOT=${SPORK_PREFIX_DATA_ROOT:-/mnt/data/zhangshuai/LLMs/Projects/spork/prefix_data}
mkdir -p "$SPORK_PREFIX_DATA_ROOT/runs"
SPORK_PREFIX_RUN=$(mktemp -d "$SPORK_PREFIX_DATA_ROOT/runs/prefix_pairs_n10_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$SPORK_PREFIX_RUN/run.log") 2>&1
printf 'Prefix collection results: %s\n' "$SPORK_PREFIX_RUN"
"$SPORK_PYTHON" scripts/collect_prefix_pairs.py \
  --model-url "${SPORK_PREFIX_URL:-http://127.0.0.1:8016/v1}" \
  --model /mnt/data/zhangshuai/LLMs/Models/Qwen3-32B \
  --output "$SPORK_PREFIX_RUN" --prefixes 10 --prefix-tokens 32 \
  --delay-tokens 64 --max-turns-per-task 2 --max-tokens 8192 \
  --context-window 40960 --tool-floor-s 2 --seed 42 "$@"
