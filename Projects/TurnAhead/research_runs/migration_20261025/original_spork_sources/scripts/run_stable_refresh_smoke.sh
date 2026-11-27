#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
SPORK_REFRESH_OUTPUT=${1:?Provide a new absolute output directory}
shift
if [[ -e "$SPORK_REFRESH_OUTPUT" ]]; then
  printf 'Output already exists: %s\n' "$SPORK_REFRESH_OUTPUT" >&2
  exit 1
fi
mkdir -p "$(dirname "$SPORK_REFRESH_OUTPUT")"
/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python scripts/run_stable_refresh_smoke.py \
  --output "$SPORK_REFRESH_OUTPUT" "$@" 2>&1 | tee "$SPORK_REFRESH_OUTPUT.launch.log"
