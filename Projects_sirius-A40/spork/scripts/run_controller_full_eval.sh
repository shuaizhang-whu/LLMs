#!/usr/bin/env bash
# User starts in tmux. Optional first argument is the new/resume output directory.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/spork_env.sh"
SPORK_FULL_PYTHON=${SPORK_FULL_PYTHON:-$SPORK_PYTHON}
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
mkdir -p "$SPORK_REPO_ROOT/results"
if [[ $# -gt 0 && "$1" != --* ]]; then
  SPORK_FULL_OUTPUT="$1"
  shift
else
  SPORK_FULL_OUTPUT=$(mktemp -d "$SPORK_REPO_ROOT/results/controller_full_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi
SPORK_FULL_LOG=$(mktemp "$SPORK_REPO_ROOT/results/controller_full_launch_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf '正式实验目录：%s\n启动日志：%s\n' "$SPORK_FULL_OUTPUT" "$SPORK_FULL_LOG"
"$SPORK_FULL_PYTHON" "$SPORK_REPO_ROOT/scripts/run_controller_full_eval.py" --output "$SPORK_FULL_OUTPUT" "$@" \
  2>&1 | tee -a "$SPORK_FULL_LOG"
