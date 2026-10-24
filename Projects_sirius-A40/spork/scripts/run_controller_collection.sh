#!/usr/bin/env bash
# User starts in tmux. Default: the first 16 preassigned train/validation tasks.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/spork_env.sh"
SPORK_DATA_PYTHON=${SPORK_DATA_PYTHON:-$SPORK_PYTHON}
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1
if (( $# < 1 || $# > 2 )); then
  printf 'Usage: bash %s prepared-dataset-directory [batch-index]\n' "$0" >&2
  exit 1
fi
SPORK_DATA_ROOT=$1
SPORK_DATA_BATCH=${2:-0}
"$SPORK_DATA_PYTHON" "$SPORK_REPO_ROOT/scripts/collect_controller_dataset.py" \
  --output "$SPORK_DATA_ROOT" --batch "$SPORK_DATA_BATCH" --prepare-only
SPORK_DATA_LOG=$(mktemp "$SPORK_DATA_ROOT/collection_batch${SPORK_DATA_BATCH}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
exec > >(tee -a "$SPORK_DATA_LOG") 2>&1
"$SPORK_DATA_PYTHON" "$SPORK_REPO_ROOT/scripts/collect_controller_dataset.py" --output "$SPORK_DATA_ROOT" --batch "$SPORK_DATA_BATCH"
printf '数据采集完成。下一步读取 collection_report.json，检查标签覆盖率与精度后训练。\n'
