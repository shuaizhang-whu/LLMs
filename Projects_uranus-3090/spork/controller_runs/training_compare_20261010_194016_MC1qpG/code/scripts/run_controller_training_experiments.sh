#!/usr/bin/env bash
# User launches in tmux. Local encoder only; vLLM is not needed.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
SPORK_COMPARE_PYTHON=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python
SPORK_COMPARE_ENCODER=/mnt/data/zhangshuai/LLMs/Models/bert-tiny
if (( $# != 1 )) || [[ ! -f "$1/training_data_gate.json" ]]; then
  printf 'Usage: bash %s completed-stage3-dataset-directory\n' "$0" >&2
  exit 1
fi
SPORK_COMPARE_DATA=$1
SPORK_COMPARE_DEVICE=${CONTROLLER_DEVICE:-cpu}
if [[ "$SPORK_COMPARE_DEVICE" != cpu && "$SPORK_COMPARE_DEVICE" != cuda ]]; then
  printf 'CONTROLLER_DEVICE must be cpu or cuda.\n' >&2
  exit 1
fi
if [[ "$SPORK_COMPARE_DEVICE" == cpu ]]; then
  export CUDA_VISIBLE_DEVICES=''
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf 'For GPU training, specify CUDA_VISIBLE_DEVICES with your available GPU.\n' >&2
  exit 1
fi
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p controller_runs
SPORK_COMPARE_LOG=$(mktemp "$PWD/controller_runs/training_compare_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
SPORK_COMPARE_OUTPUT=${SPORK_COMPARE_LOG%.log}
exec > >(tee -a "$SPORK_COMPARE_LOG") 2>&1
printf '结果目录：%s\n日志：%s\n' "$SPORK_COMPARE_OUTPUT" "$SPORK_COMPARE_LOG"
"$SPORK_COMPARE_PYTHON" scripts/run_controller_training_experiments.py \
  --dataset "$SPORK_COMPARE_DATA" --encoder "$SPORK_COMPARE_ENCODER" \
  --output "$SPORK_COMPARE_OUTPUT" --device "$SPORK_COMPARE_DEVICE" \
  --cpu-threads 4 --epochs 20 --batch-size 8 --seeds 42 43 44
printf '完成。请查看：%s/summary.md\n' "$SPORK_COMPARE_OUTPUT"
