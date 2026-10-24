#!/usr/bin/env bash
# User starts in tmux. Batches -> quality gates -> first pretrained controller fit.
# The 4-GPU vLLM server is already running; the small controller trains on CPU.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/spork_env.sh"
SPORK_STAGE3_PYTHON=${SPORK_STAGE3_PYTHON:-$SPORK_PYTHON}
SPORK_STAGE3_ENCODER=${SPORK_STAGE3_ENCODER:-$SPORK_ENCODER}
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if (( $# != 1 )) || [[ ! -f "$1/manifest.json" ]]; then
  printf 'Usage: bash %s prepared-dataset-directory\n' "$0" >&2
  exit 1
fi
SPORK_STAGE3_DATA=$1
# One owner for the full stage, including the gap between batch locks.
exec 9>"$SPORK_STAGE3_DATA/stage3.lock"
flock -n 9 || { printf '该数据集已有第三阶段流程在运行。\n' >&2; exit 1; }
for SPORK_ENCODER_FILE in config.json model.safetensors vocab.txt; do
  if [[ ! -f "$SPORK_STAGE3_ENCODER/$SPORK_ENCODER_FILE" ]]; then
    printf '请先运行 scripts/download_timing_encoder.sh 下载小编码器。\n' >&2
    exit 1
  fi
done
SPORK_STAGE3_LOG=$(mktemp "$SPORK_STAGE3_DATA/stage3_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
exec > >(tee -a "$SPORK_STAGE3_LOG") 2>&1
"$SPORK_STAGE3_PYTHON" "$SPORK_REPO_ROOT/scripts/collect_controller_dataset.py" --output "$SPORK_STAGE3_DATA" --prepare-only
SPORK_STAGE3_BATCHES=$("$SPORK_STAGE3_PYTHON" - "$SPORK_STAGE3_DATA/manifest.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))['batches']))
PY
)
for (( SPORK_STAGE3_BATCH=0; SPORK_STAGE3_BATCH<SPORK_STAGE3_BATCHES; SPORK_STAGE3_BATCH++ )); do
  bash "$SPORK_REPO_ROOT/scripts/run_controller_collection.sh" "$SPORK_STAGE3_DATA" "$SPORK_STAGE3_BATCH"
  if (( SPORK_STAGE3_BATCH == 0 )); then
    if ! "$SPORK_STAGE3_PYTHON" "$SPORK_REPO_ROOT/scripts/check_controller_training_data.py" --dataset "$SPORK_STAGE3_DATA" --first-batch; then
      printf '首批标签检查未通过，已保留全部数据。请查看 first_batch_gate.json，未继续扩充或训练。\n'
      exit 2
    fi
  fi
done
if ! "$SPORK_STAGE3_PYTHON" "$SPORK_REPO_ROOT/scripts/check_controller_training_data.py" --dataset "$SPORK_STAGE3_DATA"; then
  printf '数据采集完成，但训练数据检查未通过。请查看 training_data_gate.json；未开始训练。\n'
  exit 2
fi
if [[ -f "$SPORK_STAGE3_DATA/controller_training_path.txt" ]]; then
  SPORK_STAGE3_TRAIN=$(<"$SPORK_STAGE3_DATA/controller_training_path.txt")
  if [[ -f "$SPORK_STAGE3_TRAIN/training_report.json" ]]; then
    printf '已有完成的训练结果：%s\n' "$SPORK_STAGE3_TRAIN"
    exit 0
  fi
fi
mkdir -p "$SPORK_REPO_ROOT/controller_runs"
# Training directories must not exist when train_timing_controller.py starts.
SPORK_STAGE3_TRAIN=$("$SPORK_STAGE3_PYTHON" - <<'PY'
from datetime import datetime
from pathlib import Path
import os
import uuid
print(Path(os.environ['SPORK_REPO_ROOT']) / 'controller_runs' /
      ('stage3_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]))
PY
)
printf '%s\n' "$SPORK_STAGE3_TRAIN" > "$SPORK_STAGE3_DATA/controller_training_path.txt"
"$SPORK_STAGE3_PYTHON" "$SPORK_REPO_ROOT/scripts/train_timing_controller.py" \
  --mode train --device cpu --cpu-threads 4 --encoder "$SPORK_STAGE3_ENCODER" \
  --data "$SPORK_STAGE3_DATA/training_rows.jsonl" --output "$SPORK_STAGE3_TRAIN" \
  --epochs 3 --batch-size 8 --seed 42
printf '首轮预训练编码器微调已完成：%s/training_report.json\n' "$SPORK_STAGE3_TRAIN"
