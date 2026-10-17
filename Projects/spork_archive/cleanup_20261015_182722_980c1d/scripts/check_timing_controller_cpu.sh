#!/usr/bin/env bash
# One-batch CPU functional check; no vLLM, GPU, model download or formal training.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
SPORK_CONTROLLER_PYTHON=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
SPORK_CONTROLLER_DATA=$("$SPORK_CONTROLLER_PYTHON" - <<'PY'
import json
from pathlib import Path
root=Path('/mnt/data/zhangshuai/LLMs/Projects/spork/prefix_data')
index=json.loads((root/'index.json').read_text())
run=index.get('latest_validated_run')
if not run:
    raise SystemExit('No validated prefix dataset is registered.')
print(root/'runs'/run/'training_rows.jsonl')
PY
)
mkdir -p "$PWD/controller_runs"
SPORK_CONTROLLER_RUN=$(mktemp -d "$PWD/controller_runs/cpu_check_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$SPORK_CONTROLLER_RUN/run.log") 2>&1
printf 'CPU check output: %s\n' "$SPORK_CONTROLLER_RUN"
"$SPORK_CONTROLLER_PYTHON" scripts/train_timing_controller.py \
  --mode cpu-check --device cpu --data "$SPORK_CONTROLLER_DATA" \
  --output "$SPORK_CONTROLLER_RUN/artifacts" --max-length 128 --head-hidden-size 32
