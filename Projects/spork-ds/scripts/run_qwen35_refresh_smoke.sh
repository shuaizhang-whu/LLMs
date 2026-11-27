#!/usr/bin/env bash
# Qwen3.5-9B stable-refresh smoke in tmux train.
# Usage: bash scripts/run_qwen35_refresh_smoke.sh [output_dir] [max_turns] [repetitions]
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
OUT="${1:-results/qwen35_refresh_smoke_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$OUT" ]]; then
  printf 'Refusing to reuse existing output directory: %s\n' "$OUT" >&2
  exit 2
fi
TASK_IDS="${TASK_IDS:-airline:2,airline:5,airline:4,retail:3,retail:0,airline:3}"
PROBE_MODE="${PROBE_MODE:-name}"
MODES="${MODES:-spork eta refresh off}"
MAX_CHECKS="${MAX_CHECKS:-15}"
PYTHONPATH=. /mnt/data/zhangshuai/anaconda3/envs/spork/bin/python \
  scripts/run_qwen35_refresh_smoke.py \
  --output "$OUT" \
  --task-ids "$TASK_IDS" \
  --max-turns "${2:-1}" \
  --repetitions "${3:-1}" \
  --probe-mode "$PROBE_MODE" \
  --max-checks "$MAX_CHECKS" \
  --keep-going \
  --modes $MODES \
  2>&1 | tee "$OUT.launch.log"
