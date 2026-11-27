#!/usr/bin/env bash
# Full Qwen3.5-9B stable-refresh evaluation on the reference GDN server.
# Launch in tmux train; the runner resumes sealed records on re-run.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
OUT="${1:-results/qwen35_refresh_full_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$OUT" ]]; then
  printf 'Refusing to reuse existing output directory: %s\n' "$OUT" >&2
  exit 2
fi
PYTHONPATH=. /mnt/data/zhangshuai/anaconda3/envs/spork/bin/python \
  scripts/run_qwen35_refresh_full_eval.py \
  --output "$OUT" \
  --configs spork,eta,refresh,off \
  --scope all \
  --max-turns 8 \
  --repetitions 1 \
  --probe-mode name \
  --check-interval 50 \
  --refresh-policy read_only \
  2>&1 | tee "$OUT.launch.log"
