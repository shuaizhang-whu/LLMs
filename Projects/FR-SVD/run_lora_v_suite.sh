#!/bin/bash
set -euo pipefail

GPU="${GPU:-4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUITE_OUTPUT="${SUITE_OUTPUT:-/data/zhangs/LLMs/FR-SVD/result/FR_SVD_v1.4_lora_v_suite}"

mkdir -p "$SUITE_OUTPUT"

echo "[suite] GPU=$GPU"
echo "[suite] output=$SUITE_OUTPUT"

bash "$SCRIPT_DIR/run_lora_vonly_baseline.sh" \
    --gpu "$GPU" \
    --output_base "$SUITE_OUTPUT/v_only" 2>&1 | tee "$SUITE_OUTPUT/v_only.stdout.log"

bash "$SCRIPT_DIR/run_lora_v_then_u.sh" \
    --gpu "$GPU" \
    --output_base "$SUITE_OUTPUT/v_then_u" 2>&1 | tee "$SUITE_OUTPUT/v_then_u.stdout.log"
