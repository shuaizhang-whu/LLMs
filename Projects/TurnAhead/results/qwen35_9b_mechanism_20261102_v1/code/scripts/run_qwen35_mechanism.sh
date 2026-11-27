#!/usr/bin/env bash
set -euo pipefail
export SPORK_Q35_REFERENCE_GDN=1
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen35_env.sh"
cd "$SPORK_CODEX_ROOT"
export CUDA_VISIBLE_DEVICES=''
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--run' ) ]]; then
    printf 'Usage: bash scripts/run_qwen35_mechanism.sh [--run]\n' >&2
    exit 2
fi
SPORK_MECHANISM_OUT="$SPORK_CODEX_ROOT/results/${SPORK_Q35_MECHANISM_TAG:-qwen35_9b_mechanism_20261102_v1}"
mkdir -p "$SPORK_MECHANISM_OUT"
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_qwen35_mechanism.py \
    --output "$SPORK_MECHANISM_OUT" --model "$SPORK_CODEX_MODEL" \
    --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1" "$@" \
    2>&1 | tee -a "$SPORK_MECHANISM_OUT.launch.log"
