#!/usr/bin/env bash
# Default is preparation only. --run is explicitly user-launched in tmux train.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
SPORK_CODEX_MODE="${1:---prepare}"
if [[ $# -gt 1 || ! "$SPORK_CODEX_MODE" =~ ^--(prepare|run)$ ]]; then
    printf 'Usage: bash scripts/run_codex_full_eval.sh [--prepare|--run]\n' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=''
SPORK_CODEX_OUTPUT="$SPORK_CODEX_ROOT/results/successor_full_20261025_codex"
SPORK_CODEX_RUN_ARGS=()
if [[ "$SPORK_CODEX_MODE" == --run ]]; then
    "$SPORK_CODEX_ENV/bin/python" scripts/check_codex_runtime.py --server
    SPORK_CODEX_RUN_ARGS=(--run)
fi
mkdir -p "$SPORK_CODEX_ROOT/results"
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_successor_full_eval.py \
    --output "$SPORK_CODEX_OUTPUT" --scope all --configs spork,raw \
    --max-turns 8 --repetitions 1 --seed 42 --max-tokens 8192 --tool-floor 2 \
    --model "$SPORK_CODEX_MODEL" --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1" \
    --task-split "$SPORK_CODEX_ROOT/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json" \
    "${SPORK_CODEX_RUN_ARGS[@]}" 2>&1 | tee -a "$SPORK_CODEX_OUTPUT.launch.log"
