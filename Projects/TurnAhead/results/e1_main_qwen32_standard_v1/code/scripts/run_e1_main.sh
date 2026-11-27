#!/usr/bin/env bash
# E1: no speculation vs. speculative tool dispatch vs. TurnAhead, 155 tau2 tasks, standard kernels.
# Default only freezes the plan. Pass --run yourself in tmux train.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
SPORK_E1_MODE="${1:---prepare}"
if [[ $# -gt 1 || ! "$SPORK_E1_MODE" =~ ^--(prepare|run)$ ]]; then
    printf 'Usage: bash scripts/run_e1_main.sh [--prepare|--run]\n' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=''
SPORK_E1_OUTPUT="$SPORK_CODEX_ROOT/results/${SPORK_E1_TAG:-e1_main_qwen32_standard_v1}"
SPORK_E1_RUN_ARGS=()
if [[ "$SPORK_E1_MODE" == --run ]]; then
    "$SPORK_CODEX_ENV/bin/python" scripts/check_codex_runtime.py --server --serving-mode standard
    SPORK_E1_RUN_ARGS=(--run)
fi
mkdir -p "$SPORK_CODEX_ROOT/results"
# Repetition 0 of all tasks completes before repetition 1 starts; a stopped run resumes.
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_successor_full_eval.py \
    --output "$SPORK_E1_OUTPUT" --scope all --configs nospec,spork,raw \
    --max-turns 8 --repetitions "${SPORK_E1_REPETITIONS:-2}" --seed 42 --max-tokens 8192 --tool-floor 2 \
    --serving-mode standard \
    --model "$SPORK_CODEX_MODEL" --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1" \
    --task-split "$SPORK_CODEX_ROOT/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json" \
    "${SPORK_E1_RUN_ARGS[@]}" 2>&1 | tee -a "$SPORK_E1_OUTPUT.launch.log"
