#!/usr/bin/env bash
# Default only freezes plans; user runs GPU stages in tmux train.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen35_env.sh"
cd "$SPORK_CODEX_ROOT"
export CUDA_VISIBLE_DEVICES=''
SPORK_Q35_STAGE="${1:-prepare}"
if [[ $# -gt 1 || ! "$SPORK_Q35_STAGE" =~ ^(prepare|preflight|ablation)$ ]]; then
    printf 'Usage: bash scripts/run_qwen35_experiments.sh [prepare|preflight|ablation]\n' >&2
    exit 2
fi
SPORK_Q35_BASE="$SPORK_CODEX_ROOT/results/$SPORK_Q35_RUN_TAG"
SPORK_Q35_COMMON=(
    --model "$SPORK_CODEX_MODEL" --model-family qwen35
    --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1"
    --task-split "$SPORK_CODEX_ROOT/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json"
    --scope train --configs spork,serial,actor_only,raw
    --max-turns 2 --max-tokens 8192 --tool-floor 2 --seed 42
    --require-identical-trajectories
)
q35_run() {
    local phase="$1" run_flag="$2" task_ids repetitions
    if [[ "$phase" == preflight ]]; then
        task_ids='airline:2,retail:0'; repetitions=1
    else
        task_ids='airline:2,airline:3,airline:4,airline:6,retail:0,retail:1,retail:3,retail:4'
        repetitions=2
    fi
    local extra=()
    [[ "$run_flag" == run ]] && extra=(--run)
    "$SPORK_CODEX_ENV/bin/python" -u scripts/run_successor_full_eval.py \
        "${SPORK_Q35_COMMON[@]}" --task-ids "$task_ids" --repetitions "$repetitions" \
        --output "${SPORK_Q35_BASE}_${phase}" "${extra[@]}" \
        2>&1 | tee -a "${SPORK_Q35_BASE}_${phase}.launch.log"
}
mkdir -p results
if [[ "$SPORK_Q35_STAGE" == prepare ]]; then
    "$SPORK_CODEX_ENV/bin/python" scripts/check_qwen35_experiment.py --local
    q35_run preflight prepare
    q35_run ablation prepare
    exit 0
fi
"$SPORK_CODEX_ENV/bin/python" scripts/check_qwen35_experiment.py --server
if [[ "$SPORK_Q35_STAGE" == ablation ]]; then
    "$SPORK_CODEX_ENV/bin/python" scripts/check_qwen35_experiment.py \
        --results "${SPORK_Q35_BASE}_preflight" --stage preflight
fi
q35_run "$SPORK_Q35_STAGE" run
"$SPORK_CODEX_ENV/bin/python" scripts/check_qwen35_experiment.py \
    --results "${SPORK_Q35_BASE}_${SPORK_Q35_STAGE}" --stage "$SPORK_Q35_STAGE"
