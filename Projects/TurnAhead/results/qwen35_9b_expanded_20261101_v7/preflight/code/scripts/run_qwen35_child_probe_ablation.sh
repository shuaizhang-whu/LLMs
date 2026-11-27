#!/usr/bin/env bash
# One bounded follow-up: same eight train tasks, three arms, two repeats.
set -euo pipefail
export SPORK_Q35_REFERENCE_GDN=1
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen35_env.sh"
cd "$SPORK_CODEX_ROOT"
export CUDA_VISIBLE_DEVICES=''
SPORK_Q35_PROBE_STAGE="${1:-prepare}"
if [[ $# -gt 1 || ! "$SPORK_Q35_PROBE_STAGE" =~ ^(prepare|run)$ ]]; then
    printf 'Usage: bash scripts/run_qwen35_child_probe_ablation.sh [prepare|run]\n' >&2
    exit 2
fi
SPORK_Q35_PROBE_OUTPUT="${SPORK_Q35_CHILD_PROBE_OUTPUT:-$SPORK_CODEX_ROOT/results/qwen35_9b_20261101_gdnref_v4_child_probe}"
SPORK_Q35_PROBE_ARGS=(
    --model "$SPORK_CODEX_MODEL" --model-family qwen35
    --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1"
    --task-split "$SPORK_CODEX_ROOT/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json"
    --scope train --configs spork,raw,raw_no_probe
    --task-ids airline:2,airline:3,airline:4,airline:6,retail:0,retail:1,retail:3,retail:4
    --repetitions 2 --max-turns 2 --max-tokens 8192 --tool-floor 2 --seed 42
    --presence-penalty "$SPORK_Q35_PRESENCE_PENALTY"
    --require-identical-trajectories --output "$SPORK_Q35_PROBE_OUTPUT"
)
if [[ "$SPORK_Q35_PROBE_STAGE" == run ]]; then
    "$SPORK_CODEX_ENV/bin/python" scripts/check_qwen35_experiment.py --server
    SPORK_Q35_PROBE_ARGS+=(--run)
else
    "$SPORK_CODEX_ENV/bin/python" scripts/check_qwen35_experiment.py --local
fi
mkdir -p results
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_successor_full_eval.py \
    "${SPORK_Q35_PROBE_ARGS[@]}" 2>&1 | tee -a "${SPORK_Q35_PROBE_OUTPUT}.launch.log"
if [[ "$SPORK_Q35_PROBE_STAGE" == run ]]; then
    "$SPORK_CODEX_ENV/bin/python" scripts/check_qwen35_experiment.py \
        --results "$SPORK_Q35_PROBE_OUTPUT" --stage ablation
    "$SPORK_CODEX_ENV/bin/python" scripts/analyze_qwen35_formal.py "$SPORK_Q35_PROBE_OUTPUT"
fi
