#!/usr/bin/env bash
# Phase 2 on batch-invariant kernels with the code frozen for E1/E2; start it yourself in tmux train.
#   train          the 109 train tasks x nospec,spork,raw (with E2 this covers all 155 tasks)
#   tool10, tool20 tool latency 10 s / 20 s on the 43 tasks that completed in E2, five configurations
# Default only freezes the plan. Keep the server that ran E2: a restarted server yields other tokens.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
SPORK_P2_STAGE="${1:-}"
SPORK_P2_MODE="${2:---prepare}"
if [[ $# -gt 2 || ! "$SPORK_P2_STAGE" =~ ^(train|tool10|tool20)$ || ! "$SPORK_P2_MODE" =~ ^--(prepare|run)$ ]]; then
    printf 'Usage: bash scripts/run_phase2_det.sh train|tool10|tool20 [--prepare|--run]\n' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=''
SPORK_P2_SPLIT="$SPORK_CODEX_ROOT/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json"
if [[ "$SPORK_P2_STAGE" == train ]]; then
    SPORK_P2_TAG=e2b_train_qwen32_invariant_v1
    SPORK_P2_SPLITS=train
    SPORK_P2_SKIP=''
    SPORK_P2_CONFIGS=nospec,spork,raw
    SPORK_P2_FLOOR=2
else
    SPORK_P2_FLOOR="${SPORK_P2_STAGE#tool}"
    SPORK_P2_TAG="e5_tool${SPORK_P2_FLOOR}_qwen32_invariant_v1"
    SPORK_P2_SPLITS=validation,test
    # These hit the token limit in every E2 configuration; tool latency does not change the tokens.
    SPORK_P2_SKIP=airline:25,retail:7,retail:111
    SPORK_P2_CONFIGS=nospec,spork,raw,actor_only,raw_no_probe
fi
SPORK_P2_TASKS=$("$SPORK_CODEX_ENV/bin/python" -c '
import json, sys
splits, skip = sys.argv[2].split(","), sys.argv[3].split(",")
tasks = [t for t in json.load(open(sys.argv[1]))["tasks"] if t["domain"] in ("airline", "retail")
         and t["split"] in splits and t["group_id"] not in skip]
print(",".join(t["group_id"] for t in sorted(tasks, key=lambda t: (t["domain"], int(t["task_id"])))))
' "$SPORK_P2_SPLIT" "$SPORK_P2_SPLITS" "$SPORK_P2_SKIP")
SPORK_P2_OUTPUT="$SPORK_CODEX_ROOT/results/$SPORK_P2_TAG"
SPORK_P2_RUN_ARGS=()
if [[ "$SPORK_P2_MODE" == --run ]]; then
    "$SPORK_CODEX_ENV/bin/python" scripts/check_codex_runtime.py --server --serving-mode invariant
    SPORK_P2_RUN_ARGS=(--run)
fi
mkdir -p "$SPORK_CODEX_ROOT/results"
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_successor_full_eval.py \
    --output "$SPORK_P2_OUTPUT" --scope all --task-ids "$SPORK_P2_TASKS" --configs "$SPORK_P2_CONFIGS" \
    --max-turns 8 --repetitions 1 --seed 42 --max-tokens 8192 --tool-floor "$SPORK_P2_FLOOR" \
    --serving-mode invariant --model "$SPORK_CODEX_MODEL" --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1" \
    --task-split "$SPORK_P2_SPLIT" "${SPORK_P2_RUN_ARGS[@]}" 2>&1 | tee -a "$SPORK_P2_OUTPUT.launch.log"
