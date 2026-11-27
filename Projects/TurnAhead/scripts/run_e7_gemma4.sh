#!/usr/bin/env bash
# E7: Gemma-4-31B-it on tau2 airline+retail, no speculation / speculative tool dispatch / TurnAhead.
# Start it yourself in tmux train, against scripts/serve_gemma4_eval.sh <mode> in tmux vllm.
#   all    2-task smoke, concurrency measurement, then main; invariant also checks the first-turn prompts
#   smoke  2 train tasks, at most 4 turns (invariant: stops at the first trajectory divergence)
#   main   the 46 validation+test tasks (invariant: 1 run each; standard: 2 runs each)
#   rest   the other 109 train tasks; with main, all 155 tasks (invariant only)
# Default only freezes the plan; --run executes. Interrupted stages resume with the same command.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
SPORK_E7_STAGE="${1:-}"
SPORK_E7_SERVING="${2:-}"
SPORK_E7_MODE="${3:---prepare}"
if [[ $# -lt 2 || $# -gt 3 || ! "$SPORK_E7_STAGE" =~ ^(all|smoke|main|rest)$
      || ! "$SPORK_E7_SERVING" =~ ^(invariant|standard)$ || ! "$SPORK_E7_MODE" =~ ^--(prepare|run)$ ]]; then
    printf 'Usage: bash scripts/run_e7_gemma4.sh all|smoke|main|rest invariant|standard [--prepare|--run]\n' >&2
    exit 2
fi
if [[ "$SPORK_E7_STAGE" == rest && "$SPORK_E7_SERVING" != invariant ]]; then
    printf 'The rest stage completes the deterministic 155-task comparison; use invariant.\n' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=''
SPORK_GEMMA_MODEL="${SPORK_GEMMA_MODEL:-$SPORK_MODEL_ROOT/gemma-4-31B-it}"
e7_revision=v3
if [[ "$SPORK_E7_STAGE" == all ]]; then
    e7_run=()
    if [[ "$SPORK_E7_MODE" == --run ]]; then
        e7_run=(--run)
    fi
    e7_base_url="http://127.0.0.1:$SPORK_CHECK_PORT/v1"
    if [[ "$SPORK_E7_SERVING" == invariant ]]; then
        if [[ "$SPORK_E7_MODE" == --run ]]; then
            "$SPORK_CODEX_ENV/bin/python" scripts/check_codex_runtime.py --server --serving-mode invariant
        fi
        "$SPORK_CODEX_ENV/bin/python" -u scripts/check_gemma4_feasibility.py \
            --model "$SPORK_GEMMA_MODEL" --base-url "$e7_base_url" "${e7_run[@]}"
    fi
    e7_script="$SPORK_CODEX_ROOT/scripts/run_e7_gemma4.sh"
    e7_prefix="${SPORK_E7_TAG:-e7_gemma4}"
    SPORK_E7_TAG="${e7_prefix}_smoke_${SPORK_E7_SERVING}_${e7_revision}" bash "$e7_script" smoke "$SPORK_E7_SERVING" "$SPORK_E7_MODE"
    "$SPORK_CODEX_ENV/bin/python" -u scripts/check_concurrency_slowdown.py \
        --model-family gemma4 --model "$SPORK_GEMMA_MODEL" --base-url "$e7_base_url" \
        --output "$SPORK_CODEX_ROOT/results/concurrency_gemma4_${SPORK_E7_SERVING}.json" "${e7_run[@]}"
    export SPORK_E7_TAG="${e7_prefix}_main_${SPORK_E7_SERVING}_${e7_revision}"
    exec bash "$e7_script" main "$SPORK_E7_SERVING" "$SPORK_E7_MODE"
fi
SPORK_E7_SPLIT="$SPORK_CODEX_ROOT/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json"
SPORK_E7_TURNS=8
SPORK_E7_REPS=1
SPORK_E7_EXTRA=()
case "$SPORK_E7_STAGE" in
    smoke) SPORK_E7_TASKS=retail:51,retail:61; SPORK_E7_TURNS=4 ;;
    main)  SPORK_E7_SPLITS=validation,test ;;
    rest)  SPORK_E7_SPLITS=train ;;
esac
if [[ "$SPORK_E7_SERVING" == standard && "$SPORK_E7_STAGE" == main ]]; then
    SPORK_E7_REPS=2
fi
if [[ "$SPORK_E7_SERVING" == invariant && "$SPORK_E7_STAGE" == smoke ]]; then
    SPORK_E7_EXTRA=(--require-identical-trajectories)
fi
if [[ -z "${SPORK_E7_TASKS:-}" ]]; then
    SPORK_E7_TASKS=$("$SPORK_CODEX_ENV/bin/python" -c '
import json, sys
splits = sys.argv[2].split(",")
tasks = [t for t in json.load(open(sys.argv[1]))["tasks"]
         if t["domain"] in ("airline", "retail") and t["split"] in splits]
print(",".join(t["group_id"] for t in sorted(tasks, key=lambda t: (t["domain"], int(t["task_id"])))))
' "$SPORK_E7_SPLIT" "$SPORK_E7_SPLITS")
fi
SPORK_E7_TAG="${SPORK_E7_TAG:-e7_gemma4_${SPORK_E7_STAGE}_${SPORK_E7_SERVING}_${e7_revision}}"
SPORK_E7_OUTPUT="$SPORK_CODEX_ROOT/results/$SPORK_E7_TAG"
SPORK_E7_RUN_ARGS=()
if [[ "$SPORK_E7_MODE" == --run ]]; then
    "$SPORK_CODEX_ENV/bin/python" scripts/check_codex_runtime.py --server --serving-mode "$SPORK_E7_SERVING"
    SPORK_E7_RUN_ARGS=(--run)
fi
mkdir -p "$SPORK_CODEX_ROOT/results"
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_successor_full_eval.py \
    --output "$SPORK_E7_OUTPUT" --scope all --task-ids "$SPORK_E7_TASKS" --configs nospec,spork,raw \
    --max-turns "$SPORK_E7_TURNS" --repetitions "$SPORK_E7_REPS" --seed 42 --max-tokens 8192 --tool-floor 2 \
    --serving-mode "$SPORK_E7_SERVING" --model-family gemma4 --model "$SPORK_GEMMA_MODEL" \
    --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1" --task-split "$SPORK_E7_SPLIT" \
    "${SPORK_E7_EXTRA[@]}" "${SPORK_E7_RUN_ARGS[@]}" 2>&1 | tee -a "$SPORK_E7_OUTPUT.launch.log"
