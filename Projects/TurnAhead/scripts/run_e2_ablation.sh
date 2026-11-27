#!/usr/bin/env bash
# E2: paired ablation on the 46 validation+test tau2 tasks, batch-invariant kernels.
# Default only freezes the plan. Pass --preflight, then --run, yourself in tmux train.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
SPORK_E2_MODE="${1:---prepare}"
if [[ $# -gt 1 || ! "$SPORK_E2_MODE" =~ ^--(prepare|preflight|run)$ ]]; then
    printf 'Usage: bash scripts/run_e2_ablation.sh [--prepare|--preflight|--run]\n' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=''
SPORK_E2_TAG="${SPORK_E2_TAG:-e2_ablation_qwen32_invariant_v1}"
SPORK_E2_SPLIT="$SPORK_CODEX_ROOT/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json"
if [[ "$SPORK_E2_MODE" == --preflight ]]; then
    # Short train tasks whose windows commit, miss on the call, and miss on the next input.
    SPORK_E2_OUTPUT="$SPORK_CODEX_ROOT/results/${SPORK_E2_TAG}_preflight"
    SPORK_E2_ARGS=(--scope train --task-ids retail:51,retail:61 --max-turns 4 --run)
else
    SPORK_E2_OUTPUT="$SPORK_CODEX_ROOT/results/$SPORK_E2_TAG"
    SPORK_E2_TASKS=$("$SPORK_CODEX_ENV/bin/python" -c '
import json, sys
tasks = [t for t in json.load(open(sys.argv[1]))["tasks"]
         if t["domain"] in ("airline", "retail") and t["split"] in ("validation", "test")]
print(",".join(t["group_id"] for t in sorted(tasks, key=lambda t: (t["domain"], int(t["task_id"])))))
' "$SPORK_E2_SPLIT")
    SPORK_E2_ARGS=(--scope all --task-ids "$SPORK_E2_TASKS" --max-turns 8)
    if [[ "$SPORK_E2_MODE" == --run ]]; then
        SPORK_E2_ARGS+=(--run)
    fi
fi
if [[ "$SPORK_E2_MODE" != --prepare ]]; then
    "$SPORK_CODEX_ENV/bin/python" scripts/check_codex_runtime.py --server --serving-mode invariant
fi
mkdir -p "$SPORK_CODEX_ROOT/results"
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_successor_full_eval.py \
    --output "$SPORK_E2_OUTPUT" "${SPORK_E2_ARGS[@]}" \
    --configs "${SPORK_E2_CONFIGS:-nospec,spork,raw,actor_only,raw_no_probe,serial}" \
    --repetitions 1 --seed 42 --max-tokens 8192 --tool-floor 2 --serving-mode invariant \
    --model "$SPORK_CODEX_MODEL" --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1" \
    --task-split "$SPORK_E2_SPLIT" 2>&1 | tee -a "$SPORK_E2_OUTPUT.launch.log"
if [[ "$SPORK_E2_MODE" == --preflight ]]; then
    "$SPORK_CODEX_ENV/bin/python" - "$SPORK_E2_OUTPUT/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
differ = [f'{c["group_id"]}/{c["config"]}' for c in s["comparisons"]
          if not (c["same_actor_tokens"] and c["same_actions_results_state"])]
ok = s["complete"] and not s["failures"] and not differ
print(f'E2 preflight {"passed" if ok else "FAILED"}: {s["recorded"]}/{s["expected"]} runs, '
      f'{len(s["failures"])} with errors, {len(differ)}/{len(s["comparisons"])} differ from the spork run {differ}')
sys.exit(not ok)
PY
fi
