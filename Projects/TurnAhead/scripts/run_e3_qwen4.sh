#!/usr/bin/env bash
# E3: Qwen3-4B on one GPU, tau2 airline + retail. Only --run sends evaluation requests.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
export SPORK_CODEX_MODEL="$SPORK_MODEL_ROOT/Qwen3-4B"
export SPORK_CHECK_PORT=8027

stage=${1:-all}
if [[ "$stage" == serve ]]; then
    gpu=${2:-0}
    if [[ $# -gt 2 || ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "Usage: bash scripts/run_e3_qwen4.sh serve [GPU index]" >&2
        exit 2
    fi
    export SPORK_SERVE_DEVICES="$gpu" SPORK_SERVE_TP=1
    export SPORK_EXPERIMENT_PROFILE=qwen3_4b SPORK_TOOL_CALL_OPEN_TOKEN_ID=151657
    exec bash scripts/serve_qwen32_eval.sh invariant
fi

mode=${2:---prepare}
if [[ $# -gt 2 || ! "$stage" =~ ^(all|smoke|main)$ || ! "$mode" =~ ^--(prepare|run)$ ]]; then
    echo "Usage: bash scripts/run_e3_qwen4.sh all|smoke|main [--prepare|--run]" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=''
python="$SPORK_CODEX_ENV/bin/python"
base_url="http://127.0.0.1:$SPORK_CHECK_PORT/v1"
run_args=()
if [[ "$mode" == --run ]]; then
    run_args=(--run)
fi

if [[ "$stage" == all ]]; then
    bash "$SPORK_CODEX_ROOT/scripts/run_e3_qwen4.sh" smoke "$mode"
    "$python" -u scripts/check_concurrency_slowdown.py \
        --model-family qwen3 --model "$SPORK_CODEX_MODEL" --base-url "$base_url" \
        --output "$SPORK_CODEX_ROOT/results/concurrency_qwen3_4b_invariant_v1.json" "${run_args[@]}"
    exec bash "$SPORK_CODEX_ROOT/scripts/run_e3_qwen4.sh" main "$mode"
fi

if [[ "$stage" == smoke ]]; then
    selection=(--scope train --task-ids airline:3 --max-turns 3 --require-identical-trajectories)
else
    selection=(--scope all --max-turns 8)
fi
output="$SPORK_CODEX_ROOT/results/e3_qwen3_4b_${stage}_invariant_v1"
if [[ "$mode" == --run ]]; then
    "$python" scripts/check_codex_runtime.py --server --serving-mode invariant
fi
"$python" -u scripts/run_successor_full_eval.py \
    --workload airline-retail --model-family qwen3 --model "$SPORK_CODEX_MODEL" \
    --base-url "$base_url" --output "$output" --serving-mode invariant \
    --configs nospec,spork,raw --repetitions 1 --max-tokens 8192 --tool-floor 2 --seed 42 \
    "${selection[@]}" "${run_args[@]}" 2>&1 | tee -a "$output.launch.log"

if [[ "$stage" == smoke && "$mode" == --run ]]; then
    "$python" - "$output/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if not summary["complete"] or summary["failures"]:
    sys.exit(f"E3 smoke failed. See {sys.argv[1]}")
print("E3 smoke passed.")
PY
fi
