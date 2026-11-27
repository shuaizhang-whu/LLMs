#!/usr/bin/env bash
# User launches GPU stages in tmux train. Default prepares without generation.
set -euo pipefail
export SPORK_Q35_REFERENCE_GDN=1
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen35_env.sh"
cd "$SPORK_CODEX_ROOT"
export CUDA_VISIBLE_DEVICES=''
SPORK_EXPANDED_STAGE="${1:-prepare}"
if [[ $# -gt 1 || ! "$SPORK_EXPANDED_STAGE" =~ ^(prepare|usercheck|preflight|expanded|all|check)$ ]]; then
    printf 'Usage: bash scripts/run_qwen35_expanded.sh [prepare|usercheck|preflight|expanded|all|check]\n' >&2
    exit 2
fi
SPORK_EXPANDED_TAG="${SPORK_Q35_EXPANDED_TAG:-qwen35_9b_expanded_20261101_v7}"
SPORK_EXPANDED_ROOT="$SPORK_CODEX_ROOT/results/$SPORK_EXPANDED_TAG"
SPORK_EXPANDED_EXTRA=()
if [[ "$SPORK_EXPANDED_STAGE" =~ ^(usercheck|preflight|expanded|all)$ ]]; then
    SPORK_EXPANDED_EXTRA=(--run)
fi
if [[ -n "${SPORK_Q35_PREFLIGHT_REFERENCE:-}" ]]; then
    SPORK_EXPANDED_EXTRA+=(--preflight-reference "$SPORK_Q35_PREFLIGHT_REFERENCE")
fi
mkdir -p "$SPORK_EXPANDED_ROOT"
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_qwen35_expanded.py \
    --output-root "$SPORK_EXPANDED_ROOT" --stage "$SPORK_EXPANDED_STAGE" \
    --model "$SPORK_CODEX_MODEL" --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1" \
    "${SPORK_EXPANDED_EXTRA[@]}" \
    2>&1 | tee -a "$SPORK_EXPANDED_ROOT/$SPORK_EXPANDED_STAGE.launch.log"
