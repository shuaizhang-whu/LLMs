#!/usr/bin/env bash
SPORK_Q35_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export SPORK_CODEX_MODEL="${SPORK_Q35_MODEL:-$(dirname -- "$(dirname -- "$SPORK_Q35_ROOT")")/Models/Qwen3.5-9B}"
export SPORK_CHECK_PORT="${SPORK_Q35_PORT:-8027}"
source "$SPORK_Q35_ROOT/scripts/codex_env.sh"
export SPORK_Q35_DEVICES="${SPORK_Q35_DEVICES:-2,3}"
export SPORK_Q35_TP="${SPORK_Q35_TP:-2}"
export SPORK_TOOL_CALL_OPEN_TOKEN_ID=248058
export SPORK_EXPERIMENT_PROFILE=qwen35_9b_successor_v1
export SPORK_Q35_BATCH_INVARIANT=0
export SPORK_Q35_RUN_TAG="${SPORK_Q35_RUN_TAG:-qwen35_9b_20261101}"
