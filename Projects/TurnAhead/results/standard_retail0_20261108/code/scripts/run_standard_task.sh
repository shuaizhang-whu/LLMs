#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
export CUDA_VISIBLE_DEVICES=''
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --run ) ]]; then
    printf 'Usage: bash scripts/run_standard_task.sh [--run]\n' >&2
    exit 2
fi
SPORK_STANDARD_OUTPUT="$SPORK_CODEX_ROOT/results/standard_retail0_20261108"
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_standard_task.py --output "$SPORK_STANDARD_OUTPUT" "$@" \
  2>&1 | tee -a "$SPORK_STANDARD_OUTPUT.launch.log"
