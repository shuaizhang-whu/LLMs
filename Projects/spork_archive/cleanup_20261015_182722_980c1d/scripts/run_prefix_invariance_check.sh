#!/usr/bin/env bash
# User-launched bounded verification: the same two prefixes x 3 modes x 3 repeats.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
SPORK_CHECK_PYTHON=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if (( $# > 1 )); then
  printf 'Usage: bash %s [prepared-run-directory]\n' "$0" >&2
  exit 1
fi
SPORK_CHECK_ARGS=(--require-invariance --repetitions 3)
if (( $# == 1 )); then
  SPORK_CHECK_RUN=$1
  if [[ ! -f "$SPORK_CHECK_RUN/manifest.json" ]]; then
    printf 'Prepared manifest not found: %s\n' "$SPORK_CHECK_RUN" >&2
    exit 1
  fi
  SPORK_CHECK_ARGS+=(--resume)
else
  mkdir -p "$PWD/prefix_data/diagnostic_runs"
  SPORK_CHECK_RUN=$(mktemp -d "$PWD/prefix_data/diagnostic_runs/batch_invariance_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi
if [[ -n "${SPORK_PREFIX_URL:-}" ]]; then
  SPORK_CHECK_ARGS+=(--model-url "$SPORK_PREFIX_URL")
fi
# Prepare before tee creates run.log; never reuse the previous experiment.
"$SPORK_CHECK_PYTHON" scripts/diagnose_prefix_forks.py \
  --output "$SPORK_CHECK_RUN" --prepare-only "${SPORK_CHECK_ARGS[@]}"
exec > >(tee -a "$SPORK_CHECK_RUN/run.log") 2>&1
printf 'Bounded invariance check: %s\n' "$SPORK_CHECK_RUN"
"$SPORK_CHECK_PYTHON" scripts/diagnose_prefix_forks.py \
  --output "$SPORK_CHECK_RUN" --require-invariance --repetitions 3 --resume
printf 'Passed on these two prefixes. Proceed to staged collection with ongoing quality checks.\n'
printf 'Report: %s/fork_diagnostic_report.json\n' "$SPORK_CHECK_RUN"
