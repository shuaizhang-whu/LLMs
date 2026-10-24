#!/usr/bin/env bash
# User starts manually in tmux after the invariant D3 server is ready.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/spork_env.sh"
SPORK_ONLINE_PYTHON=${SPORK_ONLINE_PYTHON:-$SPORK_PYTHON}
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$SPORK_REPO_ROOT/results"
exec 9>"$SPORK_REPO_ROOT/results/controller_online_8016.lock"
flock -n 9 || { printf '8016 上已有控制器检查在运行。\n' >&2; exit 1; }
SPORK_ONLINE_LOG=$(mktemp "$SPORK_REPO_ROOT/results/controller_online_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
exec > >(tee -a "$SPORK_ONLINE_LOG") 2>&1
printf '在线检查日志：%s\n' "$SPORK_ONLINE_LOG"
"$SPORK_ONLINE_PYTHON" "$SPORK_REPO_ROOT/scripts/check_controller_online.py" "$@"
