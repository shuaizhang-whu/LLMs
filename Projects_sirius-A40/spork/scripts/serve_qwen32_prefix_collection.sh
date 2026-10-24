#!/usr/bin/env bash
# Start manually in tmux. Dedicated server for the paired-prefix collector.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/spork_env.sh"
spork_configure_cuda_paths
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3}
export VLLM_USE_V2_MODEL_RUNNER=0
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
SPORK_PREFIX_PORT=${SPORK_PREFIX_PORT:-8016}
SPORK_PREFIX_MODEL=${SPORK_PREFIX_MODEL:-$SPORK_QWEN_MODEL}
"$SPORK_PYTHON" - "$SPORK_PREFIX_PORT" <<'PY'
import os, socket, subprocess, sys
from importlib.metadata import version
if not version('vllm').startswith('0.29.'):
    raise SystemExit('This D3 launcher requires vLLM 0.29.x.')
with socket.socket() as s:
    try:
        s.bind(('127.0.0.1', int(sys.argv[1])))
    except OSError as e:
        raise SystemExit(f'Port {sys.argv[1]} is in use: {e}')
rows = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'
], text=True)
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if len(devices) != 4 or len(set(devices)) != 4 or not all(d.isdigit() for d in devices):
    raise SystemExit('This TP=4 launcher requires four distinct numeric CUDA_VISIBLE_DEVICES indices.')
indices = {int(d) for d in devices}
available = {int(row.split(',')[0]) for row in rows.splitlines()}
if not indices <= available:
    raise SystemExit('Selected GPU indices are not available on this host.')
busy = [row.strip() for row in rows.splitlines()
        if int(row.split(',')[0]) in indices and int(row.split(',')[1]) > 2048]
if busy:
    raise SystemExit('Selected GPUs are busy. Current index, MiB: ' + '; '.join(busy))
PY
mkdir -p "$SPORK_REPO_ROOT/results"
SPORK_PREFIX_LOG=$(mktemp "$SPORK_REPO_ROOT/results/server_prefix_collection_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf 'Server log: %s\n' "$SPORK_PREFIX_LOG"
"$SPORK_PYTHON" "$SPORK_REPO_ROOT/launch_vllm_spork_http.py" \
  --model "$SPORK_PREFIX_MODEL" --served-model-name "$SPORK_PREFIX_MODEL" \
  --host 127.0.0.1 --port "$SPORK_PREFIX_PORT" --tensor-parallel-size 4 \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.85 \
  --trust-remote-code --max-logprobs 20 --max-num-seqs 4 \
  --max-num-batched-tokens 2048 --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  2>&1 | tee "$SPORK_PREFIX_LOG"
