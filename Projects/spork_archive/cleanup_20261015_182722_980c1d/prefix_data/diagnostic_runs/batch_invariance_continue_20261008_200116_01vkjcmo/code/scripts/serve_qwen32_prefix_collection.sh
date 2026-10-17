#!/usr/bin/env bash
# Start manually in tmux. Dedicated server for the paired-prefix collector.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
SPORK_ENV=/mnt/data/zhangshuai/anaconda3/envs/spork
export PATH="$SPORK_ENV/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME="$SPORK_ENV"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export NVCC_PREPEND_FLAGS="-I$SPORK_ENV/lib/python3.12/site-packages/nvidia/curand/include"
export LD_LIBRARY_PATH="$SPORK_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_V2_MODEL_RUNNER=0
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
SPORK_PREFIX_PORT=${SPORK_PREFIX_PORT:-8016}
SPORK_PREFIX_MODEL=/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B
"$SPORK_ENV/bin/python" - "$SPORK_PREFIX_PORT" <<'PY'
import socket, subprocess, sys
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
busy = [row.strip() for row in rows.splitlines()
        if int(row.split(',')[0]) in (0, 1, 2, 3) and int(row.split(',')[1]) > 2048]
if busy:
    raise SystemExit('Free GPUs 0,1,2,3 before starting this server. Current index, MiB: ' + '; '.join(busy))
PY
mkdir -p results
SPORK_PREFIX_LOG=$(mktemp "$PWD/results/server_prefix_collection_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf 'Server log: %s\n' "$SPORK_PREFIX_LOG"
"$SPORK_ENV/bin/python" launch_vllm_spork_http.py \
  --model "$SPORK_PREFIX_MODEL" --served-model-name "$SPORK_PREFIX_MODEL" \
  --host 127.0.0.1 --port "$SPORK_PREFIX_PORT" --tensor-parallel-size 4 \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.85 \
  --trust-remote-code --max-logprobs 20 --max-num-seqs 4 \
  --max-num-batched-tokens 2048 --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  2>&1 | tee "$SPORK_PREFIX_LOG"
