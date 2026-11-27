#!/usr/bin/env bash
# Dedicated small-experiment server. Prompt logprobs need extra logits memory.
# All compared methods must use this same serving configuration.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
SPORK_CHECK_ENV=/mnt/data/zhangshuai/anaconda3/envs/spork
export PATH="$SPORK_CHECK_ENV/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME="$SPORK_CHECK_ENV"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export NVCC_PREPEND_FLAGS="-I$SPORK_CHECK_ENV/lib/python3.12/site-packages/nvidia/curand/include"
export LD_LIBRARY_PATH="$SPORK_CHECK_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_BATCH_INVARIANT=1
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPORK_CHECK_PORT=${SPORK_CHECK_PORT:-8017}
"$SPORK_CHECK_ENV/bin/python" - "$SPORK_CHECK_PORT" <<'PY'
import socket, subprocess, sys
with socket.socket() as s:
    s.bind(('127.0.0.1', int(sys.argv[1])))
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                '--format=csv,noheader,nounits'], text=True)
if any(int(r.split(',')[1]) > 2048 for r in rows.splitlines()):
    raise SystemExit('GPUs are occupied; not starting another server')
PY
mkdir -p results
SPORK_CHECK_LOG=$(mktemp "$PWD/results/server_candidate_checks_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf 'Server log: %s\n' "$SPORK_CHECK_LOG"
"$SPORK_CHECK_ENV/bin/python" launch_vllm_spork_http.py \
  --model /mnt/data/zhangshuai/LLMs/Models/Qwen3-32B \
  --served-model-name /mnt/data/zhangshuai/LLMs/Models/Qwen3-32B \
  --host 127.0.0.1 --port "$SPORK_CHECK_PORT" --tensor-parallel-size 4 \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.80 \
  --trust-remote-code --max-logprobs 20 --max-num-seqs 4 \
  --max-num-batched-tokens 512 --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  2>&1 | tee "$SPORK_CHECK_LOG"
