#!/usr/bin/env bash
# Start manually in tmux after stopping the old GPU service.
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
LOG_FILE=$(mktemp "$PWD/results/server_d3_formal_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf 'D3 formal server log: %s\n' "$LOG_FILE"
"$SPORK_ENV/bin/python" launch_vllm_spork_http.py \
  --model /mnt/data/zhangshuai/LLMs/Models/Qwen3-32B \
  --served-model-name /mnt/data/zhangshuai/LLMs/Models/Qwen3-32B \
  --host 127.0.0.1 --port 8015 --tensor-parallel-size 4 \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.85 \
  --trust-remote-code --max-logprobs 20 --max-num-seqs 4 \
  --max-num-batched-tokens 2048 --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  2>&1 | tee "$LOG_FILE"
