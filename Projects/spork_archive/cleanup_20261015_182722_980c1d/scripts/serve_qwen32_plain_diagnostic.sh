#!/usr/bin/env bash
# User launches this in tmux only if no matching plain vLLM server is running.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
SPORK_ENV=/mnt/data/zhangshuai/anaconda3/envs/spork
export PATH="$SPORK_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME="$SPORK_ENV"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export NVCC_PREPEND_FLAGS="-I$SPORK_ENV/lib/python3.12/site-packages/nvidia/curand/include"
export LD_LIBRARY_PATH="$SPORK_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1
LOG_FILE=$(mktemp "$PWD/results/server_first_turn_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf 'Server log: %s\n' "$LOG_FILE"
"$SPORK_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model /mnt/data/zhangshuai/LLMs/Models/Qwen3-32B \
  --served-model-name /mnt/data/zhangshuai/LLMs/Models/Qwen3-32B \
  --host 127.0.0.1 --port 8013 --tensor-parallel-size 4 \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.85 \
  --trust-remote-code --max-logprobs 20 2>&1 | tee "$LOG_FILE"
