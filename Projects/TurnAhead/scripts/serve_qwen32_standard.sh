#!/usr/bin/env bash
# User starts this ordinary non-speculative server in tmux vllm.
set -euo pipefail
export SPORK_CODEX_MODEL="/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B"
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME="$SPORK_CODEX_ENV"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export NVCC_PREPEND_FLAGS="-I$SPORK_CODEX_ENV/lib/python3.12/site-packages/nvidia/curand/include"
export LD_LIBRARY_PATH="$SPORK_CODEX_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_BATCH_INVARIANT=1
export VLLM_SERVER_DEV_MODE=1
export PATH="$SPORK_CODEX_ENV/bin:$PATH"
"$SPORK_CODEX_ENV/bin/python" - <<'PY'
import socket,subprocess
with socket.socket() as sock:sock.bind(('127.0.0.1',8029))
rows=subprocess.check_output(['nvidia-smi','--id=0,1,2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
if any(int(r.split(',')[1])>2048 for r in rows.splitlines()):
    raise SystemExit('GPUs 0–3 are occupied. Stop the existing model service yourself before starting the standard baseline.')
PY
# No --speculative-config: token drafting is disabled as well as Agent speculation.
"$SPORK_CODEX_ENV/bin/vllm" serve "$SPORK_CODEX_MODEL" \
 --served-model-name "$SPORK_CODEX_MODEL" --host 127.0.0.1 --port 8029 \
 --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 40960 \
 --gpu-memory-utilization .80 --trust-remote-code --max-num-seqs 4 \
 --max-num-batched-tokens 512 --no-async-scheduling --enable-prefix-caching
