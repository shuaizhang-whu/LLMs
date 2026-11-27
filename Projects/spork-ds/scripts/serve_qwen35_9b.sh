#!/usr/bin/env bash
# Qwen3.5-9B plain vLLM OpenAI server for the stable-refresh smoke/full run.
# No D3 sidecar: the refresh runner recomputes full probes and needs no
# draft-token injection. GPUs 1,2 by default; GPU 0 is left to other users.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SPORK_ENV="${SPORK_ENV:-/mnt/data/zhangshuai/anaconda3/envs/spork}"
MODEL="${SPORK_Q35_MODEL:-/mnt/data/zhangshuai/LLMs/Models/Qwen3.5-9B}"
DEVICES="${SPORK_Q35_DEVICES:-1,2}"
TP="${SPORK_Q35_TP:-2}"
PORT="${SPORK_Q35_PORT:-8027}"

export CUDA_VISIBLE_DEVICES="$DEVICES"
export CUDA_HOME="$SPORK_ENV"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export NVCC_PREPEND_FLAGS="-I$SPORK_ENV/lib/python3.12/site-packages/nvidia/curand/include"
export LD_LIBRARY_PATH="$SPORK_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$SPORK_ENV/bin:$PATH"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_BATCH_INVARIANT=0
export VLLM_GDN_DECODE_KERNEL=triton
export VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

"$SPORK_ENV/bin/python" - "$PORT" "$DEVICES" "$TP" <<'PY'
import socket, subprocess, sys
port, devices, tp = int(sys.argv[1]), sys.argv[2].split(","), int(sys.argv[3])
if len(set(devices)) != len(devices) or len(devices) != tp:
    raise SystemExit("Visible GPU count must equal TP size, without duplicate devices")
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"Port {port} is already in use: {exc}")
result = subprocess.check_output(
    ["nvidia-smi", "--id=" + sys.argv[2], "--query-gpu=index,memory.used",
     "--format=csv,noheader,nounits"], text=True)
if any(int(row.split(",")[1]) > 2048 for row in result.splitlines()):
    raise SystemExit("Selected GPUs are occupied; choose free SPORK_Q35_DEVICES")
print("Selected GPUs:", result.strip())
PY

mkdir -p "$ROOT/results"
LOG="$(mktemp "$ROOT/results/server_qwen35_9b_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
printf 'Server log: %s\n' "$LOG"

"$SPORK_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP" \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.80 \
  --language-model-only --generation-config vllm \
  --hf-overrides '{"head_dtype":"float32"}' \
  --enable-prefix-caching --mamba-cache-mode align --mamba-ssm-cache-dtype float32 \
  --trust-remote-code --max-logprobs 20 --max-num-seqs 4 \
  --max-num-batched-tokens 4096 --block-size 640 --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  2>&1 | tee "$LOG"
