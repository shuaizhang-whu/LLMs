#!/usr/bin/env bash
# User launches this script in tmux vllm; no model is loaded by prepare-only.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen35_env.sh"
cd "$SPORK_CODEX_ROOT"
export CUDA_VISIBLE_DEVICES="$SPORK_Q35_DEVICES"
export CUDA_HOME="$SPORK_CODEX_ENV"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export NVCC_PREPEND_FLAGS="-I$SPORK_CODEX_ENV/lib/python3.12/site-packages/nvidia/curand/include"
export LD_LIBRARY_PATH="$SPORK_CODEX_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$SPORK_CODEX_ENV/bin:$PATH"
export VLLM_USE_V2_MODEL_RUNNER=0
# vLLM 0.29 rejects batch-invariant mode for Qwen3.5 GDN.
# The experiment retains empirical exact token/action/state checks.
export VLLM_BATCH_INVARIANT="$SPORK_Q35_BATCH_INVARIANT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$SPORK_CODEX_ENV/bin/python" - "$SPORK_CHECK_PORT" "$SPORK_Q35_DEVICES" "$SPORK_Q35_TP" <<'PY'
import socket, subprocess, sys
devices = sys.argv[2].split(",")
if len(set(devices)) != len(devices) or len(devices) != int(sys.argv[3]):
    raise SystemExit("Visible GPU count must equal TP size, without duplicate devices")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
result = subprocess.check_output(["nvidia-smi", "--id=" + sys.argv[2],
    "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
if any(int(row.split(",")[1]) > 2048 for row in result.splitlines()):
    raise SystemExit("Selected GPUs are occupied; choose free SPORK_Q35_DEVICES and matching SPORK_Q35_TP")
print("Selected GPUs:", result.strip())
PY
CUDA_VISIBLE_DEVICES='' "$SPORK_CODEX_ENV/bin/python" scripts/check_qwen35_experiment.py --local
mkdir -p results
SPORK_Q35_LOG="$(mktemp "$PWD/results/server_qwen35_9b_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
printf 'Server log: %s\n' "$SPORK_Q35_LOG"
"$SPORK_CODEX_ENV/bin/python" launch_vllm_spork_http.py \
  --model "$SPORK_CODEX_MODEL" --served-model-name "$SPORK_CODEX_MODEL" \
  --host 127.0.0.1 --port "$SPORK_CHECK_PORT" --tensor-parallel-size "$SPORK_Q35_TP" \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.80 \
  --language-model-only --generation-config vllm \
  --enable-prefix-caching --mamba-cache-mode align --mamba-ssm-cache-dtype float32 \
  --trust-remote-code --max-logprobs 20 --max-num-seqs 4 \
  --max-num-batched-tokens 512 --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  2>&1 | tee "$SPORK_Q35_LOG"
