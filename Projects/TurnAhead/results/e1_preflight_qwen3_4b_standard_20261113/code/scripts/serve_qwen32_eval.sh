#!/usr/bin/env bash
# Evaluation server; start it yourself in tmux vllm.
# standard: default vLLM kernels. invariant: batch-invariant kernels (slower, reproducible).
set -euo pipefail
SPORK_SERVING_MODE="${1:-}"
if [[ $# -ne 1 || ! "$SPORK_SERVING_MODE" =~ ^(standard|invariant)$ ]]; then
    printf 'Usage: bash scripts/serve_qwen32_eval.sh standard|invariant\n' >&2
    exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
SPORK_SERVE_DEVICES="${SPORK_SERVE_DEVICES:-0,1,2,3}"
SPORK_SERVE_TP="${SPORK_SERVE_TP:-4}"
export PATH="$SPORK_CODEX_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$SPORK_SERVE_DEVICES"
export CUDA_HOME="$SPORK_CODEX_ENV"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export NVCC_PREPEND_FLAGS="-I$SPORK_CODEX_ENV/lib/python3.12/site-packages/nvidia/curand/include"
export LD_LIBRARY_PATH="$SPORK_CODEX_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$SPORK_SERVING_MODE" == invariant ]]; then
    export VLLM_BATCH_INVARIANT=1
else
    unset VLLM_BATCH_INVARIANT
fi
"$SPORK_CODEX_ENV/bin/python" - "$SPORK_CHECK_PORT" "$SPORK_SERVE_DEVICES" <<'PY'
import socket, subprocess, sys
with socket.socket() as s:
    s.bind(('127.0.0.1', int(sys.argv[1])))
rows = subprocess.check_output(['nvidia-smi', '--id=' + sys.argv[2], '--query-gpu=index,memory.used',
                                '--format=csv,noheader,nounits'], text=True)
if any(int(r.split(',')[1]) > 2048 for r in rows.splitlines()):
    raise SystemExit('Selected GPUs are occupied; not starting another server')
PY
mkdir -p results
SPORK_SERVE_LOG=$(mktemp "$PWD/results/server_eval_${SPORK_SERVING_MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf 'Server log: %s\n' "$SPORK_SERVE_LOG"
# Same options as the server of the earlier successor runs; only the kernel mode differs.
"$SPORK_CODEX_ENV/bin/python" launch_vllm_spork_http.py \
  --model "$SPORK_CODEX_MODEL" \
  --served-model-name "$SPORK_CODEX_MODEL" \
  --host 127.0.0.1 --port "$SPORK_CHECK_PORT" --tensor-parallel-size "$SPORK_SERVE_TP" \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.80 \
  --trust-remote-code --max-logprobs 20 --max-num-seqs 4 \
  --max-num-batched-tokens 512 --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  2>&1 | tee "$SPORK_SERVE_LOG"
