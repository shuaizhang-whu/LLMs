#!/usr/bin/env bash
# Gemma-4-31B-it evaluation server; start it yourself in tmux vllm (needs all four GPUs).
# standard: default vLLM kernels. invariant: batch-invariant kernels (slower, reproducible).
# Same request-side settings as scripts/serve_qwen32_eval.sh. Gemma-specific: text-only model
# (vision/audio towers not loaded) and the D3 trigger token <|tool_call> (id 48).
set -euo pipefail
SPORK_SERVING_MODE="${1:-}"
if [[ $# -ne 1 || ! "$SPORK_SERVING_MODE" =~ ^(standard|invariant)$ ]]; then
    printf 'Usage: bash scripts/serve_gemma4_eval.sh standard|invariant\n' >&2
    exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
SPORK_GEMMA_MODEL="${SPORK_GEMMA_MODEL:-$SPORK_MODEL_ROOT/gemma-4-31B-it}"
SPORK_SERVE_DEVICES="${SPORK_SERVE_DEVICES:-0,1,2,3}"
SPORK_SERVE_TP="${SPORK_SERVE_TP:-4}"
SPORK_GEMMA_GPU_UTIL="${SPORK_GEMMA_GPU_UTIL:-0.80}"
export PATH="$SPORK_CODEX_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$SPORK_SERVE_DEVICES"
export CUDA_HOME="$SPORK_CODEX_ENV"
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export NVCC_PREPEND_FLAGS="-I$SPORK_CODEX_ENV/lib/python3.12/site-packages/nvidia/curand/include"
export LD_LIBRARY_PATH="$SPORK_CODEX_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPORK_TOOL_CALL_OPEN_TOKEN_ID=48
export SPORK_EXPERIMENT_PROFILE=gemma4_31b_it
if [[ "$SPORK_SERVING_MODE" == invariant ]]; then
    export VLLM_BATCH_INVARIANT=1
else
    unset VLLM_BATCH_INVARIANT
fi
mkdir -p results
SPORK_SERVE_LOG=$(mktemp "$PWD/results/server_gemma4_${SPORK_SERVING_MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf 'Server log: %s\n' "$SPORK_SERVE_LOG"
"$SPORK_CODEX_ENV/bin/python" launch_vllm_spork_http.py \
  --model "$SPORK_GEMMA_MODEL" \
  --served-model-name "$SPORK_GEMMA_MODEL" \
  --host 127.0.0.1 --port "$SPORK_CHECK_PORT" --tensor-parallel-size "$SPORK_SERVE_TP" \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization "$SPORK_GEMMA_GPU_UTIL" \
  --language-model-only --max-logprobs 20 --max-num-seqs 4 \
  --max-num-batched-tokens 512 --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  2>&1 | tee "$SPORK_SERVE_LOG"
