#!/usr/bin/env bash
# Start manually in tmux after stopping the old GPU service.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/spork_env.sh"
spork_configure_cuda_paths
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3}
export VLLM_USE_V2_MODEL_RUNNER=0
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
mkdir -p "$SPORK_REPO_ROOT/results"
LOG_FILE=$(mktemp "$SPORK_REPO_ROOT/results/server_d3_formal_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf 'D3 formal server log: %s\n' "$LOG_FILE"
"$SPORK_PYTHON" "$SPORK_REPO_ROOT/launch_vllm_spork_http.py" \
  --model "$SPORK_QWEN_MODEL" \
  --served-model-name "$SPORK_QWEN_MODEL" \
  --host 127.0.0.1 --port 8015 --tensor-parallel-size 4 \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.85 \
  --trust-remote-code --max-logprobs 20 --max-num-seqs 4 \
  --max-num-batched-tokens 2048 --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  2>&1 | tee "$LOG_FILE"
