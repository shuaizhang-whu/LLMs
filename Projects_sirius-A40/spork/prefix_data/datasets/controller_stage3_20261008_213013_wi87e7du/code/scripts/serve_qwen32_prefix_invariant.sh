#!/usr/bin/env bash
# Run manually in tmux after stopping the previous 8016 server.
# Preserve the original model, TP=4, attention backend, concurrency and D3.
set -euo pipefail
export VLLM_BATCH_INVARIANT=1
printf 'Requesting vLLM batch-invariant computation; validate outputs before using labels.\n'
exec bash /mnt/data/zhangshuai/LLMs/Projects/spork/scripts/serve_qwen32_prefix_collection.sh
