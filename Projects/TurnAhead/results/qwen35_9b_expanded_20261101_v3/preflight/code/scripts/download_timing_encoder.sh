#!/usr/bin/env bash
# User-launched local download; does not load weights or run a GPU job.
set -euo pipefail
SPORK_ENCODER_HF=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/hf
HF_HUB_OFFLINE=0 "$SPORK_ENCODER_HF" download google/bert_uncased_L-2_H-128_A-2 \
  config.json model.safetensors vocab.txt \
  --revision 30b0a37ccaaa32f332884b96992754e246e48c5f \
  --local-dir /mnt/data/zhangshuai/LLMs/Models/bert-tiny
