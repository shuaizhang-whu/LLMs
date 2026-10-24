#!/usr/bin/env bash
# User-launched local download; does not load weights or run a GPU job.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/spork_env.sh"
SPORK_ENCODER_HF=${SPORK_ENCODER_HF:-$(dirname -- "$SPORK_PYTHON")/hf}
HF_HUB_OFFLINE=0 "$SPORK_ENCODER_HF" download google/bert_uncased_L-2_H-128_A-2 \
  config.json model.safetensors vocab.txt \
  --revision 30b0a37ccaaa32f332884b96992754e246e48c5f \
  --local-dir "$SPORK_ENCODER"
