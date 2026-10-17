#!/bin/bash
set -euo pipefail

# SAES-SVD v3 multi-model launcher.
# Examples:
#   bash run_saes_svd_v3.sh --model_tag qwen2_5_7b --gpu 0
#   bash run_saes_svd_v3.sh --model_tag qwen3_moe --gpu 0 --skip_moe_experts
#   bash run_saes_svd_v3.sh --model /path/to/model --gpu 0

GPU="${GPU:-0}"
MODEL_TAG="${MODEL_TAG:-qwen2_5_7b}"
MODEL=""
RATIO="${RATIO:-0.2}"
NSAMPLES="${NSAMPLES:-256}"
DATASET="${DATASET:-wikitext2}"
SEQ_LEN="${SEQ_LEN:-2048}"
RHO="${RHO:-0.95}"
ALPHA_MIN="${ALPHA_MIN:-0.4}"
ALPHA_MAX="${ALPHA_MAX:-0.6}"
EPS="${EPS:-1e-4}"
MIN_DIM="${MIN_DIM:-64}"
SKIP_MOE_EXPERTS="${SKIP_MOE_EXPERTS:-0}"
OUT_BASE="${OUT_BASE:-/data/zhangs/LLMs/SAES-SVD/result/SAES_SVD_v3}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)              GPU="$2"; shift 2;;
        --model_tag)        MODEL_TAG="$2"; MODEL=""; shift 2;;
        --model)            MODEL="$2"; MODEL_TAG=""; shift 2;;
        --ratio)            RATIO="$2"; shift 2;;
        --nsamples)         NSAMPLES="$2"; shift 2;;
        --dataset)          DATASET="$2"; shift 2;;
        --seq_len)          SEQ_LEN="$2"; shift 2;;
        --rho)              RHO="$2"; shift 2;;
        --alpha_min)        ALPHA_MIN="$2"; shift 2;;
        --alpha_max)        ALPHA_MAX="$2"; shift 2;;
        --eps)              EPS="$2"; shift 2;;
        --min_dim)          MIN_DIM="$2"; shift 2;;
        --skip_moe_experts) SKIP_MOE_EXPERTS=1; shift 1;;
        --out_base)         OUT_BASE="$2"; shift 2;;
        --out_dir)          OUT_DIR_OVERRIDE="$2"; shift 2;;
        *)                  EXTRA_ARGS+=("$1"); shift 1;;
    esac
done

PYTHON="/data/zhangs/anaconda3/envs/compress_Qwen3.5/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAES_PY="${SCRIPT_DIR}/SAES-SVD_v3.py"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi

if [[ -n "${OUT_DIR_OVERRIDE:-}" ]]; then
    OUT_DIR="$OUT_DIR_OVERRIDE"
else
    MODEL_NAME="${MODEL_TAG:-custom_model}"
    TS="$(date +%Y%m%d_%H%M%S)"
    OUT_DIR="${OUT_BASE}/${MODEL_NAME}/ns${NSAMPLES}_rho${RHO}_${TS}"
fi
mkdir -p "$OUT_DIR"

CMD=(
    "$PYTHON" "$SAES_PY"
    --ratio "$RATIO"
    --nsamples "$NSAMPLES"
    --dataset "$DATASET"
    --model_seq_len "$SEQ_LEN"
    --rho "$RHO"
    --alpha_min "$ALPHA_MIN"
    --alpha_max "$ALPHA_MAX"
    --eps "$EPS"
    --min_dim "$MIN_DIM"
    --save_path "$OUT_DIR"
    --DEV cuda:0
)

if [[ -n "$MODEL" ]]; then
    CMD+=(--model "$MODEL")
else
    CMD+=(--model_tag "$MODEL_TAG")
fi

if [[ "$SKIP_MOE_EXPERTS" == "1" ]]; then
    CMD+=(--skip_moe_experts)
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "=========================================="
echo "SAES-SVD v3 Compression"
echo "GPU:        $GPU"
echo "MODEL_TAG:  ${MODEL_TAG:-<manual>}"
echo "MODEL:      ${MODEL:-<preset>}"
echo "Output:     $OUT_DIR"
echo "=========================================="
echo "Running: ${CMD[*]}"

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" 2>&1 | tee "$OUT_DIR/run.log"
