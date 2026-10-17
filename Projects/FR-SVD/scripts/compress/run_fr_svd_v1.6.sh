#!/bin/bash
set -euo pipefail

# ================================
# FR-SVD v1.6: LLaMA-7B compression + timed evaluation
# ================================
# 默认任务：
#   /data/zhangs/LLMs/Models/llama-7b
#   20% compression, wikitext2, nsamples=256
#
# 默认模式：
#   fisher_mode=none 且不启用 innovation2，即 v1.6 的 SAES baseline 路径。
#   如需完整 FR-SVD，可加：
#     --fisher_mode guide --enable_innovation2
# ================================

GPU="${GPU:-0}"
MODEL="${MODEL:-/data/zhangs/LLMs/Models/llama-7b}"
RATIO="${RATIO:-0.2}"
NSAMPLES="${NSAMPLES:-256}"
DATASET="${DATASET:-wikitext2}"
SEQ_LEN="${SEQ_LEN:-2048}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
R1_RATIO="${R1_RATIO:-0.92}"
ALPHA_MIN="${ALPHA_MIN:-0.4}"
ALPHA_MAX="${ALPHA_MAX:-0.6}"
RHO="${RHO:-0.95}"
EPS="${EPS:-1e-4}"
FISHER_MODE="${FISHER_MODE:-none}"
ENABLE_INNOVATION2="${ENABLE_INNOVATION2:-0}"
MAX_FISHER_SAMPLES="${MAX_FISHER_SAMPLES:-32}"
LAMBDA_PHI="${LAMBDA_PHI:-1e-4}"
LAMBDA_PSI="${LAMBDA_PSI:-1e-4}"
FISHER_WEIGHT_MIN="${FISHER_WEIGHT_MIN:-0.05}"
FISHER_WEIGHT_MAX="${FISHER_WEIGHT_MAX:-20.0}"
MAX_FISHER_DIM="${MAX_FISHER_DIM:-4096}"
EVAL_LARGE="${EVAL_LARGE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
PYTHON="${PYTHON:-/data/zhangs/anaconda3/envs/compress/bin/python}"
OUT_BASE="${OUT_BASE:-/data/zhangs/LLMs/Projects/FR-SVD/result/FR_SVD_v1.6}"
OUT_DIR="${OUT_DIR:-}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)                  GPU="$2"; shift 2;;
        --model)                MODEL="$2"; shift 2;;
        --ratio)                RATIO="$2"; shift 2;;
        --nsamples)             NSAMPLES="$2"; shift 2;;
        --dataset)              DATASET="$2"; shift 2;;
        --seq_len)              SEQ_LEN="$2"; shift 2;;
        --eval_batch_size)      EVAL_BATCH_SIZE="$2"; shift 2;;
        --r1_ratio)             R1_RATIO="$2"; shift 2;;
        --alpha_min)            ALPHA_MIN="$2"; shift 2;;
        --alpha_max)            ALPHA_MAX="$2"; shift 2;;
        --rho)                  RHO="$2"; shift 2;;
        --eps)                  EPS="$2"; shift 2;;
        --fisher_mode)          FISHER_MODE="$2"; shift 2;;
        --enable_innovation2)   ENABLE_INNOVATION2=1; shift 1;;
        --max_fisher_samples)   MAX_FISHER_SAMPLES="$2"; shift 2;;
        --lambda_phi)           LAMBDA_PHI="$2"; shift 2;;
        --lambda_psi)           LAMBDA_PSI="$2"; shift 2;;
        --fisher_weight_min)    FISHER_WEIGHT_MIN="$2"; shift 2;;
        --fisher_weight_max)    FISHER_WEIGHT_MAX="$2"; shift 2;;
        --max_fisher_dim)       MAX_FISHER_DIM="$2"; shift 2;;
        --eval_large)           EVAL_LARGE=1; shift 1;;
        --no_eval)              RUN_EVAL=0; shift 1;;
        --python)               PYTHON="$2"; shift 2;;
        --out_base)             OUT_BASE="$2"; shift 2;;
        --out_dir)              OUT_DIR="$2"; shift 2;;
        *)                      EXTRA_ARGS+=("$1"); shift 1;;
    esac
done

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi

if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: model directory not found: $MODEL" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FR_SVD_PY="${PROJECT_ROOT}/FR_SVD_v1.6.py"
if [[ ! -f "$FR_SVD_PY" ]]; then
    echo "ERROR: FR-SVD v1.6 script not found: $FR_SVD_PY" >&2
    exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    MODEL_TAG="$(basename "$MODEL" | tr '-' '_')"
    I2_TAG="i2off"
    if [[ "$ENABLE_INNOVATION2" == "1" ]]; then
        I2_TAG="i2on"
    fi
    OUT_DIR="${OUT_BASE}/${MODEL_TAG}/ns${NSAMPLES}_ratio${RATIO}_fisher${FISHER_MODE}_${I2_TAG}_${TS}"
fi

mkdir -p "$OUT_DIR"

RUN_LOG="${OUT_DIR}/run.log"
COMPRESS_LOG="${OUT_DIR}/compress.log"
EVAL_LOG="${OUT_DIR}/eval.log"
SUMMARY_TXT="${OUT_DIR}/time_summary.txt"
SUMMARY_CSV="${OUT_DIR}/time_summary.csv"

hms() {
    local seconds="$1"
    printf "%02d:%02d:%02d" "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

log_main() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUN_LOG"
}

find_latest_model() {
    find "$OUT_DIR" -maxdepth 1 -type f -name '*.pt' -printf '%T@ %p\n' \
        | sort -nr | head -n 1 | cut -d' ' -f2-
}

INNOVATION2_ARGS=()
if [[ "$ENABLE_INNOVATION2" == "1" ]]; then
    INNOVATION2_ARGS+=(--enable_innovation2)
fi

EVAL_LARGE_ARGS=()
if [[ "$EVAL_LARGE" == "1" ]]; then
    EVAL_LARGE_ARGS+=(--eval_large)
fi

COMPRESS_CMD=(
    "$PYTHON" "$FR_SVD_PY"
    --model "$MODEL"
    --ratio "$RATIO"
    --nsamples "$NSAMPLES"
    --dataset "$DATASET"
    --model_seq_len "$SEQ_LEN"
    --r1_ratio "$R1_RATIO"
    --alpha_min "$ALPHA_MIN"
    --alpha_max "$ALPHA_MAX"
    --rho "$RHO"
    --eps "$EPS"
    --fisher_mode "$FISHER_MODE"
    --max_fisher_samples "$MAX_FISHER_SAMPLES"
    --lambda_phi "$LAMBDA_PHI"
    --lambda_psi "$LAMBDA_PSI"
    --fisher_weight_min "$FISHER_WEIGHT_MIN"
    --fisher_weight_max "$FISHER_WEIGHT_MAX"
    --max_fisher_dim "$MAX_FISHER_DIM"
    --save_path "$OUT_DIR"
    --DEV cuda:0
    --no_eval_after_compress
    "${INNOVATION2_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
)

{
    echo "=========================================="
    echo "FR-SVD v1.6: LLaMA-7B Compression + Timed Eval"
    echo "=========================================="
    echo "GPU:                  $GPU"
    echo "MODEL:                $MODEL"
    echo "PYTHON:               $PYTHON"
    echo "Ratio:                $RATIO"
    echo "Nsamples:             $NSAMPLES"
    echo "Dataset:              $DATASET"
    echo "SeqLen:               $SEQ_LEN"
    echo "Eval batch size:      $EVAL_BATCH_SIZE"
    echo "r1_ratio:             $R1_RATIO"
    echo "Alpha/rho:            [$ALPHA_MIN, $ALPHA_MAX], rho=$RHO"
    echo "eps:                  $EPS"
    echo "Fisher mode:          $FISHER_MODE"
    echo "Innovation2:          $ENABLE_INNOVATION2"
    echo "lambda_phi/psi:       $LAMBDA_PHI / $LAMBDA_PSI"
    echo "Output:               $OUT_DIR"
    echo "=========================================="
} | tee "$RUN_LOG"

log_main "Compression command: ${COMPRESS_CMD[*]}"
compress_start="$(date +%s)"
set +e
(
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "${COMPRESS_CMD[@]}"
) 2>&1 | tee "$COMPRESS_LOG"
compress_rc="${PIPESTATUS[0]}"
set -e
compress_end="$(date +%s)"
compress_seconds="$((compress_end - compress_start))"

if [[ "$compress_rc" != "0" ]]; then
    log_main "Compression failed with exit code $compress_rc after ${compress_seconds}s ($(hms "$compress_seconds"))."
    exit "$compress_rc"
fi

MODEL_PT="$(find_latest_model)"
if [[ -z "$MODEL_PT" || ! -f "$MODEL_PT" ]]; then
    log_main "Compression finished, but no .pt model was found in $OUT_DIR."
    exit 1
fi

log_main "Compression finished: ${compress_seconds}s ($(hms "$compress_seconds"))."
log_main "Compressed model: $MODEL_PT"

eval_seconds=0
eval_rc=0
if [[ "$RUN_EVAL" == "1" ]]; then
    EVAL_CMD=(
        "$PYTHON" "$FR_SVD_PY"
        --eval_only
        --model_path "$MODEL_PT"
        --model_seq_len "$SEQ_LEN"
        --eval_batch_size "$EVAL_BATCH_SIZE"
        --eval_dtype fp16
        --DEV cuda:0
        "${EVAL_LARGE_ARGS[@]}"
    )

    log_main "Evaluation command: ${EVAL_CMD[*]}"
    eval_start="$(date +%s)"
    set +e
    (
        cd "$SCRIPT_DIR"
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "${EVAL_CMD[@]}"
    ) 2>&1 | tee "$EVAL_LOG"
    eval_rc="${PIPESTATUS[0]}"
    set -e
    eval_end="$(date +%s)"
    eval_seconds="$((eval_end - eval_start))"

    if [[ "$eval_rc" != "0" ]]; then
        log_main "Evaluation failed with exit code $eval_rc after ${eval_seconds}s ($(hms "$eval_seconds"))."
        exit "$eval_rc"
    fi
    log_main "Evaluation finished: ${eval_seconds}s ($(hms "$eval_seconds"))."
else
    log_main "Evaluation skipped by --no_eval."
fi

total_seconds="$((compress_seconds + eval_seconds))"
{
    echo "method=FR-SVD v1.6"
    echo "model=$MODEL"
    echo "model_pt=$MODEL_PT"
    echo "gpu=$GPU"
    echo "ratio=$RATIO"
    echo "nsamples=$NSAMPLES"
    echo "dataset=$DATASET"
    echo "seq_len=$SEQ_LEN"
    echo "fisher_mode=$FISHER_MODE"
    echo "enable_innovation2=$ENABLE_INNOVATION2"
    echo "compress_seconds=$compress_seconds"
    echo "compress_time=$(hms "$compress_seconds")"
    echo "eval_seconds=$eval_seconds"
    echo "eval_time=$(hms "$eval_seconds")"
    echo "total_seconds=$total_seconds"
    echo "total_time=$(hms "$total_seconds")"
    echo "compress_log=$COMPRESS_LOG"
    echo "eval_log=$EVAL_LOG"
} | tee "$SUMMARY_TXT"

{
    echo "method,model,gpu,ratio,nsamples,fisher_mode,enable_innovation2,compress_seconds,eval_seconds,total_seconds,out_dir,model_pt"
    echo "FR-SVD_v1.6,$MODEL,$GPU,$RATIO,$NSAMPLES,$FISHER_MODE,$ENABLE_INNOVATION2,$compress_seconds,$eval_seconds,$total_seconds,$OUT_DIR,$MODEL_PT"
} > "$SUMMARY_CSV"

log_main "All done. Total time: ${total_seconds}s ($(hms "$total_seconds"))."
log_main "Logs: $RUN_LOG | $COMPRESS_LOG | $EVAL_LOG"
