#!/bin/bash
# ============================================================
# FR-SVD 诊断：r1_ratio=1.0 (no Stage 2) 对照实验
# 如果 PPL ≈ SAES-SVD (8.05), 则问题在 Stage 2
# 如果 PPL >> 8, 则问题在 Stage 1
# ============================================================

GPU="${1:-5}"

PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATE_STAMP=$(date +"%Y%m%d_%H%M%S")

MODEL_PATH="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
OUTPUT_DIR="result/FR_SVD_diag"

echo "============================================"
echo "FR-SVD 诊断实验: r1_ratio=1.0 (no Stage 2)"
echo "============================================"
echo "GPU: $GPU"
echo "Time: $(date)"
echo "============================================"

# 实验1: r1_ratio=1.0 (等同于 SAES-SVD Stage 1 only)
echo ""
echo ">>> 实验1: r1_ratio=1.0, rho=0.95 (should match SAES-SVD)"
RESULT_DIR1="${OUTPUT_DIR}/r1ratio_1.0_rho0.95_s3"
mkdir -p "$RESULT_DIR1"
LOG1="${RESULT_DIR1}/log_${DATE_STAMP}.txt"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT_DIR/FR_SVD_v1.0.py" \
    --model "$MODEL_PATH" \
    --ratio 0.2 \
    --dataset wikitext2 \
    --nsamples 128 \
    --seed 3 \
    --model_seq_len 2048 \
    --eps 1e-4 \
    --r1_ratio 1.0 \
    --rho 0.95 \
    --alpha_min 0.4 \
    --alpha_max 0.6 \
    --result_dir "$RESULT_DIR1" \
    --DEV cuda:0 2>&1 | tee "$LOG1"

PPL1=$(grep "PPL after" "$LOG1" | grep -oP "[0-9]+\.[0-9]+")
echo "r1_ratio=1.0: PPL = $PPL1"

# 实验2: r1_ratio=0.8 + rho=0.8 (match SAES-SVD rho) 
echo ""
echo ">>> 实验2: r1_ratio=0.8, rho=0.8 (match Contrast-A rho)"
RESULT_DIR2="${OUTPUT_DIR}/r1ratio_0.8_rho0.8_s3"
mkdir -p "$RESULT_DIR2"
LOG2="${RESULT_DIR2}/log_${DATE_STAMP}.txt"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT_DIR/FR_SVD_v1.0.py" \
    --model "$MODEL_PATH" \
    --ratio 0.2 \
    --dataset wikitext2 \
    --nsamples 128 \
    --seed 3 \
    --model_seq_len 2048 \
    --eps 1e-4 \
    --r1_ratio 0.8 \
    --rho 0.8 \
    --alpha_min 0.4 \
    --alpha_max 0.6 \
    --result_dir "$RESULT_DIR2" \
    --DEV cuda:0 2>&1 | tee "$LOG2"

PPL2=$(grep "PPL after" "$LOG2" | grep -oP "[0-9]+\.[0-9]+")
echo "r1_ratio=0.8, rho=0.8: PPL = $PPL2"

# 汇总
echo ""
echo "============================================"
echo "汇总 (对照: SAES-SVD PPL=8.05)"
echo "============================================"
echo "r1_ratio=1.0, rho=0.95: PPL = $PPL1"
echo "r1_ratio=0.8, rho=0.8:  PPL = $PPL2"
echo "r1_ratio=0.8, rho=0.95: PPL = 22.62 (已知结果)"
echo "============================================"
