#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# 修复 psi_decomp bug 后重跑 Exp4 (Fisher + Stage2)

PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${PROJECT_ROOT}/FR_SVD_v1.2.py"
MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"

COMMON_ARGS="--model $MODEL --dataset wikitext2 --nsamples 128 --seed 3 \
--model_seq_len 2048 --ratio 0.2 --rho 0.6 --alpha_min 0.4 --alpha_max 0.6"

LOG_DIR="${PROJECT_ROOT}/result/ablation_v1.2"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=5

echo "===== [Fix] Exp 4: Fisher=ON, Stage2=ON ====="
echo "开始时间: $(date)"

$PYTHON $SCRIPT $COMMON_ARGS \
    --r1_ratio 0.9 \
    --use_fisher \
    --DEV cuda:0 \
    --save_path "$LOG_DIR/exp4_fisher_stage2" \
    2>&1 | tee "$LOG_DIR/exp4_fisher_stage2.log"

echo ""
echo "[Exp 4] 完成时间: $(date)"
