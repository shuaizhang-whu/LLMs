#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# FR-SVD v1.2 消融实验 — GPU4 运行 Baseline + A only（串行）

PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${SCRIPT_DIR}/FR_SVD_v1.2.py"
MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"

COMMON_ARGS="--model $MODEL --dataset wikitext2 --nsamples 128 --seed 3 \
--model_seq_len 2048 --ratio 0.2 --rho 0.6 --alpha_min 0.4 --alpha_max 0.6"

LOG_DIR="./result/ablation_v1.2"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=4

echo "===== GPU4: Baseline + A only ====="
echo "开始时间: $(date)"

# Exp 1: Baseline (no A, no B)
echo ""
echo "[Exp 1] Baseline: Fisher=OFF, Stage2=OFF"
$PYTHON $SCRIPT $COMMON_ARGS \
    --r1_ratio 1.0 \
    --DEV cuda:0 \
    --save_path "$LOG_DIR/exp1_baseline" \
    2>&1 | tee "$LOG_DIR/exp1_baseline.log"

echo ""
echo "[Exp 1] 完成时间: $(date)"
echo ""

# Exp 2: A only (Fisher ON, no Stage2)
echo "[Exp 2] A only: Fisher=ON, Stage2=OFF"
$PYTHON $SCRIPT $COMMON_ARGS \
    --r1_ratio 1.0 \
    --use_fisher \
    --DEV cuda:0 \
    --save_path "$LOG_DIR/exp2_fisher_only" \
    2>&1 | tee "$LOG_DIR/exp2_fisher_only.log"

echo ""
echo "[Exp 2] 完成时间: $(date)"
echo "===== GPU4 全部完成: $(date) ====="
