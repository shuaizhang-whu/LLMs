#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# FR-SVD v1.2 消融实验 — GPU5 运行 B only + A+B（串行）

PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${SCRIPT_DIR}/FR_SVD_v1.2.py"
MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"

COMMON_ARGS="--model $MODEL --dataset wikitext2 --nsamples 128 --seed 3 \
--model_seq_len 2048 --ratio 0.2 --rho 0.6 --alpha_min 0.4 --alpha_max 0.6"

LOG_DIR="./result/ablation_v1.2"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=5

echo "===== GPU5: B only + A+B ====="
echo "开始时间: $(date)"

# Exp 3: B only (no Fisher, Stage2 ON)
echo ""
echo "[Exp 3] B only: Fisher=OFF, Stage2=ON (r1_ratio=0.9)"
$PYTHON $SCRIPT $COMMON_ARGS \
    --r1_ratio 0.9 \
    --DEV cuda:0 \
    --save_path "$LOG_DIR/exp3_stage2_only" \
    2>&1 | tee "$LOG_DIR/exp3_stage2_only.log"

echo ""
echo "[Exp 3] 完成时间: $(date)"
echo ""

# Exp 4: A+B (Fisher ON + Stage2 ON)
echo "[Exp 4] A+B: Fisher=ON, Stage2=ON (r1_ratio=0.9)"
$PYTHON $SCRIPT $COMMON_ARGS \
    --r1_ratio 0.9 \
    --use_fisher \
    --DEV cuda:0 \
    --save_path "$LOG_DIR/exp4_fisher_stage2" \
    2>&1 | tee "$LOG_DIR/exp4_fisher_stage2.log"

echo ""
echo "[Exp 4] 完成时间: $(date)"
echo "===== GPU5 全部完成: $(date) ====="
