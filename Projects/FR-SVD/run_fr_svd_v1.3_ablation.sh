#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# FR-SVD v1.3 消融实验：4 组实验
# 模型: LLaMA-7B, ratio=0.2, wikitext2, nsamples=128, seed=3
# v1.3 改进: alpha=[0.25,0.75], rho=1.0, CEALC-aware Stage 2, Fisher soft-reweighting

MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${SCRIPT_DIR}/FR_SVD_v1.3.py"
RESULT_BASE="${SCRIPT_DIR}/result/ablation_v1.3"
mkdir -p ${RESULT_BASE}

echo "============================================"
echo "FR-SVD v1.3 消融实验"
echo "Alpha 范围: [0.25, 0.75] (论文最优)"
echo "rho: 1.0 (不收缩)"
echo "Stage 2: CEALC-aware (含 Delta 信息)"
echo "Fisher: soft-reweighting (ACES 目标权重)"
echo "============================================"

# Exp1: Baseline - 两个创新点都关闭 (r1_ratio=1.0, use_fisher OFF)
echo ""
echo ">>> Exp1: Baseline (Fisher OFF, Stage2 OFF)"
CUDA_VISIBLE_DEVICES=4 ${PYTHON} ${SCRIPT} \
    --model ${MODEL} \
    --ratio 0.2 \
    --dataset wikitext2 \
    --nsamples 128 \
    --seed 3 \
    --DEV cuda:0 \
    --model_seq_len 2048 \
    --r1_ratio 1.0 \
    --rho 1.0 \
    --alpha_min 0.25 \
    --alpha_max 0.75 \
    --result_dir ${RESULT_BASE}/exp1_baseline \
    2>&1 | tee ${RESULT_BASE}/exp1_baseline.log

# Exp2: A only - Fisher soft-reweighting ON, Stage 2 OFF
echo ""
echo ">>> Exp2: A only (Fisher ON, Stage2 OFF)"
CUDA_VISIBLE_DEVICES=4 ${PYTHON} ${SCRIPT} \
    --model ${MODEL} \
    --ratio 0.2 \
    --dataset wikitext2 \
    --nsamples 128 \
    --seed 3 \
    --DEV cuda:0 \
    --model_seq_len 2048 \
    --r1_ratio 1.0 \
    --use_fisher \
    --rho 1.0 \
    --alpha_min 0.25 \
    --alpha_max 0.75 \
    --result_dir ${RESULT_BASE}/exp2_fisher_only \
    2>&1 | tee ${RESULT_BASE}/exp2_fisher_only.log

# Exp3: B only - Fisher OFF, CEALC-aware Stage 2 ON (r1_ratio=0.9)
echo ""
echo ">>> Exp3: B only (Fisher OFF, Stage2 ON, r1_ratio=0.9)"
CUDA_VISIBLE_DEVICES=4 ${PYTHON} ${SCRIPT} \
    --model ${MODEL} \
    --ratio 0.2 \
    --dataset wikitext2 \
    --nsamples 128 \
    --seed 3 \
    --DEV cuda:0 \
    --model_seq_len 2048 \
    --r1_ratio 0.9 \
    --rho 1.0 \
    --alpha_min 0.25 \
    --alpha_max 0.75 \
    --result_dir ${RESULT_BASE}/exp3_stage2_only \
    2>&1 | tee ${RESULT_BASE}/exp3_stage2_only.log

# Exp4: A+B - 两个创新点都开启
echo ""
echo ">>> Exp4: A+B (Fisher ON, Stage2 ON, r1_ratio=0.9)"
CUDA_VISIBLE_DEVICES=4 ${PYTHON} ${SCRIPT} \
    --model ${MODEL} \
    --ratio 0.2 \
    --dataset wikitext2 \
    --nsamples 128 \
    --seed 3 \
    --DEV cuda:0 \
    --model_seq_len 2048 \
    --r1_ratio 0.9 \
    --use_fisher \
    --rho 1.0 \
    --alpha_min 0.25 \
    --alpha_max 0.75 \
    --result_dir ${RESULT_BASE}/exp4_fisher_stage2 \
    2>&1 | tee ${RESULT_BASE}/exp4_fisher_stage2.log

echo ""
echo "============================================"
echo "全部 4 组消融实验完成！"
echo "结果汇总："
for exp in exp1_baseline exp2_fisher_only exp3_stage2_only exp4_fisher_stage2; do
    PPL=$(grep -oP "PPL after SVD:\s*\K[\d.]+" ${RESULT_BASE}/${exp}.log 2>/dev/null || echo "N/A")
    echo "  ${exp}: PPL=${PPL}"
done
echo "============================================"
