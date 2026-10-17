#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Phase 1: 算法验证实验（Exp5-8b）共 6 组
# 验证 3 个算法改动：ACES线性退化分支、CEALC-aware Stage 2、Fisher soft-reweighting
# 注意：ACES线性退化分支已合入代码，Exp5/5b 通过对比历史结果验证其效果

MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${SCRIPT_DIR}/FR_SVD_v1.3.py"
RESULT_BASE="${SCRIPT_DIR}/result/ablation_v1.3_phase1"
mkdir -p ${RESULT_BASE}

echo "============================================"
echo "Phase 1: 算法验证（6 组实验）"
echo "============================================"
echo "验证 3 个算法改动："
echo "  ① ACES 线性退化分支（已合入代码）"
echo "  ② CEALC-aware Stage 2"
echo "  ③ Fisher soft-reweighting"
echo ""
echo "历史对照组："
echo "  v1.2 Exp1 (无退化分支, alpha=[0.4,0.6], rho=0.95): PPL=7.894"
echo "  v1.2 Exp2 (Psi直接变换):                            PPL=7.871"
echo "  v1.2 Exp3 (H-weighted Stage2):                      PPL=7.869"
echo "  v1.2 Exp4 (v1.2 Both):                              PPL=7.848"
echo "  v1.3 Exp1 (无退化分支, alpha=[0.25,0.75], rho=1.0): PPL=8.175"
echo "  v1.3 Exp4 (无退化分支, v1.3 Both):                  待完成"
echo "============================================"

# ─────────── 验证 ACES 线性退化分支 ───────────

# Exp5: 基线 + 窄 alpha 范围（对照 v1.2 Exp1=7.894）
# 若 Exp5 ≈ 7.894 → 窄范围下退化分支无影响
# 若 Exp5 < 7.894 → 窄范围下退化分支有正面效果
echo ""
echo ">>> Exp5: Baseline (v1.2 params, 验证窄范围退化分支)"
CUDA_VISIBLE_DEVICES=5 ${PYTHON} ${SCRIPT} \
    --model ${MODEL} \
    --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 \
    --DEV cuda:0 --model_seq_len 2048 \
    --r1_ratio 1.0 \
    --rho 0.95 --alpha_min 0.4 --alpha_max 0.6 \
    --result_dir ${RESULT_BASE}/exp5_baseline \
    2>&1 | tee ${RESULT_BASE}/exp5_baseline.log

# Exp5b: 基线 + 宽 alpha 范围（对照 v1.3 Exp1=8.175，跑在退化分支修复前）
# 若 Exp5b < 8.175 → 宽范围下退化分支有效
# 若 Exp5b ≈ 8.175 → 宽范围下退化分支未触发或无效
echo ""
echo ">>> Exp5b: Baseline (v1.3 params, 验证宽范围退化分支)"
CUDA_VISIBLE_DEVICES=5 ${PYTHON} ${SCRIPT} \
    --model ${MODEL} \
    --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 \
    --DEV cuda:0 --model_seq_len 2048 \
    --r1_ratio 1.0 \
    --rho 1.0 --alpha_min 0.25 --alpha_max 0.75 \
    --result_dir ${RESULT_BASE}/exp5b_wide_alpha \
    2>&1 | tee ${RESULT_BASE}/exp5b_wide_alpha.log

# ─────────── 验证 CEALC-aware Stage 2 ───────────

# Exp6: CEALC-aware Stage 2 only（对照 v1.2 Exp3=7.869）
echo ""
echo ">>> Exp6: B only (CEALC-aware Stage 2, v1.2 params)"
CUDA_VISIBLE_DEVICES=5 ${PYTHON} ${SCRIPT} \
    --model ${MODEL} \
    --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 \
    --DEV cuda:0 --model_seq_len 2048 \
    --r1_ratio 0.9 \
    --rho 0.95 --alpha_min 0.4 --alpha_max 0.6 \
    --result_dir ${RESULT_BASE}/exp6_cealc_stage2 \
    2>&1 | tee ${RESULT_BASE}/exp6_cealc_stage2.log

# ─────────── 验证 Fisher soft-reweighting ───────────

# Exp7: Fisher soft-reweight only（对照 v1.2 Exp2=7.871）
echo ""
echo ">>> Exp7: A only (Fisher soft-reweight, v1.2 params)"
CUDA_VISIBLE_DEVICES=5 ${PYTHON} ${SCRIPT} \
    --model ${MODEL} \
    --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 \
    --DEV cuda:0 --model_seq_len 2048 \
    --r1_ratio 1.0 \
    --use_fisher \
    --rho 0.95 --alpha_min 0.4 --alpha_max 0.6 \
    --result_dir ${RESULT_BASE}/exp7_fisher_soft \
    2>&1 | tee ${RESULT_BASE}/exp7_fisher_soft.log

# ─────────── 验证联合效果 ───────────

# Exp8: 两个创新点 + v1.2 参数（对照 v1.2 Exp4=7.848）
echo ""
echo ">>> Exp8: A+B (Both, v1.2 params)"
CUDA_VISIBLE_DEVICES=5 ${PYTHON} ${SCRIPT} \
    --model ${MODEL} \
    --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 \
    --DEV cuda:0 --model_seq_len 2048 \
    --r1_ratio 0.9 \
    --use_fisher \
    --rho 0.95 --alpha_min 0.4 --alpha_max 0.6 \
    --result_dir ${RESULT_BASE}/exp8_both \
    2>&1 | tee ${RESULT_BASE}/exp8_both.log

# Exp8b: 两个创新点 + v1.3 参数（对照 v1.3 Exp4，跑在退化分支修复前）
echo ""
echo ">>> Exp8b: A+B (Both, v1.3 params, 验证宽范围下联合 + 退化分支)"
CUDA_VISIBLE_DEVICES=5 ${PYTHON} ${SCRIPT} \
    --model ${MODEL} \
    --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 \
    --DEV cuda:0 --model_seq_len 2048 \
    --r1_ratio 0.9 \
    --use_fisher \
    --rho 1.0 --alpha_min 0.25 --alpha_max 0.75 \
    --result_dir ${RESULT_BASE}/exp8b_both_wide \
    2>&1 | tee ${RESULT_BASE}/exp8b_both_wide.log

echo ""
echo "============================================"
echo "Phase 1 全部完成！结果汇总："
echo "============================================"
echo ""
echo "─── ACES 线性退化分支验证 ───"
P5=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" ${RESULT_BASE}/exp5_baseline.log 2>/dev/null || echo "N/A")
P5b=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" ${RESULT_BASE}/exp5b_wide_alpha.log 2>/dev/null || echo "N/A")
echo "  Exp5  窄alpha基线 (对照v1.2 Exp1=7.894): PPL=${P5}"
echo "  Exp5b 宽alpha基线 (对照v1.3 Exp1=8.175): PPL=${P5b}"
echo ""
echo "─── 算法改进验证（v1.2 参数下）───"
P6=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" ${RESULT_BASE}/exp6_cealc_stage2.log 2>/dev/null || echo "N/A")
P7=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" ${RESULT_BASE}/exp7_fisher_soft.log 2>/dev/null || echo "N/A")
P8=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" ${RESULT_BASE}/exp8_both.log 2>/dev/null || echo "N/A")
P8b=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" ${RESULT_BASE}/exp8b_both_wide.log 2>/dev/null || echo "N/A")
echo "  Exp6  CEALC-aware Stage2 (对照v1.2 Exp3=7.869): PPL=${P6}"
echo "  Exp7  Fisher soft-reweight (对照v1.2 Exp2=7.871): PPL=${P7}"
echo "  Exp8  Both v1.2参数 (对照v1.2 Exp4=7.848):       PPL=${P8}"
echo "  Exp8b Both v1.3参数 (对照v1.3 Exp4=?):           PPL=${P8b}"
echo "============================================"
