#!/bin/bash
# ============================================================
# Plan A: FR_SVD_v1.1 — Stage 2 使用额外秩（不从 Stage 1 抢占）
# 目标：Stage 1 保持与 SAES-SVD 相同的完整秩，Stage 2 额外补偿
# GPU: 5
# ============================================================

# nohup bash /data/zhangs/LLMs/Projects/SVD-LLM/run_planA_gpu5.sh > /data/zhangs/LLMs/Projects/SVD-LLM/result/planA_gpu5.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_BASE="${PROJECT_ROOT}/result/FR_SVD_planA"
GPU=5
DATE_STAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_BASE}/planA_all_${DATE_STAMP}.log"
mkdir -p "$OUTPUT_BASE"

echo "============================================================" | tee "$LOG_FILE"
echo "Plan A: Stage 2 额外秩补偿（FR_SVD_v1.1.py）" | tee -a "$LOG_FILE"
echo "对照基线: SAES-SVD rho=0.8 PPL=8.054" | tee -a "$LOG_FILE"
echo "核心改动: Stage 1 用完整秩 r, Stage 2 用额外 r2 秩" | tee -a "$LOG_FILE"
echo "GPU: $GPU, Time: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# 实验1: r1_ratio=0.8, rho=0.8（r2 = 20% 额外秩，与 SAES-SVD 最优 rho 一致）
echo "" | tee -a "$LOG_FILE"
echo ">>> 实验1: r1_ratio=0.8 (r2=20%额外), rho=0.8" | tee -a "$LOG_FILE"
bash "${PROJECT_ROOT}/scripts/compress/run_fr_svd.sh" \
    --gpu $GPU \
    --script FR_SVD_v1.1.py \
    --r1_ratio 0.8 \
    --rho 0.8 \
    --output_dir "${OUTPUT_BASE}/r1_0.8_rho0.8_s3" 2>&1 | tee -a "$LOG_FILE"

# 实验2: r1_ratio=0.9, rho=0.8（r2 = 10% 额外秩）
echo "" | tee -a "$LOG_FILE"
echo ">>> 实验2: r1_ratio=0.9 (r2=10%额外), rho=0.8" | tee -a "$LOG_FILE"
bash "${PROJECT_ROOT}/scripts/compress/run_fr_svd.sh" \
    --gpu $GPU \
    --script FR_SVD_v1.1.py \
    --r1_ratio 0.9 \
    --rho 0.8 \
    --output_dir "${OUTPUT_BASE}/r1_0.9_rho0.8_s3" 2>&1 | tee -a "$LOG_FILE"

# 实验3: r1_ratio=0.85, rho=0.8（r2 = 15% 额外秩）
echo "" | tee -a "$LOG_FILE"
echo ">>> 实验3: r1_ratio=0.85 (r2=15%额外), rho=0.8" | tee -a "$LOG_FILE"
bash "${PROJECT_ROOT}/scripts/compress/run_fr_svd.sh" \
    --gpu $GPU \
    --script FR_SVD_v1.1.py \
    --r1_ratio 0.85 \
    --rho 0.8 \
    --output_dir "${OUTPUT_BASE}/r1_0.85_rho0.8_s3" 2>&1 | tee -a "$LOG_FILE"

# 实验4: r1_ratio=0.8, rho=0.95（额外20% + 高 rho）
echo "" | tee -a "$LOG_FILE"
echo ">>> 实验4: r1_ratio=0.8 (r2=20%额外), rho=0.95" | tee -a "$LOG_FILE"
bash "${PROJECT_ROOT}/scripts/compress/run_fr_svd.sh" \
    --gpu $GPU \
    --script FR_SVD_v1.1.py \
    --r1_ratio 0.8 \
    --rho 0.95 \
    --output_dir "${OUTPUT_BASE}/r1_0.8_rho0.95_s3" 2>&1 | tee -a "$LOG_FILE"

# 汇总
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Plan A Results Summary" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "对照:" | tee -a "$LOG_FILE"
echo "  SAES-SVD rho=0.8:  PPL=8.054 (秩=r)" | tee -a "$LOG_FILE"
echo "  FR-SVD v1.0 r1=1.0 rho=0.95: PPL=8.084 (秩=r)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "本次实验（秩=r+r2，额外补偿）:" | tee -a "$LOG_FILE"

for config in "r1_0.8_rho0.8" "r1_0.9_rho0.8" "r1_0.85_rho0.8" "r1_0.8_rho0.95"; do
    dir="${OUTPUT_BASE}/${config}_s3"
    ppl=$(grep -r "PPL after" "$dir"/ 2>/dev/null | grep -oP "[0-9]+\.[0-9]+" | tail -1)
    echo "  ${config}: PPL=${ppl:-N/A}" | tee -a "$LOG_FILE"
done

echo "============================================================" | tee -a "$LOG_FILE"
echo "Plan A completed at $(date)" | tee -a "$LOG_FILE"
