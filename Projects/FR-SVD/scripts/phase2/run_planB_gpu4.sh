#!/bin/bash
# ============================================================
# Plan B: 在 FR_SVD_v1.0 上调参（r1_ratio=1.0, 无 Stage 2）
# 目标：通过调 rho 参数超过 SAES-SVD 的 PPL=8.054
# GPU: 4
# ============================================================

# nohup bash /data/zhangs/LLMs/Projects/SVD-LLM/run_planB_gpu4.sh > /data/zhangs/LLMs/Projects/SVD-LLM/result/planB_gpu4.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_BASE="${PROJECT_ROOT}/result/FR_SVD_planB"
GPU=4
DATE_STAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_BASE}/planB_all_${DATE_STAMP}.log"
mkdir -p "$OUTPUT_BASE"

echo "============================================================" | tee "$LOG_FILE"
echo "Plan B: r1_ratio=1.0 参数调优" | tee -a "$LOG_FILE"
echo "对照基线: SAES-SVD rho=0.8 PPL=8.054" | tee -a "$LOG_FILE"
echo "已知: FR-SVD r1_ratio=1.0 rho=0.95 PPL=8.084" | tee -a "$LOG_FILE"
echo "GPU: $GPU, Time: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# 实验1: rho=0.8 (与 SAES-SVD 最优 rho 一致)
echo "" | tee -a "$LOG_FILE"
echo ">>> 实验1: r1_ratio=1.0, rho=0.8" | tee -a "$LOG_FILE"
bash "${PROJECT_ROOT}/scripts/compress/run_fr_svd.sh" \
    --gpu $GPU \
    --script FR_SVD_v1.0.py \
    --r1_ratio 1.0 \
    --rho 0.8 \
    --output_dir "${OUTPUT_BASE}/r1_1.0_rho0.8_s3" 2>&1 | tee -a "$LOG_FILE"

# 实验2: rho=0.7 (更激进的收缩)
echo "" | tee -a "$LOG_FILE"
echo ">>> 实验2: r1_ratio=1.0, rho=0.7" | tee -a "$LOG_FILE"
bash "${PROJECT_ROOT}/scripts/compress/run_fr_svd.sh" \
    --gpu $GPU \
    --script FR_SVD_v1.0.py \
    --r1_ratio 1.0 \
    --rho 0.7 \
    --output_dir "${OUTPUT_BASE}/r1_1.0_rho0.7_s3" 2>&1 | tee -a "$LOG_FILE"

# 实验3: rho=0.6
echo "" | tee -a "$LOG_FILE"
echo ">>> 实验3: r1_ratio=1.0, rho=0.6" | tee -a "$LOG_FILE"
bash "${PROJECT_ROOT}/scripts/compress/run_fr_svd.sh" \
    --gpu $GPU \
    --script FR_SVD_v1.0.py \
    --r1_ratio 1.0 \
    --rho 0.6 \
    --output_dir "${OUTPUT_BASE}/r1_1.0_rho0.6_s3" 2>&1 | tee -a "$LOG_FILE"

# 实验4: rho=0.85 (在 0.8 和 0.95 之间)
echo "" | tee -a "$LOG_FILE"
echo ">>> 实验4: r1_ratio=1.0, rho=0.85" | tee -a "$LOG_FILE"
bash "${PROJECT_ROOT}/scripts/compress/run_fr_svd.sh" \
    --gpu $GPU \
    --script FR_SVD_v1.0.py \
    --r1_ratio 1.0 \
    --rho 0.85 \
    --output_dir "${OUTPUT_BASE}/r1_1.0_rho0.85_s3" 2>&1 | tee -a "$LOG_FILE"

# 汇总
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Plan B Results Summary" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "对照:" | tee -a "$LOG_FILE"
echo "  SAES-SVD rho=0.8:  PPL=8.054" | tee -a "$LOG_FILE"
echo "  SAES-SVD rho=0.95: PPL=8.154" | tee -a "$LOG_FILE"
echo "  FR-SVD r1=1.0 rho=0.95: PPL=8.084 (已知)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "本次实验:" | tee -a "$LOG_FILE"

for rho_val in 0.8 0.7 0.6 0.85; do
    dir="${OUTPUT_BASE}/r1_1.0_rho${rho_val}_s3"
    ppl=$(grep -r "PPL after" "$dir"/ 2>/dev/null | grep -oP "[0-9]+\.[0-9]+" | tail -1)
    echo "  rho=${rho_val}: PPL=${ppl:-N/A}" | tee -a "$LOG_FILE"
done

echo "============================================================" | tee -a "$LOG_FILE"
echo "Plan B completed at $(date)" | tee -a "$LOG_FILE"
