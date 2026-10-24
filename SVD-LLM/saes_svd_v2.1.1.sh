#!/bin/bash

# SAES-SVD v2.1.1 运行脚本
# 关键改进: 根据论文Section 5.3设置正确的β范围 [0.2, 0.4286]

RATIO=${1:-0.2}
GPU_ID=${2:-0}

echo "=========================================="
echo "SAES-SVD v2.1.1 参数优化版"
echo "=========================================="
echo "Compression ratio: $RATIO"
echo "GPU: $GPU_ID"
echo "=========================================="

# 设置路径
MODEL_PATH="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
RESULT_DIR="./result/SAES_SVD_v2.1.1/llama-7b-hf/ratio_${RATIO}/dataset_wikitext2_samples_128_seed_0"
mkdir -p "${RESULT_DIR}/logs"
mkdir -p "${RESULT_DIR}/compressed_model"

# 设置日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULT_DIR}/logs/saes_svd_v2.1.1_ratio${RATIO}_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始压缩 ratio=${RATIO}..." | tee "${LOG_FILE}"

# 运行压缩
CUDA_VISIBLE_DEVICES=$GPU_ID python SAES-SVD_v2.1.1.py \
    --model "$MODEL_PATH" \
    --ratio $RATIO \
    --result_dir "$RESULT_DIR" \
    --dataset wikitext2 \
    --nsamples 128 \
    --eps 1e-4 \
    2>&1 | tee -a "${LOG_FILE}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 压缩完成！" | tee -a "${LOG_FILE}"

# 提取关键结果
echo ""
echo "📊 提取关键结果..."
grep -E "PPL|β.*range|Average β|energy" "${LOG_FILE}" | tail -10
