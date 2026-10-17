#!/bin/bash
# SAES-SVD 修复版运行脚本

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GPU_DEVICE="1"
MODEL_PATH="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
RATIO="0.8"
DATASET="wikitext2"
NSAMPLES="128"
SEED="0"
OUTPUT_BASE="./result/SAES_SVD/llama-7b-hf/ratio_${RATIO}/dataset_${DATASET}_samples_${NSAMPLES}_seed_${SEED}"
PROFILING_PATH="${OUTPUT_BASE}/compress_output/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_profiling_${DATASET}_${NSAMPLES}_${SEED}.pt"
LOG_FILE="${OUTPUT_BASE}/logs/saes_svd_${TIMESTAMP}.log"

echo "=========================================="
echo "SAES-SVD 修复版运行"
echo "=========================================="
echo "GPU: $GPU_DEVICE"
echo "模型: $MODEL_PATH"
echo "压缩比: $RATIO"
echo "Profiling: $PROFILING_PATH"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""

# 确保日志目录存在
mkdir -p "$(dirname $LOG_FILE)"

# Step 1: 压缩（复用 profiling）
# echo "Step 1: SAES-SVD 压缩（复用 profiling）..."
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python SAES-SVD.py \
#     --step 1 \
#     --model "$MODEL_PATH" \
#     --ratio "$RATIO" \
#     --dataset "$DATASET" \
#     --whitening_nsamples "$NSAMPLES" \
#     --seed "$SEED" \
#     --DEV cuda:0 \
#     --save_path "${OUTPUT_BASE}/compress_output" \
#     --profiling_mat_path "$PROFILING_PATH" \
#     2>&1 | tee "$LOG_FILE"

# echo ""
# echo "✅ Step 1 完成！"
# echo ""

# Step 4: 评估
KEEP_RATIO=$(printf "%.1f" $(echo "1 - ${RATIO}" | bc -l))
COMPRESSED_MODEL="${OUTPUT_BASE}/compress_output/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_${KEEP_RATIO}.pt"

if [ -f "$COMPRESSED_MODEL" ]; then
    echo "Step 4: 评估 PPL..."
    echo "压缩模型: $COMPRESSED_MODEL"
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python SAES-SVD_v1.py \
        --step 4 \
        --model "$MODEL_PATH" \
        --model_path "$COMPRESSED_MODEL" \
        --DEV cuda:0 \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Step 4 完成！"
    else
        echo "❌ Step 4 失败！"
        exit 1
    fi
else
    echo "❌ 未找到压缩模型: $COMPRESSED_MODEL"
    exit 1
fi

echo "=========================================="
echo "完成时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "=========================================="

