#!/bin/bash
# SAES-SVD 完整工作流脚本（压缩 + 微调 + 评估）
# 参考 compress_llama_v2.sh 的完整流程

# ================================
# 配置变量
# ================================
CUDA_DEVICE=6
BASE_MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
DATA_PATH="/data/zhangs/LLMs/Datasets/alpaca-cleaned"
COMPRESSION_RATIO=0.2
DATASET="wikitext2"
WHITENING_SAMPLES=256
SEED=0

# 提取模型名称
MODEL_NAME=$(basename ${BASE_MODEL})

# 构建输出路径（参考 SVD_V2 结构）
RESULT_BASE="./result/SAES_SVD/${MODEL_NAME}/ratio_${COMPRESSION_RATIO}"
OUTPUT_PATH="${RESULT_BASE}/dataset_${DATASET}_samples_${WHITENING_SAMPLES}_seed_${SEED}"

# 设置 CUDA
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# 创建目录
mkdir -p ${OUTPUT_PATH}/compress_output
mkdir -p ${OUTPUT_PATH}/first_half
mkdir -p ${OUTPUT_PATH}/second_half
mkdir -p ${OUTPUT_PATH}/logs

# 日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_PATH}/logs/full_pipeline_${TIMESTAMP}.log"

log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

log "======================================"
log "SAES-SVD 完整工作流"
log "======================================"
log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
log "GPU 设备: ${CUDA_DEVICE}"
log "基础模型: ${BASE_MODEL}"
log "压缩比率: ${COMPRESSION_RATIO}"
log "数据集: ${DATASET}"
log "微调数据: ${DATA_PATH}"
log "输出目录: ${OUTPUT_PATH}"
log "======================================"

# ================================
# Step 1: 白化压缩
# ================================
log ""
log "Step 1: 执行 SAES-SVD 白化压缩..."
log "----------------------------------------"

python SAES-SVD.py \
    --model ${BASE_MODEL} \
    --ratio ${COMPRESSION_RATIO} \
    --dataset ${DATASET} \
    --whitening_nsamples ${WHITENING_SAMPLES} \
    --save_path ${OUTPUT_PATH}/compress_output \
    --seed ${SEED} \
    --step 1 2>&1 | tee -a "${LOG_FILE}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ Step 1 失败！"
    exit 1
fi

log "✅ Step 1 完成！"

# 查找压缩后的模型
KEEP_RATIO=$(echo "1 - ${COMPRESSION_RATIO}" | bc)
COMPRESS_OUTPUT=$(ls ${OUTPUT_PATH}/compress_output/*_whitening_only_${KEEP_RATIO}.pt 2>/dev/null | head -n 1)

if [ -z "${COMPRESS_OUTPUT}" ]; then
    log "❌ 找不到压缩模型文件"
    exit 1
fi

log "压缩模型: ${COMPRESS_OUTPUT}"

# ================================
# Step 2: 评估压缩后的困惑度
# ================================
log ""
log "Step 2: 评估压缩后模型的困惑度..."
log "----------------------------------------"

python SAES-SVD.py \
    --model ${BASE_MODEL} \
    --model_path ${COMPRESS_OUTPUT} \
    --step 4 2>&1 | tee -a "${LOG_FILE}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ Step 2 失败！"
    exit 1
fi

log "✅ Step 2 完成！"

# ================================
# Step 3: 第一阶段 LoRA 微调（更新 U）
# ================================
log ""
log "Step 3: 第一阶段 LoRA 微调（更新 U 投影）..."
log "----------------------------------------"

python utils/LoRA.py \
    --prune_model ${COMPRESS_OUTPUT} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_PATH}/first_half \
    --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj \
    --lora_r 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 64 2>&1 | tee -a "${LOG_FILE}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ Step 3 失败！"
    exit 1
fi

log "✅ Step 3 完成！"

# ================================
# Step 4: 合并第一阶段并评估
# ================================
log ""
log "Step 4: 合并第一阶段 LoRA 权重并评估..."
log "----------------------------------------"

python SAES-SVD.py \
    --model ${BASE_MODEL} \
    --model_path ${COMPRESS_OUTPUT} \
    --lora ${OUTPUT_PATH}/first_half \
    --step 4 2>&1 | tee -a "${LOG_FILE}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ Step 4 失败！"
    exit 1
fi

log "✅ Step 4 完成！"

# ================================
# Step 5: 第二阶段 LoRA 微调（更新 V）
# ================================
log ""
log "Step 5: 第二阶段 LoRA 微调（更新 V 投影）..."
log "----------------------------------------"

python utils/LoRA.py \
    --prune_model ${OUTPUT_PATH}/first_half/merge.pt \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_PATH}/second_half \
    --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj \
    --lora_r 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 64 2>&1 | tee -a "${LOG_FILE}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ Step 5 失败！"
    exit 1
fi

log "✅ Step 5 完成！"

# ================================
# Step 6: 验证第一阶段模型（防止被覆盖）
# ================================
log ""
log "Step 6: 验证第一阶段模型..."
log "----------------------------------------"

python SAES-SVD.py \
    --model ${BASE_MODEL} \
    --model_path ${COMPRESS_OUTPUT} \
    --lora ${OUTPUT_PATH}/first_half \
    --step 4 2>&1 | tee -a "${LOG_FILE}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ Step 6 失败！"
    exit 1
fi

log "✅ Step 6 完成！"

# ================================
# Step 7: 最终模型合并
# ================================
log ""
log "Step 7: 合并第二阶段 LoRA 权重（最终模型）..."
log "----------------------------------------"

python SAES-SVD.py \
    --model ${BASE_MODEL} \
    --model_path ${OUTPUT_PATH}/first_half/merge.pt \
    --lora ${OUTPUT_PATH}/second_half \
    --step 4 2>&1 | tee -a "${LOG_FILE}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ Step 7 失败！"
    exit 1
fi

log "✅ Step 7 完成！"

# ================================
# Step 8: 效率评估
# ================================
log ""
log "Step 8: 测量最终模型的推理效率..."
log "----------------------------------------"

python SAES-SVD.py \
    --model ${BASE_MODEL} \
    --model_path ${OUTPUT_PATH}/second_half/merge.pt \
    --step 5 2>&1 | tee -a "${LOG_FILE}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ Step 8 失败！"
    exit 1
fi

log "✅ Step 8 完成！"

# ================================
# 完成总结
# ================================
log ""
log "======================================"
log "SAES-SVD 完整工作流执行完毕！"
log "======================================"
log "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
log ""
log "📁 输出文件位置:"
log "  压缩模型: ${COMPRESS_OUTPUT}"
log "  第一阶段微调: ${OUTPUT_PATH}/first_half/"
log "  第二阶段微调: ${OUTPUT_PATH}/second_half/"
log "  最终模型: ${OUTPUT_PATH}/second_half/merge.pt"
log "  日志文件: ${LOG_FILE}"
log ""
log "📊 模型信息:"
log "  原始模型: ${BASE_MODEL}"
log "  压缩比率: ${COMPRESSION_RATIO}"
log "  保留比率: ${KEEP_RATIO}"
log ""
log "🎉 全部完成！"
log ""
