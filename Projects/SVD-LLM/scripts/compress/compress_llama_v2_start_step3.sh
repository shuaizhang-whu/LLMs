#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ================================
# 配置变量
# ================================
BASE_MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
DATA_PATH="/data/zhangs/LLMs/Datasets/alpaca-cleaned"
COMPRESS_OUTPUT_PATH="/data/zhangs/LLMs/Projects/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.2"
FINE_TUNE_PATH="/data/zhangs/LLMs/Projects/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.2/first_V_then_U"
CUDA_DEVICE=7

# 压缩输出文件（使用已经压缩好的模型）
COMPRESS_OUTPUT="$COMPRESS_OUTPUT_PATH/compress_output/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.8.pt"

# 日志保存目录（使用reverse标识区分）
LOG_DIR="$FINE_TUNE_PATH/logs_reverse"
mkdir -p "$LOG_DIR"

# 总日志文件
ALL_LOG="$LOG_DIR/all_stages_reverse.log"
echo "======================================== 开始时间: $(date) ========================================" > "$ALL_LOG"
echo "======== 反向微调顺序: 先微调V，再微调U ========" | tee -a "$ALL_LOG"

# ================================
# 反向顺序微调：先V后U
# ================================

# 3、第一阶段微调，使用LoRA对压缩模型进行微调（先更新V）
# echo "======== 阶段3: 第一阶段LoRA微调（先更新V） ========" | tee -a "$ALL_LOG"
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/utils/LoRA.py" --prune_model $COMPRESS_OUTPUT --data_path $DATA_PATH --output_dir $FINE_TUNE_PATH/first_half_reverse --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 2>&1 | tee -a "$ALL_LOG"

# 4、一阶段结束，合并微调后的模型，并进行评估
echo "======== 阶段4: 一阶段微调后PPL评估（V微调后） ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/SVDLLMv2.py" --model_path $COMPRESS_OUTPUT --lora $FINE_TUNE_PATH/first_half_reverse --step 4 2>&1 | tee -a "$ALL_LOG"

# 5、第二阶段 LoRA 微调（再更新U）
echo "======== 阶段5: 第二阶段LoRA微调（再更新U） ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/utils/LoRA.py" --prune_model $FINE_TUNE_PATH/first_half_reverse/merge.pt --data_path $DATA_PATH --output_dir $FINE_TUNE_PATH/second_half_reverse --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 2>&1 | tee -a "$ALL_LOG"

# 6、二阶段结束，再次评估一阶段微调模型
echo "======== 阶段6: 再次评估一阶段微调模型（V微调后） ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/SVDLLMv2.py" --model_path $COMPRESS_OUTPUT --lora $FINE_TUNE_PATH/first_half_reverse --step 4 2>&1 | tee -a "$ALL_LOG"

# 7、最终模型合并
echo "======== 阶段7: 最终模型合并并评估（V+U微调后） ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/SVDLLMv2.py" --model_path $FINE_TUNE_PATH/first_half_reverse/merge.pt --lora $FINE_TUNE_PATH/second_half_reverse --step 4 2>&1 | tee -a "$ALL_LOG"

# 8、测量模型吞吐速率
echo "======== 阶段8: 测量模型吞吐速率 ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/SVDLLMv2.py" --step 5 --model_path $FINE_TUNE_PATH/second_half_reverse/merge.pt 2>&1 | tee -a "$ALL_LOG"

echo "======================================== 结束时间: $(date) ========================================" | tee -a "$ALL_LOG"

# ================================
# 打印关键结果摘要
# ================================
echo ""
echo "========================================"
echo "所有阶段执行完成！（反向微调顺序：先V后U）"
echo "========================================"
echo ""
echo "关键结果摘要："
echo "----------------------------------------"
grep -i "ppl\|perplexity\|throughput\|speed\|tokens/s" "$ALL_LOG" || echo "未找到评估结果"
echo "----------------------------------------"
echo ""
echo "完整日志已保存到: $ALL_LOG"
echo "========================================"
