#!/bin/bash

# ================================
# 配置变量
# ================================
BASE_MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
DATA_PATH="/data/zhangs/LLMs/Datasets/alpaca-cleaned"
FINE_TUNE_PATH="/data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.2"
CUDA_DEVICE=6

# 压缩输出文件（根据实际生成的文件名调整）
COMPRESS_OUTPUT="$FINE_TUNE_PATH/compress_output/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.8.pt"

# 日志保存目录
LOG_DIR="$FINE_TUNE_PATH/logs"
mkdir -p "$LOG_DIR"

# 总日志文件
ALL_LOG="$LOG_DIR/all_stages.log"
echo "======================================== 开始时间: $(date) ========================================" > "$ALL_LOG"

# ================================
# 使用SVD压缩LLaMA-7B的示例
# ================================

# 1、以20%的压缩比运行数据白化处理
echo "======== 阶段1: 数据白化处理 ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model $BASE_MODEL --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $FINE_TUNE_PATH/compress_output 2>&1 | tee -a "$ALL_LOG"

# 2、对压缩后的模型进行困惑度（Perplexity, PPL）评估
echo "======== 阶段2: 压缩后模型PPL评估 ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --step 4 --model_path $COMPRESS_OUTPUT 2>&1 | tee -a "$ALL_LOG"

# 3、第一阶段微调，使用LoRA对压缩模型进行微调（更新U）
echo "======== 阶段3: 第一阶段LoRA微调（更新U） ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/LoRA.py --prune_model $COMPRESS_OUTPUT --data_path $DATA_PATH --output_dir $FINE_TUNE_PATH/first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 2>&1 | tee -a "$ALL_LOG"

# 4、一阶段结束，合并微调后的模型，并进行评估
echo "======== 阶段4: 一阶段微调后PPL评估 ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model_path $COMPRESS_OUTPUT --lora $FINE_TUNE_PATH/first_half --step 4 2>&1 | tee -a "$ALL_LOG"

# 5、第二阶段 LoRA 微调（更新V）
echo "======== 阶段5: 第二阶段LoRA微调（更新V） ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/LoRA.py --prune_model $FINE_TUNE_PATH/first_half/merge.pt --data_path $DATA_PATH --output_dir $FINE_TUNE_PATH/second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 2>&1 | tee -a "$ALL_LOG"

# 6、二阶段结束，担心一阶段微调模型 first_half/merge.pt 在第二阶段训练中被意外修改，所以再评估一次
echo "======== 阶段6: 再次评估一阶段微调模型 ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model_path $COMPRESS_OUTPUT --lora $FINE_TUNE_PATH/first_half --step 4 2>&1 | tee -a "$ALL_LOG"

# 7、最终模型合并
echo "======== 阶段7: 最终模型合并并评估 ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model_path $FINE_TUNE_PATH/first_half/merge.pt --lora $FINE_TUNE_PATH/second_half --step 4 2>&1 | tee -a "$ALL_LOG"

# 8、测量模型吞吐速率
echo "======== 阶段8: 测量模型吞吐速率 ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --step 5 --model_path $FINE_TUNE_PATH/second_half/merge.pt 2>&1 | tee -a "$ALL_LOG"

echo "======================================== 结束时间: $(date) ========================================" | tee -a "$ALL_LOG"

# ================================
# 打印关键结果摘要
# ================================
echo ""
echo "========================================"
echo "所有阶段执行完成！"
echo "========================================"
echo ""
echo "关键结果摘要："
echo "----------------------------------------"
grep -i "ppl\|perplexity\|throughput\|speed\|tokens/s" "$ALL_LOG" || echo "未找到评估结果"
echo "----------------------------------------"
echo ""
echo "完整日志已保存到: $ALL_LOG"
echo "========================================"

