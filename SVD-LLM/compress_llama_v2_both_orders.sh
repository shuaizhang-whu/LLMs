#!/bin/bash

# ================================
# 配置变量
# ================================
BASE_MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
DATA_PATH="/data/zhangs/LLMs/Datasets/alpaca-cleaned"
FINE_TUNE_PATH="/data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.4"
CUDA_DEVICE=5

# 压缩输出文件(根据实际生成的文件名调整)
COMPRESS_OUTPUT="$FINE_TUNE_PATH/compress_output/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.6.pt"

# 日志保存目录
LOG_DIR="$FINE_TUNE_PATH/logs"
mkdir -p "$LOG_DIR"

# 总日志文件
ALL_LOG="$LOG_DIR/all_stages_both_orders.log"
echo "======================================== 开始时间: $(date) ========================================" > "$ALL_LOG"

# 确保压缩输出目录存在
mkdir -p "$FINE_TUNE_PATH/compress_output"

# 确保 LoRA 输出目录存在
mkdir -p "$FINE_TUNE_PATH/first_half_U"
mkdir -p "$FINE_TUNE_PATH/second_half_UV"
mkdir -p "$FINE_TUNE_PATH/first_half_V"
mkdir -p "$FINE_TUNE_PATH/second_half_VU"

# ================================
# 使用SVD压缩LLaMA-7B的示例
# ================================

# 1、以20%的压缩比运行数据白化处理
echo "======== 阶段1: 数据白化处理 ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model $BASE_MODEL --step 1 --ratio 0.4 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $FINE_TUNE_PATH/compress_output 2>&1 | tee -a "$ALL_LOG"

# 2、对压缩后的模型进行困惑度(Perplexity, PPL)评估
echo "======== 阶段2: 压缩后模型PPL评估 ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --step 4 --model_path $COMPRESS_OUTPUT 2>&1 | tee -a "$ALL_LOG"

echo "" | tee -a "$ALL_LOG"
echo "========================================" | tee -a "$ALL_LOG"
echo "开始第一种微调顺序:先微调U.再微调V" | tee -a "$ALL_LOG"
echo "========================================" | tee -a "$ALL_LOG"
echo "" | tee -a "$ALL_LOG"

# 3、第一阶段微调.使用LoRA对压缩模型进行微调(更新U)
echo "======== 阶段3-U: 第一阶段LoRA微调(更新U) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/LoRA.py --prune_model $COMPRESS_OUTPUT --data_path $DATA_PATH --output_dir $FINE_TUNE_PATH/first_half_U --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 2>&1 | tee -a "$ALL_LOG"

# 4、一阶段结束.合并微调后的模型.并进行评估
echo "======== 阶段4-U: 一阶段微调后PPL评估(U微调后) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model_path $COMPRESS_OUTPUT --lora $FINE_TUNE_PATH/first_half_U --step 4 2>&1 | tee -a "$ALL_LOG"

# 5、第二阶段 LoRA 微调(更新V)
echo "======== 阶段5-U: 第二阶段LoRA微调(更新V) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/LoRA.py --prune_model $FINE_TUNE_PATH/first_half_U/merge.pt --data_path $DATA_PATH --output_dir $FINE_TUNE_PATH/second_half_UV --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 2>&1 | tee -a "$ALL_LOG"

# 6、二阶段结束.再次评估一阶段微调模型
echo "======== 阶段6-U: 再次评估一阶段微调模型(U微调后) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model_path $COMPRESS_OUTPUT --lora $FINE_TUNE_PATH/first_half_U --step 4 2>&1 | tee -a "$ALL_LOG"

# 7、最终模型合并
echo "======== 阶段7-U: 最终模型合并并评估(U→V顺序) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model_path $FINE_TUNE_PATH/first_half_U/merge.pt --lora $FINE_TUNE_PATH/second_half_UV --step 4 2>&1 | tee -a "$ALL_LOG"

# 8、测量模型吞吐速率
echo "======== 阶段8-U: 测量模型吞吐速率(U→V顺序) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --step 5 --model_path $FINE_TUNE_PATH/second_half_UV/merge.pt 2>&1 | tee -a "$ALL_LOG"

echo "" | tee -a "$ALL_LOG"
echo "========================================" | tee -a "$ALL_LOG"
echo "开始第二种微调顺序:先微调V.再微调U" | tee -a "$ALL_LOG"
echo "========================================" | tee -a "$ALL_LOG"
echo "" | tee -a "$ALL_LOG"

# 3、第一阶段微调.使用LoRA对压缩模型进行微调(先更新V)
echo "======== 阶段3-V: 第一阶段LoRA微调(更新V) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/LoRA.py --prune_model $COMPRESS_OUTPUT --data_path $DATA_PATH --output_dir $FINE_TUNE_PATH/first_half_V --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 2>&1 | tee -a "$ALL_LOG"

# 4、一阶段结束.合并微调后的模型.并进行评估
echo "======== 阶段4-V: 一阶段微调后PPL评估(V微调后) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model_path $COMPRESS_OUTPUT --lora $FINE_TUNE_PATH/first_half_V --step 4 2>&1 | tee -a "$ALL_LOG"

CUDA_VISIBLE_DEVICES=5 python SVDLLMv2.py --model_path /data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.4/compress_output/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.6.pt --lora /data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.4/first_half_V --step 4

# 5、第二阶段 LoRA 微调(再更新U)
echo "======== 阶段5-V: 第二阶段LoRA微调(更新U) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/LoRA.py --prune_model $FINE_TUNE_PATH/first_half_V/merge.pt --data_path $DATA_PATH --output_dir $FINE_TUNE_PATH/second_half_VU --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 2>&1 | tee -a "$ALL_LOG"

CUDA_VISIBLE_DEVICES=7 python utils/LoRA.py --prune_model /data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.4/first_half_V/merge.pt --data_path /data/zhangs/LLMs/Datasets/alpaca-cleaned --output_dir /data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.4/second_half_VU --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64

# 6、二阶段结束.再次评估一阶段微调模型
echo "======== 阶段6-V: 再次评估一阶段微调模型(V微调后) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model_path $COMPRESS_OUTPUT --lora $FINE_TUNE_PATH/first_half_V --step 4 2>&1 | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=7 python SVDLLMv2.py --model_path /data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.4/compress_output/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.6.pt --lora /data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.4/first_half_V --step 4

# 7、最终模型合并
echo "======== 阶段7-V: 最终模型合并并评估(V→U顺序) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --model_path $FINE_TUNE_PATH/first_half_V/merge.pt --lora $FINE_TUNE_PATH/second_half_VU --step 4 2>&1 | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=7 python SVDLLMv2.py --model_path /data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.4/first_half_V/merge.pt --lora /data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.4/second_half_VU --step 4

# 8、测量模型吞吐速率
echo "======== 阶段8-V: 测量模型吞吐速率(V→U顺序) ========" | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SVDLLMv2.py --step 5 --model_path $FINE_TUNE_PATH/second_half_VU/merge.pt 2>&1 | tee -a "$ALL_LOG"
CUDA_VISIBLE_DEVICES=7 python SVDLLMv2.py --step 5 --model_path /data/zhangs/LLMs/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.4/second_half_VU/merge.pt

echo "======================================== 结束时间: $(date) ========================================" | tee -a "$ALL_LOG"

# ================================
# 打印关键结果摘要
# ================================
echo ""
echo "========================================"
echo "所有阶段执行完成！"
echo "========================================"
echo ""
echo "======== U→V 微调顺序结果 ========" 
grep "阶段.*-U.*PPL\|阶段.*-U.*ppl\|阶段.*-U.*perplexity\|阶段.*-U.*throughput\|阶段.*-U.*speed" "$ALL_LOG" | head -20 || echo "未找到U→V顺序结果"
echo ""
echo "======== V→U 微调顺序结果 ========"
grep "阶段.*-V.*PPL\|阶段.*-V.*ppl\|阶段.*-V.*perplexity\|阶段.*-V.*throughput\|阶段.*-V.*speed" "$ALL_LOG" | head -20 || echo "未找到V→U顺序结果"
echo ""
echo "完整日志已保存到: $ALL_LOG"
echo "========================================"