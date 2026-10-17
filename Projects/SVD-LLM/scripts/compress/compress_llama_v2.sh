#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# 使用SVD-LLM V2来压缩和微调
# ================================
# 配置变量
# ================================
BASE_MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
DATA_PATH="/data/zhangs/LLMs/Datasets/alpaca-cleaned"
FINE_TUNE_PATH="/data/zhangs/LLMs/Projects/SVD-LLM/result/SVD_V2/llama-7B-hf/ratio_0.2"
CUDA_DEVICE=6

# 压缩输出文件（根据实际生成的文件名调整）
COMPRESS_OUTPUT="$FINE_TUNE_PATH/compress_output/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.8.pt"

# ================================
# 使用SVD压缩LLaMA-7B的示例
# ================================

# 1、以20%的压缩比运行数据白化处理
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/SVDLLMv2.py" --model $BASE_MODEL --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $FINE_TUNE_PATH/compress_output

# 2、对压缩后的模型进行困惑度（Perplexity, PPL）评估
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/SVDLLMv2.py" --step 4 --model_path $COMPRESS_OUTPUT

# 3、第一阶段微调，使用LoRA对压缩模型进行微调（更新U）
# 顺序LoRA，目标模块选择*_u_proj
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/utils/LoRA.py" --prune_model $COMPRESS_OUTPUT --data_path $DATA_PATH --output_dir $FINE_TUNE_PATH/first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64

# 4、一阶段结束，合并微调后的模型，并进行评估
# 将上一步 first_half 目录下的 LoRA 微调参数（adapter_model.bin）并到 SVD-LLM 压缩模型中，形成新的权重
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/SVDLLMv2.py" --model_path $COMPRESS_OUTPUT --lora $FINE_TUNE_PATH/first_half --step 4

# 5、第二阶段 LoRA 微调（更新V）
# 顺序LoRA，目标模块选择*_v_proj，更新V
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/utils/LoRA.py" --prune_model $FINE_TUNE_PATH/first_half/merge.pt --data_path $DATA_PATH --output_dir $FINE_TUNE_PATH/second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64

# 6、二阶段结束，担心一阶段微调模型 first_half/merge.pt 在第二阶段训练中被意外修改，所以再评估一次
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/SVDLLMv2.py" --model_path $COMPRESS_OUTPUT --lora $FINE_TUNE_PATH/first_half --step 4

# 7、最终模型合并
# 用第一阶段微调合并后的模型（merge.pt）为基准，进一步合并第二阶段 LoRA 微调结果，得到最终压缩+微调的 Llama-7B 模型
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/SVDLLMv2.py" --model_path $FINE_TUNE_PATH/first_half/merge.pt --lora $FINE_TUNE_PATH/second_half --step 4

# 8、测量模型吞吐速率
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${PROJECT_ROOT}/SVDLLMv2.py" --step 5 --model_path $FINE_TUNE_PATH/second_half/merge.pt
