#!/bin/bash

# 使用SVD压缩LLaMA-7B的示例
FINE_TUNE_PATH="."
# 1、以20%的压缩比运行数据白化处理
CUDA_VISIBLE_DEVICES=1 python SVDLLM.py --model /mnt/data/zhangshuai/LLMs/Models/jeffwan/llama-7b-hf --step 1 --ratio 0.5 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path /mnt/share/zhangshuai/LLMs/SVD-LLM/compress_output

# 作用:
# 对 Llama-7B 模型执行白化(whitening)和主成分分解(SVD)分析。
# --step 1:指定执行第1步,通常是白化和主成分提取。
# --ratio 0.2:压缩比为 20%(即保留 20% 能量/秩)。
# --whitening_nsamples 256:用 256 条样本做白化统计。
# --dataset wikitext2:白化用的数据集。
# --seed 3:设定随机种子保证可复现。
# --model_seq_len 2048:序列长度。
# --save_path .:保存路径。

# 低资源模式压缩
## 您还可以执行以下命令以适配低资源GPU(例如,Llama 7B模型仅需15G显存即可完成压缩)或用于大规模语言模型的压缩(例如Llama 65B模型)。
# python SVDLLM.py --model /home/zhangshuai/LLMs/Models/jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path ./ --run_low_resource

# 2、主成分压缩后模型权重导出
# --step 4:第4步,通常是用前面白化结果,正式进行模型结构的 SVD 压缩与导出。
# --model_path:白化后权重文件作为输入,生成压缩后模型。
CUDA_VISIBLE_DEVICES=1 python SVDLLM.py --step 4 --model_path /mnt/share/zhangshuai/LLMs/SVD-LLM/compress_output/_mnt_data_zhangshuai_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.5_2025_10_17_20_56_04.pt --eval_batch_size 2 --model_seq_len 2048
# /mnt/share/zhangshuai/LLMs/SVD-LLM/compress_output/_mnt_data_zhangshuai_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.5_2025_10_17_20_56_04.pt


# 3、第一阶段微调，使用LoRA对压缩模型进行微调（更新U）
# 顺序LoRA，目标模块选择*_u_proj，
CUDA_VISIBLE_DEVICES=1 python utils/LoRA.py --prune_model /mnt/share/zhangshuai/LLMs/SVD-LLM/compress_output/_mnt_data_zhangshuai_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.5_2025_10_17_20_56_04.pt --data_path /mnt/data/zhangshuai/LLMs/Datasets/alpaca-cleaned --output_dir first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64

: <<'END'
作用:
对 SVD-LLM 剪枝后模型进行第一阶段 LoRA 微调。
--prune_model:表示对压缩后模型再剪枝。
--data_path yahma/alpaca-cleaned:微调训练用数据集。
--output_dir $FINE_TUNE_PATH/first_half:输出微调结果到 first_half 目录。
--lora_target_modules ...:指定哪些模块应用 LoRA。
--lora_r 8:LoRA 子空间秩 r=8。
--num_epochs 3:微调3轮。
--learning_rate 1e-4、--batch_size 64:超参数。
END

# 4、第一阶段合并， LoRA 合并到 SVD-LLM 模型
# 将上一步 first_half 目录下的 LoRA 微调参数合（adapter_model.bin）并到 SVD-LLM 压缩模型中，形成新的权重。
CUDA_VISIBLE_DEVICES=1 python SVDLLM.py --model_path /mnt/share/zhangshuai/LLMs/SVD-LLM/compress_output/_mnt_data_zhangshuai_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.5_2025_10_17_20_56_04.pt --lora first_half --step 4  --eval_batch_size 2 --model_seq_len 512


# 5、第二阶段 LoRA 微调（更新V）
# 顺序LoRA，目标模块选择*_v_proj，更新V
CUDA_VISIBLE_DEVICES=2 python utils/LoRA.py --prune_model first_half/merge.pt --data_path /mnt/data/zhangshuai/LLMs/Datasets/alpaca-cleaned --output_dir second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64

# 6、第二阶段合并
CUDA_VISIBLE_DEVICES=2 python SVDLLM.py --model_path /mnt/share/zhangshuai/LLMs/SVD-LLM/compress_output/_mnt_data_zhangshuai_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.5_2025_10_17_20_56_04.pt --lora first_half --step 4  --eval_batch_size 2 --model_seq_len 512

# 7、最终模型合并
# 用第一阶段微调合并后的模型（merge.pt）为基准，进一步合并第二阶段 LoRA 微调结果，得到最终压缩+微调的 Llama-7B 模型。
CUDA_VISIBLE_DEVICES=0 python SVDLLM.py --model_path first_half/merge.pt --lora second_half --step 4 --eval_batch_size 2 --model_seq_len 512

# 8、测量模型精度
CUDA_VISIBLE_DEVICES=0 python SVDLLM.py --step 5 --model_path second_half/merge.pt --eval_batch_size 1 --model_seq_len 256

