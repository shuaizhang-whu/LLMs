#!/bin/bash
# 独立评估 LLaMA-30B 压缩模型的 PPL（逐层模式，不需要整模型上 GPU）
# 用法：bash run_eval_llama30b.sh [GPU_ID]

GPU=${1:-0}
MODEL_PT="/data/zhangs/LLMs/FR-SVD/result/v1.5/llama_30b/ns64_r1_0.92_rho1.0_20260418_230949/_data_zhangs_LLMs_Models_llama_30b_fr_svd_0.8.pt"

export CUDA_VISIBLE_DEVICES=${GPU}

echo "============================================"
echo "  LLaMA-30B PPL 评估（逐层模式）"
echo "  GPU: ${GPU}"
echo "  模型: ${MODEL_PT}"
echo "============================================"

/data/zhangs/anaconda3/envs/compress/bin/python /data/zhangs/LLMs/FR-SVD/FR_SVD_v1.5.py \
    --eval_only \
    --model_path "${MODEL_PT}" \
    --model_seq_len 2048 \
    --eval_batch_size 1 \
    --eval_large \
    --DEV cuda:0
