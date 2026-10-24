#!/bin/bash
# ================================
# Task 1: 测试基线模型的PPL
# ================================
# 用途: 确认原始LLaMA-7B在wikitext2上的困惑度
# 预期: 论文报告基线PPL=5.68

export CUDA_VISIBLE_DEVICES=5

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

MODEL_PATH="/data1/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
DEVICE="cuda:0"

echo "=========================================="
echo "Task 1: Testing Baseline Model PPL"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: wikitext2"
echo "Device: $DEVICE"
echo "=========================================="

# 直接使用Python评估原始模型
python -c "
import os
import torch
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('torch.cuda.device_count():', torch.cuda.device_count())
print('torch.cuda.current_device():', torch.cuda.current_device())
print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))

import sys
sys.path.append('/data1/zhangs/LLMs/SVD-LLM')
from utils.model_utils import get_model_from_huggingface
from evaluater import ppl_eval

print('📦 Loading baseline model...')
model, tokenizer = get_model_from_huggingface('$MODEL_PATH')
model = model.eval().float().to('$DEVICE')

print('🔍 Evaluating perplexity on wikitext2...')
ppl_eval(model, tokenizer, datasets=['wikitext2'], 
         model_seq_len=1024, batch_size=2, device='$DEVICE')
print('✅ Baseline evaluation complete!')
"

echo ""
echo "✅ 基线测试完成！"
echo "请对比论文报告的基线PPL=5.68"
