#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# ================================
# Task 1: 测试基线模型的PPL
# ================================
# 用途: 确认原始LLaMA-7B在wikitext2上的困惑度
# 预期: 论文报告基线PPL=5.68

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
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
import sys
sys.path.append('/data/zhangs/LLMs/Projects/SVD-LLM')
from utils.model_utils import get_model_from_huggingface
from evaluater import ppl_eval

print('📦 Loading baseline model...')
model, tokenizer = get_model_from_huggingface('$MODEL_PATH')
model = model.eval().float().to('$DEVICE')

print('🔍 Evaluating perplexity on wikitext2...')
ppl_eval(model, tokenizer, datasets=['wikitext2'], 
         model_seq_len=2048, batch_size=4, device='$DEVICE')
print('✅ Baseline evaluation complete!')
"

echo ""
echo "✅ 基线测试完成！"
echo "请对比论文报告的基线PPL=5.68"
