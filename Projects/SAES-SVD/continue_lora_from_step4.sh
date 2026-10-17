#!/bin/bash
# ================================
# 继续执行LoRA微调 Steps 4-7
# ================================

# ========== 路径配置 ==========
COMPRESSED_MODEL="/data/zhangs/LLMs/SAES-SVD/alpha_ablation_gpu1/Exp-B5_rho0.8_eps1e-4_alpha0.4-0.6/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_saes_svd_v2.1.3_0.8.pt"
DATA_PATH="/data/zhangs/LLMs/Datasets/alpaca-cleaned"
OUTPUT_BASE="/data/zhangs/LLMs/SAES-SVD/exp_b5_lora_tuning"
FIRST_STAGE_DIR="$OUTPUT_BASE/first_half"
SECOND_STAGE_DIR="$OUTPUT_BASE/second_half"
CUDA_DEVICE=0

echo "=========================================="
echo "Continuing LoRA Fine-tuning from Step 4"
echo "=========================================="

# ========== 步骤4: 评估第一阶段结果 ==========
echo ""
echo "▶ Step 4: Evaluating first-stage model..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -c "
import torch
import sys
sys.path.append('/data/zhangs/LLMs/SAES-SVD')
from evaluater import ppl_eval
from transformers import AutoTokenizer

# 加载合并后的模型（只保存了model，不是dict）
print('Loading first-stage merged model...')
model = torch.load('$FIRST_STAGE_DIR/merge.pt', map_location='cpu', weights_only=False)
model = model.float().to('cuda:0')
model.eval()

# 加载tokenizer
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf', use_fast=False)

print('Evaluating first-stage model...')
ppl_eval(model, tokenizer, datasets=['wikitext2'], 
         model_seq_len=2048, batch_size=4, device='cuda:0')
" 2>&1 | tee "$OUTPUT_BASE/first_stage_eval.log"

# ========== 步骤5: 第二阶段LoRA微调 (更新V) ==========
echo ""
echo "▶ Step 5: Second-stage LoRA tuning (updating V projections)..."
echo "Target modules: q_v_proj, k_v_proj, v_v_proj, o_v_proj, gate_v_proj, down_v_proj, up_v_proj"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/LoRA.py \
    --prune_model "$FIRST_STAGE_DIR/merge.pt" \
    --data_path "$DATA_PATH" \
    --output_dir "$SECOND_STAGE_DIR" \
    --lora_target_modules "q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --micro_batch_size 4 \
    --val_set_size 2000 2>&1 | tee "$OUTPUT_BASE/second_stage_training.log"

# ========== 步骤6: 合并第二阶段LoRA权重 ==========
echo ""
echo "▶ Step 6: Merging second-stage LoRA weights..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -c "
import torch
import sys
import os
sys.path.append('/data/zhangs/LLMs/SAES-SVD')

# 加载第一阶段合并后的模型
print('Loading first-stage merged model...')
model = torch.load('$FIRST_STAGE_DIR/merge.pt', map_location='cpu', weights_only=False)

# 加载LoRA adapter
from peft import PeftModel
lora_path = '$SECOND_STAGE_DIR'
print(f'Loading second-stage LoRA adapter from {lora_path}...')

# 合并LoRA权重
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

# 保存最终模型
merge_path = os.path.join('$SECOND_STAGE_DIR', 'merge.pt')
print(f'Saving final merged model to {merge_path}...')
torch.save(model, merge_path)
print('✅ Second-stage merge complete!')
size_mb = os.path.getsize(merge_path) / (1024*1024)
print(f'Model size: {size_mb:.1f} MB')
" 2>&1 | tee "$OUTPUT_BASE/second_stage_merge.log"

# ========== 步骤7: 评估最终模型 ==========
echo ""
echo "▶ Step 7: Evaluating final fine-tuned model..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -c "
import torch
import sys
sys.path.append('/data/zhangs/LLMs/SAES-SVD')
from evaluater import ppl_eval
from transformers import AutoTokenizer

print('Loading final merged model...')
model = torch.load('$SECOND_STAGE_DIR/merge.pt', map_location='cpu', weights_only=False)
model = model.float().to('cuda:0')
model.eval()

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf', use_fast=False)

print('Evaluating final model...')
ppl_eval(model, tokenizer, datasets=['wikitext2'], 
         model_seq_len=2048, batch_size=4, device='cuda:0')
" 2>&1 | tee "$OUTPUT_BASE/final_eval.log"

# ========== 生成总结报告 ==========
echo ""
echo "=========================================="
echo "✅ LoRA Fine-tuning Complete!"
echo "=========================================="
echo ""
echo "📊 Result Summary:"
echo ""
echo "Baseline (Compressed only): PPL = 8.054761"
echo ""
echo "After First-stage LoRA (U projections):"
grep -i "wikitext2\|ppl" "$OUTPUT_BASE/first_stage_eval.log" | tail -3
echo ""
echo "After Second-stage LoRA (V projections):"
grep -i "wikitext2\|ppl" "$OUTPUT_BASE/final_eval.log" | tail -3
echo ""
echo "=========================================="
echo "Output files:"
echo "  - First stage model: $FIRST_STAGE_DIR/merge.pt"
echo "  - Second stage model: $SECOND_STAGE_DIR/merge.pt"
echo "  - Logs: $OUTPUT_BASE/*.log"
echo "=========================================="
echo ""
echo "预期效果："
echo "  - Baseline (仅压缩): PPL = 8.05"
echo "  - First LoRA: PPL ≈ 7.5-7.8"
echo "  - Second LoRA: PPL ≈ 7.2-7.5"
echo "=========================================="
