#!/bin/bash
# ================================
# 执行LoRA Steps 5-7（跳过Step 4）
# ================================

# ========== 路径配置 ==========
# ★ 使用原始压缩模型作为second-stage的输入
COMPRESSED_MODEL="/data/zhangs/LLMs/Projects/SAES-SVD/alpha_ablation_gpu1/Exp-B5_rho0.8_eps1e-4_alpha0.4-0.6/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_saes_svd_v2.1.3_0.8.pt"
DATA_PATH="/data/zhangs/LLMs/Datasets/alpaca-cleaned"
OUTPUT_BASE="/data/zhangs/LLMs/Projects/SAES-SVD/exp_b5_lora_tuning"
FIRST_STAGE_DIR="$OUTPUT_BASE/first_half"
SECOND_STAGE_DIR="$OUTPUT_BASE/second_half"
CUDA_DEVICE=7

echo "=========================================="
echo "Executing LoRA Steps 5-7"
echo "=========================================="
echo "First-stage PPL: 7.47 (completed)"
echo "Now training second-stage LoRA on V projections..."
echo "=========================================="

# ========== 步骤5: 第二阶段LoRA微调 (更新V) ==========
echo ""
echo "▶ Step 5: Second-stage LoRA tuning (updating V projections)..."
echo "Input: First-stage merged model (compressed + U adapter)"

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
sys.path.append('/data/zhangs/LLMs/Projects/SAES-SVD')

# 加载第一阶段merged模型 (compressed + U adapter)
print('Loading first-stage merged model...')
checkpoint = torch.load('$FIRST_STAGE_DIR/merge.pt', map_location='cpu', weights_only=False)
model = checkpoint['model']
tokenizer = checkpoint['tokenizer']

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
torch.save({'model': model, 'tokenizer': tokenizer}, merge_path)
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
sys.path.append('/data/zhangs/LLMs/Projects/SAES-SVD')
from evaluater import ppl_eval

checkpoint = torch.load('$SECOND_STAGE_DIR/merge.pt', map_location='cpu', weights_only=False)
model = checkpoint['model']
tokenizer = checkpoint['tokenizer']

model = model.float().to('cuda:0')
model.eval()

ppl_eval(model, tokenizer, datasets=['wikitext2'], 
         model_seq_len=2048, batch_size=4, device='cuda:0')
" 2>&1 | tee "$OUTPUT_BASE/final_eval.log"

# ========== 总结 ==========
echo ""
echo "=========================================="
echo "✅ LoRA Fine-tuning Complete!"
echo "=========================================="
echo ""
echo "📊 Result Summary:"
echo "  - Baseline (compressed): PPL = 8.05"
echo "  - After First-stage (U): PPL = 7.47"
echo "  - After Second-stage (V):"
grep -i "wikitext2\|ppl" "$OUTPUT_BASE/final_eval.log" | tail -3
echo ""
echo "=========================================="
