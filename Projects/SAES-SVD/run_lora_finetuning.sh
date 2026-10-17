#!/bin/bash
# ================================
# Task 4: LoRA微调Exp-B5压缩模型
# ================================
# 参考SVDLLM的两阶段LoRA微调流程
# 对Exp-B5最优权重进行精度恢复

# ========== 路径配置 ==========
COMPRESSED_MODEL="/data/zhangs/LLMs/SAES-SVD/alpha_ablation_gpu1/Exp-B5-fp32_fixed/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_saes_svd_v2.1.4_0.8.pt"
DATA_PATH="/data/zhangs/LLMs/Datasets/alpaca-cleaned"
OUTPUT_BASE="/data/zhangs/LLMs/SAES-SVD/exp_b5_lora_tuning_fp32"
CUDA_DEVICE=6

echo "=========================================="
echo "Task 4: LoRA Fine-tuning for Exp-B5"
echo "=========================================="
echo "Compressed Model: $COMPRESSED_MODEL"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_BASE"
echo "Device: CUDA:$CUDA_DEVICE"
echo "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# ========== 步骤1: 评估压缩模型基线 ==========
echo ""
echo "▶ Step 1: Evaluating compressed model baseline..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -c "
import torch
import sys
sys.path.append('/data/zhangs/LLMs/SAES-SVD')
from evaluater import ppl_eval

# 加载压缩模型
print('Loading compressed model...')
checkpoint = torch.load('$COMPRESSED_MODEL', map_location='cpu', weights_only=False)
model = checkpoint['model']
tokenizer = checkpoint['tokenizer']

model = model.float().to('cuda:0')
model.eval()

# 评估
print('Evaluating baseline...')
ppl_eval(model, tokenizer, datasets=['wikitext2'], 
         model_seq_len=2048, batch_size=4, device='cuda:0')
" 2>&1 | tee "$OUTPUT_BASE/baseline_eval.log"

# ========== 步骤2: 第一阶段LoRA微调 (更新U) ==========
echo ""
echo "▶ Step 2: First-stage LoRA tuning (updating U projections)..."
echo "Target modules: q_u_proj, k_u_proj, v_u_proj, o_u_proj, gate_u_proj, down_u_proj, up_u_proj"

FIRST_STAGE_DIR="$OUTPUT_BASE/first_half"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/LoRA.py \
    --prune_model "$COMPRESSED_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$FIRST_STAGE_DIR" \
    --lora_target_modules "q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --micro_batch_size 4 \
    --val_set_size 2000 2>&1 | tee "$OUTPUT_BASE/first_stage_training.log"

# ========== 步骤3: 合并第一阶段LoRA权重 ==========
echo ""
echo "▶ Step 3: Merging first-stage LoRA weights..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -c "
import torch
import sys
import os
sys.path.append('/data/zhangs/LLMs/SAES-SVD')

# 加载原始压缩模型
checkpoint = torch.load('$COMPRESSED_MODEL', map_location='cpu', weights_only=False)
model = checkpoint['model']
tokenizer = checkpoint['tokenizer']

# 加载LoRA adapter
from peft import PeftModel
lora_path = '$FIRST_STAGE_DIR'
print(f'Loading LoRA adapter from {lora_path}...')

# 合并LoRA权重
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

# 保存合并后的模型
merge_path = os.path.join('$FIRST_STAGE_DIR', 'merge.pt')
print(f'Saving merged model to {merge_path}...')
torch.save({'model': model, 'tokenizer': tokenizer}, merge_path)
print('✅ First-stage merge complete!')
" 2>&1 | tee "$OUTPUT_BASE/first_stage_merge.log"

# ========== 步骤4: 评估第一阶段结果 ==========
echo ""
echo "▶ Step 4: Evaluating first-stage model..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -c "
import torch
import sys
sys.path.append('/data/zhangs/LLMs/SAES-SVD')
from evaluater import ppl_eval

checkpoint = torch.load('$FIRST_STAGE_DIR/merge.pt', map_location='cpu', weights_only=False)
model = checkpoint['model']
tokenizer = checkpoint['tokenizer']

model = model.float().to('cuda:0')
model.eval()

ppl_eval(model, tokenizer, datasets=['wikitext2'], 
         model_seq_len=2048, batch_size=4, device='cuda:0')
" 2>&1 | tee "$OUTPUT_BASE/first_stage_eval.log"

# ========== 步骤5: 第二阶段LoRA微调 (更新V) ==========
echo ""
echo "▶ Step 5: Second-stage LoRA tuning (updating V projections)..."
echo "Target modules: q_v_proj, k_v_proj, v_v_proj, o_v_proj, gate_v_proj, down_v_proj, up_v_proj"

SECOND_STAGE_DIR="$OUTPUT_BASE/second_half"

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
checkpoint = torch.load('$FIRST_STAGE_DIR/merge.pt', map_location='cpu', weights_only=False)
model = checkpoint['model']
tokenizer = checkpoint['tokenizer']

# 加载LoRA adapter
from peft import PeftModel
lora_path = '$SECOND_STAGE_DIR'
print(f'Loading LoRA adapter from {lora_path}...')

# 合并LoRA权重
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

# 保存最终模型
merge_path = os.path.join('$SECOND_STAGE_DIR', 'merge.pt')
print(f'Saving final merged model to {merge_path}...')
torch.save({'model': model, 'tokenizer': tokenizer}, merge_path)
print('✅ Second-stage merge complete!')
" 2>&1 | tee "$OUTPUT_BASE/second_stage_merge.log"

# ========== 步骤7: 评估最终模型 ==========
echo ""
echo "▶ Step 7: Evaluating final fine-tuned model..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -c "
import torch
import sys
sys.path.append('/data/zhangs/LLMs/SAES-SVD')
from evaluater import ppl_eval

checkpoint = torch.load('$SECOND_STAGE_DIR/merge.pt', map_location='cpu', weights_only=False)
model = checkpoint['model']
tokenizer = checkpoint['tokenizer']

model = model.float().to('cuda:0')
model.eval()

ppl_eval(model, tokenizer, datasets=['wikitext2'], 
         model_seq_len=2048, batch_size=4, device='cuda:0')
" 2>&1 | tee "$OUTPUT_BASE/final_eval.log"

# ========== 步骤8: 生成总结报告 ==========
echo ""
echo "=========================================="
echo "✅ LoRA Fine-tuning Complete!"
echo "=========================================="
echo ""
echo "📊 Result Summary:"
echo ""
echo "Baseline (Compressed only):"
grep "wikitext2" "$OUTPUT_BASE/baseline_eval.log" | tail -1
echo ""
echo "After First-stage LoRA (U projections):"
grep "wikitext2" "$OUTPUT_BASE/first_stage_eval.log" | tail -1
echo ""
echo "After Second-stage LoRA (V projections):"
grep "wikitext2" "$OUTPUT_BASE/final_eval.log" | tail -1
echo ""
echo "=========================================="
echo "Output files:"
echo "  - Baseline: $OUTPUT_BASE/baseline_eval.log"
echo "  - First stage: $FIRST_STAGE_DIR/"
echo "  - Second stage: $SECOND_STAGE_DIR/"
echo "  - Final model: $SECOND_STAGE_DIR/merge.pt"
echo "=========================================="
echo ""
echo "预期效果："
echo "  - Baseline (仅压缩): PPL ≈ 8.05"
echo "  - First LoRA: PPL ≈ 7.5-7.8"
echo "  - Second LoRA: PPL ≈ 7.2-7.5"
echo "  - 论文目标: PPL ≈ 5.7-6.0 (需要更多优化)"
echo "=========================================="
