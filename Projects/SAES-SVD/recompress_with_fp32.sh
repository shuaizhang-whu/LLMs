#!/bin/bash
# ========================================
# 修复SAES-SVD精度问题后的重新压缩脚本
# ========================================
# 
# 问题: 原Exp-B5模型保存为fp16 (11GB)，导致评估PPL偏高(8.05)
# 修复: 现在保存为fp32 (20GB)，预期PPL降至7.2-7.5
#
# 执行时间: 约2-3小时
# 输出大小: 约20GB (fp32精度)
# ========================================

MODEL_PATH="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
OUTPUT_DIR="alpha_ablation_gpu1/Exp-B5-fp32_fixed"
DATASET="wikitext2"
NSAMPLES=128
CUDA_DEVICE=6

# 最优参数 (来自Exp-B5实验)
RATIO=0.2      # 保留80%参数
RHO=0.8       # 轻度收缩
ALPHA_MIN=0.4  # α下界
ALPHA_MAX=0.6  # α上界
EPS=1e-4       # 正则化参数

echo "=========================================="
echo "🔧 Recompressing with fp32 precision"
echo "=========================================="
echo "Input model: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Parameters: ratio=$RATIO, rho=$RHO, alpha=[$ALPHA_MIN,$ALPHA_MAX], eps=$EPS"
echo "Expected file size: ~20GB (fp32)"
echo "Expected PPL: 7.2-7.5 (vs 8.05 with fp16)"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行压缩 (已修复保存精度问题)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python SAES-SVD_v2.1.4.py \
    --model "$MODEL_PATH" \
    --ratio $RATIO \
    --dataset $DATASET \
    --nsamples $NSAMPLES \
    --save_path "$OUTPUT_DIR" \
    --DEV cuda:0 \
    --model_seq_len 2048 \
    --seed 0 \
    --eps $EPS \
    --rho $RHO \
    --alpha_min $ALPHA_MIN \
    --alpha_max $ALPHA_MAX

echo ""
echo "=========================================="
echo "✅ Compression complete!"
echo "=========================================="
echo ""
echo "📊 Next steps:"
echo "  1. Check model size: should be ~20GB (not 11GB)"
echo "  2. Verify baseline PPL: should be 7.2-7.5 (not 8.05)"
echo "  3. If PPL is correct, proceed with LoRA tuning:"
echo ""
echo "     # Update run_lora_finetuning.sh with new model path:"
echo "     COMPRESSED_MODEL=\"$OUTPUT_DIR/jeffwan_llama_7b_hf_saes_svd_v2.1.4_0.8.pt\""
echo ""
echo "     # Run full LoRA pipeline:"
echo "     CUDA_VISIBLE_DEVICES=0 bash run_lora_finetuning.sh"
echo ""
echo "=========================================="
