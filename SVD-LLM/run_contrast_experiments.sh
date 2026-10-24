#!/bin/bash
# ================================
# Task 2: 修正参数对照实验
# ================================
# 基于Exp-B5最优配置，测试v2.1.4修正后的效果
# 对比: v2.1.3(rho=0.8) vs v2.1.4(rho=0.95)

MODEL_PATH="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
OUTPUT_DIR="v2.1.4_contrast_experiments"
DATE_STAMP=$(date +"%Y%m%d")

echo "=========================================="
echo "Task 2: Parameter Correction Experiments"
echo "=========================================="
echo "Testing v2.1.4 parameter optimizations"
echo "Base config: Exp-B5 (rho=0.8, eps=1e-4, alpha=[0.4,0.6])"
echo "Modified: rho=0.95, beta_cap=0.375"
echo "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 初始化CSV结果文件
CSV_FILE="$OUTPUT_DIR/results.csv"
echo "ExpID,Version,rho,eps,alpha_range,PPL,avg_beta,avg_delta,time_min,status" > "$CSV_FILE"

# ========== 实验1: v2.1.3基线重现 (Exp-B5配置) ==========
EXP_ID="Contrast-A"
echo ""
echo "▶ Running $EXP_ID: v2.1.3 baseline (rho=0.8, alpha=[0.4,0.6])"
LOG_FILE="$OUTPUT_DIR/${EXP_ID}_${DATE_STAMP}.log"
RESULT_DIR="$OUTPUT_DIR/${EXP_ID}_v2.1.3_rho0.8"

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python SAES-SVD_v2.1.3.py \
    --model "$MODEL_PATH" \
    --ratio 0.2 \
    --dataset wikitext2 \
    --nsamples 128 \
    --seed 3 \
    --model_seq_len 2048 \
    --eps 1e-4 \
    --rho 0.8 \
    --alpha_min 0.4 \
    --alpha_max 0.6 \
    --result_dir "$RESULT_DIR" \
    --DEV cuda:0 2>&1 | tee "$LOG_FILE"

END_TIME=$(date +%s)
DURATION=$(( (END_TIME - START_TIME) / 60 ))

# 提取结果
if grep -q "Average β across all layers" "$LOG_FILE"; then
    AVG_BETA=$(grep "Average β across all layers" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    AVG_DELTA=$(grep "Average Δ/H ratio" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    PPL=$(grep "wikitext2" "$LOG_FILE" | grep -oP "(?<=')[0-9.]+(?=')" | tail -1)
    echo "$EXP_ID,v2.1.3,0.8,1e-4,[0.4-0.6],$PPL,$AVG_BETA,$AVG_DELTA,${DURATION}m,success" >> "$CSV_FILE"
else
    echo "$EXP_ID,v2.1.3,0.8,1e-4,[0.4-0.6],FAILED,N/A,N/A,${DURATION}m,failed" >> "$CSV_FILE"
fi

# ========== 实验2: v2.1.4修正版 (rho=0.95) ==========
EXP_ID="Contrast-B"
echo ""
echo "▶ Running $EXP_ID: v2.1.4 optimized (rho=0.95, alpha=[0.4,0.6])"
LOG_FILE="$OUTPUT_DIR/${EXP_ID}_${DATE_STAMP}.log"
RESULT_DIR="$OUTPUT_DIR/${EXP_ID}_v2.1.4_rho0.95"

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python SAES-SVD_v2.1.4.py \
    --model "$MODEL_PATH" \
    --ratio 0.2 \
    --dataset wikitext2 \
    --nsamples 128 \
    --seed 3 \
    --model_seq_len 2048 \
    --eps 1e-4 \
    --rho 0.95 \
    --alpha_min 0.4 \
    --alpha_max 0.6 \
    --result_dir "$RESULT_DIR" \
    --DEV cuda:0 2>&1 | tee "$LOG_FILE"

END_TIME=$(date +%s)
DURATION=$(( (END_TIME - START_TIME) / 60 ))

# 提取结果
if grep -q "Average β across all layers" "$LOG_FILE"; then
    AVG_BETA=$(grep "Average β across all layers" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    AVG_DELTA=$(grep "Average Δ/H ratio" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    PPL=$(grep "wikitext2" "$LOG_FILE" | grep -oP "(?<=')[0-9.]+(?=')" | tail -1)
    echo "$EXP_ID,v2.1.4,0.95,1e-4,[0.4-0.6],$PPL,$AVG_BETA,$AVG_DELTA,${DURATION}m,success" >> "$CSV_FILE"
else
    echo "$EXP_ID,v2.1.4,0.95,1e-4,[0.4-0.6],FAILED,N/A,N/A,${DURATION}m,failed" >> "$CSV_FILE"
fi

# ========== 实验3: v2.1.4 + rho=1.0 (无收缩) ==========
EXP_ID="Contrast-C"
echo ""
echo "▶ Running $EXP_ID: v2.1.4 no shrinkage (rho=1.0, alpha=[0.4,0.6])"
LOG_FILE="$OUTPUT_DIR/${EXP_ID}_${DATE_STAMP}.log"
RESULT_DIR="$OUTPUT_DIR/${EXP_ID}_v2.1.4_rho1.0"

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python SAES-SVD_v2.1.4.py \
    --model "$MODEL_PATH" \
    --ratio 0.2 \
    --dataset wikitext2 \
    --nsamples 128 \
    --seed 3 \
    --model_seq_len 2048 \
    --eps 1e-4 \
    --rho 1.0 \
    --alpha_min 0.4 \
    --alpha_max 0.6 \
    --result_dir "$RESULT_DIR" \
    --DEV cuda:0 2>&1 | tee "$LOG_FILE"

END_TIME=$(date +%s)
DURATION=$(( (END_TIME - START_TIME) / 60 ))

# 提取结果
if grep -q "Average β across all layers" "$LOG_FILE"; then
    AVG_BETA=$(grep "Average β across all layers" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    AVG_DELTA=$(grep "Average Δ/H ratio" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    PPL=$(grep "wikitext2" "$LOG_FILE" | grep -oP "(?<=')[0-9.]+(?=')" | tail -1)
    echo "$EXP_ID,v2.1.4,1.0,1e-4,[0.4-0.6],$PPL,$AVG_BETA,$AVG_DELTA,${DURATION}m,success" >> "$CSV_FILE"
else
    echo "$EXP_ID,v2.1.4,1.0,1e-4,[0.4-0.6],FAILED,N/A,N/A,${DURATION}m,failed" >> "$CSV_FILE"
fi

# ========== 总结 ==========
echo ""
echo "=========================================="
echo "✅ 对照实验完成！"
echo "=========================================="
echo "结果文件: $CSV_FILE"
cat "$CSV_FILE"
echo ""
echo "预期结果："
echo "  - Contrast-A (v2.1.3, rho=0.8): avg_beta≈0.287 (过低)"
echo "  - Contrast-B (v2.1.4, rho=0.95): avg_beta≈0.33-0.35 (理想)"
echo "  - Contrast-C (v2.1.4, rho=1.0): avg_beta≈0.36-0.37 (可能过高)"
echo "=========================================="
