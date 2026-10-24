#!/bin/bash
# ================================
# Task 3: NSAMPLES消融实验
# ================================
# 基于Task 2的最优配置，测试不同校准样本数的影响
# NSAMPLES: 64, 128, 256, 512, 1024

MODEL_PATH="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
OUTPUT_DIR="nsamples_ablation"
DATE_STAMP=$(date +"%Y%m%d")

# ⚠️ 这里需要根据Task 2结果手动设置最优配置
# 默认使用 v2.1.4 + rho=0.95 + alpha=[0.4,0.6]
BEST_VERSION="SAES-SVD_v2.1.4.py"
BEST_RHO=0.95
BEST_EPS="1e-4"
BEST_ALPHA_MIN=0.4
BEST_ALPHA_MAX=0.6

echo "=========================================="
echo "Task 3: NSAMPLES Ablation Experiments"
echo "=========================================="
echo "Testing calibration sample sizes: 64, 128, 256, 512, 1024"
echo "Best config from Task 2:"
echo "  Version: $BEST_VERSION"
echo "  rho: $BEST_RHO"
echo "  eps: $BEST_EPS"
echo "  alpha: [$BEST_ALPHA_MIN, $BEST_ALPHA_MAX]"
echo "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 初始化CSV结果文件
CSV_FILE="$OUTPUT_DIR/results.csv"
echo "ExpID,NSAMPLES,rho,eps,alpha_range,PPL,avg_beta,avg_delta,time_min,status" > "$CSV_FILE"

# 样本数数组
NSAMPLES_ARRAY=(64 128 256 512 1024)

# 遍历每个NSAMPLES
for NSAMPLES in "${NSAMPLES_ARRAY[@]}"; do
    EXP_ID="NS-${NSAMPLES}"
    echo ""
    echo "=========================================="
    echo "▶ Running $EXP_ID: NSAMPLES=$NSAMPLES"
    echo "=========================================="
    
    LOG_FILE="$OUTPUT_DIR/${EXP_ID}_${DATE_STAMP}.log"
    RESULT_DIR="$OUTPUT_DIR/${EXP_ID}_nsamples${NSAMPLES}"
    
    START_TIME=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=0 python "$BEST_VERSION" \
        --model "$MODEL_PATH" \
        --ratio 0.2 \
        --dataset wikitext2 \
        --nsamples "$NSAMPLES" \
        --seed 3 \
        --model_seq_len 2048 \
        --eps "$BEST_EPS" \
        --rho "$BEST_RHO" \
        --alpha_min "$BEST_ALPHA_MIN" \
        --alpha_max "$BEST_ALPHA_MAX" \
        --result_dir "$RESULT_DIR" \
        --DEV cuda:0 2>&1 | tee "$LOG_FILE"
    
    END_TIME=$(date +%s)
    DURATION=$(( (END_TIME - START_TIME) / 60 ))
    
    # 提取结果
    if grep -q "Average β across all layers" "$LOG_FILE"; then
        AVG_BETA=$(grep "Average β across all layers" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        AVG_DELTA=$(grep "Average Δ/H ratio" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        PPL=$(grep "wikitext2" "$LOG_FILE" | grep -oP "(?<=')[0-9.]+(?=')" | tail -1)
        echo "$EXP_ID,$NSAMPLES,$BEST_RHO,$BEST_EPS,[$BEST_ALPHA_MIN-$BEST_ALPHA_MAX],$PPL,$AVG_BETA,$AVG_DELTA,${DURATION}m,success" >> "$CSV_FILE"
        echo "✅ $EXP_ID completed: PPL=$PPL, avg_beta=$AVG_BETA, time=${DURATION}m"
    else
        echo "$EXP_ID,$NSAMPLES,$BEST_RHO,$BEST_EPS,[$BEST_ALPHA_MIN-$BEST_ALPHA_MAX],FAILED,N/A,N/A,${DURATION}m,failed" >> "$CSV_FILE"
        echo "❌ $EXP_ID failed!"
    fi
done

# ========== 总结 ==========
echo ""
echo "=========================================="
echo "✅ NSAMPLES消融实验完成！"
echo "=========================================="
echo "结果文件: $CSV_FILE"
cat "$CSV_FILE"
echo ""
echo "分析方向："
echo "  1. PPL vs NSAMPLES 曲线 (期望：NSAMPLES↑ → PPL↓)"
echo "  2. avg_beta vs NSAMPLES (检查β稳定性)"
echo "  3. 时间成本 vs 性能收益 (找到性价比最优点)"
echo "  4. 对比论文可能使用的NSAMPLES (推测256-512)"
echo "=========================================="
