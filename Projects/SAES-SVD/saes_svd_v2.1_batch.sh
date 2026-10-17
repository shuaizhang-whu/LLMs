#!/bin/bash
# SAES-SVD 运行脚本（逐层闭环版本 - 多比率批量测试）

# ==================== 配置参数 ====================
GPU_DEVICE="1"
MODEL_PATH="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"

# ★ 压缩比例数组（可自定义多个比率）
# 格式: RATIOS=("0.2" "0.4" "0.6")
# 0.2 = 移除20%保留80%, 0.4 = 移除40%保留60%, 0.6 = 移除60%保留40%
RATIOS=("0.2" "0.4" "0.6")

DATASET="wikitext2"
NSAMPLES="128"
SEED="0"
EPS="1e-4"               # 正则化参数
MODEL_SEQ_LEN="2048"

# 输出路径
OUTPUT_BASE="./result/SAES_SVD_v2.1"
MODEL_NAME=$(basename $MODEL_PATH)
GLOBAL_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${OUTPUT_BASE}/${MODEL_NAME}/batch_summary_${GLOBAL_TIMESTAMP}.txt"

# ==================== 开始运行 ====================
echo "=========================================="
echo "SAES-SVD v2.1 批量压缩测试"
echo "=========================================="
echo "GPU 设备: $GPU_DEVICE"
echo "模型路径: $MODEL_PATH"
echo "压缩比例: ${RATIOS[@]}"
echo "数据集: $DATASET"
echo "样本数: $NSAMPLES"
echo "正则化参数 eps: $EPS"
echo "输出目录: $OUTPUT_BASE/${MODEL_NAME}/"
echo "批量摘要: $SUMMARY_FILE"
echo "=========================================="
echo ""

# 检查 Python 脚本是否存在
if [ ! -f "SAES_SVD_v2.1.py" ]; then
    if [ ! -f "SAES-SVD_v2.1.py" ]; then
        echo "❌ 错误: SAES_SVD_v2.1.py 或 SAES-SVD_v2.1.py 不存在！"
        exit 1
    else
        SCRIPT_NAME="SAES-SVD_v2.1.py"
    fi
else
    SCRIPT_NAME="SAES_SVD_v2.1.py"
fi

echo "使用脚本: $SCRIPT_NAME"
echo ""

# 创建批量摘要文件
mkdir -p "$(dirname "$SUMMARY_FILE")"
echo "=========================================="  > "$SUMMARY_FILE"
echo "SAES-SVD v2.1 批量压缩测试摘要"               >> "$SUMMARY_FILE"
echo "=========================================="  >> "$SUMMARY_FILE"
echo "开始时间: $(date)"                          >> "$SUMMARY_FILE"
echo "模型: $MODEL_PATH"                          >> "$SUMMARY_FILE"
echo "测试比率: ${RATIOS[@]}"                     >> "$SUMMARY_FILE"
echo "数据集: $DATASET (样本数: $NSAMPLES)"      >> "$SUMMARY_FILE"
echo "=========================================="  >> "$SUMMARY_FILE"
echo ""                                            >> "$SUMMARY_FILE"

# 用于收集所有结果
declare -A ALL_RESULTS

# ==================== 循环执行不同比率 ====================
TOTAL_RATIOS=${#RATIOS[@]}
CURRENT_NUM=0

for RATIO in "${RATIOS[@]}"; do
    CURRENT_NUM=$((CURRENT_NUM + 1))
    
    echo ""
    echo "=========================================="
    echo "[$CURRENT_NUM/$TOTAL_RATIOS] 开始测试 ratio=$RATIO"
    echo "=========================================="
    
    # 计算保留比例
    KEEP_PERCENT=$(echo "scale=1; (1-$RATIO)*100" | bc)
    
    # 为当前比率创建输出目录
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}/ratio_${RATIO}/dataset_${DATASET}_samples_${NSAMPLES}_seed_${SEED}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${OUTPUT_DIR}/logs/saes_svd_v2_${TIMESTAMP}.log"
    
    echo "压缩比例: $RATIO (保留 ${KEEP_PERCENT}%)"
    echo "输出目录: $OUTPUT_DIR"
    echo "日志文件: $LOG_FILE"
    echo ""
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR/logs"
    mkdir -p "$OUTPUT_DIR/compressed_model"
    
    # 执行压缩
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始压缩 ratio=$RATIO..." | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    START_TIME=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python $SCRIPT_NAME \
        --model "$MODEL_PATH" \
        --ratio "$RATIO" \
        --dataset "$DATASET" \
        --nsamples "$NSAMPLES" \
        --seed "$SEED" \
        --DEV cuda:0 \
        --eps "$EPS" \
        --model_seq_len "$MODEL_SEQ_LEN" \
        --save_path "${OUTPUT_DIR}/compressed_model" \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    
    # 检查执行结果
    if [ $EXIT_CODE -eq 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ ratio=$RATIO 压缩完成！耗时: ${ELAPSED_MIN}m ${ELAPSED_SEC}s" | tee -a "$LOG_FILE"
        
        # 提取关键结果
        echo "" | tee -a "$LOG_FILE"
        echo "📊 提取关键结果..." | tee -a "$LOG_FILE"
        
        # 提取 Beta 统计
        AVG_BETA=$(grep "Average β across all layers:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        # 提取 Δ/H 统计
        AVG_DELTA_H=$(grep "Average Δ/H ratio:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        # 提取 PPL
        PPL=$(grep -E "wikitext2.*:" "$LOG_FILE" | grep -v "Running" | tail -1 | awk -F': ' '{print $2}' | awk '{print $1}' | tr -d '}')
        # 提取 Weight Memory
        WEIGHT_MEM=$(grep "Weight Memory:" "$LOG_FILE" | tail -1 | awk '{print $(NF-1)}')
        
        # 保存到关联数组
        ALL_RESULTS["${RATIO}_beta"]="$AVG_BETA"
        ALL_RESULTS["${RATIO}_delta_h"]="$AVG_DELTA_H"
        ALL_RESULTS["${RATIO}_ppl"]="$PPL"
        ALL_RESULTS["${RATIO}_mem"]="$WEIGHT_MEM"
        ALL_RESULTS["${RATIO}_time"]="${ELAPSED_MIN}m${ELAPSED_SEC}s"
        ALL_RESULTS["${RATIO}_status"]="✅"
        
        # 生成当前比率的摘要
        echo "==========================================" | tee -a "${LOG_FILE}.summary.txt"
        echo "Ratio $RATIO 运行摘要" | tee -a "${LOG_FILE}.summary.txt"
        echo "==========================================" | tee -a "${LOG_FILE}.summary.txt"
        echo "完成时间: $(date)" | tee -a "${LOG_FILE}.summary.txt"
        echo "耗时: ${ELAPSED_MIN}m ${ELAPSED_SEC}s" | tee -a "${LOG_FILE}.summary.txt"
        echo "" | tee -a "${LOG_FILE}.summary.txt"
        echo "📊 关键指标:" | tee -a "${LOG_FILE}.summary.txt"
        echo "  - Average Beta: ${AVG_BETA}" | tee -a "${LOG_FILE}.summary.txt"
        echo "  - Average Δ/H: ${AVG_DELTA_H}" | tee -a "${LOG_FILE}.summary.txt"
        echo "  - PPL (wikitext2): ${PPL}" | tee -a "${LOG_FILE}.summary.txt"
        echo "  - Weight Memory: ${WEIGHT_MEM} MiB" | tee -a "${LOG_FILE}.summary.txt"
        echo "" | tee -a "${LOG_FILE}.summary.txt"
        
    else
        echo "" | tee -a "$LOG_FILE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ ratio=$RATIO 压缩失败！" | tee -a "$LOG_FILE"
        
        ALL_RESULTS["${RATIO}_status"]="❌"
        ALL_RESULTS["${RATIO}_time"]="${ELAPSED_MIN}m${ELAPSED_SEC}s"
        
        echo "⚠️ ratio=$RATIO 执行失败，继续下一个比率..."
    fi
    
    echo ""
    echo "[$CURRENT_NUM/$TOTAL_RATIOS] ratio=$RATIO 完成"
    echo ""
    
    # 短暂休息，清理GPU内存
    sleep 5
done

# ==================== 生成批量摘要 ====================
echo ""
echo "=========================================="
echo "生成批量测试摘要"
echo "=========================================="

echo ""                                                          >> "$SUMMARY_FILE"
echo "=========================================="                >> "$SUMMARY_FILE"
echo "各比率测试结果对比"                                        >> "$SUMMARY_FILE"
echo "=========================================="                >> "$SUMMARY_FILE"
printf "%-8s | %-8s | %-10s | %-10s | %-10s | %-12s | %-8s\n" \
    "Ratio" "Status" "Avg Beta" "Avg Δ/H" "PPL" "Memory(MiB)" "Time" >> "$SUMMARY_FILE"
echo "-----------------------------------------------------------------------------------------" >> "$SUMMARY_FILE"

for RATIO in "${RATIOS[@]}"; do
    STATUS="${ALL_RESULTS[${RATIO}_status]:-N/A}"
    BETA="${ALL_RESULTS[${RATIO}_beta]:-N/A}"
    DELTA_H="${ALL_RESULTS[${RATIO}_delta_h]:-N/A}"
    PPL="${ALL_RESULTS[${RATIO}_ppl]:-N/A}"
    MEM="${ALL_RESULTS[${RATIO}_mem]:-N/A}"
    TIME="${ALL_RESULTS[${RATIO}_time]:-N/A}"
    
    printf "%-8s | %-8s | %-10s | %-10s | %-10s | %-12s | %-8s\n" \
        "$RATIO" "$STATUS" "$BETA" "$DELTA_H" "$PPL" "$MEM" "$TIME" >> "$SUMMARY_FILE"
done

echo ""                                                          >> "$SUMMARY_FILE"
echo "=========================================="                >> "$SUMMARY_FILE"
echo "结束时间: $(date)"                                         >> "$SUMMARY_FILE"
echo "=========================================="                >> "$SUMMARY_FILE"
echo ""                                                          >> "$SUMMARY_FILE"
echo "📁 详细日志位置:"                                          >> "$SUMMARY_FILE"
for RATIO in "${RATIOS[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}/ratio_${RATIO}/dataset_${DATASET}_samples_${NSAMPLES}_seed_${SEED}"
    echo "  - Ratio $RATIO: ${OUTPUT_DIR}/logs/" >> "$SUMMARY_FILE"
done
echo ""                                                          >> "$SUMMARY_FILE"

# 显示摘要
echo ""
echo "=========================================="
echo "✅ 批量测试完成！"
echo "=========================================="
echo ""
cat "$SUMMARY_FILE"
echo ""
echo "📝 批量摘要已保存到: $SUMMARY_FILE"
echo ""
