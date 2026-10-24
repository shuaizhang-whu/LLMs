#!/bin/bash

################################################################################
# SAES-SVD v2.1.3 - eps×NSAMPLES参数消融实验
# 固定: rho=0.9 (当前最优)
# 变化: eps ∈ {1e-3, 5e-4, 1e-4}, NSAMPLES ∈ {128, 256}
################################################################################

cd "$(dirname "$0")"


# 设置使用GPU 1（避免与rho实验冲突）
export CUDA_VISIBLE_DEVICES=1
echo "🎮 使用GPU: 1 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="eps_nsamples_ablation_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  SAES-SVD v2.1.3 - eps×NSAMPLES消融实验"
echo "============================================================"
echo "固定参数: rho=0.9"
echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

TOTAL_START=$(date +%s)

# 固定参数
FIXED_RHO="0.9"
SEED=3

# 实验配置
experiments=(
    "Exp-E1:1e-3:128"
    "Exp-E2:1e-4:256"
    "Exp-E3:5e-4:128"
)

# 运行每个实验
exp_count=0
for exp_config in "${experiments[@]}"; do
    IFS=':' read -r EXP_ID EPS_VAL NS_VAL <<< "$exp_config"
    exp_count=$((exp_count + 1))
    
    echo ""
    echo "============================================================"
    echo "🚀 实验 [$exp_count/3]: $EXP_ID"
    echo "============================================================"
    echo "  eps=$EPS_VAL, rho=$FIXED_RHO, NSAMPLES=$NS_VAL"
    
    LOG_FILE="${OUTPUT_DIR}/${EXP_ID}_eps${EPS_VAL}_ns${NS_VAL}.log"
    EXP_START=$(date +%s)
    
    # 修改参数
    sed -i "s/^EPS=.*/EPS=\"${EPS_VAL}\"/" saes_svd_v2.1.3.sh
    sed -i "s/^RHO=.*/RHO=\"${FIXED_RHO}\"/" saes_svd_v2.1.3.sh
    sed -i "s/^NSAMPLES=.*/NSAMPLES=\"${NS_VAL}\"/" saes_svd_v2.1.3.sh
    
    echo "  ✓ 参数已设置"
    echo "  ⏳ 开始压缩..."
    
    # 运行实验
    {
        echo "[$EXP_ID] $(date '+%Y-%m-%d %H:%M:%S')"
        bash saes_svd_v2.1.3.sh 0.2
        echo "[$EXP_ID] 完成: $(date '+%Y-%m-%d %H:%M:%S')"
    } > "$LOG_FILE" 2>&1
    
    EXP_END=$(date +%s)
    EXP_MINUTES=$(((EXP_END - EXP_START) / 60))
    
    echo "  ✅ 完成！耗时: ${EXP_MINUTES}分钟"
    
    # 提取结果
    LATEST_LOG=$(find ./result/SAES_SVD_v2.1.3/ -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -f "$LATEST_LOG" ]; then
        PPL=$(grep "PPL after pruning:" "$LATEST_LOG" | tail -1 | grep -oP "'wikitext2':\s*\K[\d.]+")
        AVG_BETA=$(grep "Average β" "$LATEST_LOG" | tail -1 | grep -oP ":\s*\K[\d.]+")
        
        echo "  📊 PPL=$PPL, β=$AVG_BETA"
        echo "$EXP_ID,$EPS_VAL,$FIXED_RHO,$NS_VAL,$PPL,$AVG_BETA,${EXP_MINUTES}m" >> "${OUTPUT_DIR}/results.csv"
    fi
    
    [ "$exp_count" -lt 3 ] && sleep 10
done

# 总结
TOTAL_END=$(date +%s)
TOTAL_HOURS=$(((TOTAL_END - TOTAL_START) / 3600))
TOTAL_MINUTES=$((((TOTAL_END - TOTAL_START) % 3600) / 60))

echo ""
echo "============================================================"
echo "🎉 所有实验完成！"
echo "============================================================"
echo "总耗时: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo ""
echo "结果汇总:"
[ -f "${OUTPUT_DIR}/results.csv" ] && cat "${OUTPUT_DIR}/results.csv"
echo ""
echo "对比基线: Exp2 (eps=1e-4, rho=0.9, NS=128) → PPL=8.169"
echo "============================================================"
