#!/bin/bash

################################################################################
# SAES-SVD v2.1.3 - rho参数细粒度消融实验
# 当前最优: eps=1e-4, rho=0.9, NSAMPLES=128 → PPL=8.169
# 本脚本将依次测试 rho = 0.88, 0.85, 0.8
################################################################################

cd "$(dirname "$0")"

# 设置使用GPU 0
export CUDA_VISIBLE_DEVICES=0
echo "🎮 使用GPU: 0 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="rho_ablation_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  SAES-SVD v2.1.3 - rho参数消融实验"
echo "============================================================"
echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

TOTAL_START=$(date +%s)

# 定义实验
declare -A experiments
experiments=(
    ["Exp-A1"]="0.88"
    ["Exp-A2"]="0.85"
    ["Exp-A3"]="0.8"
)

# 运行每个实验
for exp_id in "Exp-A1" "Exp-A2" "Exp-A3"; do
    rho_value="${experiments[$exp_id]}"
    
    echo ""
    echo "============================================================"
    echo "🚀 开始实验: $exp_id (rho=$rho_value)"
    echo "============================================================"
    
    LOG_FILE="${OUTPUT_DIR}/${exp_id}_rho${rho_value}.log"
    EXP_START=$(date +%s)
    
    # 修改saes_svd_v2.1.3.sh中的RHO值
    sed -i "s/^RHO=.*/RHO=\"${rho_value}\"/" saes_svd_v2.1.3.sh
    
    # 确认修改成功
    echo "  参数配置:"
    grep "^RHO=" saes_svd_v2.1.3.sh | sed 's/^/    /'
    grep "^EPS=" saes_svd_v2.1.3.sh | sed 's/^/    /'
    grep "^NSAMPLES=" saes_svd_v2.1.3.sh | sed 's/^/    /'
    echo "------------------------------------------------------------"
    
    # 运行实验
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始实验 $exp_id (rho=$rho_value)..." | tee "$LOG_FILE"
    
    if bash saes_svd_v2.1.3.sh 0.2 >> "$LOG_FILE" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 实验完成！" | tee -a "$LOG_FILE"
        EXP_STATUS="success"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 实验失败！" | tee -a "$LOG_FILE"
        EXP_STATUS="failed"
    fi
    
    EXP_END=$(date +%s)
    EXP_DURATION=$((EXP_END - EXP_START))
    EXP_MINUTES=$((EXP_DURATION / 60))
    EXP_SECONDS=$((EXP_DURATION % 60))
    
    echo "✅ $exp_id 完成！耗时: ${EXP_MINUTES}m ${EXP_SECONDS}s (状态: $EXP_STATUS)"
    
    # 只在成功时提取结果
    if [ "$EXP_STATUS" = "success" ]; then
        LATEST_LOG=$(find ./result/SAES_SVD_v2.1.3/ -name "saes_svd_v2.1.3_ratio0.2_*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        
        if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
            PPL=$(grep "PPL after pruning:" "$LATEST_LOG" | tail -1 | grep -oP "'wikitext2':\s*\K[\d.]+" || echo "N/A")
            AVG_BETA=$(grep "Average β across all layers:" "$LATEST_LOG" | tail -1 | grep -oP ":\s*\K[\d.]+" || echo "N/A")
            AVG_DELTA=$(grep "Average Δ/H ratio:" "$LATEST_LOG" | tail -1 | grep -oP ":\s*\K[\d.]+" || echo "N/A")
            
            echo "  📈 PPL: $PPL"
            echo "  🎯 avg_β: $AVG_BETA"
            echo "  📊 avg_Δ/H: $AVG_DELTA"
            
            echo "$exp_id,rho=$rho_value,PPL=$PPL,β=$AVG_BETA,Δ/H=$AVG_DELTA,time=${EXP_MINUTES}m" >> "${OUTPUT_DIR}/results.csv"
        else
            echo "  ⚠️  未找到结果日志"
            echo "$exp_id,rho=$rho_value,PPL=N/A,β=N/A,Δ/H=N/A,time=${EXP_MINUTES}m,status=no_log" >> "${OUTPUT_DIR}/results.csv"
        fi
    else
        echo "  ❌ 实验失败，跳过结果提取"
        echo "$exp_id,rho=$rho_value,PPL=FAILED,β=N/A,Δ/H=N/A,time=${EXP_MINUTES}m,status=failed" >> "${OUTPUT_DIR}/results.csv"
    fi
    
    # 等待10秒
    if [ "$exp_id" != "Exp-A3" ]; then
        echo "⏳ 等待10秒..."
        sleep 10
    fi
done

# 总结
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "============================================================"
echo "🎉 所有实验完成！"
echo "============================================================"
echo "总耗时: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo ""
echo "结果汇总:"
cat "${OUTPUT_DIR}/results.csv"
echo ""
echo "对比基线: Exp2 (rho=0.9) → PPL=8.169"
echo "============================================================"
