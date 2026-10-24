#!/bin/bash

################################################################################
# SAES-SVD v2.1.3 - Alpha区间消融实验 (GPU 1)
# 实验范围: 配置B后2组 + 配置C全部5组 = 7组实验
################################################################################

cd "$(dirname "$0")"

# 设置使用GPU 1
export CUDA_VISIBLE_DEVICES=1
echo "🎮 使用GPU: 1 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

OUTPUT_DIR="alpha_ablation_gpu1"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  SAES-SVD v2.1.3 - Alpha区间消融实验 (GPU 1)"
echo "============================================================"
echo "实验数量: 7组"
echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

TOTAL_START=$(date +%s)

# 固定参数
MODEL_PATH="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
NSAMPLES=128
SEED=3
RATIO=0.2
DATE_STAMP=$(date +"%Y%m%d")

# 实验配置: ExpID:rho:eps:alpha_min:alpha_max
experiments=(
    # 配置B: rho=0.8, eps=1e-4 (后2组)
    "Exp-B4:0.8:1e-4:0.3:0.7"
    "Exp-B5:0.8:1e-4:0.4:0.6"
    
    # 配置C: rho=0.9, eps=1e-3 (5组)
    "Exp-C1:0.9:1e-3:0.0:1.0"
    "Exp-C2:0.9:1e-3:0.1:0.9"
    "Exp-C3:0.9:1e-3:0.2:0.8"
    "Exp-C4:0.9:1e-3:0.3:0.7"
    "Exp-C5:0.9:1e-3:0.4:0.6"
)

# 结果CSV初始化
RESULT_CSV="${OUTPUT_DIR}/results.csv"
echo "ExpID,Config,rho,eps,alpha_range,PPL,avg_beta,avg_delta,time_min,status" > "$RESULT_CSV"

# 运行每个实验
exp_count=0
for exp_config in "${experiments[@]}"; do
    IFS=':' read -r EXP_ID RHO_VAL EPS_VAL ALPHA_MIN ALPHA_MAX <<< "$exp_config"
    exp_count=$((exp_count + 1))
    
    # 确定配置组
    if [[ "$EXP_ID" == Exp-B* ]]; then
        CONFIG_NAME="ConfigB(ρ=0.8,ε=1e-4)"
    else
        CONFIG_NAME="ConfigC(ρ=0.9,ε=1e-3)"
    fi
    
    echo ""
    echo "============================================================"
    echo "🚀 实验 [$exp_count/7]: $EXP_ID"
    echo "============================================================"
    echo "  配置: $CONFIG_NAME"
    echo "  rho=$RHO_VAL, eps=$EPS_VAL"
    echo "  α区间: [$ALPHA_MIN, $ALPHA_MAX]"
    echo "  NSAMPLES=$NSAMPLES"
    
    # 为每个实验设置独立的结果目录（包含权重文件）
    EXP_RESULT_DIR="${OUTPUT_DIR}/${EXP_ID}_rho${RHO_VAL}_eps${EPS_VAL}_alpha${ALPHA_MIN}-${ALPHA_MAX}"
    mkdir -p "$EXP_RESULT_DIR"
    
    LOG_FILE="${OUTPUT_DIR}/${EXP_ID}_rho${RHO_VAL}_eps${EPS_VAL}_alpha${ALPHA_MIN}-${ALPHA_MAX}_${DATE_STAMP}.log"
    EXP_START=$(date +%s)
    
    echo "  ⏳ 开始压缩..."
    echo "  📁 结果目录: $EXP_RESULT_DIR"
    
    # 运行实验，使用--result_dir指定独立的保存目录
    if conda run -n compress python SAES-SVD_v2.1.3.py \
        --model "$MODEL_PATH" \
        --ratio "$RATIO" \
        --dataset wikitext2 \
        --nsamples "$NSAMPLES" \
        --seed "$SEED" \
        --DEV cuda:0 \
        --eps "$EPS_VAL" \
        --rho "$RHO_VAL" \
        --alpha_min "$ALPHA_MIN" \
        --alpha_max "$ALPHA_MAX" \
        --model_seq_len 2048 \
        --result_dir "$EXP_RESULT_DIR" \
        > "$LOG_FILE" 2>&1; then
        
        EXP_STATUS="success"
        echo "  ✅ 实验完成！"
    else
        EXP_STATUS="failed"
        echo "  ❌ 实验失败！"
    fi
    
    EXP_END=$(date +%s)
    EXP_MINUTES=$(((EXP_END - EXP_START) / 60))
    
    # 提取结果
    if [ "$EXP_STATUS" = "success" ]; then
        PPL=$(grep "'wikitext2':" "$LOG_FILE" | tail -1 | grep -oP ":\s*\K[\d.]+" || echo "N/A")
        AVG_BETA=$(grep "Average β across all layers:" "$LOG_FILE" | tail -1 | grep -oP ":\s*\K[\d.]+" || echo "N/A")
        AVG_DELTA=$(grep "Average Δ/H ratio:" "$LOG_FILE" | tail -1 | grep -oP ":\s*\K[\d.]+" || echo "N/A")
        
        echo "  📊 PPL=$PPL, β=$AVG_BETA, Δ/H=$AVG_DELTA"
        echo "$EXP_ID,$CONFIG_NAME,$RHO_VAL,$EPS_VAL,[$ALPHA_MIN-$ALPHA_MAX],$PPL,$AVG_BETA,$AVG_DELTA,${EXP_MINUTES}m,$EXP_STATUS" >> "$RESULT_CSV"
    else
        echo "  ⚠️  提取失败，记录为失败"
        echo "$EXP_ID,$CONFIG_NAME,$RHO_VAL,$EPS_VAL,[$ALPHA_MIN-$ALPHA_MAX],FAILED,N/A,N/A,${EXP_MINUTES}m,$EXP_STATUS" >> "$RESULT_CSV"
    fi
    
    # 等待10秒再进行下一个实验
    if [ "$exp_count" -lt 7 ]; then
        echo "  ⏳ 等待10秒..."
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
echo "🎉 GPU 1 所有实验完成！"
echo "============================================================"
echo "总耗时: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo ""
echo "结果汇总:"
cat "$RESULT_CSV"
echo ""
echo "详细日志保存在: $OUTPUT_DIR"
echo "权重文件保存在各实验独立目录中"
echo "============================================================"
