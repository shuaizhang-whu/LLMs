#!/bin/bash

# SAES-SVD v2.1.2: 逐层闭环压缩 (ACES修正版)
# 修复护栏机制，α范围对应论文Table 4最优设置

# 参数检查
if [ $# -ne 1 ]; then
    echo "使用方法: $0 <compression_ratio>"
    echo "示例: $0 0.2  # 移除20%参数，保留80%"
    echo "目前rho=1, eps=1e-4固定"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=1

RATIO="$1"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# 配置
MODEL_PATH="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
DATASET="wikitext2"
NSAMPLES="128"
DEVICE="cuda:0"
EPS="1e-4"
SEED="3"

# 输出路径
OUTPUT_BASE="./result/SAES_SVD_v2.1.2/llama-7b-hf/ratio_${RATIO}/dataset_${DATASET}_samples_${NSAMPLES}_seed_${SEED}"
mkdir -p "$OUTPUT_BASE"

# 日志设置
LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/saes_svd_v2.1.2_ratio${RATIO}_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始压缩 ratio=${RATIO}..."

# 记录开始时间
start_time=$(date +%s)

# 执行压缩 (使用conda环境)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate compress
PYTHONPATH="/data/zhangs/LLMs/SAES-SVD:$PYTHONPATH" \
python3 -u SAES-SVD_v2.1.2.py \
    --model "$MODEL_PATH" \
    --ratio "$RATIO" \
    --dataset "$DATASET" \
    --nsamples "$NSAMPLES" \
    --DEV "$DEVICE" \
    --eps "$EPS" \
    --save_path "$OUTPUT_BASE/compressed_model" \
    --model_seq_len 2048 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_FILE"

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))
duration_min=$((duration / 60))
duration_sec=$((duration % 60))

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ ratio=${RATIO} 压缩完成！耗时: ${duration_min}m ${duration_sec}s"
echo ""

# 提取关键结果
echo "📊 提取关键结果..."
if [ -f "$LOG_FILE" ]; then
    echo "✅ 日志保存至: $LOG_FILE"
    
    # 提取PPL结果
    ppl_result=$(grep -o "PPL after pruning: {'[^']*': [0-9.]*}" "$LOG_FILE" | tail -1)
    if [ -n "$ppl_result" ]; then
        echo "📈 $ppl_result"
    fi
    
    # 提取平均β和Δ/H比率
    avg_beta=$(grep -o "Average β across all layers: [0-9.]*" "$LOG_FILE" | tail -1)
    avg_delta=$(grep -o "Average Δ/H ratio: [0-9.]*" "$LOG_FILE" | tail -1) 
    beta_range=$(grep -o "β range: \[[0-9.]\+, [0-9.]\+\]" "$LOG_FILE" | tail -1)
    
    if [ -n "$avg_beta" ]; then
        echo "🎯 $avg_beta"
    fi
    if [ -n "$avg_delta" ]; then
        echo "📊 $avg_delta"
    fi
    if [ -n "$beta_range" ]; then
        echo "📏 $beta_range"
    fi
else
    echo "❌ 日志文件未找到"
fi

echo "========================"
echo "v2.1.2 期望改进:"
echo "- β 不再卡在 0.675"
echo "- β 分布在 [0.2, 0.4286] 范围内"
echo "- α 对应论文最优 [0.25, 0.75]"
echo "- PPL 应接近论文的 7.17"
echo "========================"
