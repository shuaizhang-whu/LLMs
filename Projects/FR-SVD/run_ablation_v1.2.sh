#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# FR-SVD v1.2 消融实验
# 4组实验验证创新点A (Fisher Psi加权) 和创新点B (H-weighted残差补偿)
# GPU4: Baseline + A only
# GPU5: B only + A+B

PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${SCRIPT_DIR}/FR_SVD_v1.2.py"
MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
DATASET="wikitext2"
NSAMPLES=128
SEED=3
SEQ_LEN=2048
RATIO=0.2
RHO=0.6
ALPHA_MIN=0.4
ALPHA_MAX=0.6

COMMON_ARGS="--model $MODEL --dataset $DATASET --nsamples $NSAMPLES --seed $SEED \
--model_seq_len $SEQ_LEN --ratio $RATIO --rho $RHO --alpha_min $ALPHA_MIN --alpha_max $ALPHA_MAX"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./result/ablation_v1.2_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "===== FR-SVD v1.2 消融实验 ====="
echo "日志目录: $LOG_DIR"
echo "开始时间: $(date)"
echo ""

# ====== Exp 1: Baseline (no A, no B) — GPU 4 ======
echo "[Exp 1] Baseline: Fisher=OFF, Stage2=OFF (r1_ratio=1.0)"
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON $SCRIPT $COMMON_ARGS \
    --r1_ratio 1.0 \
    --DEV cuda:0 \
    --save_path "$LOG_DIR/exp1_baseline" \
    > "$LOG_DIR/exp1_baseline.log" 2>&1 &
PID1=$!
echo "  PID=$PID1, GPU=4"

# 等30秒让GPU4的显存分配稳定
sleep 30

# ====== Exp 2: A only (Fisher ON, no Stage2) — GPU 4 等exp1完成后 ======
# 由于GPU4一次只能跑一个实验，exp2在exp1完成后自动运行
(
    wait $PID1
    echo "[Exp 2] A only: Fisher=ON, Stage2=OFF (r1_ratio=1.0)"
    echo "  开始时间: $(date)"
    CUDA_VISIBLE_DEVICES=4 $PYTHON $SCRIPT $COMMON_ARGS \
        --r1_ratio 1.0 \
        --use_fisher \
        --DEV cuda:0 \
        --save_path "$LOG_DIR/exp2_fisher_only" \
        > "$LOG_DIR/exp2_fisher_only.log" 2>&1
    echo "[Exp 2] 完成时间: $(date)"
) &

# ====== Exp 3: B only (no Fisher, Stage2 ON) — GPU 5 ======
echo "[Exp 3] B only: Fisher=OFF, Stage2=ON (r1_ratio=0.9)"
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON $SCRIPT $COMMON_ARGS \
    --r1_ratio 0.9 \
    --DEV cuda:0 \
    --save_path "$LOG_DIR/exp3_stage2_only" \
    > "$LOG_DIR/exp3_stage2_only.log" 2>&1 &
PID3=$!
echo "  PID=$PID3, GPU=5"

# ====== Exp 4: A+B (Fisher ON + Stage2 ON) — GPU 5 等exp3完成后 ======
(
    wait $PID3
    echo "[Exp 4] A+B: Fisher=ON, Stage2=ON (r1_ratio=0.9)"
    echo "  开始时间: $(date)"
    CUDA_VISIBLE_DEVICES=5 $PYTHON $SCRIPT $COMMON_ARGS \
        --r1_ratio 0.9 \
        --use_fisher \
        --DEV cuda:0 \
        --save_path "$LOG_DIR/exp4_fisher_stage2" \
        > "$LOG_DIR/exp4_fisher_stage2.log" 2>&1
    echo "[Exp 4] 完成时间: $(date)"
) &

echo ""
echo "所有实验已提交！"
echo "查看进度:"
echo "  tail -f $LOG_DIR/exp1_baseline.log"
echo "  tail -f $LOG_DIR/exp3_stage2_only.log"
echo ""
echo "查看PPL结果:"
echo "  grep -r 'wikitext2' $LOG_DIR/*.log"
