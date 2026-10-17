#!/bin/bash
# ============================================================
# FR-SVD Compression Script（v1.5 — Qwen3.5-35B-A3B 适配）
# ============================================================
# 使用方式：
#   bash run_fr_svd.sh                         # 使用默认 Qwen3.5 配置
#   bash run_fr_svd.sh --gpu 5                 # 指定 GPU
#   bash run_fr_svd.sh --ratio 0.3             # 修改压缩率
#
# 后台执行：
#   nohup bash run_fr_svd.sh --gpu 1 \
#     > /data/zhangs/LLMs/FR-SVD/result/v1.5/qwen3_moe/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ============================================================


# ==================== 可修改参数 ====================

GPU="${GPU:-1}"
# 目标模型：Qwen3.5-35B-A3B (MoE, 35B 总参 / ~3B 激活参, 40 层, hidden=2048,
# 混合注意力 3x linear_attention + 1x full_attention, 256 experts)
MODEL_PATH="${MODEL_PATH:-/data/zhangs/LLMs/Models/Qwen/Qwen3.5-35B-A3B}"
DATASET="${DATASET:-wikitext2}"
RATIO="${RATIO:-0.2}"                   # 压缩率 (0.2 = 去掉 20%)
# Qwen3.5-35B-A3B 单层 GPU 占用 ~8-12GB（fp32 compute），保守 nsamples
NSAMPLES="${NSAMPLES:-128}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-3}"

# FR-SVD 参数
# 注意：Qwen3.5 MoE 与 LLaMA-7B 结构差异较大，不沿用 LLaMA-7B 最优的
# rho=0.42 / alpha=[0.45,0.55] / r1_ratio=0.95 / lambda_psi=1e-5，
# 改用 v1.3 论文默认（alpha=[0.25,0.75], rho=1.0, r1_ratio=1.0）作为稳健起点。
R1_RATIO="${R1_RATIO:-1.0}"             # 1.0 = Stage 2 关闭（纯 CEALC 单阶段，最稳）
EPS="${EPS:-1e-4}"
RHO="${RHO:-1.0}"
ALPHA_MIN="${ALPHA_MIN:-0.25}"
ALPHA_MAX="${ALPHA_MAX:-0.75}"
FISHER_MODE="${FISHER_MODE:-none}"      # 新架构先不开 Fisher，避免 Psi 估计不稳
LAMBDA_PSI="${LAMBDA_PSI:-1e-4}"

# MoE：256 个 experts 每个只见到 (nsamples*seqlen*8/256) 个 token，
# H 估计不稳定 → 默认跳过 experts，只压 attention + shared_expert。
SKIP_MOE_EXPERTS="${SKIP_MOE_EXPERTS:-1}"

# Python 脚本
FR_SVD_SCRIPT="${FR_SVD_SCRIPT:-FR_SVD_v1.5.py}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-result/v1.5}"

# ==================== 命令行参数覆盖 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)              GPU="$2"; shift 2;;
        --model)            MODEL_PATH="$2"; shift 2;;
        --dataset)          DATASET="$2"; shift 2;;
        --ratio)            RATIO="$2"; shift 2;;
        --nsamples)         NSAMPLES="$2"; shift 2;;
        --seq_len)          SEQ_LEN="$2"; shift 2;;
        --seed)             SEED="$2"; shift 2;;
        --r1_ratio)         R1_RATIO="$2"; shift 2;;
        --eps)              EPS="$2"; shift 2;;
        --rho)              RHO="$2"; shift 2;;
        --alpha_min)        ALPHA_MIN="$2"; shift 2;;
        --alpha_max)        ALPHA_MAX="$2"; shift 2;;
        --fisher_mode)      FISHER_MODE="$2"; shift 2;;
        --lambda_psi)       LAMBDA_PSI="$2"; shift 2;;
        --skip_moe_experts) SKIP_MOE_EXPERTS=1; shift 1;;
        --no_skip_moe_experts) SKIP_MOE_EXPERTS=0; shift 1;;
        --output_dir)       OUTPUT_DIR="$2"; shift 2;;
        --script)           FR_SVD_SCRIPT="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# ==================== 准备 ====================

# Qwen3.5 需要 compress_Qwen3.5 环境；其他模型回退到 compress
PYTHON="/data/zhangs/anaconda3/envs/compress_Qwen3.5/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATE_STAMP=$(date +"%Y%m%d_%H%M%S")

MODEL_SHORT=$(basename "$MODEL_PATH" | sed 's/[-\.]/_/g')
KEEP_RATIO=$(python3 -c "print(round(1-${RATIO}, 2))")

EXP_TAG="${MODEL_SHORT}_r${RATIO}_r1${R1_RATIO}_rho${RHO}_a${ALPHA_MIN}-${ALPHA_MAX}_ns${NSAMPLES}_fis${FISHER_MODE}_skexp${SKIP_MOE_EXPERTS}_s${SEED}"
RESULT_DIR="${OUTPUT_DIR}/${EXP_TAG}"

mkdir -p "$RESULT_DIR"

LOG_FILE="${RESULT_DIR}/log_${DATE_STAMP}.txt"

# ==================== 打印配置 ====================

echo "============================================================" | tee "$LOG_FILE"
echo "FR-SVD v1.5 Compression" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Time:             $(date)" | tee -a "$LOG_FILE"
echo "GPU:              $GPU" | tee -a "$LOG_FILE"
echo "Python:           $PYTHON" | tee -a "$LOG_FILE"
echo "Model:            $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Dataset:          $DATASET" | tee -a "$LOG_FILE"
echo "Ratio:            $RATIO (keep ${KEEP_RATIO})" | tee -a "$LOG_FILE"
echo "Samples:          $NSAMPLES" | tee -a "$LOG_FILE"
echo "Seq Length:       $SEQ_LEN" | tee -a "$LOG_FILE"
echo "Seed:             $SEED" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "r1_ratio:         $R1_RATIO" | tee -a "$LOG_FILE"
echo "eps:              $EPS" | tee -a "$LOG_FILE"
echo "rho:              $RHO" | tee -a "$LOG_FILE"
echo "alpha:            [$ALPHA_MIN, $ALPHA_MAX]" | tee -a "$LOG_FILE"
echo "fisher_mode:      $FISHER_MODE (lambda_psi=$LAMBDA_PSI)" | tee -a "$LOG_FILE"
echo "skip_moe_experts: $SKIP_MOE_EXPERTS" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "Script:           $FR_SVD_SCRIPT" | tee -a "$LOG_FILE"
echo "Output:           $RESULT_DIR" | tee -a "$LOG_FILE"
echo "Log:              $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== 运行 ====================

START_TIME=$(date +%s)

MOE_ARGS=()
if [[ "$SKIP_MOE_EXPERTS" == "1" ]]; then
    MOE_ARGS+=(--skip_moe_experts)
fi

# v1.5 在压缩完成后会自动调用 ppl_eval 评估 wikitext2，无需额外评估步骤。
CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT_DIR/$FR_SVD_SCRIPT" \
    --model "$MODEL_PATH" \
    --ratio "$RATIO" \
    --dataset "$DATASET" \
    --nsamples "$NSAMPLES" \
    --seed "$SEED" \
    --model_seq_len "$SEQ_LEN" \
    --eps "$EPS" \
    --r1_ratio "$R1_RATIO" \
    --rho "$RHO" \
    --alpha_min "$ALPHA_MIN" \
    --alpha_max "$ALPHA_MAX" \
    --fisher_mode "$FISHER_MODE" \
    --lambda_psi "$LAMBDA_PSI" \
    --result_dir "$RESULT_DIR" \
    "${MOE_ARGS[@]}" \
    --DEV cuda:0 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
DURATION=$(( (END_TIME - START_TIME) / 60 ))

# ==================== 总结 ====================

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Completed successfully in ${DURATION} minutes" | tee -a "$LOG_FILE"
else
    echo "FAILED (exit code $EXIT_CODE) after ${DURATION} minutes" | tee -a "$LOG_FILE"
fi
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
