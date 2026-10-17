#!/bin/bash
set -euo pipefail

# 独立评估 Qwen2.5-7B FR-SVD 压缩模型。
# 压缩命令先保留为注释，当前脚本只执行评估。
#
# 压缩命令示例（暂不执行）：
# bash /data/zhangs/LLMs/Projects/FR-SVD/run_fr_svd_v1.5.sh \
#     --model_tag qwen2_5_7b \
#     --gpu 0 \
#     --ratio 0.2 \
#     --nsamples 256 \
#     --r1_ratio 0.92 \
#     --rho 1.0 \
#     --out_dir /data/zhangs/LLMs/Projects/FR-SVD/result/FR_SVD_v1.5/qwen2_5_7b/ns256_r1_0.92_rho1.0_20260417_211008

GPU=${1:-0}
RESULT_DIR="/data/zhangs/LLMs/Projects/FR-SVD/result/FR_SVD_v1.5/qwen2_5_7b/ns256_r1_0.92_rho1.0_20260417_211008"
MODEL_PT="${RESULT_DIR}/_data_zhangs_LLMs_Models_Qwen_Qwen2.5_7B_fr_svd_0.8.pt"
LOG_FILE="${RESULT_DIR}/eval.log"
PYTHON="/data/zhangs/anaconda3/envs/compress_Qwen3.5/bin/python"
FR_SVD_PY="/data/zhangs/LLMs/Projects/FR-SVD/FR_SVD_v1.5.py"

if [[ ! -f "$MODEL_PT" ]]; then
    echo "ERROR: model checkpoint not found: $MODEL_PT" >&2
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found or not executable: $PYTHON" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"

echo "============================================"
echo "  Qwen2.5-7B PPL 评估（仅评估，不执行压缩）"
echo "  GPU:    ${GPU}"
echo "  模型:   ${MODEL_PT}"
echo "  日志:   ${LOG_FILE}"
echo "============================================"

CUDA_VISIBLE_DEVICES="${GPU}" "$PYTHON" "$FR_SVD_PY" \
    --eval_only \
    --model_path "$MODEL_PT" \
    --model_seq_len 2048 \
    --eval_batch_size 1 \
    --DEV cuda:0 \
    2>&1 | tee "$LOG_FILE"
