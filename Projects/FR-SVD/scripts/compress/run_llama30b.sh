#!/bin/bash
# LLaMA-30B FR-SVD v1.5 压缩脚本
# 参数：nsamples=64, fisher_mode=soft, r1_ratio=0.92
# 用法：bash run_llama30b.sh [GPU_ID] [OUT_DIR]
#       GPU_ID 默认由调用方通过 $1 传入，或环境变量 GPU 传入
#       OUT_DIR 由调用方（watch脚本）预先创建并传入，所有日志统一存放于此

set -euo pipefail

GPU="${1:-${GPU:-0}}"
# 若调用方预先创建了 OUT_DIR（第2个参数或环境变量），则透传给 run_fr_svd_v1.5.sh
OUT_DIR_ARG="${2:-${OUT_DIR:-}}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "启动 LLaMA-30B 压缩 (GPU=${GPU})..."

EXTRA=()
if [[ -n "$OUT_DIR_ARG" ]]; then
    EXTRA+=(--out_dir "$OUT_DIR_ARG")
fi

CUDA_VISIBLE_DEVICES="${GPU}" bash "${SCRIPT_DIR}/run_fr_svd_v1.5.sh" \
    --model_tag    llama_30b \
    --gpu          "${GPU}" \
    --ratio        0.2 \
    --nsamples     64 \
    --dataset      wikitext2 \
    --r1_ratio     0.92 \
    --rho          1.0 \
    --alpha_min    0.25 \
    --alpha_max    0.75 \
    --eps          1e-4 \
    --fisher_mode  soft \
    --lambda_psi   1e-5 \
    "${EXTRA[@]}"
