#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Phase 2 - GPU 7: r1_ratio 扫描 + rho 微调，alpha=[0.4,0.6]
# 6 组实验

MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${PROJECT_ROOT}/FR_SVD_v1.3.py"
BASE="${PROJECT_ROOT}/result/phase2"
mkdir -p ${BASE}

COMMON="--model ${MODEL} --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 --DEV cuda:0 --model_seq_len 2048"

# r1_ratio 扫描：rho=0.6, alpha=[0.4,0.6]
for R1 in 0.75 0.8 0.85 0.95; do
    TAG="rho0.6_a04_06_r1_${R1}"
    echo ">>> ${TAG}"
    CUDA_VISIBLE_DEVICES=7 ${PYTHON} ${SCRIPT} ${COMMON} \
        --rho 0.6 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio ${R1} \
        --result_dir ${BASE}/${TAG} \
        2>&1 | tee ${BASE}/${TAG}.log
done

# rho 微调：rho=0.55 和 0.65, alpha=[0.4,0.6], r1_ratio=0.9
for RHO in 0.55 0.65; do
    TAG="rho${RHO}_a04_06_r1_0.9"
    echo ">>> ${TAG}"
    CUDA_VISIBLE_DEVICES=7 ${PYTHON} ${SCRIPT} ${COMMON} \
        --rho ${RHO} --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 \
        --result_dir ${BASE}/${TAG} \
        2>&1 | tee ${BASE}/${TAG}.log
done

echo "GPU7 全部完成"
for f in ${BASE}/rho0.6_a04_06_r1_0.75.log ${BASE}/rho0.6_a04_06_r1_0.8.log ${BASE}/rho0.6_a04_06_r1_0.85.log ${BASE}/rho0.6_a04_06_r1_0.95.log ${BASE}/rho0.55_a04_06_r1_0.9.log ${BASE}/rho0.65_a04_06_r1_0.9.log; do
    TAG=$(basename $f .log)
    PPL=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" $f 2>/dev/null || echo "N/A")
    echo "  ${TAG}: PPL=${PPL}"
done
