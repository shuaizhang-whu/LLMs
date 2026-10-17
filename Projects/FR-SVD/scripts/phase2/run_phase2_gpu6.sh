#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Phase 2 - GPU 6: alpha 扫描 + Fisher direct 对比实验
# 6 组 alpha 扫描 + 3 组 Fisher direct 对比 = 9 组实验

MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${PROJECT_ROOT}/FR_SVD_v1.3.py"
BASE="${PROJECT_ROOT}/result/phase2"
mkdir -p ${BASE}

COMMON="--model ${MODEL} --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 --DEV cuda:0 --model_seq_len 2048 --r1_ratio 0.9"

# rho=0.6 + 不同 alpha 范围
for PAIR in "0.6 0.25 0.75 a025_075" "0.6 0.3 0.6 a03_06" "0.6 0.35 0.65 a035_065"; do
    set -- $PAIR
    RHO=$1; AMIN=$2; AMAX=$3; ATAG=$4
    TAG="rho${RHO}_${ATAG}"
    echo ">>> ${TAG}"
    CUDA_VISIBLE_DEVICES=6 ${PYTHON} ${SCRIPT} ${COMMON} \
        --rho ${RHO} --alpha_min ${AMIN} --alpha_max ${AMAX} \
        --result_dir ${BASE}/${TAG} \
        2>&1 | tee ${BASE}/${TAG}.log
done

# rho=0.5 + 不同 alpha 范围
for PAIR in "0.5 0.4 0.6 a04_06" "0.5 0.3 0.6 a03_06" "0.5 0.35 0.65 a035_065"; do
    set -- $PAIR
    RHO=$1; AMIN=$2; AMAX=$3; ATAG=$4
    TAG="rho${RHO}_${ATAG}"
    echo ">>> ${TAG}"
    CUDA_VISIBLE_DEVICES=6 ${PYTHON} ${SCRIPT} ${COMMON} \
        --rho ${RHO} --alpha_min ${AMIN} --alpha_max ${AMAX} \
        --result_dir ${BASE}/${TAG} \
        2>&1 | tee ${BASE}/${TAG}.log
done

echo "=== Alpha 扫描完成，开始 Fisher direct 对比实验 ==="

# Fisher direct 对比实验：用 rho=0.6 + alpha=[0.4,0.6]（Phase 1 验证过的最佳参数）
# 对比组1：fisher_mode=none (基线，已由 GPU4 rho0.6_a04_06 覆盖)
# 对比组2：fisher_mode=direct
# 对比组3：fisher_mode=direct + rho=0.5

for PAIR in "0.6 0.4 0.6 fisher_direct_rho0.6_a04_06" "0.5 0.4 0.6 fisher_direct_rho0.5_a04_06" "0.6 0.3 0.7 fisher_direct_rho0.6_a03_07"; do
    set -- $PAIR
    RHO=$1; AMIN=$2; AMAX=$3; TAG=$4
    echo ">>> ${TAG}"
    CUDA_VISIBLE_DEVICES=6 ${PYTHON} ${SCRIPT} ${COMMON} \
        --rho ${RHO} --alpha_min ${AMIN} --alpha_max ${AMAX} \
        --fisher_mode direct \
        --result_dir ${BASE}/${TAG} \
        2>&1 | tee ${BASE}/${TAG}.log
done

echo "GPU6 全部完成"
for f in ${BASE}/rho0.6_a025_075.log ${BASE}/rho0.6_a03_06.log ${BASE}/rho0.6_a035_065.log ${BASE}/rho0.5_a04_06.log ${BASE}/rho0.5_a03_06.log ${BASE}/rho0.5_a035_065.log ${BASE}/fisher_direct_rho0.6_a04_06.log ${BASE}/fisher_direct_rho0.5_a04_06.log ${BASE}/fisher_direct_rho0.6_a03_07.log; do
    TAG=$(basename $f .log)
    PPL=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" $f 2>/dev/null || echo "N/A")
    echo "  ${TAG}: PPL=${PPL}"
done
