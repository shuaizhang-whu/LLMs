#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Phase 2 - GPU 5: rho 扫描，alpha=[0.3, 0.7]，r1_ratio=0.9
# 6 组实验，rho = 0.3, 0.4, 0.5, 0.6, 0.7, 0.8

MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${SCRIPT_DIR}/FR_SVD_v1.3.py"
BASE="${SCRIPT_DIR}/result/phase2"
mkdir -p ${BASE}

COMMON="--model ${MODEL} --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 --DEV cuda:0 --model_seq_len 2048 --r1_ratio 0.9"

for RHO in 0.3 0.4 0.5 0.6 0.7 0.8; do
    TAG="rho${RHO}_a03_07"
    echo ">>> ${TAG}"
    CUDA_VISIBLE_DEVICES=5 ${PYTHON} ${SCRIPT} ${COMMON} \
        --rho ${RHO} --alpha_min 0.3 --alpha_max 0.7 \
        --result_dir ${BASE}/${TAG} \
        2>&1 | tee ${BASE}/${TAG}.log
done

echo "GPU5 全部完成"
for f in ${BASE}/rho*_a03_07.log; do
    TAG=$(basename $f .log)
    PPL=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" $f 2>/dev/null || echo "N/A")
    echo "  ${TAG}: PPL=${PPL}"
done
