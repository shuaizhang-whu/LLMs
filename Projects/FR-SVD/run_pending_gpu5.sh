#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# GPU5: 7 个待跑实验，每批 2-3 个并行，用 PID wait 确保顺序
MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${SCRIPT_DIR}/FR_SVD_v1.3.py"
BASE="${SCRIPT_DIR}/result/phase2"
mkdir -p ${BASE}

COMMON="--model ${MODEL} --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 --DEV cuda:0 --model_seq_len 2048"

run_exp() {
    local TAG=$1; shift
    echo "[$(date '+%H:%M:%S')] >>> GPU5: ${TAG}"
    CUDA_VISIBLE_DEVICES=5 ${PYTHON} ${SCRIPT} ${COMMON} "$@" \
        --result_dir ${BASE}/${TAG} \
        > ${BASE}/${TAG}.log 2>&1
    PPL=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" ${BASE}/${TAG}.log 2>/dev/null | head -1 || echo "N/A")
    echo "[$(date '+%H:%M:%S')] <<< GPU5: ${TAG} PPL=${PPL}"
}

# 批次1：2 并行
echo "[$(date '+%H:%M:%S')] === GPU5 批次1 ==="
run_exp "rho0.6_a04_06_r1_0.8" --rho 0.6 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.8 &
P1=$!
run_exp "rho0.6_a03_07"        --r1_ratio 0.9 --rho 0.6 --alpha_min 0.3 --alpha_max 0.7 &
P2=$!
wait $P1 $P2
echo "[$(date '+%H:%M:%S')] === GPU5 批次1 完成 ==="

# 批次2：2 并行
echo "[$(date '+%H:%M:%S')] === GPU5 批次2 ==="
run_exp "rho0.5_a03_06"        --r1_ratio 0.9 --rho 0.5 --alpha_min 0.3 --alpha_max 0.6 &
P1=$!
run_exp "rho0.7_a04_06"        --r1_ratio 0.9 --rho 0.7 --alpha_min 0.4 --alpha_max 0.6 &
P2=$!
wait $P1 $P2
echo "[$(date '+%H:%M:%S')] === GPU5 批次2 完成 ==="

# 批次3：2 并行
echo "[$(date '+%H:%M:%S')] === GPU5 批次3 ==="
run_exp "rho0.7_a03_07"        --r1_ratio 0.9 --rho 0.7 --alpha_min 0.3 --alpha_max 0.7 &
P1=$!
run_exp "rho0.5_a035_065"      --r1_ratio 0.9 --rho 0.5 --alpha_min 0.35 --alpha_max 0.65 &
P2=$!
wait $P1 $P2
echo "[$(date '+%H:%M:%S')] === GPU5 批次3 完成 ==="

# 批次4：1 个
echo "[$(date '+%H:%M:%S')] === GPU5 批次4 ==="
run_exp "rho0.55_a04_06_r1_0.9" --rho 0.55 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9
echo "[$(date '+%H:%M:%S')] === GPU5 全部完成 ==="
