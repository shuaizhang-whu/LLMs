#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# GPU5 续跑: 等待当前 3 个实验(rho0.6_a03_07, rho0.6_a04_06_r1_0.8, rho0.5_a03_06)结束
# 然后跑剩余 4 个实验，每批 2 个并行
MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${PROJECT_ROOT}/FR_SVD_v1.3.py"
BASE="${PROJECT_ROOT}/result/phase2"
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

# 等待当前 GPU5 上的 3 个 Python 进程结束
echo "[$(date '+%H:%M:%S')] 等待 GPU5 当前批次完成（进程 3192783 3192785 3192787）..."
while kill -0 3192783 2>/dev/null || kill -0 3192785 2>/dev/null || kill -0 3192787 2>/dev/null; do
    sleep 30
done
echo "[$(date '+%H:%M:%S')] GPU5 当前批次完成，开始续跑"

# 批次2：2 并行
echo "[$(date '+%H:%M:%S')] === GPU5 批次2 ==="
run_exp "rho0.7_a04_06"   --r1_ratio 0.9 --rho 0.7 --alpha_min 0.4 --alpha_max 0.6 &
P1=$!
run_exp "rho0.7_a03_07"   --r1_ratio 0.9 --rho 0.7 --alpha_min 0.3 --alpha_max 0.7 &
P2=$!
wait $P1 $P2
echo "[$(date '+%H:%M:%S')] === GPU5 批次2 完成 ==="

# 批次3：2 并行
echo "[$(date '+%H:%M:%S')] === GPU5 批次3 ==="
run_exp "rho0.5_a035_065"       --r1_ratio 0.9 --rho 0.5 --alpha_min 0.35 --alpha_max 0.65 &
P1=$!
run_exp "rho0.55_a04_06_r1_0.9" --rho 0.55 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 &
P2=$!
wait $P1 $P2
echo "[$(date '+%H:%M:%S')] === GPU5 全部完成 ==="
