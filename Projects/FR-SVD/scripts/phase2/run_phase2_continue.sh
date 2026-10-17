#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Phase 2 续跑脚本 - 等待第一组4个实验跑完后，接续启动所有后续实验
# 第一组（已在跑）：
#   GPU4: rho0.3_a04_06
#   GPU5: rho0.3_a03_07  
#   GPU6: rho0.6_a025_075
#   GPU7: rho0.6_a04_06_r1_0.75

MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="${PROJECT_ROOT}/FR_SVD_v1.3.py"
BASE="${PROJECT_ROOT}/result/phase2"
mkdir -p ${BASE}

COMMON="--model ${MODEL} --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 --DEV cuda:0 --model_seq_len 2048"

run_exp() {
    local GPU=$1
    local TAG=$2
    shift 2
    echo "[$(date '+%H:%M:%S')] >>> GPU${GPU}: ${TAG}"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} ${SCRIPT} ${COMMON} "$@" \
        --result_dir ${BASE}/${TAG} \
        2>&1 | tee ${BASE}/${TAG}.log
    local PPL=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" ${BASE}/${TAG}.log 2>/dev/null || echo "N/A")
    echo "[$(date '+%H:%M:%S')] <<< GPU${GPU}: ${TAG} PPL=${PPL}"
}

# ========== 等待第一组跑完 ==========
echo "等待第一组实验完成..."
while ps aux | grep -v grep | grep "FR_SVD_v1.3.py" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date '+%H:%M:%S')] 第一组完成，启动后续实验"

# ========== 第二组 ==========
run_exp 4 "rho0.4_a04_06" --r1_ratio 0.9 --rho 0.4 --alpha_min 0.4 --alpha_max 0.6 &
run_exp 5 "rho0.4_a03_07" --r1_ratio 0.9 --rho 0.4 --alpha_min 0.3 --alpha_max 0.7 &
run_exp 6 "rho0.6_a03_06" --r1_ratio 0.9 --rho 0.6 --alpha_min 0.3 --alpha_max 0.6 &
run_exp 7 "rho0.6_a04_06_r1_0.8" --rho 0.6 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.8 &
wait
echo "[$(date '+%H:%M:%S')] 第二组完成"

# ========== 第三组 ==========
run_exp 4 "rho0.5_a04_06" --r1_ratio 0.9 --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 &
run_exp 5 "rho0.5_a03_07" --r1_ratio 0.9 --rho 0.5 --alpha_min 0.3 --alpha_max 0.7 &
run_exp 6 "rho0.6_a035_065" --r1_ratio 0.9 --rho 0.6 --alpha_min 0.35 --alpha_max 0.65 &
run_exp 7 "rho0.6_a04_06_r1_0.85" --rho 0.6 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.85 &
wait
echo "[$(date '+%H:%M:%S')] 第三组完成"

# ========== 第四组 ==========
run_exp 4 "rho0.6_a04_06" --r1_ratio 0.9 --rho 0.6 --alpha_min 0.4 --alpha_max 0.6 &
run_exp 5 "rho0.6_a03_07" --r1_ratio 0.9 --rho 0.6 --alpha_min 0.3 --alpha_max 0.7 &
run_exp 6 "rho0.5_a03_06" --r1_ratio 0.9 --rho 0.5 --alpha_min 0.3 --alpha_max 0.6 &
run_exp 7 "rho0.6_a04_06_r1_0.95" --rho 0.6 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.95 &
wait
echo "[$(date '+%H:%M:%S')] 第四组完成"

# ========== 第五组 ==========
run_exp 4 "rho0.7_a04_06" --r1_ratio 0.9 --rho 0.7 --alpha_min 0.4 --alpha_max 0.6 &
run_exp 5 "rho0.7_a03_07" --r1_ratio 0.9 --rho 0.7 --alpha_min 0.3 --alpha_max 0.7 &
run_exp 6 "rho0.5_a035_065" --r1_ratio 0.9 --rho 0.5 --alpha_min 0.35 --alpha_max 0.65 &
run_exp 7 "rho0.55_a04_06_r1_0.9" --rho 0.55 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 &
wait
echo "[$(date '+%H:%M:%S')] 第五组完成"

# ========== 第六组 ==========
run_exp 4 "rho0.8_a04_06" --r1_ratio 0.9 --rho 0.8 --alpha_min 0.4 --alpha_max 0.6 &
run_exp 5 "rho0.8_a03_07" --r1_ratio 0.9 --rho 0.8 --alpha_min 0.3 --alpha_max 0.7 &
run_exp 6 "rho0.5_a04_06" --r1_ratio 0.9 --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 &
run_exp 7 "rho0.65_a04_06_r1_0.9" --rho 0.65 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 &
wait
echo "[$(date '+%H:%M:%S')] 第六组完成"

# ========== 第七组：Fisher direct 对比实验 ==========
run_exp 4 "fisher_direct_rho0.6_a04_06" --r1_ratio 0.9 --rho 0.6 --alpha_min 0.4 --alpha_max 0.6 --fisher_mode direct &
run_exp 5 "fisher_direct_rho0.6_a03_07" --r1_ratio 0.9 --rho 0.6 --alpha_min 0.3 --alpha_max 0.7 --fisher_mode direct &
run_exp 6 "fisher_direct_rho0.5_a04_06" --r1_ratio 0.9 --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --fisher_mode direct &
run_exp 7 "fisher_direct_rho0.6_a04_06_r1_1.0" --rho 0.6 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 1.0 --fisher_mode direct &
wait
echo "[$(date '+%H:%M:%S')] 第七组（Fisher direct）完成"

# ========== 汇总结果 ==========
echo ""
echo "=========================================="
echo "Phase 2 全部实验结果汇总"
echo "=========================================="
for f in ${BASE}/*.log; do
    TAG=$(basename $f .log)
    PPL=$(grep -oP "PPL after pruning: \{'wikitext2': \K[\d.]+" $f 2>/dev/null || echo "N/A")
    echo "  ${TAG}: PPL=${PPL}"
done | sort
