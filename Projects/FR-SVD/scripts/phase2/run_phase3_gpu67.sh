#!/bin/bash
# Phase 3: 等待 GPU6/7 空闲后自动启动下一批实验
# 判断标准：显存占用 < 500MB 持续 10 分钟（600秒）

MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
BASE="/data/zhangs/LLMs/Projects/FR-SVD/result/phase2"
SCRIPT="/data/zhangs/LLMs/Projects/FR-SVD/FR_SVD_v1.3.py"
FD="--fisher_mode direct"
COM="--model ${MODEL} --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 --model_seq_len 2048 --DEV cuda:0"

IDLE_MB=500          # 低于此值视为空闲
IDLE_CHECKS=10       # 需要连续 10 次（每次 60s）= 10 分钟
CHECK_INTERVAL=60

wait_for_gpu_idle() {
    local GPU=$1
    local count=0
    echo "[$(date '+%H:%M:%S')] 等待 GPU${GPU} 空闲（连续 ${IDLE_CHECKS} 次 < ${IDLE_MB}MB）..."
    while true; do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU} 2>/dev/null | tr -d ' ')
        if [ -n "$USED" ] && [ "$USED" -lt "$IDLE_MB" ]; then
            count=$((count + 1))
            echo "[$(date '+%H:%M:%S')] GPU${GPU} 空闲确认 ${count}/${IDLE_CHECKS}（已用 ${USED}MB）"
            if [ "$count" -ge "$IDLE_CHECKS" ]; then
                echo "[$(date '+%H:%M:%S')] GPU${GPU} 确认空闲，启动实验！"
                return 0
            fi
        else
            if [ "$count" -gt 0 ]; then
                echo "[$(date '+%H:%M:%S')] GPU${GPU} 重新变忙（已用 ${USED}MB），重置计数"
            fi
            count=0
        fi
        sleep $CHECK_INTERVAL
    done
}

run_exp() {
    local GPU=$1; local TAG=$2; shift 2
    echo "[$(date '+%H:%M:%S')] GPU${GPU} >> ${TAG}"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} ${SCRIPT} ${COM} ${FD} "$@" \
        --result_dir ${BASE}/${TAG} > ${BASE}/${TAG}.log 2>&1
    PPL=$(grep -oP "wikitext2.*?\K[0-9]+\.[0-9]+" ${BASE}/${TAG}.log 2>/dev/null | tail -1)
    echo "[$(date '+%H:%M:%S')] GPU${GPU} << ${TAG} PPL=${PPL:-N/A}"
}

mkdir -p ${BASE}

# ============================================================
# GPU6：等待空闲后，串行跑 6 个实验（探索 eps/nsamples/rho细粒度）
# ============================================================
gpu6_worker() {
    wait_for_gpu_idle 6

    echo "[$(date '+%H:%M:%S')] ===== GPU6 Phase3 开始 ====="

    run_exp 6 "fisher_direct_rho0.5_a04_06_eps1e3"   --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --eps 1e-3
    run_exp 6 "fisher_direct_rho0.5_a04_06_eps1e5"   --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --eps 1e-5
    run_exp 6 "fisher_direct_rho0.42_a04_06"          --rho 0.42 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9
    run_exp 6 "fisher_direct_rho0.48_a04_06"          --rho 0.48 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9
    run_exp 6 "fisher_direct_rho0.5_a04_06_ns64"      --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --nsamples 64
    run_exp 6 "fisher_direct_rho0.5_a04_06_ns256"     --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --nsamples 256

    echo "[$(date '+%H:%M:%S')] ===== GPU6 Phase3 全部完成 ====="
}

# ============================================================
# GPU7：等待空闲后，串行跑 6 个实验（seed稳定性/r1精细/lambda_psi）
# ============================================================
gpu7_worker() {
    wait_for_gpu_idle 7

    echo "[$(date '+%H:%M:%S')] ===== GPU7 Phase3 开始 ====="

    run_exp 7 "fisher_direct_rho0.5_a04_06_seed0"     --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --seed 0
    run_exp 7 "fisher_direct_rho0.5_a04_06_seed42"    --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --seed 42
    run_exp 7 "fisher_direct_rho0.5_a04_06_r1_0.88"   --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.88
    run_exp 7 "fisher_direct_rho0.5_a04_06_r1_0.92"   --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.92
    run_exp 7 "fisher_direct_rho0.5_a04_06_lam1e3"    --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --lambda_psi 1e-3
    run_exp 7 "fisher_direct_rho0.5_a04_06_lam1e5"    --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --lambda_psi 1e-5

    echo "[$(date '+%H:%M:%S')] ===== GPU7 Phase3 全部完成 ====="
}

# 两个 GPU 并行等待 + 执行
gpu6_worker &
GPU6_PID=$!
gpu7_worker &
GPU7_PID=$!

echo "[$(date '+%H:%M:%S')] Phase3 监控已启动 GPU6_PID=${GPU6_PID} GPU7_PID=${GPU7_PID}"
wait $GPU6_PID $GPU7_PID
echo "[$(date '+%H:%M:%S')] ===== Phase3 全部完成 ====="
