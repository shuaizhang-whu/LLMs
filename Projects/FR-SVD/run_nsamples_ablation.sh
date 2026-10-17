#!/bin/bash
# nsamples 消融实验 —— 自动触发版本
# =========================================================
# 背景：ns=64→8.425（差），ns=128→7.837（基线），ns=256→7.702（突破 -0.134）
#       需要明确：
#         Q1. ns=256 的提升是否泛化到其他参数组合？
#         Q2. ns=512 是否继续提升？提升空间多大？
#         Q3. 最优 alpha=[0.45,0.55] + ns=256 能否更好？
#         Q4. lambda_psi=1e-5（7.826@128）+ ns=256 能叠加吗？
#
# GPU 分工（各自独立监控，空闲 10min 后启动）：
#   GPU6: 5 个实验 —— ns=256 在不同参数上的泛化验证
#   GPU7: 5 个实验 —— ns=512 探索 + 组合最优
# =========================================================

MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
BASE="/data/zhangs/LLMs/FR-SVD/result/phase2"
SCRIPT="/data/zhangs/LLMs/FR-SVD/FR_SVD_v1.3.py"
FD="--fisher_mode direct"
COM="--model ${MODEL} --ratio 0.2 --dataset wikitext2 --seed 3 --model_seq_len 2048 --DEV cuda:0"

MONITOR_LOG="${BASE}/nsamples_monitor.log"
IDLE_MB=500
IDLE_CHECKS=10
CHECK_INTERVAL=60

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "${MONITOR_LOG}"
}

wait_for_gpu_idle() {
    local GPU=$1
    local count=0
    log "GPU${GPU} 开始监控（等待连续 ${IDLE_CHECKS} 次 < ${IDLE_MB}MB）..."
    while true; do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU} 2>/dev/null | tr -d ' ')
        if [ -n "$USED" ] && [ "$USED" -lt "$IDLE_MB" ]; then
            count=$((count + 1))
            log "GPU${GPU} 空闲确认 ${count}/${IDLE_CHECKS}（${USED}MB）"
            [ "$count" -ge "$IDLE_CHECKS" ] && { log "GPU${GPU} 已就绪，启动实验！"; return 0; }
        else
            [ "$count" -gt 0 ] && log "GPU${GPU} 被重新占用（${USED}MB），重置计数"
            count=0
        fi
        sleep ${CHECK_INTERVAL}
    done
}

run_exp() {
    local GPU=$1
    local TAG=$2
    shift 2
    log "GPU${GPU} >> 启动 ${TAG}"
    mkdir -p "${BASE}/${TAG}"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} ${SCRIPT} ${COM} ${FD} "$@" \
        --result_dir "${BASE}/${TAG}" > "${BASE}/${TAG}.log" 2>&1
    PPL=$(grep -oP "wikitext2.*?\K[0-9]+\.[0-9]+" "${BASE}/${TAG}.log" 2>/dev/null | tail -1)
    log "GPU${GPU} << 完成 ${TAG}  PPL=${PPL:-N/A}"
}

mkdir -p "${BASE}"

# ============================================================
# GPU6：ns=256 泛化验证（5个）
# 固定 ns=256，测不同参数是否仍比 ns=128 更好
# ============================================================
gpu6_worker() {
    wait_for_gpu_idle 6

    log "===== GPU6 nsamples=256 泛化验证启动 ====="

    # 最优 alpha 窄范围 + ns=256（最高优先级：ns=128 时是 7.828，256 会到多少？）
    run_exp 6 "ns256_rho0.5_a045_055" \
        --rho 0.5 --alpha_min 0.45 --alpha_max 0.55 --r1_ratio 0.9 --nsamples 256

    # lambda_psi=1e-5 叠加 ns=256（ns=128 时 lam1e5=7.826，256 能叠加吗？）
    run_exp 6 "ns256_rho0.5_a04_06_lam1e5" \
        --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --nsamples 256 --lambda_psi 1e-5

    # 最优 rho + ns=256（rho=0.42 在 ns=128 时是 7.829）
    run_exp 6 "ns256_rho0.42_a04_06" \
        --rho 0.42 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --nsamples 256

    # r1_ratio=0.95 + ns=256（组合两个已知改善点）
    run_exp 6 "ns256_rho0.5_a04_06_r1_0.95" \
        --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.95 --nsamples 256

    # 全部最优参数叠加 + ns=256（终极组合）
    run_exp 6 "ns256_rho0.42_a045_055_r1_0.95_lam1e5" \
        --rho 0.42 --alpha_min 0.45 --alpha_max 0.55 --r1_ratio 0.95 --nsamples 256 --lambda_psi 1e-5

    log "===== GPU6 全部完成 ====="
}

# ============================================================
# GPU7：ns=512 探索（5个）
# 探索更多样本是否继续提升，以及收益递减规律
# ============================================================
gpu7_worker() {
    wait_for_gpu_idle 7

    log "===== GPU7 nsamples=512 探索启动 ====="

    # ns=512 基线（与 ns=128/256 直接对比）
    run_exp 7 "ns512_rho0.5_a04_06" \
        --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --nsamples 512

    # ns=512 + 最优 alpha
    run_exp 7 "ns512_rho0.5_a045_055" \
        --rho 0.5 --alpha_min 0.45 --alpha_max 0.55 --r1_ratio 0.9 --nsamples 512

    # ns=512 + lambda_psi=1e-5
    run_exp 7 "ns512_rho0.5_a04_06_lam1e5" \
        --rho 0.5 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --nsamples 512 --lambda_psi 1e-5

    # ns=512 + rho=0.42（当前最优 rho）
    run_exp 7 "ns512_rho0.42_a04_06" \
        --rho 0.42 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.9 --nsamples 512

    # ns=512 全部最优叠加（如果以上有提升再做这个）
    run_exp 7 "ns512_rho0.42_a045_055_r1_0.95_lam1e5" \
        --rho 0.42 --alpha_min 0.45 --alpha_max 0.55 --r1_ratio 0.95 --nsamples 512 --lambda_psi 1e-5

    log "===== GPU7 全部完成 ====="
}

# ============================================================
# 主逻辑：GPU6/7 并行独立监控
# ============================================================
log "========================================"
log "nsamples 消融实验监控已启动"
log "  GPU6: ns=256 泛化验证  [5 exp]"
log "    - rho0.5_a045_055 + ns256"
log "    - rho0.5_a04_06 + lam1e5 + ns256"
log "    - rho0.42_a04_06 + ns256"
log "    - rho0.5_a04_06 + r1=0.95 + ns256"
log "    - 全部最优叠加 + ns256"
log "  GPU7: ns=512 探索      [5 exp]"
log "    - rho0.5_a04_06 + ns512 (基线对比)"
log "    - rho0.5_a045_055 + ns512"
log "    - rho0.5_a04_06 + lam1e5 + ns512"
log "    - rho0.42_a04_06 + ns512"
log "    - 全部最优叠加 + ns512"
log "  总计: 10 个实验，≈ 10~12 GPU·小时"
log "  注：ns=512 约比 ns=128 慢 2~3 倍"
log "========================================"

gpu6_worker &
GPU6_PID=$!

gpu7_worker &
GPU7_PID=$!

log "Worker PIDs: GPU6=${GPU6_PID} GPU7=${GPU7_PID}"

wait $GPU6_PID $GPU7_PID

log "========================================"
log "所有 nsamples 消融实验已完成！"
log ""
log "===== nsamples 消融结果汇总 ====="
cd "${BASE}" && \
for f in ns256_*.log ns512_*.log; do
    [ -f "$f" ] || continue
    PPL=$(grep -oP "wikitext2.*?\K[0-9]+\.[0-9]+" "$f" 2>/dev/null | tail -1)
    [ -n "$PPL" ] && echo "${PPL} | ${f%.log}" >> "${MONITOR_LOG}"
done
echo "--- ns=256 结果 ---" && grep "ns256" "${MONITOR_LOG}" | grep "^7\." | sort -n
echo "--- ns=512 结果 ---" && grep "ns512" "${MONITOR_LOG}" | grep "^7\." | sort -n
log "========================================="
