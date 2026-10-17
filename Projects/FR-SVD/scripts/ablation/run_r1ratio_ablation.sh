#!/bin/bash
# r1_ratio 消融实验 — 自动触发版本
# =========================================================
# 背景：发现 r1_ratio=0.8→PPL=8.142（极差），r1_ratio=0.9→7.837，0.95→7.831
#        需要系统消融找出"性能悬崖"位置，以及最优 r1:r2 分配比例
#
# 固定最优参数（当前全局最优 PPL=7.8299）:
#   --fisher_mode direct --rho 0.35 --alpha_min 0.4 --alpha_max 0.6
# 辅助验证参数（亚军 PPL=7.8303）:
#   --fisher_mode direct --rho 0.40 --alpha_min 0.35 --alpha_max 0.65
#
# GPU 分工：
#   GPU4: rho=0.35 + r1_ratio ∈ {0.60, 0.70, 0.75, 0.80, 0.82, 0.84}   (6 exp, 低范围)
#   GPU6: rho=0.35 + r1_ratio ∈ {0.86, 0.88, 0.90, 0.92, 0.95, 1.00}   (6 exp, 高范围)
#   GPU7: rho=0.40 + r1_ratio ∈ {0.80, 0.84, 0.86, 0.88, 0.90, 0.95}   (6 exp, 次优参数验证)
#
# 触发机制：各 GPU 独立监控，各自空闲满 10 分钟后立即启动本 GPU 的队列
# =========================================================

MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
BASE="/data/zhangs/LLMs/Projects/FR-SVD/result/phase2"
SCRIPT="/data/zhangs/LLMs/Projects/FR-SVD/FR_SVD_v1.3.py"
FD="--fisher_mode direct"
COM="--model ${MODEL} --ratio 0.2 --dataset wikitext2 --nsamples 128 --seed 3 --model_seq_len 2048 --DEV cuda:0"

MONITOR_LOG="${BASE}/r1ablation_monitor.log"
IDLE_MB=500          # 低于此值视为空闲
IDLE_CHECKS=10       # 连续 10 次（每次 60s）= 10 分钟
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
# GPU4 Worker: rho=0.35 低范围 r1_ratio（找悬崖下方）
# r1_ratio: 0.60 / 0.70 / 0.75 / 0.80 / 0.82 / 0.84
# ============================================================
gpu4_worker() {
    wait_for_gpu_idle 4

    log "===== GPU4 r1_ratio 消融（rho=0.35，低范围）启动 ====="

    run_exp 4 "r1abl_rho035_a04_06_r1_0.60" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.60

    run_exp 4 "r1abl_rho035_a04_06_r1_0.70" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.70

    run_exp 4 "r1abl_rho035_a04_06_r1_0.75" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.75

    run_exp 4 "r1abl_rho035_a04_06_r1_0.80" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.80

    run_exp 4 "r1abl_rho035_a04_06_r1_0.82" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.82

    run_exp 4 "r1abl_rho035_a04_06_r1_0.84" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.84

    log "===== GPU4 全部完成 ====="
}

# ============================================================
# GPU6 Worker: rho=0.35 高范围 r1_ratio（找最优点）
# r1_ratio: 0.86 / 0.88 / 0.90 / 0.92 / 0.95 / 1.00
# ============================================================
gpu6_worker() {
    wait_for_gpu_idle 6

    log "===== GPU6 r1_ratio 消融（rho=0.35，高范围）启动 ====="

    run_exp 6 "r1abl_rho035_a04_06_r1_0.86" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.86

    run_exp 6 "r1abl_rho035_a04_06_r1_0.88" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.88

    run_exp 6 "r1abl_rho035_a04_06_r1_0.90" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.90

    run_exp 6 "r1abl_rho035_a04_06_r1_0.92" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.92

    run_exp 6 "r1abl_rho035_a04_06_r1_0.95" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 0.95

    run_exp 6 "r1abl_rho035_a04_06_r1_1.00" \
        --rho 0.35 --alpha_min 0.4 --alpha_max 0.6 --r1_ratio 1.00

    log "===== GPU6 全部完成 ====="
}

# ============================================================
# GPU7 Worker: rho=0.40 辅助验证（alpha=[0.35,0.65]）
# 在"悬崖区"附近密集采样，确认最优 r1_ratio 是参数无关的
# r1_ratio: 0.80 / 0.84 / 0.86 / 0.88 / 0.90 / 0.95
# ============================================================
gpu7_worker() {
    wait_for_gpu_idle 7

    log "===== GPU7 r1_ratio 消融（rho=0.40，次优参数验证）启动 ====="

    run_exp 7 "r1abl_rho04_a035_065_r1_0.80" \
        --rho 0.40 --alpha_min 0.35 --alpha_max 0.65 --r1_ratio 0.80

    run_exp 7 "r1abl_rho04_a035_065_r1_0.84" \
        --rho 0.40 --alpha_min 0.35 --alpha_max 0.65 --r1_ratio 0.84

    run_exp 7 "r1abl_rho04_a035_065_r1_0.86" \
        --rho 0.40 --alpha_min 0.35 --alpha_max 0.65 --r1_ratio 0.86

    run_exp 7 "r1abl_rho04_a035_065_r1_0.88" \
        --rho 0.40 --alpha_min 0.35 --alpha_max 0.65 --r1_ratio 0.88

    run_exp 7 "r1abl_rho04_a035_065_r1_0.90" \
        --rho 0.40 --alpha_min 0.35 --alpha_max 0.65 --r1_ratio 0.90

    run_exp 7 "r1abl_rho04_a035_065_r1_0.95" \
        --rho 0.40 --alpha_min 0.35 --alpha_max 0.65 --r1_ratio 0.95

    log "===== GPU7 全部完成 ====="
}

# ============================================================
# 主逻辑：3 个 GPU 并行独立监控，各自就绪后立即启动
# ============================================================
log "========================================"
log "r1_ratio 消融实验监控已启动"
log "  GPU4: rho=0.35, r1_ratio={0.60~0.84}  [6 exp]"
log "  GPU6: rho=0.35, r1_ratio={0.86~1.00}  [6 exp]"
log "  GPU7: rho=0.40, r1_ratio={0.80~0.95}  [6 exp]"
log "  总计: 18 个实验，≈ 15 GPU·小时"
log "========================================"

gpu4_worker &
GPU4_PID=$!

gpu6_worker &
GPU6_PID=$!

gpu7_worker &
GPU7_PID=$!

log "Worker PIDs: GPU4=${GPU4_PID} GPU6=${GPU6_PID} GPU7=${GPU7_PID}"

wait $GPU4_PID $GPU6_PID $GPU7_PID
log "========================================"
log "所有 r1_ratio 消融实验已完成！"
log ""

# ============================================================
# 自动汇总：打印当前所有 r1abl 实验结果，按 PPL 排序
# ============================================================
log "===== r1_ratio 消融结果汇总 ====="
cd "${BASE}" && \
for f in r1abl_*.log; do
    PPL=$(grep -oP "wikitext2.*?\K[0-9]+\.[0-9]+" "$f" 2>/dev/null | tail -1)
    [ -n "$PPL" ] && echo "${PPL} | ${f%.log}" >> "${MONITOR_LOG}"
done
grep "^7\." "${MONITOR_LOG}" | sort -n | tee -a "${MONITOR_LOG}"
log "========================================="
