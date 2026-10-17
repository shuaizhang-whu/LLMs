#!/bin/bash
set -euo pipefail

# ============================================================
# FR-SVD LoRA GPU Monitor
#
# 功能:
#   1. 持续监控 GPU 0-7
#   2. 一旦发现空闲 GPU，立即启动 run_lora_finetuning.sh
#   3. 自动把发现的空闲 GPU 号传给 LoRA 脚本
#   4. 保存监控日志和任务日志
#
# 用法:
#   bash monitor_lora_finetuning.sh
#   nohup bash monitor_lora_finetuning.sh > /tmp/fr_svd_lora_monitor.out 2>&1 &
#
# 可选环境变量:
#   SCAN_INTERVAL   扫描间隔，默认 30 秒
#   UTIL_THRESH     空闲判定的 GPU 利用率阈值，默认 5 (%)
#   MEM_USED_THRESH 空闲判定的已用显存阈值，默认 1500 (MiB)
#   MIN_FREE_MEM    空闲判定的可用显存阈值，默认 20000 (MiB)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/lora/run_lora_finetuning.sh"
LOG_DIR="${PROJECT_ROOT}/result/FR_SVD_v1.4_lora_monitor"

SCAN_INTERVAL="${SCAN_INTERVAL:-30}"
UTIL_THRESH="${UTIL_THRESH:-5}"
MEM_USED_THRESH="${MEM_USED_THRESH:-1500}"
MIN_FREE_MEM="${MIN_FREE_MEM:-20000}"

mkdir -p "${LOG_DIR}"

MONITOR_LOG="${LOG_DIR}/monitor_$(date +%Y%m%d_%H%M%S).log"
JOB_LOG=""

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MONITOR_LOG}"
}

query_gpu_stats() {
    local gpu_id="$1"
    nvidia-smi --id="${gpu_id}" \
        --query-gpu=index,name,utilization.gpu,memory.used,memory.free \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

gpu_is_free() {
    local gpu_id="$1"
    local stats util mem_used mem_free

    stats="$(query_gpu_stats "${gpu_id}")"
    [[ -z "${stats}" ]] && return 1

    util="$(echo "${stats}" | cut -d',' -f3)"
    mem_used="$(echo "${stats}" | cut -d',' -f4)"
    mem_free="$(echo "${stats}" | cut -d',' -f5)"

    [[ "${util}" -le "${UTIL_THRESH}" ]] &&
    [[ "${mem_used}" -le "${MEM_USED_THRESH}" ]] &&
    [[ "${mem_free}" -ge "${MIN_FREE_MEM}" ]]
}

find_free_gpu() {
    local gpu_id
    for gpu_id in {0..7}; do
        if gpu_is_free "${gpu_id}"; then
            echo "${gpu_id}"
            return 0
        fi
    done
    return 1
}

launch_job() {
    local gpu_id="$1"
    JOB_LOG="${LOG_DIR}/lora_job_gpu${gpu_id}_$(date +%Y%m%d_%H%M%S).log"

    log "发现空闲 GPU${gpu_id}，立即启动 LoRA 微调脚本。"
    log "任务日志: ${JOB_LOG}"
    log "执行命令: bash ${RUN_SCRIPT} --gpu ${gpu_id}"

    bash "${RUN_SCRIPT}" --gpu "${gpu_id}" 2>&1 | tee -a "${JOB_LOG}"
    local exit_code=${PIPESTATUS[0]}

    if [[ "${exit_code}" -eq 0 ]]; then
        log "LoRA 微调任务在 GPU${gpu_id} 上执行完成。"
    else
        log "LoRA 微调任务在 GPU${gpu_id} 上退出，exit_code=${exit_code}。"
    fi

    return "${exit_code}"
}

log "============================================================"
log "FR-SVD LoRA GPU Monitor started"
log "监控范围: GPU 0-7"
log "扫描间隔: ${SCAN_INTERVAL}s"
log "空闲条件: util<=${UTIL_THRESH}% , mem_used<=${MEM_USED_THRESH}MiB , mem_free>=${MIN_FREE_MEM}MiB"
log "LoRA 脚本: ${RUN_SCRIPT}"
log "监控日志: ${MONITOR_LOG}"
log "============================================================"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    log "错误: 未找到运行脚本 ${RUN_SCRIPT}"
    exit 1
fi

while true; do
    found_gpu=""

    for gpu_id in {0..7}; do
        stats="$(query_gpu_stats "${gpu_id}")"
        if [[ -z "${stats}" ]]; then
            log "GPU${gpu_id}: unavailable"
            continue
        fi

        gpu_name="$(echo "${stats}" | cut -d',' -f2)"
        util="$(echo "${stats}" | cut -d',' -f3)"
        mem_used="$(echo "${stats}" | cut -d',' -f4)"
        mem_free="$(echo "${stats}" | cut -d',' -f5)"
        log "GPU${gpu_id}: ${gpu_name}, util=${util}%, mem_used=${mem_used}MiB, mem_free=${mem_free}MiB"

        if [[ "${util}" -le "${UTIL_THRESH}" ]] &&
           [[ "${mem_used}" -le "${MEM_USED_THRESH}" ]] &&
           [[ "${mem_free}" -ge "${MIN_FREE_MEM}" ]]; then
            found_gpu="${gpu_id}"
            break
        fi
    done

    if [[ -n "${found_gpu}" ]]; then
        launch_job "${found_gpu}"
        exit $?
    fi

    log "当前没有空闲 GPU，${SCAN_INTERVAL}s 后继续扫描。"
    sleep "${SCAN_INTERVAL}"
done
