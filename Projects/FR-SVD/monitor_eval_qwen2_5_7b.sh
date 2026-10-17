#!/bin/bash
set -euo pipefail

# 监控 GPU 0-7，发现空闲 GPU 后启动一次 Qwen2.5-7B 评估脚本。
#
# 可选环境变量:
#   SCAN_INTERVAL   扫描间隔，默认 30 秒
#   UTIL_THRESH     空闲判定的 GPU 利用率阈值，默认 5 (%)
#   MEM_USED_THRESH 空闲判定的已用显存阈值，默认 1500 (MiB)
#   MIN_FREE_MEM    空闲判定的可用显存阈值，默认 30000 (MiB)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_eval_qwen2_5_7b.sh"
RESULT_DIR="${SCRIPT_DIR}/result/FR_SVD_v1.5/qwen2_5_7b/ns256_r1_0.92_rho1.0_20260417_211008"

SCAN_INTERVAL="${SCAN_INTERVAL:-30}"
UTIL_THRESH="${UTIL_THRESH:-5}"
MEM_USED_THRESH="${MEM_USED_THRESH:-1500}"
MIN_FREE_MEM="${MIN_FREE_MEM:-30000}"

mkdir -p "$RESULT_DIR"

MONITOR_LOG="${RESULT_DIR}/monitor_eval_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

query_gpu_stats() {
    local gpu_id="$1"
    local output line idx name util mem_used mem_free

    if ! output="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null)"; then
        return 1
    fi

    while IFS= read -r line; do
        IFS=',' read -r idx name util mem_used mem_free <<< "$line"
        idx="${idx// /}"
        if [[ "$idx" == "$gpu_id" ]]; then
            util="${util// /}"
            mem_used="${mem_used// /}"
            mem_free="${mem_free// /}"
            echo "${idx},${name},${util},${mem_used},${mem_free}"
            return 0
        fi
    done <<< "$output"

    return 1
}

log "============================================================"
log "Qwen2.5-7B eval GPU monitor started"
log "监控范围: GPU 0-7"
log "扫描间隔: ${SCAN_INTERVAL}s"
log "空闲条件: util<=${UTIL_THRESH}% , mem_used<=${MEM_USED_THRESH}MiB , mem_free>=${MIN_FREE_MEM}MiB"
log "评估脚本: ${RUN_SCRIPT}"
log "监控日志: ${MONITOR_LOG}"
log "评估日志: ${RESULT_DIR}/eval.log"
log "============================================================"

if [[ ! -f "$RUN_SCRIPT" ]]; then
    log "错误: 未找到评估脚本 ${RUN_SCRIPT}"
    exit 1
fi

while true; do
    found_gpu=""

    for gpu_id in {0..7}; do
        stats="$(query_gpu_stats "$gpu_id")"
        if [[ -z "$stats" ]]; then
            log "GPU${gpu_id}: unavailable"
            continue
        fi

        IFS=',' read -r _idx gpu_name util mem_used mem_free <<< "$stats"
        log "GPU${gpu_id}: ${gpu_name}, util=${util}%, mem_used=${mem_used}MiB, mem_free=${mem_free}MiB"

        if [[ "$util" -le "$UTIL_THRESH" ]] &&
           [[ "$mem_used" -le "$MEM_USED_THRESH" ]] &&
           [[ "$mem_free" -ge "$MIN_FREE_MEM" ]]; then
            found_gpu="$gpu_id"
            break
        fi
    done

    if [[ -n "$found_gpu" ]]; then
        log "发现空闲 GPU${found_gpu}，启动 Qwen2.5-7B 评估。"
        log "执行命令: bash ${RUN_SCRIPT} ${found_gpu}"
        bash "$RUN_SCRIPT" "$found_gpu"
        exit_code=$?
        log "评估脚本退出，exit_code=${exit_code}。"
        exit "$exit_code"
    fi

    log "当前没有空闲 GPU，${SCAN_INTERVAL}s 后继续扫描。"
    sleep "$SCAN_INTERVAL"
done
