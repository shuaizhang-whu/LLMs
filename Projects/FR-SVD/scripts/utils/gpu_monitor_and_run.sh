#!/bin/bash
# ============================================================
# FR-SVD GPU Monitor + Auto-Launch + Auto-Fix
#
# 功能:
#   Phase 1: 每300秒扫描 GPU 0-7，找到空闲 GPU 自动启动
#   Phase 2: 运行压缩任务，捕获日志
#   Phase 3: 失败时调用 claude CLI 自动修复，最多3次重试
#
# 使用:
#   bash gpu_monitor_and_run.sh [--gpu 5] [--ratio 0.2] ...
#   nohup bash gpu_monitor_and_run.sh > monitor.log 2>&1 &
# ============================================================

# ==================== 可修改参数 ====================

UTIL_THRESH=10          # GPU-Util 空闲阈值 (%)
MEM_THRESH=4000         # Memory-Used 空闲阈值 (MiB)
SCAN_INTERVAL=300       # 扫描间隔 (秒)
MAX_FIX_RETRIES=3       # 自动修复最大重试次数

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FR_SVD_SCRIPT="${PROJECT_ROOT}/FR_SVD_v1.0.py"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/compress/run_fr_svd.sh"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
CLAUDE="claude"

# 结果目录（与 run_fr_svd.sh 保持一致）
BASE_OUTPUT="${PROJECT_ROOT}/result/FR_SVD_v1"
mkdir -p "$BASE_OUTPUT"

MONITOR_LOG="$BASE_OUTPUT/monitor_$(date +%Y%m%d_%H%M%S).log"

# 透传给 run_fr_svd.sh 的额外参数（命令行传入）
EXTRA_ARGS=("$@")

# ==================== 工具函数 ====================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

# 获取指定 GPU 的 Util% 和 Memory-Used(MiB)
get_gpu_stats() {
    local gpu_id=$1
    nvidia-smi --id="$gpu_id" \
        --query-gpu=utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' '
}

# 检查 GPU 是否空闲
is_gpu_free() {
    local gpu_id=$1
    local stats
    stats=$(get_gpu_stats "$gpu_id")
    [ -z "$stats" ] && return 1

    local util mem
    util=$(echo "$stats" | cut -d',' -f1)
    mem=$(echo  "$stats" | cut -d',' -f2)

    [ "$util" -lt "$UTIL_THRESH" ] && [ "$mem" -lt "$MEM_THRESH" ]
}

# 从日志提取 Python traceback（最后40行里含 Error/Traceback 的部分）
extract_error() {
    local logfile=$1
    tail -80 "$logfile" | grep -A 40 -E "(Traceback|Error|Exception)" | head -60
}

# ==================== Phase 1: GPU 监控 ====================

log "============================================================"
log "FR-SVD GPU Monitor started"
log "  Scan interval : ${SCAN_INTERVAL}s"
log "  Free criteria : Util < ${UTIL_THRESH}%, Mem < ${MEM_THRESH} MiB"
log "  GPUs monitored: 0-7"
log "  Monitor log   : $MONITOR_LOG"
log "============================================================"

FREE_GPU=""

while true; do
    log "Scanning GPUs 0-7 ..."
    for gpu_id in {0..7}; do
        stats=$(get_gpu_stats "$gpu_id")
        if [ -z "$stats" ]; then
            log "  GPU $gpu_id: [unavailable]"
            continue
        fi
        util=$(echo "$stats" | cut -d',' -f1)
        mem=$(echo  "$stats" | cut -d',' -f2)
        log "  GPU $gpu_id: Util=${util}%  Mem=${mem}MiB"

        if [ "$util" -lt "$UTIL_THRESH" ] && [ "$mem" -lt "$MEM_THRESH" ]; then
            FREE_GPU=$gpu_id
            log "  >>> GPU $gpu_id is FREE — launching job"
            break
        fi
    done

    if [ -n "$FREE_GPU" ]; then
        break
    fi

    log "No free GPU found. Next scan in ${SCAN_INTERVAL}s ..."
    sleep "$SCAN_INTERVAL"
done

# ==================== Phase 2 & 3: 启动 + 自动修复 ====================

DATE_STAMP=$(date +%Y%m%d_%H%M%S)
JOB_LOG="$BASE_OUTPUT/job_gpu${FREE_GPU}_${DATE_STAMP}.log"

attempt=0
success=false

while [ $attempt -le $MAX_FIX_RETRIES ]; do
    if [ $attempt -eq 0 ]; then
        log "------------------------------------------------------------"
        log "Launching FR-SVD on GPU $FREE_GPU (attempt 1)"
    else
        log "------------------------------------------------------------"
        log "Re-launching after fix (attempt $((attempt + 1))/$((MAX_FIX_RETRIES + 1)))"
    fi
    log "Job log: $JOB_LOG"

    # 运行压缩脚本，GPU 和输出目录通过参数传入
    bash "$RUN_SCRIPT" \
        --gpu "$FREE_GPU" \
        --output_dir "$BASE_OUTPUT" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$JOB_LOG"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        log "============================================================"
        log "Job completed successfully on GPU $FREE_GPU"
        log "============================================================"
        success=true
        break
    fi

    log "Job FAILED (exit code $EXIT_CODE)"

    attempt=$((attempt + 1))
    if [ $attempt -gt $MAX_FIX_RETRIES ]; then
        log "Max fix retries ($MAX_FIX_RETRIES) reached. Giving up."
        break
    fi

    # ────────── Phase 3: 调用 claude CLI 自动修复 ──────────
    log "Extracting error from log ..."
    ERROR_SNIPPET=$(extract_error "$JOB_LOG")

    if [ -z "$ERROR_SNIPPET" ]; then
        log "Could not extract error traceback. Skipping auto-fix."
        break
    fi

    log "Error snippet:"
    echo "$ERROR_SNIPPET" | tee -a "$MONITOR_LOG"

    log "Calling claude CLI to fix FR_SVD.py (fix attempt $attempt/$MAX_FIX_RETRIES) ..."

    FIX_PROMPT="I am running FR_SVD.py (path: $FR_SVD_SCRIPT) and it crashed with the following error. Please read the file, identify the bug, and fix it. Only fix the actual bug — do not change logic unrelated to the error.

Error output:
$ERROR_SNIPPET"

    $CLAUDE -p "$FIX_PROMPT" 2>&1 | tee -a "$MONITOR_LOG"
    FIX_EXIT=${PIPESTATUS[0]}

    if [ $FIX_EXIT -ne 0 ]; then
        log "claude CLI returned non-zero ($FIX_EXIT). Stopping."
        break
    fi

    log "Fix applied. Retrying job ..."
    # 追加新的 job log section
    echo "" >> "$JOB_LOG"
    echo "======= Retry attempt $attempt after auto-fix =======" >> "$JOB_LOG"
done

if [ "$success" = false ]; then
    log "============================================================"
    log "FAILED after all attempts. Check job log: $JOB_LOG"
    log "============================================================"
    exit 1
fi