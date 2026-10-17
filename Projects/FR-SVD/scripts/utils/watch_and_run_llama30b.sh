#!/bin/bash
# ============================================================
# LLaMA-30B GPU 监控 + 自动启动脚本
#
# 功能：每 60 秒扫描 GPU 0-7，发现显存使用 < 5 GB 的空闲 GPU
#       后预先创建结果目录，将监控日志和压缩日志均写入该目录
#
# 用法：
#   bash watch_and_run_llama30b.sh
#   nohup bash watch_and_run_llama30b.sh 2>&1 &
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCAN_INTERVAL=60
FREE_THRESH=5120   # MiB，< 此值视为空闲
GPUS=(0 1 2 3 4 5 6 7)

# 与 run_fr_svd_v1.5.sh 保持一致的目录公式（用于预先创建）
OUT_BASE="/data/zhangs/LLMs/Projects/FR-SVD/result/v1.5"
MODEL_TAG="llama_30b"
NSAMPLES=64
R1_RATIO=0.92
RHO=1.0

# 监控日志暂存到临时文件，找到空闲 GPU 后统一迁移到结果目录
TMP_LOG=$(mktemp /tmp/watch_llama30b_XXXXXX.log)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$TMP_LOG"
}

get_mem_used() {
    nvidia-smi --id="$1" \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

log "=== LLaMA-30B 监控启动 ==="
log "扫描 GPU: ${GPUS[*]}"
log "空闲阈值: < ${FREE_THRESH} MiB 已用"
log "扫描间隔: ${SCAN_INTERVAL}s"

while true; do
    log "--- 扫描 GPU 状态 ---"
    CHOSEN=""
    for gpu in "${GPUS[@]}"; do
        used=$(get_mem_used "$gpu")
        if [[ -z "$used" ]]; then
            log "  GPU ${gpu}: 获取状态失败，跳过"
            continue
        fi
        log "  GPU ${gpu}: ${used} MiB used"
        if [[ "$used" -lt "$FREE_THRESH" && -z "$CHOSEN" ]]; then
            CHOSEN="$gpu"
        fi
    done

    if [[ -n "$CHOSEN" ]]; then
        # 预先创建结果目录，与 run_fr_svd_v1.5.sh 公式一致
        TS=$(date +%Y%m%d_%H%M%S)
        OUT_DIR="${OUT_BASE}/${MODEL_TAG}/ns${NSAMPLES}_r1_${R1_RATIO}_rho${RHO}_${TS}"
        mkdir -p "$OUT_DIR"

        log "发现空闲 GPU ${CHOSEN}！结果目录: ${OUT_DIR}"
        log "迁移监控日志到结果目录..."

        # 将监控前置日志迁移进结果目录
        cp "$TMP_LOG" "${OUT_DIR}/watch.log"
        rm -f "$TMP_LOG"

        # 后续所有输出（监控 + 压缩）直接写入结果目录的 watch.log
        # run_fr_svd_v1.5.sh 会在同目录生成 run.log
        log() {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${OUT_DIR}/watch.log"
        }

        log "启动 LLaMA-30B 压缩 (GPU=${CHOSEN})..."
        bash "${PROJECT_ROOT}/scripts/compress/run_llama30b.sh" "${CHOSEN}" "${OUT_DIR}"

        log "压缩任务完成，退出监控。"
        exit 0
    else
        log "暂无空闲 GPU，${SCAN_INTERVAL}s 后重试..."
        sleep "$SCAN_INTERVAL"
    fi
done
