#!/bin/bash
# Ratio 0.4 / 0.6 with r1 ablation (2 parameter groups x 5 r1 values)
# Dynamic scheduler on GPU0-7: when any GPU becomes idle, it pulls next task.
# Total experiments: 2 ratios * 2 groups * 5 r1 values = 20

set -u

MODEL="/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"
PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT="/data/zhangs/LLMs/Projects/FR-SVD/FR_SVD_v1.3.py"
BASE="/data/zhangs/LLMs/Projects/FR-SVD/result/phase2"

MONITOR_LOG="${BASE}/ratio46_r1_monitor_allgpu.log"
SUMMARY_TSV="${BASE}/ratio46_r1_summary_allgpu.tsv"
QUEUE_FILE="${BASE}/ratio46_r1_queue.txt"
QUEUE_LOCK="${BASE}/ratio46_r1_queue.lock"

COMMON_ARGS=(
  --model "${MODEL}"
  --dataset wikitext2
  --seed 3
  --model_seq_len 2048
  --DEV cuda:0
  --fisher_mode direct
)

# Monitor all GPUs
TARGET_GPUS=(0 1 2 3 4 5 6 7)
IDLE_MB=500
IDLE_CHECKS=10
CHECK_INTERVAL=60

# Experiment matrix
RATIOS=(0.4 0.6)
R1_VALUES=(0.88 0.90 0.92 0.94 0.95)

# 2 strongest parameter groups selected from existing best results
GROUP_NAMES=(
  "g1_rho042_a045055_l1e5_ns512"
  "g2_rho05_a04_06_l1e5_ns512"
)
GROUP_RHO=(0.42 0.5)
GROUP_ALPHA_MIN=(0.45 0.4)
GROUP_ALPHA_MAX=(0.55 0.6)
GROUP_LAMBDA=(1e-5 1e-5)
GROUP_NS=(512 512)

mkdir -p "${BASE}"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "${MONITOR_LOG}"
}

ppl_from_log() {
  local f="$1"
  grep -oP "wikitext2.*?\K[0-9]+\.[0-9]+" "$f" 2>/dev/null | tail -1
}

wait_for_gpu_idle() {
  local gpu="$1"
  local count=0

  log "GPU${gpu}: waiting for idle (${IDLE_CHECKS} checks with memory < ${IDLE_MB}MB)"
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')

    if [[ -n "$used" && "$used" -lt "$IDLE_MB" ]]; then
      count=$((count + 1))
      log "GPU${gpu}: idle ${count}/${IDLE_CHECKS} (${used}MB)"
      if [[ "$count" -ge "$IDLE_CHECKS" ]]; then
        log "GPU${gpu}: confirmed idle"
        return 0
      fi
    else
      if [[ "$count" -gt 0 ]]; then
        log "GPU${gpu}: busy again (${used:-NA}MB), reset counter"
      fi
      count=0
    fi

    sleep "$CHECK_INTERVAL"
  done
}

run_exp() {
  local gpu="$1"
  local tag="$2"
  shift 2

  local log_file="${BASE}/${tag}.log"
  local out_dir="${BASE}/${tag}"

  # Skip already finished experiments
  if [[ -f "$log_file" ]]; then
    local existing
    existing=$(ppl_from_log "$log_file")
    if [[ -n "$existing" ]]; then
      log "GPU${gpu}: skip ${tag} (already done, PPL=${existing})"
      return 0
    fi
  fi

  mkdir -p "$out_dir"
  log "GPU${gpu}: start ${tag}"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --result_dir "$out_dir" > "$log_file" 2>&1

  local ppl
  ppl=$(ppl_from_log "$log_file")
  log "GPU${gpu}: finish ${tag} PPL=${ppl:-N/A}"
}

build_tag() {
  local ratio="$1"
  local gname="$2"
  local r1="$3"
  local ratio_tag

  ratio_tag=${ratio/./}
  echo "ratio${ratio_tag}_${gname}_r1_${r1}"
}

init_queue() {
  : > "$QUEUE_FILE"
  : > "$QUEUE_LOCK"

  local ratio g r1
  for ratio in "${RATIOS[@]}"; do
    for g in "${!GROUP_NAMES[@]}"; do
      for r1 in "${R1_VALUES[@]}"; do
        # ratio|group_index|r1
        echo "${ratio}|${g}|${r1}" >> "$QUEUE_FILE"
      done
    done
  done
}

pop_task() {
  local task=""

  exec 200<>"$QUEUE_LOCK"
  flock -x 200
  if [[ -s "$QUEUE_FILE" ]]; then
    task=$(head -n 1 "$QUEUE_FILE")
    tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp"
    mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
  fi
  flock -u 200
  exec 200>&-

  echo "$task"
}

worker() {
  local gpu="$1"
  log "GPU${gpu}: worker started"

  while true; do
    wait_for_gpu_idle "$gpu"

    local task
    task=$(pop_task)
    if [[ -z "$task" ]]; then
      log "GPU${gpu}: no remaining task, worker exit"
      break
    fi

    IFS='|' read -r ratio g_idx r1 <<< "$task"
    local gname="${GROUP_NAMES[$g_idx]}"
    local tag
    tag=$(build_tag "$ratio" "$gname" "$r1")

    run_exp "$gpu" "$tag" \
      --ratio "$ratio" \
      --rho "${GROUP_RHO[$g_idx]}" \
      --alpha_min "${GROUP_ALPHA_MIN[$g_idx]}" \
      --alpha_max "${GROUP_ALPHA_MAX[$g_idx]}" \
      --lambda_psi "${GROUP_LAMBDA[$g_idx]}" \
      --nsamples "${GROUP_NS[$g_idx]}" \
      --r1_ratio "$r1"
  done
}

write_summary() {
  : > "$SUMMARY_TSV"
  echo -e "ratio\tgroup\tr1_ratio\tppl\ttag" >> "$SUMMARY_TSV"

  local ratio g r1
  for ratio in "${RATIOS[@]}"; do
    for g in "${!GROUP_NAMES[@]}"; do
      for r1 in "${R1_VALUES[@]}"; do
        local tag
        tag=$(build_tag "$ratio" "${GROUP_NAMES[$g]}" "$r1")
        local ppl
        ppl=$(ppl_from_log "${BASE}/${tag}.log")
        [[ -z "$ppl" ]] && ppl="NA"
        echo -e "${ratio}\t${GROUP_NAMES[$g]}\t${r1}\t${ppl}\t${tag}" >> "$SUMMARY_TSV"
      done
    done
  done

  log "summary written: ${SUMMARY_TSV}"

  # Best r1 per (ratio, group)
  log "best r1 per (ratio, group):"
  awk -F'\t' '
    NR>1 && $4!="NA" {
      key=$1"|"$2
      if (!(key in best) || $4+0 < best[key]+0) {
        best[key]=$4
        best_r1[key]=$3
      }
    }
    END {
      for (k in best) {
        split(k, a, "|")
        printf("ratio=%s group=%s best_r1=%s best_ppl=%s\n", a[1], a[2], best_r1[k], best[k])
      }
    }
  ' "$SUMMARY_TSV" | sort | while read -r line; do log "$line"; done
}

init_queue

log "============================================================"
log "Start ratio(0.4/0.6) x 2groups x r1(0.88,0.90,0.92,0.94,0.95)"
log "Total experiments: 20"
log "Dynamic scheduler GPUs: ${TARGET_GPUS[*]}"
log "Queue file: ${QUEUE_FILE}"
log "Monitor log: ${MONITOR_LOG}"
log "============================================================"

PIDS=()
for gpu in "${TARGET_GPUS[@]}"; do
  worker "$gpu" &
  PIDS+=("$!")
done

log "worker pids: ${PIDS[*]}"

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

write_summary
log "All ratio/r1 ablation experiments completed."
