#!/bin/bash
set -euo pipefail

# Benchmark LLaMA-7B compression/evaluation wall time for:
#   1. SVD-LLM
#   2. SAES-SVD v3
#   3. FR-SVD v1.5 ablations:
#      - Fisher only
#      - Innovation 2 / CEALC-aware Stage 2 only
#      - Fisher + Innovation 2
#
# All outputs are written under /data/zhangs/LLMs/Projects/FR-SVD/result.

LLAMA_MODEL="${LLAMA_MODEL:-/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf}"
PYTHON="${PYTHON:-/data/zhangs/anaconda3/envs/compress/bin/python}"

SVDLLM_DIR="/data/zhangs/LLMs/Projects/SVD-LLM"
SVDLLM_PY="${SVDLLM_DIR}/SVDLLM.py"
SAES_DIR="/data/zhangs/LLMs/Projects/SAES-SVD"
SAES_PY="${SAES_DIR}/SAES-SVD_v3.py"
FR_DIR="/data/zhangs/LLMs/Projects/FR-SVD"
FR_PY="${FR_DIR}/FR_SVD_v1.5.py"

GPU_A="${GPU_A:-3}"
GPU_B="${GPU_B:-6}"
SAES_GPU="${SAES_GPU:-$GPU_B}"
RATIO="${RATIO:-0.2}"
NSAMPLES="${NSAMPLES:-256}"
DATASET="${DATASET:-wikitext2}"
SEQ_LEN="${SEQ_LEN:-2048}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
RUN_ONLY="${RUN_ONLY:-all}"
FRESH_JOB_OUTPUT="${FRESH_JOB_OUTPUT:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${FR_DIR}/result/llama7b_time_benchmark_${TS}}"
SUMMARY="${OUT_ROOT}/summary.csv"
MAIN_LOG="${OUT_ROOT}/benchmark.log"
LOCK_FILE="${OUT_ROOT}/summary.lock"

mkdir -p "$OUT_ROOT"
if [[ "$RUN_ONLY" == "saes_svd_v3" ]]; then
    mkdir -p "${OUT_ROOT}/saes_svd_v3"
    MAIN_LOG="${OUT_ROOT}/saes_svd_v3/benchmark.log"
fi
SUMMARY_HEADER="method,variant,gpu,status,compress_seconds,eval_seconds,total_seconds,output_dir,model_path,compress_log,eval_log"
if [[ ! -f "$SUMMARY" ]]; then
    echo "$SUMMARY_HEADER" > "$SUMMARY"
fi

safe_model_name() {
    local s="$1"
    s="${s//\//_}"
    s="${s//-/_}"
    echo "$s"
}

log_main() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

append_summary() {
    local row="$1"
    flock "$LOCK_FILE" bash -c 'echo "$1" >> "$2"' _ "$row" "$SUMMARY"
}

archive_existing_job_output() {
    local out_dir="$1"
    local stamp="$2"

    for path in "$out_dir"/compress.log "$out_dir"/eval.log "$out_dir"/*.pt; do
        [[ -e "$path" ]] || continue
        mv "$path" "${path}.backup_${stamp}"
    done
}

run_logged() {
    local gpu="$1"
    local cwd="$2"
    local log_file="$3"
    shift 3

    mkdir -p "$(dirname "$log_file")"
    (
        cd "$cwd"
        echo "__BENCH_CMD_START_EPOCH=$(date +%s)"
        echo "__BENCH_CMD_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "__BENCH_CMD_CWD=$cwd"
        echo "__BENCH_CMD_GPU=$gpu"
        echo "__BENCH_CMD=$*"
        set +e
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$@" 2>&1
        rc=$?
        set -e
        echo "__BENCH_CMD_END_EPOCH=$(date +%s)"
        echo "__BENCH_CMD_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "__BENCH_CMD_EXIT_CODE=$rc"
        exit "$rc"
    ) >> "$log_file" 2>&1
}

first_epoch_value() {
    local key="$1"
    local log_file="$2"
    grep -m1 "$key" "$log_file" | sed -E 's/.*=([0-9]+).*/\1/' || true
}

last_epoch_value() {
    local key="$1"
    local log_file="$2"
    grep "$key" "$log_file" | tail -1 | sed -E 's/.*=([0-9]+).*/\1/' || true
}

run_auto_eval_job() {
    local method="$1"
    local variant="$2"
    local gpu="$3"
    local cwd="$4"
    local out_dir="$5"
    shift 5

    mkdir -p "$out_dir"
    local compress_log="${out_dir}/compress.log"
    local eval_log="${out_dir}/eval.log"
    local status="ok"

    if [[ "$FRESH_JOB_OUTPUT" == "1" ]]; then
        archive_existing_job_output "$out_dir" "$(date +%Y%m%d_%H%M%S)"
    fi

    log_main "START ${method}/${variant} compression on GPU${gpu}; output=${out_dir}"
    if ! run_logged "$gpu" "$cwd" "$compress_log" "$@"; then
        status="failed"
    fi

    local model_path
    model_path="$(find "$out_dir" -maxdepth 1 -type f -name '*.pt' ! -name '*profiling*' | head -1 || true)"

    if [[ "$status" == "ok" && -n "$model_path" && -f "$model_path" ]]; then
        log_main "START ${method}/${variant} evaluation on GPU${gpu}; model=${model_path}"
        if [[ "$method" == "saes_svd_v3" ]]; then
            if ! run_logged "$gpu" "$SAES_DIR" "$eval_log" \
                "$PYTHON" "$SAES_PY" \
                --eval_only \
                --model_path "$model_path" \
                --model_seq_len "$SEQ_LEN" \
                --eval_batch_size "$EVAL_BATCH_SIZE" \
                --DEV cuda:0; then
                status="failed"
            fi
        else
            if ! run_logged "$gpu" "$FR_DIR" "$eval_log" \
                "$PYTHON" "$FR_PY" \
                --eval_only \
                --model_path "$model_path" \
                --model_seq_len "$SEQ_LEN" \
                --eval_batch_size "$EVAL_BATCH_SIZE" \
                --DEV cuda:0; then
                status="failed"
            fi
        fi
    else
        status="failed"
        echo "Compressed model not found under ${out_dir}" >> "$eval_log"
    fi

    local c_start c_end e_start e_end compress_seconds="" eval_seconds="" total_seconds=""
    c_start="$(first_epoch_value "__BENCH_CMD_START_EPOCH" "$compress_log")"
    c_end="$(last_epoch_value "__BENCH_CMD_END_EPOCH" "$compress_log")"
    e_start="$(first_epoch_value "__BENCH_CMD_START_EPOCH" "$eval_log")"
    e_end="$(last_epoch_value "__BENCH_CMD_END_EPOCH" "$eval_log")"
    if [[ -n "$c_start" && -n "$c_end" ]]; then
        compress_seconds=$((c_end - c_start))
    fi
    if [[ -n "$e_start" && -n "$e_end" ]]; then
        eval_seconds=$((e_end - e_start))
    fi
    if [[ -n "$compress_seconds" && -n "$eval_seconds" ]]; then
        total_seconds=$((compress_seconds + eval_seconds))
    fi

    append_summary "${method},${variant},${gpu},${status},${compress_seconds},${eval_seconds},${total_seconds},${out_dir},${model_path},${compress_log},${eval_log}"
    log_main "END ${method}/${variant} on GPU${gpu}; status=${status}; compress=${compress_seconds}s eval=${eval_seconds}s total=${total_seconds}s"
}

run_svdllm_job() {
    local gpu="$1"
    local variant="whitening_only"
    local out_dir="${OUT_ROOT}/svdllm_${variant}"
    mkdir -p "$out_dir"

    local compress_log="${out_dir}/compress.log"
    local eval_log="${out_dir}/eval.log"
    local status="ok"

    log_main "START svdllm/${variant} compression on GPU${gpu}; output=${out_dir}"
    if ! run_logged "$gpu" "$SVDLLM_DIR" "$compress_log" \
        "$PYTHON" "$SVDLLM_PY" \
        --model "$LLAMA_MODEL" \
        --ratio "$RATIO" \
        --dataset "$DATASET" \
        --whitening_nsamples "$NSAMPLES" \
        --model_seq_len "$SEQ_LEN" \
        --save_path "$out_dir" \
        --DEV cuda:0 \
        --step 1; then
        status="failed"
    fi

    local safe_name model_path
    safe_name="$(safe_model_name "$LLAMA_MODEL")"
    model_path="${out_dir}/${safe_name}_whitening_only_0.8.pt"
    if [[ ! -f "$model_path" ]]; then
        model_path="$(find "$out_dir" -maxdepth 1 -type f -name '*_whitening_only_*.pt' | head -1 || true)"
    fi

    if [[ "$status" == "ok" && -n "$model_path" && -f "$model_path" ]]; then
        log_main "START svdllm/${variant} evaluation on GPU${gpu}; model=${model_path}"
        if ! run_logged "$gpu" "$SVDLLM_DIR" "$eval_log" \
            "$PYTHON" "$SVDLLM_PY" \
            --model "$LLAMA_MODEL" \
            --model_path "$model_path" \
            --ratio "$RATIO" \
            --dataset "$DATASET" \
            --model_seq_len "$SEQ_LEN" \
            --eval_batch_size "$EVAL_BATCH_SIZE" \
            --DEV cuda:0 \
            --step 4; then
            status="failed"
        fi
    else
        status="failed"
        echo "Compressed model not found: ${model_path}" | tee -a "$eval_log"
    fi

    local c_start c_end e_start e_end compress_seconds="" eval_seconds="" total_seconds=""
    c_start="$(first_epoch_value "__BENCH_CMD_START_EPOCH" "$compress_log")"
    c_end="$(last_epoch_value "__BENCH_CMD_END_EPOCH" "$compress_log")"
    e_start="$(first_epoch_value "__BENCH_CMD_START_EPOCH" "$eval_log")"
    e_end="$(last_epoch_value "__BENCH_CMD_END_EPOCH" "$eval_log")"
    if [[ -n "$c_start" && -n "$c_end" ]]; then
        compress_seconds=$((c_end - c_start))
    fi
    if [[ -n "$e_start" && -n "$e_end" ]]; then
        eval_seconds=$((e_end - e_start))
    fi
    if [[ -n "$compress_seconds" && -n "$eval_seconds" ]]; then
        total_seconds=$((compress_seconds + eval_seconds))
    fi

    append_summary "svdllm,${variant},${gpu},${status},${compress_seconds},${eval_seconds},${total_seconds},${out_dir},${model_path},${compress_log},${eval_log}"
    log_main "END svdllm/${variant} on GPU${gpu}; status=${status}; compress=${compress_seconds}s eval=${eval_seconds}s total=${total_seconds}s"
}

run_saes_job() {
    local gpu="$1"
    local out_dir="${OUT_ROOT}/saes_svd_v3"
    run_auto_eval_job "saes_svd_v3" "single_stage" "$gpu" "$SAES_DIR" "$out_dir" \
        "$PYTHON" "$SAES_PY" \
        --model "$LLAMA_MODEL" \
        --ratio "$RATIO" \
        --dataset "$DATASET" \
        --nsamples "$NSAMPLES" \
        --model_seq_len "$SEQ_LEN" \
        --rho 0.95 \
        --alpha_min 0.4 \
        --alpha_max 0.6 \
        --eps 1e-4 \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --no_eval_after_compress \
        --save_path "$out_dir" \
        --DEV cuda:0
}

run_fr_job() {
    local variant="$1"
    local gpu="$2"
    local r1_ratio="$3"
    local fisher_mode="$4"
    local out_dir="${OUT_ROOT}/fr_svd_${variant}"

    run_auto_eval_job "fr_svd_v1.5" "$variant" "$gpu" "$FR_DIR" "$out_dir" \
        "$PYTHON" "$FR_PY" \
        --model "$LLAMA_MODEL" \
        --ratio "$RATIO" \
        --dataset "$DATASET" \
        --nsamples "$NSAMPLES" \
        --model_seq_len "$SEQ_LEN" \
        --r1_ratio "$r1_ratio" \
        --fisher_mode "$fisher_mode" \
        --rho 1.0 \
        --alpha_min 0.25 \
        --alpha_max 0.75 \
        --eps 1e-4 \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --no_eval_after_compress \
        --save_path "$out_dir" \
        --DEV cuda:0
}

lane_gpu_a() {
    run_svdllm_job "$GPU_A"
    run_fr_job "fisher_only" "$GPU_A" "1.0" "soft"
    run_fr_job "fisher_plus_innovation2" "$GPU_A" "0.92" "soft"
}

lane_gpu_b() {
    run_saes_job "$GPU_B"
    run_fr_job "innovation2_only" "$GPU_B" "0.92" "none"
}

log_main "Benchmark output root: $OUT_ROOT"
log_main "Summary CSV: $SUMMARY"
log_main "GPU lane A: $GPU_A; GPU lane B: $GPU_B"
log_main "Model: $LLAMA_MODEL; ratio=$RATIO; nsamples=$NSAMPLES; eval_batch_size=$EVAL_BATCH_SIZE"
log_main "Run mode: $RUN_ONLY"

if [[ "$RUN_ONLY" == "saes_svd_v3" ]]; then
    log_main "Running only saes_svd_v3 on GPU${SAES_GPU}"
    run_saes_job "$SAES_GPU"
    log_main "SAES-SVD v3 rerun finished. Summary:"
    cat "$SUMMARY" | tee -a "$MAIN_LOG"
    exit 0
fi

if [[ "$RUN_ONLY" != "all" ]]; then
    echo "Unsupported RUN_ONLY=${RUN_ONLY}. Use 'all' or 'saes_svd_v3'." >&2
    exit 2
fi

lane_gpu_a &
PID_A=$!
lane_gpu_b &
PID_B=$!

log_main "Started lane A pid=$PID_A and lane B pid=$PID_B"

set +e
wait "$PID_A"
RC_A=$?
wait "$PID_B"
RC_B=$?
set -e

log_main "Lane A exit code: $RC_A"
log_main "Lane B exit code: $RC_B"
log_main "Benchmark finished. Summary:"
cat "$SUMMARY" | tee -a "$MAIN_LOG"

if [[ "$RC_A" -ne 0 || "$RC_B" -ne 0 ]]; then
    exit 1
fi
