#!/bin/bash
set -euo pipefail
# ================================
# FR-SVD v1.5：多模型压缩运行脚本
# ================================
# 支持：
#   - LLaMA-7B / Llama-2-7b / Llama-13b / Llama-30b
#   - Qwen2.5-7B（GQA + q/k/v bias）
#   - Qwen3.5-35B-A3B（混合注意力 + MoE，建议加 --skip_moe_experts）
#
# 用法：
#   bash run_fr_svd_v1.5.sh                               # 默认 LLaMA-7B
#   bash run_fr_svd_v1.5.sh --model_tag llama2_7b         # 按预设选择模型
#   bash run_fr_svd_v1.5.sh --model_tag qwen2_5_7b --gpu 4
#   bash run_fr_svd_v1.5.sh --model_tag qwen3_moe --skip_moe_experts
# ================================

# ========== 可修改参数 ==========
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/data/zhangshuai/LLMs}"
GPU="${GPU:-1}"
MODEL_TAG="${MODEL_TAG:-llama_7b}"
RATIO="${RATIO:-0.2}"
NSAMPLES=""   # 留空：由模型预设自动设置
DATASET="${DATASET:-wikitext2}"
SEQ_LEN="${SEQ_LEN:-2048}"
R1_RATIO="${R1_RATIO:-1.0}"
ALPHA_MIN="${ALPHA_MIN:-0.25}"
ALPHA_MAX="${ALPHA_MAX:-0.75}"
RHO="${RHO:-1.0}"
EPS="${EPS:-1e-4}"
FISHER_MODE="${FISHER_MODE:-none}"
LAMBDA_PSI="${LAMBDA_PSI:-1e-5}"
SKIP_MOE_EXPERTS="${SKIP_MOE_EXPERTS:-0}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/FR-SVD/result/FR_SVD_v1.5}"

# ========== 命令行参数覆盖 ==========
_NSAMPLES_SET=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)              GPU="$2"; shift 2;;
        --model_tag)        MODEL_TAG="$2"; shift 2;;
        --ratio)            RATIO="$2"; shift 2;;
        --nsamples)         NSAMPLES="$2"; _NSAMPLES_SET=1; shift 2;;
        --dataset)          DATASET="$2"; shift 2;;
        --seq_len)          SEQ_LEN="$2"; shift 2;;
        --r1_ratio)         R1_RATIO="$2"; shift 2;;
        --alpha_min)        ALPHA_MIN="$2"; shift 2;;
        --alpha_max)        ALPHA_MAX="$2"; shift 2;;
        --rho)              RHO="$2"; shift 2;;
        --eps)              EPS="$2"; shift 2;;
        --fisher_mode)      FISHER_MODE="$2"; shift 2;;
        --lambda_psi)       LAMBDA_PSI="$2"; shift 2;;
        --skip_moe_experts) SKIP_MOE_EXPERTS=1; shift 1;;
        --out_base)         OUT_BASE="$2"; shift 2;;
        --out_dir)          _OUT_DIR_OVERRIDE="$2"; shift 2;;
        --model)            MODEL_OVERRIDE="$2"; shift 2;;
        *) EXTRA_ARGS+=("$1"); shift 1;;
    esac
done

# ========== 模型预设 ==========
# 每个模型标签设置路径和推荐 NSAMPLES（用户 --nsamples 可覆盖）
#
# A40 (46 GB VRAM) 显存预算分析：
#   逐层压缩只需 1 个 student 层 (fp32) + 1 个 teacher 层 (fp16) 在 GPU 上，
#   其余参数在 CPU。瓶颈是 CPU 内存（两份模型 + 子层 inp_cache）。
#
#   模型          两份模型(CPU)  推荐 NSAMPLES  子层 cache    CPU 合计
#   ─────────────────────────────────────────────────────────────────
#   LLaMA-7B      ~28 GB        256            ~20 GB        ~48 GB
#   Llama-2-7b    ~28 GB        256            ~20 GB        ~48 GB
#   LLaMA-13b     ~52 GB        256            ~36 GB        ~88 GB
#   LLaMA-30b     ~120 GB       128            ~20 GB        ~140 GB
#   Qwen2.5-7B    ~28 GB        256            ~26 GB        ~54 GB
#   Qwen3.5-35B   ~140 GB       128            ~8 GB         ~148 GB
case "$MODEL_TAG" in
    llama_7b)
        MODEL="${PROJECT_ROOT}/Models/llama-7b"
        _DEFAULT_NSAMPLES=256
        ;;
    llama2_7b)
        MODEL="${PROJECT_ROOT}/Models/Llama-2-7b"
        _DEFAULT_NSAMPLES=256
        ;;
    llama_13b)
        MODEL="${PROJECT_ROOT}/Models/llama-13b"
        _DEFAULT_NSAMPLES=256
        ;;
    llama_30b)
        MODEL="${PROJECT_ROOT}/Models/llama-30b"
        _DEFAULT_NSAMPLES=128
        ;;
    qwen2_5_7b)
        MODEL="${PROJECT_ROOT}/Models/Qwen2.5-7B"
        _DEFAULT_NSAMPLES=256
        ;;
    qwen3_moe)
        MODEL="${PROJECT_ROOT}/Models/Qwen3.5-35B-A3B"
        _DEFAULT_NSAMPLES=128
        # MoE experts 使用 packed 参数张量（非 nn.Linear），find_layers 不会发现，
        # 实际只压缩 attention + shared_expert，skip_moe_experts 标志作为安全护栏保留
        SKIP_MOE_EXPERTS=1
        ;;
    *)
        echo "Unknown MODEL_TAG: $MODEL_TAG" >&2
        echo "支持: llama_7b, llama2_7b, llama_13b, llama_30b, qwen2_5_7b, qwen3_moe" >&2
        exit 1
        ;;
esac

# 允许直接覆盖模型路径
if [[ -n "${MODEL_OVERRIDE:-}" ]]; then
    MODEL="$MODEL_OVERRIDE"
fi

# 如果用户未通过 --nsamples 显式指定，使用模型推荐值
if [[ "$_NSAMPLES_SET" == "0" ]]; then
    NSAMPLES="${_DEFAULT_NSAMPLES:-256}"
fi

# Qwen 模型用 compress_Qwen3.5 环境；LLaMA 系列用 compress 环境（旧版 transformers，
# 与 sentencepiece-only tokenizer 兼容，无 v5 递归 bug）
case "$MODEL_TAG" in
    qwen*)
        PYTHON="${PYTHON:-/mnt/data/zhangshuai/anaconda3/envs/compress_Qwen3.5/bin/python}"
        ;;
    *)
        PYTHON="${PYTHON:-/mnt/data/zhangshuai/anaconda3/envs/compress/bin/python}"
        ;;
esac
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON" >&2; exit 1
fi
if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: Model directory not found at $MODEL" >&2; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FR_SVD_PY="${SCRIPT_DIR}/FR_SVD_v1.5.py"

if [[ -n "${_OUT_DIR_OVERRIDE:-}" ]]; then
    OUT_DIR="$_OUT_DIR_OVERRIDE"
else
    TS=$(date +%Y%m%d_%H%M%S)
    OUT_DIR="${OUT_BASE}/${MODEL_TAG}/ns${NSAMPLES}_r1_${R1_RATIO}_rho${RHO}_${TS}"
fi
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "FR-SVD v1.5: Multi-Model Compression"
echo "=========================================="
echo "PROJECT_ROOT:      $PROJECT_ROOT"
echo "GPU:              $GPU"
echo "MODEL_TAG:        $MODEL_TAG"
echo "MODEL:            $MODEL"
echo "PYTHON:           $PYTHON"
echo "Ratio:            $RATIO (keep $(python3 -c "print(1-$RATIO)"))"
echo "Nsamples:         $NSAMPLES"
echo "Dataset:          $DATASET"
echo "SeqLen:           $SEQ_LEN"
echo "r1_ratio:         $R1_RATIO"
echo "Alpha:            [$ALPHA_MIN, $ALPHA_MAX], rho=$RHO"
echo "eps:              $EPS"
echo "Fisher:           $FISHER_MODE (lambda_psi=$LAMBDA_PSI)"
echo "Skip MoE experts: $SKIP_MOE_EXPERTS"
echo "Output:           $OUT_DIR"
echo "=========================================="

MOE_ARGS=()
if [[ "$SKIP_MOE_EXPERTS" == "1" ]]; then
    MOE_ARGS+=(--skip_moe_experts)
fi

CMD=(
    "$PYTHON" "$FR_SVD_PY"
    --model "$MODEL"
    --ratio "$RATIO"
    --nsamples "$NSAMPLES"
    --dataset "$DATASET"
    --model_seq_len "$SEQ_LEN"
    --r1_ratio "$R1_RATIO"
    --alpha_min "$ALPHA_MIN"
    --alpha_max "$ALPHA_MAX"
    --rho "$RHO"
    --eps "$EPS"
    --fisher_mode "$FISHER_MODE"
    --lambda_psi "$LAMBDA_PSI"
    --save_path "$OUT_DIR"
    --DEV cuda:0
    "${MOE_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
)

echo "Running: ${CMD[*]}"
echo ""

RUN_START_EPOCH=$(date +%s)
RUN_START_ISO=$(date --iso-8601=seconds)

set +e
(
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
) 2>&1 | tee "$OUT_DIR/run.log"
CMD_STATUS=${PIPESTATUS[0]}
set -e

RUN_END_EPOCH=$(date +%s)
RUN_END_ISO=$(date --iso-8601=seconds)
TOTAL_SECONDS=$((RUN_END_EPOCH - RUN_START_EPOCH))

cat > "$OUT_DIR/run_timing.tsv" <<EOF
metric	value
run_start	$RUN_START_ISO
run_end	$RUN_END_ISO
total_wall_seconds	$TOTAL_SECONDS
exit_status	$CMD_STATUS
EOF

echo ""
echo "=========================================="
if [[ "$CMD_STATUS" == "0" ]]; then
    echo "Done. Output saved to: $OUT_DIR"
else
    echo "FAILED with exit status $CMD_STATUS. Partial output saved to: $OUT_DIR"
fi
echo "Timing summary: $OUT_DIR/timing_summary.json"
echo "Run timing:     $OUT_DIR/run_timing.tsv"
echo "=========================================="
exit "$CMD_STATUS"
