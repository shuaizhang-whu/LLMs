#!/bin/bash
set -euo pipefail
# ================================
# FR-SVD v1.4：两阶段 LoRA 微调完整流程
# ================================
# 参考 SVDLLM / SAES-SVD 的两阶段 LoRA 微调
# 先微调 U 投影，合并；再微调 V 投影，合并
#
# 用法：
#   bash run_lora_finetuning_full.sh
#   bash run_lora_finetuning_full.sh --gpu 4
#   bash run_lora_finetuning_full.sh --compressed_model <path>
# ================================

# ========== 可修改参数 ==========
GPU="${GPU:-5}"
COMPRESSED_MODEL="${COMPRESSED_MODEL:-/data/zhangs/LLMs/Projects/FR-SVD/result/phase2/ns512_rho0.42_a045_055_r1_0.95_lam1e5/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_fr_svd_0.8.pt}"
DATA_PATH="${DATA_PATH:-/data/zhangs/LLMs/Datasets/alpaca-cleaned}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/zhangs/LLMs/Projects/FR-SVD/result/FR_SVD_v1.4_lora}"

# LoRA 超参
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
VAL_SET_SIZE="${VAL_SET_SIZE:-2000}"
MODEL_SEQ_LEN="${MODEL_SEQ_LEN:-2048}"

# ========== 命令行参数覆盖 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)               GPU="$2"; shift 2;;
        --compressed_model)  COMPRESSED_MODEL="$2"; shift 2;;
        --data_path)         DATA_PATH="$2"; shift 2;;
        --output_base)       OUTPUT_BASE="$2"; shift 2;;
        --lora_r)            LORA_R="$2"; shift 2;;
        --lora_alpha)        LORA_ALPHA="$2"; shift 2;;
        --lora_dropout)      LORA_DROPOUT="$2"; shift 2;;
        --num_epochs)        NUM_EPOCHS="$2"; shift 2;;
        --learning_rate)     LEARNING_RATE="$2"; shift 2;;
        --batch_size)        BATCH_SIZE="$2"; shift 2;;
        --micro_batch_size)  MICRO_BATCH_SIZE="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FR_SVD_PY="${PROJECT_ROOT}/FR_SVD_v1.4.py"

echo "=========================================="
echo "FR-SVD v1.4: Two-Stage LoRA Fine-tuning"
echo "=========================================="
echo "GPU:              $GPU"
echo "Compressed Model: $COMPRESSED_MODEL"
echo "Data:             $DATA_PATH"
echo "Output:           $OUTPUT_BASE"
echo "LoRA r/alpha:     $LORA_R / $LORA_ALPHA"
echo "Epochs:           $NUM_EPOCHS, LR=$LEARNING_RATE"
echo "Batch:            $BATCH_SIZE (micro=$MICRO_BATCH_SIZE)"
echo "=========================================="

mkdir -p "$OUTPUT_BASE"

if [[ ! -f "$COMPRESSED_MODEL" ]]; then
    echo "Compressed model not found: $COMPRESSED_MODEL" >&2
    exit 1
fi

FIRST_STAGE_DIR="$OUTPUT_BASE/first_half_U"
SECOND_STAGE_DIR="$OUTPUT_BASE/second_half_UV"

# ========== 步骤 1: baseline 评估 ==========
echo ""
echo "▶ Step 1: Evaluating compressed model baseline (before LoRA)..."
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$FR_SVD_PY" \
    --eval_only \
    --model_path "$COMPRESSED_MODEL" \
    --eval_dtype fp16 \
    --model_seq_len $MODEL_SEQ_LEN \
    --DEV cuda:0 2>&1 | tee "$OUTPUT_BASE/baseline_eval.log"

# ========== 步骤 2: 第一阶段 LoRA 微调（更新 U 投影） ==========
echo ""
echo "▶ Step 2: First-stage LoRA tuning (updating U projections)..."
echo "Target modules: q_u_proj, k_u_proj, v_u_proj, o_u_proj, gate_u_proj, down_u_proj, up_u_proj"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "${PROJECT_ROOT}/utils/LoRA.py" \
    --prune_model "$COMPRESSED_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$FIRST_STAGE_DIR" \
    --lora_target_modules "q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --val_set_size $VAL_SET_SIZE 2>&1 | tee "$OUTPUT_BASE/first_stage_training.log"

if [[ ! -f "$FIRST_STAGE_DIR/adapter_model.bin" ]] && [[ ! -f "$FIRST_STAGE_DIR/adapter_model.safetensors" ]]; then
    echo "First-stage LoRA adapter not found in $FIRST_STAGE_DIR" >&2
    exit 1
fi

# ========== 步骤 3: 合并第一阶段 LoRA 并评估 ==========
echo ""
echo "▶ Step 3: Merging first-stage LoRA and evaluating..."

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$FR_SVD_PY" \
    --eval_only \
    --model_path "$COMPRESSED_MODEL" \
    --lora "$FIRST_STAGE_DIR" \
    --eval_dtype fp16 \
    --model_seq_len $MODEL_SEQ_LEN \
    --DEV cuda:0 2>&1 | tee "$OUTPUT_BASE/first_stage_eval.log"

if [[ ! -f "$FIRST_STAGE_DIR/merge.pt" ]]; then
    echo "Missing merged model for second-stage input: $FIRST_STAGE_DIR/merge.pt" >&2
    exit 1
fi

# ========== 步骤 4: 第二阶段 LoRA 微调（更新 V 投影） ==========
echo ""
echo "▶ Step 4: Second-stage LoRA tuning (updating V projections)..."
echo "Target modules: q_v_proj, k_v_proj, v_v_proj, o_v_proj, gate_v_proj, down_v_proj, up_v_proj"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "${PROJECT_ROOT}/utils/LoRA.py" \
    --prune_model "$FIRST_STAGE_DIR/merge.pt" \
    --data_path "$DATA_PATH" \
    --output_dir "$SECOND_STAGE_DIR" \
    --lora_target_modules "q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --val_set_size $VAL_SET_SIZE 2>&1 | tee "$OUTPUT_BASE/second_stage_training.log"

if [[ ! -f "$SECOND_STAGE_DIR/adapter_model.bin" ]] && [[ ! -f "$SECOND_STAGE_DIR/adapter_model.safetensors" ]]; then
    echo "Second-stage LoRA adapter not found in $SECOND_STAGE_DIR" >&2
    exit 1
fi

# ========== 步骤 5: 合并第二阶段 LoRA 并评估最终模型 ==========
echo ""
echo "▶ Step 5: Merging second-stage LoRA and evaluating final model..."

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$FR_SVD_PY" \
    --eval_only \
    --model_path "$FIRST_STAGE_DIR/merge.pt" \
    --lora "$SECOND_STAGE_DIR" \
    --eval_dtype fp16 \
    --model_seq_len $MODEL_SEQ_LEN \
    --DEV cuda:0 2>&1 | tee "$OUTPUT_BASE/final_eval.log"

# ========== 步骤 6: 总结 ==========
echo ""
echo "=========================================="
echo "FR-SVD v1.4 LoRA Fine-tuning Complete!"
echo "=========================================="
echo ""
echo "Result Summary:"
echo ""
echo "Baseline (Compressed only):"
grep -E "wikitext2|PPL after pruning" "$OUTPUT_BASE/baseline_eval.log" | tail -1
echo ""
echo "After First-stage LoRA (U projections):"
grep -E "wikitext2|PPL after pruning" "$OUTPUT_BASE/first_stage_eval.log" | tail -1
echo ""
echo "After Second-stage LoRA (V projections):"
grep -E "wikitext2|PPL after pruning" "$OUTPUT_BASE/final_eval.log" | tail -1
echo ""
echo "Output files:"
echo "  - Baseline:    $OUTPUT_BASE/baseline_eval.log"
echo "  - First stage: $FIRST_STAGE_DIR/"
echo "  - Second stage:$SECOND_STAGE_DIR/"
echo "  - Final model: $SECOND_STAGE_DIR/merge.pt"
echo "=========================================="
