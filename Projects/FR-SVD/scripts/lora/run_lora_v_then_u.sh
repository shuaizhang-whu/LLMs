#!/bin/bash
set -euo pipefail

GPU="${GPU:-4}"
COMPRESSED_MODEL="${COMPRESSED_MODEL:-/data/zhangs/LLMs/Projects/FR-SVD/result/phase2/ns512_rho0.42_a045_055_r1_0.95_lam1e5/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_fr_svd_0.8.pt}"
DATA_PATH="${DATA_PATH:-/data/zhangs/LLMs/Datasets/alpaca-cleaned}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/zhangs/LLMs/Projects/FR-SVD/result/FR_SVD_v1.4_lora_v_then_u}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
FIRST_STAGE_LR="${FIRST_STAGE_LR:-1e-4}"
SECOND_STAGE_LR="${SECOND_STAGE_LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
VAL_SET_SIZE="${VAL_SET_SIZE:-2000}"
MODEL_SEQ_LEN="${MODEL_SEQ_LEN:-2048}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        --compressed_model) COMPRESSED_MODEL="$2"; shift 2 ;;
        --data_path) DATA_PATH="$2"; shift 2 ;;
        --output_base) OUTPUT_BASE="$2"; shift 2 ;;
        --lora_r) LORA_R="$2"; shift 2 ;;
        --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --first_stage_lr) FIRST_STAGE_LR="$2"; shift 2 ;;
        --second_stage_lr) SECOND_STAGE_LR="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --micro_batch_size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

PYTHON="/data/zhangs/anaconda3/envs/compress/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FR_SVD_PY="${PROJECT_ROOT}/FR_SVD_v1.4.py"
LORA_PY="${PROJECT_ROOT}/utils/LoRA.py"
FIRST_STAGE_DIR="${OUTPUT_BASE}/first_half_V"
SECOND_STAGE_DIR="${OUTPUT_BASE}/second_half_VU"

mkdir -p "$OUTPUT_BASE"

if [[ ! -f "$COMPRESSED_MODEL" ]]; then
    echo "Compressed model not found: $COMPRESSED_MODEL" >&2
    exit 1
fi

echo "=========================================="
echo "FR-SVD v1.4: V->U LoRA"
echo "=========================================="
echo "GPU:              $GPU"
echo "Compressed Model: $COMPRESSED_MODEL"
echo "Data:             $DATA_PATH"
echo "Output:           $OUTPUT_BASE"
echo "LoRA r/alpha:     $LORA_R / $LORA_ALPHA"
echo "Epochs:           $NUM_EPOCHS"
echo "Stage LR:         V=$FIRST_STAGE_LR, U=$SECOND_STAGE_LR"
echo "Batch:            $BATCH_SIZE (micro=$MICRO_BATCH_SIZE)"
echo "=========================================="

echo
echo "▶ Step 1: Evaluating compressed model baseline..."
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$FR_SVD_PY" \
    --eval_only \
    --model_path "$COMPRESSED_MODEL" \
    --eval_dtype fp16 \
    --model_seq_len $MODEL_SEQ_LEN \
    --DEV cuda:0 2>&1 | tee "$OUTPUT_BASE/baseline_eval.log"

echo
echo "▶ Step 2: First-stage V LoRA with alpaca+wikitext2 validation..."
echo "Target modules: q_v_proj, k_v_proj, v_v_proj, o_v_proj, gate_v_proj, down_v_proj, up_v_proj"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$LORA_PY" \
    --prune_model "$COMPRESSED_MODEL" \
    --data_path "$DATA_PATH" \
    --extra_val_dataset wikitext2 \
    --output_dir "$FIRST_STAGE_DIR" \
    --lora_target_modules "q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $FIRST_STAGE_LR \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --val_set_size $VAL_SET_SIZE 2>&1 | tee "$OUTPUT_BASE/first_stage_training.log"

if [[ ! -f "$FIRST_STAGE_DIR/adapter_model.bin" ]] && [[ ! -f "$FIRST_STAGE_DIR/adapter_model.safetensors" ]]; then
    echo "First-stage V adapter not found in $FIRST_STAGE_DIR" >&2
    exit 1
fi

echo
echo "▶ Step 3: Merging first-stage V LoRA and evaluating..."
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$FR_SVD_PY" \
    --eval_only \
    --model_path "$COMPRESSED_MODEL" \
    --lora "$FIRST_STAGE_DIR" \
    --eval_dtype fp16 \
    --model_seq_len $MODEL_SEQ_LEN \
    --DEV cuda:0 2>&1 | tee "$OUTPUT_BASE/first_stage_eval.log"

if [[ ! -f "$FIRST_STAGE_DIR/merge.pt" ]]; then
    echo "Missing merged V-stage model: $FIRST_STAGE_DIR/merge.pt" >&2
    exit 1
fi

echo
echo "▶ Step 4: Second-stage U LoRA with lower LR and alpaca+wikitext2 validation..."
echo "Target modules: q_u_proj, k_u_proj, v_u_proj, o_u_proj, gate_u_proj, down_u_proj, up_u_proj"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$LORA_PY" \
    --prune_model "$FIRST_STAGE_DIR/merge.pt" \
    --data_path "$DATA_PATH" \
    --extra_val_dataset wikitext2 \
    --output_dir "$SECOND_STAGE_DIR" \
    --lora_target_modules "q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $SECOND_STAGE_LR \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --val_set_size $VAL_SET_SIZE 2>&1 | tee "$OUTPUT_BASE/second_stage_training.log"

if [[ ! -f "$SECOND_STAGE_DIR/adapter_model.bin" ]] && [[ ! -f "$SECOND_STAGE_DIR/adapter_model.safetensors" ]]; then
    echo "Second-stage U adapter not found in $SECOND_STAGE_DIR" >&2
    exit 1
fi

echo
echo "▶ Step 5: Merging second-stage U LoRA and evaluating final VU model..."
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$FR_SVD_PY" \
    --eval_only \
    --model_path "$FIRST_STAGE_DIR/merge.pt" \
    --lora "$SECOND_STAGE_DIR" \
    --eval_dtype fp16 \
    --model_seq_len $MODEL_SEQ_LEN \
    --DEV cuda:0 2>&1 | tee "$OUTPUT_BASE/final_eval.log"

echo
echo "=========================================="
echo "V->U Run Complete"
echo "=========================================="
echo "Baseline PPL:"
grep -E "PPL after pruning" "$OUTPUT_BASE/baseline_eval.log" | tail -1
echo "After V-stage PPL:"
grep -E "PPL after pruning" "$OUTPUT_BASE/first_stage_eval.log" | tail -1
echo "After U-stage PPL:"
grep -E "PPL after pruning" "$OUTPUT_BASE/final_eval.log" | tail -1
echo "Latest V-stage Alpaca/Wikitext2 eval:"
grep -E "eval_.*alpaca-cleaned_loss|eval_wikitext2_loss" "$OUTPUT_BASE/first_stage_training.log" | tail -2
echo "Latest U-stage Alpaca/Wikitext2 eval:"
grep -E "eval_.*alpaca-cleaned_loss|eval_wikitext2_loss" "$OUTPUT_BASE/second_stage_training.log" | tail -2
