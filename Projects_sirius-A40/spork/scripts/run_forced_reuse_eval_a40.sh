#!/usr/bin/env bash
# Direct end-to-end evaluation. Launch manually in tmux train.
set -euo pipefail
SPORK_EVAL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SPORK_EVAL_ROOT"
SPORK_EVAL_VARIANT=forced_reuse
SPORK_EVAL_SMALL=0
while [[ ${1:-} == --* ]]; do
  case "$1" in
    --deferred) SPORK_EVAL_VARIANT=deferred ;;
    --cadence) SPORK_EVAL_VARIANT=cadence ;;
    --smoke) SPORK_EVAL_VARIANT=deferred; SPORK_EVAL_SMALL=1 ;;
    *) printf '用法：bash %s [--deferred | --cadence | --smoke] [结果目录]\n' "$0" >&2; exit 2 ;;
  esac
  shift
done
if [[ -z ${SPORK_PYTHON:-} && -z ${SPORK_ENV:-} && ${CONDA_DEFAULT_ENV:-} != spork ]]; then
  SPORK_EVAL_CONDA=${CONDA_EXE:-$(command -v conda || true)}
  if [[ -z "$SPORK_EVAL_CONDA" ]]; then
    printf '请先 conda activate spork，或设置 SPORK_PYTHON\n' >&2
    exit 2
  fi
  SPORK_EVAL_BASE=$("$SPORK_EVAL_CONDA" info --base)
  set +u
  set +o pipefail
  source "$SPORK_EVAL_BASE/etc/profile.d/conda.sh"
  conda activate spork
  set -u
  set -o pipefail
fi
source scripts/spork_env.sh
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 LOGURU_LEVEL=ERROR LITELLM_LOCAL_MODEL_COST_MAP=True
export JE_ARROW_MALLOC_CONF=background_thread:false
SPORK_EVAL_TASKS=(airline:2 airline:15 airline:30 airline:48 retail:0 retail:39 retail:76 retail:112)
SPORK_EVAL_MODES=(spork eta eta_retarget eta_retarget_forced_reuse)
SPORK_EVAL_REPEATS=1
if [[ $SPORK_EVAL_VARIANT == deferred ]]; then
  SPORK_EVAL_MODES=(eta_retarget_forced_reuse eta_retarget_deferred_reuse)
  SPORK_EVAL_REPEATS=3
fi
if [[ $SPORK_EVAL_VARIANT == cadence ]]; then
  SPORK_EVAL_MODES=(eta_retarget_deferred_reuse eta_retarget_deferred_reuse_fast eta_retarget_deferred_reuse_adaptive)
  SPORK_EVAL_REPEATS=3
fi
if (( SPORK_EVAL_SMALL )); then
  SPORK_EVAL_TASKS=(airline:15 airline:48)
  SPORK_EVAL_MODES=(eta_retarget_forced_reuse eta_retarget_deferred_reuse)
  SPORK_EVAL_REPEATS=1
  SPORK_EVAL_VARIANT=deferred_smoke
fi
SPORK_EVAL_OUTPUT=${1:-results/a40_32b_${SPORK_EVAL_VARIANT}_eval_$(date +%Y%m%d_%H%M%S)_${RANDOM}}
SPORK_EVAL_OUTPUT=$(realpath -m -- "$SPORK_EVAL_OUTPUT")
SPORK_EVAL_ARGS=(
  --model "$SPORK_QWEN_MODEL" --base-url "${SPORK_BASE_URL:-http://127.0.0.1:8017/v1}"
  --output "$SPORK_EVAL_OUTPUT" --eta-reference paper
  --tasks "${SPORK_EVAL_TASKS[@]}"
  --modes "${SPORK_EVAL_MODES[@]}"
  --repeats "$SPORK_EVAL_REPEATS"
  --max-steps 60 --max-tokens 8192 --tool-floor 2
)
if [[ -f "$SPORK_EVAL_OUTPUT/manifest.json" ]]; then
  SPORK_EVAL_ARGS+=(--resume)
fi
mkdir -p "$SPORK_EVAL_OUTPUT"
printf '直接运行端到端评估：%s 个任务 × %s 组 × %s 次重复 = %s 次。\n结果和日志：%s\n' \
  "${#SPORK_EVAL_TASKS[@]}" "${#SPORK_EVAL_MODES[@]}" "$SPORK_EVAL_REPEATS" \
  "$(( ${#SPORK_EVAL_TASKS[@]} * ${#SPORK_EVAL_MODES[@]} * SPORK_EVAL_REPEATS ))" "$SPORK_EVAL_OUTPUT"
set +e
"$SPORK_PYTHON" -u scripts/run_interactive_smoke.py "${SPORK_EVAL_ARGS[@]}" \
  2>&1 | tee -a "$SPORK_EVAL_OUTPUT/run.log"
SPORK_EVAL_CODES=("${PIPESTATUS[@]}")
set -e
SPORK_EVAL_EXIT=${SPORK_EVAL_CODES[0]}
if (( SPORK_EVAL_EXIT == 0 && SPORK_EVAL_CODES[1] != 0 )); then
  SPORK_EVAL_EXIT=${SPORK_EVAL_CODES[1]}
fi
printf '%s\n' "$SPORK_EVAL_EXIT" > "$SPORK_EVAL_OUTPUT/exit_code.txt"
exit "$SPORK_EVAL_EXIT"
