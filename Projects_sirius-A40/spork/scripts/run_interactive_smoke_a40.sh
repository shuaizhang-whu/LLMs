#!/usr/bin/env bash
# Manually launch in train after the history-check experiment has finished.
set -euo pipefail
SPORK_INTERACTIVE_RESUME=0
SPORK_INTERACTIVE_CHECK=0
SPORK_INTERACTIVE_OUTPUT=
SPORK_INTERACTIVE_REFERENCE=conservative
while (( $# )); do
  case "$1" in
    --help|-h)
      printf '%s\n' '用法：bash scripts/run_interactive_smoke_a40.sh [结果目录]' \
        '      bash scripts/run_interactive_smoke_a40.sh --check-only' \
        '      bash scripts/run_interactive_smoke_a40.sh --resume 结果目录' \
        '      bash scripts/run_interactive_smoke_a40.sh --eta-reference paper [结果目录]' \
        '4 个已观察过的训练任务 × 3 种策略，共 12 次真实交互诊断。' \
        '复用已有模型服务；运行锁繁忙时立即退出，不排队，不启动服务。'
      exit 0 ;;
    --resume) SPORK_INTERACTIVE_RESUME=1 ;;
    --check-only) SPORK_INTERACTIVE_CHECK=1 ;;
    --eta-reference)
      if (( $# < 2 )) || [[ "$2" != paper && "$2" != conservative ]]; then
        printf -- '--eta-reference 需要 paper 或 conservative\n' >&2; exit 2
      fi
      SPORK_INTERACTIVE_REFERENCE=$2
      shift ;;
    --*) printf '未知参数：%s\n' "$1" >&2; exit 2 ;;
    *)
      if [[ -n "$SPORK_INTERACTIVE_OUTPUT" ]]; then printf '只能指定一个结果目录\n' >&2; exit 2; fi
      SPORK_INTERACTIVE_OUTPUT=$1 ;;
  esac
  shift
done
if (( SPORK_INTERACTIVE_RESUME )) && [[ -z "$SPORK_INTERACTIVE_OUTPUT" ]]; then
  printf -- '--resume 需要指定已有结果目录\n' >&2
  exit 2
fi
SPORK_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SPORK_REPO_ROOT"
if [[ -z ${SPORK_PYTHON:-} && -z ${SPORK_ENV:-} && ${CONDA_DEFAULT_ENV:-} != spork ]]; then
  SPORK_INTERACTIVE_CONDA=${CONDA_EXE:-$(command -v conda || true)}
  if [[ -z "$SPORK_INTERACTIVE_CONDA" ]]; then
    printf '请先 conda activate spork，或设置 SPORK_PYTHON\n' >&2
    exit 2
  fi
  SPORK_INTERACTIVE_CONDA_BASE=$("$SPORK_INTERACTIVE_CONDA" info --base)
  set +u
  set +o pipefail
  source "$SPORK_INTERACTIVE_CONDA_BASE/etc/profile.d/conda.sh"
  conda activate spork
  set -u
  set -o pipefail
fi
source scripts/spork_env.sh
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
export JE_ARROW_MALLOC_CONF=background_thread:false PYTHONDONTWRITEBYTECODE=1 LOGURU_LEVEL=ERROR
export LITELLM_LOCAL_MODEL_COST_MAP=True
SPORK_INTERACTIVE_OUTPUT=${SPORK_INTERACTIVE_OUTPUT:-results/a40_32b_interactive_$(date +%Y%m%d_%H%M%S)_${RANDOM}}
SPORK_INTERACTIVE_OUTPUT=$(realpath -m -- "$SPORK_INTERACTIVE_OUTPUT")
SPORK_INTERACTIVE_ARGS=(--model "$SPORK_QWEN_MODEL" --base-url "${SPORK_BASE_URL:-http://127.0.0.1:8017/v1}"
  --output "$SPORK_INTERACTIVE_OUTPUT" --eta-reference "$SPORK_INTERACTIVE_REFERENCE")
"$SPORK_PYTHON" scripts/run_interactive_smoke.py "${SPORK_INTERACTIVE_ARGS[@]}" --check-only
if (( SPORK_INTERACTIVE_CHECK )); then exit 0; fi
if (( ! SPORK_INTERACTIVE_RESUME )); then
  "$SPORK_PYTHON" scripts/run_interactive_smoke.py "${SPORK_INTERACTIVE_ARGS[@]}" --prepare-only
fi
printf '运行 12 次交互诊断；结果和日志：%s\n' "$SPORK_INTERACTIVE_OUTPUT"
set +e
"$SPORK_PYTHON" -u scripts/run_interactive_smoke.py "${SPORK_INTERACTIVE_ARGS[@]}" --resume \
  2>&1 | tee -a "$SPORK_INTERACTIVE_OUTPUT/run.log"
SPORK_INTERACTIVE_CODES=("${PIPESTATUS[@]}")
set -e
SPORK_INTERACTIVE_EXIT=${SPORK_INTERACTIVE_CODES[0]}
if (( SPORK_INTERACTIVE_EXIT == 0 && SPORK_INTERACTIVE_CODES[1] != 0 )); then
  SPORK_INTERACTIVE_EXIT=${SPORK_INTERACTIVE_CODES[1]}
fi
printf '%s\n' "$SPORK_INTERACTIVE_EXIT" > "$SPORK_INTERACTIVE_OUTPUT/exit_code.txt"
exit "$SPORK_INTERACTIVE_EXIT"
