#!/usr/bin/env bash
# Source only: shared local paths. Does not change cwd, activate an environment,
# download weights, or start any experiment. Activate the intended env first,
# or set SPORK_PYTHON / SPORK_ENV explicitly.
SPORK_REPO_ROOT=$(cd -- "${SPORK_REPO_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")/..}" && pwd)
if [[ -z "${SPORK_MODEL_ROOT:-}" ]]; then
  case "$(basename -- "$(dirname -- "$SPORK_REPO_ROOT")")" in
    Projects|Projects_*) SPORK_MODEL_ROOT="$SPORK_REPO_ROOT/../../Models" ;;
    *) SPORK_MODEL_ROOT="$SPORK_REPO_ROOT/models" ;;
  esac
elif [[ "$SPORK_MODEL_ROOT" != /* ]]; then
  SPORK_MODEL_ROOT="$SPORK_REPO_ROOT/$SPORK_MODEL_ROOT"
fi
SPORK_MODEL_ROOT=$(realpath -m -- "$SPORK_MODEL_ROOT")
SPORK_PYTHON=${SPORK_PYTHON:-${SPORK_ENV:+$SPORK_ENV/bin/python}}
SPORK_PYTHON=$(command -v -- "${SPORK_PYTHON:-python}") || {
  printf 'Activate the SPORK Python environment or set SPORK_PYTHON.\n' >&2
  return 1
}
if ! "$SPORK_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  printf 'SPORK requires Python >= 3.10; activate that environment or set SPORK_PYTHON.\n' >&2
  return 1
fi
SPORK_QWEN_MODEL=${SPORK_QWEN_MODEL:-$SPORK_MODEL_ROOT/Qwen3-32B}
SPORK_ENCODER=${SPORK_ENCODER:-$SPORK_MODEL_ROOT/bert-tiny}
export SPORK_REPO_ROOT SPORK_MODEL_ROOT SPORK_PYTHON
export PYTHONPATH="$SPORK_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

spork_configure_cuda_paths() {
  local spork_env_prefix spork_curand
  spork_env_prefix=${SPORK_ENV:-$("$SPORK_PYTHON" -c 'import sys; print(sys.prefix)')}
  export CUDA_HOME=${CUDA_HOME:-$spork_env_prefix}
  export CUDA_PATH=${CUDA_PATH:-$CUDA_HOME}
  export FLASHINFER_NVCC=${FLASHINFER_NVCC:-$CUDA_HOME/bin/nvcc}
  for spork_curand in "$spork_env_prefix"/lib/python*/site-packages/nvidia/curand/include; do
    if [[ -d "$spork_curand" ]]; then
      export NVCC_PREPEND_FLAGS="-I$spork_curand${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"
      break
    fi
  done
  export LD_LIBRARY_PATH="$spork_env_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}
