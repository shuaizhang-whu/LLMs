#!/usr/bin/env bash
# Launch from the user's vllm tmux session. Explicit GPU selection only.
# Example: bash scripts/serve_qwen_a40.sh Qwen3-32B 0,1
# Example: bash scripts/serve_qwen_a40.sh Qwen3-4B 0
set -euo pipefail
ulimit -c 0
export PYTHONFAULTHANDLER=1
source "$(dirname -- "${BASH_SOURCE[0]}")/spork_env.sh"
SPORK_A40_MODEL=${1:-Qwen3-32B}
SPORK_A40_DEVICES=${2:?Pass dedicated GPU indices, e.g. 0,1 or 0}
case "$SPORK_A40_MODEL" in
  Qwen3-32B) SPORK_A40_TP=2 ;;
  Qwen3-4B) SPORK_A40_TP=1 ;;
  *) printf 'Supported models: Qwen3-32B or Qwen3-4B\n' >&2; exit 2 ;;
esac
SPORK_A40_MODEL_PATH="$SPORK_MODEL_ROOT/$SPORK_A40_MODEL"
SPORK_A40_PORT=${SPORK_A40_PORT:-8017}
spork_configure_cuda_paths
SPORK_A40_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
if (( ${SPORK_A40_DRIVER%%.*} < 580 )); then
  SPORK_A40_COMPAT=${SPORK_CUDA_COMPAT_DIR:-$SPORK_REPO_ROOT/.runtime/cuda-compat-13-0/usr/local/cuda-13.0/compat}
  if [[ ! -f "$SPORK_A40_COMPAT/libcuda.so.1" ]]; then
    printf 'CUDA 13 requires local compatibility libraries on this driver. Run scripts/install_cuda_compat.sh first.\n' >&2
    exit 2
  fi
  export LD_LIBRARY_PATH="$SPORK_A40_COMPAT:$LD_LIBRARY_PATH"
  # FlashInfer JIT links with -lcuda; the conda toolkit omits driver stubs.
  export LIBRARY_PATH="$SPORK_A40_COMPAT${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
export CUDA_VISIBLE_DEVICES="$SPORK_A40_DEVICES"
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
# Local GDB trace: pyarrow's bundled jemalloc background thread crashes during
# import on this host. Scope the workaround to this server and its workers.
export JE_ARROW_MALLOC_CONF="${JE_ARROW_MALLOC_CONF:-background_thread:false}"
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$SPORK_PYTHON" - "$SPORK_A40_MODEL_PATH" "$SPORK_A40_TP" "$SPORK_A40_PORT" <<'PY'
import json, os, socket, struct, subprocess, sys
from pathlib import Path
from importlib.metadata import version
model, tp, port = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
if not version('vllm').startswith('0.29.'):
    raise SystemExit('The D3 bridge requires vLLM 0.29.x')
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if len(devices) != tp or len(set(devices)) != tp or not all(x.isdigit() for x in devices):
    raise SystemExit(f'Expected {tp} distinct numeric GPU indices')
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'], text=True)
gpus = {r.split(',')[0].strip(): [int(x) for x in r.split(',')[1:]] for r in rows.splitlines()}
if any(d not in gpus or gpus[d][0] > 512 or gpus[d][1] > 5 for d in devices):
    raise SystemExit(f'Selected GPUs are unavailable or busy: {devices}')
index = json.loads((model / 'model.safetensors.index.json').read_text())
for shard in sorted(set(index['weight_map'].values())):
    path = model / shard
    with path.open('rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        if header_size > 32 * 1024 * 1024:
            raise SystemExit(f'Invalid weight header: {path}')
        header = json.loads(f.read(header_size))
    expected = 8 + header_size + max(v['data_offsets'][1] for k,v in header.items() if k != '__metadata__')
    if path.stat().st_size != expected:
        raise SystemExit(f'Truncated weight shard: {path}')
with socket.socket() as s:
    s.bind(('127.0.0.1', port))
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != tp:
    raise SystemExit('CUDA initialization failed')
print(json.dumps({'model':str(model), 'gpus':devices, 'tensor_parallel':tp,
                  'torch':torch.__version__, 'cuda':torch.version.cuda,
                  'device':torch.cuda.get_device_name(0), 'port':port}, indent=2), flush=True)
PY
mkdir -p "$SPORK_REPO_ROOT/results"
SPORK_A40_LOG=$(mktemp "$SPORK_REPO_ROOT/results/server_a40_${SPORK_A40_MODEL}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf 'Server log: %s\n' "$SPORK_A40_LOG"
cd "$SPORK_REPO_ROOT"
"$SPORK_PYTHON" launch_vllm_spork_http.py \
  --model "$SPORK_A40_MODEL_PATH" --served-model-name "$SPORK_A40_MODEL_PATH" \
  --host 127.0.0.1 --port "$SPORK_A40_PORT" --tensor-parallel-size "$SPORK_A40_TP" \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization "${SPORK_GPU_MEMORY_UTILIZATION:-0.85}" \
  --trust-remote-code --max-logprobs 20 --max-num-seqs 4 \
  --max-num-batched-tokens 512 --no-async-scheduling --enforce-eager \
  --speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  2>&1 | tee "$SPORK_A40_LOG"
