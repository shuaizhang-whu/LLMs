#!/usr/bin/env bash
# Run manually in tmux against the same existing batch-invariant D3 server.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
SPORK_CONTINUE_PYTHON=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if (( $# != 1 )) || [[ ! -f "$1/imported_measurements.json" ]]; then
  printf 'Usage: bash %s prepared-continuation-directory\n' "$0" >&2
  exit 1
fi
SPORK_CONTINUE_RUN=$1
"$SPORK_CONTINUE_PYTHON" scripts/diagnose_prefix_forks.py \
  --output "$SPORK_CONTINUE_RUN" --repetitions 3 --require-invariance \
  --continue-on-truncation --resume --prepare-only
exec > >(tee -a "$SPORK_CONTINUE_RUN/run.log") 2>&1
if "$SPORK_CONTINUE_PYTHON" scripts/diagnose_prefix_forks.py \
    --output "$SPORK_CONTINUE_RUN" --repetitions 3 --require-invariance --continue-on-truncation --resume; then
  printf '所有对照已完成，完整输出一致性检查通过。\n'
else
  SPORK_CONTINUE_EXIT=$?
  "$SPORK_CONTINUE_PYTHON" - "$SPORK_CONTINUE_RUN" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]) / 'fork_diagnostic_report.json'
if path.exists():
    report = json.loads(path.read_text())
    if report.get('collection_complete'):
        print('所有对照已记录；完整验证未通过，截断和其他失败均保留在报告中。退出码 2 不表示程序再次在截断处中止。')
    else:
        print('采集尚未完成，请查看 run.log 中的异常。')
    print('Report:', path)
PY
  exit "$SPORK_CONTINUE_EXIT"
fi
printf 'Report: %s/fork_diagnostic_report.json\n' "$SPORK_CONTINUE_RUN"
