#!/usr/bin/env bash
# User launches this in tmux. It sends inference requests; do not auto-run.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
PYTHON=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python
MODEL_PATH=/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B
MODEL_NAME=${MODEL_NAME:-$MODEL_PATH}
MODEL_URL=${MODEL_URL:-http://127.0.0.1:8013/v1}
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR=$(mktemp -d "$PWD/results/first_turn_diagnostic_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
printf 'Results: %s\n' "$RUN_DIR"
printf 'Model: %s\nAPI: %s\n' "$MODEL_NAME" "$MODEL_URL"
# Read-only preflight, then the user-authorized inference requests below.
"$PYTHON" - "$MODEL_URL" "$MODEL_NAME" "$RUN_DIR" <<'PY'
import json, sys, urllib.request
from pathlib import Path
from spork_unified_eval import load_tau2_tasks
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(sys.argv[1].rstrip('/')+'/models',timeout=10) as response:
    data=json.load(response)
assert sys.argv[2] in [m['id'] for m in data['data']], 'Model name does not match the running server'
task=load_tau2_tasks(42)[6]
assert (task['domain'],str(task['task_id']))==('retail','43'), 'Task order changed; do not silently change the diagnostic case'
Path(sys.argv[3],'preflight.json').write_text(json.dumps({'server_models':data,'task':{'domain':task['domain'],'task_id':task['task_id']},'purpose':'Repeated first-turn diagnostic, not full-task evaluation'},indent=2))
PY
orders=('baseline,d1,d1_d2' 'd1,d1_d2,baseline' 'd1_d2,baseline,d1')
for round in 1 2 3; do
  "$PYTHON" spork_unified_eval.py \
    --model-url "$MODEL_URL" --model-name "$MODEL_NAME" \
    --tokenizer-path "$MODEL_PATH" --benchmark tau2 \
    --configs "${orders[$((round-1))]}" --n 155 --seed 42 --start 6 --end 7 \
    --enable-thinking --max-tokens 32768 --completion-max-tokens 8192 \
    --max-turns 1 --no-forced-conclusion --task-timeout-s 300 \
    --tau2-floor 2 --warmup-requests 1 --warmup-max-tokens 128 \
    --server-tp 4 --server-max-model-len 40960 \
    --server-tag qwen3-32b-plain-first-turn-diagnostic \
    --results-root "$RUN_DIR/round_$round"
done
# Report text/action repeatability; do not interpret truncated-task reward.
"$PYTHON" - "$RUN_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for p in sorted(root.glob('round_*/**/retail_43.json')):
 r=json.loads(p.read_text());t=r['turns'][0]
 rows.append({'round':p.relative_to(root).parts[0],'config':r['config'],'error':t.get('error'),
              'text_sha256':hashlib.sha256(t.get('baseline_text','').encode()).hexdigest(),
              'action':[t.get('baseline_tool_name'),t.get('baseline_tool_args')],
              'spec_used':t.get('spec_used',False),'probe_name':t.get('committed_probe_name',t.get('probe_tool_name')),
              'total_decode_tokens':t.get('total_decode_tokens'),'wall_s':r['wall_s']})
(root/'repeatability.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
print(json.dumps(rows,indent=2,ensure_ascii=False))
if len(rows)!=9 or any(x['error'] for x in rows):
 raise SystemExit('Diagnostic incomplete or contains errors; inspect run.log')
PY
