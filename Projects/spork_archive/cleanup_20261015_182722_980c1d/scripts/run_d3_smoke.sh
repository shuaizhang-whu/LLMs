#!/usr/bin/env bash
# Manual tmux command. One task, up to three turns/config; not a benchmark.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
PYTHON=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python
MODEL=/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR=$(mktemp -d "$PWD/results/d3_smoke_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
printf 'D3 smoke results: %s\n' "$RUN_DIR"
"$PYTHON" - "$RUN_DIR" "$MODEL" <<'PY'
import json,sys,urllib.request
from pathlib import Path
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open('http://127.0.0.1:8014/v1/models',timeout=15) as r: models=json.load(r)
assert sys.argv[2] in [m['id'] for m in models['data']], 'Wrong model served on port 8014'
with opener.open('http://127.0.0.1:8014/spork/status',timeout=30) as r: status=json.load(r)
assert status.get('has_drafter'), 'No ngram drafter; check D3 server startup flags'
req=urllib.request.Request('http://127.0.0.1:8014/spork/clear',data=b'{}',headers={'Content-Type':'application/json'})
with opener.open(req,timeout=30) as r: cleared=json.load(r)
Path(sys.argv[1],'preflight.json').write_text(json.dumps({'models':models,'before':status,'clear':cleared},indent=2))
print('D3 sidecar and ngram drafter ready.')
PY
"$PYTHON" spork_unified_eval.py \
  --model-url http://127.0.0.1:8014/v1 --model-name "$MODEL" \
  --tokenizer-path "$MODEL" --benchmark tau2 --n 1 --seed 42 \
  --configs baseline,d1_d2,d1_d2_d3 --enable-thinking \
  --max-tokens 32768 --completion-max-tokens 8192 \
  --max-turns 3 --no-forced-conclusion --task-timeout-s 300 --tau2-floor 2 \
  --warmup-requests 1 --warmup-max-tokens 128 \
  --server-tp 4 --server-max-model-len 40960 --server-max-num-batched-tokens 2048 \
  --server-tag qwen3-32b-d3-smoke-vllm029-sync-v1-eager \
  --server-speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  --results-root "$RUN_DIR/eval"
"$PYTHON" - "$RUN_DIR" <<'PY'
import json,sys,urllib.request
from pathlib import Path
root=Path(sys.argv[1]);opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open('http://127.0.0.1:8014/spork/status',timeout=30) as r: after=json.load(r)
before=json.loads((root/'preflight.json').read_text())['clear']
keys=['set_count','inject_count','draft_token_count','verified_injections','verified_draft_tokens','accepted_draft_tokens','dropped_injections','prefix_mismatches','accounting_errors']
delta={k:after.get(k,0)-before.get(k,0) for k in keys}
tasks=[]; errors=[];sets=[]
for p in sorted((root/'eval').glob('**/*.json')):
 r=json.loads(p.read_text())
 if 'turns' not in r:continue
 tasks.append({'config':r['config'],'task_id':r['task_id'],'turns':r['n_turns']})
 for t in r['turns']:
  if t.get('error'):errors.append({'config':r['config'],'error':t['error']})
  for probe in t.get('probes',[]):
   if probe.get('d3_set') is not None:sets.append(probe['d3_set'])
   if probe.get('d3_set_error'):errors.append({'d3_set_error':probe['d3_set_error']})
state='tokens_accepted' if delta['accepted_draft_tokens']>0 else ('verified_but_zero_accepted' if delta['verified_injections'] else ('injected_but_not_verified' if delta['inject_count'] else 'no_injection'))
report={'stage':state,'counter_delta':delta,'tasks':tasks,'errors':errors,'registrations':sets,
        'active_requests_after_eval':after.get('active_requests',0),
        'pending_verifications_after_eval':after.get('pending_verifications',0),
        'note':'D3-only counts from worker verification, excluding ngram drafts and bonus/correction tokens. This truncated smoke is not a latency/quality benchmark.'}
(root/'d3_smoke_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
(root/'status_after.json').write_text(json.dumps(after,indent=2))
print(json.dumps(report,indent=2,ensure_ascii=False))
# Cleanup even if the evaluator recorded task-level errors.
req=urllib.request.Request('http://127.0.0.1:8014/spork/clear',data=b'{}',headers={'Content-Type':'application/json'})
with opener.open(req,timeout=30) as r: final_clear=json.load(r)
(root/'final_clear.json').write_text(json.dumps(final_clear,indent=2))
if errors or delta['accounting_errors'] or len(tasks)!=3:
 raise SystemExit('D3 smoke contains errors; inspect report and server log before extending.')
if not delta['verified_injections']:
 raise SystemExit('No verified D3 injection yet; send d3_smoke_report.json and server log for diagnosis.')
print('D3 verification path exercised; see accepted_draft_tokens for acceptance, not just registration.')
PY
