#!/usr/bin/env bash
# User starts in tmux. Full 155-task comparison on ONE non-eager D3 server.
set -euo pipefail
cd /mnt/data/zhangshuai/LLMs/Projects/spork
PYTHON=/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python
MODEL=/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR=$(mktemp -d "$PWD/results/d3_formal_n155_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
printf 'Formal results: %s\n' "$RUN_DIR"
"$PYTHON" - "$RUN_DIR" "$MODEL" <<'PY'
import json,sys,urllib.request,subprocess,importlib.metadata as meta
from pathlib import Path
from spork_unified_eval import load_tau2_tasks
root=Path(sys.argv[1]);opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open('http://127.0.0.1:8015/v1/models',timeout=15) as r: models=json.load(r)
assert sys.argv[2] in [m['id'] for m in models['data']], 'Wrong model on port 8015'
with opener.open('http://127.0.0.1:8015/spork/status',timeout=30) as r: status=json.load(r)
assert status.get('has_drafter'), 'D3/NG​RAM service required'
req=urllib.request.Request('http://127.0.0.1:8015/spork/clear',data=b'{}',headers={'Content-Type':'application/json'})
with opener.open(req,timeout=30) as r: cleared=json.load(r)
tasks=load_tau2_tasks(42)
assert len(tasks)==155, f'Task set changed: {len(tasks)} instead of 155'
(root/'task_manifest.json').write_text(json.dumps([{'domain':t['domain'],'task_id':t['task_id']} for t in tasks],indent=2))
(root/'preflight.json').write_text(json.dumps({'models':models,'before':status,'clear':cleared},indent=2))
(root/'provenance.json').write_text(json.dumps({'packages':{k:meta.version(k) for k in ['vllm','torch','transformers','tau2']},'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'server_port':8015,'configs':['baseline','d1_d2','d1_d2_d3'],'n':155,'seed':42,'max_turns':8,'forced_conclusion':True,'task_timeout_s':0,'server_mode':'D3 HTTP; V1; sync; non-eager; ngram20; TP4','warmup_requests':2},indent=2))
(root/'working_changes.patch').write_text(subprocess.check_output(['git','diff'],text=True))
print('Preflight passed: correct model, D3 sidecar, 155 tasks.')
PY
"$PYTHON" spork_unified_eval.py \
  --model-url http://127.0.0.1:8015/v1 --model-name "$MODEL" \
  --tokenizer-path "$MODEL" --benchmark tau2 --n 155 --seed 42 \
  --configs baseline,d1_d2,d1_d2_d3 --enable-thinking \
  --max-tokens 32768 --completion-max-tokens 8192 --max-turns 8 \
  --task-timeout-s 0 --tau2-floor 2 \
  --warmup-requests 2 --warmup-max-tokens 512 \
  --d2-max-retries 5 --d2-retry-token-step 50 \
  --d2-confidence-threshold 0.90 --d2-probe-max-tokens 160 \
  --server-tp 4 --server-max-model-len 40960 --server-max-num-batched-tokens 2048 \
  --server-tag qwen3-32b-d3-formal-vllm029-sync-v1-non-eager \
  --server-speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":20,"prompt_lookup_min":2}' \
  --results-root "$RUN_DIR/eval"
"$PYTHON" - "$RUN_DIR" <<'PY'
import json,sys,urllib.request,collections
from pathlib import Path
root=Path(sys.argv[1]);opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open('http://127.0.0.1:8015/spork/status',timeout=30) as r: after=json.load(r)
before=json.loads((root/'preflight.json').read_text())['clear']
keys=['set_count','inject_count','draft_token_count','verified_injections','verified_draft_tokens','accepted_draft_tokens','dropped_injections','prefix_mismatches','accounting_errors']
delta={k:after.get(k,0)-before.get(k,0) for k in keys}
paths=list((root/'eval').glob('**/summary.json'));assert len(paths)==1
summary=json.loads(paths[0].read_text())
(root/'summary.json').write_text(json.dumps(summary,indent=2))
set_errors=collections.Counter();clear_errors=[]
for p in (root/'eval').glob('**/d1_d2_d3/*.json'):
 r=json.loads(p.read_text())
 for t in r['turns']:
  ds=t.get('d3_status')
  if ds and (ds.get('error') or ds.get('clear_status')!='cleared'):
   clear_errors.append({'task':p.name,'turn':t.get('turn'),'status':ds})
  for probe in t.get('probes',[]):
   ds=probe.get('d3_set')
   if ds and ds.get('status')!='ok':set_errors[str(ds.get('error','unknown'))]+=1
   if probe.get('d3_set_error'):set_errors[probe['d3_set_error']]+=1
report={'counter_delta':delta,'draft_acceptance_rate':delta['accepted_draft_tokens']/delta['verified_draft_tokens'] if delta['verified_draft_tokens'] else None,'registration_errors':dict(set_errors),'clear_errors':clear_errors,'active_requests_after_eval':after.get('active_requests',0),'pending_verifications_after_eval':after.get('pending_verifications',0),'note':'Counts exclude ngram drafts and bonus/correction tokens. Compare D1+D2 vs D1+D2+D3 on this same server; baseline here includes ngram.'}
(root/'d3_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
(root/'status_after.json').write_text(json.dumps(after,indent=2))
req=urllib.request.Request('http://127.0.0.1:8015/spork/clear',data=b'{}',headers={'Content-Type':'application/json'})
with opener.open(req,timeout=30) as r: final_clear=json.load(r)
(root/'final_clear.json').write_text(json.dumps(final_clear,indent=2))
print(json.dumps({'configs':summary['configs'],'D3':report},indent=2,ensure_ascii=False))
print('Summary:',root/'summary.json');print('D3 report:',root/'d3_report.json')
if any(v['n_tasks']!=155 or v['n_errors'] for v in summary['configs'].values()) or delta['accounting_errors'] or clear_errors:
 raise SystemExit('Results saved, but errors require review before performance claims.')
if not delta['verified_injections']:
 raise SystemExit('Results saved, but D3 was never verified; inspect logs.')
PY
