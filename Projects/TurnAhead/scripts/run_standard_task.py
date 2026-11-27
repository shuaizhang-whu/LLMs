#!/usr/bin/env python3
"""Measure a standard Qwen3-32B task with ALL speculative methods disabled.

Default prepares only. --run requires a dedicated ordinary vLLM server whose
/server_info proves speculative_config=None. Reuses the fixed-user task and
8-turn protocol of the historical SPORK/raw comparison, and compares the
complete generated outputs and tool states before emitting a three-arm table.
"""
from __future__ import annotations
import argparse,asyncio,copy,json,hashlib,sys,time,traceback,fcntl
from pathlib import Path
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spork_core import paths
from spork_core.evaluation_artifacts import digest,freeze_plan,write_json,read_completed,complete_record
from spork_core.executors import Tau2ToolExecutor
from spork_core.benchmarks.tau2 import get_environment,load_system_prompt,load_tools
from spork_core.qwen import QwenToolCallAdapter
from spork_core.prefix_replay import PrefixHttpClient
from spork_core.turn_runner import RetryTurnConfig
from spork_core.timed_retry import run_timed_retry_turn
from spork_core.vllm_client import VllmClient
from spork_unified_eval import load_tau2_tasks
from scripts.run_successor_full_eval import evaluation_session


def plain_server(config):
    c=config.get('vllm_config')
    if not isinstance(c,dict) or 'speculative_config' not in c:
        raise ValueError('Server must expose JSON vllm_config including speculative_config')
    if c['speculative_config'] is not None:
        raise ValueError('Standard baseline requires ALL token speculation disabled at server startup')
    if c.get('parallel_config',{}).get('tensor_parallel_size')!=4:
        raise ValueError('Use TP=4, matching the historical 32B serving configuration')
    if not config.get('vllm_env',{}).get('VLLM_BATCH_INVARIANT'):
        raise ValueError('Batch invariance must be enabled as in the reference')
    return c


def off_cfg():
    return RetryTurnConfig(max_retries=0,confidence_threshold=.9,retry_token_step=50,
        probe_max_tokens=160,baseline_max_tokens=8192,context_window_tokens=40960,
        enable_thinking=True,seed=42,d3_set=None,d3_clear=None,launch_mode='off')


def same_trajectory(a,b):
    fields=('baseline_text','main_token_ids','baseline_has_tool_call','baseline_tool_name',
            'baseline_tool_args','tool_result_sha256','environment_after')
    return (not a['errors'] and not b['errors'] and len(a['turns'])==len(b['turns'])
        and all(all(x.get(k)==y.get(k) for k in fields) for x,y in zip(a['turns'],b['turns'])))


async def execute(args,task,tok,ref):
    adapter=QwenToolCallAdapter(tok,load_tools(task['domain']))
    initial=[{'role':'system','content':load_system_prompt(task['domain'])},
             {'role':'user','content':task['user_message']}]
    ids=tok.encode(adapter.render_main_prompt(initial,enable_thinking=True),add_special_tokens=False)
    path=args.output/'records/standard.json'
    if path.with_name(path.name+'.seal').exists():
        row=read_completed(path)
    else:
        http=PrefixHttpClient(args.base_url,args.model,900,presence_penalty=0.)
        client=VllmClient(args.base_url,args.model,900)
        async with evaluation_session() as session:
            async with session.get(http.root+'/server_info?config_format=json',timeout=90) as response:
                response.raise_for_status();info=await response.json()
            cfg=plain_server(info)
            if str(Path(cfg.get('model_config',{}).get('model','')).resolve())!=str(Path(args.model).resolve()):
                raise ValueError('Serving model is not the reference Qwen3-32B')
            write_json(args.output/'server_info.json',info)
            async with session.post(http.root+'/reset_prefix_cache',timeout=30) as response:
                response.raise_for_status();reset=await response.json()
                if reset.get('success') is not True:raise RuntimeError('Dedicated server cache reset refused')
            await http.complete(session,ids,max_tokens=1,seed=42)
            messages=copy.deepcopy(initial);executor=Tau2ToolExecutor(get_environment(task['domain']),floor_s=2.)
            row={'group_id':args.task_id,'config':'standard','model':args.model,'errors':[],
                'turns':[],'initial_prompt_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                'token_speculation':False,'action_speculation':False,'seed':42,'max_turns':8}
            started=time.perf_counter()
            try:
                for turn in range(1,9):
                    entry=await run_timed_retry_turn(session,client,adapter,messages,executor,turn,off_cfg(),http=http)
                    if entry['num_probes'] or entry.get('spec_dispatched') or entry.get('d3_set_count'):
                        raise RuntimeError('Standard path launched speculative work')
                    row['turns'].append(entry)
                    write_json(args.output/'in_progress.json',row)
                    print(f'  standard {args.task_id} turn {turn}: {entry.get("baseline_tool_name") or "reply"}',flush=True)
                    if not entry['baseline_has_tool_call']:break
            except Exception as exc:
                row['errors'].append(f'{type(exc).__name__}: {exc}');row['traceback']=traceback.format_exc()
            finally:row['wall_s']=time.perf_counter()-started
            complete_record(path,row)
    matching={k:same_trajectory(row,r) for k,r in ref.items()}
    result={'task':args.task_id,'same_outputs_tools_states':matching,
        'all_three_comparable':not row['errors'] and all(matching.values()),
        'standard_s':row['wall_s'],'spork_s':ref['spork']['wall_s'],'raw_s':ref['raw']['wall_s'],
        'reference_source':str(args.reference),'standard_errors':row['errors'],
        'scope':'Single fixed-user Qwen3-32B task; standard measured with no token/action speculation. Historical SPORK/raw measured in earlier session; source dates/configurations retained, not a contemporaneous three-arm repetition.'}
    write_json(args.output/'three_way_comparison.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if not result['all_three_comparable']:raise RuntimeError('Saved output differs from reference; inspect before plotting as matched task')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task-id',default='retail:0')
    p.add_argument('--reference',type=Path,default=ROOT/'results/successor_full_20261025_codex')
    p.add_argument('--model',default='/mnt/data/zhangshuai/LLMs/Models/Qwen3-32B')
    p.add_argument('--base-url',default='http://127.0.0.1:8029/v1')
    p.add_argument('--run',action='store_true');args=p.parse_args()
    args.output=args.output.resolve();args.reference=args.reference.resolve()
    d,tid=args.task_id.split(':');ref={}
    for c in ['spork','raw']:
        path=args.reference/f'records/{d}_{tid}_r0_{c}.json';ref[c]=read_completed(path)
        if ref[c] is None or ref[c]['errors']:raise ValueError('Both historical comparison arms must be valid sealed runs')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    task=next(t for t in load_tau2_tasks(42) if f"{t['domain']}:{t['task_id']}"==args.task_id)
    adapter=QwenToolCallAdapter(tok,load_tools(d));messages=[{'role':'system','content':load_system_prompt(d)},{'role':'user','content':task['user_message']}]
    ids=tok.encode(adapter.render_main_prompt(messages,enable_thinking=True),add_special_tokens=False)
    h=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if any(r['initial_prompt_sha256']!=h for r in ref.values()):raise ValueError('Current task/prompt differs from frozen SPORK/raw reference')
    sources=sorted(set([Path(__file__),ROOT/'scripts/run_standard_task.sh',ROOT/'scripts/serve_qwen32_standard.sh',ROOT/'spork_unified_eval.py',ROOT/'tau2_tool_parser.py']+list((ROOT/'spork_core').rglob('*.py'))+list((paths.TAU2_ROOT/'src/tau2').rglob('*.py'))))
    manifest={'purpose':'standard_no_speculation_single_task','group_id':args.task_id,'model':args.model,
        'base_url':args.base_url,'initial_prompt_sha256':h,'standard':'No successor, no D1/D2/D3, no server speculative_config',
        'seed':42,'max_turns':8,'max_tokens':8192,'tool_floor_s':2.,
        'reference_sha256':{c:digest(args.reference/f'records/{d}_{tid}_r0_{c}.json') for c in ref},
        'benchmark_data_sha256':{str(f):digest(f) for f in (paths.TAU2_ROOT/f'data/tau2/domains/{d}').rglob('*') if f.is_file()},
        'model_metadata_sha256':{name:digest(Path(args.model)/name) for name in ['config.json','tokenizer_config.json','generation_config.json']},
        'code_sha256':{str(f.relative_to(ROOT)):digest(f) for f in sources}}
    freeze_plan(args.output,manifest,ROOT,sources)
    print(json.dumps({'output':str(args.output),'task':args.task_id,'reference_prompt_exact':True,'generation_requests':'enabled' if args.run else 0},ensure_ascii=False),flush=True)
    if args.run:
        with ExitStack() as stack:
            for sibling in ROOT.parent.glob('spork*'):
                if (sibling/'spork_core').is_dir():
                    lock=sibling/'results/controller_online_8029.lock';lock.parent.mkdir(exist_ok=True)
                    fcntl.flock(stack.enter_context(lock.open('a+')).fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            asyncio.run(execute(args,task,tok,ref))

if __name__=='__main__':main()
