"""Offline audit of saved SPORK runs. No model serving or tool execution.

Outputs are written to a NEW directory. Input JSON is never rewritten.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spork_core.qwen import token_split, parse_baseline_tool_call
from spork_core.tool_calls import args_exact_match

CONFIGS = ('baseline', 'd1', 'd1_d2')

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def action(t):
    if t is None:
        return {'end_of_trace': True}
    if not t.get('baseline_has_tool_call'):
        return None
    return {'name': t['baseline_tool_name'], 'arguments': t.get('baseline_tool_args') or {}}

def trace(r):
    return [action(t) for t in r['turns'] if t.get('baseline_has_tool_call')]

def visible_history(t):
    text = re.sub(r'^\s*<think>.*?</think>\s*', '', t.get('baseline_text', ''), count=1, flags=re.DOTALL)
    return text, t.get('tool_result_len'), t.get('tool_result_preview')

def excerpt(text, at, radius=180):
    return text[max(0, at-radius):at+radius]

def first_text_difference(a, b):
    x, y = (a or {}).get('baseline_text',''), (b or {}).get('baseline_text','')
    i = next((i for i,(u,v) in enumerate(zip(x,y)) if u != v), min(len(x), len(y)))
    return {'char_offset': i, 'baseline_excerpt': excerpt(x,i), 'variant_excerpt': excerpt(y,i)}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tokenizer', required=True)
    args = ap.parse_args()
    paths = [args.input/'summary.json'] + [p for c in CONFIGS for p in sorted((args.input/c).glob('*.json'))]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows = {c: {p.name: json.loads(p.read_text()) for p in sorted((args.input/c).glob('*.json'))} for c in CONFIGS}
    assert rows['baseline'], 'No results found'
    assert all(set(rows[c]) == set(rows['baseline']) for c in CONFIGS), 'Task sets differ'
    for c in CONFIGS:
        for name,r in rows[c].items():
            assert (r['domain'], r['task_id']) == (rows['baseline'][name]['domain'], rows['baseline'][name]['task_id'])
            assert not any(t.get('error') for t in r['turns']), f'Errored task: {c}/{name}'
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    metrics = {}; tasks_csv = []; turns_csv = []; violations = []; corrected = []; history_samples = []
    for cfg in CONFIGS:
        cfg_turns = []; cfg_tasks = []
        for name, r in rows[cfg].items():
            task_turns = []
            for t in r['turns']:
                text = t.get('baseline_text','')
                m = token_split(text, tok)
                assert m['cot_tokens'] + m['tool_call_tokens'] + m['post_tool_call_tokens'] == m['total_decode_tokens']
                assert m['total_decode_tokens'] == t['total_decode_tokens'], f'Tokenizer/count drift: {cfg}/{name}'
                parsed = parse_baseline_tool_call(text)
                # Independent JSON decoder check; saved calls should match the text.
                independent = None
                if '<tool_call>' in text:
                    body = text.split('<tool_call>',1)[1].lstrip()
                    try:
                        obj, end = json.JSONDecoder().raw_decode(body)
                        if isinstance(obj,dict) and obj.get('name'):
                            independent = {'name':obj['name'], 'arguments':obj.get('arguments') or {}}
                    except (ValueError, TypeError):
                        pass
                logged = action(t)
                for label, call in [('parser_vs_log', parsed), ('independent_decoder_vs_log', independent)]:
                    if call != logged:
                        violations.append({'config':cfg,'file':name,'turn':t['turn'],'kind':label,'parsed':call,'logged':logged})
                if t.get('spec_used'):
                    pred_name = t.get('committed_probe_name', t.get('probe_tool_name'))
                    pred_args = t.get('committed_probe_args', t.get('probe_tool_args'))
                    if not (t.get('gate_match') and logged and pred_name == logged['name'] and args_exact_match(pred_args,logged['arguments'])):
                        violations.append({'config':cfg,'file':name,'turn':t['turn'],'kind':'reuse_gate_mismatch'})
                probes = t.get('probes', [])
                probe_n = len(probes) if 'probes' in t else int(t.get('probe_dispatch_ms') is not None)
                probe_wall = sum(p.get('probe_wall_s',0) or 0 for p in probes) if 'probes' in t else (t.get('probe_wall_s',0) or 0)
                main = t.get('main_end_ms',0)/1000
                turn_end = t.get('turn_end_ms',0)/1000
                tr = dict(config=cfg, file=name, turn=t['turn'], mode=t['mode'],
                    tool_name=t.get('baseline_tool_name'), tool_turn=bool(logged),
                    total_tokens=m['total_decode_tokens'], pre_tool_tokens=m['cot_tokens'],
                    tool_span_tokens=m['tool_call_tokens'], post_tool_tokens=m['post_tool_call_tokens'],
                    tool_closed=m['tool_call_closed'], tool_parse_ok=m['tool_call_parse_ok'],
                    recorded_main_boundary_s=main, after_main_boundary_s=turn_end-main,
                    turn_wall_s=turn_end, logged_overlap_s=t.get('real_overlap_ms',0)/1000,
                    probe_count=probe_n, summed_probe_request_wall_s=probe_wall,
                    spec_dispatched=bool(t.get('spec_dispatched')), spec_used=bool(t.get('spec_used')),
                    spec_wasted_flag=bool(t.get('spec_wasted')),
                    dispatched_without_reuse=bool(t.get('spec_dispatched') and not t.get('spec_used')),
                    first_token_timeout=bool(t.get('first_token_timeout')),
                    forced_conclusion=t.get('mode')=='forced_conclusion')
                task_turns.append(tr); turns_csv.append(tr); cfg_turns.append(tr)
                corrected.append({'config':cfg,'file':name,'turn':t['turn'],
                                  'original':{k:t.get(k) for k in m},'corrected':m})
                if logged and not m['tool_call_closed'] and len(history_samples)<3:
                    # Exercise the actual history append + actual chat template.
                    from spork_core.turn_runner import _append_tool_messages
                    msgs=[{'role':'user','content':'Check the order.'}]
                    _append_tool_messages(msgs,text,'AUDIT_TOOL_RESULT_PLACEHOLDER')
                    rendered=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=True)
                    history_samples.append({'config':cfg,'file':name,'turn':t['turn'],
                        'saved_close_tag_present': '</tool_call>' in text,
                        'appended_close_tag_present':'</tool_call>' in msgs[-2]['content'],
                        'rendered_close_tag_present':'</tool_call>' in rendered,
                        'rendered_excerpt':rendered[-1800:],
                        'note':'Synthetic user/result placeholders; this verifies formatting propagation, not the historical full prompt.'})
            task = dict(config=cfg, file=name, domain=r['domain'], task_id=r['task_id'], task_idx=r['task_idx'],
                        wall_s=r['wall_s'], turns=len(task_turns), success=r['tau2_success']['reward'])
            for k in ['tool_turn','total_tokens','pre_tool_tokens','tool_span_tokens','post_tool_tokens',
                      'recorded_main_boundary_s','after_main_boundary_s','turn_wall_s','logged_overlap_s',
                      'probe_count','summed_probe_request_wall_s','spec_dispatched','spec_used',
                      'spec_wasted_flag','dispatched_without_reuse','forced_conclusion']:
                task[k]=sum(x[k] for x in task_turns)
            task['outside_turn_timers_s']=r['wall_s']-task['turn_wall_s']
            tasks_csv.append(task); cfg_tasks.append(task)
        n=len(cfg_tasks)
        metrics[cfg]={'n_tasks':n,'n_turns':len(cfg_turns),'successes':sum(x['success'] for x in cfg_tasks),
            'wall_mean_s':statistics.mean(x['wall_s'] for x in cfg_tasks),
            'totals':{k:sum(x[k] for x in cfg_tasks) for k in cfg_tasks[0] if isinstance(cfg_tasks[0][k],(int,float)) and k not in ('task_idx',)},
            'means':{k:statistics.mean(x[k] for x in cfg_tasks) for k in cfg_tasks[0] if isinstance(cfg_tasks[0][k],(int,float)) and k not in ('task_idx',)},
            'missing_close_on_tool_turns':sum(x['tool_turn'] and not x['tool_closed'] for x in cfg_turns),
            'parsed_tool_turns_with_zero_tool_span':sum(x['tool_turn'] and x['tool_span_tokens']==0 for x in cfg_turns),
            'retry_turns':sum(t.get('num_probes',0)>1 for r in rows[cfg].values() for t in r['turns'])}
        print(cfg, 'processed', len(cfg_turns),'turns',flush=True)
    pairs=[]; divergence=[]; comparisons={}
    task_lookup={(t['config'],t['file']):t for t in tasks_csv}
    for cfg in CONFIGS[1:]:
        first_action_counts=collections.Counter(); first_text_counts=collections.Counter()
        for name, b in rows['baseline'].items():
            v=rows[cfg][name]
            diff_action = next((i for i,(x,y) in enumerate(itertools.zip_longest(b['turns'],v['turns'])) if action(x)!=action(y)),None)
            diff_text = next((i for i,(x,y) in enumerate(itertools.zip_longest(b['turns'],v['turns'])) if (x or {}).get('baseline_text')!=(y or {}).get('baseline_text')),None)
            if diff_action is not None:first_action_counts[diff_action+1]+=1
            if diff_text is not None:first_text_counts[diff_text+1]+=1
            bm,vm=task_lookup['baseline',name],task_lookup[cfg,name]
            pair=dict(config=cfg,file=name,domain=b['domain'],task_id=b['task_id'],task_idx=b['task_idx'],
                same_action_trace=trace(b)==trace(v),first_action_diff_turn=None if diff_action is None else diff_action+1,
                first_text_diff_turn=None if diff_text is None else diff_text+1,
                baseline_success=b['tau2_success']['reward'],variant_success=v['tau2_success']['reward'],
                baseline_wall_s=b['wall_s'],variant_wall_s=v['wall_s'],saved_wall_s=b['wall_s']-v['wall_s'])
            for k in ['total_tokens','recorded_main_boundary_s','after_main_boundary_s','outside_turn_timers_s','tool_turn']:
                pair['baseline_minus_variant_'+k]=bm[k]-vm[k]
            pairs.append(pair)
            if diff_action is not None:
                a=b['turns'][diff_action] if diff_action<len(b['turns']) else None
                d=v['turns'][diff_action] if diff_action<len(v['turns']) else None
                prev_b=b['turns'][:diff_action]; prev_v=v['turns'][:diff_action]
                divergence.append({**pair,'baseline_action':action(a),'variant_action':action(d),
                    'prior_saved_history_equal':len(prev_b)==len(prev_v) and all(visible_history(x)==visible_history(y) for x,y in zip(prev_b,prev_v)),
                    'history_equality_limit':'Only saved assistant text + tool-result length/preview; full outputs and states were not logged.',
                    'baseline_raw_text':(a or {}).get('baseline_text',''), 'variant_raw_text':(d or {}).get('baseline_text',''),
                    'raw_text_difference':first_text_difference(a,d),
                    'variant_used_spec_at_divergence':(d or {}).get('spec_used',False),
                    'variant_probe_name':(d or {}).get('committed_probe_name',(d or {}).get('probe_tool_name')),
                    'variant_probe_args':(d or {}).get('committed_probe_args',(d or {}).get('probe_tool_args'))})
        pp=[p for p in pairs if p['config']==cfg]
        same=[p for p in pp if p['same_action_trace']]
        comparisons[cfg]={'same_action_trace_tasks':len(same),'different_action_trace_tasks':len(pp)-len(same),
            'first_action_diff_turn_histogram':dict(first_action_counts),'first_text_diff_turn_histogram':dict(first_text_counts),
            'quality_gain_tasks':sum(p['variant_success']>p['baseline_success'] for p in pp),
            'quality_loss_tasks':sum(p['variant_success']<p['baseline_success'] for p in pp),
            'faster_tasks':sum(p['saved_wall_s']>0 for p in pp),'paired_mean_saving_s':statistics.mean(p['saved_wall_s'] for p in pp),
            'same_action_trace_mean_saving_s':statistics.mean(p['saved_wall_s'] for p in same),
            'same_action_trace_warning':'Post-treatment descriptive subset; not a causal estimate.',
            'mean_saving_decomposition':{k:statistics.mean(p['baseline_minus_variant_'+k] for p in pp) for k in ['recorded_main_boundary_s','after_main_boundary_s','outside_turn_timers_s']}}
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==before[str(p)], f'Input changed during audit: {p}'
    write_csv(args.output/'turn_metrics.csv',turns_csv)
    write_csv(args.output/'task_metrics.csv',tasks_csv)
    write_csv(args.output/'paired_tasks.csv',pairs)
    with (args.output/'token_corrections.jsonl').open('w') as f:
        for row in corrected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump(args.output/'summary.json',{'configs':metrics,'paired_comparisons':comparisons,'violations':violations,
        'measurement_limits':[
            'Token counts re-tokenize returned text; omitted stop tokens cannot be reconstructed as measured generation usage.',
            'cot_tokens is pre-tool text or whole final-answer text, not pure reasoning.',
            'main_end_ms is recorded after controller awaits main; D1/D2 may already have waited for probes. It is not isolated model decode time.',
            'real_overlap_ms is a controller-timestamp estimate capped by tool duration, not an independent measurement of physical overlap.',
            'Probe request wall times overlap with main generation; their sum is neither GPU compute cost nor additive E2E overhead.',
            'Tool-result previews do not establish equality of full observations or environment state.',
            'Success is the local action-match guardrail, not official tau2 task reward.',
            'Single fixed-order run with no discarded warmup; model-length and runtime drift are not identified causally.'
        ]})
    dump(args.output/'first_divergences.json',divergence)
    dump(args.output/'history_format_check.json',history_samples)
    dump(args.output/'input_sha256.json',before)
    dump(args.output/'provenance.json',{'argv':sys.argv,'tokenizer':args.tokenizer,'input':str(args.input),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'tokenizer_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(args.tokenizer).glob('tokenizer*') if p.is_file()},
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('spork_core/qwen.py'),Path('spork_core/turn_runner.py'),Path('spork_core/vllm_client.py')]},
        'inputs_verified_unchanged':True})
    (args.output/'working_changes.patch').write_text(subprocess.check_output(['git','diff'],text=True))
    print(json.dumps({'configs':metrics,'paired_comparisons':comparisons,'violations':violations},indent=2))

if __name__=='__main__':
    main()
