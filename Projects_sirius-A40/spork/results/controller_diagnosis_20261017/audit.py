"""Short, stdlib-only audit of sealed local results; never runs inference."""
import collections as C
import csv
import hashlib
import json
from pathlib import Path
import statistics as S

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/controller_full_20261017_021939_Wid7sh'
DATA = ROOT / 'prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du'
OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
mean = lambda xs: S.mean(xs) if xs else None

manifest = read(RUN / 'manifest.json')
records = []
for item in manifest['schedule']:
    p = RUN / item['result']
    marker = p.with_suffix('.complete.json')
    if marker.exists():
        assert sha(p) == read(marker)['sha256'], p
        records.append(read(p))
grouped = C.defaultdict(dict)
for r in records:
    grouped[(r['group_id'], r['repeat'])][r['config']] = r
fields = ('baseline_has_tool_call', 'baseline_tool_name', 'baseline_tool_args',
          'tool_result_sha256', 'environment_after', 'main_token_ids')
signature = lambda r: ([{k: t[k] for k in fields} for t in r['turns']], r['action_reward'])
valid = {k: v for k, v in grouped.items() if set(v) == set(manifest['configs'])
         and not any(r['errors'] for r in v.values())
         and all(signature(r) == signature(v['timing_off']) for r in v.values())}
report = {'provenance': {'run': str(RUN.relative_to(ROOT)), 'manifest_sha256': sha(RUN / 'manifest.json'),
                       'training_rows_sha256': sha(DATA / 'training_rows.jsonl'),
                       'note': 'Local snapshot; one live repeat; empirical maxima select timing noise.'},
          'sealed_runs': len(records), 'failed_runs': sum(bool(r['errors']) for r in records),
          'valid_tasks': len(valid), 'splits': dict(C.Counter(v['timing_off']['split'] for v in valid.values())),
          'online': {}, 'by_turn': {}, 'labels': {}, 'cross_collection': {}}
flat = []
for mode in manifest['configs']:
    rs = [v[mode] for v in valid.values()]
    ts = [t for r in rs for t in r['turns']]
    probes = [p for t in ts for p in t['probes']]
    first = [t['probes'][0] for t in ts if t['probes']]
    used = [t for t in ts if t['spec_used']]
    slack = [(t['main_end_ms'] - t['spec_end_ms']) / 1000 for t in used]
    tools = [t for t in ts if t['baseline_has_tool_call']]
    calls = [t['launch_timing']['controller_inference_ms'] for t in ts if t['launch_timing']['controller_calls']]
    report['online'][mode] = {
        'task_mean_s': mean([r['wall_s'] for r in rs]), 'turns': len(ts), 'tool_turns': len(tools),
        'first_probe_mean_s': mean([p['probe_dispatch_ms']/1000 for p in first]),
        'first_probe_prefix_counts': dict(C.Counter(p['tokens_at_dispatch'] for p in first)),
        'probe_calls': len(probes), 'turns_with_retries': sum(len(t['probes']) > 1 for t in ts),
        'spec_used': len(used), 'spec_wasted': sum(t['spec_wasted'] for t in ts),
        'used_tools_ready_before_actor': sum(x >= 0 for x in slack),
        'used_tool_slack_median_s': S.median(slack) if slack else None,
        'used_tool_slack_min_s': min(slack) if slack else None,
        'main_mean_s': mean([t['main_end_ms']/1000 for t in ts]),
        'tool_wait_mean_s': mean([(t['tool_result_ready_ms']-t['main_end_ms'])/1000 for t in ts]),
        'controller_mean_ms': mean(calls),
        'controller_sum_per_task_s': sum(calls)/1000/len(rs),
        'policies': dict(C.Counter(t['launch_timing']['policy'] for t in ts)),
        'no_tool_turn_probe_calls': sum(len(t['probes']) for t in ts if not t['baseline_has_tool_call']),
        'tool_real_mean_s': mean([t.get('spec_real_tool_wall_s', t['real_tool_wall_s']) for t in tools]),
    }
    for label in ('first_two', 'later'):
        sub = [t for t in ts if (t['turn'] <= 2) == (label == 'first_two')]
        report['by_turn'].setdefault(label, {})[mode] = {
            'n': len(sub), 'mean_turn_s': mean([t['turn_end_ms']/1000 for t in sub]),
            'policies': dict(C.Counter(t['launch_timing']['policy'] for t in sub))}
for (group, rep), modes in valid.items():
    for i in range(len(modes['timing_off']['turns'])):
        row = {'group': group, 'split': modes['timing_off']['split'], 'turn': i+1}
        for mode, record in modes.items():
            t = record['turns'][i]
            row[mode + '_turn_s'] = t['turn_end_ms']/1000
            row[mode + '_first_probe_s'] = t['probes'][0]['probe_dispatch_ms']/1000 if t['probes'] else None
            row[mode + '_policy'] = t['launch_timing']['policy']
        row['bert_saved_vs_wait_s'] = row['timing_wait_turn_s'] - row['timing_bert_turn_s']
        flat.append(row)
rows = [json.loads(x) for x in (DATA / 'training_rows.jsonl').read_text().splitlines()]
for split in ('train', 'validation'):
    rs = [r for r in rows if r['split'] == split and r['label_valid']]
    audits = [read(DATA / 'tasks' / r['group_id'].replace(':', '_') /
                   f"turn_{r['metadata']['turn']:03d}" / 'label_audit.json') for r in rs]
    deltas = [a['nominal_gain_intervals_s']['now_minus_wait'] for a in audits]
    report['labels'][split] = {
        'n': len(rs), 'tasks': len({r['group_id'] for r in rs}),
        'oracle_policy_counts': dict(C.Counter(max(('off','now','wait'), key=lambda p:
                         0 if p == 'off' else r['targets']['gain_'+p+'_s']) for r in rs)),
        'median_abs_now_wait_gap_s': S.median(abs(d['mean']) for d in deltas),
        'median_now_wait_interval_half_width_s': S.median(d['half_width'] for d in deltas),
        'now_wait_interval_includes_zero': sum(d['lower'] <= 0 <= d['upper'] for d in deltas),
        'all_features_budgets': sorted(set((r['input']['visible_generated_tokens'], r['input']['wait_tokens']) for r in rs)),
        'oracle_over_wait_mean_per_prefix_s': mean([max(0,*r['targets'].values())-r['targets']['gain_wait_s'] for r in rs]),
        'now_best_rows': [{'id': r['id'], 'gain_over_wait_s': r['targets']['gain_now_s']-r['targets']['gain_wait_s']}
                         for r in rs if r['targets']['gain_now_s'] > max(0,r['targets']['gain_wait_s'])],
    }
lookup = {(r['group_id'],r['metadata']['turn']): r for r in rows if r['label_valid']}
matched_training = []
for (group, _), modes in valid.items():
    for i, t in enumerate(modes['timing_bert']['turns'],1):
        old = lookup.get((group,i))
        if old:
            d = {'group':group, 'turn':i, 'prefix_equal': t['launch_timing']['decision_prefix_text']==old['input']['prefix_text'],
                 'replay_now_better': old['targets']['gain_now_s'] > old['targets']['gain_wait_s'],
                 'live_now_better': modes['timing_now']['turns'][i-1]['turn_end_ms'] < modes['timing_wait']['turns'][i-1]['turn_end_ms']}
            matched_training.append(d)
report['cross_collection'] = {'shared_prefixes':len(matched_training),
    'identical_prefix_text':sum(d['prefix_equal'] for d in matched_training),
    'same_now_wait_winner':sum(d['replay_now_better']==d['live_now_better'] for d in matched_training),
    'rows':matched_training}
tok = read(ROOT/'controller_runs/training_compare_e200_20261016_165218_ifqRYU/full_seed_42/tokenization_report.json')
report['encoding'] = {k: {'clipped_rows':sum(t['fields'][k]['used'] < t['fields'][k]['original'] for t in tok),
                        'mean_tokens_used':mean([t['fields'][k]['used'] for t in tok])}
                      for k in ['task','context','observation','reasoning']}
report['provenance'].update(audit_script_sha256=sha(Path(__file__)), local_snapshot_only=True)
exceptions = sorted(report['labels']['validation']['now_best_rows'],
                    key=lambda r: r['gain_over_wait_s'], reverse=True)
report['validation_gain_concentration'] = {
    'top_two_prefix_ids': [r['id'] for r in exceptions[:2]],
    'oracle_extra_gain_fraction': sum(r['gain_over_wait_s'] for r in exceptions[:2]) /
        (report['labels']['validation']['oracle_over_wait_mean_per_prefix_s'] * report['labels']['validation']['n']),
    'note': 'These are checkpoint-selection validation labels, not held-out online evidence.'}
reuse = C.Counter((a['spec_used'], b['spec_used']) for modes in valid.values()
                  for a, b in zip(modes['timing_now']['turns'], modes['timing_wait']['turns']))
report['online_now_wait_reuse_patterns'] = {'both_reuse': reuse[True, True], 'now_only': reuse[True, False],
                                          'wait_only': reuse[False, True], 'neither': reuse[False, False]}
report['single_run_posthoc_headroom'] = {}
for name, choices in [('now_wait', ['timing_now', 'timing_wait']),
                      ('including_early_and_off', ['timing_off', 'timing_spork', 'timing_now', 'timing_wait'])]:
    gains = [r['timing_wait_turn_s'] - min(r[m + '_turn_s'] for m in choices) for r in flat]
    report['single_run_posthoc_headroom'][name] = {
        'mean_per_turn_s': mean(gains), 'total_per_task_s': sum(gains)/len(valid),
        'scope': 'Post-hoc minimum of single-run turn times; includes timing noise, not a reliable speedup ceiling.'}
OUT.mkdir(exist_ok=True)
(OUT/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (OUT/'per_turn.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
print(json.dumps({k:v for k,v in report.items() if k not in ['cross_collection','by_turn']},ensure_ascii=False,indent=2))
print('BY_TURN',json.dumps(report['by_turn']))
print('CROSS_COLLECTION',{k:v for k,v in report['cross_collection'].items() if k!='rows'})
print('OUTPUT',OUT)
