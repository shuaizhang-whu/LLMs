"""Read existing experiment text logs only; export a provenance-rich audit.

Never loads checkpoints or executes training/evaluation. Raw metric strings
and LF-based line numbers are retained. Launcher copies are indexed separately.
"""
from pathlib import Path
import csv
import html
import json
import re
from collections import Counter

OUT = Path(__file__).resolve().parent
FR = OUT.parent
SAES = FR.parent / 'SAES-SVD'
NUM = r'([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)'
PPL = re.compile(r"PPL after pruning:\s*\{[^\n}]*?'wikitext2':\s*(?:np\.float64\()?" + NUM)


def read(p):
    return p.read_bytes().decode('utf-8', errors='replace')


def pick(s, pattern, default='', flags=re.M):
    m = re.search(pattern, s, flags)
    return m.group(1).strip() if m else default


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def parse(config, metric=None, **overrides):
    s = read(config)
    # Read the first header; later diagnostic lines can use different versions.
    head = '\n'.join(s.split('\n')[:85])
    metric = metric or config
    ms = read(metric)
    matches = list(PPL.finditer(ms))
    assert len(matches) <= 1, f'Aggregate log passed as single experiment: {metric}'
    match = matches[0] if matches else None
    if match:
        status = 'measured'
    elif re.search(r'out of memory|CUDA OOM', s + ms, re.I):
        status = 'OOM_no_PPL'
    elif 'Traceback (most recent call last)' in s + ms:
        status = 'error_no_PPL'
    else:
        status = 'no_PPL_in_saved_log'
    method = 'SAES-SVD' if 'SAES-SVD' in head else 'FR-SVD'
    version = pick(head, r'(?:SAES|FR)-SVD\s+(v[0-9.]+)')
    model = pick(head, r'^\s*(?:模型|Model):\s*(\S+)')
    if model == 'None':
        model = ''
    alpha = re.search(r'(?i)Alpha(?: range| 范围)?\s*[:=]\s*\[' + NUM + r',\s*' + NUM + r'\]', head)
    rho = pick(head, r'(?i)\brho\s*[:=]\s*' + NUM)
    fisher = pick(head, r'^\s*Fisher 模式:\s*([^\n]+)')
    if not fisher:
        fm = re.search(r'创新点A \(([^)]+)\):\s*(ON|OFF)', head)
        if fm:
            fisher = fm.group(1) if fm.group(2) == 'ON' else 'OFF'
    stage2 = pick(head, r'(?:创新点B \([^\n]+?\)|Innovation2[^:\n]*):\s*(ON|OFF)')
    code = pick(head, r'^Script:\s*(\S+\.py)')
    record = {
        'experiment': config.parent.name if config.name in ('compress.log','run.log') else config.stem,
        'group': '', 'scope': '', 'method': method, 'code_version': version,
        'code_file': code, 'version_evidence': 'log_header' if version else '',
        'model_path': model, 'compression_ratio_removed': pick(head, r'(?:压缩率:|Compression ratio:|Ratio:)\s*' + NUM),
        'dataset': pick(head, r'(?:数据集|Dataset):\s*([\w-]+)'),
        'nsamples': pick(head, r'(?:样本数:|Samples:)\s*(\d+)'),
        'seq_len': pick(head, r'(?:Seq Length:|SeqLen:)\s*(\d+)'),
        'seed_cli': pick(head, r'(?:^\s*Seed:)\s*(\d+)'),
        'rho': rho, 'alpha_min': alpha.group(1) if alpha else '',
        'alpha_max': alpha.group(2) if alpha else '',
        'r1_ratio': pick(head, r'r1_ratio:\s*' + NUM),
        'eps': pick(head, r'(?:\beps\s*[:=]|Epsilon:)\s*' + NUM),
        'lambda_psi': pick(head, r'lambda_psi\s*[:=]\s*' + NUM),
        'lambda_phi': pick(head, r'lambda_phi\s*[:=]\s*' + NUM),
        'fisher_mode': fisher, 'stage2': stage2,
        'ppl': match.group(1) if match else '', 'status': status,
        'config_log': str(config), 'metric_log': str(metric),
        'metric_line': ms[:match.start()].count('\n') + 1 if match else '',
        'supplement_evidence': '', 'notes': '',
    }
    record.update(overrides)
    if not record['code_file'] and re.fullmatch(r'v[0-9.]+', record['code_version']):
        record['code_file'] = ('SAES-SVD_' if method == 'SAES-SVD' else 'FR_SVD_') + record['code_version'] + '.py'
    return record


rows = []
# Phase2: one named experiment file each. Do not treat monitoring copies as runs.
for p in sorted((FR / 'result/phase2').glob('*.log')):
    if not re.match(r'(?:fisher_direct_|rho\d|r1abl_|ratio04_|ns\d|（[\d.]+）ns\d)', p.name):
        continue
    r = parse(p, code_version='v1.3', code_file='FR_SVD_v1.3.py')
    r['seq_len'], r['seed_cli'] = '2048', '3'
    seed = re.search(r'_seed(\d+)', p.name)
    if seed:
        r['seed_cli'] = seed.group(1)
    r['lambda_psi'] = '1e-5' if ('lam1e5' in p.name or '_l1e5_' in p.name) else ('1e-3' if 'lam1e3' in p.name else '1e-4')
    r['supplement_evidence'] = 'scripts/phase2/run_phase2_gpu4.sh,run_phase2_gpu5.sh,run_phase2_gpu6.sh,run_phase2_gpu7.sh,run_phase3_gpu67.sh; scripts/ablation/run_nsamples_ablation.sh,run_r1ratio_ablation.sh,run_ratio46_r1_ablation.sh; FR_SVD_v1.3.py defaults'
    r['notes'] = 'seq_len/seed由对应脚本补充；lambda_psi按对应脚本与当前代码默认补充，非日志头直接打印；v1.3代码文件可修改，未记录commit/hash。'
    if seed:
        r['notes'] += '校准调用未传args.seed，当前代码默认校准seed=3，不能视为独立校准种子。'
    if r['fisher_mode'] == 'OFF':
        r['lambda_psi'] = ''
    if float(r['compression_ratio_removed']) == 0.4:
        r['group'] = 'FR v1.3 ratio0.4 联合参数'
        r['scope'] = 'joint_parameters'
    elif int(r['nsamples']) != 128:
        r['group'] = 'FR v1.3 校准样本数及联合参数'
        r['scope'] = 'joint_parameters'
    elif seed or re.search(r'_eps|_lam', p.name):
        r['group'] = 'FR v1.3 阻尼或名义seed对照'
        r['scope'] = 'other_parameter_control'
    elif float(r['r1_ratio']) != 0.9 or p.name.startswith('r1abl_'):
        r['group'] = 'FR v1.3 r1与rho/alpha联合对照'
        r['scope'] = 'joint_parameters'
    else:
        r['group'] = 'FR v1.3 Fisher OFF rho/alpha扫描' if r['fisher_mode'] == 'OFF' else 'FR v1.3 Fisher direct rho/alpha扫描'
        r['scope'] = 'rho_alpha_sweep'
    rows.append(r)

# SAES alpha grid: version is v2.1.3, NOT v2.1.4.
for folder in ('alpha_ablation_gpu0', 'alpha_ablation_gpu1', 'v2.1.4_contrast_experiments', 'nsamples_ablation'):
    for p in sorted((SAES / 'result' / folder).glob('*.log')):
        r = parse(p, seq_len='2048', seed_cli='3', fisher_mode='not_applicable', stage2='not_applicable')
        if 'alpha_ablation' in folder:
            r.update(group='SAES v2.1.3 alpha扫描', scope='rho_alpha_sweep', code_version='v2.1.3', code_file='SAES-SVD_v2.1.3.py')
            script = 'run_alpha_ablation_' + folder.rsplit('_', 1)[1] + '.sh'
            r['notes'] = '日志头及启动脚本均确认v2.1.3；当前项目未找到该版本py文件。'
            # The failed B1 log still provides a header; do not impute a PPL.
        elif folder.startswith('v2.1.4'):
            r.update(group='SAES v2.1.4 rho对照', scope='rho_alpha_sweep')
            script = 'run_contrast_experiments.sh'
        else:
            r.update(group='SAES v2.1.4 校准样本数对照', scope='other_parameter_control')
            script = 'run_nsamples_ablation.sh'
        r['supplement_evidence'] = str(SAES / 'scripts/ablation' / script)
        rows.append(r)

# Early FR implementation records. Preserve uncertainty of overwritten v1 code.
for folder in ('FR_SVD_v1', 'FR_SVD_diag', 'FR_SVD_planA', 'FR_SVD_planB'):
    for p in sorted((FR / 'result' / folder).rglob('log*.txt')):
        r = parse(p, seq_len='2048', seed_cli='3')
        if folder == 'FR_SVD_planA':
            r.update(code_version='v1.1', code_file='FR_SVD_v1.1.py', version_evidence='Script field in run log',
                     group='FR v1.1 额外秩方案', scope='historical_joint_parameters')
            r['notes'] = '该版本Stage1使用完整r，Stage2增加额外秩；名义ratio=.2不保证与总秩固定版本相同实际压缩量。当前v1.1代码说明Fisher Phase B禁用。'
            r['fisher_mode'] = 'disabled (current v1.1 code; no run hash)'
        elif folder == 'FR_SVD_planB':
            r.update(code_version='v1.0', code_file='FR_SVD_v1.0.py', version_evidence='Script field in run log',
                     group='FR v1.0 单阶段rho扫描', scope='rho_alpha_sweep')
            r['fisher_mode'] = 'disabled (current v1.0 code; no run hash)'
        elif folder == 'FR_SVD_diag':
            r.update(code_version='v1.0', code_file='FR_SVD_v1.0.py', version_evidence='scripts/utils/run_fr_svd_diagnostic.sh',
                     group='FR v1.0 诊断rho/r1对照', scope='historical_joint_parameters')
            r['fisher_mode'] = 'disabled (current v1.0 code; no run hash)'
        else:
            r.update(code_version='early v1 historical snapshot; exact file unconfirmed',
                     code_file='', version_evidence='directory FR_SVD_v1; no script name/hash in saved log',
                     group='FR 早期v1 历史记录', scope='historical_joint_parameters')
            r['notes'] = '不能由目录v1认定当前FR_SVD_v1.0.py就是当时快照；两条日志算法横幅不同，不能把差异全归因rho/alpha。'
            if 'Fisher 样本数' in read(p):
                r['fisher_mode'] = 'early Fisher-guided; max 32 samples in log'
        if not r['stage2']:
            r['stage2'] = 'OFF (r2=0)' if r['r1_ratio'] == '1.0' else 'ON (rank allocation in log)'
        rows.append(r)

for folder, version in (('ablation_v1.2','v1.2'),('ablation_v1.3','v1.3'),('ablation_v1.3_phase1','v1.3')):
    for p in sorted((FR / 'result' / folder).glob('exp*.log')):
        r = parse(p, seq_len='2048', seed_cli='3', code_version=version, code_file=f'FR_SVD_{version}.py',
                  group='FR '+version+' 组件与rho/alpha组合 ('+folder+')', scope='component_and_joint_control')
        r['version_evidence'] = 'log header and scripts/ablation/'+ ('run_ablation_v1.2.sh' if version=='v1.2' else ('run_fr_svd_v1.3_phase1.sh' if folder.endswith('phase1') else 'run_fr_svd_v1.3_ablation.sh'))
        r['supplement_evidence'] = r['version_evidence'] + '; lambda_psi defaults in current code'
        r['lambda_psi'] = '1e-4' if r['fisher_mode'] != 'OFF' else ''
        r['notes'] = '同名v1.3早期日志使用soft-reweighting；phase2有none/direct，按Fisher模式分别看。'
        rows.append(r)

# Later versions: compression header takes precedence over eval-only defaults.
for p in sorted((FR / 'result/FR_SVD_v1.6').rglob('compress.log')):
    evals = sorted(p.parent.glob('*eval.log'))
    r = parse(p, evals[0] if evals else p, group='FR v1.6 固定rho/alpha组件对照', scope='fixed_rho_alpha_other_experiment')
    runner = p.parent / 'run.log'
    run = read(runner) if runner.exists() else ''
    r['seq_len'] = pick(run, r'SeqLen:\s*(\d+)')
    r['supplement_evidence'] = str(runner)
    r['notes'] = '配置取compress.log，不取仅评估时的默认参数。'+pick(read(p),r'^FR-SVD v1\.6: ([^\n]+)')
    if r['stage2'] == 'OFF':
        r['notes'] += '；日志说明r1_ratio在关闭innovation2时忽略，全部秩用于Stage1。'
    rows.append(r)

bench = FR / 'result/llama7b_time_benchmark_20260427_021522'
for p in sorted(bench.glob('*/compress.log')):
    if p.parent.name == 'svdllm_whitening_only':
        continue  # no rho/alpha mechanism; indexed in raw metrics.
    r = parse(p, p.parent/'eval.log', group='LLaMA7B时间基准 固定参数/组件对照', scope='fixed_rho_alpha_other_experiment')
    r['seq_len'] = pick(read(p), r'--model_seq_len\s+(\d+)')
    r['supplement_evidence'] = 'command captured in '+str(p)+':5'
    r['notes'] = '不是独立rho/alpha扫描；FR与SAES默认rho/alpha不同。Fisher soft与创新点开关需单独看。'
    rows.append(r)

for p in sorted((SAES / 'result/SAES_SVD_v3').rglob('run.log')):
    r = parse(p, group='SAES v3 Qwen固定rho/alpha重复开发运行', scope='fixed_rho_alpha_other_experiment')
    r['notes'] = 'rho/alpha固定；不能把多次运行的大幅差异归因为rho/alpha变化。无历史commit，代码/缓存等其他变化未锁定。'
    rows.append(r)
for p in sorted((FR / 'result/FR_SVD_v1.5').rglob('run.log')) + sorted((FR / 'result/FR_SVD_v1.5_saes_baseline').rglob('run.log')):
    ev = p.parent / 'eval.log'
    metric = ev if ev.exists() and PPL.search(read(ev)) else p
    r = parse(p, metric, group='FR v1.5 其它模型固定配置', scope='fixed_rho_alpha_other_experiment')
    r['notes'] = '不是独立rho/alpha扫描；压缩配置取run.log，eval日志默认配置不能作为压缩参数。'
    rows.append(r)

# Qwen has eval only: report uncertain compression metadata instead of eval defaults.
p = FR / 'result/FR_SVD_v1.5/qwen2_5_7b/ns256_r1_0.92_rho1.0_20260417_211008/eval.log'
rows.append(parse(p, code_version='v1.5', code_file='FR_SVD_v1.5.py',
                  group='FR v1.5 其它模型固定配置', scope='fixed_rho_alpha_other_experiment',
                  model_path='Qwen2.5-7B (directory label; original path unverified)',
                  nsamples='256 (directory label)', r1_ratio='0.92 (directory label)', rho='1.0 (directory label)',
                  compression_ratio_removed='', alpha_min='', alpha_max='', eps='', fisher_mode='', stage2='',
                  notes='只有eval.log；ns256/r1=.92/rho=1由目录标签指示，不能把仅评估默认ns128/r1=.8/OFF当成压缩配置；其余参数未确认。'))

# One failed 30B launch with a real compression header, not a monitoring copy.
p = FR / 'result/FR_SVD_v1.5/llama_30b/llama_30b_gpu5_r1_0.92.log'
rows.append(parse(p, group='FR v1.5 其它模型固定配置', scope='fixed_rho_alpha_other_experiment', notes='失败/未完成启动，和后续ns64运行分列。'))

# Stable order and row IDs. Equal numeric results remain distinct saved records.
rows.sort(key=lambda r:(r['group'],r['config_log']))
for i, r in enumerate(rows, 1):
    r['id'] = f'R{i:03d}'

measured = [r for r in rows if r['status']=='measured']
pending = [r for r in rows if r['status']!='measured']
write_csv('all_related_runs.csv', rows)
write_csv('measured_results.csv', measured)
write_csv('incomplete_runs.csv', pending)
write_csv('rho_alpha_sweeps.csv', [r for r in rows if r['scope']=='rho_alpha_sweep'])

# Inventory every text log and every raw PPL occurrence, including launcher copies
# and LoRA evaluations. This documents audit coverage without miscounting repeats.
canonical = {r['metric_log']:r['id'] for r in rows}
configs = {r['config_log']:r['id'] for r in rows}
occurrences, inventory = [], []
for root in (FR/'result', SAES/'result'):
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix not in ('.log','.txt','.out','.csv','.json','.md'):
            continue
        s = read(p)
        hits = list(PPL.finditer(s))
        disposition = 'canonical_metric' if str(p) in canonical else ('canonical_config' if str(p) in configs else ('LoRA/fine-tuning; not rho-alpha sensitivity' if 'lora' in str(p).lower() else ('SVDLLM whitening baseline; no rho/alpha' if 'svdllm_whitening_only' in str(p) else 'auxiliary/aggregate/reference; not counted as independent run')))
        inventory.append({'path':str(p),'ppl_occurrences':len(hits),'disposition':disposition,'record_id':canonical.get(str(p),configs.get(str(p),''))})
        for m in hits:
            occurrences.append({'source_log':str(p),'line':s[:m.start()].count('\n')+1,'ppl':m.group(1),'disposition':disposition,'record_id':canonical.get(str(p),'')})
write_csv('raw_ppl_occurrences.csv', occurrences)
write_csv('file_inventory.csv', inventory)

# Verify provenance: each reported PPL is exactly the literal at the saved line.
for r in measured:
    line = read(Path(r['metric_log'])).split('\n')[int(r['metric_line'])-1]
    assert PPL.search(line).group(1) == r['ppl'], r
    assert all(r[k] for k in ('rho','nsamples','code_version')), r
summary = {
    'audited_text_files':len(inventory), 'raw_ppl_occurrences_including_copies':len(occurrences),
    'related_saved_runs':len(rows), 'measured_saved_runs':len(measured), 'incomplete_saved_runs':len(pending),
    'groups':dict(Counter(r['group'] for r in measured)),
    'pure_rho_alpha_measured_rows':sum(r['scope']=='rho_alpha_sweep' for r in measured),
    'scope':'Two user-specified result roots only; no predictions; no new experiments.',
    'limitations':'Version labels are not immutable snapshots. Equal results do not prove independent repeats. Unknown fields remain blank. LoRA and SVDLLM indexed in raw inventory, not treated as rho/alpha sensitivity.'
}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(OUT/'all_related_runs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')

# A plain text, self-contained copy is convenient for reading without a browser.
report = [
    'ρ / αmin / αmax 真实结果全量核查（不含预测）',
    '范围：'+str(FR/'result')+'；'+str(SAES/'result'),
    f'共扫描{len(inventory)}个文本文件；关联保存记录{len(rows)}条，其中实测PPL {len(measured)}条，未完成/失败{len(pending)}条。',
    f'原始日志输出{len(occurrences)}次含launcher复制；不等于{len(occurrences)}次独立实验。相同数值记录仍保留，不能当作重复测量样本计算统计量。',
    '以下ratio是删除比例（.2=去掉20%、保留80%）。PlanA v1.1采用额外秩，实际参数量需另核。',
    'N=校准样本数；lambda_psi/seed/seq_len等由脚本补充时见该条supplement_evidence。空白表示未记录/未确认。',
    '版本是日志或脚本标识，不是不可变commit；历史同名py可能有修改。仅评估程序的默认参数没有当作压缩配置。',
    'LoRA评估与无rho/alpha的SVDLLM基线在raw_ppl_occurrences.csv保留，不作为rho/alpha敏感性记录。',
    '',
]
for group in sorted({r['group'] for r in rows}):
    report += ['='*72, group, '='*72]
    for r in (x for x in rows if x['group']==group):
        report += [
            f"{r['id']}  {r['experiment']}",
            f"  PPL={r['ppl'] or '无实测输出'}；状态={r['status']}；范围={r['scope']}",
            f"  模型={r['model_path'] or '未确认'}；数据集={r['dataset']}；压缩ratio={r['compression_ratio_removed'] or '未确认'}；N={r['nsamples']}；seq_len={r['seq_len'] or '未记录'}；CLI seed={r['seed_cli'] or '未记录'}",
            f"  ρ={r['rho']}；α=[{r['alpha_min']},{r['alpha_max']}]；r1/r={r['r1_ratio'] or '未记录/不适用'}；eps={r['eps']}；lambda_psi={r['lambda_psi'] or '未记录/不适用'}；lambda_phi={r['lambda_phi'] or '未记录/不适用'}",
            f"  Fisher={r['fisher_mode'] or '未记录/不适用'}；Stage2={r['stage2'] or '未记录/不适用'}",
            f"  版本={r['code_version']}；代码={r['code_file'] or '精确文件未确认'}；依据={r['version_evidence']}",
            f"  配置日志={r['config_log']}",
            f"  PPL日志={r['metric_log']}:{r['metric_line']}",
            f"  补充依据={r['supplement_evidence']}",
            f"  说明={r['notes']}", '',
        ]
(OUT/'all_results.txt').write_text('\n'.join(report),encoding='utf-8')

# Standalone local HTML report with filterable complete tables.
esc = html.escape
fields = [('id','ID'),('experiment','实验'),('code_version','代码版本'),('model_path','模型'),
          ('compression_ratio_removed','压缩率'),('nsamples','校准N'),('seq_len','序列长'),('seed_cli','CLI seed'),
          ('rho','ρ'),('alpha_min','αmin'),('alpha_max','αmax'),('r1_ratio','r1/r'),('eps','eps'),
          ('lambda_psi','λψ'),('lambda_phi','λφ'),('fisher_mode','Fisher'),('stage2','Stage2'),('ppl','WikiText-2 PPL'),('status','状态')]
doc = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ρ/α 实测结果全量核查</title>',
       '<style>body{font:15px/1.6 system-ui,sans-serif;margin:24px;color:#17212b}h1{font-size:26px}h2{font-size:20px;margin-top:32px}.box{background:#eef4fa;padding:16px}input{font:inherit;padding:10px;width:min(650px,90%)}.scroll{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}th,td{border:1px solid #ccd6df;padding:7px;vertical-align:top}th{background:#edf2f6;position:sticky;top:0}td{min-width:55px}.details td{background:#fafafa;white-space:normal}a{color:#165ea6}code{overflow-wrap:anywhere}tr:hover{background:#f3f8fc}</style>',
       '<h1>ρ、αmin、αmax：真实实验结果全量核查</h1>',
       '<p>扫描范围：Projects/FR-SVD/result 与 Projects/SAES-SVD/result。所有PPL直接来自已有日志，没有预测，没有新实验。</p>',
       '<div class="box"><b>结论修正：</b>大波动真实存在，尤其SAES v2.1.3的ρ=.9、α=[.2,.8]（9.545778），以及FR v1.3 Fisher OFF中ρ=.4的α范围变化（7.854064→8.132422）。之前局部Fisher direct近邻结果不能概括全部历史实验。<br>代码版本按日志/脚本标注，不等于不可变历史代码快照；旧FR目录v1未记载精确脚本名。ratio=.2表示名义压缩20%、保留80%。PlanA增加额外秩，不能只按该ratio认定实际参数量相同。<br>空白=未记录或无法确认；不是0。PPL相同的保存记录不能直接视为独立复现实验。仅评估模式打印的默认参数不能当作压缩参数。</div>',
       '<p>文本文件：'+str(len(inventory))+'；含PPL的原始输出次数（含副本）：'+str(len(occurrences))+'；选定关联记录：'+str(len(rows))+'；有实测PPL：'+str(len(measured))+'；无PPL：'+str(len(pending))+'.</p>',
       '<p><a href="measured_results.csv">实测CSV</a> · <a href="rho_alpha_sweeps.csv">直接rho/alpha扫描CSV</a> · <a href="all_related_runs.csv">所有关联记录CSV</a> · <a href="all_results.txt">完整逐条文字清单</a> · <a href="incomplete_runs.csv">未完成CSV</a> · <a href="raw_ppl_occurrences.csv">全部原始PPL输出索引（含副本/LoRA）</a> · <a href="file_inventory.csv">扫描清单</a></p>',
       '<input id="filter" placeholder="筛选实验、代码版本、模型或参数（例：rho0.4、v2.1.3）" aria-label="筛选结果">',
       '<p>核心扫描 scope=rho_alpha_sweep；joint/component/control/历史/其它模型分别附列，避免把它们当作只改变ρ/α的对照。每行下方可查看证据、说明与代码文件名。</p>']
for group in sorted({r['group'] for r in rows}):
    group_rows = [r for r in rows if r['group']==group]
    doc += ['<section><h2>'+esc(group)+'</h2><div class="scroll"><table><thead><tr>'+''.join('<th>'+label+'</th>' for _,label in fields)+'</tr></thead>']
    for r in group_rows:
        doc += ['<tbody class="record"><tr>'+''.join('<td>'+esc(str(r[k]))+'</td>' for k,_ in fields)+'</tr>']
        details = f"scope={r['scope']}；code={r['code_file'] or '未确认'}；版本依据={r['version_evidence']}；说明={r['notes']}"
        doc += ['<tr class="details"><td colspan="'+str(len(fields))+'"><details><summary>来源与配置说明</summary><p>'+esc(details)+'</p><p>配置日志：<code>'+esc(r['config_log'])+'</code><br>指标日志：<code>'+esc(r['metric_log'])+':'+str(r['metric_line'])+'</code><br>补充依据：<code>'+esc(r['supplement_evidence'])+'</code></p></details></td></tr></tbody>']
    doc += ['</table></div></section>']
doc += ['<script>document.getElementById("filter").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll("tbody.record").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))})</script></html>']
(OUT/'report.html').write_text('\n'.join(doc),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
