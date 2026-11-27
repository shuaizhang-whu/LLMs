from pathlib import Path
import json,subprocess,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
P=Path(__file__).resolve().parent
fontpath=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
fm.fontManager.addfont(fontpath);plt.rcParams.update({'font.family':fm.FontProperties(fname=fontpath).get_name(),'axes.unicode_minus':False,'font.size':21,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
D=json.loads((P/'experiment-data.json').read_text());old=D['research_runs/full_eval_review_20261026/audit.json'];full=D['results/qwen35_9b_expanded_20261101_v7/expanded/summary.json'];micro=D['results/qwen35_9b_mechanism_20261102_v2/summary.json'];rows=[]
for name,title,labels,vals,detail in [
 ('32b','Qwen3-32B｜固定用户消息',['SPORK','TurnAhead (raw)'],[old['splits']['all']['spork_mean_s'],old['splits']['all']['successor_mean_s']],'148 个有效配对 · 最多 8 轮'),
 ('9b','Qwen3.5-9B｜完整用户交互',['SPORK','raw','no_probe','one_probe'],[full['methods'][m]['mean_agent_pipeline_s'] for m in ['spork','raw','raw_no_probe','raw_one_probe']],'20 个严格配对 · 首次重复')]:
 fig,ax=plt.subplots(figsize=(7.6,3.7 if name=='9b' else 3.7));colors=['#77818C','#A60000','#00807B','#394B69'][:len(vals)]
 ax.barh(range(len(vals)),vals,color=colors,height=.50)
 ax.set_yticks(range(len(vals)),labels);ax.invert_yaxis();ax.set_xlim(0,450);ax.tick_params(axis='y',length=0,labelsize=21)
 ax.set_xlabel('平均耗时（秒）',fontsize=21);ax.xaxis.grid(True,color='#E4E7EB',zorder=0);ax.set_axisbelow(True)
 for i,v in enumerate(vals):
  pct=100*(1-v/vals[0]);suffix=''
  ax.text(v+6,i,f'{v:.1f}{suffix}',va='center',fontsize=21)
  rows.append([name,labels[i],v,pct])
 ax.set_title(title+'\n'+detail,loc='left',fontsize=23,pad=20)
 fig.subplots_adjust(left=.30,right=.98,bottom=.22,top=.74)
 fig.savefig(P/f'assets/figures/results-{name}.png',dpi=220);fig.savefig(P/f'assets/figures/results-{name}.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(12,3.65));colors={'raw':'#A60000','raw_no_probe':'#00807B'}
labels=['airline:2 · 第1–2轮','retail:28 · 第20–21轮','retail:28 · 第22–23轮']
for j,mode in enumerate(['raw','raw_no_probe']):
 for i,case in enumerate(['gain','adopted_slow','candidate_miss']):
  r=next(r for r in micro['timing_by_case'] if r['case']==case and r['config']==mode);v=r['mean_saved_s'];samples=r['saved_s_each_repeat'];y=i+(j-.5)*.26
  ax.barh(y,v,height=.22,color=colors[mode],label=mode if i==0 else None,xerr=[[v-min(samples)],[max(samples)-v]],error_kw={'ecolor':'#25303D','capsize':4,'elinewidth':1.1})
  ax.text((max(samples)+.45 if v>=0 else min(samples)-.45),y,f'{v:+.2f} s',va='center',ha='left' if v>=0 else 'right',fontsize=17)
  rows.append([case,mode,v,''])
ax.set_yticks(range(3),labels,fontsize=18);ax.invert_yaxis();ax.axvline(0,color='#515B68',linewidth=1);ax.set_xlim(-5.7,22);ax.tick_params(axis='x',labelsize=18);ax.set_xlabel('相对 SPORK 节省时间（秒；正值更快）',fontsize=18);ax.tick_params(axis='y',length=0);ax.xaxis.grid(True,color='#E7EAED');ax.set_axisbelow(True);ax.legend(loc='lower right',bbox_to_anchor=(1,1.05),frameon=False,ncol=2,fontsize=18);fig.subplots_adjust(left=.26,right=.98,bottom=.23,top=.78)
fig.savefig(P/'assets/figures/mechanism-savings.png',dpi=220);fig.savefig(P/'assets/figures/mechanism-savings.svg');plt.close(fig)
with (P/'assets/figures/plot-data.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['protocol_or_case','method','latency_or_saved_seconds','reduction_percent']);w.writerows(rows)
print('Saved 3 source-data plots, SVGs and CSV.')
