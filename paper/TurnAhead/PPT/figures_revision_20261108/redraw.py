from pathlib import Path
import json,copy,re,hashlib,statistics,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
P=Path(__file__).resolve().parent;ROOT=P.parent
font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip();fm.fontManager.addfont(font)
plt.rcParams.update({'font.family':fm.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':19,'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
A=Path('/mnt/data/zhangshuai/LLMs/Projects/spork-codex/results/successor_full_20261025_codex');records={};bad=[]
for f in (A/'records').glob('*.json'):
 r=json.loads(f.read_text());seal=f.with_name(f.name+'.seal')
 if hashlib.sha256(f.read_bytes()).hexdigest()!=json.loads(seal.read_text())['sha256']:bad.append(f.name)
 records.setdefault(r['group_id'],{})[r['config']]=r
assert not bad
pairs=[g for g in records.values() if set(g)=={'spork','raw'} and not any(r['errors'] for r in g.values())]
assert len(pairs)==148
vals={k:statistics.mean(g[k]['wall_s'] for g in pairs) for k in ['spork','raw']}
task_a=records['retail:0']['spork'];task_b=records['retail:0']['raw']
assert not task_a['errors'] and not task_b['errors'] and len(task_a['turns'])==len(task_b['turns'])==6
assert all(a['main_token_ids']==b['main_token_ids'] for a,b in zip(task_a['turns'],task_b['turns']))
assert task_a['turns'][-1]['environment_after']==task_b['turns'][-1]['environment_after']
fig,axes=plt.subplots(1,2,figsize=(14,5.1))
for ax,title,vv in [(axes[0],'整体平均｜148 个有效配对',[vals['spork'],vals['raw']]),(axes[1],'单任务示例｜retail:0 · 6 轮',[task_a['wall_s'],task_b['wall_s']])]:
 ax.barh([0,1],vv,color=['#77818C','#A60000'],height=.48)
 ax.set_yticks([0,1],['SPORK','raw']);ax.invert_yaxis();ax.set_xlim(0,440);ax.set_xlabel('任务耗时（秒）',fontsize=21);ax.tick_params(axis='y',length=0,labelsize=22);ax.xaxis.grid(True,color='#E4E7EB');ax.set_axisbelow(True)
 for y,v in enumerate(vv):ax.text(v+6,y,f'{v:.1f}',fontsize=22,va='center')
 ax.set_title(title,fontsize=22,loc='left',pad=45)
 ax.text(.01,1.015,f'耗时下降 {100*(1-vv[1]/vv[0]):.2f}%',transform=ax.transAxes,fontsize=19,color='#A60000',va='bottom')
fig.subplots_adjust(left=.105,right=.97,bottom=.21,top=.76,wspace=.38)
fig.savefig(P/'实验结果-整体与单任务.png',dpi=240);fig.savefig(P/'实验结果-整体与单任务.svg');plt.close(fig)
# Reuse the user's revised deck, update results, and replace the introductory
# timeline with a native-editable three-arm flow. No result records are changed.
prs=Presentation(ROOT/'revision_20261108/TurnAhead-组会汇报.pptx');assert len(prs.slides)==6
RED='A60000';TEAL='00807B';INK='1A2230';GREY='647080';WHITE='FFFFFF';BLUE='DDE8F4';ORANGE='FFF0DC';LRED='FBE8E8'
def remove(sh):sh._element.getparent().remove(sh._element)
def tb(s,x,y,w,h,t,size=18,color=INK,bold=False,align=PP_ALIGN.LEFT):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True;tf.margin_top=tf.margin_bottom=Inches(.015);tf.margin_left=tf.margin_right=Inches(.03)
 for i,line in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.alignment=align;p.space_after=Pt(0);p.line_spacing=1.02
  for chunk in re.findall(r'[\x00-\x7f]+|[^\x00-\x7f]+',line):
   r=p.add_run();r.text=chunk;r.font.name='Times New Roman' if chunk.isascii() else 'Microsoft YaHei';r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=RGBColor.from_string(color)
 return sh
def box(s,x,y,w,h,label,fill=WHITE,color=TEAL,size=17):
 sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill);sh.line.color.rgb=RGBColor.from_string(color);sh.line.width=Pt(1)
 for z in sh._element.xpath('./p:style/a:effectRef'):z.set('idx','0')
 sh._element.spPr.append(OxmlElement('a:effectLst'));tb(s,x+.02,y+.025,w-.04,h-.04,label,size,INK,align=PP_ALIGN.CENTER);return sh
def arrow(s,x,y,x2,y2,col=TEAL,dashed=False):
 sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x),Inches(y),Inches(x2),Inches(y2));sh.line.color.rgb=RGBColor.from_string(col);sh.line.width=Pt(1.3)
 ln=sh.line._get_or_add_ln();end=OxmlElement('a:tailEnd');end.set('type','triangle');ln.append(end)
 if dashed:
  dash=OxmlElement('a:prstDash');dash.set('val','dash');ln.append(dash)
 sh._element.spPr.append(OxmlElement('a:effectLst'))
 for z in sh._element.xpath('./p:style/a:effectRef'):z.set('idx','0')
 return sh
s=prs.slides[0]
for sh in list(s.shapes):
 if sh.shape_id not in {2,3}:remove(sh)
tb(s,.46,1.05,12.4,.6,'TurnAhead｜SPORK、serial 与 raw 的流程对比',28,RED,True)
tb(s,.5,1.82,12.25,.49,'raw 在真实结果就绪时启动下一轮，父轮确认后验证并采用。',22,INK,True)
# Each row shows the parent, real candidate result and the second turn.
for i,(name,desc,fill) in enumerate([
 ('SPORK','正式输入启动下一轮；保留 D1/D2/D3',BLUE),
 ('serial','验证后启动候选下一轮；匹配子轮关闭 D3',BLUE),
 ('raw','结果就绪即启动候选下一轮；匹配子轮关闭 D3',LRED)]):
 y=2.58+i*1.29
 tb(s,.5,y,1.2,.32,name,22,RED if name=='raw' else TEAL,True)
 tb(s,1.7,y-.03,10.98,.3,desc,17,GREY)
 # Main parent (full reasoning), ending with its formal tool1 call.
 box(s,1.75,y+.31,4.46,.35,'父轮 cot₁ → 正式 tool₁',BLUE,size=17)
 tb(s,6.34,y+.30,1.50,.36,'父轮确认',17,TEAL,True)
 arrow(s,6.23,y+.49,6.32,y+.49)
 # Real candidate tool observation becomes available before parent ends.
 box(s,2.05,y+.77,2.82,.34,'probe₁ → tool₁ → 结果₁',ORANGE,'C18842',16)
 if name!='raw':
  arrow(s,4.92,y+.94,6.45,y+.94,GREY,dashed=True)
  tb(s,5.04,y+.63,1.28,.28,'等待确认',16,GREY)
  arrow(s,7.81,y+.49,8.13,y+.49)
  box(s,8.20,y+.31,4.47,.44,'下一轮：cot₂ ＋ 工具处理',fill,size=17)
  arrow(s,6.57,y+.92,8.13,y+.58,GREY,dashed=True)
 else:
  arrow(s,4.94,y+.94,5.18,y+.94)
  box(s,5.24,y+.75,5.58,.38,'沙箱 B：cot₂ ＋ tool₂ 提前执行',LRED,RED,17)
  arrow(s,10.86,y+.94,11.08,y+.94,RED)
  box(s,11.13,y+.75,1.54,.38,'验证并采用',WHITE,TEAL,16)
  arrow(s,7.09,y+.66,11.78,y+.71,TEAL,dashed=True)
tb(s,.5,6.63,12.25,.43,'图示为候选匹配路径；serial 是消融对照，仍保留当前轮 SPORK 投机。',16,GREY)
s.notes_slide.notes_text_frame.text='三组都保留当前轮的 SPORK 派发：先预测候选调用，再拿到真实工具结果。区别在于下一轮什么时候开始。\n\nSPORK 等父轮确认后，用正式输入开始下一轮。serial 也等确认，它把我们方案中的提前启动关掉，保留候选输入和沙箱这套处理；匹配的子轮关闭 D3。raw 则在结果一到就启动下一轮，让模型和工具在沙箱 B 里一起向前运行，父轮结束后再验证并采用。子轮内部保留 D1/D2，因此第二次工具调用也可以和第二轮推理重叠。'
s=prs.slides[-1]
for sh in list(s.shapes):
 if sh.shape_id not in {2,3}:remove(sh)
tb(s,.46,1.05,12.4,.6,'实验结果｜整体平均与单任务示例',28,RED,True)
tb(s,.5,1.82,12.25,.45,'Qwen3-32B：整体平均耗时下降 19.41%；单任务展示实际执行时间。',21,INK,True)
s.shapes.add_picture(str(P/'实验结果-整体与单任务.png'),Inches(.5),Inches(2.5),width=Inches(12.3),height=Inches(4.48))
s.notes_slide.notes_text_frame.text='左边是148个有效配对任务的整体平均耗时，SPORK为364.0秒，TurnAhead为293.3秒，下降19.41%。右边单独展示retail:0这个任务，两组都执行了六轮，SPORK用了307.97秒，TurnAhead用了200.90秒。两组生成的token和最终状态一致。这样可以同时看到整体收益和一个具体任务的执行时间。这组实验采用固定用户消息，每个任务最多八轮。'
prs.save(P/'TurnAhead-组会汇报-图表更新.pptx')
# Standalone native-editable flow slide.
flow=Presentation(P/'TurnAhead-组会汇报-图表更新.pptx')
for sid in list(flow.slides._sldIdLst)[1:]:flow.part.drop_rel(sid.rId);flow.slides._sldIdLst.remove(sid)
flow.save(P/'SPORK-serial-raw-流程对比.pptx')
(P/'data-and-scope.json').write_text(json.dumps({'source':str(A),'records_verified':310,'valid_pairs':148,'means':vals,'no_speculation_mean':None,'no_speculation_status':'Searched spork-codex/results: no Qwen3-32B no-speculation task time found. User accepted a normal single-task example; figure uses retail:0 instead of an unmeasured bar.','single_task':{'group_id':'retail:0','spork_s':task_a['wall_s'],'raw_s':task_b['wall_s'],'turns':6,'same_tokens':True,'same_final_state':True},'serial_semantics':'Serial ablation retains parent D1/D2/D3, builds candidate ticket and sandbox; launches the child after parent validation; matched child D3 is disabled while D1/D2 remains.','raw_semantics':'Child begins when real candidate tool result is ready and parent still reasons; tools execute in isolated B; commit follows exact-input contract; child D3 is disabled.'},ensure_ascii=False,indent=2)+'\n')
print('Saved data chart, native flow diagram, and updated deck.')
