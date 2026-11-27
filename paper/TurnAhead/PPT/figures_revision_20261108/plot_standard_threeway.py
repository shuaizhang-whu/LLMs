from pathlib import Path
import argparse,json,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args()
d=json.loads(a.result.read_text())
if not d.get('all_three_comparable'):raise ValueError('Three arms must have the same outputs and tool states')
P=Path(__file__).resolve().parent
font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip();fm.fontManager.addfont(font)
plt.rcParams.update({'font.family':fm.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':21,'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
values=[d['standard_s'],d['spork_s'],d['raw_s']]
fig,ax=plt.subplots(figsize=(11,4.8));ax.barh([0,1,2],values,color=['#394B69','#77818C','#A60000'],height=.5);ax.set_yticks([0,1,2],['标准执行','SPORK','TurnAhead (raw)']);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.22);ax.tick_params(axis='y',length=0);ax.xaxis.grid(True,color='#E4E7EB');ax.set_axisbelow(True);ax.set_xlabel('任务耗时（秒）')
for y,v in enumerate(values):ax.text(v+max(values)*.015,y,f'{v:.2f}',va='center')
ax.set_title('Qwen3-32B｜retail:0｜同一任务三种执行方式',loc='left',pad=20)
fig.subplots_adjust(left=.29,right=.95,bottom=.2,top=.79)
for ext in ['png','svg']:fig.savefig(P/f'标准-SPORK-raw-单任务耗时.{ext}',dpi=240)
plt.close(fig)
prs=Presentation(P/'TurnAhead-组会汇报-图表更新.pptx');sl=prs.slides[-1]
for sh in list(sl.shapes):
 if sh.shape_id not in {2,3}:sh._element.getparent().remove(sh._element)
def text(x,y,w,h,t,size,color='1A2230'):
 sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
 for i,line in enumerate(t.split('\n')):
  para=tf.paragraphs[0] if i==0 else tf.add_paragraph();r=para.add_run();r.text=line;r.font.name='Microsoft YaHei';r.font.size=Pt(size);r.font.color.rgb=RGBColor.from_string(color)
text(.46,1.05,12.4,.62,'实验结果｜标准执行、SPORK 与 raw',28,'A60000')
text(.5,1.82,12.2,.52,'同一任务 retail:0：关闭全部投机 / SPORK / TurnAhead',21)
sl.shapes.add_picture(str(P/'标准-SPORK-raw-单任务耗时.png'),Inches(1.3),Inches(2.5),width=Inches(10.7),height=Inches(4.67))
sl.notes_slide.notes_text_frame.text=f"这一页比较同一个任务的三种执行方式。标准执行关闭动作投机和token投机，耗时{values[0]:.2f}秒；SPORK耗时{values[1]:.2f}秒；TurnAhead耗时{values[2]:.2f}秒。三组输出和工具状态经过一致性检查。标准执行是本次补测，另外两组采用此前该任务的执行记录。"
prs.save(P/'TurnAhead-组会汇报-三组耗时.pptx')
print('Saved measured three-arm figure and PPT:',P/'TurnAhead-组会汇报-三组耗时.pptx')
