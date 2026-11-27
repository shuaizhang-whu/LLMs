from pathlib import Path
import json,re,copy,subprocess
from PIL import Image
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
S=json.loads((P/'slides-content.json').read_text())
RED='A60000';TEAL='00807B';INK='1A2230';GREY='647080';BORDER='D8DEE5';WHITE='FFFFFF'
prs=Presentation(str(P/'assets/template/source-template.pptx'))
for sid in list(prs.slides._sldIdLst):
 prs.part.drop_rel(sid.rId);prs.slides._sldIdLst.remove(sid)
prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
logo=P/'assets/template/whu-cse-logo.png'
maps=[]
def rec(sl,shape,kind,source='本次讲解布局',editable=True):
 maps.append({'slide':len(prs.slides),'shape_id':shape.shape_id,'type':kind,'x':shape.left/914400,'y':shape.top/914400,'w':shape.width/914400,'h':shape.height/914400,'text':shape.text if shape.has_text_frame else None,'source':str(source),'editable':editable})
 return shape

def textbox(sl,x,y,w,h,text,size=20,color=INK,bold=False,align=PP_ALIGN.LEFT,margin=.02):
 sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.clear();tf.word_wrap=True
 tf.margin_left=tf.margin_right=Inches(margin);tf.margin_top=tf.margin_bottom=Inches(margin)
 for i,line in enumerate(text.split('\n')):
  para=tf.paragraphs[0] if i==0 else tf.add_paragraph();para.alignment=align;para.space_before=Pt(0);para.space_after=Pt(2);para.line_spacing=1.08
  for chunk in re.findall(r'[\x00-\x7f]+|[^\x00-\x7f]+',line):
   r=para.add_run();r.text=chunk;r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=RGBColor.from_string(color)
   r.font.name='Times New Roman' if chunk.isascii() else 'Microsoft YaHei'
   r._r.get_or_add_rPr().set('lang','zh-CN')
 return rec(sl,sh,'textbox')

def rect(sl,x,y,w,h,border=BORDER,fill=WHITE,radius=False):
 sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill);sh.line.color.rgb=RGBColor.from_string(border);sh.line.width=Pt(1.1)
 for r in sh._element.xpath('./p:style/a:effectRef'):r.set('idx','0')
 sh._element.spPr.append(OxmlElement('a:effectLst'))
 return rec(sl,sh,'shape')

def arrow(sl,x1,y1,x2,y2,color=TEAL):
 sh=sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));sh.line.color.rgb=RGBColor.from_string(color);sh.line.width=Pt(2)
 end=OxmlElement('a:tailEnd');end.set('type','triangle');sh.line._get_or_add_ln().append(end)
 for r in sh._element.xpath('./p:style/a:effectRef'):r.set('idx','0')
 sh._element.spPr.append(OxmlElement('a:effectLst'))
 return rec(sl,sh,'connector')

def picture(sl,path,x,y,w,h):
 im=Image.open(path);iw,ih=im.size;scale=min(w/iw,h/ih);pw,ph=iw*scale,ih*scale
 sh=sl.shapes.add_picture(str(path),Inches(x+(w-pw)/2),Inches(y+(h-ph)/2),width=Inches(pw),height=Inches(ph))
 return rec(sl,sh,'image',path,False)

def formula(name,tex,size=28):
 f=P/'assets/formulas'/f'{name}.png';fig=plt.figure(figsize=(.01,.01));fig.patch.set_alpha(0);t=fig.text(0,0,tex,fontsize=size,color='#'+INK,math_fontfamily='stix')
 fig.savefig(f,dpi=240,transparent=True,bbox_inches='tight',pad_inches=.05);plt.close(fig);return f

F={
 'slack':formula('slack',r'$\tau_t=d_t+L_t,\quad \sigma_t=\max(R_t-\tau_t,0)$',26),
 'base':formula('base_time',r'$T_{\mathrm{SPORK}}=\max(R_t,\tau_t)+B_{t+1}$',26),
 'ahead':formula('ahead_time',r'$T_{\mathrm{TurnAhead}}=\max(R_t,\tau_t+B_{t+1})$',26),
 'gain':formula('ideal_gain',r'$\Delta T_t=\min(B_{t+1},\sigma_t)$',38),
 'input':formula('candidate_input',r'$\widehat{X}_{t+1}=F(H_t\oplus\langle\varnothing,\widehat{a}_t,\widehat{o}_t\rangle)$',25)}

def base(s):
 sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=RGBColor.from_string(WHITE)
 picture(sl,logo,.45,.17,3.45,.71)
 textbox(sl,9.1,.35,3.75,.35,'TurnAhead · 方法与实验',14,GREY,align=PP_ALIGN.RIGHT)
 textbox(sl,.46,1.06,12.4,.62,s['title'],28,RED,True)
 rect(sl,.47,1.87,.055,.37,TEAL,TEAL)
 textbox(sl,.67,1.83,12.15,.54,s['claim'],22,INK,True)
 rect(sl,.46,7.0,12.4,.012,RED,RED)
 textbox(sl,.46,7.11,8.8,.22,'来源：'+s['source'],10.5,GREY)
 textbox(sl,9.4,7.1,3.45,.27,s['footer'],11,RED,align=PP_ALIGN.RIGHT)
 sl.notes_slide.notes_text_frame.text=s['notes']+'\n来源：'+s['source']+'\n主稿：'+str(P.parent/'ACL/main.tex')
 return sl

# Batch 1 is generated first and rendered before the remaining two slides.
for s in S[:5]:
 sl=base(s);n=s['n']
 if n==1:
  picture(sl,P/'assets/figures/figure1-timeline.png',.75,2.4,11.85,3.46)
  for i,(a,b) in enumerate(s['items']):
   x=.47+i*4.18;rect(sl,x,5.95,4.0,.73,TEAL if i else RED)
   textbox(sl,x+.12,6.01,3.76,.62,a+'：'+b,16,INK)
  textbox(sl,.48,6.72,12.2,.26,s['bottom'],14,GREY)
 elif n==2:
  for i,t in enumerate(['候选动作','真实观测','执行后状态','就绪时间']):
   x=.48+i*3.15;rect(sl,x,2.52,3.0,.62,TEAL);textbox(sl,x,2.66,3,.32,t,21,TEAL,True,PP_ALIGN.CENTER)
  boxes=[(.48,3.6,2.38,'当前正式状态','已提交状态 Sₜ'),(4.25,3.48,3.52,'沙箱 A','候选动作后的状态\n保留给父轮采用'),(9.22,3.48,3.62,'沙箱 B','从 A 再复制\n后继工具只修改 B')]
  for x,y,w,t,b in boxes:
   rect(sl,x,y,w,1.37,TEAL);textbox(sl,x+.1,y+.10,w-.2,.37,t,23,RED,True,PP_ALIGN.CENTER);textbox(sl,x+.12,y+.61,w-.24,.66,b,18,INK,align=PP_ALIGN.CENTER)
  arrow(sl,2.93,4.15,4.12,4.15);textbox(sl,2.85,3.56,1.35,.5,'复制并执行\n候选动作',16,TEAL,align=PP_ALIGN.CENTER)
  arrow(sl,7.85,4.15,9.1,4.15);textbox(sl,7.91,3.65,1.15,.32,'再复制',18,TEAL,align=PP_ALIGN.CENTER)
  rect(sl,.48,5.25,12.36,.65,TEAL);textbox(sl,.66,5.39,12,.36,'因果、隔离、可验证：只读已知前缀；候选在私有副本执行；各字段可比较。',18,INK)
  textbox(sl,.5,6.17,12.3,.55,s['bottom'],18,INK)
 elif n==3:
  table=sl.shapes.add_table(5,2,Inches(.48),Inches(2.49),Inches(12.36),Inches(2.85))
  rec(sl,table,'native_table');table.table.columns[0].width=Inches(2.3);table.table.columns[1].width=Inches(10.06)
  for i,(a,b) in enumerate(s['items']):
   for j,t in enumerate([a,b]):
    cell=table.table.cell(i,j);cell.text=t;cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string(WHITE);cell.margin_left=Inches(.18);cell.margin_top=Inches(.07);cell.margin_bottom=Inches(.05)
    for para in cell.text_frame.paragraphs:
     para.space_after=Pt(0)
     for r in para.runs:
      r.font.name='Microsoft YaHei';r.font.size=Pt(20);r.font.bold=j==0;r.font.color.rgb=RGBColor.from_string(RED if j==0 else INK)
    for name in ['lnL','lnR','lnT','lnB']:
     ln=OxmlElement('a:'+name);ln.set('w','9000');fill=OxmlElement('a:solidFill');col=OxmlElement('a:srgbClr');col.set('val',TEAL);fill.append(col);ln.append(fill);cell._tc.get_or_add_tcPr().append(ln)
  picture(sl,F['input'],.9,5.55,7.1,.65)
  textbox(sl,.55,6.3,12.2,.48,s['bottom'],18,INK)
 elif n==4:
  ys=[2.62,3.7]
  labels=[('契约通过',['等待后继完成','再次验证','单次采用输出与状态'],TEAL),('失配或失败',['取消请求','等待在途工具结束','从正式输入与状态继续'],RED)]
  for y,(label,steps,color) in zip(ys,labels):
   textbox(sl,.49,y+.2,1.98,.45,label,21,color,True)
   for i,t in enumerate(steps):
    x=2.65+i*3.43;rect(sl,x,y,3.15,.81,color);textbox(sl,x+.10,y+.2,2.95,.44,t,18,INK,align=PP_ALIGN.CENTER)
    if i<2:arrow(sl,x+3.2,y+.4,x+3.38,y+.4,color)
  textbox(sl,.5,4.9,12.1,.42,'父轮直接回复：丢弃后继，交回用户控制权。',20,GREY)
  rect(sl,.48,5.54,12.36,.59,TEAL);textbox(sl,.65,5.68,12,.32,'保证有前提：目标与环境确定性；隔离且单次提交；原子采用；局部渲染 R。',17.5,INK)
  textbox(sl,.5,6.32,12.2,.42,'保证对象：可见输出、动作、观测序列与最终状态；greedy 与 seed 相同不足以保证。',17,INK)
 elif n==5:
  picture(sl,F['slack'],.53,2.53,6.7,.64)
  picture(sl,F['base'],.53,3.33,6.7,.63)
  picture(sl,F['ahead'],.53,4.11,6.7,.63)
  rect(sl,7.5,2.72,5.33,1.8,TEAL);picture(sl,F['gain'],7.72,3.25,4.9,.75)
  textbox(sl,7.75,2.9,4.85,.33,'理想节省量',19,TEAL,True)
  for x,w,a,b in [(.48,6.0,*s['items'][2]),(6.72,6.12,*s['items'][3])]:
   rect(sl,x,5.08,w,1.31,TEAL);textbox(sl,x+.13,5.19,w-.26,.36,a,20,TEAL,True);textbox(sl,x+.13,5.63,w-.26,.68,b,17.5,INK)
  textbox(sl,.5,6.5,12.2,.32,s['bottom'],16,GREY)
prs.save(str(P/'batch-methods.pptx'))

if '--methods-only' in __import__('sys').argv:raise SystemExit()
for s in S[5:]:
 sl=base(s)
 if s['n']==6:
  picture(sl,P/'assets/figures/results-32b.png',.37,2.4,6.1,3.24)
  picture(sl,P/'assets/figures/results-9b.png',6.77,2.4,6.1,3.24)
  textbox(sl,.5,2.62,1.65,.77,'耗时下降\n19.41%',20,RED,True)
  textbox(sl,.49,5.83,5.95,.44,'左：任务耗时；固定用户消息，最多 8 轮。',17,INK)
  textbox(sl,6.86,5.77,5.95,.66,'右：扣除用户生成；98/384 次未完成。\nno_probe / one_probe：后继预测 0 / 1 次。',16.5,INK)
  textbox(sl,.49,6.53,12.25,.33,'raw 降幅 95% 区间：−2.88%～+5.09%；不同模型与交互协议分别比较。',17,INK)
 elif s['n']==7:
  picture(sl,P/'assets/figures/mechanism-savings.png',.42,2.43,12.46,3.63)
  textbox(sl,.5,6.14,12.25,.31,'获益例：父轮 +2.50 s、后续等待 −18.37 s；未命中例：约 194 个子分支 token 被丢弃。',16.5,INK)
  textbox(sl,.5,6.62,12.25,.3,s['bottom'],16,GREY)
prs.save(str(P/'TurnAhead-组会汇报.pptx'))
(P/'element-map.json').write_text(json.dumps(maps,ensure_ascii=False,indent=2)+'\n')
print('Saved 7 editable slides and element map.')
