from pathlib import Path
from copy import deepcopy
import re,json,hashlib
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.oxml.xmlchemy import OxmlElement
P=Path(__file__).resolve().parent
SRC=Path('/mnt/data/zhangshuai/.codex/attachments/1cadfb24-846c-4c72-8d1f-3a286b396302/TurnAhead-组会汇报.pptx')
prs=Presentation(SRC)
assert len(prs.slides)==6,'Inspect unexpected attachment version'
notes=[
'这张图展示了我们想利用的时间窗口。最上面是串行执行：这一轮想完了，才去调工具，再开始下一轮。SPORK 已经把工具调用提前到了推理过程中。\n\nTurnAhead 再往前走一步：工具结果一回来，就开始算下一轮。这样，当前轮继续推理的同时，下一轮也在推进。接下来我讲这个提前启动的起点怎么构造。',
'我们把提前启动的起点叫作 Anchor，它带着四样信息：候选动作、真实工具结果、执行后的状态，以及结果就绪的时间。\n\n这里用了两个沙箱。A 保存执行完候选动作的状态，等父轮确认后采用。接着再复制出 B，让下一轮在 B 里运行。这样父轮和下一轮各自的状态就分开了。候选调用沿用 SPORK 的自投机派发，下一轮也由同一个目标模型来生成。',
'提前算出的这一轮，最后要过五项检查：动作、工具结果、起始状态、完整输入，以及生成配置。五项都对齐，才有采用的依据。\n\n其中，完整输入检查会把聊天模板和分词后的 token 逐个比较。父轮如果多了一段可见说明，下一轮的输入也会跟着变化。状态检查对应的是下一轮刚启动的时刻。等它运行结束，我们还会再检查一次，然后提交输出和新状态。',
'这里有两条处理路径。检查通过，就等下一轮完成，再核对一次，把输出和状态一起采用。出现失配或生成失败，就取消这个分支，等在途工具结束，再从已经确认的输入和状态继续执行。\n\n父轮如果直接回答用户，就把控制权交回用户。这套处理依靠确定的模型和环境行为、私有状态隔离，以及原子提交来保持正式轨迹。图里这几步对应了提前计算到正式采用的整个过程。',
'收益可以从两个完成时间直接看出来。SPORK 先等父轮和工具结果都就绪，再花时间跑下一轮；TurnAhead 把下一轮提前放进了这个等待窗口。\n\n所以，理想节省量就是下一轮时长和剩余推理窗口里较小的那个。比如父轮还要推理六秒，下一轮需要十秒，提前启动就能重叠其中六秒。实际运行还要扣掉父子轮相互拖慢，以及验证和排空的开销。我们用深度一的窗口来控制这部分成本。',
'这组 Qwen3-32B 实验里，平均任务耗时下降了 19.41%。灰色是 SPORK，平均 364.0 秒；红色是 TurnAhead，平均 293.3 秒，相当于每个任务省了大约 70.7 秒。\n\n两组各运行了 155 个任务，图中统计的是 148 个有效配对。这个设置使用固定用户消息，每个任务最多执行八轮。它展示了跨轮重叠在这套设置下带来的时间收益。'
]
replacements={
(3,6):'后续历史保留可见输出；完整输入由文本、动作与观测共同决定。',
(3,12):'候选输入按空可见文本构造，最终逐项核对完整 token 序列。',
(4,6):'分支在副本中运行，通过验证后再把输出与状态正式提交。',
(4,30):'正确性条件：目标与环境确定性；隔离且单次提交；原子采用；局部渲染 R。',
(4,31):'保证对象：可见输出、动作、观测序列与最终状态。',
(5,21):'深度一窗口；按用户交互划分；外部副作用在验证后提交。',
(5,22):'理想分析：Anchor 正确，忽略竞争与管理开销。R：父轮，L：工具，B：后继轮时长。',
(6,4):'实验结果｜Qwen3-32B 固定用户消息评测',
(6,6):'148 个有效配对任务，平均任务耗时下降 19.41%。',
(6,13):'固定用户消息；每个任务最多执行 8 轮。'
}
def remove(sh):
 el=sh._element;el.getparent().remove(el)
def replace(sh,text):
 tf=sh.text_frame
 runs=[r for p in tf.paragraphs for r in p.runs]
 props=deepcopy(runs[0]._r.get_or_add_rPr()) if runs else None
 p0=tf.paragraphs[0];ppr=deepcopy(p0._p.pPr) if p0._p.pPr is not None else None
 tf.clear();p=tf.paragraphs[0]
 if ppr is not None:
  old=p._p.pPr
  if old is not None:p._p.remove(old)
  p._p.insert(0,ppr)
 for chunk in re.findall(r'[\x00-\x7f]+|[^\x00-\x7f]+',text):
  r=p.add_run();r.text=chunk
  if props is not None:r._r.insert(0,deepcopy(props))
  r.font.name='Times New Roman' if chunk.isascii() else 'Microsoft YaHei'
removed=[]
# Remove only display metadata, preserving body mentions of methods and baselines.
for collection in [prs.slides,prs.slide_masters,*[m.slide_layouts for m in prs.slide_masters]]:
 for obj in collection:
  for sh in list(obj.shapes):
   ph=sh._element.xpath('.//p:ph')
   txt=sh.text if sh.has_text_frame else ''
   auto=any(x.get('type') in {'sldNum','ftr'} for x in ph)
   footer=(sh.top is not None and sh.top>prs.slide_height*.85 and (re.search(r'(来源|出处|Source|References|\d+\s*/\s*\d+)',txt,re.I)))
   if auto or footer:removed.append(txt or 'master/footer placeholder');remove(sh)
for i,sl in enumerate(prs.slides,1):
 for sh in sl.shapes:
  if (i,sh.shape_id) in replacements:replace(sh,replacements[(i,sh.shape_id)])
 sl.notes_slide.notes_text_frame.text=notes[i-1]
 # Suppress inherited footer and slide-number display as well.
 for el in sl._element.xpath('./p:hf'):sl._element.remove(el)
OUT=P/'TurnAhead-组会汇报.pptx';prs.save(OUT)
(P/'口播备注.txt').write_text('\n\n'.join(f'【{i}】\n{t}' for i,t in enumerate(notes,1))+'\n')
(P/'revision-record.json').write_text(json.dumps({'source':str(SRC),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'output':str(OUT),'pages':6,'edits':'Edited user attachment directly; retained its 5 method pages and 32B-only experiment chart; removed residual 9B discussion; rewrote all spoken notes; cleaned source footers and page numbers from slides/masters/layouts.','footer_objects_removed':removed,'source_record':'Experiment audit remains in ../source-notes.txt and the historical result files; no new data created.'},ensure_ascii=False,indent=2)+'\n')
print('Saved',OUT)
