from pathlib import Path
import json,zipfile,hashlib
import pymupdf as fitz
from pptx import Presentation
from PIL import Image
P=Path(__file__).resolve().parent
f=P/'TurnAhead-组会汇报.pptx';prs=Presentation(f);doc=fitz.open(P/'renders/TurnAhead-组会汇报.pdf')
assert len(prs.slides)==len(doc)==7
issues=[];rows=[]
for i,(sl,pg) in enumerate(zip(prs.slides,doc),1):
 assert sl.background.fill.fore_color.rgb.__str__()=='FFFFFF'
 assert sl.notes_slide.notes_text_frame.text.strip()
 counts={'textboxes':0,'tables':0,'pictures':0,'connectors':0,'shapes':0}
 for sh in sl.shapes:
  if sh.left<0 or sh.top<0 or sh.left+sh.width>prs.slide_width+100 or sh.top+sh.height>prs.slide_height+100:issues.append(f'slide{i}/shape{sh.shape_id}: out of canvas')
  if sh.has_table:counts['tables']+=1
  elif sh.has_text_frame and sh.text.strip():counts['textboxes']+=1
  elif sh.shape_type==13:counts['pictures']+=1
  elif sh.shape_type==9:counts['connectors']+=1
  else:counts['shapes']+=1
 pg.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(str(P/f'renders/slide_{i:02d}.png'))
 im=Image.open(P/f'renders/slide_{i:02d}.png').convert('RGB');w,h=im.size
 samples=[im.getpixel(q) for q in [(0,0),(w-1,0),(0,h-1),(w-1,h-1),(w//2,5)]]
 assert all(c==(255,255,255) for c in samples)
 rows.append({'slide':i,**counts,'render_size':im.size,'background_samples_all_white':True,'notes_present':True})
assert not issues,issues
# XML media/notes consistency, and embedded source-logo byte fidelity.
with zipfile.ZipFile(f) as z:
 bad=z.testzip();assert bad is None
 logo_hash=hashlib.sha256((P/'assets/template/whu-cse-logo.png').read_bytes()).hexdigest()
 assert any(hashlib.sha256(z.read(n)).hexdigest()==logo_hash for n in z.namelist() if n.startswith('ppt/media/'))
qa={'pages':7,'method_pages':5,'experiment_pages':2,'native_objects':rows,'out_of_canvas_objects':issues,
 'archive_integrity':'passed','source_logo_bytes_preserved':True,'font_limit':'微软雅黑 / Times New Roman are declared in PPTX; this Linux preview uses Noto Sans CJK / Liberation Serif fallback. Formula PNG uses STIX.',
 'editable':'Titles, body, diagrams, arrows and the five-row contract table are native PPT objects. Original paper figure, equations, source-data charts and original logo are separate images.',
 'visual_checks':'All 7 rendered slides inspected; corrected first-page card line height, page5 cost-box height, chart label clipping and legend overlap.',
 'imagegen_limit':'Draft images retain near-white raster noise at some blank-area pixels. Final PPT background is explicitly #FFFFFF, and sampled final renders are exact white. Drafts are layout references, never full-slide backgrounds.',
 'source_data_limit':'32B fixed-user and 9B full interaction are distinct protocols. v7 is partial: 98/384, 20 strict paired tasks. Microbenchmarks are 3 retrospectively selected prefixes with 3 same-seed repeats.',
 'powerpoint_limit':'Reopened with python-pptx and rendered with LibreOffice; not tested in Microsoft PowerPoint.',
 'file':str(f),'bytes':f.stat().st_size}
(P/'qa-record.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
(P/'qa-record.txt').write_text('TurnAhead：7页（5页动机方法 + 2页实验）\n已检查：页数、备注、原始校徽字节、PPTX压缩包、原生对象、画布边界、显式白底、7页渲染。\n已修正：第1页卡片行高、第5页执行边界换行、图表数值标签裁切、图例覆盖横条。\n主要文字/流程/表格可编辑；公式、原图、数据图为独立图片；数据图配套SVG、CSV及plot_results.py。\n字体：Linux预览采用Noto Sans CJK / Liberation Serif；PPTX保留微软雅黑/Times New Roman设置，公式STIX。\n生成效果图部分近白背景带轻微像素噪声；最终PPT底层白色且渲染留白采样为#FFFFFF。\n不同模型/协议分别报告；完整交互评估未完成；局部机制样本事后选取。\n未在Microsoft PowerPoint实际打开；已用python-pptx重开、LibreOffice渲染检查。\n')
print(json.dumps({'slides':7,'bytes':f.stat().st_size,'issues':issues,'objects':rows},ensure_ascii=False))
