from pathlib import Path
import json,hashlib
import pymupdf as fitz
P=Path(__file__).resolve().parent
slides=[
 dict(title='TurnAhead｜让下一轮提前开始',claim='当前轮继续推理，下一轮利用真实候选观测并行推进。',kind='opportunity',source='§2.1–2.3；§4 开头',items=[('原有依赖','本轮推理 → 正式调用 → 工具结果 → 下一轮'),('可利用机会','候选工具已返回，主模型仍在继续推理'),('核心方法','提前运行同一个目标模型的完整下一轮')],bottom='Anchor → Anchored Lookahead → Exact-Input Commit / Drain-and-Resume',notes='我们关注的是两轮之间的等待。SPORK 已经能在思考时提前调用工具；当真实候选结果已经拿到，下一轮所需的信息可能已经齐全，但正式执行仍等待当前轮结束。TurnAhead 把这段时间用于下一轮。候选可能错误，所以提前计算与正式采用必须分开。',layout='三张横排卡片；下方一条从 Anchor 到提交/恢复的流程。',time='75 秒',transition='先看三种执行方式的时间线。'),
 dict(title='从工具重叠走向跨轮重叠',claim='真实工具结果就绪后，还可以重叠下一轮模型计算。',kind='timeline',source='Figure 1，PDF 第 3 页；§4；命题 3',items=[('串行执行','本轮结束后，依次执行工具和下一轮'),('SPORK','在本轮推理中隐藏工具等待'),('TurnAhead','从真实候选观测出发，提前执行下一轮')],bottom='原稿 Figure 1｜时长为机制示意，不是实验测量。',notes='原图比较串行、仅工具重叠和跨轮重叠。TurnAhead 保留完整父轮推理，橙色工具段结束时启动绿色后继轮。绿色分支先在隔离状态运行，到正式验证时才决定是否采用。原图中的长度用于解释关系，不对应实测秒数。',layout='上方 70% 放论文原始时间线；下方三列解释。',time='75 秒',transition='要提前启动，首先要定义一个可验证的就绪点。'),
 dict(title='Anchor｜把候选调用变成可验证的起点',claim='Anchor 包含真实执行得到的观测和状态，不预测工具返回值。',kind='anchor',source='§4.1，定义 2',items=[('因果性','只使用已提交历史、当前状态和已生成前缀'),('隔离性','候选动作在当前环境的私有副本执行'),('可验证性','动作、观测与执行后状态均可精确比较')],bottom='当前实例：SPORK 自投机派发候选；无需额外模型或训练。',notes='Anchor 的四个字段是候选动作、真实观测、执行后的候选状态、就绪时间。SPORK 的 D1/D2 提供候选，工具在隔离环境执行，实际结果形成 Anchor。接口也允许其他满足三项属性的来源；本文不声称已验证其他来源的性能。',layout='上方四字段公式；中间三列属性；底部实例说明。',time='85 秒',transition='拿到 Anchor 后，怎样构造还没有正式发生的下一轮输入？'),
 dict(title='下一轮输入由哪些内容决定？',claim='调用相同仍不够；必须检查渲染后的完整输入 token。',kind='input',source='§2.1，式 (2)；§4.1，式 (3)、命题 1',items=[('历史策略','已完成的思考不写入后续历史'),('提前构造','已提交历史 + 空可见文本 + 候选动作 + 真实观测'),('潜在失配','相同调用 + 不同可见前缀 → 可能得到不同输入')],bottom='充分情形：可见文本为空，且动作与观测正确；最终仍检查完整 token 序列。',notes='在当前历史策略下，下一轮依赖可见文本、动作和工具结果。候选路径暂按可见文本为空来构造。如果正式父轮真的没有额外可见文本，并且动作与观测相同，下一轮输入就能提前确定。若父轮保留说明性文字，哪怕调用相同也可能无法复用。输入不匹配时回退，历史策略影响机会而不是放宽检查。',layout='左右对照：正式历史与候选历史；下方强调完整 token 比较。',time='95 秒',transition='输入可以提前构造，执行状态也要单独隔离。'),
 dict(title='Anchored Lookahead｜用第二个沙箱运行完整下一轮',claim='保留 Anchor 后状态，再复制一份给后继轮执行。',kind='sandboxes',source='§4.2，Trigger / Second sandbox / Successor execution',items=[('启动条件','首个 Anchor 就绪，且父轮仍处于推理阶段'),('沙箱 A','候选动作后的状态；保留给父轮正式采用'),('沙箱 B','从 A 再复制；后继轮的工具只修改 B')],bottom='后继轮使用同一目标模型与相同生成配置；不递归启动更深分支。',notes='只用一个副本是不够的。父轮采用候选动作时，需要的是刚执行完父动作的状态。如果后继轮直接修改这份状态，就可能把未来动作提前提交。所以保留 A，再复制 B 给后继轮。只有后续契约通过，B 的结果才能被正式采用。sandboxed 策略让后继工具立即在 B 内运行。',layout='主状态、沙箱 A、沙箱 B 三节点分叉图；分离候选执行与正式采用路径。',time='90 秒',transition='下一页给出正式采用必须同时满足的五项条件。'),
 dict(title='Exact-Input Commit｜五项同时通过才能采用',claim='验证后继轮的起始条件，而不只验证工具名和参数。',kind='contract',source='§4.3，式 (4)',items=[('动作','工具名与全部参数相同'),('观测','真实工具输出的 SHA-256 相同'),('起始状态','环境指纹相同：Agent / User 数据库'),('完整输入','下一轮输入 token 序列逐项相同'),('生成配置','模型、seed、预算、上下文、思考、停止与采样配置相同')],bottom='先检查契约；等待后继轮完成后，再检查一次，再采用。',notes='动作匹配只验证父调用。观测摘要有助于定位拒绝原因，虽然它通常已由输入相同隐含。状态检查针对后继轮起点；后继轮执行后产生的新状态本来就可能不同。完整输入检查包括聊天模板和 tokenization。最后还要检查生成配置。等后继完成后再次验证，避免等待期间条件改变。',layout='五行原生可编辑表格；下方用五个 AND 连接的检查标识。',time='100 秒',transition='契约通过时采用，失败时怎样恢复？'),
 dict(title='Drain-and-Resume｜失败分支先排空，再恢复正式路径',claim='取消请求后，还要等待在途工具结束，才能继续。',kind='recovery',source='§4.3；Algorithm 1',items=[('契约通过','等待后继完成 → 再次验证 → 采用输出与状态'),('失配或生成失败','取消后继请求 → 排空在途工具 → 正式生成下一轮'),('父轮直接回复','丢弃后继分支，将控制权交回用户')],bottom='单次采用；未验证分支的状态始终留在私有副本。',notes='取消异步包装不保证底层工具线程已经停止。排空用于确保旧分支不再运行或争用资源。失配后从父轮已经确认的输入和状态重新生成下一轮。若正式路径自身也产生无效输出，不能通过无限重试把它包装成成功。父轮若直接回答用户，就没有需要接续的后继轮。',layout='纵向决策流程；左侧青色通过路径，右侧红色恢复路径；独立灰色回复出口。',time='85 秒',transition='这些检查在哪些假设下足以保持正式轨迹？'),
 dict(title='正确性结论是有条件的',claim='满足执行假设与渲染局部性时，可按窗口归纳保持轨迹。',kind='guarantees',source='§4.4，假设 1、命题 2；附录 A.1–A.2',items=[('A1｜目标确定性','同输入与配置下，提交输出不受并发请求和草稿影响'),('A2｜环境确定性','状态与动作决定结果；指纹足以区分相关状态'),('A3｜隔离与单次提交','投机只修改私有副本，每轮最多采用一次'),('A4｜原子提交','最终检查与采用之间，正式历史和状态不变')],bottom='另需逐消息局部渲染条件 R。保证对象：可见输出、动作、观测序列与最终状态。',notes='证明对窗口归纳。通过契约的分支具有相同输入、配置和起始状态，因此在假设下给出相同正式输出与环境转换。未通过分支没有正式影响，恢复路径与参考执行相同。渲染局部性保证采用的历史在以后继续渲染时仍等价。这是理论前提，不能由 greedy 或 seed 相同直接推出所有运行时都满足 A1。',layout='2×2 条件卡片；底部一行归纳关系与保证对象。',time='100 秒',transition='正确性与性能分开：通过验证也不一定更快。'),
 dict(title='理想收益来自观测就绪后的剩余推理窗口',claim='理想节省量等于：后继轮时长与可重叠窗口的较小者。',kind='latency',source='§4.4，命题 3、式 (5)',items=[('观测就绪时刻','τₜ = dₜ + Lₜ'),('剩余推理窗口','σₜ = max(Rₜ − τₜ, 0)'),('两种完成时间','SPORK：max(Rₜ, τₜ) + Bₜ₊₁；TurnAhead：max(Rₜ, τₜ + Bₜ₊₁)')],bottom='条件：Anchor 正确、dₜ ≤ Rₜ；忽略并发竞争与管理开销。',notes='R 是父轮生成时长，L 是候选工具时长，d 是派发时刻，B 是后继轮在基底下的时长。先写两种执行完成时间，再相减得到 min(B, sigma)。工具更快会让观测更早就绪，从而在父轮时长固定时增大窗口。这个公式只描述理想可重叠部分，不是实测收益保证。',layout='左侧三条符号定义与完成时间；右侧大公式与简洁时间轴。',time='105 秒',transition='最后把理想收益与真实系统代价放在一起。'),
 dict(title='执行边界与实际收益',claim='有效重叠要覆盖并发拖慢、验证和失败分支的排空成本。',kind='scope',source='§4.2 Windows / 工具策略；附录 A.3–A.5',items=[('深度受控','按窗口配对；不跨用户消息或最终回复；预算不足时单轮执行'),('工具策略','sandboxed：副本内立即执行；deferred：等父轮验证后再执行'),('收益条件','关注成功提交与重叠长度的联合分布，以及父轮与子轮拖慢'),('适用边界','真实外部副作用无法复制时，需延后工具执行')],bottom='TurnAhead：真实观测触发提前计算，完整输入契约约束正式采用。',notes='当前设计是深度一的窗口化执行，连续滚动和递归加深不属于本稿已经验证的方案。deferred 策略适合不能复制外部效果的工具，代价是无法提前隐藏这部分工具延迟。附录把真实收益分成重叠收益、父子轮拖慢以及拒绝排空成本；提交率不能单独决定收益。本次先展示方法，实验图表将在后续补入。',layout='四个横向信息区；中间收益减成本关系；底部方法收束。',time='90 秒',transition='后续接入实验结果，检验收益和边界。')
]
for i,s in enumerate(slides,1):s['n']=i;s['footer']=f'TurnAhead · 方法汇报   |   {i:02d} / 10'
(P/'slides-content.json').write_text(json.dumps(slides,ensure_ascii=False,indent=2)+'\n')
sections=[]
for s in slides:
 body='\n'.join(f'{a}：{b}' for a,b in s['items'])
 formulas={'anchor':'Anchor =（候选动作，真实观测，执行后状态，就绪时间）；原文定义 2。','input':'候选输入：X̂ₜ₊₁ = F(Hₜ ⊕ ⟨∅, âₜ, ôₜ⟩)；原文式 (3)。','contract':'Iₜ = 1[动作相同 ∧ 观测相同 ∧ 起点状态相同 ∧ 完整输入相同 ∧ 生成配置相同]；式 (4)。','latency':'ΔTₜ = min(Bₜ₊₁, σₜ)；由两种完成时间相减，原文式 (5)。','scope':'实际净收益 = 有效重叠 − 父子轮拖慢 − 验证与排空开销；附录 A 的讲解性概括。'}
 fig='原稿 Figure 1：main.pdf 第 3 页上方时间线，等比裁切图形区域，保留原始标签。' if s['kind']=='timeline' else '依据所引章节绘制原生可编辑讲解流程；非论文原图。'
 sections.append(f'''第{s['n']}页｜{s['title']}
【本页作用】围绕“{s['claim']}”展开方法讲解。
【本页核心结论】{s['claim']}
【页面实际展示的全部文字】
标题：{s['title']}
核心结论条：{s['claim']}
正文要点：
{body}
流程标签/图注：{s['bottom']}
页脚：{s['footer']}；来源：{s['source']}
【方法流程、公式或表格】{formulas.get(s['kind'],'按正文要点组织流程或对照；无独立公式。')}
【排版布局】{s['layout']} 校徽在左上，使用原始模板素材。
【配图及来源】{fig} 来源：{s['source']}。
【预计讲解时间】{s['time']}
【口播备注草稿】{s['notes']}
【与前后内容的过渡语】{s['transition']}
''')
(P/'TurnAhead-PPT文字大纲.md').write_text('\n'.join(sections))
source=P.parent/'ACL';notes=['方法材料：main.tex 及其 input 的 sections/02_background.tex、04_method.tex、A_proofs.tex。','主 PDF：main.pdf，8 页。论文原图：Figure 1，PDF 第 3 页。','slides 1–10 不包含实测加速率或任务成功率；其他章节中的 Planned 段落不作为完成事实。','理论：输入可确定性命题1；无损性命题2（A1–A4 + 渲染局部性R）；窗口延迟命题3（忽略竞争与管理开销）。','继承机制：SPORK D1/D2/D3 是基底；TurnAhead 讲解观测锚定跨轮执行。','当前稿方法中的第二沙箱、Exact-Input Commit、Drain-and-Resume 为本次展示重点。','模板：'+str(P/'assets/template/source-template.pptx'),'原始校徽：从既有 SPORK deck 的 ppt/media/image1.png 提取，保留字节。','演示文稿专用工具/技能未在当前工具列表提供；可编辑重建采用本地 python-pptx 与 LibreOffice 渲染。','字体：服务器无微软雅黑和 Times New Roman；渲染使用 Noto Sans CJK / Liberation Serif 回退，公式使用 STIX。']
for f in [source/'main.tex',source/'main.pdf',source/'sections/04_method.tex',source/'sections/A_proofs.tex']:
 notes.append(str(f)+' sha256='+hashlib.sha256(f.read_bytes()).hexdigest())
(P/'source-notes.txt').write_text('\n'.join(notes)+'\n')
doc=fitz.open(str(source/'main.pdf'));page=doc[2]
page.get_pixmap(matrix=fitz.Matrix(4,4),clip=fitz.Rect(75,65,520,195),alpha=False).save(str(P/'assets/figures/figure1-timeline.png'))
print('Saved 10-slide outline and source notes.')
