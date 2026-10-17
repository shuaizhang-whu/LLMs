# A：Agent 动作投机——精讲论文 PPT 文字大纲

更新时间：2026-08-03

说明：现有 `Speculative Actions` 8 页组会 PPT 保留作为第 1 篇，不重复制作。本文档对 DSP、PASTE、SPORK 和 Joint Agent–Speculator RL 给出每篇 8 页的可上屏文字、排版和原图建议。所有幻灯片均遵循 16:9、纯白背景、左上角校院 Logo 预留、红/青绿色强调的现有模板。

## 1. Dynamic Speculative Agent Planning（8 页）

```text
P1｜封面
标题：Dynamic Speculative Agent Planning
副标题：用在线预测的动态投机深度控制延迟—成本权衡
页脚：论文精读｜Agent 动作投机 Survey｜汇报人：张帅｜2026.08
版式：上方保留学院 Logo；居中大标题；下方用一条“Approximation 快草拟 → Target 并行验证 → 动态 k”时间线作视觉锚点。
配图：论文 fig/online_graph.tex 简化后的局部时间线。

P2｜背景、问题与贡献总览
标题：为什么固定投机步数 k 不够？
左栏｜背景
• Agent 规划是多步串行过程，慢 Target Agent 保证质量，快 Approximation Agent 可提前草拟未来步骤。
• 投机步数 k 决定并行度：k 太小隐藏不了足够延迟；k 太大会产生大量被取消的冗余调用。
• 同一任务的不同阶段难度不同，不存在始终最优的固定 k。
右栏｜三项贡献
1. 画出延迟—成本 Pareto 前沿，将 k 明确为核心控制量。
2. 使用 DistilBERT 状态价值预测器在线估计动态 k，不需预部署数据和训练。
3. 用 expectile 偏置或 offset 提供用户可控的“更快 ↔ 更省”调节。
底部结论条：最终轨迹始终由 Target 决定，动态预测只改变提前执行多远。
版式：左 42% 问题，右 58% 三张编号卡片；底部用红色结论条。
配图：fig/fix_k_small.tex 与 fig/fix_k_large.tex 并排，作“k 小/k 大”对照。

P3｜固定 k 基线：延迟如何被隐藏，成本如何被浪费
标题：固定 k：加速来自并行，浪费来自失配之后
流程：
Step 1：Approximation Agent A 连续草拟 k 个未来步骤。
Step 2：Target Agent T 从同一状态出发，并行生成权威步骤并逐步对齐。
Step 3：A_i = T_i 时确认并保留对应分支；首个失配点 j 出现时，取消 j 之后所有未确认线程。
Step 4：从 T_j 继续，因此输出与串行 Target 轨迹一致。
关键取舍：预计连续命中长度 m 时，k>m 的部分几乎只增加 token/API 成本；k<m 则留下未利用的并行窗口。
版式：上半页是 A/T 双轨时间线，下半页左侧四步流程，右侧为 k<m、k≈m、k>m 三态对照。
配图：fig/direct_react_*_acc_trend.pdf 中的准确率随 k 变化图，或重画 fig/fix_k_small.tex / fig/fix_k_large.tex。

P4｜DSP 总体框架
标题：DSP：在每个投机 episode 中动态决定 k
主流程：
1. 状态 s_i：当前 prompt + 已验证的局部轨迹。
2. Predictor：异步输出 k_i = round(V_theta(s_i))，与 A 的当前步骤同时执行，隐藏预测延迟。
3. Parallel planning：A 最多向前草拟 k_i 步；T 在并行分支中验证。
4. Stop / recover：命中计算完 k_i 或出现首个 mismatch 即结束 episode，取消失配后续工作。
5. Online trainer：根据 T 给出的实际连续命中长度构造 (s,k) 训练对，后台更新 predictor，完成后热切换到 inference copy。
并行关系：主执行线（A/T）、Predictor inference 线和 Trainer 线相互解耦，训练不阻塞当前任务。
版式：中央放论文总体图，右侧用 1–5 号圆形标注；下方单独标出三条并行线。
配图：fig/online_graph.tex（论文原图，优先完整使用）。

P5｜将动态 k 建模为状态价值预测
标题：预测不是选一个动作，而是估计“还能连续命中几步”
定义：
• 状态 s_t：已验证的 token 轨迹；转移 s_(t+1)=s_t | a_t；折扣 gamma=1。
• 回报 G_t：从 s_t 开始到首次失配前的实际连续命中步数。
• 价值 V_pi(s)=E_pi[G_t | s_t=s]：当前状态的期望可投机深度。
核心损失（公式图）：
L_theta = E[(G_t^lambda - V_theta(s_t))^2]
G_t^lambda = (1-lambda) sum_{n=1}^{T-t-1} lambda^{n-1} G_{t:t+n} + lambda^{T-t-1}G_t
解读：lambda=0 更偏 1-step TD，lambda=1 且 gamma=1 时退化为在线监督回归 k；DistilBERT 只做价值回归，不生成动作。
版式：左侧为 s_t 到 mismatch 的短轨迹示意，右侧为两个公式卡片与 lambda 滑杆。
配图：原图不必要；用论文公式重排为 Times New Roman 透明 PNG。

P6｜用户可控的延迟—成本权衡
标题：一个预测器，两种“更激进 / 更保守”的控制方式
方式 A｜Biased value / expectile
• 用非对称损失让预测器更倾向高估或低估可命中长度。
• tau > 0.5：更激进，投机更深，延迟低但成本高。
• tau < 0.5：更保守，冗余少，但隐藏的延迟也少。
方式 B｜Unbiased value + offset
• 保持价值预测器无偏，部署时使用 k = clip(round(V_theta(s))+beta)。
• beta>0 向速度端移动，beta<0 向成本端移动；不需重新训练。
读图方法：每个 tau/beta 对应延迟—成本平面上一个运行点，选型是 Pareto 前沿上的部署决策，不是准确率调参。
版式：左右两张方法卡，中间一条从“低成本”到“低延迟”的滑杆；底部放 Pareto 散点图。
配图：fig/openagi_react_cost_comp.pdf 或 fig/openagi_mad_cost_comp.pdf。

P7｜实验：速度、成本与在线收敛
标题：动态 k 追平最快固定策略，但减少无效调用
设置：OpenAGI + TravelPlanner；Direct/ReAct/CoT/Multi-Agent 规划；对比串行、固定 k、DSP 与启发式调优。
三个结论：
1. Latency：DSP 保留接近最快 fixed-k 的并行加速。
2. Cost：总成本最多下降约 30%，不必要成本最多下降约 60%。
3. Control：tau/beta 可稳定地沿 Pareto 边界移动，说明用户控制量确实生效。
实验阅读限制：最终质量不降来自“始终跟随 T”，实验核心证据是成本与延迟，而不是新的任务准确率。
版式：左侧 65% 为两张原图，右侧 35% 为三个大数字结论。
配图：fig/openagi_react_gen_vs_time.pdf + fig/openagi_react_cost_comp.pdf。

P8｜总结与适用边界
标题：DSP 解决的是“投机多远”，不是“如何预测动作”
三句总结：
• 核心对象：动态投机深度 k，让系统适应不同状态的可预测性。
• 方法关键：将连续命中长度写成状态价值，在线、异步、轻量更新。
• 系统意义：在不改变 Target 轨迹的前提下，把速度—成本变成可显式配置的运行点。
局限：
1. 并未动态选择 Approximation Agent，效果仍受 A/T 速度比与一致性限制。
2. 严格步骤对齐可能拒绝语义等价但表面不同的规划。
3. 在线冷启动期与更多域的泛化仍需评估。
与 Survey 主线的位置：Speculative Actions 定义“预测—验证—提交”；DSP 在此之上增加“深度控制”。
版式：上方三句总结卡片，下方左侧局限，右侧放 Survey 技术链定位图。
```

## 2. PASTE（8 页）

```text
P1｜封面
标题：PASTE
主标题：Parallelizing Tool Execution and LLM Generation for Low-Latency Agent Serving
副标题：从重复工具模式到全栈 LLM–Tool 协同调度
页脚：论文精读｜Agent 动作投机 Survey｜汇报人：张帅｜2026.08
版式：中央标题，下方是 LLM reasoning 与 Tool execution 由串行变为重叠的时间线。
配图：images/ATC/PASTE-v2.pdf 的主框架局部或 motivation/agentic_tool_loop.pdf。

P2｜背景、问题与贡献总览
标题：Agent Serving 的关键路径不只是 LLM，工具占 45%–57%
左栏｜两个观察
• 现有执行：LLM 生成 → 工具调用 → 等待结果 → LLM 恢复，语义依赖被实现成物理串行。
• 在 deep research、coding、AI4Science 工作负载中，工具执行占 E2E 延迟 45%–57%。
• 优化 token 速度无法隐藏 exposed tool time；要在 LLM 还在生成时得到“带具体参数的未来工具调用”。
右栏｜三项贡献
1. Pattern Analyzer：从历史轨迹抽取控制流序列和隐式数据流映射，实例化具体候选调用。
2. Tool Speculation Scheduler：准入、隔离、确认/提升/丢弃提前执行，保证权威调用优先。
3. LLM–Tool Co-Scheduler：根据工具完成和 GPU 负载节奏化恢复会话，避免“工具变快、GPU 队列变慢”。
底部数字：平均任务完成时间 -43.5%｜平均观测工具延迟 1.8×更低。
版式：左侧串行时间线，右侧三组系统组件，下方大数字。
配图：motivation/tool_llm_percentage_solo.pdf + tool_llm_concurrency_e2elatency.pdf。

P3｜总体框架
标题：PASTE 不是单一预测器，而是三个闭环调度组件
系统流程：
Step 1｜Offline / continuous mining：Pattern Analyzer 从历史 session 发现稳定的工具转移和参数派生规则。
Step 2｜Live instantiation：用当前 session 历史中已知字段填充 pattern，得到可执行的 (tool, args)。
Step 3｜Selective speculation：Tool Scheduler 结合预计收益、置信度、工具容量决定是否提前运行，结果先进隔离区。
Step 4｜Authoritative match：LLM 真实调用到达后规范化比较 tool name + arguments；命中复用，未命中丢弃或将正在运行的同一调用提升为 authoritative。
Step 5｜Return pacing：Co-Scheduler 基于 active request、decode batch、context/KV pressure 控制 session 返回 LLM 的节奏。
不可破坏的约束：预测层不向权威 session 写入内容；未确认结果对 Agent 不可见。
版式：原图占左侧 68%，右侧 32% 放 1–5 的简短标注；底部红色约束条。
配图：images/ATC/PASTE-v2.pdf（优先使用论文原图）。

P4｜Pattern Analyzer
标题：从“经常这样做”到“现在可以提前执行”
两类可利用结构：
• 控制流 pattern：Search → Visit，Edit → Test，Grep → FileEditor。
• 数据流 pattern：下一工具参数可从上一输出/当前 session 中确定提取；例如 Visit(url) 的 url 来自 Search 结果。
生成候选的步骤：
1. 对历史轨迹中工具序列计数，形成 pattern pool 和转移置信度。
2. 学习参数与 session 字段之间的派生关系，保存为可实例化规则。
3. 在线匹配当前上下文，只有所需输入已出现时才生成具体候选。
4. 多个 pattern 冲突时不在分析器内强行决策，交给 Tool Scheduler 按收益与资源仲裁。
案例：Search("DSP paper") 已返回 URL → LLM 还在生成时，从返回文本抽取 URL 并预先 Visit(url)。
版式：上方是两条 pattern 示例，中间是 1–4 流程，下方放一条具体 Search→Visit 案例。
配图：从 images/ATC/PASTE-v2.pdf 裁切 Pattern Analyzer 部分，配合重画的 Search→Visit 示意。

P5｜Tool Speculation Scheduler
标题：提前运行不等于提前提交：完整的投机生命周期
状态机：Candidate → Admitted → Running → {Reuse / Promote / Discard / Preempt}
准入：只当预计可隐藏的 exposed tool time 大于调度与资源机会成本时才启动。
隔离：预执行结果进 speculative store，不更改 authoritative session state。
匹配：对工具名和规范化参数进行一致性检查。
四种结果：
• Reuse：已完成且完全命中，立即返回。
• Promote：权威调用与正在运行的投机调用相同，将该 job 提升而非重启。
• Discard：未命中且不再有价值，丢弃隔离结果。
• Preempt：工具资源紧张时，权威调用抢占投机工作。
安全条件：优先只读/幂等/可隔离工具；副作用工具必须有显式策略、sandbox 或非变更版本。
版式：中央水平状态机，下方四张结果卡，右下角放安全约束红框。
配图：speculative_tool_execution.pdf（论文原图）。

P6｜LLM–Tool Co-Scheduler
标题：工具侧变快之后，为什么 E2E 反而可能变慢？
失败链条：
投机让工具更早完成 → 更多 session 在短时间内回到 LLM → GPU decode batch / KV pressure 上升 → queueing time 增大 → 工具侧收益被抵消。
Co-Scheduler 观测：工具完成、reuse/promotion、预计已隐藏时间、active LLM requests、decode batch pressure、context length 与 KV pressure。
决策：
1. 对带有高投机收益的 returning session 保留优先级。
2. 在 GPU 过载时限制同时恢复数，避免回流尖峰。
3. 在 GPU 利用不足时快速放行，不把 pacing 变成新等待。
关键消融：Tool-only 不是完整解法；Agent Serving 的优化目标是跨 LLM 与 Tool 两个资源域的 task E2E。
版式：左侧用红色链条表示拥塞，右侧用青绿色反馈环表示协同调度，底部放 Tool-only / LLM-only / PASTE 对照。
配图：tool_llm_concurrency_percentage.pdf 或 workload_trace_10min.pdf，加 ablation.pdf 局部。

P7｜实验：全栈协同才转化为 E2E 收益
标题：PASTE 在多种 Agent 工作负载下同时降低平均与尾延迟
设置：deep research、coding、scientific-agent；vLLM 全栈实现；对比 vLLM、Agentix、ORION、SpecFaaS。
核心结果：
• 平均 E2E 延迟最多下降 43.5%，p99 最多改善 55.4%。
• 平均观测工具侧延迟相对主要系统基线降低约 1.8×。
• 消融：Tool-only 使回流更集中，平均 LLM queueing 上升到 65s，E2E 419s；完整 PASTE 降到 45s 和 270s。
实验解读：命中率只是中间量；最终需证明提前执行没有把延迟挪到另一个队列。
版式：左侧放 E2E_overall 原图，右侧放 ablation 原图，上方叠放 43.5% / 55.4% / 1.8× 三个数字标签。
配图：images/ATC/E2E_overall.pdf + images/ATC/ablation.pdf。

P8｜总结与适用边界
标题：PASTE 把“动作投机”推进为“Agent Serving 全栈并行化”
三句总结：
• 预测来源：不调用额外 LLM，而是从重复轨迹中提取可执行的控制流+数据流模式。
• 无损核心：观测式预测、结果隔离、权威调用优先、严格 name+args 确认。
• 系统核心：工具侧提前执行必须与 LLM 侧节奏调度一起设计，才能实现 task E2E 收益。
局限：
1. 依赖可重复的 Agent 轨迹和可从已知上下文提取的具体参数。
2. 低频工具、开放参数和长尾 pattern 的命中收益有限。
3. 有外部副作用、不可撤销的工具不能直接投机执行。
与 Survey 主线的位置：Speculative Actions 给出通用语义；PASTE 展示真实多会话 Serving 中“预测、执行、资源调度”如何合为一个系统。
版式：上半页三张总结卡，下半页左侧为限制，右侧为 Survey 定位图。
```

## 3. SPORK（8 页）

```text
P1｜封面
标题：SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference
副标题：不加辅助模型、不用历史轨迹、不训练的同模型自投机
页脚：论文精读｜Agent 动作投机 Survey｜汇报人：张帅｜2026.08
版式：中央为一条主生成流，在首 token 处用“fork”图标分出 forced tool-call probe。
配图：figures/fig_teaser_prompt.pdf 或 fig2_method_overview_v2.png 的局部。

P2｜背景、问题与贡献总览
标题：外部 Speculator 的成本，也是 Agent 投机的系统瓶颈
问题：
• 辅助小模型需要第二份权重/KV cache，可能与 Actor 的轨迹分布不一致。
• 历史 pattern 对长尾工具与少见参数容易失效，而当前 Actor 的隐状态可能已经包含下一调用。
研究问题：运行中的 LLM 能否足够早、足够准，且失败后仍可回收地预测自己的下一个工具调用？
三项贡献：
1. D1 Prefix-KV Fork：主流第一个 token 后强制工具前缀，缓存命中将 probe 开销从 1.6s 降到约 0.35s。
2. D2 Confidence Gate：使用 probe 自身 tool-name token 的 logprob，选择最早可信提交时刻。
3. D3 Partial-token Reuse：严格匹配拒绝 probe 时，把共享的工具调用前缀作为 speculative decoding 草稿。
版式：左侧问题，右侧 D1/D2/D3 三层卡片，底部一句“training-free self-speculation”。
配图：figures/fig2_method_overview_v2.png 缩略图。

P3｜三个经验发现
标题：同一模型确实“早知道、知道何时可信、失败也不全浪费”
Insight 1｜早知道工具名
• Qwen3-32B 在首部探测时工具名准确率 74.6%–99.6%，但完整参数准确率只有 7.6%–83.9%。
Insight 2｜参数随 CoT 变得可预测
• GAIA 上 args-exact 从 CoT 开始的 7.6% 单调上升到 think-end 的 97.5%；越晚越准，但可隐藏的工具时间越小。
Insight 3｜拒绝仍有可验证前缀
• 拒绝 probe 与主流工具调用共享的 token 前缀中位数 18，75% 的拒绝样本至少共享 14 tokens。
结论：投机时机不应是固定 fork 位置，而应是“最早跨过置信阈值的位置”。
版式：三等分栏，每栏一张原图+一个大数字结论。
配图：fig1b_per_turn_fork_accuracy_top.png + fig_insight2_cot_curve_gaia.png + fig_insight3_prefix_distribution.png。

P4｜总体流程
标题：SPORK 每个 Agent turn 的“分叉—门控—严格验收—前缀回收”
流程：
1. Main 以流式 chat completion 正常生成 CoT。
2. 收到首 token 后启动 raw-completion fork，复用 Prefix KV，强制生成 tool-call body。
3. Probe 生成 tool name 时计算置信度 c；只有 c≥theta 才真正调度 speculative tool。
4. Main 到达工具边界时严格比较工具名+序列化参数。
5a. Hit：复用已完成结果，将工具时间隐藏在剩余 decode 之下。
5b. Miss：串行运行真实工具；同时 D3 将 probe body 作为草稿，验证并接受最长相同 token 前缀。
无损语义：只有与 Main 完整结构调用一致才复用工具结果；否则主轨迹按串行路径执行。
版式：论文方法总图占页面 75%，右下角增加 Hit/Miss 两个标注。
配图：figures/fig2_method_overview_v2.png（完整使用）。

P5｜D1 与成本模型
标题：D1 先把“多跑一次模型”的前缀开销压到足够低
基准成本模型（公式图）：
Speedup ratio = T_base / (T_base* - alpha * t_overlap + T_oh)
Break-even：alpha * t_overlap >= T_oh
参数解读：
• alpha：严格命中并接受已执行工具的 turn 比率。
• t_overlap：工具执行与主生成真正重叠的时间。
• T_oh：probe 的 prefill+decode 以及干扰开销。
D1 实现：不在 main 发起同时启动 probe，而等首 token 让前缀 KV 已入 cache；probe 只 decode 约 50 个工具 token。
实测：prefill 约 1.3s 降为缓存命中约 0.05s，整个 probe 从约 1.6s 降到约 0.35s；主流 TPOT 干扰约 +0.22%。
版式：上半页放公式和 break-even 条，下半页左侧是 D1 缓存时间线，右侧是 1.6s→0.35s 数字对比。
配图：figures/fig_d1_overhead.png；公式使用 Times New Roman 透明 PNG。

P6｜D2 与 D3
标题：D2 提高“期望重叠”，D3 降低“失败分支”的浪费
左栏｜D2 Adaptive Confidence Gate
置信度（公式图）：c = min_{i=2..L} exp(l_i)
• 取 tool-name span 中除开头模板 token 外的最小 top-1 概率，用最弱 token 表示工具选择的保守信心。
• theta=0.90 时精确率 88%、召回率 100%、F1=0.937，过滤 77% 不必要 probe。
右栏｜D3 Partial-token Accept
• 当主流进入 tool-call opener，SporkProposer 将 probe body 注入为草稿 token。
• 主模型并行验证草稿，接受最长一致前缀 k，只生成首个 mismatch 之后的后缀。
• 工具级命中获得重叠；工具级失配仍获得 token 级接受，两层投机复用同一 probe。
底部映射：D1↓T_oh｜D2↑alpha*t_overlap｜D3↓T_base* 的 miss re-decode。
版式：左右两栏；D2 放置信分布图，D3 放 token 验证图；底部用三段箭头总结。
配图：fig_d2_confidence_gate.png + fig_d3_spec_dec.pdf。

P7｜实验：真实工具任务的尾延迟
标题：SPORK 的收益不大但稳健：不改准确率，主要改善 P95
设置：GAIA、HotpotQA、tau2-bench、BrowseComp、BFCL；Qwen3-32B 为主，附加 Qwen3-4B；真实网络工具与 vLLM。
结果：
• 端到端 P95 延迟下降约 6%–18%，任务准确率与串行基线相差约在 1 个百分点内。
• 收益最大的区间：工具时间较长、CoT 留有足够重叠窗口、严格调用命中率较高。
• 收益受限的区间：工具很快、参数只在 CoT 末端才确定、原生 XML/非标准工具格式导致 forced-prefix 不稳定。
消融阅读：D1 控开销，D2 决定是否真正发工具，D3 主要让失配不是纯浪费。
版式：左侧 hero E2E/P95 结果图，右侧是“有利/不利”运行区间地图。
配图：fig_hero_e2e_speedup.png + fig_operating_regime_map.png。

P8｜总结与适用边界
标题：SPORK 是 Agent 动作投机与 Token 投机解码之间最直接的桥
三句总结：
• 预测源：当前 Actor 自身的 Prefix-KV fork，没有额外模型和训练数据。
• 接受层：工具级依赖完整 name+args 严格匹配，失败分支再退回 token 级 speculative verification。
• 系统经验：自投机是否有益可用 alpha*t_overlap 与 T_oh 的 break-even 条件解释。
局限：
1. 严格参数命中仍是核心瓶颈，工具名早确定不等于完整调用早确定。
2. D3 需进入推理引擎修改 proposer path；应用层 HTTP 控制器只包含 D1+D2。
3. 对有副作用工具，置信门控不能取代隔离/确认机制。
与 Survey 主线的位置：PASTE 用轨迹模式预测；SPORK 改为同模型强制前缀探测，并首次系统化回收拒绝 probe 的 token 前缀。
版式：上半页是两层接受桥接图，下半页左侧局限，右侧 Survey 定位。
```

## 4. Speculate While You Reason（8 页）

```text
P1｜封面
标题：Speculate While You Reason
主标题：Teaching Agents to Predict Their Next Tool Call via Joint Agent–Speculator RL
副标题：一个模型、两种模式：用联合强化学习提高自投机 Hit@1
页脚：论文精读｜Agent 动作投机 Survey｜汇报人：张帅｜2026.08
版式：中央放同一 4B 模型分成 Agent mode 和 Speculator mode 的双通路图。
配图：figures/intro_self_speculation.png 或 figures/example-speculation.png。

P2｜背景、问题与贡献总览
标题：小外部 Speculator 不一定既准又便宜，关键是与当前 Agent 对齐
左栏｜问题
• 外部小模型带来第二份权重和 KV cache，但下一工具预测仍常与 4B Agent 的真实调用分布不一致。
• 直接自投机没有额外权重，也能复用 Prefix KV，但通用 Agent 未被教过“从部分轨迹猜自己的下一调用”。
• 预测目标会随 Agent policy 更新而漂移，离线旧轨迹会快速过时。
右栏｜三项贡献
1. Self-speculating agent：同一 policy pi_theta 用固定后缀在 Agent / Speculator 两种模式间切换，共享权重和 Prefix KV。
2. On-policy target construction：从当前 Agent 的 rollout 中截取每个部分轨迹，把它实际发出的下一调用当作 Speculator 目标。
3. Joint RL recipe：双模式 SFT warmup + 4:8 Agent/Speculator 交替 DAPO 更新 + 模式切换时重置 optimizer。
底部数字：Qwen3-4B Hit@1 44.1→61.2｜Qwen3.5-4B 48.9→66.3，同时保持匹配域 task success。
版式：左问题+右贡献，底部两个大数字对比。
配图：intro_self_speculation.png 缩略图。

P3｜为什么要同模型自投机
标题：“更小”的外部模型并不自然意味“更好”的 Speculator
对比一｜外部 Speculator
• 额外模型常驻，增加显存和 serving 调度复杂度。
• 预测的是 Target Agent 将来的结构化调用，小模型的任务能力与其模仿 Target 的能力并不等价。
对比二｜Off-the-shelf self-speculation
• 同一模型与自己的动作分布天然更接近，且可复用 Prefix KV，但未经训练时 Hit@1 仍不足。
经验观察：在 Qwen3 系列中，最强外部小模型平均 Hit@1 约 18.6，主 Agent 自投机约 29.1；说明自身 policy 是更好的预测起点。
需要学习的新任务：给定 h_t = 当前部分轨迹，直接输出下一个结构调用 a_t，而不继续完整 CoT。
版式：左右对比“外部小模型”与“同模型”，中间放参数/KV 占用示意，下方放 18.6 vs 29.1 对比。
配图：figures/intro_self_speculation.png + tables/offshelf_results.tex 中的主要数字重绘。

P4｜双模式数据构造
标题：同一条 Agent rollout 同时提供任务学习和投机目标
轨迹：tau=(x, r_1,a_1,o_1, ..., r_T,a_T,o_T,y)，其中 a_t=(n_t,sigma_t) 为结构化工具调用。
Agent mode：输入 x，生成完整轨迹 tau，用最终答案评估 R_ag(tau)。
Speculator mode：
1. 在每个工具轮 t 截取调用前缀 h_t。
2. 追加固定模式后缀 s_sp，构造 h_t ⊕ s_sp。
3. 让同一 pi_theta 直接生成 predicted call â_t。
4. 目标就是当前 policy 在原 rollout 中实际产生的 a_t，保持 on-policy 对齐。
关键差异：两种模式共享权重，但输出长度和分布完全不同：Agent 是长 reasoning trajectory，Speculator 是单个 structured call。
版式：中间放一条完整轨迹，从每个 a_t 向下分出部分轨迹+后缀的训练样本；右侧放 Agent/Speculator 两种输出。
配图：figures/example-speculation.png（优先使用）。

P5｜投机奖励
标题：部署需要完全匹配，训练则需要可学的稠密信号
工具名奖励：R_name = 1[ñ_t = n_t]
参数奖励：
R_args = (1/|keys(sigma_t)|) * sum_{k in keys(sigma_t)} F1(ô_t[k], sigma_t[k])
最终奖励：R_sp = R_name * R_args
设计解读：
• 工具名错误时立即得 0，与“不同工具结果无法复用”一致。
• 工具名正确后，参数按 key 做 macro token-F1，对接近但不完全相同的字符串给局部信号。
• 评估 Hit@1 仍要求解析后工具名和完整 argument dictionary 精确一致；稠密奖励不改变部署复用标准。
对象关系：R_ag 优化“任务做对”，R_sp 优化“提前猜对自己的调用”，两者依赖同一 policy 但不在同一步混合。
版式：左侧为三层奖励公式，右侧为一个“工具名错/参数近似/完全命中”小例子，底部放训练 vs 部署标准。
配图：公式重排为 Times New Roman PNG，其他元素重画。

P6｜Joint Agent–Speculator RL
标题：数据收集耦合，优化步骤分离：一个 policy 交替学两个目标
完整算法：
0. SFT warmup：混合成功 Agent 完整轨迹和中间轨迹投机样本，先建立两种格式稳定性。
1. 每轮采样 query batch Q，用当前 pi_theta 生成 G_ag 组 Agent rollout。
2. 同一批 rollout 同时提供 Agent 最终奖励 R_ag 和每步 (h_t,a_t) 投机目标。
3. 调度 m(k) 按 4:8 循环：4 次 Agent DAPO update，接着 8 次 Speculator DAPO update。
4. 模式切换时重置 optimizer state，避免上一奖励的 momentum / adaptive statistics 干扰下一模式。
5. Agent reward 额外惩罚无意义重复工具调用，抑制为追求投机奖励而崩塌成连续 tool calling。
核心原则：不将 R_ag 和 R_sp 粗暴线性相加后同步更新，而是在新鲜 on-policy 数据上分块优化。
版式：中央环形训练流程，4 个红色 Agent block + 8 个青绿 Speculator block；右侧放三个稳定技巧。
配图：tables/training_recipe_algorithm.tex 内容重画为流程图。

P7｜实验：Hit@1 提升且匹配域任务成功率保持
标题：联合 RL 把自投机从“可用起点”提高到“更高严格命中率”
设置：HotpotQA、MuSiQue、BrowseComp-Plus、tau-bench Airline/Retail；Qwen3-4B 与 Qwen3.5-4B；指标为完整 structured-call Hit@1 和 task success。
主结果：
• Qwen3-4B：Base self-spec 29.1 → SFT 44.1 → Joint RL 61.2 Hit@1；task success 26.4 → 26.6 → 27.7。
• Qwen3.5-4B：Base self-spec 33.2 → SFT 48.9 → Joint RL 66.3 Hit@1；task success 49.0 → 49.2 → 50.6。
• 五个 benchmark 的 Hit@1 均同向提升，说明收益不是由单一数据集驱动。
消融：移除 SFT warmup、optimizer reset 或 alternating schedule 都会使两种 reward 的收敛更弱或更不稳定。
版式：左侧重画两条 Base→SFT→RL 阶梯，右侧放论文消融 reward 曲线。
配图：tables/main_self_speculation_results.tex 主要数字重绘 + figures/reward_ablation_trends.pdf。

P8｜总结、泛化与严格“无损”界定
标题：这篇论文优化的是 Speculator–Agent 对齐，代价是改变 Agent 本身
三句总结：
• 架构：同一模型双模式，消除额外 speculator 权重并共享 Prefix KV。
• 学习：下一工具目标来自当前 Agent 的新鲜 rollout，通过交替 RL 维持两种行为。
• 结果：大幅提高严格 Hit@1，并在匹配训练域内保持任务成功率。
必须讲清的边界：
1. 它不是 Speculative Actions 意义下的严格轨迹无损：RL 直接修改了 Agent policy，只能说实验上 task success 未下降。
2. 跨域实验显示 Hit@1 可能提升，但任务成功率不总是稳定；对训练分布仍敏感。
3. 当前主要证据是预测命中率与 Agent 质量，真实全栈 E2E 延迟/资源结果不如 PASTE、SPORK 完整。
与 Survey 主线的位置：SPORK 是训练免费的同模型 probe；本文则直接把“预测自己”变成双目标训练任务。
版式：上半页三张总结卡，下半页用红色标注严格无损边界，右下放 SPORK ↔ Joint RL 对照。
```

## 5. 五篇精讲论文的组会串联顺序

```text
Speculative Actions：定义动作层“预测—验证—提交”与无损执行语义
→ DSP：在不改变预测来源的前提下，动态决定投机多远
→ PASTE：从历史轨迹产生具体工具调用，并将算法放入多会话全栈 Serving
→ SPORK：去掉外部 Speculator，从当前 Actor 的 Prefix-KV 直接分叉预测，失配时回收 token 前缀
→ Joint Agent–Speculator RL：不只探测 Actor，而是主动训练 Actor 学会预测自己
```
