# B．端云协同 Agent——PPT 文字大纲（24 页）

```text
第 1 页｜封面
标题：B．端云协同 Agent
副标题：从请求路由到步级协作、选择性披露与可信执行
页脚：Agent 系统 Survey｜武汉大学国家网络安全学院｜张帅｜2026.08
排版：纯白背景；左上角预留校徽；标题居中偏左；右侧放“设备—网络—云”三层线框图。
配图：自绘端—云协同概念图，不放复杂论文截图。

第 2 页｜为什么不能只选“全本地”或“全云端”？
标题：端云矛盾不是二选一，而是多目标约束
左栏：全本地
• 优点：隐私、离线可用、无 API 费用、低网络依赖
• 缺点：模型小、长轨迹脆弱、端侧 prefill/decode 慢
右栏：全云端
• 优点：推理和多模态能力强、工具生态成熟
• 缺点：网络延迟、持续成本、原始 UI/本地数据暴露
底部结论：协同的目标不是“尽量少用云”，而是在成功率约束下最小化云调用、延迟、通信与隐私暴露。
排版：左右两张对称卡片；底部横向目标函数条。
配图：本地手机与云模型的对照图标。

第 3 页｜统一分类：端云协同的四个层次
标题：同叫“端云协同”，实际可能发生在四个不同层次
层次 1 任务层：云端分解，本地执行子任务（Minions）
层次 2 轨迹层：每个 Agent step 动态路由（Hera）
层次 3 表示层：只上传 UI block / feature / KV（CORE、split inference）
层次 4 执行层：本地过滤与守卫、云端语义判断（Cognitive Firewall、AgenTEE）
右下角提示：比较论文前，必须先问“路由的单位是什么？”
排版：四层阶梯或纵向堆栈；每层标注代表论文。
配图：自绘四层堆栈。

第 4 页｜论文地图与本章主线
标题：B 方向代表论文：从协作协议走向可信 Agent runtime
时间线：2025 Minions → 2025 CORE → 2026 Mobile-Agent-v3.5 → 2026 Hera → 2026 Agent-X / AgenTEE / Cognitive Firewall
主讲种子：Hera、CORE
补充桥梁：Minions、Agent-X、Mobile-Agent-v3.5、AgenTEE、Cognitive Firewall
阅读问题：谁决定？传什么？何时上云？如何证明收益与安全？
排版：横向时间线；种子论文用深蓝，桥接论文用青色。
配图：各论文标题缩写卡片。

第 5 页｜Minions：从请求路由到协作协议
标题：Minions：让云端“指导”，而不是让云端读取全部本地上下文
问题：本地小模型能读私有长文档，但复杂推理弱；云端强模型能力高，却不应接收全文。
Naive Minion：本地与云端来回对话；远程成本降 30.4×，但仅恢复 87% cloud-only 性能。
失败原因：本地模型难以执行复杂多步指令，也难以在长上下文中稳定推理。
转折：把难任务拆成短、局部、可并行的子任务。
排版：左侧问题；右侧 naive 对话链；底部失败原因。
配图：自绘 local-context / remote-planner 通信图。

第 6 页｜MinionS 方法
标题：云端分解与聚合，本地在私有数据上并行执行
Step 1：云端基于任务生成结构化子任务。
Step 2：本地在不同文档 chunk 上并行执行短子任务。
Step 3：只把局部结果/证据返回云端，而非发送完整上下文。
Step 4：云端聚合、必要时验证并形成答案。
关键点：协同收益来自“能力分工 + 数据最小化 + 并行执行”，不只是模型路由。
排版：中央四步流水线；本地阶段画成并行扇出/聚合。
配图：论文方法图若能获取则优先，否则按论文协议重绘为可编辑流程图。

第 7 页｜Minions 结果与局限
标题：质量可接近云端，但它还不是长轨迹 Agent 路由器
结果：MinionS 平均降低 5.7× 远程成本，恢复 97.9% remote-only 性能。
优点：本地数据不离端；子任务并行；协议容易理解。
局限：面向长文档推理；环境基本静态；路由单位是子任务，不是 Agent 每一步。
过渡：当 Agent 与环境交互后，每一步难度都在变化——需要 state-level coordination。
排版：左侧大数字 5.7× / 97.9%；右侧优点/局限；底部箭头指向 Hera。

第 8 页｜Hera 动机：每一步都需要云吗？
标题：长轨迹中，云端价值集中在少数关键决策点
设备模型：低延迟、低成本，但复杂规划和状态跟踪容易失败。
云端模型：能力强，但每一步调用都会累积费用和网络等待。
论文观察：设备失败而云成功的轨迹中，仍有大量步骤端云动作一致；理想 router 无需把整条轨迹交给云端。
难点：某一步是否关键，与轨迹位置、推理长度、预测熵和环境状态共同相关。
排版：左侧成功/失败轨迹；右侧端云动作一致性示意。
配图：本地原图 figures/motivating_example.pdf 或 figures/teaser.pdf。

第 9 页｜Hera 问题形式化
标题：在 cloud-only 性能约束下，最小化整条轨迹的云调用
动作混合：â(s,x) = (1-rφ(s,x))·πd(s,x) + rφ(s,x)·πc(s,x)
其中 rφ(s,x)∈{0,1}：0 选择设备模型，1 升级到云模型。
优化目标：minφ E[Σt rφ(st,x)]
约束：E[R(τmix)] ≥ E[R(τcloud)]
解释：router 不是估计单步“哪个回答更好”，而是在轨迹回报约束下管理未来云成本。
排版：左侧符号表；右侧两行公式；下方用自然语言解释。
配图：无，公式为 Times New Roman PNG 或 Office 公式。

第 10 页｜Hera 第一阶段：动作一致性冷启动
标题：Stage I：用云轨迹回放构造 step-level 路由标签
Step 1：分别 rollout device policy 与 cloud policy。
Step 2：选取 cloud 成功、device 明显较差的任务集合 Xdiff。
Step 3：在 cloud trajectory 的每个 state 上重放 device model。
Step 4：device action 与 cloud action 不一致 → 标注“需要云”；一致 → 标注“本地足够”。
Step 5：用 BCE 训练 500M 二分类 router。
优点：把离散路由转为可监督学习；缺点：动作相同只是 proxy，不直接等于最终成功。
排版：五步水平流程；一致/不一致用绿/橙分支。
配图：本地原图 figures/teaser.pdf 的 IL 部分或自绘。

第 11 页｜Hera 第二阶段：成本感知 RL
标题：Stage II：从“像云端”转向“整条轨迹成功且少上云”
Step 1：对同一初始状态采样 N 条混合路由轨迹。
Step 2：将多条轨迹中相同 state 分组。
Step 3：分别估计本地/云决策后的 expected return R̂(d;s)。
Step 4：估计从当前 state 开始的 future cloud calls Ĉ(d;s)。
Step 5：优先选回报更高的决策；回报近似时选未来云调用更少者。
标签规则：若 |R̂1-R̂0|≤ε，则 y(s)=argmin_d Ĉ(d;s)。
排版：左侧多轨迹汇聚；中间 state grouping；右侧回报/成本二级选择。
配图：本地原图 figures/rl_training.pdf。

第 12 页｜Hera 实验：形成成功率–成本 Pareto 前沿
标题：Hera 只在 46.3% 的 steps 调云，达到 92.5% cloud-only 成功率
Benchmark：ALFWorld、WebShop、AppWorld。
综合结果：92.5% cloud-only success；53.6% cloud cost；69.3% end-to-end latency。
Router overhead：约 61 ms/step。
消融：IL 提供稳定冷启动；RL 进一步提高成功率并减少云调用。
关键解释：Agent 路由应使用 trajectory-level feedback，而非把通用 request router 机械地逐步调用。
排版：左侧三项大数字；右侧 Pareto/训练曲线；底部结论。
配图：本地原图 figures/exps/clouassist_*.pdf、figures/exps/latency_break.pdf 或 figures/rl_training.pdf。

第 13 页｜从“何时上云”到“上传什么”
标题：Hera 选择计算位置；CORE 进一步选择可见信息
Hera 的未解决问题：上云时通常把完整 state/context 交给云模型。
GUI 场景的风险：每一步 screenshot/XML 都可能包含与当前动作无关的联系人、消息、账号和历史内容。
新问题：能否只暴露与当前 sub-task 相关的 UI 局部，同时保留云模型的决策能力？
排版：左侧完整 UI 上传（红色大框）；右侧局部 block 上传（绿色小框）。
配图：CORE 原图 pics/intro_v3.pdf。

第 14 页｜CORE 总览
标题：CORE：本地模型做粗筛，云模型做精决策
组件 1：Layout-aware block partitioning——按 XML 层级与语义邻近将页面分块。
组件 2：Co-planning——本地与云端共同判断当前 sub-task。
组件 3：Co-decision-making——本地排序相关 block，云端只在 top block 内选择具体 UI element。
组件 4：Multi-round accumulation——本地误判时逐轮扩大候选上下文。
目标：在 task success 与 UI exposure 之间形成可控权衡。
排版：中央 pipeline；本地模块蓝色，云端模块紫色，上传信息青色。
配图：本地原图 pics/method_pipeline_v3.pdf。

第 15 页｜CORE：布局感知分块
标题：先把页面变成语义 block，再决定哪一块可以上云
输入：Android XML hierarchy 与 UI elements。
Step 1：根据父子层级、空间邻近和容器关系形成初始 block。
Step 2：限制 block 大小，避免完整页面退化为一个 block。
Step 3：保留可操作元素和必要文本，去除与当前动作无关的冗余。
Step 4：为每个 block 生成可供本地模型排序的结构化描述。
意义：协同单位从“整个 state”缩小为“候选信息块”。
排版：左侧 XML tree；中间 partition；右侧若干 UI block。
配图：本地原图 pics/method_block_partitioin_v2.pdf。

第 16 页｜CORE：共规划、共决策与多轮累积
标题：用两阶段协作避免“小模型直接决定最终动作”
Co-planning：端云模型先对当前目标/子任务达成一致。
Local ranking：本地模型根据 sub-task 为 UI blocks 排序。
Cloud selection：云模型只读取候选 block，在其中选择具体 element/action。
Accumulation：若无法可靠决策，则追加下一候选 block，而非一次上传全页。
风险控制：本地误判可能漏掉关键元素；累积机制以更多暴露换取恢复能力。
排版：上半部两阶段流程；下半部 round 1→round 2→round 3 的逐步扩窗。
配图：本地原图 pics/method_pipeline_v3.pdf，必要时裁切方法区域。

第 17 页｜CORE 结果：隐私不是抽象口号
标题：UI exposure 最多减少 55.6%，成功率接近 cloud-only
比较对象：cloud-only、local-only 与 CORE。
主要指标：task success、uploaded UI elements/tokens、latency。
核心结果：在多种 app/task 上显著减少上传；最高 UI exposure reduction 55.6%。
观察：简单页面可激进裁剪；复杂页面需要更多轮累积。
局限：XML 质量、布局树完整性和纯视觉页面决定可用性。
排版：左侧雷达/成功率图；右侧 55.6% 大数字与三条观察。
配图：本地原图 pics/radar_chart_sr_upload_colored_labels.pdf、pics/tokens_on_both_new.pdf、pics/latency_on_both_new.pdf。

第 18 页｜协同之前：本地 Agent 自身也要足够快
标题：Agent-X：端侧 Agent 的瓶颈同时存在于 prefill 与 decode
观察：端侧 Planner/Arbiter 占主要延迟；与云服务不同，prefill 也不可忽略。
PromptWeaver：重写 prompt，使大部分静态内容可复用 prefix KV cache。
ExSpec：从 few-shot/examples 构建轻量 lookup draft，进行 LLM-free speculative decoding。
定位：不是端云 router，而是让 B 中的“设备 Agent”成为可用执行端。
排版：左侧 latency breakdown；右侧 Agent-X 双模块。
配图：本地论文原图 fig/agent_breakdown_latency.pdf、fig/proposed_overview.pdf。

第 19 页｜Agent-X 结果与 A–B 桥接
标题：端侧 Agent 全流水线加速 1.61×，且不改变任务准确率
PromptWeaver：prefill 1.97×。
ExSpec：decode 1.73×。
组合：end-to-end 1.61×，accuracy-preserving。
A–B 区别：Speculative Actions 预测未来环境动作；ExSpec 预测动作文本 token。
共同点：都依赖廉价候选生成与强语义路径的无损接受。
排版：左侧三项数字；右侧“两种 speculation”对比表。
配图：本地论文原图 fig/eval_end_to_end_speedup.pdf、fig/proposed_decode_overview.pdf。

第 20 页｜协同不止路由：模型层 split inference
标题：表示层协同把 KV、draft 与 verification 分布到端和云
端侧：输入预处理、embedding、adaptive feature、KV authentication、speculation、低维 head。
云端：decoder、KV 管理、token verification、高维 vocabulary projection。
传输：量化并加密的中间表示/截断 logits，而非原始 prompt。
结果：相对 split-inference baseline，per-token latency 最多降 46.1%，downlink 最多降 67.4%。
边界：这是 LLM inference 协同，不直接决定 Agent 的环境 action。
排版：左右端云模块 + 中央加密通道；底部结果。
配图：按论文摘要/架构重绘简图。

第 21 页｜多平台模型谱系：Mobile-Agent-v3.5
标题：从单一端云 pair 走向 2B–235B 的云边 GUI Agent 家族
GUI-Owl-1.5：2B/4B/8B/32B/235B，instruct 与 thinking 版本。
统一能力：desktop、mobile、browser、grounding、Tool/MCP、memory。
训练：hybrid data flywheel + unified thought synthesis + multi-platform MRPO。
结果：OSWorld 56.5；AndroidWorld 71.6；WebArena 48.4；ScreenSpotPro 80.3。
意义：Hera/CORE 需要的本地与云端 policy 可以来自同一跨尺寸模型谱系。
排版：上方模型尺寸阶梯；下方四项 benchmark 数字。
配图：论文模型谱系/训练总览原图（若无本地源则重绘）。

第 22 页｜可信端侧执行：AgenTEE
标题：数据留在本地不等于运行时可信
威胁：端侧 OS/hypervisor 可能被攻破；system prompt、模型权重、运行状态和第三方 app 相互不信任。
方法：Agent runtime、inference engine、third-party applications 分别放入独立 attested cVM。
通信：通过显式、可验证 channel 协作；基于 Arm CCA 远程证明。
结果：相对普通 OS 多进程部署，runtime overhead <5.15%。
定位：解决执行隔离，不解决模型幻觉或路由错误。
排版：三个隔离 cVM 盒子 + verified channel；右下角 overhead。
配图：按论文架构重绘。

第 23 页｜B–C–D 桥梁：Cognitive Firewall
标题：安全也可以端云协同：低成本本地筛查，云端语义判断，本地执行守卫
Gate 1 Edge Sentinel：本地发现不可见文本、样式欺骗等 presentation-layer 攻击。
Gate 2 Cloud Deep Planner：处理需要强语义理解的间接 prompt injection。
Gate 3 Deterministic Guard：在副作用动作提交前执行规则约束，fail closed。
结果：1,000 adversarial samples 上 ASR <1%；本地先过滤可避免无效云调用。
意义：端云协同的目标函数从 accuracy/cost 扩展为 latency/privacy/security。
排版：三道门横向流程；每道门上方标位置 local/cloud/local。
配图：按论文三阶段架构重绘。

第 24 页｜B 小结与向 C 的过渡
标题：端云协同 Agent 的研究单位正在持续细化
四个结论：
1. 从 request routing 走向 trajectory-aware step routing。
2. 从“是否上云”走向“上传哪一部分状态”。
3. 从模型推理分工走向执行、隐私与安全共同设计。
4. 比较方法必须同时报告 success、cloud use、latency、exposure 与 risk。
向 C 过渡：路由器需要真实状态与动作空间；GUI Agent 提供最具体、最具副作用的端云协同场景。
排版：左侧四结论；右侧 A–B–C–D 环形图，突出 B→C。
配图：自绘统一框架。
```

