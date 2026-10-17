# D．可靠性与 Agent 幻觉——PPT 文字大纲（25 页）

```text
第 1 页｜封面
标题：D．可靠性与 Agent 幻觉
副标题：从动作不忠实、根因归因到工具缓解与轨迹审计
页脚：Agent 系统 Survey｜武汉大学国家网络安全学院｜张帅｜2026.08
排版：纯白背景；左上角预留校徽；右侧用轨迹节点、工具、放大镜与盾牌构成线框图。

第 2 页｜先划边界：错误、幻觉与攻击不是同义词
标题：D 研究的第一步，是避免把所有 Agent failure 都叫 hallucination
Wrong action：不利于任务，但可能是合理探索或策略低效。
Hallucinated action：不被 task instruction、interaction history 或 environment observation 支持。
System failure：工具异常、超时、环境崩溃、格式错误，不一定源于模型幻觉。
Security compromise：prompt injection/越权导致错误动作，诱因来自对抗输入。
结论：可靠性是总集合；幻觉是其中“缺乏证据支撑”的子类。
排版：集合图或四象限；用交集说明边界。

第 3 页｜从最终答案走向完整轨迹
标题：Agent 可靠性至少有四个观察粒度
Response-level：最终答案是否正确/有事实依据。
Action-level：当前动作是否忠实于任务、历史和环境。
Step-level attribution：哪一步首次偏离、为什么。
Trajectory/system-level：错误如何传播，是否跨 Agent/工具/角色边界。
底部提示：最终任务成功也可能掩盖不可靠过程；最终失败也不一定来自最末一步。
排版：由点到线到系统的四层放大图。

第 4 页｜论文地图：定义—定位—机理—缓解—审计
标题：本章不按 benchmark 堆叠，而按可靠性问题链组织
定义/诱发：MIRAGE-Bench。
定位/调试：AgentHallu、AgentDebug。
机理：Reasoning Trap、Tool-Induced Myopia。
训练缓解：Relign。
运行时缓解：AgentProp-Bench interceptor、GUI verifier。
轨迹审计：Trajel。
跨方向安全：Cognitive Firewall。
排版：水平问题链；论文位于对应节点。

第 5 页｜MIRAGE：什么是 hallucinated action？
标题：动作不忠实于任务、历史或观察，才属于 Agent hallucination
Task-unfaithful：忽略约束、越出请求范围、宣称不要求的目标。
History-unfaithful：引用不存在结果、重复已失败动作、假设用户已回复。
Observation-unfaithful：编造按钮/文件/页面状态、假设环境已转移或任务已完成。
关键区分：utility=0 的错误动作可能是 hallucination；但合理探索失败不必算 hallucination。
排版：三瓣 taxonomy；中心写 Hallucinated Action。
配图：论文 taxonomy/overview 原图（本地源可复用 Agent幻觉目录或从 arXiv 提取）。

第 6 页｜MIRAGE：用风险 snapshot 稳定诱发与评测
标题：冻结高风险决策点，避免完整 rollout 的不可复现性
风险场景：out-of-scope query、unexpected transition、unachievable goal、ill-specified instruction、flawed history、popup distraction。
方法：从 WebArena、OSWorld、WorkArena、SWE-Bench、τ-bench 等环境审计风险点；冻结 context snapshot；让模型预测下一动作。
评测：risk-aware LLM judge 给 Utility 0/1/2，并统计 hallucination rate。
结论：闭源模型只略优于开源；编造 UI/完成状态与忽略环境变化普遍存在。
局限：snapshot 提高可复现性，但弱化真实长期传播。
排版：左侧完整轨迹，截取一个 snapshot；右侧风险列表和评分。

第 7 页｜AgentHallu：从二分类到根因归因
标题：知道“有幻觉”远远不够，还要找到最早 responsible step
任务 1 Where：定位轨迹中首次导致后续错误的 hallucination-responsible step。
任务 2 Why：解释该步为什么不被证据支持，并归入 taxonomy。
五大类：Planning、Retrieval、Reasoning、Human-Interaction、Tool-Use；共 14 子类。
轨迹表示：Thought–Action–Observation 多步序列。
排版：一条 6–8 步轨迹；高亮第 2 步为 root cause，后续为 propagation。
配图：AgentHallu teaser 原图。

第 8 页｜AgentHallu 数据与结果
标题：检测相对容易，精准定位仍很难
数据：693 高质量 trajectories；443 hallucinated、250 clean；7 种 agent framework、5 个 domain；平均约 7.6 steps。
评测：13 个闭源/开源模型；多层标注覆盖 binary、step、causal explanation。
结果：最佳 step localization accuracy 仅 41.1%；tool-use attribution 仅 11.6%。
趋势：轨迹越长越难；step-by-step prompting 有帮助，但 token 成本增加。
研究含义：需要低成本、可在线触发的 attribution，而不是每次交给最大 judge 全轨迹复盘。
排版：左侧数据卡；右侧 41.1% / 11.6% 大数字与长度趋势。

第 9 页｜AgentDebug：把 root cause 连接到局部重跑
标题：Where LLM Agents Fail：模块化诊断，而不是只改最后一个 action
Agent Error Taxonomy：Memory、Reflection、Planning、Action、System。
Stage 1：逐模块分析当前轨迹。
Stage 2：定位 critical/root-cause error。
Stage 3：生成针对性 feedback，从 critical step 重新 rollout。
关键点：失败可能由早期 memory/planning 产生，末尾动作只是症状。
排版：左侧五模块；右侧 detect→feedback→rerollout。
配图：本地 Agent幻觉论文 4 的方法图，优先原图。

第 10 页｜AgentDebug 结果与成本
标题：正确的重启位置比“从头再来”或“只改最后一步”更有效
数据：200 条专家标注失败轨迹，覆盖 ALFWorld、WebShop、GAIA。
检测：AgentDebug 的 all-correct 与 step accuracy 优于 direct prompting/self-refine 等 baseline。
修复：论文报告下游 task success 相对最多提升 26%；多个模型/环境均受益。
代价：强 judge + 部分 trajectory replay；general failure taxonomy 不完全等同于 hallucination。
启示：D 的最终目标应是 attribution-guided intervention，而非只做审计报告。
排版：左侧定位结果；右侧三种重跑策略对比。

第 11 页｜为什么工具幻觉是 Agent 可靠性的核心切口
标题：工具把“语言错误”转化为可执行副作用
类型 1 Tool selection：调用不存在/不相关工具，或调用时机错误。
类型 2 Tool usage：JSON/schema 错、参数名/类型错、参数内容编造。
传播链：fabricated parameter → tool output → reasoning accepts output → wrong final action/answer。
特殊能力：rejection（执行前拒绝坏参数）与 recovery（接受后能纠错）并非同一种能力。
排版：上方四类 taxonomy；下方错误传播链。

第 12 页｜Reasoning Trap：推理增强会自动降低工具幻觉吗？
标题：控制实验给出反直觉答案：think harder 可能更敢于调用错误工具
Benchmark：SimpleToolHalluBench。
NTA：任务需要工具，但当前没有任何工具；正确行为是说明无法完成/询问用户。
DT：只提供 distractor tool；正确行为仍是不误用。
对照：tool RL、math GRPO、R1 distillation、thinking on/off、prompt、DPO。
研究问题：是工具训练过拟合，还是 reasoning enhancement 本身改变了行动倾向？
排版：左侧 NTA/DT 两个场景；右侧多种增强方式。
配图：论文 Figure 1/实验矩阵原图。

第 13 页｜Reasoning Trap 发现：能力–可靠性存在张力
标题：RL、蒸馏和推理模式均出现“性能越高，工具幻觉越多”
发现 1：tool-specific RL 提高任务 reward，同时提高 hallucination。
发现 2：即使只做 math GRPO，也会提高后续 tool hallucination，说明不只是工具数据过拟合。
发现 3：R1 distillation 与 thinking mode 同样放大问题。
缓解：prompt engineering 很弱；DPO 可降低 hallucination，但牺牲 utility。
结论：训练目标必须显式优化 abstention/solvability/reliability，不能假设 reasoning 会自然带来谨慎。
排版：左侧能力↑/可靠性↓示意曲线；右侧四条发现。

第 14 页｜Tool-Induced Myopia：工具正确也可能推理失真
标题：最终答案更准，不代表工具增强过程更可信
现象：模型把 Code Interpreter 输出当作证明的替代，跳过必要逻辑、假设与创造性推理。
PyMath：1,679 道 Python 有帮助但不充分的竞赛数学题。
结果：tool-augmented model 最终准确率最高可提升 19.3pp；但过程 pairwise win、miss rate、PRM 等变差；高风险案例约 55% 出现 TIM。
启示：工具 observation 进入最终结论前，还需要 reasoning faithfulness check。
排版：左侧“答案对/过程错”案例；右侧 final accuracy vs reasoning quality 双轴。

第 15 页｜Relign：先把工具幻觉定义清楚
标题：四类错误 + 三类结果，构成可训练的可靠工具调用目标
四类：tool type、tool timing、tool format、tool content hallucination。
RelyToolBench：在 StableToolBench 基础上加入 Missing Parameter 与 Unmatched Tools。
指标：Tool Hallucination Rate、Reliable Pass Rate、Benefit–Cost Utility。
三类响应：correct tool action；indecisive action；hallucinated action。
排版：左侧 2×2 taxonomy；右侧三层质量排序。
配图：Relign 原论文 Figure 1/2。

第 16 页｜Relign 方法：扩展 action space，而不是强迫模型每次都调用工具
标题：条件不足时，ChangeTools / TalkToUser 比编造参数更可靠
Reliable SFT：构造工具替换与参数缺失样本，教会模型输出 indecisive actions。
Reliable DPO：从多个轨迹构造偏好对：correct > indecisive > hallucinated。
意义：将“拒绝、澄清、换工具”视为有效 Agent action，而非任务失败。
公式化偏好：a_correct ≻ a_indecisive ≻ a_hallucinated。
局限：固定的两类 indecisive action 仍较粗；timing/重复调用涉及轨迹记忆。
排版：上半部 action space；下半部 SFT→sampling/evaluator→DPO。

第 17 页｜Relign 结果
标题：可靠性对齐同时降低工具幻觉和无效调用
ToolLLaMA：overall THR 约 56.5 → 18.3。
Llama-3.1-8B：约 50.8 → 14.6，RePR 提升到约 77.2。
Qwen2.5-7B：约 49.2 → 17.6。
观察：常规 tool-learning 可提高 pass rate，却不自动降低 hallucination/cost。
局限：依赖 GPT-4o evaluator 训练与评测；对长轨迹 logical/referential 幻觉覆盖有限。
排版：三组 before/after bar；右侧观察与局限。

第 18 页｜AgentProp-Bench：先验证 judge 是否可信
标题：如果自动 judge 与人类接近随机，所有缓解结论都会失真
数据：约 2,000 tasks、2,300 traces，Calendar/Weather/Medical/Knowledge；Retail held-out。
干预：把首次关键参数替换为 schema 合法但语义错误的值。
三个阶段：S1 注入发生；S2 工具执行受影响；S3 最终答案错误。
Judge 结果：substring heuristic 与人类 κ=0.049；3-LLM ensemble κ=0.432，但偏保守。
排版：左侧注入链；右侧 judge-human κ 对比。

第 19 页｜AgentProp：传播、拒绝、恢复与运行时拦截
标题：错误参数传播到最终错误的概率约 0.62
发现：不同模型传播概率约 0.46–0.73。
能力分解：rejection 与 recovery Spearman ρ=0.126，近似独立。
Runtime interceptor：L1 schema/tool error；L2 thought keyword monitor；L3 unsupported output consistency。
结果：GPT-4o-mini hallucination 55.8%→32.8%，下降 23pp；对高拒绝率 Gemini 效果不显著。
结论：runtime mitigation 必须针对具体 failure mode 与 model behavior 调整。
排版：上方错误传播 Sankey；下方三层 interceptor 与结果。

第 20 页｜Trajel：工业多 Agent 的轨迹级审计
标题：只审最终 response，会漏掉 procedural 与 scope hallucination
数据：225 条专家标注 AssetOps multi-agent trajectories，来自 42 个工业问题与 6 个模型配置。
五类：Factual、Referential、Logical、Procedural、Scope。
检测：subtask-level BERT/NLI、trajectory-level Longformer、LLM-as-a-Judge。
特点：Agent 角色边界与工业流程顺序也是 evidence；越权或跳步即使最终答案合理仍不可靠。
排版：左侧工业 workflow；右侧五类 taxonomy。
配图：本地论文 Architecture.png 与原论文 taxonomy 图。

第 21 页｜Trajel 结果：多类型与细粒度仍困难
标题：68.3% 轨迹被人工识别为 hallucinated，近半包含多种类型
统计：procedural 约 38.5%，factual 26.3%，scope 19.8%。
LLM judge：binary F1 约 0.855、recall 高但偏过度标注；logical/referential F1 仅约 0.258/0.222。
简单 execution-quality signals 很强：clarity/justification AUC 约 0.908。
当 justification 与 result verification 同时缺失，hallucination rate 约 97.1%。
启示：低成本运行时 kill-switch 可能来自可解释执行信号，而非更大分类器。
排版：左侧类型占比；右侧 detector 指标与 97.1% 风险框。

第 22 页｜C–D 桥梁：把可靠性嵌入 GUI 执行
标题：运行时缓解可以分布在动作提交前后
Pre-action：GUI-Critic-R1 检查 candidate outcome，防止不可逆动作。
Post-effect：VeriGUI 比较 expectation 与 actual state，避免失败动作循环。
Post-trajectory：VAGEN/IRA 主动取证，避免“看起来完成”的幻觉。
与 Relign/AgentProp 的对应：policy alignment 决定是否行动；runtime verifier 决定是否提交/继续/确认完成。
排版：同一 GUI trajectory 上标三道检查点。

第 23 页｜B–C–D 桥梁：Cognitive Firewall
标题：概率模型负责判断，确定性 Guard 负责限制副作用
本地 Sentinel：低成本过滤 presentation-layer prompt injection。
云端 Deep Planner：处理高语义复杂度的间接攻击。
本地 deterministic Guard：在执行边界限制敏感 side-effect actions，fail closed。
结果：ASR <1%；同时减少不必要云推理。
可靠性启示：即使 semantic detector 漏判，也不应让模型自由提交所有外部副作用。
排版：三道门 + 攻击流；突出最后执行边界。

第 24 页｜统一可靠性闭环
标题：Detect → Attribute → Decide → Verify → Recover → Audit
Detect：动作/轨迹是否缺乏支持。
Attribute：定位 root cause step 与错误类型。
Decide：allow / abstain / ask / change tool / escalate / block。
Verify：检查工具输出、GUI state transition 与完成条件。
Recover：从 critical step 重试、回滚或换策略。
Audit：记录证据、角色边界、judge 决策与干预成本。
设计目标：在 task utility、hallucination risk、latency/cost 与 over-refusal 之间形成 Pareto。
排版：六阶段闭环；中央放多目标天平。

第 25 页｜D 小结与 A–D 合流
标题：可靠性不是最后附加的模块，而是其他三个方向成立的条件
对 A：投机动作只有被可靠验证、可停止/回滚时才真正 lossless。
对 B：路由必须同时考虑能力、成本、隐私与风险；高风险语义可升级云端，副作用在本地守卫。
对 C：GUI success 必须由可见与隐藏状态共同证明，且错误恢复应单独评测。
总观点：A 决定何时并行，B 决定在哪里执行，C 决定执行什么，D 决定能否信任。
排版：四象限 When/Where/What/Trust；底部一句总观点。
```

