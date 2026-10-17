# A–D 统一框架与桥接论文矩阵

## 1. 四个方向不是四条平行线

统一 Agent runtime：

`Observation → Reasoning/Planning → Candidate Actions → Placement/Execution → Environment Feedback → Verification/Recovery`

- A（动作投机）回答：**何时并行、预测多少未来动作、如何接受/回退？**
- B（端云协同）回答：**每个阶段放在哪里执行、何时升级到云端、传输什么？**
- C（GUI 执行）回答：**在什么真实环境中观察、定位、点击、调用 API 并完成任务？**
- D（可靠性/幻觉）回答：**动作是否被证据支持、出错如何定位、阻断、恢复和审计？**

可压缩成四个正交坐标：

1. When：串行还是投机（A）
2. Where：端、边、云如何放置（B）
3. What：API、工具、GUI、OS 动作（C）
4. Trust：接受、验证、回滚、审计（D）

## 2. 桥接论文矩阵

| 论文 | A | B | C | D | 主报告定位 |
|---|:---:|:---:|:---:|:---:|---|
| Speculative Actions | ● | ○ | ○ | ● | A 起点；lossless verification 是 A–D 接口 |
| Dynamic Speculative Agent Planning | ● | ○ |  | ○ | 动态投机与成本/延迟策略 |
| PASTE | ● | ○ |  | ○ | 工具依赖图与并行执行 |
| Agent-X | ● | ● | ○ | ○ | 端侧 LLM-free speculation，A–B 桥 |
| Cassandra | ● | ● |  | ○ | edge self-speculative decoding，A–B 补充 |
| Hera | ○ | ● | ○ | ○ | state-level cloud escalation，未来可由风险/投机置信度驱动 |
| CORE |  | ● | ● | ● | UI 选择性披露；B–C 核心桥，隐私属于广义可靠性 |
| UFO² | ● | ○ | ● | ● | speculative multi-action + OS execution + safeguard |
| Mobile-Agent-v3.5 | ○ | ● | ● | ○ | 多尺寸、多平台 cloud-edge GUI Agent |
| GUI-Critic-R1 | ○ | ○ | ● | ● | 执行前 critic，可验证投机/GUI candidate action |
| VeriGUI | ○ |  | ● | ● | action-effect verification 与自恢复 |
| VAGEN / IRA |  | ○ | ● | ● | 交互式 verifier/奖励，读取隐藏环境状态 |
| Cognitive Firewall |  | ● | ● | ● | edge sentinel + cloud planner + deterministic guard |
| AgentHallu / AgentDebug | ○ | ○ | ○ | ● | 为所有 Agent 轨迹提供 root-cause attribution/debugging |
| Relign / AgentProp | ○ | ○ | ○ | ● | 工具调用的 policy alignment 与 runtime interception |

图例：● 直接贡献；○ 可形成自然接口但非论文主贡献。

## 3. 必须额外汇报的桥接论文

### CORE（B+C，兼具隐私可靠性）

- 不在 B、C 两章重复整篇讲。
- B 章强调“传什么到云端”：UI block selection 与 exposure metric。
- C 章只引用其移动 GUI pipeline 与 XML 观察假设。
- 桥接页总结：协同路由可以在数据粒度而非模型粒度发生。

### UFO²（A+C+D）

- C 章主讲系统架构。
- A 章回链 speculative multi-action：一次 LLM 生成多个动作，以 control-state check 接受/中止。
- D 章回链 safeguard、隔离 PiP 和执行状态验证。

### Cognitive Firewall（B+C+D）

- 本地低成本视觉检测 → 云端深语义判断 → 本地确定性动作守卫。
- 是 A–D 主报告中最清楚的“性能、部署、安全三目标共同设计”案例。

### Agent-X（A+B）

- 说明 Agent speculation 不只发生在 environment action，也发生在本地 LLM token generation。
- 关键区别：Speculative Actions 预测未来环境动作；Agent-X ExSpec 预测动作文本 token。

### GUI-Critic-R1 / VeriGUI（C+D，并可服务 A）

- pre-critic 适合验证高副作用候选动作；post-effect verifier 适合确认投机动作是否真的生效。
- 构成“执行前阻断 + 执行后恢复”的两道门。

## 4. 主报告过渡语

- A→B：即使未来动作能被投机预测，Speculator、Actor、Verifier 放在端还是云，决定真实收益与成本。
- B→C：端云协同需要具体环境才能定义“状态、隐私与动作”；GUI 是最具代表性的落点。
- C→D：GUI 动作会产生真实副作用，success-only 评测无法回答动作是否可靠，因此必须进入验证、恢复与幻觉审计。
- D→总体：可靠性不是最后附加的一层；它决定哪些投机可接受、哪些状态可上传、哪些 GUI 动作可提交。

## 5. 主报告建议页数

| 模块 | 页数 | 内容 |
|---|---:|---|
| 开场与统一框架 | 5 | 问题、Agent loop、四坐标、论文地图 |
| A 精简回顾 | 12 | 种子 + 代表方法 + 对 B/C/D 的接口 |
| B 端云协同 | 18 | Minions、Hera、CORE、Agent-X、trustworthy edge |
| C GUI 执行 | 22 | OSWorld、UFO²、GUI-Owl、pre/during/post verification |
| D 可靠性/幻觉 | 22 | definition、attribution、mechanism、mitigation、audit |
| 跨方向桥接与结论 | 8 | CORE/UFO²/Cognitive Firewall/Agent-X、统一对比、结论 |
| 合计 | 87 | 适合 70–90 页主报告；详细方法放方向附件 |

