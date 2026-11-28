#### Official Review of Submission11343 by Reviewer uXrn

Official Reviewby Reviewer uXrn17 Nov 2026, 03:57 (modified: 25 Nov 2026, 01:46)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer uXrn, Authors[Revisions](https://openreview.net/revisions?id=WkxRzDwZhM)

**Paper Summary:**

FR-SVD introduces a two-stage SVD-based model compression method with a fixed rank budget. In the first stage, it performs task-aware low-rank truncation by leveraging Fisher Information, activation statistics, and cumulative input alignment. In the second stage, it employs ridge regression to compensate for output errors—arising from weight truncation and shifts in upstream input distributions—within the constraints of the remaining rank budget. Experimental results demonstrate that this approach improves both perplexity and zero-shot accuracy across various LLaMA and Qwen models; specifically, for LLaMA-7B at a 60% compression rate, it achieves a 12% reduction in WikiText-2 perplexity compared to SAES-SVD.

**Summary Of Strengths:**

1. The proposed method pushes the performance of SVD-based LLM compression to a new state-of-the-art.
2. The proposed method provides a comprehensive optimization by simultaneously addressing weight reconstruction error, hidden-feature drift, and output error.

**Summary Of Weaknesses:**

1. All experiments were conducted on standard dense models; the performance on hybrid architectures or Mixture-of-Experts (MoE) models was not evaluated.
2. The second stage of FR-SVD appears quite similar to the approach used in AA-SVD [1].
3. Regarding the GPU memory cost of compression for FR-SVD: given that the process requires computing full gradients, are the reported execution times based on a setup where all model weights and gradients are stored in GPU memory?
4. I am interested in the generalization capability of FR-SVD across different data distributions—specifically, how it performs when the calibration dataset and the benchmark dataset originate from different domains. This would help clarify whether the error gradients primarily capture calibration information related to the model's intrinsic characteristics, or if they contain task-specific information.

[1] AA-SVD: Anchored and Adaptive SVD for Large Language Model Compression

##### 中文译文

**论文概述**

FR-SVD 提出一种在固定秩预算下分两阶段进行的 SVD 模型压缩方法。第一阶段利用 Fisher 信息、激活统计量和累积输入对齐进行任务感知的低秩截断。第二阶段在剩余秩预算内使用岭回归，补偿权重截断和上游输入分布偏移造成的输出误差。实验表明，该方法在多个 LLaMA 和 Qwen 模型上改善了困惑度和零样本准确率。具体而言，当 LLaMA-7B 的压缩率为 60% 时，相比 SAES-SVD，FR-SVD 将 WikiText-2 困惑度降低了 12%。

**优点**

1. 该方法将基于 SVD 的大语言模型压缩性能提升到了新的最先进水平。
2. 该方法同时处理权重重构误差、隐藏特征漂移和输出误差，优化较为全面。

**缺点**

1. 所有实验均在标准稠密模型上进行，尚未评估该方法在混合架构或混合专家（Mixture-of-Experts，MoE）模型上的表现。
2. FR-SVD 的第二阶段似乎与 AA-SVD [1] 使用的方法相当相似。
3. 关于 FR-SVD 压缩过程的 GPU 显存开销，既然该过程需要计算完整梯度，论文报告的运行时间是否是在所有模型权重和梯度都存放在 GPU 显存中的条件下测得的？
4. 我想了解 FR-SVD 在不同数据分布下的泛化能力，尤其是在校准数据集和评测数据集来自不同领域时的表现。这有助于判断误差梯度主要捕捉的是与模型内在特性有关的校准信息，还是也包含任务特定的信息。

[1] AA-SVD: Anchored and Adaptive SVD for Large Language Model Compression





#### Official Review of Submission11343 by Reviewer QwTt

Official Reviewby Reviewer QwTt14 Nov 2026, 15:05 (modified: 25 Nov 2026, 01:46)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer QwTt, Authors[Revisions](https://openreview.net/revisions?id=FVPA4PeyfE)

**Paper Summary:**

This paper studies low-rank LLM compression without post-compression fine-tuning. Under a fixed per-layer total rank , FR-SVD first performs cumulative-error-aware truncation weighted by activation statistics and gradient-fitted Kronecker Fisher factors, then uses the remaining rank for a ridge-regression correction toward full-precision outputs. The compensation addresses both current-layer truncation and upstream-input shifts. With the calibration statistics and alignment coefficient fixed, the authors derive an optimal solution to each stage's objective. They also show that exact compensation cannot increase the calibration residual when the first stage is held fixed. Experiments cover six decoder-only backbones: LLaMA-7B at 20%, 40%, and 60% compression, and the others at 20%. At 60% compression on LLaMA-7B, the reported WikiText-2 perplexity decreases from SAES-SVD's 22.01 to 19.36, a 12.0% relative reduction; 22.01 is a reference value taken from the original SAES-SVD paper.

**Summary Of Strengths:**

**S1.** The objectives and theoretical conditions are reasonably clear. Pages 3–5 and Appendices A–B use invertible transformations to reduce the two fixed rank-constrained problems to low-rank approximation, yielding explicit solutions and a residual bound. The appendices also distinguish the mixed surrogate metric from the Hessian and the regularized objective from the unregularized residual, and discuss finite damping and approximate SVD. These analyses are useful.

**S2.** The experiments have reasonable breadth and include meaningful parameter controls. Table 3 on page 7 fixes the total rank within each compression ratio and compares Fisher-only compression, ordinary CEALC plus compensation, and the complete method. The complete method improves over both variants, with absolute perplexity reductions of 0.06, 0.21, and 1.06 relative to the Fisher-only variant. Table 2 and Appendix D provide cross-model evidence, although the 13B and 30B experiments report perplexity only.

**S3.** The paper discloses computational costs and scope limitations. Page 8 reports calibration sensitivity and one-time compression time; page 9 acknowledges missing direct comparisons with the closest methods and the fixed rank split; page 17 explains that inference executes the factors directly without reconstructing dense weights.

**Summary Of Weaknesses:**

**W1. Paired comparability of the key baselines and evaluation against the closest methods remain insufficient.**

Section 5.1 on page 6 explicitly states that the LLaMA-7B SAES-SVD WikiText-2 values in Tables 1 and 3 are taken from the original paper; the study also uses an adapted third-party SAES-SVD implementation. This disclosure is helpful, but a complete provenance mapping and paired results under a common configuration are missing. Equal nominal compression ratios do not establish equal compressed-matrix sets, per-layer ranks and actual parameter budgets, procedures for calibration and configuration selection, or evaluation implementations. I recommend reporting the provenance of each result and rerunning the key comparisons using common weights, actual budgets, and evaluation procedures. The unusually high FWSVD perplexities and near-zero task accuracies in Table 1 also warrant configuration details, diagnostics, or raw-output checks.

Moreover, page 2 discusses GFWSVD, EoRA, and AA-SVD, and page 9 explicitly acknowledges the absence of matched-budget direct comparisons. The existing Table 3 ablations support the internal combined design, but are not sufficient to establish the gain over these neighboring methods. The authors should prioritize the most relevant comparable methods. Where adaptation is infeasible in the present framework, the paper should explain why and limit the conclusions accordingly. This is a comparative-evidence concern, not a missing-citation criticism.

**W2. Independence between hyperparameter selection and final testing is unclear.**

Page 6 specifies calibration using 256 WikiText-2 training sequences of length 2048. However, Table 4 on pages 7–8 compares and selects  and  using WikiText-2 perplexity, and Appendix E.3 on page 17 uses this tuned setting for the rank-allocation study, with the default  near the best region. The paper does not clearly distinguish calibration data, validation data for configuration selection, and the final reported WikiText-2 test data, or state when these settings were fixed and transferred to other models. Please provide the splits and selection rules. If the final test data participated in selection, fix the configurations on independent validation data and rerun the evaluation. This is an unresolved issue affecting interpretation of the gains.

**W3. Numerical fallback behavior and practical solver details leave some ablation evidence unresolved.**

Appendix A.5 on page 14 gives implicit Kronecker leading-pair computation and initialization, and Appendix B.5 on page 16 analyzes approximate SVD. Nevertheless, the actual iteration limits, stopping tolerances, and truncation implementation are unclear. The statement of half precision on page 6 does not specify the precision used to accumulate statistics and perform matrix decompositions. More concretely, the Figure 6 caption on page 17 attributes the +0.0779 at L31/down_proj to a numerical fallback artifact in the Stage-1-only baseline. Please explain its trigger, replacement operation, and affected scope, and confirm whether the F variant in Table 3 on page 7 is affected. If the solver or fallback policy requires correction, rerun the affected local diagnostics after that correction. Equation (57) measures the local target , whereas compensation targets .

The five-seed and calibration-size experiments in Figure 3 are useful controls, but cover only LLaMA-7B at 20% compression and do not give paired variance for the main method differences. At higher compression, repeating at least the key method comparisons with the same random seeds and reporting variability in their differences would help establish robustness. This concern is about numerical and empirical reliability.

**W4. Deployment and quantization results need more complete cost accounting.**

The compression times in Table 5 on page 8, 1 h 14 min 06 s versus 1 h 35 min 29 s, correspond to a 28.9% increase in one-time cost. Figure 4 also specifies one A40, batch size 4, prefill length 1024, and decoding length 256. Under this configuration, the model at 60% nominal compression has 5.73 GiB of weights and a 2.10× generation speedup. It would strengthen the paper to also report absolute throughput and latency, together with peak memory during compression and inference. Explain how the timing procedure handles warm-up, synchronization, and repeated measurements, and distinguish nominal parameter reduction in compressed linear weights from whole-model storage. Broader deployment claims would require representative batch sizes and context lengths. Appendix F explains shared statistics across variants, so their times cannot simply be subtracted to isolate component costs.

Table 6 on page 18 specifies a common 2.8 GB weight budget, GPTQ at 3 bits, and 30% FR-SVD compression followed by 4-bit GPTQ on the factors. The combined perplexity is 9.65 versus 16.28 for GPTQ at 3 bits, a 40.7% relative reduction. The missing information concerns GPTQ configurations and calibration settings, whether retained high-precision weights and metadata are included in memory, and the scope of a single-model, single-budget result. Please supply these accounting details and restrict the current conclusion to the tested configuration; the bit width is not the missing factor in this comparison.

**Comments Suggestions And Typos:**

**C1.** Compute Avg and Drop consistently from unrounded task results and provide higher precision or raw outputs. For LLaMA-3-8B in Table 2, averaging the six printed accuracies gives approximately 0.613333 for the original model and 0.565000 for FR-SVD, implying about 7.88% Drop. Using the rounded Avg values 0.61 and 0.57 instead gives about 6.6%, as printed. The reported precision may explain the discrepancy, so this is not a confirmed calculation error, but the precision rule needs clarification.

**C2.** Editorial improvements include explicitly giving the dimensions of the candidate matrix  and correction matrix  at first use, connecting CEALC directly to the subsection's expanded name, using different symbols for Algorithm 2's candidate set and quadratic coefficient C, and providing a more direct pointer to the fixed-projection definition. The paper distinguishes the true energy ratio in Equation (9) from the Appendix A.4 proxy. A small representative-layer comparison of both quantities and independent validation error as  varies could empirically explain the heuristic. Definitions may appear in adjacent text or referenced appendices; every equation does not need a separate “where” clause.

**C3.** Increase the tick and cell-value sizes in Figures 3 and 6. Figure 8 should include a clear legend and markers that do not rely on red and green hues for correct versus incorrect caption fragments. Figure 4 uses bars, open circles, and a dashed line; its main presentation issue is the contrast of small text.

##### 中文译文

**论文概述**

本文研究无需在压缩后进行微调的大语言模型低秩压缩。在每层总秩固定的条件下，FR-SVD 首先结合激活统计量和用梯度拟合得到的 Kronecker Fisher 因子进行加权，并在截断时考虑累积误差。随后，该方法利用剩余秩预算进行岭回归校正，使压缩模型的输出逼近全精度模型的输出。补偿同时考虑当前层的截断误差和上游输入偏移。在校准统计量和对齐系数固定时，作者推导了各阶段目标的最优解。作者还证明，在第一阶段固定时，精确求解的补偿不会增大校准残差。实验覆盖六个仅解码器架构的骨干模型，其中 LLaMA-7B 分别采用 20%、40% 和 60% 的压缩率，其余模型采用 20%。当 LLaMA-7B 的压缩率为 60% 时，论文报告的 WikiText-2 困惑度由 SAES-SVD 的 22.01 降至 19.36，相对降低 12.0%。22.01 是取自 SAES-SVD 原论文的参考值。

**优点**

**S1.** 目标函数和理论条件较为清楚。第 3–5 页及附录 A–B 通过可逆变换，将两个具有固定秩约束的问题化为低秩近似问题，得到显式解及残差界。附录还区分了混合代理度量与 Hessian，以及正则化目标与未正则化残差，并讨论了有限阻尼和近似 SVD。这些分析有帮助。

**S2.** 实验覆盖面合理，参数控制也有意义。第 7 页表 3 在每种压缩率下固定总秩，比较了仅使用 Fisher 的压缩方法、普通 CEALC 加补偿，以及完整方法。完整方法优于两种变体，相比仅使用 Fisher 的变体，困惑度分别降低 0.06、0.21 和 1.06。表 2 和附录 D 提供了跨模型的实验结果，但 13B 和 30B 模型的实验只报告了困惑度。

**S3.** 论文披露了计算成本和适用范围的局限。第 8 页报告了对校准设置的敏感性和一次性压缩耗时。第 9 页承认缺少与最接近方法的直接对比，并指出两阶段的秩分配是固定的。第 17 页说明，推理时直接使用低秩因子计算，无需重构稠密权重。

**缺点**

**W1. 关键基线的配对可比性不足，对最接近方法的评估也不充分。**

第 6 页第 5.1 节明确说明，表 1 和表 3 中 SAES-SVD 在 LLaMA-7B 上的 WikiText-2 数值取自原论文。本研究也使用了一个经过适配的第三方 SAES-SVD 实现。这些说明有帮助，但论文缺少逐项对应的完整结果来源，也缺少统一配置下的配对结果。相同的名义压缩率无法保证被压缩的矩阵集合、逐层秩、实际参数预算、校准与配置选择流程以及评测实现都一致。建议逐项报告结果来源，并使用相同的模型权重、实际预算和评测流程重跑关键对比。表 1 中 FWSVD 的困惑度异常偏高，任务准确率接近零，也需要补充配置细节、诊断结果或原始输出核查。

此外，第 2 页讨论了 GFWSVD、EoRA 和 AA-SVD，第 9 页也明确承认缺少在相同预算下的直接对比。现有表 3 的消融实验支持方法内部的组合设计，但不足以证明它相对于这些相近方法的增益。作者应优先比较相关性最高且具备可比性的方法。如果某个方法无法适配现有框架，应解释原因，并相应收窄结论。这里关注的是对比证据是否充分，引用本身已经给出。

**W2. 超参数选择与最终测试是否独立尚不清楚。**

第 6 页指出，校准使用 WikiText-2 训练集中的 256 条序列，每条长度为 2048。然而，第 7–8 页表 4 使用 WikiText-2 困惑度比较和选择两个超参数，随后第 17 页附录 E.3 又在秩分配研究中使用了这套调参后的设置，其中一个参数的默认值接近最优区间。〔此处文档中的英文缺少两个超参数的符号及默认值所对应的符号。〕论文没有明确区分校准数据、用于配置选择的验证数据和最终报告所用的 WikiText-2 测试数据，也没有说明这些配置何时确定、何时迁移到其他模型。请给出数据划分和选择规则。如果最终测试数据参与了配置选择，请在独立的验证数据上确定配置，然后重新评估。这个问题尚未得到澄清，会影响对增益的解读。

**W3. 数值回退行为和求解器实现细节仍使部分消融证据存疑。**

第 14 页附录 A.5 给出了 Kronecker 主导因子对的隐式计算与初始化，第 16 页附录 B.5 分析了近似 SVD。不过，实际的迭代次数上限、停止容差和截断实现仍不清楚。第 6 页提到使用半精度，但没有说明累积统计量和进行矩阵分解时采用的精度。更具体地说，第 17 页图 6 的图注将 L31/down_proj 处的 +0.0779 归因于仅使用第一阶段的基线发生数值回退所产生的异常。请说明回退的触发条件、替代操作和影响范围，并确认第 7 页表 3 的 F 变体是否受到影响。如果求解器或回退策略需要修正，请在修正后重跑受影响的局部诊断。公式（57）衡量的是局部目标，而补偿针对的是另一个目标。〔此处文档中的英文缺少两个目标的公式符号。〕

图 3 中的五个随机种子实验和校准集大小实验提供了有用的控制，但只覆盖 LLaMA-7B 在 20% 压缩率下的情况，也未报告主要方法之间差值的配对方差。在更高压缩率下，至少用相同的随机种子重复关键方法对比，并报告方法间差值的波动程度，有助于验证结果的稳健性。这里关注的是数值和实验结果的可靠性。

**W4. 部署与量化结果需要更完整的成本核算。**

第 8 页表 5 给出的压缩耗时分别为 1 小时 14 分 06 秒和 1 小时 35 分 29 秒，对应一次性成本增加 28.9%。图 4 还说明，实验使用一张 A40，批大小为 4，预填充长度为 1024，解码长度为 256。在这套配置下，名义压缩率为 60% 的模型权重占用 5.73 GiB，生成速度提升至 2.10 倍。建议同时报告绝对吞吐量、延迟，以及压缩和推理期间的峰值显存。请说明计时流程如何处理预热、同步和重复测量，并区分被压缩线性权重的名义参数缩减与整个模型的存储占用。若希望得出更广泛的部署结论，应覆盖具有代表性的批大小和上下文长度。附录 F 说明不同变体共享统计量，因此无法直接用耗时相减分离各组件的成本。

第 18 页表 6 采用统一的 2.8 GB 权重预算，对比 3 位 GPTQ 与先做 30% 的 FR-SVD 压缩、再对低秩因子做 4 位 GPTQ 的组合方案。组合方案的困惑度为 9.65，3 位 GPTQ 为 16.28，相对降低 40.7%。有待补充的是 GPTQ 配置和校准设置、保留的高精度权重与元数据是否计入占用，以及单一模型和单一预算下这一结果的适用范围。请提供这些核算细节，并将当前结论限定在已测试的配置内。表中已明确给出量化位宽。

**评论、建议与勘误**

**C1.** 使用未四舍五入的任务结果统一计算 Avg 和 Drop，并提供更高精度的数值或原始输出。对于表 2 中的 LLaMA-3-8B，直接对表中列出的六项准确率取平均，原模型约为 0.613333，FR-SVD 约为 0.565000，据此得到的 Drop 约为 7.88%。如果使用表中的舍入后平均值 0.61 和 0.57，则会得到约 6.6%，与表中所列一致。报告精度可能解释这个差异，因此暂不能认定有计算错误，但需要澄清精度规则。

**C2.** 写作方面，建议在候选矩阵和校正矩阵首次出现时明确其维度，将 CEALC 缩写直接关联到小节标题中的全称，在算法 2 中为候选集合与二次项系数 C 使用不同符号，并更直接地指向固定投影的定义。论文区分了公式（9）中的真实能量比与附录 A.4 中的代理量。可以选取少数有代表性的层，比较这两个量，同时观察独立验证集上的误差随相应参数变化的情况，为该启发式方法提供实证解释。〔此处文档中的英文缺少两个矩阵的符号和被改变的参数符号。〕定义可以放在相邻正文中，也可以指向附录，无需为每个公式单列一段符号说明。

**C3.** 请增大图 3 和图 6 中的刻度标签和单元格数字。图 8 应提供清晰的图例，并使用不依赖红绿色彩区分的标记来表示图注中正确和错误的片段。图 4 使用了柱形、空心圆和虚线，主要的呈现问题是小号文字的对比度。





#### Official Review of Submission11343 by Reviewer khDv

Official Reviewby Reviewer khDv30 Oct 2026, 15:46 (modified: 25 Nov 2026, 01:46)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer khDv, Authors[Revisions](https://openreview.net/revisions?id=qQUKDTym89)

**Paper Summary:**

The authors propose FR-SVD, a two-stage low-rank compression framework that combines Fisher-Guided Cumulative Error-Aware Truncation with Eigenspace Residual Compensation to address two key problems of traditional SVD compression: it may discard important weight directions that are sensitive to the downstream task loss, and compression of upstream layers can shift the input distribution of subsequent layers and thereby accumulate reconstruction error. In the first stage, building on the cumulative error-aware compression objective of SAES-SVD, the method constructs Kronecker-factored Fisher weights from the calibration loss gradient so that different low-rank directions receive different importance and those most sensitive to model loss are preferentially retained. In the second stage the remaining rank budget is used to fit the residual between the compressed-model output and the full-precision-model output via regularized low-rank ridge regression, simultaneously compensating for the current-layer SVD truncation error and the upstream input shift. The authors provide a closed-form optimal solution under fixed statistics and evaluate the approach on six models, reporting solid performance together with actual model weight sizes and inference speed-up results.

**Summary Of Strengths:**

1. The motivation is fairly clear. Both problems have relatively precise mathematical definitions. Figure 1 further shows that FR-SVD reduces intermediate-layer input shift compared with SAES-SVD, providing experimental support for the research motivation.
2. The two-stage design has a clear logical division of labor and feels natural.
3. The experiments are relatively thorough, covering multiple models and multiple benchmarks.

**Summary Of Weaknesses:**

1. The novelty of the core method is limited. It is mainly a combination of existing compression techniques, and the paper’s contribution leans more toward integrating and improving current SVD compression mechanisms.
2. Several key baselines are missing, such as GFWSVD and AA-SVD, which respectively correspond to the core technical ideas of the two stages in FR-SVD.

**Comments Suggestions And Typos:**

1. I suggest directly validating whether Fisher-guided truncation preserves more loss-sensitive singular directions. Specifically, the authors could compare the calibration-loss sensitivity of the singular directions discarded by Fisher-guided and activation-aware SVD under identical rank budgets. This would provide more direct evidence that Fisher weighting improves the selection of task-relevant directions, rather than merely changing the approximation metric.
2. The heavy compression results are reported for only one model. Could the authors also show the results for the remaining three models so that the method’s behavior under different compression ratios can be better observed?
3. Could you explain in more detail why it is not possible to compare against key methods such as GFWSVD and AA-SVD?

##### 中文译文

**论文概述**

作者提出了 FR-SVD，这是一种由 Fisher 引导的累积误差感知截断（Fisher-Guided Cumulative Error-Aware Truncation）和特征空间残差补偿（Eigenspace Residual Compensation）组成的两阶段低秩压缩框架。它针对传统 SVD 压缩的两个主要问题。其一，传统方法可能丢弃对下游任务损失敏感的重要权重方向。其二，上游层压缩会导致后续层的输入分布发生偏移，从而使重构误差累积。第一阶段以 SAES-SVD 的累积误差感知压缩目标为基础，从校准损失的梯度构造 Kronecker 分解形式的 Fisher 权重，赋予不同低秩方向不同的重要性，优先保留对模型损失最敏感的方向。第二阶段使用剩余的秩预算，通过带正则化约束的低秩岭回归拟合压缩模型输出与全精度模型输出之间的残差，同时补偿当前层的 SVD 截断误差和上游输入偏移。作者给出了统计量固定时的闭式最优解，并在六个模型上进行了评估，报告了较好的实验表现，以及实际模型权重大小和推理加速结果。

**优点**

1. 研究动机相当清楚，两个问题都有相对精确的数学定义。图 1 还显示，与 SAES-SVD 相比，FR-SVD 减轻了中间层的输入偏移，为研究动机提供了实验支持。
2. 两个阶段的分工清晰，设计自然。
3. 实验较为充分，覆盖了多个模型和多个基准。

**缺点**

1. 核心方法的创新性有限。该方法主要组合了现有的压缩技术，论文的贡献更偏向整合和改进现有的 SVD 压缩机制。
2. 缺少若干关键基线，例如 GFWSVD 和 AA-SVD，它们分别对应 FR-SVD 两个阶段的核心技术思路。

**评论、建议与勘误**

1. 建议直接验证 Fisher 引导的截断能否保留更多对损失敏感的奇异方向。具体来说，可以在相同的秩预算下，对比 Fisher 引导的 SVD 和激活感知 SVD 各自丢弃的奇异方向对校准损失的敏感性。这将更直接地说明 Fisher 加权能否更好地选择与任务相关的方向，以及观察到的差异是否仅来自近似度量的改变。
2. 高压缩率下的结果只在一个模型上报告。能否也给出其余三个模型的结果，以便更好地观察该方法在不同压缩率下的表现？
3. 能否更详细地解释为什么无法与 GFWSVD 和 AA-SVD 等关键方法进行比较？
