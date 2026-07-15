# SAES-SVD: SELF-ADAPTIVE SUPPRESSION OF AC-CUMULATED AND LOCAL ERRORS FOR SVD-BASED LLM COMPRESSION

**Anonymous authors**

Paper under double-blind review

## ABSTRACT

The rapid growth in the parameter scale of large language models (LLMs) has cre-ated a high demand for efficient compression techniques. As a hardware-agnostic and highly compatible technique, low-rank compression has been widely adopted. However,existing methods typically compress each layer independently by mini-mizing per-layer reconstruction error, overlooking a critical limitation: the recon-struction error propagates and accumulates through the network, which leads to amplified global deviations from the full-precision baseline. To address this,we propose Self-Adaptive Error Suppression SVD (SAES-SVD), a LLMs compres-sion framework that jointly optimizes intra-layer reconstruction and inter-layer error compensation. SAES-SVD is composed of two novel components: ❶ Cu-mulative Error-Aware Layer Compression (CEALC), which formulates the com-pression objective as a combination of local reconstruction and weighted cumula-tive error compensation. Based on it, we derive a closed-form low-rank solution relied on second-order activation statistics, which explicitly aligns each layer's output with its full-precision counterpart to compensate for accumulated errors. **❷** Adaptive Collaborative Error Suppression (ACES), which automatically ad-justs the weighting coefficient to enhance the low-rank structure of the compres-sion objective in CEALC. Specifically, the coefficient is optimized to maximize the ratio between the Frobenius norm of the compressed layer's output and that of the compression objective under a fixed rank, thus ensuring that the rank budget is utilized effectively. Extensive experiments across multiple LLM architectures and tasks show that, without fine-tuning or mixed-rank strategies, SAES-SVD consistently improves post-compression performance. For example, at a 0.2 com-pression ratio on LLaMA-7B, existing methods exhibit an average accuracy drop exceeding 0.05,whereas SAES-SVD restricts the drop to only 0.02. These im-provements underscore the potential of SAES-SVD to effectively narrow the gap between compressed models and their full-precision counterparts, paving the way for more reliable compression of LLMs.

## 1 INTRODUCTION

Large language models (LLMs), including GPT Family (Achiam et al., 2023;Dettmers et al., 2022), LLaMA Family (Touvron et al., 2023a;b; Dubey et al., 2024), and Qwen Family (Bai et al., 2023; Team, 2024; Yang et al., 2025), have achieved state-of-the-art performance in tasks such as com-monsense reasoning, document summarization, and few-shot learning. However, these advances come at a substantial cost: LLMs with billions of parameters demand enormous computational and memory resources, rendering them impractical for deployment on edge devices. To address this challenge, model compression has become a key research direction, with approaches such as quan-tization (Frantar et al., 2022; Hu1 et al., 2025), network pruning (Ashkboos et al., 2024), knowledge distillation (Hinton et al., 2014), and low-rank decomposition (Yuan et al., 2023; Li et al., 2025). Among these, low-rank compression, typically implemented through Truncated Singular Value De-composition (Eckart & Young, 1936), directly reduces both parameter count and computational cost by factorizing weight matrices into low-rank components. This efficiency makes it particularly ap-pealing for LLMs deployment, as the method is hardware-agnostic and can be seamlessly integrated with other compression techniques.

<!-- 1 -->

<!-- Adaptive Err Suppression 0.95 운AaIM ALelEIS euIso Severe Err 0.90 Accumulation 0.85 Baseline (SVD-LLM) SAES-SVD (Ours) 0.80 0 2 4 6 8 10 12 14 16 Layer Index -->
![](https://web-api.textin.com/ocr_image/external/74199f84f437aed8.jpg)

(a)

<!-- ASVD 104 SVD-LLM AaIxeldJed PIM SAES-SVD (Ours) 102 102 0.2 0.3 0.4 0.5 0.6 0.7 0.8 Compression Ratio -->
![](https://web-api.textin.com/ocr_image/external/6d3e24286f997a2d.jpg)

(b)

<!-- max=0.9530 max=0.9530 으lea A6JeuE PeuIe2e 0.95 0.94 0.93 min=0. 9304 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 β Value -->
![](https://web-api.textin.com/ocr_image/external/4fbd1121be03c344.jpg)

<!-- 0.40 Ours: 0.405 KSeJno2y 0.35 Fixed a (Compression Ratio 0.2) 0.30 Adaptive Method (Compression Ratio 0.2) 0.0 0.2 0.4 0.6 0.8 1.0 β Value -->
![](https://web-api.textin.com/ocr_image/external/5932520723c10153.jpg)

(c) (d)

Figure 1: (a) Cumulative error analysis on LLaMA-3-8B, showing the cosine similarity of com-pressed outputs with the original reference across layers. SAES-SVD effectively mitigates error accumulation by adaptively suppressing upstream deviations. (b) Perplexity on WikiText2 under varying compression ratios. SAES-SVD consistently achieves lower perplexity than ASVD and SVD-LLM. (c) Retained Energy Ratio achieved by using different weighting cofficients for the cu-mulative error compensation term in with rank=128compression for k-proj in the 14th Layer of LLaMA-3-8B. Here, for the sake of convenience, we useβ=α/(1+α)as the horizontal axis. (d) Effect of fixed versus adaptive weighting cofficients for the cumulative error compensation on average accuracy across 7 zero-shot benchmarks.

Existing low-rank compression methods typically focus on layer-wise optimization.For example, methods such as ASVD (Yuan et al., 2023), SVD-LLM (Wang et al., 2024), and AdaSVD (Li et al., 2025) aim to reduce local reconstruction error within individual layers by introducing activation-aware scaling, applying whitening transformations, and performing adaptive iterative refinement, respectively. However, these approaches share a fundamental **limitation:** **they** **tend** **to** **minimize** **the reconstruction error for each layer independently, without considering how errors prop-agate** **and** **accumulate** **through** **the** **network.** In practice, reconstruction errors from early layers alter the input distribution of subsequent layers, causing these errors to compound and leading to an increasingly larger gap from the full-precision baseline. Applying SVD-LLM (Wang et al., 2024) to LLaMA2-7B further supports this observation. Although SVD-LLM achieves the theoretical min-imum truncation error at each layer, the similarity to full-precision outputs, as shown in Figure 1a, decreases steadily from 0.97 to 0.79 across layers. This result indicates that minimizing per-layer reconstruction error alone does not guarantee end-to-end fidelity.

Building on these observations, we propose Self-Adaptive Error Suppression SVD (SAES-SVD), a low-rank compression framework that incorporates cumulative error compensation into the opti-mization objective. The core idea of SAES-SVD is to let each layer perceive accumulated com-pression errors from previous layers andadapt its compression strategy based on its sensitivity to those errors. In this way, conventional independent layer-wise optimization is transformed into a globally coordinated error-suppression mechanism. Specifically, SAES-SVD consists of two main components**: ❶ Cumulative Error-Aware Layer Compression (CEALC):** For each layer,we minimize not only the local reconstruction error but also enforce alignmment with the correspond-ing full-precision reference. This alignment helps compensate for errors propagated from upstream compressed layers. The dual objective is formulated as a weighted Frobenius norm of reconstruc-tion and alignment errors. From this objective, we derive a closed-form solution that effectively suppresses reconstruction error and compensates for cumulative error. Moreover, it achieves this with only minimal additional cost, as the method relies solely on second-order statistics of acti-vation deviations and a few extra matrix operations compared with vanilla SVD compression. **❷Adaptive Collaborative Error Suppression (ACES):** Since each layer hasdifferent sensitivity to compression ratio and cumulative error, using a fixed weighting coefficient (i.e.,α\ellin Equation 3) for cumulative error compensation often results in suboptimal solutions. As shown in Figure 1c, varying the coefficient across layers leads to different values of the Retained Energy Ratio (RER), defined as the proportion of the squared sum of the top-k singular values to the total squared sum after SVD. RER serves as a measure of the low-rank structure of the compression objective, where a higher RER indicates stronger energy concentration in the leading singular values and thus greater compression efficiency. To address this issue, we introduce an adaptive weighting coefficient for each layer(i.e.,β\ell*in Equation 14). The coefficient is automatically tuned to maximize RER. By concentrating energy on the dominant singular values, this optimization yields compact and infor-mative low-rank representations and enables energy-aware error compensation that improves com-pression efficiency under fixed rank budgets.

<!-- 2 -->

We conducted a comprehensive evaluation of SAES-SVD across diverse datasets, model architec-tures, and representative SVD-based baselines. The evaluation included language modeling,clas-sification, and generation tasks. Results from multiple benchmarks, such as Figure 1, show that SAES-SVD effectively suppresses cumulative errors introduced during compression. Figure 1a in-dicates that cosine similarity with full-precision outputs remains consistently higher across layers. Figure 1b and Figure 1d further shows that SAES-SVD achieves much lower perplexity and higher accuracy under diverse compression ratios. The main contributions of this work are threefold:

·We identify that independent layer-wise compression leads to the amplification of cumulative errors. To address this, we propose CEALC, which moves beyond independent optimization and explicitly incorporates cross-layer error accumulation into the low-rank objective. Building on this formulation, we derive an efficient closed-form low-rank solution that depends only on second-order activation statistics.

·We introduce ACES, which use an adaptive coefficient α to control the weighting of cumulative error compensation. The value of α is adjusted to maximize the proportion of retained energy. This optimnization enhances the low-rank structure of the compression objective and ensures that, under a fixed rank budget, the retained components preserve as much information as possible.

·CEALC and ACES together form the framework SAES-SVD. Experiments demonstrate that SAES-SVD is both effective and broadly applicable. At the 20% compression ratio, without requiring complex rank allocation or additional fine-tuning, it reduces the drop in zero-shot accu-racy by 58% and lowers the perplexity gap by 52% compared with Dip-SVD. These improvements substantially narrow the gap between the compressed model and the full-precision baseline.

## 2 RELATED WORKS

**Large Language Model Compression Techniques.** The rapid growth of LLMs has spurred ex-tensive research on compression methods that reduce model size and inference cost while preserving performance. Existing approaches can be broadly categorized into four classes: pruning, quantiza-tion (Nagel et al., 2021; Frantar et al., 2022; Hu et al., 2025; Lin et al., 2023; Hu et al., 2024), knowledge distillation (Hinton et al., 2014), and low-rank approximation (Yuan et al., 2023;Wang et al., 2024). Unstructured pruning (Ashkboos et al., 2024; Zhang et al., 2023) removes individ-ual weights, but its irregular sparsity patterns often yield limited hardware benefits. Structured pruning (Ma et al., 2023; Dettmers et al., 2023) eliminates entire channels or blocks, improving ef-ficiency but risking significant accuracy loss. Quantization (Frantar et al., 2022; Lin et al., 2023; Hu et al., 2025) reduces numerical precision, offering strong memory savings but typically restricted to 2-8 bits, which limits flexibility and may degrade performance under aggressive settings. Knowl-edge distillation transfers knowledge from a large teacher model to a smaller student (Liu et al., 2024), but requires costly retraining. In contrast, low-rank approximation, especially singular value decomposition (SVD), provides a hardware-agnostic, post-training alternative that can be combined with other methods, making it a particularly attractive solution for LLM compression.

**SVD-based** **CompressionforLLMs.** SVD (Golub et al., 1987) reduces matrix sizes by truncat-ing the smallest singular values. Low-rank approximation using SVD has been widely studied as a hardware-agnostic and efficient approach for compressing LLMs. Early works mainly focused on minimizing the per-layer truncation loss. ASVD (Yuan et al., 2023) introduced a learnable scaling matrix to mitigate the loss of discarded singular values, but the improvement remained local and could not guarantee the theoreticaI minimum truncation loss across all layers. SVD-LLM (Wang et al., 2024; 2025b) addressed this limitation by incorporating a whitening matrix that normalizes input activations, thereby achieving theoretically minimal truncation error. AdaSVD (Li et al.,2025) introduces an alternating optimization scheme that iteratively updates low-rank matrices and com-pensation terms, thereby progressively reducing compression error. Several subsequent methods have incorporated importance weighting. For example, FW-SVD (Hsu et a1., 2022) and GFW-SVD (Chekalina et al., 2025) leverage global importance indicators to guide single-layer compres-sion, whereas DipSVD (Ding et al., 2025) jointly exploits both local and global importance to refine the weighting process. Although these techniques improve layer-level stability, they still rely on the assumption that layer errors are independent, leaving the problem of cumulative error unresolved.

<!-- 3 -->

## 3 PRELIMINARIES

For a weight mnatrixW∈Rm×n,its SVD is given byW=UΣV\top,whereU∈Rm×mis the left singular vector matrix,V∈R×is the right singular vector matrix, andΣ∈Rm×nis a nonnegative diagonal matrix with entries arranged in descending order (with min(m,n) effective diagonal elements). The classical low-rank approximation problem can be formulated as

minA∈R×r,B∈Rr×\|AB-W\|F2, r≤min(m,n) (1)

According to the Eckart-Young-Mirsky (Eckart & Young, 1936) theorem, the rank-r optimal ap-proximation of W is[W]r=UrΣrVr\top.whereUr∈Rm×randVr∈Rn×r contain the top-r left and right singular vectors of W. By choosing A=UrΣr1/2andB=Σr1/2Vr\top,we obtain AB=[W], thus achieving a rank r decomposition for compressing W. However, direct vanilla SVD compression ignores the statistical variations of layer input activations and often leads to sig-nificant accuracy degradation in LLM compression (Yuan et al., 2023; Wang et al., 2024). To address this issue, recent methods formulate LLM compression as an activation-aware, layer-wise indepen-dent optimization problem with the following objective:

$$\min _{A_{\ell },B_{\ell }}\left\|A_{\ell }B_{\ell }X_{\ell }-W_{\ell }X_{\ell }\right\|_{F}^{2}\tag{2}$$

here,X\elldenotes the input activations to thel-th layer, generated by the compressed upstream layers. In practice, compression in upstream layers introduces distributional shifts in these inputs. As depth increases,the gap betweenX\elland its full-precision counterpartX\ellfaccumulates, eventually causing a substantial global deviation of the model outputs from the full-precision baseline.

## 4 METHOD

<!-- Original LLMs CEALC Compressed LLMs · α FP Alignment Reconstruction ·. Layer \ell-1 ABXe-Wex 1 || ABX,-WeX,I2 Layer e-1 e x X\ellf ACES Original Weight X\ell Layer e W\ell Uopuetel ABeuS α* Whiting A Layer e-1 e $x_{N}^{f}\dot {j}$ 5 + Compensation :XN Truc SVD Layer N Δ\ell Δ B Layer N a -->
![](https://web-api.textin.com/ocr_image/external/8035c46bb2f1163c.jpg)

Figure 2: Overview of the proposed SAES-SVD framework.

We propose SAES-SVD, a unified low-rank compression framework that suppresses both per-layer's reconstruction errors and cumulative deviations accumulated from compressed upstream layers. As illustrated in Figure 2, the framework comprises two key components. The first, Cumulative Error-Aware Layer Compression (CEALC), explicitly incorporates cumulative errors compensation into the compression objective, ensuring that each layer aligns closely with its full-precision counterpart. The second, Adaptive Collaborative Error Suppression (ACES), introduces an adaptive coefficientαto balance local reconstruction fidelity with alignment to the full-precision counterpart. The optimal a is chosen to maximize the proportion of ratained energy within the truncated subspace.

<!-- 4 -->

<!-- 4.1 CUMULATIVE ERROR-AWARE LAYER COMPRESSION -->

**Layer compression objective with cumulative error awareness** Traditional SVD-based com-pression methods (Li et al., 2025; Wang et al., 2024) typically approximates the output of the current layer's weight matrix on its inputX\ellby minimizing Equation 2. However, this approach neglects cumulative errors propagated from upstream compressed layers, which accumulate with increasing depth and amplify the global deviation from the full-precision (FP) baseline. To address this issue, we introduce an additional alignment term with respect to the FP reference output, leading to the following weighted objective:

$$\arg \min _{A_{\ell },B_{\ell }}\underbrace {\left\|\left(A_{\ell }B_{\ell }-W_{\ell }\right)X_{\ell }\right\|_{F}^{2}}_{\text {intra-layerreconstructionerror}}+α_{\ell }\underbrace {\left\|A_{\ell }B_{\ell }X_{\ell }-W_{\ell }X_{\ell }^{f}\right\|_{F}^{2}}_{\text {FPreferencealignment}}\tag{3}$$

Here,α\ell≥0controls the alignment strength. LetT\ell=W\ellX\ellbe the original output under current inputs andR\ell=W\ellX\ellfunder FP inputs. Then Equation 3 is equivalent to

$$\left\|\left(A_{\ell }B_{\ell }X_{\ell }-T_{\ell }\right)\right\|_{F}^{2}+α_{\ell }\left\|\left(A_{\ell }B_{\ell }X_{\ell }-R_{\ell }\right)\right\|_{F}^{2}=\left(1+α_{\ell }\right)\left\|A_{\ell }B_{\ell }X_{\ell }-Z_{\ell }\right\|_{F}^{2}+C_{\ell }\tag{4}$$

where the constantC\ell:=\|T\ell\|F2+α\ell\|R\ell\|F2-(1+α\ell)\|Z\ell\|F2is independent ofA\ell,B\ell,and

$$Z_{\ell }=\frac {T_{\ell }+α_{\ell }R_{\ell }}{1+α_{\ell }}=W_{\ell }\frac {X_{\ell }+α_{\ell }X_{\ell }^{f}}{1+α_{\ell }}.\tag{5}$$

Therefore, Equation 3 is equivalent to

$$\min _{A_{\ell },B_{\ell }}\left\|A_{\ell }B_{\ell }X_{\ell }-Z_{\ell }\right\|_{F}^{2}\tag{6}$$

**Theorem** 4.1. As illustrated by the detailed derivation in the Appendix A.2,letH\ell=X\ellX\ell\top\succ0andL\ell=H\ell-1/2.Then,for any rank constraint r\ell,

argminrank(U\ell)≤r\ell\|U\ellX\ell-Z\ell\|F2=argminrank(V\ell)≤r\ell\|V\ell-M\ellL\ell\|F2, withV\ell:=U\ellH\ell1/2 (7)

Combined with Theorem 4.1, Equation 6 is equivalent to

$$\arg \min _{A,B}\left\|ABL^{-1}-ZX^{\top }L\right\|_{F}^{2}\tag{8}$$

**Eliminating explicit inputs via second-order statistics.** Storing raw activationsX\ell and their FP counterpartsX\ellfrequires prohibitive memory overhead. For instance, even with a moderate calibration set of 128 samples and sequence length 2048, the activation storage cost exceeds the parameter size by more than two orders of magnitude, making it infeasible for practical large-scale compression. To address this, we reformulate the objective in terms of compact second-order statis-tics. Specifically, we define the input covarianceH\ell=X\ellX\ell\topand the differential covariance Δ\ell=(X\ellf-X\ell)X\ell\top,which capture the distributional characteristics and upstream-induced shifts. By substitutingX\ellfX\ell\top=H\ell+Δ\ell,we obtain

$$Z_{\ell }X_{\ell }^{\top }=\frac {W_{\ell }\left(X_{\ell }X_{\ell }^{\top }+α_{\ell }X_{\ell }^{f}X_{\ell }^{\top }\right)}{1+α_{\ell }}=W_{\ell }\left(H_{\ell }+\beta _{\ell }\Delta _{\ell }\right)\quad \beta _{\ell }:=\frac {α_{\ell }}{1+α_{\ell }}\in [0,1)\tag{9}$$

Substituting Equation 8 into Equation 6, then the optimization objective becomes

$$\arg \min _{A_{\ell },B_{\ell }}\left\|A_{\ell }B_{\ell }H_{\ell }^{1/2}-W_{\ell }\left(H_{\ell }+\beta \Delta _{\ell }\right)H_{\ell }^{-1/2}\right\|_{F}^{2}\tag{10}$$

Hence, the target in Equation 6 depends only onH\ellandΔ\ell,which avoids explicit storage ofX\elland X\ellf. In practice, these two second-order statistics are collected in batches and layer by layer, similar to the strategy used in GPTQ, which greatly reduces space complexity.

<!-- 5 -->

**Closed-form via truncated SVD.** LettingT\ell=A\ellB\ellH\ell1/2,and defining the target matrixG\ell= W\ell(H\ell+βΔ\ell)H\ell-1/2,the Eckart-Young-Mirsky (Eckartx Young, 1936) theorem guarantees that the best rank-r approximation ofG\ellis obtained by truncated SVD:

$$T_{l}^{*}=\left[G_{\ell }\right]_{r_{\ell }}=\tilde {U}_{r_{\ell }}Σ_{r_{\ell }}\tilde {V}_{r_{\ell }}^{\top }\tag{11}$$

whereΣr\ell contains the top-r\ell singular values, and $\tilde {U}_{r_{\ell }},\tilde {V}_{r_{\ell }}$  are the corresponding left and right singular vectors. Consequently, a closed-form solution forA\ellandB\ellis given by

$$A_{\ell }=\tilde {U}_{r_{\ell }}Σ_{r_{\ell }}^{1/2}\quad B_{\ell }=Σ_{r_{\ell }}^{1/2}\tilde {V}_{r_{\ell }}^{\top }L_{\ell }\tag{12}$$

with L\ell=H\ell-1/2precomputed from input statistics. At this point, we complete the derivation of the closed-form solution of CEALC. As shown in Equation 10, the approximation target consists of two weighting terms: W\ellH\ellandβΔ\ell. The coefficientβ controls the strength of compensation for upstream cumulative errors when compressing the current layer l. After applying the whitening matrixH-1/2,a truncated SVD yields the optimal theoretical solution.

### 4.2 ADAPTIVE COLLABORATIVE ERROR SUPPRESSION

**Motivation.** The formulation in Section 4.1 assumes that the alignment coefficient $\beta _{\ell }=\frac {α_{\ell }}{1+α_{\ell }}$ is fixed in advance. In practice, however, different layers have different sensitivities to accumu-lated errors from upstream layers. A single alignment strength can disrupt the low-rank structure of the original weights. As shown in Figure 1c, varying β alters the low-rank characteristics of the compression objective in Equation 10. An inappropriate choice of β\ell scatters spectral energy into non-dominant components, which in turn reduces compression efficiency. To address this,we reinterpret Equation 10 as an approximation problem for the original weight matrixW\ell.The key motivation is to permit moderate weight deviations, but only when such deviations help

·① steers the compressed output toward the FP reference, thereby mitigating cumulative error; and

·② maximizes the spectral energy preserved in the top-r\ell singular subspace, avoiding wasteful energy leakage into discarded components.

CEALC naturally satisfies condition $\textcircled {1}$ , where the coefficient β\ellcontrols the strength of the align-ment. Thus, our goal is to determine the optimal β\ellsuch that, under a rank budget r\ell, the retained subspace preserves as much information as possible. Formally,we define

G\ell(β\ell)=S\ell+β\ellD\ell, with $\left\{\begin{matrix}S_{\ell }=W_{\ell }H_{\ell }L_{\ell },\\ D_{\ell }=W_{\ell }\Delta _{\ell }L_{\ell },\\ \beta _{\ell }=\frac {\alpha _{\ell }}{1+\alpha _{\ell }}\in [0,1].\end{matrix}\right.$  (13)

The optimization target can then be expressed as maximizing the proportion of spectral energy captured by the top-r\ell singular values ofG\ell(β\ell):

$$\beta _{\ell }^{\star }=\arg \max _{\beta _{\ell }\in [0,1]}\frac {\sum _{i=1}^{r_{\ell }}σ_{i}^{2}\left(G_{\ell }\left(\beta _{\ell }\right)\right)}{\sum _{i=1}^{\min \left(m_{\ell },n_{\ell }\right)}σ_{i}^{2}\left(G_{\ell }\left(\beta _{\ell }\right)\right)}\tag{14}$$

whereσi(·) denotes the i-thsingular value in decending order after SVD. **However, singular values** **change with** β, **and each trial value would require a new SVD computation, making direct** **optimization intractable.**

**Theorem** **4.2** (First-Order Approximation of Fixed Subspace(FS-FOA)). LetS\ell∈Rm×n with singular value decompositionS\ell=U\ellΣ\ellV\ell\top,and let\ell∈{1,⋯min(m,n)-1}be the target rank.Denote byU\ell∈Rm×\ellandVr\ell∈Rn×r\ellthe matrices formed by the top--r_{\ell}left and right singular vectorsofS\ell,respectively. Define the orthogonal projection matrices

$$P_{L}^{\ell }=I-U_{r_{\ell }}U_{r_{\ell }}^{\top }\quad P_{R}^{\ell }=I-V_{r_{\ell }}V_{r_{\ell }}^{\top }\tag{15}$$

ForG\ell(β)=S\ell+βD\ell,the relative error ratio can be approximated by

$$\widetilde {ρ}_{\ell }(\beta )=\frac {\left\|P_{L}^{\ell }G_{\ell }(\beta )P_{R}^{\ell }\right\|_{F}}{\left\|G_{\ell }(\beta )\right\|_{F}}=\frac {\left\|S_{\ell }^{\bot }+\beta D_{\ell }^{\bot }\right\|_{F}}{\left\|S_{\ell }+\beta D_{\ell }\right\|_{F}}\tag{16}$$

whereS\ell⊥=PL\ellS\ellPR\ellandD\ell⊥=PL\ellD\ellPR\ellare the projections of SS\ellandD\ellonto the orthogonal complements of the top r\ellsingular subspaces ofS\ell.Detailed proof can be referred to Appendix A.3.

<!-- 6 -->

For computational tractability, we freeze the principal subspace atβ=0,leveraging Theorem4.2. The resulting approximation ratio can be written as

$$\widetilde {ρ}(\beta )=\frac {\left\|S_{\ell }^{\bot }+\beta D_{\ell }^{\bot }\right\|_{F}^{2}}{\left\|S_{\ell }+\beta D_{\ell }\right\|_{F}^{2}}=\frac {a+2b\beta +c\beta ^{2}}{A+2B\beta +C\beta ^{2}}\tag{17}$$

where

$$a=\left\|S_{\ell \bot }\right\|_{F}^{2}\quad b=\left\langle S_{\ell \bot },D_{\ell \bot }\right\rangle\quad c=\left\|D_{\ell \bot }\right\|_{F}^{2}\quad A=\left\|S_{\ell }\right|_{F}^{2}\quad B=\left\langle S_{\ell },D_{\ell }\right\rangle\quad C=\left\|D_{\ell }\right\|_{F}^{2}\tag{18}$$

Setting the derivative of\widetildeρ(β)to zero yields a quadratic equation

$$\widetilde {p}(\beta )=(cB-bC)\beta ^{2}+(cA-aC)\beta +(bA-aB)=0\tag{19}$$

We evaluate all real roots within a feasible interval [βmin,βmax](e.g.,[0,1]), along with the end-points and the minimizerβ★of\widetildeρ(β)is then selected. This adaptive selection minimizes the fraction of truncated energy, ensuring that the retained singular spectrum captures the maximum proportion of information. As illustrated in Figure 1d, the adaptive strategy consistently outperforms fixed-αsettings across different compression ratios, highlighting both its robustness and effectiveness.

Through CEALC and ACES, we establish a closed-loop compression mechanism: each layer adap-tively selects α\ell to suppress newly introduced truncation error, while residual error is accumulated into second-order statistics and propagated forward. This allows subsequent layers to anticipate and compensate for upstreamn cumulative errors. Consequently, the model achieves output fidelity close to the baseline using only a single SVD approximation per layer. The detailed implementation can be found in Appendix A.6.

## 5 EXPERIMENTS.

**Models** **and** **Datasets.** To comprehensively evaluate the performance of SAES-SVD across di-verse models and tasks, we conducted systematic experiments on mainstream open-source LLMs and standard language understanding and generation benchmarks, covering a wide range of param-eter budgets and hardware settings. Unless otherwise specified, all results are obtained using a single-pass SVD per layer without task-specific fine-tuning after compression. For model selection, we considered representative architectures including LLaMA-1 (Touvron et al.,2023a),LLama-2 (Touvron et al., 2023b) and LLaMA-3 (LLaMA Team, AI @ Meta, 2024). Regarding evalua-tion tasks and datasets, we focused on natural language reasoning and understanding benchmarks. Specifically, we measured perplexity on the WikiText2 (Merity et al., 2016) and C4 (Raffel et al., 2020) dataset with a sequence length of 2048 to assess the language coherence and modeling ca-pacity of the compressed models. In addition, we evaluated zero-shot accuracy on multiple datasets, including ARC-Challenge (Clark et al., 2018), ARC-Easy (Clark et al., 2018), HellaSwag (Zellers et al., 2019), MathQA (Amini et al., 2019), PIQA (Bisk et al., 2020),and WinoGrande (Sakaguchi et al., 2021), to comprehensively examine the generalizability of the compressed models on various tasks.

**Baselines.** In a unified calibration dataset and evaluation protocol, we compared our method with a range of well-established SVD-based low-rank compression approaches, including ASVD (Li et al., 2025), SVD-LLM (Wang et al., 2024), FW-SVD (Hsu et al., 2022), Dobi-SVD(Wang et al.,2025a), and AdaSVD (Li et al., 2025). Notably, SVD-LLM and Dobi-SVD rely on fine-tuning, while A-SVD,AdaSVD and Dobi-SVD employ mixed-rank strategies. In contrast, our proposed SAES-SVD does not require additional training nor auxiliary techniques, yet it consistently outperforms these baselines, highlighting its effectiveness and simplicity.

**Main** **Results.** We conducted a comprehensive evaluation of SAES-SVD for low-rank compres-sion of LLMs and compared it with several SVD-based baselines across different compression ratios. Overall, SAES-SVD consistently achieved superior performance across tasks and model scales. Ta-ble 1 summarizes results on the representative LLaMA-7B. SAES-SVD reduced the perplexity gap by more than 35% compared with the strongest baseline and lowered the accuracy drop by 30-40%. At the more challenging 0.6 ratio, it halved the perplexity gap relative to Dobi-SVD and SVD-LLM, while keeping the accuracy drop below 0.03, compared with morethan 0.05 for other methods. These results lead to two key observations: (i) SAES-SVD consistently reduces perplexity by a large margin-up to a2x improvement at aggressive compression levels-and (ii) it preserves or

<!-- 7 -->

Table 1: Perplexity and zero-shot evaluation of LLaMA-7B across seven benchmark datasets under varying compression ratios. The table compares SAES-SVD with competing SVD-based methods (ASVD* (Yuan et al., 2023), FWSVD (Hsu et al., 2022),SVD-LLM+ (Wang et al., 2024), Dobi-SV*(Wang et al., 2025a), Dip-SVD*). Methods with fine-tuning are marked by †; mixed-rank strategies by *. Due to the closed-source implementation of Dip-SVD, we were unable to obtain its performance at the 00.4 compression ratio.

<table border="1" ><tr>
<td>Ratio</td>
<td> Method</td>
<td>Wiki2↓</td>
<td> PTB↓</td>
<td> C4↓</td>
<td>Openb.↑</td>
<td> ARCe↑</td>
<td> ARCc↑ </td>
<td>WinoG.↑</td>
<td>HellaS.↑</td>
<td>PIQA↑</td>
<td>MathQA↑</td>
<td>Avg.↑</td>
<td>Drop↓</td>
</tr><tr>
<td>0.0</td>
<td>Baseline</td>
<td>5.68</td>
<td>8.35</td>
<td>7.34</td>
<td>0.28</td>
<td>0.67</td>
<td>0.38</td>
<td>0.67</td>
<td>0.56</td>
<td>0.78</td>
<td>0.27</td>
<td>0.52</td>
<td>0.00</td>
</tr><tr>
<td rowspan="6">0.2</td>
<td>FWSVD</td>
<td>2e5</td>
<td>3e4</td>
<td>2e3</td>
<td>0.09</td>
<td>0.11</td>
<td>0.06</td>
<td>0.05</td>
<td>0.08</td>
<td>0.10</td>
<td>0.05</td>
<td>0.02</td>
<td>96%</td>
</tr><tr>
<td>ASVD+</td>
<td>11.14</td>
<td>16.55 </td>
<td>15.93</td>
<td>0.25</td>
<td>0.53</td>
<td>0.27</td>
<td>0.64</td>
<td>0.41</td>
<td>0.68</td>
<td>0.24</td>
<td>0.43 </td>
<td>16.7%</td>
</tr><tr>
<td>SVD-LLM\dagger</td>
<td>7.94</td>
<td> 16.22 </td>
<td>15.84</td>
<td>0.22</td>
<td>0.58</td>
<td>0.29</td>
<td>0.63</td>
<td>0.43</td>
<td>0.69</td>
<td>0.24</td>
<td>0.44 </td>
<td>14.7%</td>
</tr><tr>
<td>Dobi-SVD*+</td>
<td>8.54</td>
<td> 14.83 </td>
<td>10.01</td>
<td>0.26</td>
<td>0.59</td>
<td>0.31</td>
<td>0.66</td>
<td>0.44</td>
<td>0.70</td>
<td>0.23</td>
<td>0.46 </td>
<td>10.8%</td>
</tr><tr>
<td>Dip-SVD*</td>
<td>7.95</td>
<td>15.60 </td>
<td>14.07</td>
<td>0.27</td>
<td>0.63</td>
<td>0.33</td>
<td>0.64</td>
<td>0.45</td>
<td>0.71</td>
<td>0.24</td>
<td>0.47</td>
<td>9.2%</td>
</tr><tr>
<td>Ours</td>
<td>7.17</td>
<td>15.16 </td>
<td>13.77</td>
<td>0.29</td>
<td>0.68</td>
<td>0.36</td>
<td>0.65</td>
<td>0.45</td>
<td>0.75</td>
<td>0.25</td>
<td>0.50</td>
<td>3.9%</td>
</tr><tr>
<td rowspan="6">0.4</td>
<td>FWSVD</td>
<td>2e4</td>
<td>1e4</td>
<td>1e4</td>
<td>0.06</td>
<td>0.05</td>
<td>0.02</td>
<td>0.02</td>
<td>0.00</td>
<td>0.05</td>
<td>0.03</td>
<td>0.02 </td>
<td>96.6%</td>
</tr><tr>
<td>ASVD</td>
<td>1e3</td>
<td>3e3</td>
<td>1e3</td>
<td>0.13</td>
<td>0.28</td>
<td>0.22</td>
<td>0.48</td>
<td>0.26</td>
<td>0.55</td>
<td>0.19</td>
<td>0.30 </td>
<td>41.8%</td>
</tr><tr>
<td>SVD-LLM+</td>
<td>13.11</td>
<td>63.75 </td>
<td>49.83</td>
<td>0.19</td>
<td>0.42</td>
<td>0.25</td>
<td>0.58</td>
<td>0.33</td>
<td>0.60</td>
<td>0.21</td>
<td>0.37 </td>
<td>28.3%</td>
</tr><tr>
<td>Dobi-SVD*+</td>
<td>13.54</td>
<td> 46.38 </td>
<td>23.54</td>
<td>0.22</td>
<td>0.41</td>
<td>0.27</td>
<td>0.58</td>
<td>0.34</td>
<td>0.61</td>
<td>0.23</td>
<td>0.38 </td>
<td>26.3%</td>
</tr><tr>
<td>Dip-SVD*</td>
<td>12.76</td>
<td> 46.95 </td>
<td>34.35</td>
<td>0.22</td>
<td>0.50</td>
<td>0.30</td>
<td>0.61</td>
<td>0.36</td>
<td>0.64</td>
<td>0.22</td>
<td>0.40 </td>
<td>22.8%</td>
</tr><tr>
<td>Ours</td>
<td>10.42</td>
<td>45.13 </td>
<td>32.79</td>
<td>0.23</td>
<td>0.50</td>
<td>0.29</td>
<td>0.62</td>
<td>0.36</td>
<td>0.65</td>
<td>0.23</td>
<td>0.41</td>
<td>21.1%</td>
</tr><tr>
<td rowspan="5">0.6</td>
<td>FWSVD</td>
<td>3e4</td>
<td>2e4</td>
<td>2e4</td>
<td>0.06</td>
<td>0.01</td>
<td>0.00</td>
<td>0.00</td>
<td>0.01</td>
<td>0.01</td>
<td>0.00</td>
<td>0.01</td>
<td>97.8%</td>
</tr><tr>
<td>ASVD\dagger</td>
<td>6e4</td>
<td>4e4</td>
<td>4e5</td>
<td>0.12</td>
<td>0.26</td>
<td>0.21</td>
<td>0.49</td>
<td>0.26</td>
<td>0.53</td>
<td>0.18</td>
<td>0.29 </td>
<td>43.8%</td>
</tr><tr>
<td>SVD-LLM\dagger</td>
<td>53.74</td>
<td>4e2</td>
<td>3e2</td>
<td>0.14</td>
<td>0.28</td>
<td>0.22</td>
<td>0.50</td>
<td>0.27</td>
<td>0.55</td>
<td>0.21</td>
<td>0.31 </td>
<td>39.9%</td>
</tr><tr>
<td>Dobi-SVD*\dagger</td>
<td>46.18</td>
<td>2e2</td>
<td>2e2</td>
<td>0.15</td>
<td>0.31</td>
<td>0.20</td>
<td>0.52</td>
<td>0.28</td>
<td>0.54</td>
<td>0.22</td>
<td>0.32 </td>
<td>38.0%</td>
</tr><tr>
<td>Ours</td>
<td>22.01</td>
<td> 116.83 </td>
<td>93.97</td>
<td>0.16</td>
<td>0.33</td>
<td>0.25</td>
<td>0.52</td>
<td>0.30</td>
<td>0.54</td>
<td>0.23</td>
<td>0.34 </td>
<td>34.1%</td>
</tr></table>

Table 2: Zero-shot accuracy of LLaMA-13B under different compression ratios on seven benchmark datasets. Results are reported for SVD-LLM\dagger, Dip-SVD**, and our proposed method. denotes methods that rely on fine-tuning, while * indicates methods using mixed-rank strategies.

<table border="1" ><tr>
<td>Ratio</td>
<td> Method</td>
<td>Openb.</td>
<td>ARCe</td>
<td>WinoG.</td>
<td>HellaS.</td>
<td>ARCc</td>
<td>PIQA</td>
<td>MathQA</td>
<td>Average</td>
</tr><tr>
<td rowspan="2">0.2</td>
<td>SVD-LLM</td>
<td>0.302</td>
<td>0.683</td>
<td>0.684</td>
<td>0.470</td>
<td>0.356</td>
<td>0.725</td>
<td>0.265</td>
<td>0.498</td>
</tr><tr>
<td>Dip-SVD*</td>
<td>0.306</td>
<td>0.681</td>
<td>0.692</td>
<td>0.490</td>
<td>0.369</td>
<td>0.734</td>
<td>0.258</td>
<td>0.504</td>
</tr><tr>
<td></td>
<td>Ours</td>
<td>0.308</td>
<td>0.712</td>
<td>0.684</td>
<td>0.477</td>
<td>0.388</td>
<td>0.734</td>
<td>0.265</td>
<td>0.510</td>
</tr><tr>
<td></td>
<td>SVD-LLM\dagger</td>
<td>0.222</td>
<td>0.521</td>
<td>0.639</td>
<td>0.355</td>
<td>0.248</td>
<td>0.637</td>
<td>0.228</td>
<td>0.407</td>
</tr><tr>
<td>0.4</td>
<td>Dip-SVD*</td>
<td>0.230</td>
<td>0.548</td>
<td>0.644</td>
<td>0.402</td>
<td>0.283</td>
<td>0.661</td>
<td>0.233</td>
<td>0.429</td>
</tr><tr>
<td></td>
<td>Ours</td>
<td>0.248</td>
<td>0.543</td>
<td>0.654</td>
<td>0.470</td>
<td>0.284</td>
<td>0.631</td>
<td>0.239</td>
<td>0.438</td>
</tr></table>

even improves accuracy, while other methods suffer notable degradation. Crucially, SAES-SVD outperforms not only methods requiring fine-tuning (e.g., SVD-LLM, Dobi-SVD) but also those relying on mixed-rank allocations (e.g., AdaSVD, Dobi-SVD,Dip-SVD). These findings,together with additional experiments shown in Table 5 and Table 6 in appendix, confirm that the gains of SAES-SVD generalize across architectures and model scales, underscoring both its robustness and broadapplicability.

**Comparison** **on** **larger-scale** **models.** We further evaluated our method on larger architectures, including LLaMA-13B and LLaMA-30B, against current SOTA approaches. As shown in Table 2and Figure 3, we conducted experiments across compression ratios ranging from 0.2 to 0.4. Our method consistently outperforms all baselines in terms of both average accuracy and perplexity. Notably, on LLaMA-30B, SAES-SVD reduces perplexity to 5.49, compared with 5.63 for SVD-LLM and 6.64 for Dip-SVD, while achieving a 10% accuracy improvement over Dip-SVD. These results demonstrate that SAES-SVD remains highly effective at larger scales and delivers substantial gains in compressed model performance.

**Inference** **speed.** Low-rank approximation simultaneously decreases both computational com-plexity and parameter storage. This dual reduction substantially lowers memory footprint and com-pute requirements. We evaluated the inference speed of LLaMA3-8B under different compres-sion ratios on a single NVIDIA A6000 GPU. As illustrated in Figure 4, our proposed SAES-SVD achieves consistent acceleration over the FP16 baseline, with speedups ranging from 1.29x to 3.79x as the compression ratio increases. These results demonstrate the practicality of SAES-SVD in accelerating inference while significantly reducing memory demands.

<!-- 8 -->

<!-- 22.71 LLaMA-13B 20 LLaMA-30B → 15 AaIXelded 10 6.74 6.61 5.63 6.64 6.34 5.49 5 0 ASVD SVD-LLM Dip-SVD Ours -->
![](https://web-api.textin.com/ocr_image/external/0996e4710ed8d5c7.jpg)

<!-- 80% 3.89GB 3.79x 60% 6.85GB 2.37x CIeH uoIsselduo 40% 9.86GB 1.71x Compression Levels 20% 12.84GB 1.29x Baseline 20% Compression 40% Compression 0% 15.83GB 1.00x 60% Compression 80% Compression 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 SpeedUp -->
![](https://web-api.textin.com/ocr_image/external/c48a7fb2e8837a27.jpg)

Figure 3: Wiki-PPL for two large-scale models, Figure 4: Memory usage and inference speedup of LLaMA-3-8B on varying compression ratios.

across different methods at 20% compression.

Table 3: Ablation study on the two compo-nents of our method. Avg Acc denotes the average accuracy over PIQA, ARC-e, Hel-laSwvag and WinoGrande.

Table 4: Sensitivity analysis of the upper and lower bounds of the adaptive coefficientα in ACES, conducted on LLaMA2-7B at a 20% compression ratio.

<table border="1" ><tr>
<td>Model</td>
<td>CEALC</td>
<td>ACES</td>
<td>Wiki PPL↓</td>
<td>Avg Acc↑</td>
<td>amin</td>
<td>αmax</td>
<td>Wiki PPL↓</td>
<td>Avg Acc↑</td>
</tr><tr>
<td rowspan="3">LLama2-7B</td>
<td></td>
<td>x</td>
<td>9.34</td>
<td>58.66</td>
<td>0</td>
<td>1</td>
<td>8.96</td>
<td>58.47</td>
</tr><tr>
<td>✓</td>
<td>x</td>
<td>7.66</td>
<td>62.02</td>
<td>0.1</td>
<td>0.9</td>
<td>7.88</td>
<td>62.81</td>
</tr><tr>
<td>✓</td>
<td>✓</td>
<td>7.37</td>
<td>63.03</td>
<td>0.2</td>
<td>0.8</td>
<td>7.38</td>
<td>63.02</td>
</tr><tr>
<td rowspan="3">LLama3-8B</td>
<td></td>
<td></td>
<td>16.59</td>
<td>55.76</td>
<td>0.25</td>
<td>0.75</td>
<td>7.37</td>
<td>63.03</td>
</tr><tr>
<td>x<br>✓</td>
<td>x<br>x</td>
<td>12.25</td>
<td>58.82</td>
<td>0.3</td>
<td>0.7</td>
<td>7.36</td>
<td>63.01</td>
</tr><tr>
<td>✓</td>
<td>✓</td>
<td>11.48</td>
<td>60.18</td>
<td>0.4</td>
<td>0.6</td>
<td>7.63</td>
<td>62.87</td>
</tr></table>

**Ablation** **Experiments.** The proposed SAES-SVD framework consists of two complementary components: Cumulative Error-Aware Layer Compression (CEALC) and Adaptive Collaborative Error Suppression (ACES). CEALC addresses the limitation of prior methods that treat layer com-pression as independent processes; as shown in Figure 1a, it effectively mitigates cumulative com-pression errors. Building on this, ACES further enhances the low-rank properties of the compression objective shaped by CEALC, leading to superior performance under limited rank budgets, as illus-trated in Figure 1d. The ablation results presented in Table 3 confirm these contributions: CEALC significantly improves both generative and zero-shot tasks, while ACES provides additional perfor-mance gains on top of CEALC.

We conducted a sensitivity analysis of the adaptive coefficient α by varying its lower and upper bounds. Experiments were performed on LLaMA-2 7B at a 20% compression ratio, with results summarized in Table 4. The analysis shows that performance is relatively stable within moderate ranges of α. Specifically,when αmin=025andαmax=075,the model achieves the best trade-off,wvith an average zero-shot accuracy of 63.03%. In contrast, setting the lower bound too high(e.g.,.αmin&gt;08)slightly degrades perplexity while reducing average accuracy significantly, indicating potential overfitting calibration dataset. These results validate our choice of setting the lower bound to 0.25 and the upper bound to 0.75 in practice.

## 6 CONCLUSION

In this work, we proposed SAES-SVD, a principled framework for low-rank compression of large language models that explicitly addresses both local reconstruction error and cross-layer accu-mulated error. By introducing Cumulative Error-Aware Layer Compression (CEALC) and Adap-tive Collaborative Error Suppression (ACES), our method effectively mitigates error propagation while preserving critical low-rank structure. Extensive experiments across diverse models and tasks demonstrate that SAES-SVD consistently outperforms prior SVD-based methods in terms of per-plexity and zero-shot accuracy, even under aggressive compression ratios, and does so without re-quiring fine-tuning or mixed-rank heuristics. These findings highlight the robustness, scalability, and practicality of SAES-SVD, providing a promising direction for deploying large language mod-els efficiently in real-world scenarios.

<!-- 9 -->

### REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-man, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et a1. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,2023.

Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi,and Hannaneh Ha-jishirzi Kohlmeier. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. In Proceedings of the 57th Annual Meeting of the Association for Com-putational Linguistics (ACL), pp. 2357-2367, Florence, Italy, 2019. Association for Compu-tational Linguistics. doi: 10.18653/v1/P19-1227. URL https://aclanthology.org/ P19-1227/.

Saleh Ashkboos, Maximilian L Croci, Marcelo Gennari do Nascimento, Torsten Hoefler,and James Hensman. Slicegpt: Compress large language models by deleting rows and columns. arXiv preprint arXiv:2401.15024,2024.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng,Yang Fan,Wenbin Ge, Yu Han,Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609,2023.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical com-monsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432-7439,2020.

Viktoriia Chekalina, Daniil Moskovskiy, Daria Cherniuk, Maxim Kurkin, Andrey Kuznetsov, and Evgeny Frolov. Generalized fisher-weighted svd: Scalable kronecker-factored fisher approxima-tion for compressing large language models. arXiv preprint arXiv:2505.17974, May 2025.URL https://arxiv.org/abs/2505.17974.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick,and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457,2018.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale. Advances in Neural Information Processing Systems, 35: 30318-30332,2022.

Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashk-boos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. Spqr: A sparse-quantized repre-sentation for near-lossless llm weight compression. arXiv preprint arXiv:2306.03078,2023.

Xuan Ding, Rui Sun, Yunjian Zhang, Xiu Yan, Yueqi Zhou, Kaihao Huang, Suzhong Fu, Chuanlong Xie, and Yao Zhu. Dipsvd: Dual-importance protected svd for efficient llm compression. arXiv preprint arXiv:2506.20353, June 2025. URL https://arxiv.org/abs/2506.20353.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv-2407,2024.

Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychome-trika, 1(3):211-218,1936. doi:10.1007/BF02288367.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323,2022.

G. H. Golub, Alan Hoffman, and G. W. Stewart. A generalization of the eckart-young-mirsky matrix approximation theorem. Linear Algebra and its Applications, 88/89:317-327,1987. doi: 10.1016/0024-3795(87)90114-5.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. In Advances in Neural Information Processing Systems,2014.

<!-- 10 -->

Yen-Chang Hsu, Ting Hua, Sungen Chang, Qian Lou, Yilin Shen, and Hongxia Jin. Language model compression with weighted low-rank factorization. In International Conference on Learning Representations (ICLR) 2022 Workshop, 2022. URL https://arxiv.org/abs/2207. 00112. arXiv preprint arXiv:2207.00112.

Xing Hu,Yuan Cheng,Dawei Yang,Zhihang Yuan,Jiangyong Yu,Chen Xu, and Sifan Zhou. I-llm: Efficient integer-only inference for fully-quantized low-bit large language models. arXiv preprint arXiv:2405.17849,May 2024. URL https://arxiv.org/abs/2405.17849.

Xing Hu,Yuan Cheng,Dawei Yang,Zukang Xu, Zhihang Yuan, Jiangyong Yu, Chen Xu,Zhe Jiang, and Sifan Zhou. Ostquant: Refining large language model quantization with oorthogonal and scaling transformations for better distribution fitting. arXiv preprint arXiv:2501.13987,2025.

Zhiteng Li, Mingyuan Xia, Jingyuan Zhang, Zheng Hui, Linghe Kong, Yulun Zhang, and Xiaokang Yang. Adasvd: Adaptive singular value decomposition for large language models. arXiv preprint arXiv:2502.01403,2025.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. Awq: Activation-aware weight quantization for llm compression and acceleration. arXiv preprint arXiv:2306.00978,2023.

Zechun Liu, Barlas Oguz,Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad,Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. LLM-QAT: Data-free quantization aware training for large language models. In Findings of thte Association for Computational Linguistics: ACL 2024, pp. 467-484, Bangkok, Thailand, aug 2024. Association for Computational Linguis-tics. doi: 10.18653/v1/2024.findings-acl.26. URL https://aclanthology.org/2024. findings-ac1.26/.

LLaMA Team, AI @ Meta. The llama-3 herd of models, 2024. URL https://arxiv.org/ abs/2407.21783.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. In Advances in Neural Information Processing Systems, 2023.

Stephen Merity,Caiming Xiong,James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843,2016.

Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, and Tijmen Blankevoort. A white paper on neural network quantization. arXiv preprint arXiv:2106.08295, 2021. doi: 10.48550/arXiv.2106.08295. URL https://arxiv.org/ abs/2106.08295.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research,2020.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adver-sarial winograd schema challenge at scale. Communications of the ACM, 2021.

Qwen Team. Qwen2 technical report. arXiv preprint arXiv:2407.10671,2,2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux,Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971,2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Niko-lay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open founda-tion and fine-tuned chat models. arXiv preprint arXiv:2307.09288,2023b.

Qinsi Wang, Jinghan Ke, Masayoshi Tomizuka, Yiran Chen, Kurt Keutzer, and Chenfeng Xu. Dobi-svd: Differentiable svd for llm compression and some new perspectives. arXiv preprint arXiv:2502.02723,2025a.

<!-- 11 -->

Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. Svd-llm: Truncation-aware singular value decomposition for large language model compression. arXiv preprint arXiv:2403.07378,March 2024. URL https://arxiv.org/abs/2403.07378.

Xin Wang, Samiul Alam, Zhongwei Wan, Hui Shen, and Mi Zhang. Svd-llm v2: Optimizing sin-gular value truncation for large language model compression. arXiv preprint arXiv:2503.12340, March 2025b. URL https://arxiv.org/abs/2503.12340.

An Yang, Anfeng Li,Baosong Yang,Beichen Zhang,Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388,2025.

Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang Wu, Yan Yan, and Guangyu Sun. Asvd: Activation-aware singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821,2023.

Rowan Zellers, AriHoltzman,Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a ma-chine really finish your sentence? arXiv preprint arXiv:1905.07830,2019.

Mingyang Zhang, Chunhua Shen, Zhen Yang, Linlin Ou, Xinyi Yu, Bohan Zhuang, et al. Pruning meets low-rank parameter-efficient fine-tuning. arXiv preprint arXiv:2305.18403,2023.

<!-- 12 -->

### A APPENDIX

#### A.1 LLM DISCLAIMER

The authors hereby declare the role of large language model (LLM) tools in the preparation of this manuscript: LLMs were solely utilized to assist with text polishing (including refining sentence structure, optimizing lexical expression, and enhancing language fluency) and **writing optimiza-tion** of the paper's narrative content.

It is explicitly emphasized that all core components of this research, which determine the originality, scientific validity, and academic value of the work, were independently completed by the research team through manual efforts. These components include, but are not limited to:

·The formulation and development of the overall research framework, core ideas, and logical struc-ture of the study;

·The design, coding, debugging, and validation of all algorithms and program codes involved in the research;

·The design of experimental protocols, collection and preprocessing of experimental data, ex-ecution of experiments, analysis and interpretation of experimental results, and verification of conclusions.

The use of LLM tools did not involve any participation in the conception of research content, gen-eration of technical solutions, implementation of experimental processes, or derivation of research conclusions. All content of this paper adheres to academic integrity standards, and the research team assumes full responsibility for the scientificity, authenticity, and originality of the work.

<!-- 13 -->

#### A.2 THEORETICAL PROOF OF THEOREM 4.1 AND SUBSEQUENT TRUNCATED SVD

**Problem** **Formulation.** Consider the optimization problem

$$\min _{A,B}J(A,B)=\|ABX-Z\|_{F}^{2},\tag{20}$$

where the goal is to simplify the objective functionJ(A,B)so that optimization with respect to A and B becomes more tractable. The key idea is to exploit the structure of the matrix X,particularly its row space.

**Assumption.** To ensure that the following projection operations are well-defined, we assume that X has full row rank, because in practice a small ridge parameter λ will be introduced and L\ell= (H\ell+λI)-1/2.This assumption is reasonable since the activation mmatrix X typically has far more columns than rows. If X is ap×matrix withp«,then rank :()=p, which guarantees that \top invertible.

**Orthogonal Projection Operator.** We dlefine the projection matrix

$$P=X^{\top }\left(XX^{\top }\right)^{-1}X,\tag{21}$$

which is ann×northogonal projector mapping any row vector inRnonto the row space of X. For any matrix Ywith ncolumns, the productYP projects each row of Y onto the row space of X.

The matrix P satisfies several important properties:

**·Symmetry:**P\top=P,sinceXX\topissymmetric and so is its inverse.

**·Idempotence**:P2=P,which follows by direct calculation.

**·Projection** **property**: For any matrix of the form CX, right multiplication by P leaves it un-changed,i.e.,(CX)=CX. In particular,ABXbelongs to this class.

**Decomposition** **of** Z. We decompose Z into componentslying in the row space of X and its orthogonal complement:

$$Z_{P}=ZP,\quad Z_{\bot }=Z(I-P)\tag{22}$$

Clearly,

$$Z=ZP+Z(I-P)=Z_{P}+Z_{\bot },\tag{23}$$

whereZPis the projection ofZ onto the row space ofX,and Z⊥is the projection onto its orth1ogonal complement.Moreover,Z⊥P=0.

**Reformulated** **Objective.** Using this decomposition, we expand the objective:

$$J(A,B)=\|ABX-Z\|_{F}^{2}$$

$$=\left\|ABX-\left(Z_{P}+Z_{\bot }\right)\right\|_{F}^{2}\tag{24}$$

$$=\left\|\left(ABX-Z_{P}\right)-Z_{\bot }\right\|_{F}^{2}$$

LetU=ABX-ZPandV=Z⊥.Since U lies in the row space ofX and V lies in its orthog-onal complement, they are orthogonal,i.e.,Tr(\topV)=0.By the Pythagorean theorem for the Frobenius norm, we obtain

$$J(A,B)=\left\|ABX-Z_{P}\right\|_{F}^{2}+\left\|Z_{\bot }\right\|_{F}^{2}\tag{25}$$

By the premise, we start from

$J(A,B)=\left\|ABX-ZX^{\top }\left(XX^{\top }\right)^{-1}X\right\|_{F}^{2}=\|UX-ZP\|_{F}^{2}$  whereU=AB (26)

<!-- 14 -->

**Step 1: Projection reduction is tight up to a constant.** Decompose Z into its components on the row space of X and its orthogonal complement:

$$Z=ZP+Z(I-P)=:Z_{P}+Z_{\bot },\quad Z_{\bot }P=0.\tag{27}$$

Since UX lies in the row space of X,we have(UX)P=UX.Therefore,

$$\|UX-Z\|_{F}^{2}=\left\|UX-\left(Z_{P}+Z_{\bot }\right)\right\|_{F}^{2}=\left\|\left(UX-Z_{P}\right)-Z_{\bot }\right\|_{F}^{2}\quad =\left\|UX-Z_{P}\right\|_{F}^{2}+\left\|Z_{\bot }\right\|_{F}^{2}\tag{28}$$

Thus minimizing\|UX-ZP\|F2is equivalent to minimizing\|UX-Z\|F2,since they differ by the constant\|Z⊥\|F2independent of U. Hence equation 26 is a valid surrogate objective.

**Step 2: Whitening reformulation.** LetM=ZX\top∈Rm×p.Expand the Frobenius norm using trace identities:

$$\|UX-ZP\|_{F}^{2}=\|UX-ZP\|_{F}^{2}=\|(UX-Z)P\|_{F}^{2}=\text {tr}\left((UX-Z)P(UX-Z)^{\top }\right)\quad =\text {tr}\left(UXX^{\top }U^{\top }\right)-2\text {tr}\left(UXPZ^{\top }\right)+\text {tr}\left(ZPZ^{\top }\right)=\text {tr}\left(UHU^{\top }\right)-2\text {tr}\left(UM^{\top }\right)+\text {tr}\left(ZPZ^{\top }\right)\tag{29}$$

BecauseH\succ0,its symmetric square rootsH±1/2exist and are invertible.Using

$$\text {tr}\left(UHU^{\top }\right)=\left\|UH^{1/2}\right\|_{F}^{2}\quad \text {tr}\left(UM^{\top }\right)=\left\langle UH^{1/2}\right.\quad \left.MH^{-1/2}\right\rangle _{F}\tag{30}$$

we can complete the square:

$$\|UX-ZP\|_{F}^{2}=\left\|UH^{1/2}\right\|_{F}^{2}-2\left\langle UH^{1/2},MH^{-1/2}\right\rangle _{F}+\text {tr}\left(ZPZ^{\top }\right)\quad =\left\|UH^{1/2}-MH^{-1/2}\right\|_{F}^{2}+\left(\text {tr}\left(ZPZ^{\top }\right)-\left\|MH^{-1/2}\right\|_{F}^{2}\right)\tag{31}$$

The parenthetical term is independent of U. Therefore the minimizers of equation 26 are exactly the minimizers of

$$\min _{\text {rank}(U)\leq r}\left\|UH^{1/2}-MH^{-1/2}\right\|_{F}^{2}\tag{32}$$

At this point, we have completed the proof of Theorem 4.1. Subsequently, we will perform low-rank approximation based on this.

**Step 3: Invertible change of variables and rank preservation.** Define the change of variables

$$V=UH^{1/2}\quad L=H^{-1/2}\tag{33}$$

SinceH1/2is invertible,rank(V)=rank(U), so the rank constraint is preserved. The problem is thus equivalent to

$$\min _{\text {rank}(V)\leq r}\|V-ML\|_{F}^{2}\tag{34}$$

This proves the desired whitening equivalence: minimizing\|UX-ZP\|F2overrank-rUis equiv-alent to minimizing\|V-ML\|F2 over rank-r A, with the bijection between minimizers given by U★=V★H-1/2

<!-- 15 -->

#### A.3 THEORETICAL PROOF OF THEOREM 4.2: FIXED-SUBSPACE FIRST-ORDER APPROXIMATION FOR TRUNCATION RATIOS

**Setup.** LetS∈Rm×nwith singular value decompositionS=UΣV\top,and fix a target rank r∈{1,⋯,min(m,n)-1}.Denote

Ur:=U[:,1:r], Vr:=V[:,1:r], PL:=I-UrUr\top PR:=I-VrVr\top 

Given a directionD∈Rm×nand a scalarβ∈[0,1],define the affine path

$$G(\beta ):=S+\beta D.\tag{35}$$

Let\|·\|Fbe the Frobenius norm,||·\|2the spectral norm, and letσ1(·)≥⋯denote singular values.

**True** **tail energy** **and** **ratio.** By the Eckart-Young-Mirsky theorem,

$$E(\beta ):=\min _{\text {rank}(Q)\leq r}\|G(\beta )-Q\|_{F}^{2}=\sum _{i>r}σ_{i}(G(\beta ))^{2}\tag{36}$$

We call $\sqrt {E(\beta )}$ the tail (Frobenius) norm ofG(β) at rank r. The corresponding tail ratio is

$$ρ(\beta ):=\frac {\sqrt {E(\beta )}}{\|G(\beta )\|_{F}}\tag{37}$$

**Fixed-subspace first-order approximation (FS-FOA).** Freeze the leading rank-r subspaces at β=0(those ofS), and approximate the tail by the projection onto the fixed orthogonal complement:

\widetildeE(β):=\|PLG(β)PR\|F2=\|S⊥+βD⊥\|F2, S⊥:=PLSPR D⊥:=PLDPR. (38)

The FS-FOA tail ratio is then

$$\widetilde {ρ}(\beta ):=\frac {\left\|S_{\bot }+\beta D_{\bot }\right\|_{F}}{\|S+\beta D\|_{F}}\tag{39}$$

**Assumption (spectral gap and small perturbation).** Let the spectral gap atr be

$$δ:=σ_{r}(S)-σ_{r+1}(S)>0,\tag{40}$$

and assume the relative perturbation is small:

$τ:=\frac {\beta \|D\|_{2}}{δ}\leq τ_{0}$  for some sufficiently smallτ0∈(0,1). (41)

**Theorem (FS-FOA is first-order accurate for the tail energy).** Under equation 40-equation 41, there exist absolute constantsC1,C2&gt;such that for allβ∈[0,1]

$$E(\beta )=\left\|S_{\bot }+\beta D_{\bot }\right\|_{F}^{2}+R_{E}(\beta )\quad \left|R_{E}(\beta )\right|\leq C_{1}τ^{2}\left(\|S\|_{F}^{2}+\beta ^{2}\|D\|_{F}^{2}\right)\tag{42}$$

In particular, the discrepancy between the true tail energy and its FS-FOA surrogate is second order in T.

**Corollary (FS-FOA is first-order accurate for the tail ratio).** With the same assumptions,there existsC′&gt;0such that

$$|ρ(\beta )-\widetilde {ρ}(\beta )|\leq C^{\prime }τ^{2}=O\left(\frac {\beta ^{2}\|D\|_{2}^{2}}{δ^{2}}\right)\tag{43}$$

Hence\widetildeρ(β) is a first-order accurate approximation to the true ratioρ(β)

<!-- 16 -->

**Proof sketch of equation** 42**.** LetUr(β),Vr(β)be the leadingr -dimensional singular subspaces ofG(β) and PL(β)=I-U(β)U(β)\top,PR(β)=I-V(β)V(β)\toptheir orthogonal complements. Standard DavisKahan/Wedin perturbation bounds imply

$$\left\|P_{L}(\beta )-P_{L}\right\|_{2}\lesssim τ,\quad \left\|P_{R}(\beta )-P_{R}\right\|_{2}\lesssim τ.\tag{44}$$

BecauseE(β)=\|PL(β)G(β)PR(β)\|F2,writePL(β)=PL+EL,PR(β)=PR+ERwith \|EL/R\|2=O(τ)and expand:

$$P_{L}(\beta )G(\beta )P_{R}(\beta )=\left(P_{L}+E_{L}\right)(S+\beta D)\left(P_{R}+E_{R}\right)=P_{L}(S+\beta D)P_{R}+\mathcal {E},$$

where the Frobenius norm of the remainder satisfies\|E\|F=O(τ\|G(β)\|F)by the block structure induced by(Ur,Vr). Squaring norms yields

$$E(\beta )=\left\|P_{L}GP_{R}\right\|_{F}^{2}+O\left(τ^{2}\|G(\beta )\|_{F}^{2}\right)=\left\|S_{\bot }+\beta D_{\bot }\right\|_{F}^{2}+O\left(τ^{2}\left(\|S\|_{F}^{2}+\beta ^{2}\|D\|_{F}^{2}\right)\right)$$

which is equation 42. The corollary equation 43 follows by dividing by \|G(β)\|F2and using $\sqrt {1+x}=1+O(x)$ for smallx.

**Closed form for the FS-FOA ratio and its stationary condition.** Both numerator and denomi-nator of equation 39 are **quadratic** polynomials in β:

$$\left\|S_{\bot }+\beta D_{\bot }\right\|_{F}^{2}=a+2b\beta +c\beta ^{2}\tag{45}$$

$$\|S+\beta D\|_{F}^{2}=A+2B\beta +C\beta ^{2}\tag{46}$$

where

0 :=\|S⊥\|F2

$$b:=\left\langle S_{\bot },D_{\bot }\right\rangle$$

$$c:=\left\|D_{\bot }\right\|_{F}^{2}$$

$$A:=\|S\|_{F}^{2}$$

$$B:=\langle S,D\rangle$$

C:=\|D\|F2

(47)

Let

$$φ(\beta ):=\widetilde {ρ}(\beta )=\sqrt {\frac {a+2b\beta +c\beta ^{2}}{A+2B\beta +C\beta ^{2}}}\tag{48}$$

Since the square root is monotone, arg min Φ coincides with arg min of the squared ratio. Differen-tiating and simplifying gives the quadratic stationary equation

$$(cB-bC)\beta ^{2}+(cA-aC)\beta +(bA-aB)=0.\tag{49}$$

Selection rule. Solve equation 49, keep real roots in 0,1], add the boundaries {0,1},evaluate \widetildeρ(β) on this candidate set, and choose the minimizerβ*∈[01. In applications with a trade-off parameterα≥0,one uses the bijection

$$\beta =\frac {α}{1+α}\quad Longleftrightarrow\quad α=\frac {1}{1-\beta }-1,\tag{50}$$

so that $α^{*}=\frac {1}{\beta ^{*}}-1.$ 

**When is FS-FOA accurate.** Accuracy is controlled byτ=β\|D\|2/δ.When T&lt;1(e.g., τ\lesssim05), the error in equation 42-equation 43 is second order and the stationary points of\widetildeρ reliably approximate those of ρ. In near-degenerate cases (small spectral gap or large D), one may constrain β≤βmax&lt;1or apply a mild shrinkage to the selectedβ*; a one-step refinement that recomputes the leading subspaces atG(β*)further reduces the residual error.

**Interpretation.** The numerator in equation 39 measures the energy that would be discarded by rank-r truncation (tail) when measured in the fixed orthogonal complement of the leading subspaces of S; the denominator measures the total energy. The quadratic form equation 49 encodes the balance between tail alignment (a,b,c) and total alignment (A,B,C),yielding a closed-form,one-dimensional selection of β (hence α) that minimizes the truncation ratio to first order.

<!-- 17 -->

<!-- A.4 QUANTITATIVE RESULTS -->

Table 5: Performance of Compressed Llama-2-7B Across Benchmarks and Compression Ratios.

<table border="1" ><tr>
<td>Ratio</td>
<td>Method</td>
<td>WikiText-2</td>
<td>PTB</td>
<td>C4</td>
<td>ARCe</td>
<td>WinoG.</td>
<td>HellaS.</td>
<td>PIQA</td>
<td>Average</td>
</tr><tr>
<td>0</td>
<td>Original</td>
<td>5.68</td>
<td>8.35</td>
<td>7.34</td>
<td>74.62</td>
<td>69.22</td>
<td>76.00</td>
<td>79.11</td>
<td>68.85</td>
</tr><tr>
<td rowspan="6">0.4</td>
<td>SVD</td>
<td>39661.03</td>
<td>69493.00</td>
<td>56954.00</td>
<td>26.39</td>
<td>48.62</td>
<td>25.64</td>
<td>52.99</td>
<td>38.41</td>
</tr><tr>
<td>FWSVD</td>
<td>8060.35</td>
<td>9684.10</td>
<td>7955.21</td>
<td>26.05</td>
<td>50.20</td>
<td>25.70</td>
<td>52.39</td>
<td>38.59</td>
</tr><tr>
<td>ASVD</td>
<td>1609.32</td>
<td>7319.49</td>
<td>1271.85</td>
<td>26.81</td>
<td>49.49</td>
<td>25.83</td>
<td>53.81</td>
<td>38.99</td>
</tr><tr>
<td>SVD-LLM</td>
<td>161.11</td>
<td>719.44</td>
<td>61.95</td>
<td>36.99</td>
<td>56.04</td>
<td>30.49</td>
<td>56.96</td>
<td>45.12</td>
</tr><tr>
<td>AdaSVD</td>
<td>14.76</td>
<td>304.62</td>
<td>56.98</td>
<td>41.12</td>
<td>58.17</td>
<td>31.75</td>
<td>58.49</td>
<td>47.38</td>
</tr><tr>
<td>Ours</td>
<td>11.35</td>
<td>217.20</td>
<td>40.57</td>
<td>43.27</td>
<td>57.77</td>
<td>32.14</td>
<td>58.92</td>
<td>48.03</td>
</tr><tr>
<td rowspan="6">0.5</td>
<td>SVD</td>
<td>53999.48</td>
<td>39207.00</td>
<td>58558.00</td>
<td>25.80</td>
<td>47.36</td>
<td>25.55</td>
<td>52.67</td>
<td>37.85</td>
</tr><tr>
<td>FWSVD</td>
<td>8173.21</td>
<td>8615.71</td>
<td>8024.67</td>
<td>25.84</td>
<td>48.70</td>
<td>25.64</td>
<td>52.83</td>
<td>38.25</td>
</tr><tr>
<td>ASVD</td>
<td>6977.57</td>
<td>15539.44</td>
<td>4785.15</td>
<td>25.13</td>
<td>49.17</td>
<td>25.48</td>
<td>52.94</td>
<td>38.18</td>
</tr><tr>
<td>SVD-LLM</td>
<td>272.19</td>
<td>1772.91</td>
<td>129.66</td>
<td>31.65</td>
<td>51.14</td>
<td>28.38</td>
<td>54.57</td>
<td>41.44</td>
</tr><tr>
<td>AdaSVD</td>
<td>25.58</td>
<td>593.14</td>
<td>113.84</td>
<td>34.18</td>
<td>54.06</td>
<td>28.88</td>
<td>55.50</td>
<td>43.16</td>
</tr><tr>
<td>Ours</td>
<td>14.02</td>
<td>95.48</td>
<td>48.94</td>
<td>35.56</td>
<td>55.80</td>
<td>29.58</td>
<td>56.58</td>
<td>44.38</td>
</tr><tr>
<td rowspan="6">0.6</td>
<td>SVD</td>
<td>65186.67</td>
<td>79164.00</td>
<td>70381.00</td>
<td>24.49</td>
<td>51.85</td>
<td>25.40</td>
<td>53.16</td>
<td>38.73</td>
</tr><tr>
<td>FWSVD</td>
<td>27213.30</td>
<td>24962.80</td>
<td>47284.87</td>
<td>25.38</td>
<td>48.46</td>
<td>25.61</td>
<td>51.96</td>
<td>37.85</td>
</tr><tr>
<td>ASVD</td>
<td>10003.57</td>
<td>15530.19</td>
<td>9983.83</td>
<td>26.68</td>
<td>48.86</td>
<td>25.76</td>
<td>51.80</td>
<td>38.28</td>
</tr><tr>
<td>SVD-LLM</td>
<td>89.90</td>
<td>2052.89</td>
<td>561.00</td>
<td>26.73</td>
<td>47.43</td>
<td>26.89</td>
<td>53.48</td>
<td>38.63</td>
</tr><tr>
<td>AdaSVD</td>
<td>50.33</td>
<td>1216.95</td>
<td>239.18</td>
<td>28.20</td>
<td>51.22</td>
<td>27.36</td>
<td>52.83</td>
<td>39.90</td>
</tr><tr>
<td>Ours</td>
<td>23.89</td>
<td>334.67</td>
<td>100.42</td>
<td>31.02</td>
<td>52.01</td>
<td>30.38</td>
<td>54.62</td>
<td>42.01</td>
</tr></table>

Table 6: PPL and accuracy of LLaMA-3-8B across compression ratios comparing SVD-LLM and Ours.

<table border="1" ><tr>
<td>Ratio</td>
<td>Method</td>
<td>WikiText-2(PPL)↓</td>
<td>Openb↑</td>
<td>ARCc↑</td>
<td>ARCe↑</td>
<td>WinoG.↑</td>
<td>HellaS.↑</td>
<td>PIQA↑</td>
<td>MathQA↑</td>
<td>Mean↑</td>
</tr><tr>
<td rowspan="2">0.2</td>
<td>SVD-LLM</td>
<td>13.87</td>
<td>0.242</td>
<td>0.278</td>
<td>0.579</td>
<td>0.648</td>
<td>0.393</td>
<td>0.664</td>
<td>0.259</td>
<td>0.438</td>
</tr><tr>
<td>Ours</td>
<td>11.49</td>
<td>0.252</td>
<td>0.284</td>
<td>0.593</td>
<td>0.658</td>
<td>0.393</td>
<td>0.671</td>
<td>0.268</td>
<td>0.446</td>
</tr><tr>
<td rowspan="2">0.4</td>
<td>SVD-LLM</td>
<td>80.46</td>
<td>0.134</td>
<td>0.193</td>
<td>0.325</td>
<td>0.523</td>
<td>0.274</td>
<td>0.548</td>
<td>0.208</td>
<td>0.315</td>
</tr><tr>
<td>Ours</td>
<td>23.30</td>
<td>0.162</td>
<td>0.196</td>
<td>0.340</td>
<td>0.554</td>
<td>0.296</td>
<td>0.552</td>
<td>0.218</td>
<td>0.331</td>
</tr><tr>
<td rowspan="2">0.6</td>
<td>SVD-LLM</td>
<td>729.46</td>
<td>0.104</td>
<td>0.207</td>
<td>0.272</td>
<td>0.521</td>
<td>0.264</td>
<td>0.529</td>
<td>0.211</td>
<td>0.301</td>
</tr><tr>
<td>Ours</td>
<td>63.09</td>
<td>0.136</td>
<td>0.219</td>
<td>0.285</td>
<td>0.534</td>
<td>0.278</td>
<td>0.535</td>
<td>0.225</td>
<td>0.316</td>
</tr><tr>
<td rowspan="2">0.8</td>
<td>SVD-LLM</td>
<td>6971.65</td>
<td>0.120</td>
<td>0.207</td>
<td>0.259</td>
<td>0.503</td>
<td>0.258</td>
<td>0.531</td>
<td>0.201</td>
<td>0.297</td>
</tr><tr>
<td>Ours</td>
<td>181.84</td>
<td>0.140</td>
<td>0.203</td>
<td>0.281</td>
<td>0.520</td>
<td>0.259</td>
<td>0.539</td>
<td>0.208</td>
<td>0.307</td>
</tr></table>

We have supplemented detailed experimental results on different compression ratios for LLama-2and LLama-3. As shown in Table 5 and Table 6, our method consistently achieves superior results across different compression ratios. Especially under extreme compression ratio conditions, such as Llama3-8B at a compression ratio of 0.8, our method outperforms the SVD-LLM method by nearly two orders of magnitude on the Perplexity metric.

<!-- 18 -->

#### A.5 VISUALIZATION OF SELF-ADAPTATION BETA

To clarify the impact of different beta coefficients on the low-rank property of theoptimization ob-jective in Section 4.1, we visualized the compression of the k-proj Linear layer of the 14th layer of LLama3-8B. As shown in Figure 5, beta affects the low-rank property of the optimization objective. The difference between the most suitable beta and the worst beta within the range of [0,1] is signifi-cant.When be=0.95, as shown in the first subofigure, the proportion of discarded spectral energy is7%, while when beta=0.4, only 4.7% of the spectral energy is lost.

<!-- 0.955 max=0.9530 ope 0.950 S 0.945 Jeu 0.940 PeuIaeN 0.935 0.930 min=0.9293 0.925 0.00.10.20.30.40.50.60.70.80.9 β Value -->
![](https://web-api.textin.com/ocr_image/external/6cdcfa913366a2ba.jpg)

<!-- 0.944 0.942 oTe 0.940 max=0.9392 max=0.9392 R6Jau 0.938 0.936 Peulee 0.934 min=0.9335 min=0.9335 0.932 0.930 0.00.10.20.30.40.50.60.70.80.9 β Value -->
![](https://web-api.textin.com/ocr_image/external/835a810ea9558fdd.jpg)

<!-- 0.856 0.854 ope 0.852 max=0.8515 R6Jaua 0.850 Peuieae min=0.8494 0.848 0.846 0.00.10.20.30.40.50.60.70.80.9 β Value -->
![](https://web-api.textin.com/ocr_image/external/fc2a5bdb976e4302.jpg)

Figure 5: Using different beta coefficients for the k-proj of LLama3-8B at layer 14, layer 3 and layer 20 after the SVD in Equation 11, the proportion of retained energy at rank-1228.

We visualized in Figure 6 the Self-Adaptation beta coefficients of the odd k-proj layers of the LLama3-8B model under the condition of a compression ratio of 0.4. As can be observed from the figure, in the shallow layers of the network, beta tends to be small. As the number of layers deepens, the optimal beta continuously increases. In the last few layers, a downward trend appears.

<!-- max bound=0.792 maxbound=0.792 0.8 adaptive beta (kprojbeta) 0.6 e1eg a≥adepA 0.4 0.2 min bound=0.125 012345 6 78 9 10 11 12 13 14 15 Layer Index -->
![](https://web-api.textin.com/ocr_image/external/798de212fb1fa68d.jpg)

Figure 6: The visualization of the optimal beta values obtained from k-proj of odd-numbered layers on LLaMA2-7B.

<!-- 19 -->

#### A.6 ALGORITHM

APPENDIX A. ALGORITHMIC DETAILS

**Algorithm 1 (CollectSecondOrderStats-Streaming).** This routine gathers the per-layer second-order statistics required by SAES-SVD without storing raw activations. For each layerl, it maintains a running estimate of the covarianceH\ell=XX\topand the cross-residual termΔ\ell=(Xf-X)X\top The update is fully streaming: existing statistics are reweighted by a factorγ=n\ell/(n\ell+m)and the current mini-batch is scaled by $\sqrt {2/\left(n_{\ell }+m\right)}$ before accumulation, matching the numerics used in our implementation. Shapes are normalized to(dinN)to keep the math consistent across layers. This design avoids cachingX\elloX\ellfin memory, reduces I/O, and remains robust under variable batch sizes.

**Algorithm 2 (SAES-SVD for a single layer).** Given layer weightsW\elland the statistics(H\ell,Δ\ell) we form the whitenerL\ell=(H\ell+λI)-1/and the whitened objective matrixG\ell(β)=W\ell(H\ell+. .βΔ\ell)L\ell.If no alignment strength is provided,β is selected by Algorithm 3; otherwise we use the given a viaβ=α/(1+α). A single rank r truncated SVD ofG\ell(β)yields factors $A_{\ell }=\tilde {U}_{\ell }Σ_{\ell }^{1/2}$ and $B_{\ell }=Σ_{\ell }^{1/2}\tilde {V}_{\ell }^{\top }L_{\ell }$ ,which reconstruct a low-rank approximation ofW\elltailored to the cumulative-error-aware objective. The routine uses only one SVD per layer and a Cholesky-based whitener, making it both stable and efficient.

**Algorithm** **3** **(ACESBetaSelect).** This procedure selects the adaptive alignment coefficient βwithout repeated SVDs. We compute a singleSVDofS=WHL to obtain the rank-r princi-pal subspace,project(S,D) onto the orthogonal complement to get(S⊥,D⊥),and then optimize a first-order (fixed-subspace) surrogate. Two objectives are supported: (i) the ratio objective,which minimizes the approximate tail/total energy $\widetilde {ρ}(\beta =\frac {a+2b\beta +c\beta ^{2}}{A+2B\beta +C\beta ^{2}}$ ;and (ii) the energy objective, which minimizes the tail energya+2bβ+cβ2. Candidateβ values are obtained in closed form (sta-tionary roots and interval endpoints), then filtered through guardrails: interval clipping[βminβmax] a capβcp&lt;1to prevent over-alignment, and an optional shrink factorρ∈(01]for added stability. The final choiceβ★is mapped backtoα★=β★/(1-β★)and passed to Algorithm 2.

**Complexity** **and** **implementation** **notes.** All three algorithms rely only on matrix multiplications, one Cholesky per layer for whitening, and one rank-r truncated SVD per layer (randomized or Lanczos). No iterative backpropagation or fine-tuning is required. In practice, choosing a small ridge λ ensures numerical stability when H\ellis ill-conditioned; if Cholesky fails, increase λ. The ratio objective in Algorithm 3 is recommended when cross-layer robustness is prioritized; the energy objective is a conservative alternative that further suppresses absolute tail energy.

<!-- 20 -->

##### Algorithm 1: CollectSecondOrderStats-Streaming(per-layer)

Input: Calibration dataset D; layer set{l}; routines that expose per-layer inputsX\elland FP referencesX\ellf

**Output**: Per-layer second-order statsH\ell∈Rdin×din Δ\ell∈Rdin×din

**1 foreach** layer l**do**

2 $H_{\ell }\leftarrow 0,\Delta _{\ell }\leftarrow 0,n_{\ell }\leftarrow 0$ 

// running matrices and sample counter

3 **foreach** mini-batchB⊂D**do**

// Forward once to cache both compressed-path inputs X\ell and

45

FP references X\ellf

run forward; collect{X\ell,X\ellf}\ell

**foreach** layer l **do**

//Flatten to 2D and transpose to (din,N)

6

ifim(X\ell)=**then**

7

$$X\leftarrow X_{\ell }^{\top }$$

8

$$\text {elseif}\text {ndim}\left(X_{\ell }\right)=3\text {then}$$

9

$$X\leftarrow \text {reshape}\left(X_{\ell },-1,d_{\text {in}}\right)^{\top }$$

10

**else**

11

$$\text {L}X\leftarrow \text {flatten_{to}}2\mathrm {D}\left(X_{\ell }\right)^{\top }$$

12

Do the same forX\ellfto getXfwith shape(din,N)

// Streaming reweighting (match code: scale old stats,

add scaled current batch)

13

14

$$t\leftarrow n_{\ell },\quad m\leftarrow N,\quad γ\leftarrow \frac {}{+m}$$

$$H_{\ell }\leftarrow γH_{\ell },\quad \Delta _{\ell }\leftarrow γ\Delta _{\ell }\quad n_{\ell }\leftarrow t+m$$

15

∥ Batch scaling by $\sqrt {2/n_{\ell }}$  (as in code)

$$s\leftarrow \sqrt {\frac {2}{n_{\ell }}}$$

16

\widetildeX←s·X, \widetildeXf←s·Xf 

// Update H\ell=XX\top and Δ\ell=(Xf-X)X\top in the same metric

17

$$H_{\ell }\leftarrow H_{\ell }+\widetilde {X}\widetilde {X}^{\top }$$

18

$$dX\leftarrow \widetilde {X}^{f}-\widetilde {X}$$

19

$$\Delta _{\ell }\leftarrow \Delta _{\ell }+dX\widetilde {X}^{\top }$$

**20 return**{(H\ell,Δ\ell)}\ell

##### Algorithm 2: SAES-SVD for layer l

**Input:** WeightsW\ell∈Rdout×din;second-order statisticsH\ell,Δ\ell target rank r; ridge λ≥0;(optional) a orβ

**Output:**A\ell∈Rdout×r 1L\ell←(H\ell+λI)-1/2 B\ell∈Rrxdins.t.W\ell≈A\ellB\ell

// Whitening Matrix

2 if a is given and β is not **then**

3bigqcupα/(1+α)

**4** if β is not given **then**

**5** Lβ←ACESBeaSelec(W\ell,H\ell,Δ\ell,r,λ)

∥ Adaptive β

**6**G\ell←W\ell(H\ell+βΔ\ell)L\ell

// Whitened objective matrix

7 $\left[\tilde {U}_{\ell },Σ_{\ell },\tilde {V}_{\ell }\right]\leftarrow \text {TruncatedSVD}\left(G_{\ell },r\right)$ 

// Trucated SVD Decomposition

8 $A_{\ell }\leftarrow \tilde {U}_{\ell }Σ_{\ell }^{1/2}$ 

9 $B_{\ell }\leftarrow Σ_{\ell }^{1/2}\tilde {V}_{\ell }^{\top }L_{\ell }$ 

**10 return**(A\ell,B\ell)

<!-- 21 -->

##### Algorithm 3: ACESBetaSelect (single-SVD, closed-form)

**Input:**W∈Rm×n

$$H=XX^{\top },$$

$$\Delta =\left(X^{f}-X\right)X^{\top },$$

target rank r,dampingλ&gt;0,interval[βmin,..βmax],capβcap&lt;1,shrinkρ∈(0,1],objective ∈{raio,energy}

**Output:**β★∈[βmin,βmax]andα★=β★/(1-β★)

**1 Whitening and decomposition:**

2L←(H+λI)-1/2(Cholesky-based)

3 S←WHL, D←WΔL 

4 [Ur,Σr,Vr]←TopRSVD(S,r)

$$\parallel G(\beta )=S+\beta D$$

∥ only one SVD per layer

5 **Projectors and FOA terms:**

6PL←I-UrUr\top, PR←I-VrVr\top

7

8 $\begin{array}{l}S_{\bot }\leftarrow P_{L}SP_{R},\quad D_{\bot }\leftarrow P_{L}DP_{R}\\ a\leftarrow \left\|S_{\bot }\right\|_{F}^{2}b\leftarrow \left\langle S_{\bot },D_{\bot }\right\rangle ,c\leftarrow \left\|D_{\bot }\right\|_{F}^{2}\end{array}$ 

**9** A←\|S\|F2B←〈S,D〉, ,C←\|D\|F2

**10 Candidate generation (FOA):**

11 if ob jective= ratio **then**

12

$$p(\beta )\leftarrow (cB-C)\beta ^{2}+(cA-aC)\beta +(A-aB)\quad \widetilde {ρ}(\beta )\quad \begin{array}{c}o(\beta )\\ \mathcal {C}\leftarrow \{\text {realrootsof}p(\beta )\}\cup \left\{\beta _{\min },\beta _{\max }\right\}\end{array}$$

// stationary points of

13

14

$$\beta _{0}\leftarrow \text {clip}\left(-b/c,\beta _{\min },\beta _{\max }\right)$$

//minimizes tail energy a+2bβ+cβ2

15

$$\mathcal {C}\leftarrow \left\{\beta _{0}\right\}$$

16 **Selection with guardrails:**

**17 foreach**β∈Cdo

**18**

$$\beta \leftarrow \min \left\{\max \left\{\beta ,\beta _{\min }\right\},\beta _{\max }\right\}$$

19

$$\beta \leftarrow ρ·\min \left\{\beta ,\beta _{\text {cap}}\right\}$$

// shrink & cap for stability

20

**if**objective=rati**hen**

21

$$\text {score}(\beta )\leftarrow \frac {a+2b\beta +c\beta ^{2}}{A+2B\beta +C\beta ^{2}}$$

// approx. tail/total ratio \widetildeρ(β)

22

**else**

23

score(β)←a+2bβ+cβ2

// tail energy

$$24ifobjective=ratiothen$$

25 $bigsqcup_{\beta }^{\star }\leftarrow \arg \min _{\beta \in \mathcal {C}}\text {score}(\beta )$ 

26 else

27 bigsqcupβ★←argminβ∈Cscore(β)

**28 Return:**β★andα★=β★/(1-β★)

<!-- 22 -->

