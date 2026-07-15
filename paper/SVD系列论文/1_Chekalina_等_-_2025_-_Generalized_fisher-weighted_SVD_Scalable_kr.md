# Generalized Fisher-Weighted SVD: Scalable

# Kronecker-Factored Fisher Approximation for

# Compressing Large Language Models

**Viktoriia Chekalina**

AIRI

Chekalina@airi.net

**Daria Cherniuk**

**Maxim Kurkin**

AIRI

AIRI,Skoltech

**Daniil Moskovskiy**

AIRI,Skoltech

Daniil.Moskovskiy@skoltech.ru

**Andrey Kuznetsov**

**Evgeny Frolov**

AIRI

AIRI,HSE University

## Abstract

The Fisher information is a fundamental concept for characterizing the sensitivity of parameters in neural networks. However, leveraging the full observed Fisher information is too expensive for large models, so most methods rely on simple diag-onal approximations. While efficient, this approach ignores parameter correlations, often resulting in reduced performance on downstream tasks. In this work, we miti-gate these limitations and propose Generalized Fisher-Weighted SVD (GFWSVD) -a post-training LLM compression technique that accounts for both diagonal and off-diagonal elements of the Fisher information matrix, providing a more accurate reflection of parameter importance. To make the method tractable, we introduce a scalable adaptation of the Kronecker-factored approximation algorithm for the observed Fisher information. We demonstrate the effectiveness of our method on LLM compression, showing improvements over existing compression baselines.

## 1 Introduction

The Fisher information Fisher Information Matrix (FIM) Fisher [1992] is widely employed in neural networks to enhance the efficiency of models, particularly in the context of training and inference. However, computing and leveraging the full Fisher information is computationally prohibitive for deep networks. To make the problem tractable, existing methods adopt simplified1 approximations-most commonly, assuming that the Fisher matrix is diagonal Wu et al. [2024], Frankle and Carbin [2019], Soen and Sun [2024]. While efficient, this assumption discards valuable information about parameter correlations.

One key application of FIM is low-rank compression of large language models (LLMs). However,the standard low-rank approach-Singular Value Decomposition (SVD)-often leads to suboptimal performance. To mitigate this, weighted SVD methods aim to align the optimization objective with the target task Yuan et al. [2023], Hsu et al. [2022]. Fisher-Weighted SVD (FWSVD) Hsu et al. [2022] uses Fisher information to assign importance to parameters. However, FWSVD utilizes only the diagonal part of FIM and treats each row as independent, which can lead to poor retention of task-critical components.

In contrast, we propose a more accurate weighted SVD method: **Generalized** **Fisher-Weighted** **SVD** **(GFWSVD).Our** approach leverages a Kronecker factorization of the full FIM to derive two sensitivity matrices, which are integrated into a generalized SVD framework. To overcome the high computational cost of factorizing the full Fisher matrix, we introduce a scalable adaptation of the Kronecker decomposition algorithm. We reduce the core operations of the standard algorithm to matrix multiplications shaped by the dimensions of the network's weight matrices. Thus, our method lowers the computational complexity from quartic to cubic in the weight matrix size-while preserving the full Fisher information, including off-diagonal elements. We compare our method with various low-rank compression approaches for large models-those using Fisher information (Fisher-Weighted SVD), and those leveraging activation statistics (ASVD Yuan et al. [2023],SVD-LLM Wang et al. [2025b])-and observe consistent improvements in downstream task performance.

<!-- SZ07 AeIA SzI0T.so] IAtL6LT.SOsZ:AIXTe -->

To summarize, our main contributions are as follows:

·We introduce **Generalized Fisher-Weighted SVD (GFWSVD),** **a** new weighted SVD-based method for compressing large language models, which leverages the Kronecker-decomposed Fisher information that encodes both row-wise and column-wise parameter correlations. We prove that **GFWSVD** is a generalization of FWSVD Hsu et al. [2022].

·We propose a computationally effective adaptation of the Kronecker decomposition algo-rithm for the Fisher information matrix (FIM) that captures its full structure without relying on diagonal or other simplifying approximations.

·We empirically show that our method preserves model performance under compression while maintaining computational efficiency, outperforming existing low-rank factorization techniques on most of the considered tasks.

## 2 Related Work

Fisher information is a fundamental tool for estimating parameter importance in neural networks. It has been used in continual learning to prevent catastrophic forgetting Kirkpatrick et al. [2017],in federated learning to guide local update strategies Jhunjhunwala et al. [2024],and more recently for merging fine-tuned models at the parameter level Lee et al. [2025]. Due to the computational com-plexity of the FIM, many methods introduce structural assumptions to enable efficient approximations. A prominent approach is Kronecker product factorization, which decomposes FIM into tractable com-ponents. KFAC Grosse and Martens [2016] pioneered this idea for convolutional layers,showing that structural constraints can preserve key curvature information while reducing cost. Later work Tang et al. [2021] improved training efficiency through faster Kronecker-factored approximations,while KPSVD Koroko et al. [2023] applied singular value constraints to enable memory-efficient FIM approximations in large-scale models. These efforts primarily focus on improving optimization and training. In contrast, our work leverages Kronecker-product FIM approximation for post-training model compression, aiming to preserve task sensitivity while significantly reducing model size. These methods commonly assume layer independence, reducing the full-network FIM to a block-diagonal form and enabling per-layer analysis. We adopt the same assumption in this work, focusing on improving compression at the single-layer level.

Structural approximations have shown promise in post-training model compression.For instance, SparseGPT Frantar and Alistarh [2023] ranks weights using curvature estimates for pruning, while FWSVD Hsu et al. [2022] applies diagonal FIM approximations to guide task-aware SVD compres-sion. As we later demonstrate, FWSVD emerges as a special case of our more general framework, underscoring the flexibility of our approach. Notably, many of these methods assumme independent parameter contributions, which can limit task sensitivity. In contrast, our Kronecker-factored approxi-mation of the full observed FIM captures both row- and column-wise dependencies within weight matrices, yielding more accurate importance estimates.

Separately,task-unaware low-rank methods focus on minimizing truncation error without leveraging global structure. AdaSVD Li et al. [2025] distributes compression strength across layers via adaptive compensation,while ASVD Yuan et al. [2023], NSVD Lu et al. [2025],and SVD-LLM variants Wang et al. [2025b,a] use activation statistics to refine truncation. These methods modulate projections using layer-specific signals like activation norms, distributions, or covariances.

Although our method is task-aware and structure-driven, it is potentially compatible with these activation-based refinements. Integrating such signals, as in KFAC-like schemes, is a promising direction we leave for future work. In this paper, we focus on a clean evaluation setting to isolate and highlight the core contributions of GFWSVD.

<!-- 2 -->

We note that our approach can be potentially integrated with these improved task-unaware compres-sion methods. For example, incorporating activations is possible through mechanisms similar to those of KFAC-based approaches. We leave such extensions beyond the current scope and focus on the performance analysis in a purer setting to better contrast the intrinsic properties of our approach.

## 3 Background and Problem Formulation

In this section, we establish the connection between Fisher information over matrix variables drawn from Matrix-Variate Normal (MVN) distribution and our approach to approximating the Fisher matrix via a Kronecker product decomposition. We then leverage this decomposition to develop an improved compression algorithm based on the generalized SVD formulation.

### 3.1 Layer Compression and Hessian Approximation

Consider post-training weight compression as a perturbation of a model parametersθ∈Rd.The perturbation affects the deviation of the model's loss functionL(θ) in the proximity of an optimal pointθ★. Sensitivity to such perturbation can be naturally captured by the second-order expansion of the loss determined by the quadratic term involving the HessianH=H(θ★)of the problem:

$$\nabla \mathcal {L}=\mathcal {L}(\theta )-\mathcal {L}\left(\theta ^{*}\right)\approx \frac {1}{2}\left(\theta -\theta ^{\star }\right)^{\top }H\left(\theta -\theta ^{\star }\right)\tag{1}$$

Compression optimization thus corresponds to minimizing the deviation∇Lwith respect to a compressionθ=C(θ★)while considering the structured curvature encoded in H:

$$\min _{\mathcal {C}}\left(\theta ^{\star }-\mathcal {C}\left(\theta ^{\star }\right)\right)^{\top }H\left(\theta ^{\star }-\mathcal {C}\left(\theta ^{\star }\right)\right),\tag{2}$$

where the optimization task in considered over a functional fammily of compression methods C.

In real-world settings, working directly with H is often intractable due to its size and complex structure. Hence, solving the task in Eq. 2 also requires finding good enough approximations of H that ideally capture the most important properties of the Hessian. As we show next, there is a certain class of approximations that align particularly well with our task.

### 3.2 Matrix-Variate Normal Distribution and Fisher Information

The Matrix-Variate Normal (MVN) distribution Gupta and Nagar [2018] extends the classical multivariate normal distribution to matrix-valued random variables, providing a structured approach to modeling dependencies within rows and columns. Formally, a matrixX∈R× follows an MVN distribution if its entries exhibit Gaussian properties with covariance structured across both dimensions. The distribution is defined as

$$\mathbf {X}\;\mathcal {M}\mathcal {N}\left(\mathbf {M},\boldsymbol {Σ}_{1},\boldsymbol {Σ}_{2}\right)\tag{3}$$

where M is the mean matrix, and the (non-degenerate) covariance is expressed as a Kronecker productΣ2⊗Σ1.Here,Σ1captures dependencies between rows, whileΣ2encodes dependencies across columns. This structure ensures that each row and column follows a well-defined correlated Gaussian distribution.

A crucial property of MVN is that its likelihood function inherently incorporates the inverse Kronecker-factored covariance, leading to an efficient representation of second-order dependencies. The log-probability density function of X has the form:

$$\log (p(\mathbf {X}))\propto -\frac {1}{2}\left(\text {vec}(\mathbf {X}-\mathbf {M})^{\top }\left(\boldsymbol {Σ}_{2}\otimes \boldsymbol {Σ}_{1}\right)^{-1}\text {vec}(\mathbf {X}-\mathbf {M})\right)=\quad =-\frac {1}{2}\text {tr}\left(\boldsymbol {Σ}_{1}^{-1}(\mathbf {X}-\mathbf {M})\boldsymbol {Σ}_{2}^{-1}(\mathbf {X}-\mathbf {M})^{\top }\right)\tag{4}$$

Maximization of log-likelihood leads to mninimization of trace in Eq. 4, which yields the Generalized Least Squares Matrix Decomposition problem Allen et al. [2014]:

$$\min _{\text {rank}(\mathbf {X})\leq r}\left\|\boldsymbol {Σ}_{1}^{-\frac {1}{2}}(\mathbf {X}-\mathbf {M})\boldsymbol {Σ}_{2}^{-\frac {1}{2}}\right\|_{\mathrm {F}}^{2}\tag{5}$$

<!-- 3 -->

directly connected to the Generalized Singular Value Decomposition (GSVD) Golub and Van Loan [2013].This problem can be straightforwardly solved by means of standard SVD Abdi [2007]:

$$\mathbf {X}=\boldsymbol {Σ}_{1}^{\frac {1}{2}}\hat {\mathbf {U}}\hat {\mathbf {S}}\hat {\mathbf {V}}^{\top }\boldsymbol {Σ}_{2}^{\frac {1}{2}}\tag{6}$$

where $\hat {\mathbf {U}}\hat {\mathbf {S}}\hat {\mathbf {V}}^{\top }=\text {SVD}_{r}\left(\boldsymbol {Σ}_{1}^{-\frac {1}{2}}\mathbf {M}\boldsymbol {Σ}_{2}^{-\frac {1}{2}}\right)$ .We note that the result also holds in the case when matrix square roots are replaced with the corresponding Cholesky factors, which are typically easier to find.

Under regular conditions (e.g., smooth differentiability and proper statistical properties),Fisher InformationIFserves as an expectation of the local curvature (second derivative) of the likelihood function. Importantly, by taking derivatives of the MVN likelihood function with respect to M, it is easy to show that the corresponding Hessian directly coincides with Fisher Information at the MLE solution,e.i.IF=H(M)=Σ2-1⊗Σ1-1. This formulation provides a natural bridge between the selection of an optimal compression algorithm Cfrom Eq. 2 and Fisher Informnation,which we establish next.

### 3.3 Fisher-Weighted Linear Layer Compression

Building on the established connection between MVN distributions and Fisher Information, we are now ready to formulate the rank-r linear layer compression theorem.

**Theorem 1.**LetW∈Rn×m represent some parameter weights matrix of a single-layer linear neural network. Suppose that the following conditions hold.

1.The task loss function is derived from an MLE problem.

2. The (non-degenerate) emnpirical Fisher Information has a Kronecker product structure IF=A⊗B

3.The weights **W** are drawn from the MVN distributionM(★,B-1,A-1),whereW★ is the optimal weights matrix.

Under these conditions, the best rank-r approximation that minimizes the expected increase in the loss after low-rank dlecomposition ofW★is given by:

$$\widehat {\mathbf {W}}_{r}=\mathbf {L}_{\mathbf {B}}^{-\top }\widetilde {\mathbf {W}}_{r}\mathbf {L}_{\mathbf {A}}^{-1}\tag{7}$$

whereA=LALA\topandB=LBLB\topare Cholesky factorizations,\widetildeW=LB\topW★LAis an auxiliary matrix,\widetildeWris the truncated SVD of\widetildeWof rank r.

Proof. Under the assumption that the loss function originates from MLE, the Hessian coincides with Fisher Information at the optimal point, ensuring structured sensitivity encoding. Hence, one can replace Eq. 2 with a surrogate problem

$$\min _{\mathcal {C}}\left(\theta ^{\star }-\mathcal {C}\left(\theta ^{\star }\right)\right)^{\top }\mathcal {I}_{F}\left(\theta ^{\star }-\mathcal {C}\left(\theta ^{\star }\right)\right)\tag{8}$$

forvec(W★)=θ andvec(W)=C(θ★)

SubstitutingIFwithA⊗**B**and applying Cholesky decomposition to factors A and B yields:

$$\text {vec}\left(\mathbf {W}^{\star }-\mathbf {W}\right)^{\top }\left(\mathbf {L}_{\mathbf {A}}\mathbf {L}_{\mathbf {A}}^{\top }\otimes \mathbf {L}_{\mathbf {B}}\mathbf {L}_{\mathbf {B}}^{\top }\right)\text {vec}\left(\mathbf {W}^{\star }-\mathbf {W}\right)$$

$$=\text {vec}\left(\mathbf {W}^{\star }-\mathbf {W}\right)^{\top }\left(\mathbf {L}_{\mathbf {A}}\otimes \mathbf {L}_{\mathbf {B}}\right)\left(\mathbf {L}_{\mathbf {A}}^{\top }\otimes \mathbf {L}_{\mathbf {B}}^{\top }\right)\text {vec}\left(\mathbf {W}^{\star }-\mathbf {W}\right)$$

$$=\text {vec}\left(\mathbf {L}_{\mathbf {B}}{}^{\top }\left(\mathbf {W}^{\star }-\mathbf {W}\right)\mathbf {L}_{\mathbf {A}}\right)^{\top }\text {vec}\left(\mathbf {L}_{\mathbf {B}}{}^{\top }\left(\mathbf {W}^{\star }-\mathbf {W}\right)\mathbf {L}_{\mathbf {A}}\right)$$

$$=\left\|\mathbf {L}_{\mathbf {B}}^{\top }\left(\mathbf {W}^{\star }-\mathbf {W}\right)\mathbf {L}_{\mathbf {A}}\right\|_{\mathrm {F}}^{2}\tag{9}$$

In Section 3.2, we established that the optimal solution to this problem can be obtained via the standard SVD of the auxiliary matrix \widetildeW. The final solution is found in two steps: 1) finding an optimalIrank-r solution to the auxiliary problem \widetildeWr=SVr(LB\topW★LA),and 2) recovering the optimal solution to the original problem through the inverse transformation\widehatWr=LB-\top\widetildeWrLA-1which yields the best rank r minimizer for Eq.9. Consequently, the decomposition Wrpresents an optimal compression C for Eq. 8, which in turn yields the minimal error increase in Eq. 1 for the given task defined by Eq.2. ☐

<!-- 4 -->

<!-- I I SVD x W x $\hat {Σ}_{r}$ $\hat {V}_{r}^{T}$ D x x FWSVD x W x $\hat {U}_{r}$ Generalized FWSVD (ours) LBT x W x LA -->
![](./images/df9bb1c63baf6741.jpg)

Figure 1: Generalization of the Weighted SVD frameworks. For standard SVD, the transformation matrices are identity matrices. For FWSVD, the left matrix is diagonal but not identity, and the right matrix is identity. For GFWSVD,both matrices are non-diagonal.

Linear layer factorization in this case can be computed with the following expressions:

$$\mathbf {W}_{1}=\sqrt {\hat {\mathbf {S}}_{r}}\hat {\mathbf {V}}_{r}^{\top }\mathbf {L}_{\mathbf {A}}^{-1}\in \mathbb {R}^{r\times m},\mathbf {W}_{2}=\mathbf {L}_{\mathbf {B}}^{-\top }\hat {\mathbf {U}}_{r}\sqrt {\hat {\mathbf {S}}_{r}}\in \mathbb {R}^{n\times r}\tag{10}$$

where $\hat {\mathbf {S}}_{r}$ is the diagonal matrix of the rleading singular values of the auxiliary problem.

### 3.4 Relationship to Prior Works

We show that FWSVD, presented in Hsu et al. [2022], is a special case of our generalized framework. The full justification is given in Appendix A. In FWSVD, the objective minimnizes a weighted reconstruction error using a diagonal matrix D derived from a row-wise sum of the Fisher Information. We show that this setup corresponds to a diagonal Kronecker-factored approximation of the FIM, where D arises naturally from minimizing the Kronecker approximation error. The resulting solution for the low-rank factorsW2,W1matches that of FWSVD (up to a constant), which shows that their method is a special case of our more general framework.

The connection between our generalized approach, the classical SVD and FWSVD is depicted in Figure 1. Weighted SVD approaches can be interpreted as transforming the decomposed object-here, the weight matrix-into a new space where the low-rank approximation better aligns with the target task. In this formulation, the sensitivity matrices serve as transformation matrices that reweight the importance of different directions. Under this view, vanilla SVD corresponds to using identity transformations; FWSVD applies a diagonal (but non-identity) transformation on one side while keeping the other side as identity. In contrast, our method employs full, non-diagonal transformations on both sides, capturing richer structure in the parameter space.

## 4 Kronecker Factorization Algorithm via Rank-1 SVD

Suppose that we have a linear layer of a network with a weight matrix W and defineGi∈Rn×mas a weight gradients.L(θ)|θ= on the i-thbatch,andgi=vc(Gi)∈Rn·m-isflattening version. Then,Fisher InformationIF(θ)can be defined as an empirical mean over all batches in a dataset D:

$$\mathcal {I}_{F}\left(\theta ^{\star }\right)=\mathbb {E}\left[gg^{\top }\right]=\frac {1}{|D|}\sum _{i=1}^{|D|}g_{i}g_{i}^{\top }\tag{11}$$

Kronecker product approximation is obtained by solving minimization problem:

$$\min \left\|\mathcal {I}_{F}-\mathbf {A}\otimes \mathbf {B}\right\|_{\mathrm {F}}\tag{12}$$

Kronecker product decomposition ofIFis computed from a rank-1approximation of permuted matrix $\tilde {\mathcal {I}}_{F}=\mathcal {R}\mathcal {I}_{F}\in \mathbb {R}^{m^{2}x^{2}}$ , as it is described in Loan and Pitsianis [1992].The pseudocode of the method is in Algorithm 1.

<!-- 5 -->

## Algorithm 1 Compute Kronecker Factors via Rank-1 SVD

**Require:** List of gradients{gi}i=1|D|,|D|-number of batches

1: $\mathcal {I}_{F}\leftarrow \frac {1}{|D|}\sum _{i=1}^{|D|}g_{i}g_{i}^{T}$ 

2: $\tilde {\mathcal {I}}_{F}\leftarrow \mathcal {R}\mathcal {I}_{F}$ 

3: (u,σ,vT)←Leading singular triplet

**Truncated SVD**

4: b←u·σ

$$trianglerightb=\text {vec}(\mathbf {B})$$

5: a←v

$$trianglerighta=\text {vec}(\mathbf {A})$$

6: B←reshape(b,(m,m))

7: A←reshape(a,(n,n)

8**: return**(B,A)

## 4.1 Efficient Rank-1 Computation

The primary computational bottleneck of Algorithm arises in performing SVD on the matrix $\tilde {\mathcal {I}}_{F}$ 

Standard SVD is computationally intractable for large matrices, so we employ truncated SVD using the Lanczos Lanczos [1950] method, which avoids explicit matrix construction and requires only the ability to multiply the matrix with a vector from the left or right. Even in this setting, aggregating the full second-moment gradient information across all batch samples is computationally expensive.

We can show (see Appendix B) that permutedIF for i-th batch can be defined as the Kronecker product of the corresponding gradient matrices:

$$\tilde {\mathcal {I}}_{F}=\frac {1}{|D|}\sum _{i=1}^{|D|}\mathbf {G}_{i}\otimes \mathbf {G}_{i}\tag{13}$$

then multiplication of this matrix $\tilde {\mathcal {I}}_{F}$ to a vector z from left will be:

$$\tilde {\mathcal {I}}_{F}z=\frac {1}{k}\left(\sum _{i=1}^{k}\mathbf {G}_{i}\otimes \mathbf {G}_{i}\right)z=\frac {1}{|D|}\left(\sum _{i=1}^{|D|}\mathbf {G}_{i}\otimes \mathbf {G}_{i}\right)\mathbf {Z}=z,\;\text {where}\;z=\text {vec}(\mathbf {Z}),\mathbf {Z}\in \mathbb {R}^{n\times n}.$$

(14)

Using property of the Kronecker product(K⊗L)vec(C)=vec(K\topCL) we reduce the matrix-vector multiplication to a sequence of matrx multiplicaons.

$$\tilde {\mathcal {I}}_{F}z=\frac {1}{|D|}\sum _{i=1}^{|D|}\text {vec}\left(\mathbf {G}_{i}^{\top }\mathbf {Z}\mathbf {G}_{i}\right)\tag{15}$$

The derivation for right-side multiplication is analogous (seeAppendix C). Using these operations, we can obtain an approximation of the Fisher information for layers of LLMs and boatch sizes used in practice within a reasonable time.

## 4.2 Time Complexity of the proposed Rank-1 Computation

The time complexity of computing the truncated SVD of the matrix $\tilde {\mathbf {J}}\in \mathbb {R}^{m^{2}\times n^{2}}$ consists of the cost of matrix-vector multiplications and the orthogonalization step. In the standard matrix-vector multiplication case, multiplying the Fisher matrix $\tilde {\mathbf {J}}$  by a vector has a cost ofO(m^{2}right),while orthogonalization costsO((m2+2)·2), where r is the rank of the decomposition and can typically be neglected.

However, using the structured formulation from Eq. 15, where left matrix-vector products are implemented via multiplications with matricesGi\top∈R×Z∈R×,andGi∈R×,the overall complexity is reduced toO(mn2+m2n). Applying analogous reasoning to the right matrix-vector products (see Eq. 30) one can yield the same complexity.

<!-- 6 -->

Although the proposed method exhibits cubic complex-ity with respect to the dimensions of the linear layer's weight matrix, its empirical runtime grows more slowly than that of the standard matrix-vector product,which scales quartically. In large language models, where m and n are typically on the order of 103 , this reduction yields a practically significant speedup. Table 2 reports the empirical decomposition times forIF,correspond-ing for matrices W with different sizes.

<!-- Empirical Runtime of Decomposition of Fisher matrix 43.98 s 40 (SpuoaS) 30 20 auIAUnY 12.05s 10 2.62s 2.96s 3.64s 0 12 500 6.8e4 7.5e5 2.3e6 Number of Parameters in W -->
![](./images/7974ea8825fb6420.jpg)

## 5 Numerical Experiments

Figure 2: Empirical runtime for comput-ing the Kronecker decomposition of the Fisher matrix for weight matrices of vary-ing sizes.

To validate our theoretical contributions, we conduct ex-tensive numerical experiments on several transformer ar-chitectures: the encoder-only BERT model Devlin et al. [2019] and the decoder-only LLM LLaMA 2 Touvron

et al. [2023]. Our goal is to demonstrate the practical benefits of GFWSVD in low-rank compression under fine-tuning and evaluation protocols. We conduct all experiments on a single NVIDIA A100GPU with latest CUDA drivers using Python 3.12. The code is available on GitHub:link.

<!-- GLUE 0.75 \ \ \ Full Model a6e.a2Y 0.50 SVD 0.45 ASVD 0.40 FWSVD GFWSVD (Ours) 0.35 0 100 200 300 400 500 600 FCN Layer Rank -->
![](./images/cc8acd329af9d435.jpg)

<!-- MMLU 0.45 Full Model ASVD 0.43 SVD-LLM ACeJn2Y 0.40 FWSVD GFWSVD (Ours) 0.38 A 0.35 コww 0.33 0.30 0.28 0.25 80 85 90 95 Compression Rate (%) -->
![](./images/8ac89252cae29560.jpg)

Figure 3: Macro-averaged GLUE performance of Figure 4: Average **MMLU** performance of bert-base-uncased model for different com- 11ama-2-7b-chat model for different compres-pression ranks.

sion rates.

### 5.1 Compressing the Transformer Encoder

In our experiments, we follow the "fine-tune then compress" pipeline, similar to FWSVD Hsu et al. [2022]. We begin by fine-tuning a pre-trained checkpoint¹ of the BERT-base model on a specific downstream GLUE task. Optimal fine-tuning hyperparameters (e.g., learning rate, batch size) are selected for each task using the Optuna framework Akiba et al. [2019]. During this stage, we also collect gradients to construct the FIMIFand compute its Kronecker decomposition as described in Section 4.

Using the resulting Cholesky factorsLA **and** LB,we uniformly compress the fully connected layers of BERT by factorizing them into two smaller layers, following the method detailed in Section 3.1. The chosen layer-wise ranks and the resulting overall compression rate of the model are summarized in Table 1. We reproduce the ASVD method using the original authors' code. For FWSVD,we incorporate the newly constructed FIM into the compression process.

We show average compression results in Table 2 and Figure 3, extended results are in Appendix D in Table 4. On most of the GLUE tasks and considered compression ranks, our proposed GFWSVD ap-proach consistently outperforms both FWSVD and SVD, with particularly strong gains at lower ranks. While ASVD exhibits relatively poor performance on several tasks (QQP, QNLI), it occasionally surpasses GFWSVD-notably on SST2 under aggressive compression.

'https://huggingface.co/google-bert/bert-base-uncased

<!-- 7 -->

Table 1: The correspondence between rank and entire BERT compression rate

<table border="1" ><tr>
<td>Rank</td>
<td colspan="2">C.Rate Rank</td>
<td>C.Rate</td>
</tr><tr>
<td colspan="2">1 ~40%<br>5 ~40%<br>10 ~39%<br>50 ~36%</td>
<td>100<br>250<br>500<br>600</td>
<td>~33%<br>~23%<br>~8%<br>~1%</td>
</tr></table>

Table 2: Macro-averaged GLUE performance of the bert-base-uncased for different compression ranks. Best results for each rank are in **bold.**

<table border="1" ><tr>
<td>Method/Rank</td>
<td>600 </td>
<td>500 </td>
<td>250</td>
<td> 100</td>
<td>50</td>
<td>10</td>
<td>1</td>
</tr><tr>
<td>SVD</td>
<td>0.77 </td>
<td>0.76 </td>
<td>0.65 </td>
<td>0.47 </td>
<td>0.41</td>
<td> 0.42</td>
<td> 0.37</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td> 0.75 </td>
<td>0.71</td>
<td> 0.51</td>
<td> 0.46 </td>
<td>0.45 </td>
<td>0.36</td>
<td> 0.36</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td> 0.74 </td>
<td>0.74 </td>
<td>0.68 </td>
<td>0.56 </td>
<td>0.46 </td>
<td>0.43</td>
<td> 0.38</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>0.77 </td>
<td>0.77 </td>
<td>0.75 </td>
<td>0.66 </td>
<td>0.59 </td>
<td>0.53</td>
<td> 0.51</td>
</tr></table>

### 5.2 Compressing the Transformer Decoder

We evaluate our approach on the decoder-only LLama 2 7B model² against several competitive baselines: diagonal FI-based low-rank approximation method FWSVD Hsu et al. [2022],and two activation-based methods -ASVD Yuan et al. [2023] and SVD-LLMWang et al. [2025b].Notably, ASVD and SVD-LLM both rely on activation-based weighting to gauge parameter importance, while FWSVD and ours GFWSVD derive importance scores solely from gradient information.

We measure perplexity on WikiText 2 Merity et al. [2017] and PTB Marcus et al. [1993] datasets, and 5-shot reasoning performance on the MMLU benchmark Hendrycks et al. [2021].Following prior works on low-rank approximation of LLMs Wang et al. [2025b],Yuan et al.[2023],we test several compression setups, removing from 5% to 20% of original parameters.

Following standard practice in post-training LLM compression methods Wanng et al. [2025b],Yuan et al. [2023], we use a randomly sampled set of sentences as calibration data to generate gradients for further obtaining the factor matrices. For calibration data, we choose the FineWeb dataset Penedo et al. [2024] due to its high quality and diversity, and collect gradients on a random subsample of size 1024. These gradients are then used to obtain LA and LB, as well as the data needed for FWSVD. As in LLMs, uniform layer compression can disproportionately degrade performance by over-compressing critical layers and under-utilizing redundancy in less sensitive ones, so it is essential for each method to use a compression configuration that accounts for layer sensitivity. For both ASVD and SVD-LLM, we used the corresponding code released by the authors and re-ran the necessary compression pipelines for our checkpoint with all hyperparameters set to default values. For our approach, we adopted the method of per-layer importance scores as described in the ASVD work.

Table 3 and Figure 4 shows that GFWSVD consistently outperforms both simple and strong baselines across all compression rates. In particular, at the most aggressive settings (15-20% of the original parameters), our method matches or exceeds the accuracy of activation-based methods and shows substantially lower perplexities on both WikiText-2 and PTB.

## 6 Limitations

Our method decomposes the observed Fisher information matrixIFinto a Kronecker product of two smaller matrices, Y and X (Eq. 12). While effective, this assumes exact factorization,which may not hold in practice and can limit approximation quality andtask sensitivity. In LLM experiments, we also observed cases where the estimated Kronecker factors were singular, requiring regularization (e.g.,Y←Y+αiaY)to ensure positive definiteness and numerical stability. Although this resolves instability, it introduces additional computational overhead.

2https://huggingface.co/unsloth/1lama-2-7b-chat

<!-- 8 -->

Table 3: Performance of the unsloth/1lama-2-7b-chat compressed by various methods under compression ratios from 5% to 20% on WikiText-2, PTB, and MMLU. Lower is better for perplexity (↓),higher is better for accuracy (↑).

<table border="1" ><tr>
<td>METHOD</td>
<td>WikiText-2↓</td>
<td> PTB↓ </td>
<td>Compr.</td>
<td>MMLU Avg↑</td>
<td>Humanities</td>
<td>Other↑</td>
<td>Social Sciences</td>
<td>STEM</td>
</tr><tr>
<td>Full model</td>
<td>6.94</td>
<td>25.75</td>
<td>100%</td>
<td>0.46±0.003</td>
<td>0.43±0.01\quad0.55±0.01</td>
<td></td>
<td>0.53±0.01</td>
<td>0.36±0.01</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td>7.52</td>
<td>45.25</td>
<td rowspan="4">95%</td>
<td>0.40±0.003</td>
<td>0.36±0.01\quad0.45±0.01</td>
<td></td>
<td>0.45±0.01</td>
<td>0.35±0.01</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>7.60</td>
<td>26.29</td>
<td>0.41±0.004</td>
<td>0.37±0.010.48±0.01</td>
<td></td>
<td>0.46±0.01</td>
<td>0.35±0.01</td>
</tr><tr>
<td>SVD-LLM Wang et al. [2025b]</td>
<td>8.80</td>
<td>51.28</td>
<td>0.34±0.004</td>
<td>0.31±0.01\quad0.38±0.01</td>
<td></td>
<td>0.35±0.01</td>
<td>0.31±0.01</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>7.16</td>
<td>28.55</td>
<td>0.40±0.003</td>
<td>0.38±0.01\quad0.47±0.01</td>
<td></td>
<td>0.44±0.01</td>
<td>0.33±0.01</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td>11.53</td>
<td>96.62</td>
<td rowspan="4">90%</td>
<td>0.37±0.004</td>
<td>0.34±0.01\quad0.43±0.01</td>
<td></td>
<td>0.42±0.01</td>
<td>0.33±0.01</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>8.97</td>
<td>40.12</td>
<td>0.37±0.004</td>
<td>0.33±0.01\quad0.42±0.01</td>
<td></td>
<td>0.40±0.01</td>
<td>0.33±0.01</td>
</tr><tr>
<td>SVD-LLM Wang et al. [2025b]</td>
<td>9.69</td>
<td>60.82</td>
<td>0.32±0.004</td>
<td>0.30±0.01\quad0.35±0.01</td>
<td></td>
<td>0.32±0.01</td>
<td>0.30±0.01</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>8.77</td>
<td>36.44</td>
<td>0.38±0.002</td>
<td>0.35±0.01\quad0.44±0.01</td>
<td></td>
<td>0.42±0.01</td>
<td>0.33±0.01</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td>22.06</td>
<td>411.50</td>
<td rowspan="4">85%</td>
<td>0.31±0.009</td>
<td>0.29±0.01\quad0.34±0.01</td>
<td></td>
<td>0.33±0.01</td>
<td>0.30±0.01</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>10.91</td>
<td>83.49</td>
<td>0.32±0.003</td>
<td>0.30±0.01\quad0.33±0.01</td>
<td></td>
<td>0.32±0.01</td>
<td>0.30±0.01</td>
</tr><tr>
<td>SVD-LLM Wang et al. [2025b]</td>
<td>10.36</td>
<td>72.58</td>
<td>0.30±0.004</td>
<td>0.29±0.01\quad0.34±0.01</td>
<td></td>
<td>0.31±0.01</td>
<td>0.30±0.01</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>10.06</td>
<td>42.19</td>
<td>0.36±0.004</td>
<td>0.33±0.01\quad0.41±0.01</td>
<td></td>
<td>0.38±0.01</td>
<td>0.32±0.01</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td>66.37</td>
<td>1523.00</td>
<td rowspan="4">80%</td>
<td>0.27±0.004</td>
<td>0.25±0.01\quad0.30±0.01</td>
<td></td>
<td>0.28±0.01</td>
<td>0.28±0.01</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>27.73</td>
<td>241.57</td>
<td>0.26±0.004</td>
<td>0.25±0.01\quad0.27±0.01</td>
<td></td>
<td>0.24±0.01</td>
<td>0.28±0.01</td>
</tr><tr>
<td>SVD-LLM Wang et al. [2025b]</td>
<td>11.23</td>
<td>98.91</td>
<td>0.29±0.004</td>
<td>0.27±0.01\quad0.32±0.01</td>
<td></td>
<td>0.29±0.01</td>
<td>0.29±0.01</td>
</tr><tr>
<td>GFWSVD(Ours)</td>
<td>11.13</td>
<td>50.50</td>
<td>0.32±0.003</td>
<td>0.30±0.01\quad0.35±0.01</td>
<td></td>
<td>0.34±0.01</td>
<td>0.30±0.01</td>
</tr></table>

We observed that compression effectiveness varies significantly across layers, making preliminary layer selection necessary to achieve favorable trade-offs. A key limitation of our current approach is the lack of coordination across layers during compression. For effective multi - layer com-pression-especially in large-scale models like LLMs - it is important to account for cross-layer dependencies. Future work could focus on modeling these interactions to enable joint comapression strategies.

## 7 Broader impacts

Our method, GFWSVD, enables more efficient compression of large language models, which can significantly reduce computational costs and energy consumnption. This may help democratize access to powerful NLP tools and promote more sustainable AI deployment. However, as with any compression method, there is a risk of preserving harmful biases or weakening safety mechanisms. We advocate for responsible evaluation and deployment of compressed models, especially in sensitive applications.

## 8 Conclusion and Future Work

In this work we propose **Generalized** **Fisher-Weighted** **SVD** **(GFWSVD),** a gradient-based low-rank approximation method that accurately incorporates parameter importance via the full FIM. Unlike the diagonal approximation of the FIM Hsu et al. [2022], GFWSVD considers parameter correlations by leveraging a scalable Kronecker decomposition algorithm, and provides an optimal trade-off between downstream accuracy and computational efficiency. We thoroughly evaluate GFWSVD with encoder-only model BERT on GLUE benchmark and decoder only LLM LLaMa 2 on MMLU reasoning benchmark. We demonstrate GFWSVD's superiority over diagonal Fisher-and activation-based SVD methods, particularly at extremely low ranks.

GFWSVD highlights the critical role of accurate FIM computation in compression. While our approach performs well empirically, its reliance on a rank-1 Kronecker approximation of the Fisher matrix may oversimplify important structure. Future work could explore higher-rank Kronecker series to capture richer task-relevant information, and extend the method to model cross-layer dependencies, potentially improving performance by leveraging transitive correlations across the network.

<!-- 9 -->

### References

Hervé Abdi. Singular value decomposition (svd) and generalized singular value decomposition. Encyclopedia of mneasurement and statistics, 907(912):44,2007.

Takuya Akiba, Shotaro Sano, Takeru Yanase, Toshihiko Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2623-2631. ACM,2019.

Genevera I Allen, Logan Grosenick, and Jonathan Taylor. A generalized least-square matrix decom-position. Journalof the American Statistical Association, 109(505):145-159,2014.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7,2019, Volume 1 (Long and Short Papers), pages 4171-4186. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1423. URL https://doi.org/10.18653/v1/n19-1423.

R.A. Fisher. On the Mathematical Foundations of Theoretical Statistics, pages 11-44. Springer New York, New York, NY, 1992. ISBN 978-1-4612-0919-5. doi: 10.1007/978-1-4612-0919-52.URL https://doi.org/10.1007/978-1-4612-0919-52.

Jonathan Frankle and MMichael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA,USA, May 6-9,2019.OpenReview.net, 2019. URL https://openreview.net/forum?id= rJ1-b3RcF7.

Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot. In Andreas Krause, Emma Brunskill,Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 10323-10337. PMLR,2023. URL https://proceedings.mlr.press/v202/ frantar23a.html.

Gene H Golub and Charles F Van Loan. Matrix computations. JHU press, 2013.

Roger Grosse and James Martens. A kronecker-factored approximate fisher matrix for convolution layers. In International Conference on Machine Learning, pages 573-582. PMLR, 2016.

Arjun K Gupta and Daya K Nagar. Matrix variate distributions. Chapman and Hall/CRC, 2018.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measurin1g massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7,2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=d7KBjmI3GmQ.

Yen-Chang Hsu, Ting Hua, Sungen Chang, Qian Lou, Yilin Shen, and Hongxia Jin. Language model compression with weighted low-rank factorization, 2022. URL https://openreview. net/forum?id=uPv9Y3gmAI5.

Divyansh Jhunjhunwala, Shiqiang Wang, and Gauri Joshi. Fedfisher: Leveraging fisher information for one-shot federated learning. In Sanjoy Dasgupta, Stephan Mandt, and Yingzhen Li, editors, International Conference on Artificial Intelligence and Statistics, 2-4 May 2024, Palau de Congres-sos, Valencia, Spain, volume 238 of Proceedings of Machine Learning Research, pages 1612-1620. PMLR,2024. URL https://proceedings.mlr.press/v238/jhunjhunwala24a.html.

James Kirkpatrick, Razvan Pascanu, et al. Overcoming catastrophic forgetting in neural networks. In Proceedings of the National Academy of Sciences, volume 114, pages 3521-3526,2017.

<!-- 10 -->

Abdoulaye Koroko, Ani Anciaux-Sedrakian, Ibtihel Ben Gharbia, Valérie Garès, Mounir Haddou, and Quang Huy Tran. Efficient approximations of the fisher matrix in neura1 networks using kronecker product singular value decomposition. ESAIM: Proceedings and Surveys, 73:218-237, 2023.

Cornelius Lanczos. An iteration method for the solution of the eigenvalue problem of linear differen-tial and integral operators. Journal of Research of the National Bureau of Standards, 45:255-282, 1950.

Sanwoo Lee, Jiahao Liu, Qifan Wang, Jingang Wang, Xunliang Cai, and Yunfang Wu. Dynamic fisher-weighted model merging via Bayesian optimization. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Associ-ation for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4923-4935, Albuquerque, New Mexico, April 2025. Association for Computational Linguis-tics. ISBN 979-8-89176-189-6. URL https://aclanthology.org/2025.naacl-long.254/.

Zhiteng Li, Mingyuan Xia,Jingyuan Zhang, Zheng Hui, Linghe Kong, Yulun Zhang, and Xiaokang Yang. Adasvd: Adaptive singular value decomposition for large language models. arXiv e-prints, pages arXiv-2502,2025.

Charles Van Loan and Nikos Pitsianis. Approximation with kronecker products. 1992.

Jun Lu, Tianyi Xu, Bill Ding, David Li, and Yu Kang. Large language model compression via the nested activation-aware decomposition. arXiv preprint arXiv:2503.17101,2025.

Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. Comput. Linguistics,19(2):313-330,1993.

Stephen Merity,Caiming Xiong, James Bradbury, and Richard Socher. Pointersentinel mixture models. In 5th International Conference on Learning Representations, ICLR 2017,Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. URL httpos: //openreview.net/forum?id=Byj72udxe.

Guilherme Penedo,Hynek Kydlícek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin A. Raffel, Leandro von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10- 15, 2024,2024. URL http://papers.nips.cc/paper files/paper/2024/hash/370df50ccfdf8bde18f8f9c2d9151bda-Abstract-Datasets andBenchmarksTrack.html.

Alexander Soen and Ke Sun. Trade-offs of diagonal fisher information matrix estimators. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15,2024,2024. URL http://papers.nips.cc/paperfiles/paper/ 2024/hash/0b43289db08ed60edc6451cb2132e203-Abstract-Conference.html.

Zedong Tang, Fenlong Jiang, Maoguo Gong, Hao Li, Yue Wu,Fan Yu,Zidong Wang, and Min Wang. Skfac: Training neural networks with faster kronecker-factored approximate curvature.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13479-13487,2021.

Hugo Touvron, Louis Martin, Kevin Stonne,Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra,Prajjwal Bhargava, Shruti Bhosale, Dan Bikel,Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adlina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov,Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic,Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288, 2023. doi: 10.48550/ARXIV.2307.09288. URL https://doi.org/10.48550/ arXiv.2307.09288.

<!-- 11 -->

Xin Wang, Samiul Alam, Zhongwei Wan, Hui Shen, and Mi Zhang. Svd-llm v2: Optimizing singular value truncation for large language model compression. arXiv preprint arXiv:2503.12340,2025a.

Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. SVD-LLM: Truncation-aware singular value decomposition for large language model compression. In International Conference on Learning Representations (ICLR), 2025b. URL https://openreview.net/forum?id=LNYIUouhdt.

Xiaodong Wu, Wenyi Yu, Chao Zhang, and Philip C. Woodland. An improved empirical fisher approximation for natural gradlient descent. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, ed-itors, Advances in Neural Information Processing Systems 38: Annual Conference on Neu-ral Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, Decem-ber 10 - 15, 2024, 2024. URL http://papers.nips.cc/paperfiles/paper/2024/hash/ f23098fa0cfcdef0e743b134d380eeb9-Abstract-Conference.html.

Zhihang Yuan,Yuzhang Shang, Yue Song, Qiang Wu, Yan Yan, and Guangyu Sun. ASVD: activation-aware singular value decomposition for compressing large language models. CoRR, abs/2312.05821, 2023. doi: 10.48550/ARXIV.2312.05821. URL https://doi.org/10.48550/ arXiv.2312.05821.

## A Appendix A: Special case of diagonal Fisher Information Matrix

In this section, we show that FWSVD, presented in Hsu et al. [2022], is a special case of our generalized approach.

In the work of Hsu et al. [2022], authors propose to minimize the following objective:

$$\min _{\mathbf {W}_{1},\mathbf {W}_{2}}\left\|\mathbf {D}\mathbf {W}^{\star }-\mathbf {D}\mathbf {W}_{2}\mathbf {W}_{1}\right\|_{F}^{2}\tag{16}$$

where D is the diagonal matrix $\sqrt {\text {diag}\left(\mathbb {E}\left[\mathbf {G}\mathbf {G}^{\top }\right]\right)}$ .Specifically, $\mathbf {D}_{i,i}=\sqrt {\sum _{j=1}^{m}\mathbb {E}\left(\mathbf {G}_{i,j}\right)^{2}}$ 

Similarly to 12, we approximate the Fisher Information with a Kronecker product of identity matrix Imand some diagonal matrix $\tilde {\mathbf {D}}$ . As described further in Section 4 and Appendix B, under the permutation R, the problem

$$\min _{\mathbf {D}}\left\|\mathbf {I}_{F}-\mathbf {I}_{m}\otimes \tilde {\mathbf {D}}\right\|_{\mathrm {F}}\tag{17}$$

reduces to minimization of the expression

$$\min _{\mathbf {d}}\left\|\mathbb {E}[\mathbf {G}\otimes \mathbf {G}]-\left(\mathbf {I}_{n}\odot \mathbf {I}_{n}\right)d·\text {vec}\left(\mathbf {I}_{m}\right)^{\top }\right\|_{\mathrm {F}}\tag{18}$$

where · is a Khatri-Rao product (column-wise Kronecher product) and · is a vector outer product; d is a vector diagonal of $\tilde {\mathrm {D}}$ ;E[G⊗G]is a permuted Fisher Information matrix $\tilde {\mathbf {I}}_{\mathbf {F}}$ , defined in Eq 13.

For simplicity, we will use a shorter notation. LetE=E[G⊗G]Z=In⊙In,v=vec(Im).Then, the problem 18 is equivalent to

$$\min _{\mathbf {d}}\left\|\mathbf {Z}d·v^{\top }-\mathbf {E}\right\|_{\mathrm {F}}\tag{19}$$

Applying first-order optimality conditions yields:

$$\left\langle \mathbf {Z}δd·v^{\top },\mathbf {Z}d·v^{\top }-\mathbf {E}\right\rangle =0$$

$$\left\langle δd·v^{\top },\mathbf {Z}^{\top }\mathbf {Z}d·v^{\top }-\mathbf {Z}^{\top }\mathbf {E}\right\rangle =0$$

$$\left\langle δd,\mathbf {Z}^{\top }\mathbf {Z}d·v^{\top }v-\mathbf {Z}^{\top }\mathbf {E}v\right\rangle =0$$

<!-- 12 -->

SinceZ\topZ=In,v\topv=\|v\|22=\|vec(Im)\|22=m,we have:

$$d=\frac {1}{m}\left(\mathbf {I}_{n}\odot \mathbf {I}_{n}\right)^{\top }\mathbb {E}[\mathbf {G}\otimes \mathbf {G}]\text {vec}\left(\mathbf {I}_{m}\right)=\frac {1}{m}\left(\mathbf {I}_{n}\odot \mathbf {I}_{n}\right)^{\top }\text {vec}\left(\mathbb {E}\left[GG^{\top }\right]\right)=\frac {1}{m}\text {diag}\left(\mathbb {E}\left[GG^{\top }\right]\right)\tag{20}$$

Thus, diagonal matrix $\tilde {\mathbf {D}}$ from Kronecker product approximation problem 17 equals square of matrix D from the FWSVD formulation 16 up to the constant $\frac {1}{}$ 

We apply Theorem 1 to find factorsW2W1for the obtained approximation $\mathbf {I}_{F}=\mathbf {I}_{m}\otimes \tilde {\mathbf {D}}$ 

$$\mathbf {W}_{2}=\sqrt {\tilde {\mathbf {D}}}^{-1}\hat {\mathbf {U}}_{r}\sqrt {\hat {\mathbf {S}}_{r}}=\mathbf {D}^{-1}\hat {\mathbf {U}}_{r}\sqrt {\hat {\mathbf {S}}_{r}},\mathbf {W}_{1}=\sqrt {\hat {\mathbf {S}}_{r}}\hat {\mathbf {V}}_{r}^{\top }\tag{21}$$

where $\hat {\mathbf {U}}_{r}\hat {\mathbf {S}}_{r}\hat {\mathbf {V}}_{r}^{\top }$ is r-rank SVD of $\sqrt {\tilde {\mathbf {}}}\mathbf {W}^{\star }=\mathbf {}\mathbf {W}^{\star }$ . This is the same solution that minimizes the problem 16 from FWSVD paper Hsu et al. [2022]. Consequently,FWSVD approach is a special case of diagonal Kronecker product approximation of Fisher Information.

## B Appendix B: Additional Explanations for Kronecker decomposition adaptation

Let's show that the permutedIFin the Kronecker decomposition algorithm can be expressed as the Kronecker product of the corresponding gradient matrices.

We start with the empirical Fisher information matrix defined as:

$$\mathcal {I}_{F}=\frac {1}{|D|}\sum _{i=1}^{|D|}g_{i}g_{i}^{\top }\tag{22}$$

and its reordered version:

$$\tilde {\mathcal {I}}_{F}=\mathcal {R}\mathcal {I}_{F}\tag{23}$$

Using the identity

$$\text {vec}\left(\boldsymbol {g}_{i}\boldsymbol {g}_{i}^{\top }\right)=\boldsymbol {g}_{i}\otimes \boldsymbol {g}_{i},$$

we obtain:

$$\text {vec}\left(\mathcal {I}_{F}\right)=\frac {1}{|D|}\sum _{i=1}^{|D|}\text {vec}\left(\boldsymbol {g}_{i}\boldsymbol {g}_{i}^{\top }\right)=\frac {1}{|D|}\sum _{i=1}^{|D|}\left(\boldsymbol {g}_{i}\otimes \boldsymbol {g}_{i}\right)\tag{24}$$

LetP∈R(ab)2×(ab)2be the unique permutation matrix such that for any matrices**A,**B∈Ra×b

$$\mathcal {P}·\text {vec}(\mathbf {A}\otimes \mathbf {B})=(\text {vec}(\mathbf {A})\otimes \text {vec}(\mathbf {B})).\tag{25}$$

Using this definition, we can write:

$$\mathcal {P}·\text {vec}\left(\mathbf {G}_{i}\otimes \mathbf {G}_{i}\right)=\text {vec}\left(\mathbf {G}_{i}\right)\otimes \text {vec}\left(\mathbf {G}_{i}\right)\tag{26}$$

Therefore, the vectorized Fisher information becomes:

$$\text {vec}\left(\mathcal {I}_{F}\right)=\frac {1}{|D|}\sum _{i=1}^{|D|}\mathcal {P}·\text {vec}\left(\mathbf {G}_{i}\otimes \mathbf {G}_{i}\right)=\mathcal {P}·\text {vec}\left(\frac {1}{|D|}\sum _{i=1}^{|D|}\left(\mathbf {G}_{i}\otimes \mathbf {G}_{i}\right)\right)=\mathcal {P}\text {vec}\left(\tilde {\mathcal {I}}_{F}\right)\tag{27}$$

<!-- 13 -->

So, $\tilde {\mathcal {I}}_{F}$ can be defined as $\frac {1}{|D|}\sum _{i=1}^{|D|}\left(\mathbf {G}_{i}\otimes \mathbf {G}_{i}\right)$ . This fact is used in the accelerated adaptation of the Kronecker Factorization algorithm.

Now,suppose aIFand $\tilde {\mathcal {I}}_{F}$ are connected withR∈Rn×n(see Eq. 23):

$$\text {vec}\left(\widetilde {\mathcal {I}}_{F}\right)=(I\otimes \mathcal {R})·\text {vec}\left(\mathcal {I}_{F}\right),\mathcal {P}=I\otimes \mathcal {R}\tag{28}$$

## C Appendix C: Right vector-matrix multiplication

We can define right vector-matrix multiplication as follows:

$$\mathcal {I}_{F}^{\top }z=\left(\sum _{i=1}^{|D|}\mathbf {G}_{i}\otimes \mathbf {G}_{i}\right)^{\top }z\tag{29}$$

Using property of the Kronecker product(K⊗L)vc(C)=vc(K\topCL):

IF\topz=∑i=1|D|vec(GiZGi\top)wherez=vec(Z),Z∈Rm×m (30)

## D Appendix D: Extended GLUE results

We report extended compression results on tasks of GLUE benchmark in Table 4.

<!-- 14 -->

Table 4: Performance of bert-base-uncased compressed by various methods under compression rates from 60% to 99% on GLUE benchmark. Lower is better for COLA (↓), higher is better for all other tasks(↑).

<table border="1" ><tr>
<td>METHOD/DATASET</td>
<td>MRPC</td>
<td>STSB</td>
<td>QQP</td>
<td>MNLI1</td>
<td>QNLI</td>
<td>RTE</td>
<td>COLA↓</td>
<td>SST2</td>
</tr><tr>
<td>Full model</td>
<td>0.77</td>
<td>0.87</td>
<td>0.90</td>
<td>0.83</td>
<td>0.90</td>
<td>0.56</td>
<td>0.41</td>
<td>0.91</td>
</tr><tr>
<td colspan="9">Compression Rate 99%(r=600)</td>
</tr><tr>
<td>SVD</td>
<td>0.67</td>
<td>0.84</td>
<td>0.90</td>
<td>0.67</td>
<td>0.90</td>
<td>0.56</td>
<td>0.58</td>
<td>0.91</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>0.72</td>
<td>0.73</td>
<td>0.89</td>
<td>0.83</td>
<td>0.90</td>
<td>0.56</td>
<td>0.41</td>
<td>0.91</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td>0.72</td>
<td>0.87</td>
<td>0.90</td>
<td>0.72</td>
<td>0.90</td>
<td>0.55</td>
<td>0.36</td>
<td>0.91</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>0.73</td>
<td>0.87</td>
<td>0.90</td>
<td>0.73</td>
<td>0.90</td>
<td>0.56</td>
<td>0.55</td>
<td>0.92</td>
</tr><tr>
<td></td>
<td colspan="8">Compression Rate 92%(r=500)</td>
</tr><tr>
<td>SVD</td>
<td>0.53</td>
<td>0.82</td>
<td>0.89</td>
<td>0.53</td>
<td>0.90</td>
<td>0.54</td>
<td>0.53</td>
<td>0.89</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>0.71</td>
<td>0.56</td>
<td>0.86</td>
<td>0.81</td>
<td>0.89</td>
<td>0.53</td>
<td>0.44</td>
<td>0.88</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td>0.71</td>
<td>0.87</td>
<td>0.90</td>
<td>0.71</td>
<td>0.89</td>
<td>0.56</td>
<td>0.34</td>
<td>0.91</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>0.73</td>
<td>0.87</td>
<td>0.90</td>
<td>0.73</td>
<td>0.90</td>
<td>0.56</td>
<td>0.49</td>
<td>0.92</td>
</tr><tr>
<td colspan="9">Compression Rate 77%(r=250)</td>
</tr><tr>
<td>SVD</td>
<td>0.49</td>
<td>0.68</td>
<td>0.81</td>
<td>0.49</td>
<td>0.85</td>
<td>0.50</td>
<td>0.17</td>
<td>0.57</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>0.69</td>
<td>0.08</td>
<td>0.76</td>
<td>0.50</td>
<td>0.58</td>
<td>0.47</td>
<td>0.11</td>
<td>0.75</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td>0.69</td>
<td>0.86</td>
<td>0.89</td>
<td>0.69</td>
<td>0.89</td>
<td>0.61</td>
<td>0.23</td>
<td>0.80</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>0.71</td>
<td>0.86</td>
<td>0.89</td>
<td>0.71</td>
<td>0.89</td>
<td>0.61</td>
<td>0.38</td>
<td>0.88</td>
</tr><tr>
<td colspan="9">Compression Rate 67%(r=100)</td>
</tr><tr>
<td>SVD</td>
<td>0.32</td>
<td>0.08</td>
<td>0.64</td>
<td>0.32</td>
<td>0.80</td>
<td>0.51</td>
<td>0.01</td>
<td>0.49</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>0.58</td>
<td>0.07</td>
<td>0.74</td>
<td>0.39</td>
<td>0.50</td>
<td>0.47</td>
<td>0.05</td>
<td>0.82</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td>0.69</td>
<td>0.58</td>
<td>0.87</td>
<td>0.71</td>
<td>0.86</td>
<td>0.55</td>
<td>0.21</td>
<td>0.72</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>0.71</td>
<td>0.70</td>
<td>0.87</td>
<td>0.71</td>
<td>0.86</td>
<td>0.55</td>
<td>0.21</td>
<td>0.72</td>
</tr><tr>
<td colspan="9">Compression Rate 64%(r=50)</td>
</tr><tr>
<td>SVD</td>
<td>0.32</td>
<td>0.19</td>
<td>0.57</td>
<td>0.32</td>
<td>0.78</td>
<td>0.48</td>
<td>0.02</td>
<td>0.49</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>0.68</td>
<td>-0.03</td>
<td>0.73</td>
<td>0.49</td>
<td>0.76</td>
<td>0.51</td>
<td>-0.03</td>
<td>0.80</td>
</tr><tr>
<td>FWSVD Hsu et al.[2022]</td>
<td>0.69</td>
<td>0.65</td>
<td>0.84</td>
<td>0.69</td>
<td>0.72</td>
<td>0.46</td>
<td>0.03</td>
<td>0.77</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>0.69</td>
<td>0.65</td>
<td>0.84</td>
<td>0.69</td>
<td>0.72</td>
<td>0.46</td>
<td>0.05</td>
<td>0.77</td>
</tr><tr>
<td colspan="9">Compression Rate61%(r=10)</td>
</tr><tr>
<td>SVD</td>
<td>0.32</td>
<td>0.32</td>
<td>0.67</td>
<td>0.32</td>
<td>0.61</td>
<td>0.51</td>
<td>0.00</td>
<td>0.49</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>0.61</td>
<td>-0.14</td>
<td>0.64</td>
<td>0.40</td>
<td>0.57</td>
<td>0.49</td>
<td>-0.04</td>
<td>0.76</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td>0.37</td>
<td>0.32</td>
<td>0.79</td>
<td>0.37</td>
<td>0.57</td>
<td>0.49</td>
<td>0.00</td>
<td>0.49</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>0.53</td>
<td>0.60</td>
<td>0.79</td>
<td>0.53</td>
<td>0.62</td>
<td>0.47</td>
<td>0.05</td>
<td>0.65</td>
</tr><tr>
<td colspan="9">Compression Rate 60%(r=1)</td>
</tr><tr>
<td>SVD</td>
<td>0.32</td>
<td>0.04</td>
<td>0.69</td>
<td>0.31</td>
<td>0.55</td>
<td>0.53</td>
<td>0.00</td>
<td>0.49</td>
</tr><tr>
<td>ASVD Yuan et al. [2023]</td>
<td>0.62</td>
<td>-0.10</td>
<td>0.64</td>
<td>0.42</td>
<td>0.50</td>
<td>0.49</td>
<td>-0.03</td>
<td>0.70</td>
</tr><tr>
<td>FWSVD Hsu et al. [2022]</td>
<td>0.32</td>
<td>0.18</td>
<td>0.72</td>
<td>0.32</td>
<td>0.51</td>
<td>0.50</td>
<td>0.00</td>
<td>0.49</td>
</tr><tr>
<td>GFWSVD (Ours)</td>
<td>0.42</td>
<td>0.70</td>
<td>0.74</td>
<td>0.42</td>
<td>0.65</td>
<td>0.52</td>
<td>0.05</td>
<td>0.49</td>
</tr></table>

<!-- 15 -->

