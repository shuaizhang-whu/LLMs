<!--  -->

<!-- **NVIDIA.** -->

# EoRA: Fine-tuning-free Compensation for

# Compressed LLM with Eigenspace Low-Rank

# Approximation

**Shih-Yang** Liu¹ **Maksim** Khadkevich Nai CChit Fung1 **Charbel Sakr Chao-Han Huck Yang**

Chien-Yi Wang Saurav **Muralidharan Hongxu Yin Kwang-Ting Cheng1 Jan Kautz**

**Yu-Chiang Frank Wang Pavlo Molchanov Min-Hung Chen**

**Abstract:** While post-training compression techniques effectively reduce the memory footprint,latency,and power consumption of Large Language Models (LLMs), they often result in noticeable accuracy degradation and remain limited by hardware and kernel constraints that restrict supported compression formats-ultimately reducing flexibility across a wide range of deployment scenarios. In this work, we propose EoRA-a novel, **fine-tuning-free** method that augments compressed LLMs with low-rank matrices, allowing users to rapidly enhance task-specific performance and freely balance the trade-off between accuracy and computational overhead beyond the constraints of compression formats. EoRA consistently outperforms prior training-free low-rank methods in recovering the accuracy of compressed LLMs, achieving notable accuracy improvements (e.g., **10.84%** on ARC-Challenge, 6.74% on MathQA, and 6.74% on GSM8K) for LLaMA3-8B compressed to 3-bit. We also introduce an optimized CUDA kernel, accelerating inference by up to 1.4x and reducing memory overhead through quantizing EoRA. Overall, EoRA offers a prompt solution for improving the accuracyof compressed models under varying user requirements, enabling more efficient and flexible deployment of LLMs. Code is available at https://github.com/NVlabs/EoRA.

## 1. Introduction

Although Large Language Models (LLMs) excel in various tasks, their deployment remains challenging due to high inference costs. Post-training compression methods, like quantization [Frantar et al., 2023,Lin et al., 2024, Liu et al., 2025, Tseng et al., 2024] and pruning [Ma et al., 2023, Frantar and Alistarh, 2023, Sun et al., 2024], aim to reduce computational demands but typically cause accuracy loss or face hardware/kernel constraints, limiting deployment flexibility. For instance, strict hardware-supported formats, such as 2:4sparsity on NVIDIA GPUs or integer-only quantization kernels, prevent intermediate approaches (e.g., 2.X:4sparsity or arbitrary-bit quantization) that could offer a more adaptable trade-off between accuracy and latency based on user needs.

To relax these format constraints and improve the accuracy of the compressed models on specified tasks, we formulate a new problem, termed customized compensation: Given a compressed LLM, we attach residual low-rank paths to it to compensate for compression errors and enhance task-specific accuracy, enabling more flexible control over the trade-off between accuracy and compression ratio to accommodate varying user requirements. For example, a user may wish to boost the accuracy of a 2:4 sparsity-pruned model on math reasoning tasks, accepting a modest increase in memory usage and inference latency in return. Importantly,in our problem setting, the weights of the compressed model are not modified during compensation. This enables deployment of a single, general compressed backbone alongside lightweight, task-specific low-rank modules that can be dynamically loaded as needed-allowing for efficient integration with existing multi-adapter inference frameworks (e.g., The vLLM Team) as illustrated in Figure 1. A naive solution is to apply SVD Li et al. [2024], Yao et al. [2024] for compensation; however, this neglects calibration data and thus fails to enhance task-specific performance. Alternatively, LoRA-based methods, such as Li et al. [2024], Dettmers et al. [2023] require fine-tuning, limiting their applicability for rapid task adaptation. These limitations prompt an important question: “How can we swiftly improve the task-specific accuracy for compressed LLMs without fine-tuning?”

To tackle this research challenge, we introduce fine-tuning-free Eigenspace Low-Rank Approximation **(EoRA),** a method designed to efficiently enhance the task-specific accuracy of compressed LLMs while offering users

<!-- 1 affiliated with HKUST. Work done during Shih-Yang's internship at NVIDIA Research. -->

<!-- © 2025 NVIDIA. All rights reserved. -->

<!-- S707unrS[TO.so] tATLClC.OLtC:AIXTe -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

<!-- EoRA Improvement ARC-C MathQA GSM8K +4.53% +3.48% +11.83% ON ON ON OFF ARC-C: MathQA: GSM8K: Other user- W' W′ 34.64 29.91 13.95 specified tasks LLaMA3-8B (2:4 Sparsity) BARC′ BMathQA′ B′ GSM8K B⋯′ ARC-C:30.11 AARC AMathQA AGSM8K A.. A⋯ MathQA:26.43 Task-specific Low-rank EoRA Modules GSM8K:2.12 limited task-specific calibration data optimized in minutes -->
![](./images/eb5b1364f993df80.jpg)

Figure 1 | An overview of our proposed EoRA, which enables swift task-specific accuracy enhancement for compressed LLMs without **fine-tuning,** using only a small amount of downstream calibration data. At inference time, a single compressed backbone is loaded, while lightweight, task-specific low-rank modules can be dynamically toggled on and off on demand, enabling efficient and flexible deployment. EoRA with rank 128 boosts the accuracy of the LLaMA3-8B model pruned to 2:4 structured sparsity by 4.53%,3.48%,and 11.83% on ARC-C, MathQA, and GSM8K, respectively--all achieved within minutes using just 64 calibration samples per task.

greater flexibility in managing the trade-off between accuracy and computational overhead. EoRA operates by projecting the compression error into the task-specific eigenspace of each layer's input activations, followed by applying SVD to approximate the projected error. This approach ensures that the SVD approximation error directly aligns with the task-specific compression loss. As a **fine-tuning-free** method,EoRA avoids backpropagation and completes in just a few minutes using minimal calibration data.

We validate the effectiveness of EoRA in boosting the accuracy of compressed LLMs (LLaMA2-7B/13B and LLaMA3-8B) on language generation, commonsense reasoning, and math tasks. Our method consistently outperforms other training-free baselines, especially for aggressively compressed (including pruned,quantized, and both) models (e.g., 2.65%, 3.42%, and **10.99%** improvement on ARC-Challenge, MathQA, and GSM8K when1 compensating 2:4 pruned LLaMA3-8B). To reduce redundant memory transfer overhead from running low-rank compensation, we design a fused kernel that integrates low-rank and quantization operations, achieving up to 1.4x speedup.

The summary of our contributions is as follows:

**·Flexible and Task-specific Model Compensation:** We ppropose,fine-tuning-free Eigenspace Low-Rank Approximation **(EoRA),** a training-free approach that improves the task-specific accuracy of compressed LLMs in minutes using minimal calibration data, while supporting more flexible compression ratios unconstrained by hardware or kernel-imposed format limitations.

**·Eigenspace** **Projection:** EoRA leverages calibration data to project the compression error into the task-specific eigenspace and utilizes the corresponding eigenvalues as importance indicators, effectively aligning the approximation error with task-specific compression loss.

**·Efficient** **Inference**: We develop a custom kernel that fuses part of the low-rank matrix multiplication with a quantization kernel, accelerating EoRA inference by up to 1.4x. EoRA is also robust to quaantization, further minimizing the size-overhead from low-rank compensation matrices.

<!-- 2 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

## 2. Preliminaries

**Post-training** **compression** aims to compress a well-optimized model by a targeted compression ratio utilizing only a limited set of calibration data. The compression process is often framed as a layer-wise optimization problem, aiming to minimize the layer-wise output difference between the original weight W∈Rd×kand the compressed weight $\hat {W}_{}\in \mathbb {R}^{dxk}$  for each layer l. Then the layer-wise model compression loss can be formed as:

$$\underset {\hat {W}_{l}}{\arg \min }\left\|W_{l}X_{l}-\hat {W}_{l}X_{l}\right\|_{F}\tag{1}$$

whereXl∈Rk×nis the input activation of layer l and F denotes the Frobenius error between the layer-wise output. Once the compression is complete,theWlfor each layer will be substituted with $\hat {W}_{}$ ,resulting in a smaller model size, faster inference, or both. However, their flexibility is often limited by a discrete set of compression formats (e.g.,2:4 sparsity,3/4-bit quantization), making it challenging to meet the diverse accuracy/overhead requirements of different users.

To bypass the limitations of fixed compression formats and enhance the accuracy of compressed models on user-specified tasks, we introduce a new problem, termed customized compensation: Given an already compressed model, the objective is to add residual low-rank paths that compensate for compression errors and enhance task-specific accuracy according to user-defined accuracy/overhead requirements. Crucially,the compressed model's weights remain unchanged during compensation, enabling the deployment of a single, general compressed backbone with lightweight, task-specific low-rank modules that can be dynamically loaded as needed, facilitating efficient integration with existing inference frameworks, as illustrated in Figure 1.

A simple approach to obtain lowv-rank residual paths that compensate for compression errors is to directly apply Singular Value Decomposition (SVD) [Li et al., 2024, Yao et al., 2024, Li et al., 2025]. More specifically, this method relies on a closed-form solution by using SVD to approximate the compression error $\Delta W_{l}=W_{l}-\hat {W}_{l}$ asΔWl≈UlΣlVlT,whereΣl∈Rr×r is a diagonal matrix containing the top-r largest singular value sorted in descending order,andU∈R×,V∈Rk×are orthonormal matrices, with each column representing the singular vectors corresponding to the singular values inΣ.The product ofUl and Σlcan then be treated as Bl=UlΣlwithVlTbeing treated asAl. Overall, the error approximation loss can be formulated as:

$$\underset {B_{l},A_{l}}{\arg \min }\left\|\Delta W_{l}-B_{l}A_{l}\right\|_{F}\tag{2}$$

and SVD is applied onΔWlto minimize the above equation. However, naively applying SVD to optimize error approximation loss (Eq.2) does not ensure minimization of the layer-wise compression loss (Eq.1) and ignores calibration data, making it ineffective for task-specific accuracy recovery. While LoRA-based methods Li et al. [2024], Dettmers et al. [2023] address this issue, they require fine-tuning and are less suitable for rapid adaptation. This raises a key question: “How can we swiftly improve the task-specific accuracy for compressed LLMs without fine-tuning?”. For simplicity, we omit the subscript l, which corresponds to layer l in the following sections.

## 3. Method: EoRA

To tackle the challenge of improving task-specific accuracy of compressed LLMs without fine-tuning,we introduce fine-tuning-free Eigenspace Low-Rank Approximation **(EoRA)-a** method that preserves the efficiency of existing training-free solutions while substantially improving their effectiveness in task-specific accuracy recovery.

First, we propose projecting the compression error into the eigenspace [Stewart, 2001] of the corresponding layer's input activations, ensuring a direct alignment between the error approximation loss (Eq. 2) and the overall layer-wise model compression loss(Eq.1).Inspired boy the classical Principal Component Analysis (PCA) algorithm, we leverage the eigenvalues of each activation channel as importance scores to indicate the importance of each column after the eigenprojection. This allows us to allocate more low-rank representation capacity to approximate the more critical error elements. Following PCA, we perform the eigendecomposition on $\tilde {X}\tilde {X}^{T}$  where $\tilde {X}\in \mathbb {R}^{k\times n}$ is the average of the input activations over the task-specific calibration set. The eigendecomposition $\tilde {X}$ $\tilde {X}^{T}=QΛQ^{T}$ is then used to derive the eigenspace projection matrixQ∈Rk×k,,whose columns are the eigenvectors,andΛ∈Rk×k, which is a diagonal matrix with each diagonal element being the corresponding eigenvalues of the eigenvectors in Q. WVe then propose to project the compression error ΔW into

<!-- 3 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

the eigenspace with the projection matrix $Q^{\prime }=Q\sqrt {Λ}$ to obtain the projected error ΔW′∈Rd×k=ΔWQ′. The proposed new error approximation loss, EoRA loss, can be formulated as:

$$\underset {B^{\prime },A^{\prime }}{\arg \min }\left\|\Delta W^{\prime }-B^{\prime }A^{\prime }\right\|_{F}\tag{3}$$

where SVD is applied to approximateΔW′asSVD(ΔW′)≈U′Σ′V′T,andΣ′∈Rr×r contains the top-r singular values. U′∈Rx and V′∈Rk×r are orthonormal matrices with columns representing the corresponding singular vectors. Then the low-rank matrices B′ and A′ are then assigned as B′=U′Σ′and A′=V′T. This loss function ensures that error columns associated with larger eigenvalues are approximated more accurately than those with smaller eigenvalues. We then multiply the low-rank approximation in the eigenspaceΔW′with $Q^{\prime -1}=\sqrt {Λ}^{-1}Q^{T}$ toproject back to the original space, obtaining the final task-specific compression error approximation asΔW=ΔW′Q′-1≈B′A′Q′-1.Q′is invertible because $Q^{\prime -1}=\sqrt {Λ}^{-1}Q^{T}$ and $Q^{\prime }Q^{\prime -1}=Q\sqrt {Λ}\sqrt {Λ}^{-1}Q^{T}$ . Here, the middle term $\sqrt {Λ}\sqrt {Λ}^{-1}$  simplifies to the identity matrix,and since Q is an orthogonal matrix,(QQTalso yields the identity matrix. The product of A′ and Q′-1can be consolidated into a single matrix with the same dimensions as the ooriginal A′', ensuring no additional inference latency as A=A′Q′-1. Then, the forward pass of one linear layer of the compressed model compensated with EoRA for the input activation X can be formulated as:

$$\hat {W}X+B^{\prime }AX\tag{4}$$

EoRA compensation is applied to each compressed linear layer, and the overall **fine-tuning-free** optimiza-tion of Eq. 3 across all linear layers can be completed in just a few minutes, enabling users to rapidly enhance the accuracy of compressed LLMs on their chosen downstream tasks using only a small amount of task-specific calibration data-without any need for backpropagation. EoRA can also provide better initialization for further LoRA fine-tuning, offering users the option to further improve accuracy if additional computational resources are available. Moreover, the low-rank matrices of EoRA are robust to quantization, which can further reduce the additional memory/inference cost. Please refer to Sec. 4.5 for more details.

### Algorithm 1 Eigenspace low-rank approximation (EoRA)

**Input:** $\tilde {X}$ : Average of the input activations of the current layer over the calibration set,W: Full-precision Weight, $\hat {W}$ Compressed Weight, r: Compensation rank

Output:B′,A: Two low-rank matrices for compensation.

1.ΔW=W-W

2. Run Eigendecompostion on XXT=QAQT

3.Reformulate $QΛQ^{T}=(Q\sqrt {Λ})\left(\sqrt {Λ}Q^{T}\right)=Q^{\prime }Q^{\prime T}$ 

4. Project the compression error to eigenspaceΔW′=ΔWQ′

5. Run r-rankSVD approximation on ΔW′,B′A′=U′Σ′V′=SVD(ΔW′)

6. Project the approximation back to the originalspaceA=A′Q′-1

7. The final forward pass of current layer becomes $\hat {W}X+B^{\prime }AX$ 

**Mapping EoRA loss (Eq. 3) to task-specific compression loss (Eq.** 1): When Eq. 1 is conditioned on different task-specific calibration data, it also implies the compressed model's accuracy on each corresponding task. Therefore, the objective of task-specific low-rank compensationis to approximateΔWthat minimizes Eq. 1, using input activations X derived from the calibration data of different tasks. To achieve this,we reformulate the compression objective for each layer as:

$$\underset {B,A}{\arg \min }\|WX-(\hat {W}+BA)X\|_{F}=\underset {B,A}{\arg \min }\|\Delta WX-BAX\|_{F}\tag{5}$$

Since the Frobenius norm of a matrix is equal to the square root of its Gram matrix [Sun, 1991, Wang et al., 2025],the minimization problem can be rewritten as:

$$\underset {B,A}{\arg \min }\|\Delta WX-BAX\|_{F}=\underset {B,A}{\arg \min }\left[\text {trace}\left((\Delta W-BA)XX^{T}(\Delta W-BA)^{T}\right)\right]^{\frac {1}{2}}\tag{6}$$

Directly applying SVD onΔW -initially does not guarantee the minimization of the above equation Eq. 6. To address this issue, EoRA projectsΔWinto the eigenspace before performing SVD. In the following,we demonstrate that minimizing Eq. 3 with SVD is the same as minimizing Eq. 6.

<!-- 4 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

**Theorem** 1. For an activation matrixX, whose matrix productXXThas an eigendecomposition given by XXT=QΛQT. By projecting the compression error ΔWinto the eigenspace with $Q\sqrt {Λ}$ as $\Delta W^{\prime }=\Delta WQ\sqrt {Λ},$ minimizingEq. 3 via SVD becomes equivalent to minimizing Eq.6

Proof. First,note thatXXT=QΛQT, and by substituting this into Eq. 6,we get

$$\left[\text {trace}\left((\Delta W-BA)QΛQ^{T}(\Delta W-BA)^{T}\right)\right]^{\frac {1}{2}}\quad =\left[\text {trace}\left((\Delta WQ-BAQ)Λ\left(\Delta WQ-BAQ\right)^{T}\right)\right]^{\frac {1}{2}}\tag{7}$$

Since $Λ=\sqrt {Λ}\sqrt {Λ}$ and $\sqrt {Λ}=\sqrt {Λ}^{T}$ ,the aboveEq.7can further be rewritten as:

$$\left[\text {trace}\left((\Delta WQ\sqrt {Λ}-BAQ\sqrt {Λ})\left(\Delta WQ\sqrt {Λ}-BAQ\sqrt {Λ}\right)^{T}\right)\right]^{\frac {1}{2}}\tag{8}$$

Let $Q^{\prime }=Q\sqrt {Λ},$  then Eq. 8 becomes:

$$\left[\text {trace}\left(\left(\Delta WQ^{\prime }-BAQ^{\prime }\right)\left(\Delta WQ^{\prime }-BAQ^{\prime }\right)^{T}\right)\right]^{\frac {1}{2}}\quad =\left[\text {trace}\left(\left(\Delta W^{\prime }-BAQ^{\prime }\right)\left(\Delta W^{\prime }-BAQ^{\prime }\right)^{T}\right)\right]^{\frac {1}{2}}=\left\|\Delta W^{\prime }-BAQ^{\prime }\right\|_{F}\tag{9}$$

where the square root of the Gram matrix can be transformed back to the corresponding Frobenius norm according to [Sun, 1991]. By settingBAQ′=B′A′\|ΔW′-BAQ′\|Fbecomes\|ΔW′-B′A′\|F.By the Eckart-Young theorem [Eckart and Young, 1936], the minimization of this Frobenius norm is achieved by running SVD onΔW′, therefore, we prove that minimizing \|ΔW′-B′A′\|Fvia SVD is equivalent to minimizing Eq. 6, where low-rank approximation ofΔW′isSV(ΔW′)=B′A′.Note that the above minimization is constrained to the rank of A′ and B′.

## 4. Experiments

### 4.1. Experiments Details

WVe implement EoRA in PyTorch [Paszke et al., 2017], utilizing the Hugging Face Transformers and Datasets framework [Wolf et al., 2019]. All experiments are conducted on a single NVIDIA H100 GPU. We primarily focus on evaluating EoRA for compensating LLaMA2-7B/13B and LLaMA3-8B models, compressed using SparseGPT [Frantar and Alistarh, 2023], a widely adopted pruning method, and GPTQ [Frantar et al., 2023] for quantization. Channel-wise asymmetric quantization is applied across all experiments, and we follow the settings from [Huang et al., 2024] to construct the calibration dataset for both SparseGPT and GPTQ.

We compare EoRA with **ZeroQuant-V2** Yao et al. [2024] which proposes using simple SVD for optimizing Eq. 2. Although Activation-aware Singular Value Decomposition (ASVD) Yuan et al. [2023] is designed to replace the entire model with its low-rank decomposition rather than approximating the compression errors, its strategy of incorporating activation distribution variance can also be adapted for error compensation using low-rank matrices. Specifically, we scale the compression error ΔW using a diagonal scaling matrix S, where each diagonal entrySiiis computed based on the average absolute value of the activations $\tilde {X}$  in the i-th channel as $S_{ii}=\left(\frac {1}{}\sum _{j=1}^{}\left|\tilde {X}_{ij}\right|\right)^{\frac {1}{2}}.$  Here, n denotes the number of activation entries in the i-th channel. We then apply SVD to the scaled errorΔW"=ΔWSto obtain its low-rank approximation. Since S is invertible, we can project the approximation back to the original space asΔW=ΔW"-1≈B"A"-1same as how EoRA project its compensation back to the original space. We refer to this method as **Act-S** in the remainder of this paper. We also compare EoRA with a training-based method, **ApiQ** Liao et al. [2024],which optimizes low-rank matrices (A and B) using gradient-based training to minimize Eq. 6. In our comparison, we limit the evaluation to the layer-wise variant of ApiQ, as other variants require substantially more memory or training time. These more resource-intensive versions align more closely with PEFT methods rather than training-free low-rank approximation approaches. For instance, when applied at the model level, ApiQ effectively becomnes equivalent to training LoRA on top of a compressed model-shifting its focus toward fine-tuning rather than training-free compensation, and thus falling outside the scope of this study. Note that the optimization time for both EoRA and Act-S is comparable, with each completing within minutes, whereas ApiQ requires over hours to optimize.

<!-- 5 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

We evaluate EoRA and the baselines on improving the task-specific accuracy of the compressed LLMs on language generation, commonsense reasoning, and math reasoning tasks using the LM-Evaluation-Harness framework [Gao et al., 2024]. We pick WikiText2 for the language generation task and perplexity as the evaluation metric. For commonsense reasoning, we select ARC-Challenge (ARC-C) [Clark et al., 2018],and for math reasoning ability, we choose MathQA [Amini et al., 2019] and GSM8K [Cobbe et al., 2021]. We sample 128 concatenated sentences of length 2048 from the WikiText2 training set as the calibration set for EoRA,Act-S, and ApiQ for the language generation task. For commonsense reasoning tasks, we sample 32concatenated sentences of length 2048 from the ARC training set and combine them with 32 concatenated sentences of the same length from C4 [Raffel et al., 2020] to construct the calibration set for EoRA, Act-S, and ApiQ. Similarly, for the math reasoning tasks, we sample 32 concatenated sentences oflength 2048 from the MathQA/GSM8K training set and combine them with 32 concatenated sentences from C4 to form the calibration set for the three methods.

### 4.2. Main Results

#### 4.2.1. Sparsity Error Compensation

Table 1|Perplexity and commonsense/math reasoning results for LLaMA3-8B pruned with 2:4 sparsity using SparseGPT, with all compensation methods evaluated at rank 128.

<table border="1" ><tr>
<td>Model</td>
<td>Sparsity</td>
<td>Compensation Method</td>
<td>Wikitext2↓</td>
<td>ARC-C↑</td>
<td>MathQA↑</td>
<td>GSM8K↑</td>
</tr><tr>
<td rowspan="6">LLaMA3-8B</td>
<td>-</td>
<td>-</td>
<td>6.13</td>
<td>50.42</td>
<td>40.10</td>
<td>36.23</td>
</tr><tr>
<td rowspan="5">2:4</td>
<td>-</td>
<td>12.32</td>
<td>30.11</td>
<td>26.43</td>
<td>2.12</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>11.31</td>
<td>31.99</td>
<td>26.49</td>
<td>2.956</td>
</tr><tr>
<td>Act-S</td>
<td>11.32</td>
<td>31.74</td>
<td>26.73</td>
<td>3.26</td>
</tr><tr>
<td>ApiQ</td>
<td>11.08</td>
<td>34.21</td>
<td>28.77</td>
<td>14.55</td>
</tr><tr>
<td>EoRA (Ours)</td>
<td>11.07</td>
<td>34.64</td>
<td>29.91</td>
<td>13.95</td>
</tr></table>

To assess the effectiveness of EoRA in compensating for sparsity error, we compare EoRA with all the baselines on LLaMA2-7B/13B and LLaMA3-8B models pruned with SparseGPT to 2:4 sparsity-the only sparsity format that yields actual inference speedups on GPUs. Rank of all the compensation methods is set to 128, and the results of LLaMA3-8B are presented in Table 1, while the full results, including LLaMA2-7B/13B, are provided in Table 4 in the appendix. EoRA consistently outperforms all training-free baselines,achieving gains of 2.9%, 2.1%, and 10.7% over Act-S on ARC-C, MathQA, and GSM8K, respectively. Furthermore,it surpasses ApiQ by 0.4% on ARC-C and 1.1% on MathQA, while delivering comparable results on GSM8K-all with significantly faster optimization time (15 minutes vs. 2.5 hours). Furthermore, EoRA proves robustness across different model sizes, continuing to outperform ZeroQuant-V2, Act-S and ApiQ in boosting the accuracy of 2:4 pruned LLaMA2-7B/13B across ARC-C and MathQA as shown in Table 4. We further assess the generalizability and compatibility of EoRA with pruning methods beyond SparseGPT. Specifically, we evaluate EoRA on LLaMA3-8B prunedI to 2:4 sparsity using Wanda Sun et al. [2024], where EoRA continues to outperform all the training-free baseline methods. For additional details, please refer to Section A.4.

#### 4.2.2. Quantization Error Compensation

We evaluate EoRA on LLaMA2-7B/13B and LLaMA3-8B models quantized with GPTQ to 4-bit and 3-bit to assess the effectiveness of EoRA in compensating for quantization error. The ranks for all the methods are set to 128. From Table 2, 3-bit quantization causes significant accuracy degradation,with losses up to 29.5%/17.7%/35.8% on ARC-C, MathQA, and GSM8K, respectively. By applying EoRA, we demonstrate that the accuracy loss can be reduced to 18.7%/10.9%/24.3% on ARC-C, MathQA,and GSM8K-providing 10.8%/6.7%/11.5% improvement, outperforming all the baseline methods for compensating the quantization error. On the other hand, although 4-bit quantization does not result in as much accuracy loss as 3-bit quantization,applying EoRA can still generally enhance the performance of the 4-bit model, offering up to a 2.2% and 3.14% accuracy boost on ARC-C and MathQA, respectively. Comprehensive results, including those for LLaMA2-7B/13B, are presented in Table 5 in the appendix, where a similar trend of improvement with EoRA is observed. We also explore the feasibility of using EoRA to improve ultra-compressed models

<!-- 6 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

Table 2 | Perplexity and commonsense/math reasoning results forLLaMA3-8B quantized to 3/4-bits using GPTQ, with all compensation methods evaluated at rank 128.

<table border="1" ><tr>
<td>Model</td>
<td>W bits</td>
<td>Compensation Method</td>
<td>Wikitext2↓</td>
<td>ARC-C↑</td>
<td>MathQA↑</td>
<td>GSM8K↑</td>
</tr><tr>
<td rowspan="11">LLaMA3-8B</td>
<td>-</td>
<td>-</td>
<td>6.13</td>
<td>50.42</td>
<td>40.10</td>
<td>36.23</td>
</tr><tr>
<td rowspan="5">W4</td>
<td>-</td>
<td>7.00</td>
<td>45.90</td>
<td>34.07</td>
<td>27.74</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>6.80</td>
<td>45.24</td>
<td>36.51</td>
<td>31.23</td>
</tr><tr>
<td>Act-S</td>
<td>6.82</td>
<td>47.86</td>
<td>35.84</td>
<td>29.34</td>
</tr><tr>
<td>ApiQ</td>
<td>6.87</td>
<td>46.58</td>
<td>36.18</td>
<td>30.09</td>
</tr><tr>
<td>EoRA(Ours)</td>
<td>6.80</td>
<td>47.44</td>
<td>37.21</td>
<td>30.70</td>
</tr><tr>
<td rowspan="5">W3</td>
<td>-</td>
<td>15.64</td>
<td>20.90</td>
<td>22.37</td>
<td>0.45</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>10.24</td>
<td>30.02</td>
<td>26.43</td>
<td>3.79</td>
</tr><tr>
<td>Act-S</td>
<td>10.19</td>
<td>31.28</td>
<td>25.42</td>
<td>4.09</td>
</tr><tr>
<td>ApiQ</td>
<td>10.41</td>
<td>30.46</td>
<td>26.86</td>
<td>10.79</td>
</tr><tr>
<td>EoRA (Ours)</td>
<td>10.06</td>
<td>31.74</td>
<td>29.11</td>
<td>11.90</td>
</tr></table>

that combine both pruning and quantization. In this setting, EoRA continues to outperform all baselines on ARC-C and MathQA. Further details can be found in Section A.3 of the Appendix.

### 4.3. Ablation Study: Ranks and Calibration Sizes

<!-- ARC-C 52 50 47 ZereQuant-V2 45 Act-S ApiQ AoeanaoH 42 EoRA(Ours) 40 Original Accuracy 北3 2:4 Sparse 32 30 27 64 128 256 512 Rank -->
![](./images/a55b1b33cc868638.jpg)

(a)

<!-- MathQA 40 38 ZeroQuant-V2 Act-S 36 ApiQ Kelna2y 34 EoRA(Ours) Original Accuracy 32 2:4 Sparse 30 28 26 64 128 256 512 Rank -->
![](./images/81228a0734ae4cea.jpg)

(b)

<!-- GSM8K 35 30 ZeroQuant-V2 Act-S ApiQ aiooy EoRA(Ours) Original Accuracy 2:4 Sparse 10 5 64 128 256 512 Rank -->
![](./images/81f601cded64f8b2.jpg)

(c)

Figure 2 | Results of applying EoRAand other baselines with rank set to {64,128,256,512} to improve LLaMA3-8B models pruned to 2:4 sparsity by SparseGPT on (a) ARC-C/(b)MathQA/(c) GSM8K.

Since one of the advantages of using EoRA is the greater flexibility in adjusting overall model accuracy without being constrained by specific compression formats, in this section, we investigate the influence of different ranks on adopting EoRA. We vary the rank of EoRA in {64,128,256,512} on compensating LLaMA3-8B pruned to 2:4 sparsity. As shown in Figure 2, EoRA consistently outperforms the two training-free baselines (ZeroQuant-V2 and Act-S) across all tested ranks, with the performance gap becoming more prominent at higher ranks. For instance, on GSM8K, EoRA achieves improvements of 7.43%,10.69%,11.9%,and 14.62% at ranks 64, 128, 256, and 512, respectively. In contrast, the gains on ARC-C remain relatively stable across ranks,ranging between 2% and 4%. Additionally, EoRA begins to outperform ApiQ on GSM8K at higher ranks,with improvements of 1.21% and 2.51% observed at ranks 256 and 512, respectively. These experiments prove that EoRA is robust across different rank settings, offering users a more flexible option upon existing compression configurations to effectively balance the trade-off between inference overhead and model accuracy. A similar trend is observed in the results for LLaMA2-7B/13B shown in Table 8 in the appendix.

We also compare the influence of different calibration sizes on EoRA. We vary the calibration size in {16,32,64}, and compare them on recovering the accuracy of LLaMA3-8B quantized to 3/4-bit and pruned to 2:4 sparsity. Overall, we find that EoRA demonstrates strong robustness and maintains competitive accuracy even with limited calibration data, as shown in Table 9. Notably, using as few as 32 calibration samples to compensate a 2:4 pruned model can even yield better accuracy improvements than using 64 samples.

<!-- 7 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

### 4.4. EoRA as LoRA initialization for Fine-tuning Compressed Models

Table 3 | Finetune the 4-bit compressed LLaMA3-8B models with different initialization of the low-rank matrices for Commonsense/Math reasoning tasks.

<table border="1" ><tr>
<td>Model</td>
<td>Compression Method</td>
<td>Compression Setting</td>
<td>LoRA initialization</td>
<td>ARC-C↑</td>
<td>MathQA↑</td>
</tr><tr>
<td rowspan="6">LLaMA3-8B</td>
<td rowspan="2">Full-precision</td>
<td rowspan="2">-</td>
<td>w/o finetuning</td>
<td>50.42</td>
<td>40.10</td>
</tr><tr>
<td>Standard</td>
<td>56.39</td>
<td>53.56</td>
</tr><tr>
<td rowspan="4">GPTQ</td>
<td rowspan="4">W4</td>
<td>w/o finetuning</td>
<td>45.90</td>
<td>34.07</td>
</tr><tr>
<td>QLoRA</td>
<td>54.09</td>
<td>51.42</td>
</tr><tr>
<td>LoftQ</td>
<td>54.52</td>
<td>53.96</td>
</tr><tr>
<td>EoRA</td>
<td>55.46</td>
<td>56.04</td>
</tr></table>

We show that, with additional computational resources, users can leverage the low-rank matrices from EoRA as initialization for LoRA fine-tuning, enabling further accuracy improvements for compressed models. We follow the conventional LoRA fine-tuning framework, which keeps the compressed model frozen and only tunes the low-rank residual components during fine-tuning. We conduct experiments on compressed LLaMA3-8B models with {2:4 sparsity, 4-bit, 3-bit} compression. The rank of LoRA is set to 128 and is applied to every linear layer, initialized using EoRA, SVD following LoftQ Li et al. [2024], and standard initialization following QLoRA Dettmers et al. [2023]. Fine-tuning is performed on the ARC training set for evaluating ARC-C, and on the MathQA training set for the math reasoning task. We fine-tune the models for 3 epochs with a batch size of 64, a learning rate of 1e-5,and a cosine learning rate scheduler. As shown in Table 3, initializing with EoRA substantially enhances the accuracy of compressed models, surpassing both QLoRA and LoftQ when fine-tuning 4-bit quantized LLaMA3-8B, and achieving accuracy on par with standard full-precision fine-tuning. We also observed that the improvements over QLoRA and LoftQ are more pronounced on 3-bit quantized and 2:4 pruned models, aligning with our earlier finding that EoRA is more effective when the compression error is more substantial, as shown in Table 10.

### 4.5. Kernel Optimization, Inference Speed Evaluation and Memory Overhead of EoRA

<!-- Fused Kernel A B' B′ Shared output X k-bit Weight Quantization kernel $\hat {W}$ -->
![](./images/c2c2d6f64c0428d9.jpg)

<!-- 50 FP r128(4-bit) ■r512(4-bit) ■r128(16-bit) 512(16-bit) 45 W4 r512(4-bit) r512(16-bit) 40 r512(4-bit) r512(4-bit) 20 +512(16-bit) +11 0-0A 35 r128(4-bit) r128(16-bit) r128(+-bit) +4.36% Full-precision 30 r128(16-bit) 2:4 GPTQ +19.11% GPTQ-W4+EoRA 25 +10.5 GPTQ-W3+EoRA SparseGPT W3 SparseGPT-2:4+EoRA 20 4 5 6 7 8 9 10 11 12 13 14 15 Model Size (GB) -->
![](./images/04713a30a7e97a44.jpg)

(a) (b)

Figure 3|(a) Wepropose fusing the multiplication of B with the weight quantization kernel to minimize data movement overhead and substantially improve thhe inference latency. (b) The model size and ARC-C accuracy of EoRA with rank 128/512, quantized to 4-bit for compensating LLaMA3-8B quantized to 4/3-bit or pruned to 2:4 sparsity.

While theoretically, compensating a compressed model with low-rank residual paths introduces minimal computational overhead, in practice, it leads to a noticeable increase in latency. This is primarily because input and output must transfer between L2 cache and DRAM twice as often compared to that without a low-rank residual path, shifting the inference process from being computation-bound to memory-bound. This phenomenon is also discussed in [Li et al., 2025]. To address this, we propose fusing the low-bit weight quantization kernel with the matrix multiplication of B, which shares the same output. By doing so,the shared output no longer needs to be offloaded and reloaded to the L2 cache, effectively reducing data transfer overhead as illustrated in Figure 3 (a). Implementation details of our kernel can be found in Section A.8. As shown in Table 11, our custom EoRA kernel substantially accelerates inference compared to using native PyTorch for the low-rank residuial path on top of the low-bit quantized kernel, achieving a speedup of up to 1.4x over FP16 with EoRA of rank 128 at 3-bit quantization. In contrast, without the EoRA kernel,the initial 1.7x speedup provided by the 3-bit quantized kernel drops to 1.1x. Similarly, under 4-bit quantization, the EoRA kernel delivers an extra 0.3x speedup compared to setups without the EoRA kernel.

<!-- 8 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

Finally, EoRA can also be quantized to further reduce the additional cost of residual low-rank compensation paths. In this section, we quantize EoRA of rank {128, 512} to 4/3-bit on compensating three types of compressed LLaMA3-8B models (2:4 pruned, 4-bit quantized, and 3-bit quantized). The complete results are provided in Table 12 in appendix, while the results for LLaMA3-8B are illustrated in Figure 3 (b).As shown in the figure, EoRA is robust to quantization, which means that when EoRA is quantized, the accuracy drop from full-precision EoRA is insignificant while the model size is significantly reduced. For example, when a 512-rank EoRA is quantized from 16-bits to 4-bit on 2:4 pruned LLaMA3-8B, the accuracy drops are only 0.43% on ARC-C while the total model size reduces by 16.49%. Additionally, compared to the original uncompensated 2:4 pruned model, quantizing EoRA of rank 128/512 improves accuracy by 4.4%/11.4% with a total model size increase of just 2%/7%. For 3-bit quantized LLaMA3-8B compensated with a 4-bit quantized EoRA of rank 128/512 achieves 10.6%/19.1% accuracy improvements, with a corresponding model size increase of only 3%/14%. Interestingly, we also observe that quantizing EoRA does not always result in accuracy loss; in some cases, it even slightly improves accuracy, potentially due to quantization acting as a form of regularization, as discussed in Liu et al. [2023]. Generally, we recommend users quantize EoRA to 4-bit, as this significantly reduces inference latency and model size with kernel support, without causing any noticeable drop in accuracy.

## 5. Related Works

**Post-training** **LLM** **Compression:** As LLMs scale, reducing their size is essential for efficient deployment. Traditional compression-aware training methods are impractical due to the need for full datasets and heavy retraining. Post-training compression methods like quantization and pruning have gained popularity as they require only minimal calibration data and no retraining. PTQ reduces model size by lowering bitwidths [Frantar et al., 2023, Tseng et al., 2024], while PTP removes less important weights to reduce computation [Frantar and Alistarh, 2023, Sun et al., 2024]. ODur method, EoRA, is compatible with all such compression techniques as it operates independently of the base method used.

**Low-Rank** **Decomposition:** Low-rank decomposition methods [Yuan et al.,2023,Sakr and Khailany, 2024,Wang et al., 2025, Hsu et al., 2022] compress models by replacing weights with low-rank matrices, reducing both latency and size without special kernel support. However, they are less widely adopted due to weaker accuracy-compression trade-offs. SVD-LLM [Wang et al., 2025] is conceptually close to our work which also tries to align the SVD compression error with the layer-wise compression loss-it relies on the matrix product of the activation being positive-definite, a condition often unmet in practice. Enforcing this condition typically requires additional modifications, which introduce noise into the approximation. In contrast, EoRA employs eigendecomposition, which only requires the matrix product of the activationto be symmetric-a property that naturally holds-avoiding such issues. Furthermore, although EoRA also utilizes SVD-based low-rank decomposition, its core objective is fundamentally different. Whereas prior methods aim to replace pre-trained weight matrices with low-rank approximations to reduce model size and inference cost, EoRA focuses on approximating the compression error itself. This allows for improved accuracy recovery in compressed LLMs and provides greater flexibility in balancing accuracy and computational overhead by overcoming the constraints of fixed compression formats.

## 6.Conclusion

In this work, we present EoRA, a novel fine-tuning-free approach that rapidly boosts the task-specific accuracy of compressed LLMs using minimal calibration data, while offering greater flexibility by relaxing compression format constraints. By projecting compression errors into the task-specific eigenspace of activations, EoRA uses eigenvalues to guide SVD, aligning approximation error with layer-wise compression loss-without any gradient-based training. EoRA achieves strong results across language, commonsense, and math reasoning tasks, outperforming prior low-rank methods Yao et al. [2024],Yuan et al. [2023], Liao et al. [2024]. Its training-free design allows quick adaptation to various accuracy-latency trade-offs, and it remains robust under quantization, reducing memory overhead. Additionally, it can serve as a strong initialization for LoRA fine-tuning. Overall, EoRA is a scalable, efficient solution for improving compressed LLMs across diverse deployment settings,with potential extensions to new architectures and modalities.

<!-- 9 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

# References

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. In International Conference on Learning Representations, 2023.

Ji Lin, Jiaming Tang,Haotian Tang,Shang Yang,Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang,Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. In Machine Learning and Systems, 2024.

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. Spinquant: Llm quantization with learned rotations. In International Conference on Learning Representations, 2025.

Albert Tseng,Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Quip#: Even better llm quantization with hadamard incoherence and lattice codebooks. In International Conference on Machine Learning,2024.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. In Neural Information Processing Systems, 2023.

Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International Conference on Machine Learning, 2023.

Mingjie Sun,Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning approach for large language models. In International Conference on Learning Representations,2024.

The vLLM Team. MultiLoRA Inference. https://docs.vllm.ai/en/stable/gettingstarted/examples/ multilorainference.html,2025.

Yixiao Li,Yifan Yu,Chen Liang, Nikos Karampatziakis, Pengcheng HIe, Weizhu Chen, and Tuo Zhao. Loftq: Lora-fine-tuning-aware quantization for large language models. In International Conference on Learning Representations, 2024.

Zhewei Yao,XiaoxiaWuCheng Li,Stephen Youn, and Yuxiong He. Exploring post-training quantization in llms from comprehensive study to low rank compensation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2024.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. In Neural Information Processing Systems, 2023.

Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdqunat: Absorbing outliers by low-rank components for 4-bit diffusion models. In International Conference on Learning Representations, 2025.

Gilbert W Stewart. Matrix Algorithms: Volume II: Eigensystems. SIAM, 2001.

Ji-Guang Sun. Perturbation bounds for the cholesky and qr factorizations. BIT Numerical Mathematics, 1991.

Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. Svd-llm: Truncation-aware singular value decomposition for large language model compression. In International Conference on Learning Representations, 2025.

Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika,1(3):211-218,1936.

<!-- 10 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan,Edward Yang,Zachary DeVito,Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. In Neural Information Processing Systems Autodiff Workshop, 2017.

Thomas Wolf,Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface's transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771,2019.

Wei Huang,Xudong Ma,Haotong Qin,Xingyu Zheng,Chengtao Lv, Hong Chen, Jie Luo, Xiaojuan Qi, Xianglong Liu, and Michele Magno. How good are low-bit quantized llama3 models? an empirical study. arXiv preprint arXiv:2404.14047,2024.

Zhihang Yuan,Yuzhang Shang, Yue Song, Qiang Wu,YYan Yan, and Guangyu Sun. Asvd: Activation-aware singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821, 2023.

Baohao Liao, Christian Herold, Shahram Khadivi, and Christof Monz. Apiq: Finetuning of 2-bit quantized large language model. In Empirical Methods in Natural Language Processing,2024.

Leo Gao,Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang,Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika,Eric Tang, Anish Thite, Ben Wang,Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 2024. URL https://zenodo.org/records/12608602.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457,2018.

Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. In North American Chapter of the Association for Computational Linguistics, 2019.

Karl Cobbe,Vineet Kosaraju, Mohammad Bavarian, Mark Chen,Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168,2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li,and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research,2020.

Shih-Yang Liu, Zechun Liu, and Kwang-Ting Cheng. Oscillation-free quantization for low-bit vision trans-formers. In International Conference on Machine Learning,2023.

Charbel Sakr and Brucek Khailany. Espace: Dimensionality reduction of activations for model compression. In Neural Information Processing Systems, 2024.

Yen-Chang Hsu, Ting Hua, Sungen Chang, Qian Lou, Yilin Shen, and Hongxia Jin. Language model compression with weighted low-rank factorization. In International Conference on Learning Representations, 2022.

<!-- 11 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

# A. Appendix

## A.1. Sparsity Error Compensation

Table 4 | Perplexity and Commonsense/Math reasoning results of LLaMA2/3 pruned by SparseGPT to 2:4sparsity, with low-rank compensation of rank 128.

<table border="1" ><tr>
<td>Model</td>
<td>Sparsity</td>
<td>Compensation Method</td>
<td>Wikitext2↓</td>
<td>ARC-C↑</td>
<td>MathQA↑</td>
<td>GSM8K↑</td>
</tr><tr>
<td rowspan="6">LLaMA3-8B</td>
<td>-</td>
<td>-</td>
<td>6.13</td>
<td>50.42</td>
<td>40.10</td>
<td>36.23</td>
</tr><tr>
<td rowspan="5">2:4</td>
<td></td>
<td>12.32</td>
<td>30.11</td>
<td>26.43</td>
<td>2.12</td>
</tr><tr>
<td>ZeroQuantV2</td>
<td>11.31</td>
<td>31.99</td>
<td>26.49</td>
<td>2.96</td>
</tr><tr>
<td>Activation</td>
<td>11.32</td>
<td>31.74</td>
<td>26.73</td>
<td>3.26</td>
</tr><tr>
<td>APIQ</td>
<td>11.08</td>
<td>34.21</td>
<td>28.77</td>
<td>14.55</td>
</tr><tr>
<td>EoRA</td>
<td>11.07</td>
<td>34.64</td>
<td>29.91</td>
<td>13.95</td>
</tr><tr>
<td rowspan="6">LLaMA2-7B</td>
<td>-</td>
<td>-</td>
<td>5.47</td>
<td>39.84</td>
<td>27.67</td>
<td>14.85</td>
</tr><tr>
<td rowspan="5">2:4</td>
<td>-</td>
<td>8.77</td>
<td>30.11</td>
<td>24.65</td>
<td>1.66</td>
</tr><tr>
<td>ZeroQuantV2</td>
<td>8.15</td>
<td>30.54</td>
<td>24.89</td>
<td>1.97</td>
</tr><tr>
<td>Activation</td>
<td>8.22</td>
<td>30.20</td>
<td>25.09</td>
<td>2.73</td>
</tr><tr>
<td>APIQ</td>
<td>8.03</td>
<td>32.67</td>
<td>26.36</td>
<td>7.58</td>
</tr><tr>
<td>EoRA</td>
<td>7.97</td>
<td>32.67</td>
<td>25.59</td>
<td>6.22</td>
</tr><tr>
<td rowspan="6">LLaMA2-13B</td>
<td>-</td>
<td>-</td>
<td>4.88</td>
<td>45.56</td>
<td>29.91</td>
<td>21.37</td>
</tr><tr>
<td rowspan="5">2:4</td>
<td>-</td>
<td>7.10</td>
<td>34.30</td>
<td>25.92</td>
<td>2.65</td>
</tr><tr>
<td>ZeroQuantV2</td>
<td>6.82</td>
<td>33.61</td>
<td>25.12</td>
<td>3.56</td>
</tr><tr>
<td>Activation</td>
<td>6.92</td>
<td>34.12</td>
<td>25.69</td>
<td>4.09</td>
</tr><tr>
<td>APIQ</td>
<td>6.80</td>
<td>36.68</td>
<td>27.16</td>
<td>12.13</td>
</tr><tr>
<td>EoRA</td>
<td>6.75</td>
<td>37.54</td>
<td>27.53</td>
<td>10.91</td>
</tr></table>

<!-- 12 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

## A.2. Quantization Error Compensation

Table 5 | Perplexity and Commonsense/Math reasoning results of LLaMA2/3 quantized by GPTQ with different bit-width, with low-rank compensation of rank 128.

<table border="1" ><tr>
<td>Model</td>
<td>W bits</td>
<td>Compensation Method</td>
<td>Wikitext2↓</td>
<td>ARC-C↑</td>
<td>MathQA↑</td>
<td>GSM8K↑</td>
</tr><tr>
<td rowspan="11">LLaMA3-8B</td>
<td>-</td>
<td>-</td>
<td>6.13</td>
<td>50.42</td>
<td>40.10</td>
<td>36.23</td>
</tr><tr>
<td rowspan="5">W4</td>
<td>-</td>
<td>7.00</td>
<td>45.90</td>
<td>34.07</td>
<td>27.74</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>6.80</td>
<td>45.24</td>
<td>36.51</td>
<td>31.23</td>
</tr><tr>
<td>Act-S</td>
<td>6.82</td>
<td>47.86</td>
<td>35.84</td>
<td>29.34</td>
</tr><tr>
<td>ApiQ</td>
<td>6.87</td>
<td>46.58</td>
<td>36.18</td>
<td>30.09</td>
</tr><tr>
<td>EoRA</td>
<td>6.80</td>
<td>47.44</td>
<td>37.21</td>
<td>30.70</td>
</tr><tr>
<td rowspan="5">W3</td>
<td></td>
<td>15.64</td>
<td>20.90</td>
<td>22.37</td>
<td>0.45</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>10.24</td>
<td>30.02</td>
<td>26.43</td>
<td>3.79</td>
</tr><tr>
<td>Act-S</td>
<td>10.19</td>
<td>31.28</td>
<td>25.42</td>
<td>4.09</td>
</tr><tr>
<td>ApiQ</td>
<td>10.41</td>
<td>30.46</td>
<td>26.86</td>
<td>10.79</td>
</tr><tr>
<td>EoRA</td>
<td>10.06</td>
<td>31.74</td>
<td>29.11</td>
<td>11.90</td>
</tr><tr>
<td rowspan="11">LLaMA2-7B</td>
<td>-</td>
<td>-</td>
<td>5.47</td>
<td>39.84</td>
<td>27.67</td>
<td>14.85</td>
</tr><tr>
<td rowspan="5">W4</td>
<td>-</td>
<td>5.75</td>
<td>38.13</td>
<td>26.73</td>
<td>9.93</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>5.68</td>
<td>37.62</td>
<td>27.06</td>
<td>10.15</td>
</tr><tr>
<td>Act-S</td>
<td>5.68</td>
<td>39.84</td>
<td>27.50</td>
<td>9.86</td>
</tr><tr>
<td>ApiQ</td>
<td>5.68</td>
<td>39.59</td>
<td>27.00</td>
<td>11.22</td>
</tr><tr>
<td>EoRA</td>
<td>5.68</td>
<td>38.05</td>
<td>27.13</td>
<td>11.45</td>
</tr><tr>
<td rowspan="5">W3</td>
<td>-</td>
<td>7.76</td>
<td>31.65</td>
<td>23.50</td>
<td>0.38</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>6.84</td>
<td>34.47</td>
<td>23.90</td>
<td>2.04</td>
</tr><tr>
<td>Act-S</td>
<td>6.86</td>
<td>32.67</td>
<td>25.02</td>
<td>2.57</td>
</tr><tr>
<td>ApiQ</td>
<td>6.86</td>
<td>33.70</td>
<td>26.06</td>
<td>7.13</td>
</tr><tr>
<td>EoRA</td>
<td>6.84</td>
<td>35.83</td>
<td>25.79</td>
<td>7.50</td>
</tr><tr>
<td rowspan="11">LLaMA2-13B</td>
<td>-</td>
<td>-</td>
<td>4.88</td>
<td>45.56</td>
<td>29.91</td>
<td>21.37</td>
</tr><tr>
<td rowspan="5">W4</td>
<td>-</td>
<td>5.06</td>
<td>44.28</td>
<td>29.10</td>
<td>21.00</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>5.03</td>
<td>44.19</td>
<td>28.97</td>
<td>19.48</td>
</tr><tr>
<td>Act-S</td>
<td>5.04</td>
<td>43.60</td>
<td>29.48</td>
<td>18.49</td>
</tr><tr>
<td>ApiQ</td>
<td>5.04</td>
<td>42.83</td>
<td>29.64</td>
<td>21.45</td>
</tr><tr>
<td>EoRA</td>
<td>5.03</td>
<td>44.53</td>
<td>28.90</td>
<td>22.36</td>
</tr><tr>
<td rowspan="5">W3</td>
<td>-</td>
<td>5.99</td>
<td>37.28</td>
<td>26.26</td>
<td>4.62</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>5.76</td>
<td>37.54</td>
<td>26.83</td>
<td>9.93</td>
</tr><tr>
<td>Act-S</td>
<td>5.81</td>
<td>38.90</td>
<td>26.26</td>
<td>9.17</td>
</tr><tr>
<td>ApiQ</td>
<td>5.81</td>
<td>39.67</td>
<td>27.47</td>
<td>14.32</td>
</tr><tr>
<td>EoRA</td>
<td>5.75</td>
<td>39.50</td>
<td>27.20</td>
<td>15.08</td>
</tr></table>

<!-- 13 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

## A.3. Sparse & Quantization Error Compensation

Table 6 | Perplexity and Commonsense/Math reasoning results of LLaMA2/3 models pruned to 2:4 using SparseGPT and quantized to 4-bit with GPTQ, with compensation rank set to 128.

<table border="1" ><tr>
<td>Model</td>
<td>Sparsity</td>
<td>W bits</td>
<td>Compensation Method</td>
<td>Wikitext2↓</td>
<td>ARC-C↑</td>
<td>MathQA↑</td>
<td>GSM8K↑</td>
</tr><tr>
<td rowspan="6">LLaMA3-8B</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>6.13</td>
<td>50.42</td>
<td>40.10</td>
<td>36.23</td>
</tr><tr>
<td rowspan="5">2:4</td>
<td rowspan="5">W4</td>
<td>-</td>
<td>86.15</td>
<td>18.34</td>
<td>19.89</td>
<td>0.00</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>12.84</td>
<td>29.35</td>
<td>26.86</td>
<td>1.59</td>
</tr><tr>
<td>Act-S</td>
<td>12.99</td>
<td>27.90</td>
<td>25.59</td>
<td>1.90</td>
</tr><tr>
<td>ApiQ</td>
<td>12.77</td>
<td>30.71</td>
<td>28.74</td>
<td>11.06</td>
</tr><tr>
<td>EoRA</td>
<td>12.60</td>
<td>31.22</td>
<td>29.58</td>
<td>10.16</td>
</tr><tr>
<td rowspan="6">LLaMA2-7B</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>5.47</td>
<td>39.84</td>
<td>27.67</td>
<td>14.85</td>
</tr><tr>
<td rowspan="5">2:4</td>
<td rowspan="5">W4</td>
<td>-</td>
<td>9.37</td>
<td>29.43</td>
<td>23.88</td>
<td>0.99</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>8.42</td>
<td>29.94</td>
<td>24.42</td>
<td>1.67</td>
</tr><tr>
<td>Act-S</td>
<td>8.24</td>
<td>28.92</td>
<td>24.05</td>
<td>1.97</td>
</tr><tr>
<td>ApiQ</td>
<td>8.03</td>
<td>30.63</td>
<td>24.12</td>
<td>7.05</td>
</tr><tr>
<td>EoRA</td>
<td>8.24</td>
<td>31.14</td>
<td>25.39</td>
<td>4.93</td>
</tr><tr>
<td rowspan="6">LLaMA2-13B</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>4.88</td>
<td>45.56</td>
<td>29.91</td>
<td>21.37</td>
</tr><tr>
<td rowspan="5">2:4</td>
<td rowspan="5">W4</td>
<td>-</td>
<td>7.27</td>
<td>33.10</td>
<td>24.75</td>
<td>2.20</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>6.98</td>
<td>33.27</td>
<td>25.29</td>
<td>2.65</td>
</tr><tr>
<td>Act-S</td>
<td>6.92</td>
<td>34.64</td>
<td>26.09</td>
<td>2.81</td>
</tr><tr>
<td>ApiQ</td>
<td>6.80</td>
<td>36.17</td>
<td>26.96</td>
<td>12.59</td>
</tr><tr>
<td>EoRA</td>
<td>6.89</td>
<td>35.06</td>
<td>27.06</td>
<td>9.86</td>
</tr></table>

Here, we examine the feasibility of applying EoRA to compensate for ultra-compressed models that undergo both pruning and quantization. Specifically, we prune LLaMA2-7B/13B and LLaMA3-8B to 2:4sparsity and quantize them to 4-bit. We set the ranks of both EoRA and SVD to 128 to compensate for the pruning and quantization errors. Similarly to our previous findings,LLaMA3-8B is the least resilient to compression, experiencing a significant drop in both perplexity for language generation and accuracy on commonsense and math reasoning tasks. Notably, the accuracy on ARC-C plummets to 18.33% and MathQA to 19.89%, which is worse than random guessing. However, compensating for the sparsity and quantization errors with EoRA significantly improves the accuracy of these compressed models, reducing perplexity by up to 73.55 and boosting accuracy by 12.88%/9.60%/10.16% on ARC-C/MathQA/GSM8K tasks. Additionally, EoRA consistently outperforms both ZeroQuant-V2 and Act-S across LLaMA2 and LLaMA3. For instance, EoRA exceeds ZeroQuant-V2 in compensating the compressed LLaMA2-13B on ARC-C by 1.79% and on MathQA by 1.77%, narrowing the accuracy gap wvith the uncompressed model to just 2.85% on MathQA. Overall, we find that EoRA tends to offer greater accuracy recovery when addressing more aggressive compression settings, ensuring the plausibility of adopting EoRA for mitigating severe compression error.

<!-- 14 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

## A.4. Compatibility With Various Compression Methods

Table 7| Comparison between compensation methods of rank set to 128 on compensating LLaMA3-8B models pruned to 2:4 sparsity with Wanda on Perplexity and Commonsense/Math reasoning tasks.

<table border="1" ><tr>
<td>Compression Method</td>
<td>Compression Setting</td>
<td>Compensation Method</td>
<td>Wikitext2↓</td>
<td>ARC-C↑</td>
<td>MathQA↑</td>
<td>GSM8K↑</td>
</tr><tr>
<td>Full-precision</td>
<td></td>
<td>-</td>
<td>6.13</td>
<td>50.42</td>
<td>40.10</td>
<td>36.23</td>
</tr><tr>
<td rowspan="5">Wanda</td>
<td rowspan="5">2:4</td>
<td>-</td>
<td>21.42</td>
<td>27.04</td>
<td>25.09</td>
<td>0.76</td>
</tr><tr>
<td>ZeroQuant-V2</td>
<td>17.16</td>
<td>30.46</td>
<td>26.16</td>
<td>1.28</td>
</tr><tr>
<td>Act-S</td>
<td>17.37</td>
<td>29.77</td>
<td>26.73</td>
<td>1.51</td>
</tr><tr>
<td>ApiQ</td>
<td>14.30</td>
<td>31.91</td>
<td>29.61</td>
<td>12.81</td>
</tr><tr>
<td>EoRA</td>
<td>14.04</td>
<td>34.81</td>
<td>30.05</td>
<td>11.52</td>
</tr></table>

In this section studies the generalizability and compatibility of EoRA with different pruning methods beyond SparseGPT. We adopt Wanda [Sun et al., 2024], a method that prunes weights with the smallest magnitudes scaled by their corresponding input activations. For these compression methods, we adhere to the calibration set construction detailed in 4.1, and maintain the same settings when utilizing EoRA to address compression errors. We evaluate EoRAon LLaMA3-8B pruned with Wanda to 2:4 structured sparsity. The ranks of all low-rank compensation methods are set to 128. Table 7 demonstrates that EoRAconsistently outperforms every training-free methods, both ZeroQuant-V2 and Act-S, in improving accuracy across all the tasks. For example, EoRA achieves accuracy gains of 7.77%/4.96%/10.76% on ARC-C/MathQA/GSM8K which is 5.04%/3.32%/10.01% over the improvement brought by Act-s. Furthermore, EoRA outperforms ApiQ on both ARC-C and MathQA by 2.9% and 0.44%. Overall, these findings underscore the effectiveness and generalizability of EoRA across different compression techniques.

<!-- 15 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

## A.5. Compensation With Different Ranks

Table 8 | Results of EoRA of different rank on compensating LLaMA2/3 models pruned to 2:4 sparsity by SparseGPT on Commonsense and Math reasoning tasks.

<table border="1" ><tr>
<td>Model</td>
<td>Sparsity</td>
<td>r</td>
<td>Compensation Method</td>
<td>ARC-C↑</td>
<td>MathQA↑</td>
<td>GSM8K↑</td>
</tr><tr>
<td rowspan="18">LLaMA3-8B</td>
<td></td>
<td></td>
<td></td>
<td>50.42</td>
<td>40.10</td>
<td>36.23</td>
</tr><tr>
<td rowspan="17">2:4</td>
<td></td>
<td></td>
<td>30.11</td>
<td>26.43</td>
<td>2.12</td>
</tr><tr>
<td rowspan="4">64</td>
<td>ZeroQuant-V2</td>
<td>30.97</td>
<td>26.39</td>
<td>2.27</td>
</tr><tr>
<td>Act-S</td>
<td>30.46</td>
<td>26.67</td>
<td>3.34</td>
</tr><tr>
<td>ApiQ</td>
<td>33.10</td>
<td>27.87</td>
<td>11.52</td>
</tr><tr>
<td>EoRA</td>
<td>33.10</td>
<td>28.57</td>
<td>10.77</td>
</tr><tr>
<td rowspan="4">128</td>
<td>ZeroQuant-V2</td>
<td>31.99</td>
<td>26.49</td>
<td>2.96</td>
</tr><tr>
<td>Act-S</td>
<td>31.74</td>
<td>26.73</td>
<td>3.26</td>
</tr><tr>
<td>ApiQ</td>
<td>34.21</td>
<td>28.77</td>
<td>14.55</td>
</tr><tr>
<td>EoRA</td>
<td>34.64</td>
<td>29.91</td>
<td>13.95</td>
</tr><tr>
<td rowspan="4">256</td>
<td>ZeroQuant-V2</td>
<td>34.55</td>
<td>28.74</td>
<td>4.09</td>
</tr><tr>
<td>Act-S</td>
<td>32.76</td>
<td>27.94</td>
<td>5.16</td>
</tr><tr>
<td>ApiQ</td>
<td>35.41</td>
<td>30.45</td>
<td>15.85</td>
</tr><tr>
<td>EoRA</td>
<td>37.96</td>
<td>31.59</td>
<td>17.06</td>
</tr><tr>
<td rowspan="4">512</td>
<td>ZeroQuant-V2</td>
<td>38.73</td>
<td>30.38</td>
<td>6.75</td>
</tr><tr>
<td>Act-S</td>
<td>36.18</td>
<td>29.65</td>
<td>8.64</td>
</tr><tr>
<td>ApiQ</td>
<td>36.69</td>
<td>32.63</td>
<td>20.77</td>
</tr><tr>
<td>EoRA</td>
<td>41.89</td>
<td>34.17</td>
<td>23.28</td>
</tr><tr>
<td rowspan="18">LLaMA2-7B</td>
<td></td>
<td></td>
<td></td>
<td>39.84</td>
<td>27.67</td>
<td>14.85</td>
</tr><tr>
<td rowspan="17">2:4</td>
<td></td>
<td>-</td>
<td>30.11</td>
<td>24.65</td>
<td>1.66</td>
</tr><tr>
<td rowspan="4">64</td>
<td>ZeroQuant-V2</td>
<td>30.20</td>
<td>24.48</td>
<td>1.97</td>
</tr><tr>
<td>Act-S</td>
<td>30.12</td>
<td>25.03</td>
<td>1.74</td>
</tr><tr>
<td>ApiQ</td>
<td>31.83</td>
<td>25.62</td>
<td>5.91</td>
</tr><tr>
<td>EoRA</td>
<td>32.16</td>
<td>25.62</td>
<td>5.08</td>
</tr><tr>
<td rowspan="4">128</td>
<td>ZeroQuant-V2</td>
<td>30.54</td>
<td>24.89</td>
<td>1.97</td>
</tr><tr>
<td>Act-S</td>
<td>30.20</td>
<td>25.09</td>
<td>2.73</td>
</tr><tr>
<td>ApiQ</td>
<td>32.67</td>
<td>26.36</td>
<td>7.58</td>
</tr><tr>
<td>EoRA</td>
<td>32.67</td>
<td>25.59</td>
<td>6.22</td>
</tr><tr>
<td rowspan="4">256</td>
<td>ZeroQuant-V2</td>
<td>31.99</td>
<td>25.19</td>
<td>2.88</td>
</tr><tr>
<td>Act-S</td>
<td>32.59</td>
<td>25.39</td>
<td>3.26</td>
</tr><tr>
<td>ApiQ</td>
<td>34.30</td>
<td>25.99</td>
<td>8.79</td>
</tr><tr>
<td>EoRA</td>
<td>34.47</td>
<td>26.06</td>
<td>7.88</td>
</tr><tr>
<td rowspan="4">512</td>
<td>ZeroQuant-V2</td>
<td>34.72</td>
<td>24.38</td>
<td>3.34</td>
</tr><tr>
<td>Act-S</td>
<td>34.73</td>
<td>25.76</td>
<td>3.56</td>
</tr><tr>
<td>ApiQ</td>
<td>34.98</td>
<td>26.16</td>
<td>9.70</td>
</tr><tr>
<td>EoRA</td>
<td>36.77</td>
<td>25.96</td>
<td>8.79</td>
</tr><tr>
<td rowspan="18">LLaMA2-13B</td>
<td></td>
<td></td>
<td>-</td>
<td>45.56</td>
<td>29.91</td>
<td>21.37</td>
</tr><tr>
<td rowspan="17">2:4</td>
<td></td>
<td></td>
<td>34.30</td>
<td>25.92</td>
<td>2.65</td>
</tr><tr>
<td rowspan="4">64</td>
<td>ZeroQuant-V2</td>
<td>33.95</td>
<td>25.56</td>
<td>2.81</td>
</tr><tr>
<td>Act-S</td>
<td>32.76</td>
<td>25.93</td>
<td>2.96</td>
</tr><tr>
<td>ApiQ</td>
<td>35.84</td>
<td>27.17</td>
<td>8.64</td>
</tr><tr>
<td>EoRA</td>
<td>36.00</td>
<td>26.80</td>
<td>8.19</td>
</tr><tr>
<td rowspan="4">128</td>
<td>ZeroQuant-V2</td>
<td>33.61</td>
<td>25.12</td>
<td>3.56</td>
</tr><tr>
<td>Act-S</td>
<td>34.12</td>
<td>25.69</td>
<td>4.09</td>
</tr><tr>
<td>ApiQ</td>
<td>36.68</td>
<td>27.16</td>
<td>12.13</td>
</tr><tr>
<td>EoRA</td>
<td>37.54</td>
<td>27.53</td>
<td>10.91</td>
</tr><tr>
<td rowspan="4">256</td>
<td>ZeroQuant-V2</td>
<td>35.06</td>
<td>26.06</td>
<td>4.93</td>
</tr><tr>
<td>Act-S</td>
<td>34.56</td>
<td>26.23</td>
<td>4.62</td>
</tr><tr>
<td>ApiQ</td>
<td>36.69</td>
<td>27.40</td>
<td>14.56</td>
</tr><tr>
<td>EoRA</td>
<td>38.73</td>
<td>27.77</td>
<td>13.04</td>
</tr><tr>
<td rowspan="4">512</td>
<td>ZeroQuant-V2</td>
<td>36.51</td>
<td>26.39</td>
<td>7.28</td>
</tr><tr>
<td>Act-S</td>
<td>36.86</td>
<td>26.77</td>
<td>6.14</td>
</tr><tr>
<td>ApiQ</td>
<td>38.57</td>
<td>27.71</td>
<td>17.21</td>
</tr><tr>
<td>EoRA</td>
<td>40.61</td>
<td>29.17</td>
<td>17.51</td>
</tr></table>

<!-- 16 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

## A.6. Influence of Different Calibration sizes on EoRA

Table 9 | Ablation studies of calibrating EoRA with different calibration sizes on compensating compressed LLaMA3-8B.

<table border="1" ><tr>
<td>Model</td>
<td>Compression Format</td>
<td>#Calib</td>
<td>MathQA↑</td>
</tr><tr>
<td rowspan="9">LLaMA3-8B</td>
<td rowspan="3">W4</td>
<td>16</td>
<td>36.62</td>
</tr><tr>
<td>32</td>
<td>36.93</td>
</tr><tr>
<td>64</td>
<td>37.21</td>
</tr><tr>
<td rowspan="3">W3</td>
<td>16</td>
<td>26.33</td>
</tr><tr>
<td>32</td>
<td>27.57</td>
</tr><tr>
<td>64</td>
<td>29.11</td>
</tr><tr>
<td rowspan="3">2:4</td>
<td>16</td>
<td>30.05</td>
</tr><tr>
<td>32</td>
<td>30.18</td>
</tr><tr>
<td>64</td>
<td>29.91</td>
</tr></table>

<!-- 17 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

## A.7. Fine-tuning Compressed Models with EoRA

Table 10 | Finetune the compressed LLaMA3-8B models with various compression settings and different initialization of the low-rank matrices for Commonsense/Math reasoning tasks.

<table border="1" ><tr>
<td>Model</td>
<td>Compression Method</td>
<td>Compression Setting</td>
<td>LoRA initialization</td>
<td>ARC-C↑</td>
<td>MathQA↑</td>
</tr><tr>
<td rowspan="14">LLaMA3-8B</td>
<td rowspan="2">Full-precision</td>
<td rowspan="2">-</td>
<td>w/o finetuning</td>
<td>50.42</td>
<td>40.10</td>
</tr><tr>
<td>Standard</td>
<td>56.39</td>
<td>53.56</td>
</tr><tr>
<td rowspan="4">SparseGPT</td>
<td rowspan="4">2:4</td>
<td>w/o finetuning</td>
<td>30.11</td>
<td>26.43</td>
</tr><tr>
<td>QLoRA</td>
<td>41.30</td>
<td>45.42</td>
</tr><tr>
<td>LoftQ</td>
<td>43.68</td>
<td>48.77</td>
</tr><tr>
<td>EoRA</td>
<td>48.54</td>
<td>54.67</td>
</tr><tr>
<td rowspan="4">GPTQ</td>
<td rowspan="4">W4</td>
<td>w/o finetuning</td>
<td>45.90</td>
<td>34.07</td>
</tr><tr>
<td>QLoRA</td>
<td>54.09</td>
<td>51.42</td>
</tr><tr>
<td>LoftQ</td>
<td>54.52</td>
<td>53.96</td>
</tr><tr>
<td>EoRA</td>
<td>55.46</td>
<td>56.04</td>
</tr><tr>
<td rowspan="4">GPTQ</td>
<td rowspan="4">W3</td>
<td>w/o finetuning</td>
<td>20.90</td>
<td>22.37</td>
</tr><tr>
<td>QLoRA</td>
<td>30.29</td>
<td>34.10</td>
</tr><tr>
<td>LoftQ</td>
<td>44.70</td>
<td>48.17</td>
</tr><tr>
<td>EoRA</td>
<td>47.44</td>
<td>53.90</td>
</tr></table>

<!-- 18 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

## A.8. Inference Speed Evaluation

Table 11 | Comparison of the average per-token latency (batch size 1) for 128-token generation on LLaMA3-70B between full-precision and GPTQ + EoRA (rank=128) with and without our custom EoRA kernel.

<table border="1" ><tr>
<td>Format</td>
<td>EoRA Kernel</td>
<td>Latency</td>
<td>Speedup</td>
</tr><tr>
<td>FP-16</td>
<td>-</td>
<td>60ms</td>
<td>1x</td>
</tr><tr>
<td rowspan="3">3-bit</td>
<td>-</td>
<td>35ms</td>
<td>1.7x</td>
</tr><tr>
<td>No</td>
<td>54ms</td>
<td>1.1x</td>
</tr><tr>
<td>Yes</td>
<td>43ms</td>
<td>1.4x</td>
</tr><tr>
<td rowspan="3">4-bit</td>
<td>-</td>
<td>38ms</td>
<td>1.6x</td>
</tr><tr>
<td>No</td>
<td>61ms</td>
<td>0.9x</td>
</tr><tr>
<td>Yes</td>
<td>51ms</td>
<td>1.2x</td>
</tr></table>

In language generation, the model produces tokens sequentially, making matrix-vector multiplications the primary factor impacting the inference latency. Consequently, we build our custom EoRA kernel on top of GPTQ's low-bit quantized matrix vector product kernel, pre-allocating the shared output prior to matrix vector multiplication and integrating the full-precision matrix vector multiplication of B into the quantized kernel reducing redundant memory access. We show the inference speedup of our proposed EoRA kernel in the above Table.11.

<!-- 19 -->

<!-- EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation -->

## A.9. Quantizing EoRA To Further Reduce Memory Overhead

Table 12 | Accuracy and the Model Size of quantizing EoRA of rank {128,512} to 4/3-bit on compensating LLaMA3-8B of {2:4 sparisity,4/3-bit}.

<table border="1" ><tr>
<td>Compression method</td>
<td>Config</td>
<td>r</td>
<td>W-bit of EoRA</td>
<td>Model Size (GB)</td>
<td>ARC-C↑</td>
<td>MathQA↑</td>
</tr><tr>
<td></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>15.08</td>
<td>50.42</td>
<td>40.10</td>
</tr><tr>
<td rowspan="7">SparseGPT</td>
<td rowspan="7">2:4</td>
<td>-</td>
<td>-</td>
<td>9.12</td>
<td>30.11</td>
<td>26.43</td>
</tr><tr>
<td rowspan="3">128</td>
<td>16</td>
<td>9.77</td>
<td>34.64</td>
<td>29.91</td>
</tr><tr>
<td>4</td>
<td>9.28</td>
<td>34.47</td>
<td>29.91</td>
</tr><tr>
<td>3</td>
<td>9.24</td>
<td>34.72</td>
<td>29.71</td>
</tr><tr>
<td rowspan="3">512</td>
<td>16</td>
<td>11.70</td>
<td>41.89</td>
<td>34.17</td>
</tr><tr>
<td>4</td>
<td>9.77</td>
<td>41.46</td>
<td>33.63</td>
</tr><tr>
<td>3</td>
<td>9.64</td>
<td>40.35</td>
<td>32.66</td>
</tr><tr>
<td rowspan="14">GPTQ</td>
<td rowspan="7">W4</td>
<td>-</td>
<td>-</td>
<td>5.35</td>
<td>45.90</td>
<td>34.07</td>
</tr><tr>
<td rowspan="3">128</td>
<td>16</td>
<td>6.01</td>
<td>47.44</td>
<td>37.21</td>
</tr><tr>
<td>4</td>
<td>5.50</td>
<td>47.35</td>
<td>36.78</td>
</tr><tr>
<td>3</td>
<td>5.46</td>
<td>47.18</td>
<td>36.52</td>
</tr><tr>
<td rowspan="3">512</td>
<td>16</td>
<td>7.85</td>
<td>48.29</td>
<td>38.72</td>
</tr><tr>
<td>4</td>
<td>6.01</td>
<td>48.80</td>
<td>38.92</td>
</tr><tr>
<td>3</td>
<td>5.90</td>
<td>46.92</td>
<td>36.88</td>
</tr><tr>
<td rowspan="7">W3</td>
<td>-</td>
<td>-</td>
<td>4.63</td>
<td>20.90</td>
<td>22.37</td>
</tr><tr>
<td rowspan="3">128</td>
<td>16</td>
<td>5.28</td>
<td>31.74</td>
<td>29.11</td>
</tr><tr>
<td>4</td>
<td>4.78</td>
<td>31.48</td>
<td>28.64</td>
</tr><tr>
<td>3</td>
<td>4.74</td>
<td>29.18</td>
<td>26.7</td>
</tr><tr>
<td rowspan="3">512</td>
<td>16</td>
<td>7.16</td>
<td>38.82</td>
<td>31.89</td>
</tr><tr>
<td>4</td>
<td>5.28</td>
<td>40.01</td>
<td>31.69</td>
</tr><tr>
<td>3</td>
<td>5.18</td>
<td>35.4</td>
<td>30.45</td>
</tr></table>

<!-- 20 -->

