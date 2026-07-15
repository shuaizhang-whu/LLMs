# SVD-LLM V2: Optimizing Singular Value Truncation for

# Large Language Model Compression

**Xin Wang**

**Samiul Alam**

**Zhongwei Wan**

**Hui Shen**

**Mi Zhang**

The Ohio State University

{wang.15980, alam.140, wan.512, shen.1780, mizhang.1}@osu.edu

https://github.com/AIoT-MLSys-Lab/SVD-LLM

## Abstract

Despite significant advancements, the practical deployment of Large Language Models (LLMs) is often hampered by their immense sizes, high-lighting the need for effective compression tech-niques. Singular Value Decomposition (SVD) is a promising LLM compression technique. However, existing SVD-based compression methods fall short in reducing truncation losses, leading to less competitive performance in com-pressed models. In this work, we introduce SVD-LLM V2,a SVD-based LLM compression method that optimizes singular value trunca-tion in SVD compression with two techniques. First, SVD-LLM V2 proposes to use theoretical truncation loss of weight matrices to assign a unique compression ratio to each weight ma-trix at different layers to accommodate weight redundancy heterogeneity. Second, SVD-LLM V2 proposes loss-optimized weight truncation to ensure that the truncated singular values re-sult in a lower and more stable truncation loss in practice. We evaluate SVD-LLM V2 on ten datasets and five LLMs at various scales. Our results show SVD-LLM V2 outperforms state-of-the-art SVD-based LLM compression meth-ods. Our code is available at https://github. com/AIoT-MLSys-Lab/SVD-LLM.

## 1 Introduction

Despite the outstanding performance Large Language Models (LLMs) exhibit in various tasks (Zhao et al., 2023; Gozalo-Brizuela and Garrido-Merchán, 2023; Wan et al.,2024b;Shen et al., 2024; Wan et al., 2025), the significant re-sources consumed limit their widespread acces-sibility (Wan et al., 2024a; Wang et al., 2024a; Zhou et al., 2024). Model compression (Zhu et al., 2023; Shen et al., 2025) is one effective approach to reduce resource consumption. To avoid resource-intensive retraining,LLM compression is often con-ducted in a post-training manner. Techniques such as LLM quantization (Yuan et al., 2024;Huang

<table border="1" ><tr>
<td></td>
<td colspan="2">Compression Ratio Truncation Lo</td>
<td>ss PPL</td>
</tr><tr>
<td>SVD-LLM</td>
<td>Homogeneous<br><img src="https://web-api.textin.com/ocr_image/external/1a32caa4cc4d4269.jpg"></td>
<td>L=0.8961</td>
<td>P=11.8</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>Heterogeneous</td>
<td>L=0.7351</td>
<td>P=8.01</td>
</tr></table>

Figure 1: Comparison between SVD-LLM V2 and SVD-LLM. We randomly select a weight matrix from LLaMA-3 8B and compare the normalized truncation loss and perplexity (PPL) under 20% compression ratio.

et al., 2024), unstructured pruning (Frantar and Al-istarh, 2023), and structured pruning (Ma et al., 2023;Ashkboos et al., 2024; Zhong et al.,2024) have been proposed.

Low-rank approximation, such as Singular Value Decomposition (SVD) is also an effective tech-nique for compressing LLMs. Compared with quantization and unstructured pruning, SVD com-pression is more hardware-friendly. Recently,a few SVD-based LLM compression methods have been proposed. At a high level, these methods all focus on reducing the truncation loss during SVD compression to reserve accuracy.Specifically, FWSVD (Hsu et al., 2022) reduces truncation loss by estimating weight importance and preserving more important weights. ASVD (Yuan et al., 2023) injects a scaling matrix to reduce the truncation loss but was not able to achieve theoretical min-imum truncation loss at each LLM layer. SVD-LLM (Wang et al., 2024b), on the other hand,fills this gap by proposing a whitening matrix that ob-tains theoretical minimum truncation loss at each LLM layer, demonstrating superior performance.

Despite such advantage, SVD-LLM has two limitations. First, SVD-LLM applies a homoge-neous compression ratio to all the weight matrices. This coarse-grained setup unfortunately overlooks the heterogeneity of weight redundancy across dif-ferent LLM layers. Second, SVD-LLM utilizes Cholesky decomposition for weight truncation. However, Cholesky decomposition requires the ma-trix being decomposed to be positive-definite, a condition that is challenging to fulfill in practice. Moreover, Cholesky decomposition introduces nu-merical instability throughout its iterative process. As a consequence, SVD-LLM could still lead to high truncation loss in practice.

In this paper, we propose SVD-LLM V2, a SVD-based post-training LLM compression method that effectively addresses the two limitations of SVD-**LLM.** First, to address the heterogeneity of weight redundancy across layers, SVD-LLM V2 uses the the-oretical truncation loss of weight matrices at each layer as the guidance to assign a unique compres-sion ratio to each weight matrix based on its type at different layers. Second, SVD-LLM V2 substi-tutes the Cholesky decomposition with two rounds of SVD for weight truncation, which we prove to achieve the theoretical minimum truncation un-der the optimized compression ratio. In doing so, SVD-LLM V2 is able to achieve better perplexity with lower truncation loss than SVD-LLM (Figure 1).

We evaluate SVD-LLM V2 on ten datasets cover-ing various language modeling, classification, and generation tasks as well as five LLMs with various backbones and scales. Our results demnonstrate the superiority of SVD-LLM V2 with three key findings:

·SVD-LLM V2 consistently outperforms state-of-the-art SVD-based LLM compression meth-ods across all ten datasets and five LLMs.

·SVD-LLM V2 outperforms state-of-the-art struc-tured pruning-based LLM compression meth-ods with up to 28% lower perplexity under 7 GB memory budget. When comparing to state-of-the-art 1-bit quantization-based LLM compression methods, SVD-LLM V2 outper-forms PB-LLM and achieves 5% lower per-plexity. Moreover, by combining with 2-bit quantization, SVD-LLM V2 is able to outper-form 1-bit BiLLM, demonstrating the promise of combining SVD and quantization-based methods for advancing the frontier of post-training LLM compression.

·LLMs compressed by SVD-LLM V2 achieve in-ference speedup on real hardware. In partic-ular,LLMs compressed by SVD-LLM V2 are able to achieve a throughput speedup of up to 2.71×cormpared to the original LLMs on a single NVIDIA A100 GPU.

## 2 Related Work

### 2.1 Laarge Language Model Compression

Large Language Models (LLMs) typically contain billions of parameters, making traditional model compression techniques impractical due to the need for resource-intensive retraining. To address this, post-training methods that bypass retraining during compression have been developed. These meth-ods generally fall into four categories: unstructured pruning, structured pruning, quantization, and low-rank approximation. Unstructured pruning (Fran-tar and Alistarh, 2023) sets the individual weight values to zero without changing the overall archi-tecture. However, its irregular sparsity is feasible only for speedups or memory savings on certain hardware. In contrast, structured pruning (Ma et al., 2023;Ashkboos et al., 2024; Zhong et al., 2024) removes entire channels from LLMs, simplifying hardware implementation but often suffering from accuracy degradation. Quantization (Frantar et al., 2022; Zhao et al., 2024) reduces the precision of the weight matrices for compression. However, it often fails to provide the desired inference speedups (Lin et al., 2024b) and offers a limited range of compres-sion options-typically between 2 to 8 bits-which hinders optima1 memory utilization. Recent ef-forts (Yuan et al.,2024;Huang et al., 2024) have explored 1-bit post-training quantization. Never-theless, these approaches still suffer from accuracy drop, indicating that 1-bit quantization is still chal-lenging in LLM compression.

### 2.2 SVD for LLM Compression

Singular Value Decomposition (SVD) reduces ma-trix sizes by truncating the smallest singular values. It then constructs two smaller, lower-rank matrices to approximate the original matrix (Golub et al., 1987). SVD is also feasible for LLM (Hsu et al., 2022;Yuan et al., 2023; Wang et al., 2024b; Lin et al., 2024a). To ensure better compression per-formance, existing post-training SVD-based LLM compression methods attempt to lower the trun-cation loss L in the form of Frobenius norm as follows during LLM compression:

$$L=\left\|WX-W^{\prime }X\right\|_{F}\tag{1}$$

where W is the weight matrix of the original LLLM, X is the activation of W,and W′is the com-pressed low-ranking weight matrix. For example, Yuan et al. (2023) propose ASVD, which scales the weight matrix using a diagonal matrix to normalize

<!-- Heterogeneous Compression Ratio Allocation ② Loss-optimized Weight Truncation Original LLM (Q1,Q2,⋯,QN) (L1,L2,⋯,LN) (R1,R2,⋯,RN) M W XXT us $\sqrt {s_{s}}$ uws Compressed Layer 1 LLM (K1,\;K2,⋯,\;KN) (L1,L2,⋯,LN) (R1,R2,⋯,RN) Layer 1 Layer 2 Compute Trunc.(sws) Group Allocate --. Theoretical ... --. SVD Truncation W′ Layer 2 Ratio SVD 89 ... (G1,G2,⋯,GN) Loss (L1,L2,⋯,LN) (R1,R2,⋯,Rn) Layer N us ss   LayerN (U1,U2,⋯,UN) (L1,L2,⋯,LN) (R1,R2,⋯,RN) Us uws sws us-1 -->
![](https://web-api.textin.com/ocr_image/external/d2c07e319f9e48be.jpg)

Figure 2: Overview of SVD-LLM V2.

the impact of input channels on the weights to re-duce the truncation loss. Wang et al. (2024b) make further advancement by whitening the input matrix to mitigate its impact on SVD truncation with the guarantee of minimal theoretical truncation loss. Despite these progresses, existing methods stil1 suf-fer from high truncation loss in practice,leading to accuracy degradation.

## 3 SVD-LLM V2

Figure 2 provides an overview of SVD-LLM V2. Specifically, SVD-LLM V2 groups the weight ma-trices across all the layers in the original LLM by type,such as query (Q) and key (K) in attention blocks, and Gate (G) and Up(U) in MLP blocks. It then computes the theoretical truncation loss of the weight matrices and assigns a unique compression ratio to each weight matrix within each group based on the computed truncation loss. Lastly, SVD-LLM V2 performs loss-optimized weight truncation to obtain the compressed LLM. Below, we describe the details of the two main components of SVD-LLM V2: (1) heterogeneous compression ratio allocation and (2) loss-optimized weight truncation.

### 3.1 Heterogeneous Compression Ratio Allocation

**Motivation:** Since different weight matrices in LLMs often exhibit different levels of redundancy, applying a homogeneous compression ratio to all the weight matrices would incur high truncation loss for those with low redundancy (Zhong et al., 2024; He et al., 2024; Li et al., 2024). To demon-strate this, we use SVD-LLM to measure the trun-cation loss of the query matrix across different lay-ers in LLaMA-3 8B on WikiText-2 dataset with 50% compression ratio. As shown in Figure 3, the truncation loss varies at different layers. For exam-ple, the query matrix in the 27th layer has a mmuch higher truncation loss than that of the first layer,in-dicating the 27th layer should be compressed under a smaller compression ratio, while a larger com-

<!-- x⁷ ×107 7 SVD-LLM Sso7 uoIeCun4 5 SVD-LLM V2 3 1 5 10 15 20 25 30 Layer Index -->
![](https://web-api.textin.com/ocr_image/external/17e117f4b555e2f9.jpg)

Figure 3: Comparison between SVD-LLM and SVD-LLM V2 on the truncation loss of the query weight matrix across different layers in LlaMA-3 8B on WikiText-2dataset with 50% compression ratio.

#### Algorithm 1 Pseudocode of Heterogeneous Compression Ra-tio Allocation in SVD-LLM V2

**Input**: M: Original LLLM

x: Input activation

R:Target compression ratio

**Output**:Rd:A list of allocated compression ratios

1: **procedure** RATIOALLOCATION(M,S,R)

2:

$$G\leftarrow \text {Group}(M)$$

Group the weight by types

3:

$$R_{d}\leftarrow \emptyset$$

Initialize the compression ratio list

4:

**for** g **in** G do

5:

$$L_{G}\leftarrow \emptyset$$

Initialize the loss list in the group

6:

**for** w in g do

7:

Lmin← TheoreticalLoss(w,x,R)

8:

$$L_{G}\leftarrow L_{G}\cup L_{\min }$$

9:

**end for**

10:

$$L_{G}\leftarrow 1/\mathbf {Log}\left(L_{G}\right)$$

Normalize LG

11:

12:

$$r\leftarrow \text {Ln}\left(L_{G}\right)xRxL_{\min }/\text {Sum}\left(L_{G}\right)$$

$$R_{d}\leftarrow R_{d}\cup r$$

Append r to the list Rd

13:

**end for**

14:

**return**Rd

15**: end procedure**

pression ratio should be applied to the first layer. However, existing SVD-based LLM compression methods either overlook this variation or require resource-intensive operations to determine the spe-cific compression ratios, making them impractical for compressing LLMs at larger scales. Therefore, it is essential to develop a more efficient approach to apply different compression ratios at different weight matrices to reduce the truncation loss.

**Key** **Design:** The pseudocode of the heteroge-neous compression ratio allocation is listed in Algo-rithm 1. Specifically, given that different types of weight matrices, such as query (Q) and key (K) in attention blocks, and Gate (G) and Up(U) in MLP blocks play different roles in an LLM, SVD-LLM V2 first groups the weight matrices across all the layers in the origina LLM according to their types. Next, SVD-LLM V2 computes the theoretical mini-mum truncation loss of the weight matrices,i.e., Lmin=\|C-C′\|F,where C is the original ma-trix of WX andC′is its compressed version by SVD,respectively. It then inverses and normalizes Lmin by 1/log(Lmin).FFinally, given the target model compression ratio R, the compression ratio of each weight matrix within a group is determined asLen(LG)xRxLmin/Sum(LG),where LG denotes the list of theoretical truncation lossfor all matrices within the same group, Len(LG) de-notes the group size and Sum(Lmin) denotes the sum of the loss within this group. In this way, SVD-LLM V2 bypasses the need to measure end-to-end perplexity to determine compression ratios, as done in ASVD and is time-consuming. Instead, it utilizes truncation loss, which is easy to obtain and thereby enhancing the efficiency of the algo-rithm. As shown in Figure 3, with the proposed het-erogeneous compression ratio allocation scheme, SVD-LLM V2 effectively reduces the truncation loss (the blue area) with only a small increase of several small truncation losses (the yellow area).

In the next section, we describe the details of the proposed loss-optimized weight truncation.

### 3.2 Loss-optimized Weight Truncation

**Motivation:** After determining the specific com-pression ratio for each weight matrix in the LLM, the next step is to truncate the weights accord-ing to their assigned compression ratios. To re-duce truncation lossL=\|WX-W′X\|Fdur-ing SVD compression, SVD-LLM first constructs the whitening matrix S by applying Cholesky de-composition onXXT. It then performs SVD and truncation on WS. Although SVD-LLM has been theoretically proven to achieve the lowest trunca-tion loss at a given compression ratio, our empir-ical study shows that its actual truncation loss is frequently above the theoretical minimum. This is mainly due to the numerical instability involved in performing the Cholesky decomposition on a large-scale matrix during truncation. Moreover, the Cholesky decomposition requiresXXTto be positive definite, which is often hard to satisfy.

To demonstrate this, we randomly select two weight matrices,A and B, in LLaMA-3 8B and

#### Algorithm 2 Pseudocode of Loss-optimized Weight Trunca-tion in SVD-LLM V2

**Input**: W: Original weight matrix

X: Input activation

R: Target compression ratio

**Output:**W′:Compressed weight matrix

1: **procedure** WEIGHTTRUNCATION(W,X,R)

2:

$$S\leftarrow XX^{T}$$

Construct matrix S from X

3:

Us,Ss,Vs←SVD(S)Perform SVD on matrix S

4:

$$\leftarrow W\times U_{s}\times \sqrt {S_{s}}$$

Construct matrix D

5:

$$U_{ws},S_{ws},V_{ws}\leftarrow \mathbf {SVD}(D)$$

Perform SVD on

matrix D

6:

Ts←Truncate( e(Sws,.,R)

Perform SVD

truncation on matrixSwsbased on compression ratio R

7:

$$W^{\prime }\leftarrow U_{ws}xT_{s}xS_{s}^{-1}xU_{s}^{-1}$$

ConstrucW′

8:

**return** W'

9**: end procedure**

compute their truncation loss by SVD-LLMusing 256 randomly selected data in the C4 dataset un-der compression ratios 20% and 60%. As shown in Table 1,because XXTis not positive definite, SVD-LLM fails to compress matrix A. For matrix B,even when the compression ratio is as low as 20%,SVD-LLM still achieves a larger truncation loss in practice than in theory, and this difference even increases with increasing compression ratio.

SVD-based LLM compression methods such as Balco (Ji et al., 2024) have been proposed that utilize pooled covariance matrices to precisely esti-mate the feature distribution to reduce truncation loss. However, these methods cannot guarantee their theoretical optimality during SVD truncation. Therefore, it is necessary to design a new way to optimize the truncation loss for SVD compression.

Table 1: Comparison of the normalized truncation loss (↓) between SVD-LLMM and SVD-LLM V2 on two ran-domly selected weight matrices in LLaMA-3 8B using 256 calibration data on C4 under 20% and 60% com-pression ratios. Fail means the algorithm raises an error during the SVD compression.

<table border="1" ><tr>
<td></td>
<td colspan="2">MATRIX A</td>
<td colspan="2">MATRIX B</td>
</tr><tr>
<td>RATIO</td>
<td>20%</td>
<td>60%</td>
<td>20%</td>
<td>60%</td>
</tr><tr>
<td>Theoretical</td>
<td>0.5982</td>
<td>2.3251</td>
<td>0.7351</td>
<td>3.5245</td>
</tr><tr>
<td>SVD-LLM</td>
<td>Fail</td>
<td>Fail</td>
<td>0.8961</td>
<td>5.9834</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>0.5982 </td>
<td>2.3251</td>
<td>0.7351(↓18%)</td>
<td>3.5245(↓41%)</td>
</tr></table>

**Key** **Design:** The pseudocode of the proposed loss-optimized weight truncation is provided in AAlgo-rithm 2. Different from SVD-LLM, SVD-LLM V2bypasses the Cholesky decomposition, resulting in a more straightforward process with improved numerical stability. Specifically, given the input ac-tivation X, SVD-LLM V2 conducts SVD onXXTto obtain the decomposed matricesUs,Ss,Vs,,where

Ssis the diagonal matrix with singular values. It then conducts another round of SVD onWx $U_{}\times \sqrt {S_{}}$ to obtain Uws,Sws,Vws The final compressed weight matrix W′ can be obtained viaUwsTruc.(Sws)Ss-1Us-1,where Trunc.(C) denotes the rank-k truncation of ma-trix C during SVD compression.

In the following, we provide a theoretical proof on why such truncation offers the same theoretical minimum truncation loss as SVD-LLM.

**Theorem 3.1.**IfUs,Ss,Vsare obtained by SVD decompositionofXXTandUws,Sws,Vwsare ob-tained by SVD decomposition $ofW\times U_{s}\times \sqrt {S_{s}},$ the compressed weight matrixW′=Uws×Trunc. $\left(S_{ws}\right)xV_{ws}x\sqrt {S_{s}}^{-1}xU_{s}^{-1}$ ensures the theoretical minimum truncation loss.

Proof. SinceXXTis the symmetric matrix, sup-pose that the singular vectors and values of input activationXisUx,Sx,Vx,we haveU=Ux and $\sqrt {S_{s}}=S_{x}$ . Suppose $S=U_{s}\times \sqrt {S_{s}},$ ,thus $S^{-1}=\sqrt {S_{s}}^{-1}xU_{s}^{-1}$ ,and we have:

$$S^{-1}X=\sqrt {S_{s}}^{-1}U_{s}^{-1}X=S_{x}^{-1}U_{x}^{-1}X\quad =S_{x}^{-1}U_{x}^{-1}U_{x}S_{x}V_{x}=Vx\tag{2}$$

Therefore,S-1×Xis orthogonal and\|AxS-1x X\|F=\|S-1xX\|F,and the final truncation loss could be derived as:

$$L^{2}=\left\|WX-W^{\prime }X\right\|_{F}^{2}$$

$$=\left\|WSS^{-1}X-U_{ws}x\text {Trunc.}\left(S_{ws}\right)xV_{ws}xS^{-1}X\right\|_{F}^{2}$$

$$=\left\|\left(WS-U_{ws}x\text {Trunc.}\left(S_{ws}\right)xV_{ws}\right)S^{-1}X\right\|_{F}^{2}$$

$$=\left\|WS-U_{ws}x\text {Trunc.}\left(S_{ws}\right)xV_{ws}\right\|_{F}^{2}$$

$$=\|\mathbf {SVD}(WS)\|_{F}^{2}=\left\|\mathbf {SVD}\left(W\times U_{x}\times S_{x}\right)\right\|_{F}^{2}$$

$$=\left\|\mathbf {SVD}\left(W\times U_{x}\times S_{x}\times V_{x}\right)\right\|_{F}^{2}$$

$$=\|\mathbf {SVD}(WX)\|_{F}^{2}=L_{\min }^{2}$$

(3)

Therefore, the designed SVD truncation ensures the theoretical minimum truncation loss. ☐

For a better demonstration, we also imple-ment the new loss-optimized weight truncation by SVD-LLM V2 on LLaMA-3-8B. As shown in Table 1, SVD-LLM V2 achieves better numerical stability and lower truncation loss than SVD-LLM.

## 4 Experiments and Analysis

**Baselines.** We compare SVD-LLM V2 against two groups of methods. (1) Three state-of-the-art SVD-based **LLM** compression methods: FWSVD (Hsu et a1., 2022), ASVD (Yuan et al., 2023), and SVD-LLM(Wang et al., 2024b) (Section 4.1).(2)Other types of LLM compression methods. These include three state-of-the-art pruning-based LLM compres-sion methods: LLM-Pruner (Ma et al., 2023), SliceGPT (Ashkboos et al., 2024), and Block-Pruner (Zhong et al., 2024), and two state-of-the-art quantization-based LLM compression methods: PB-LLM(Yuan et al., 2024), and BiLLM (Huang et al., 2024)(Section 4.4).

**Models** **and** **Datasets.** To demonstrate the gen-erability of our method, we evaluate the per-formance of SVD-LLM V2 on five models at var-ious scales (LLaMA-7B, 13B, 30B, LLaMA3-8B (Touvron et al., 2023),OPT-6.7B (Zhang et al., 2022)) and ten datasets including two lan-guage modeling datasets (WikiText-2 (Merityet al., 2017) and C4 (Raffel et al., 2020)), six classifi-cation1 datasets (OpenbookQA (Mihaylov et al., 2018), WinoGrande (Sakaguchi et al., 2020), Hel-laSwag (Zellers et al., 2019), Arce (Clark et al., 2018), PIQA (Bisk et al., 2020), MathQA (Amini et al., 2019)), and two generation datasets (Truth-fulQA (Lin et al., 2022) and GSM8K(Cobbe et al., 2021)) with the LM-Evaluation-Harness frame-work (Gao et al., 2023).

**Implementation Details.** We randomly select 256WikiText-2 samples as the calibration data. To mitigate the error raised by the Choleksy decom-position in SVD-LLM due to positive definite, we followed the implementation of SVD-LLM(Wang et al.,2024b) to add the small noise into the de-composed matrices. The compression ration in our experiments refers to the parameter reduction of LLM achieved through compression. All of the experiments are conducted on A100 GPUs.

### 4.1 Performance Comparison

We first compare SVD-LLM V2 against state-of-the-art SVD-based LLM compression methods from four aspects: (1) performance on different LLMs, (2) performance on LLMs with larger scales,(3) performance under different compression ratios, and (4)compression speed.

**Performance on Different LLMs.** We compare the performance between SVD-LLM V2 and the base-lines on three different LLMs, including LLaMA-7B,OPT-6.7B, and LLaMA-3 8B under 20% com-pression ratio on ten datasets. As shown in Table 2, SVD-LLM V2 consistently achieves better and more stable performance than all the SVD-based LLM compression baselines across all three LLMs and all ten datasets. In particular, SVD-LLM V2 achieves

Table 2: Performance of OPT-6.7B, LLaMA-7B, and LLaMA-3 8B compressed by SVD-LLM V2 and baselines under 20% compression ratio on two language modeling datasets (measured by perplexity (↓)), six classification datasets (measured by both individual and average accuracy (↑)), two generation datasets (TruthfulQA measured by BLEU score (↑),and GSM8K measured by Exact Match Accuracy (↑)). The best performance is marked in bold. The relative performance gain compared to the best-performing baseline is marked in green inside bracket.

<table border="1" ><tr>
<td></td>
<td>METHOD</td>
<td>WikiText-2↓</td>
<td>C4↓</td>
<td>Openb.</td>
<td>ARCe</td>
<td>WinoG.</td>
<td> HellaS.</td>
<td> PIQA</td>
<td>MathQA</td>
<td>Average↑</td>
<td>TruthfulQA↑</td>
<td>GSM8K↑</td>
</tr><tr>
<td rowspan="5">E L<br>-<br>V I A E T 1</td>
<td>Original</td>
<td>5.68</td>
<td>7.34</td>
<td>0.34</td>
<td>0.75</td>
<td>0.70</td>
<td>0.57</td>
<td>0.79</td>
<td>0.27</td>
<td>0.57</td>
<td>0.30</td>
<td>0.09</td>
</tr><tr>
<td>FWSVD</td>
<td>1727</td>
<td>1511</td>
<td>0.09</td>
<td>0.11</td>
<td>0.05</td>
<td>0.08</td>
<td>0.10</td>
<td>0.05</td>
<td>0.08</td>
<td>0.00</td>
<td>0.00</td>
</tr><tr>
<td>ASVD</td>
<td>11.14</td>
<td>15.93</td>
<td>0.29</td>
<td>0.53</td>
<td>0.64</td>
<td>0.41</td>
<td>0.68</td>
<td>0.17</td>
<td>0.45</td>
<td>0.21</td>
<td>0.04</td>
</tr><tr>
<td>SVD-LLM</td>
<td>7.94</td>
<td>15.84</td>
<td>0.31</td>
<td>0.71</td>
<td>0.68</td>
<td>0.49</td>
<td>0.71</td>
<td>0.22</td>
<td>0.52</td>
<td>0.24</td>
<td>0.06</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>7.12(↓10%)</td>
<td>10.47(↓34%)</td>
<td>0.32</td>
<td>0.72</td>
<td>0.70</td>
<td>0.52</td>
<td>0.75</td>
<td>0.24</td>
<td>0.54(↑4%)</td>
<td>0.27(+0.03)</td>
<td>0.07(+0.01)</td>
</tr><tr>
<td rowspan="5">H L<br>9<br>.<br>-<br>L a O</td>
<td>Original</td>
<td>10.87</td>
<td>12.50</td>
<td>0.28</td>
<td>0.66</td>
<td>0.65</td>
<td>0.50</td>
<td>0.76</td>
<td>0.25</td>
<td>0.52</td>
<td>0.29</td>
<td>0.01</td>
</tr><tr>
<td>FWSVD</td>
<td>14559</td>
<td>17898</td>
<td>0.03</td>
<td>0.08</td>
<td>0.02</td>
<td>0.01</td>
<td>0.05</td>
<td>0.01</td>
<td>0.03</td>
<td>0.01</td>
<td>0.00</td>
</tr><tr>
<td>ASVD</td>
<td>82</td>
<td>102</td>
<td>0.16</td>
<td>0.41</td>
<td>0.30</td>
<td>0.36</td>
<td>0.61</td>
<td>0.07</td>
<td>0.32</td>
<td>0.09</td>
<td>0.00</td>
</tr><tr>
<td>SVD-LLM</td>
<td>16.04</td>
<td>21.27</td>
<td>0.21</td>
<td>0.56</td>
<td>0.59</td>
<td>0.47</td>
<td>0.73</td>
<td>0.21</td>
<td>0.46</td>
<td>0.22</td>
<td>0.00</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>13.46(↓16%)</td>
<td>17.72(↓17%)</td>
<td>0.25</td>
<td>0.61</td>
<td>0.62</td>
<td>0.49</td>
<td>0.74</td>
<td>0.22</td>
<td>0.49(↑7%)</td>
<td>0.24(+0.02)</td>
<td>0.01(+0.01)</td>
</tr><tr>
<td rowspan="5">9<br>8<br>E -<br>V A<br>8<br>T 1</td>
<td>Original</td>
<td>6.14</td>
<td>9.47</td>
<td>0.35</td>
<td>0.80</td>
<td>0.73</td>
<td>0.60</td>
<td>0.80</td>
<td>0.40</td>
<td>0.61</td>
<td>0.49</td>
<td>0.45</td>
</tr><tr>
<td>FWSVD</td>
<td>4782</td>
<td>8195</td>
<td>0.01</td>
<td>0.04</td>
<td>0.01</td>
<td>0.02</td>
<td>0.02</td>
<td>0.01</td>
<td>0.02</td>
<td>0.00</td>
<td>0.00</td>
</tr><tr>
<td>ASVD</td>
<td>17.55</td>
<td>28.41</td>
<td>0.20</td>
<td>0.59</td>
<td>0.61</td>
<td>0.41</td>
<td>0.69</td>
<td>0.30</td>
<td>0.47</td>
<td>0.37</td>
<td>0.28</td>
</tr><tr>
<td>SVD-LLM</td>
<td>11.82</td>
<td>20.05</td>
<td>0.29</td>
<td>0.77</td>
<td>0.64</td>
<td>0.51</td>
<td>0.72</td>
<td>0.30</td>
<td>0.54</td>
<td>0.45</td>
<td>0.31</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>8.01(↓32%)</td>
<td>11.72(↓42%)</td>
<td>0.33</td>
<td>0.79</td>
<td>0.70</td>
<td>0.58</td>
<td>0.77</td>
<td>0.36</td>
<td>0.59(↑9%)</td>
<td>0.46(+0.01)</td>
<td>0.40(+0.09)</td>
</tr></table>

up tp 42% perplexity reduction and 9% accuracy improvement with better generation ability com-pared to prior state-of-the-art mnethod SVD-LLM on LLaMA-3 8B.

**Performance on LLMs with Larger Scales.** We compare the performance between SVD-LLM V2 and the baselines on LLaMA-13B and LLaMA-30B under 20% compression ratio on WikiText-2 and six classification datasets. As shown in Table 3, SVD-LLM V2 consistently outperforms all the base-lines on both 13B and 30B model sizes.

Table 3: Perplexity (↓) on WikiText-2 and average ac-curacy (↑) of six classification datasets of LLaMA-13B and LLaMA-30B under 20% compression ratio.

<table border="1" ><tr>
<td></td>
<td colspan="2">LLAMA-13B</td>
<td colspan="2">LLAMA-30B</td>
</tr><tr>
<td>METHOD</td>
<td>Perplexity↓</td>
<td>Accuracy↑</td>
<td>Perplexity↓</td>
<td>Accuracy↑</td>
</tr><tr>
<td>Original</td>
<td>5.09</td>
<td>0.59</td>
<td>4.10</td>
<td>0.61</td>
</tr><tr>
<td>FWSVD</td>
<td>15.98</td>
<td>0.43</td>
<td>20.54</td>
<td>0.42</td>
</tr><tr>
<td>ASVD</td>
<td>6.74</td>
<td>0.54</td>
<td>22.71</td>
<td>0.44</td>
</tr><tr>
<td>SVD-LLM</td>
<td>6.61</td>
<td>0.55</td>
<td>5.63</td>
<td>0.57</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>5.46(↓17%)</td>
<td>0.56(↑2%)</td>
<td>4.71(↓16%)</td>
<td> 0.60(↑5%)</td>
</tr></table>

**Performance under Different Compression Ra-tios.** We compare the performance between SVD-LLM V2 and the baselines on LLaMA-7B un-der compression ratio ranging from 20% to 80% on WikiText-2 and six classification datasets. As shown in Figure 4, SVD-LLM V2 consistently out-performs all baselines, and the performance gain compared to the best-performing baseline increases as the compression ratio increases.

**Compression** **Speed.** Besides measuring the per-formance of the compressed LLMs, we also evalu-ate the compression speed. Specifically, we mea-sure the A100 GPU houirs used by SVD-LLM V2 and the baseline methods for compressing LLaMA-7B under 20% compression ratio. Our results show that FWSVD takes about 6 GPU hours, ASVD takes about 5.5 GPU hours, SVD-LLM takes about 15 minutes, and SVD-LLMV2 takes about 18 minutes to finish the compression. FWSVD requires gradi-ent calculation, thus consumes a significant amount of time for compression. For the other methods, the main reason for such a variation is their respec-tive techniques for allocating compression ratios among weight matrices. SVD-LLM assigns the same compression ratio to all weight matrices, en-abling the fastest operation but sacrificing accuracy. ASVD,however,determines the compression ratio by regularly evaluating the end-to-end perplexity, which slows down its compression process. In con-trast, SVD-LLM V2 allocates the compression ratio directly from its truncation loss, making it signifi-cantly faster than ASVD.

### 4.2 Inference Speedup of SVD-LLM V2

To evaluate the inference speedup of models com-pressed by SVD-LLM V2, we measure the numbers of tokens generated per second from both the original LLaMA-7B and the model compressed by SVD-LLM V2 under different compression ratios on a single NVIDIA A100 GPU. For a fair comparison, we fix the batch size to 4, the prefill length to 1024, and the decoding length to 256. As shown in Figure 5,

<!-- FWSVD ASVD SVD-LLM SVD-LLM V2 0.5 AIxeldlad 50 enCCA 20% 40% 60% 80% 0.020% 40% 60% 80% Compression Ratio Compression Ratio -->
![](https://web-api.textin.com/ocr_image/external/526c7863df0d530d.jpg)

(a) Perplexity

(b) Average Accuracy

Figure 4: Perplexity on WikiText-2 and average accu-racy on six classification datasets of LLaMA-7B com-pressed by SVD-LLM V2 and other SVD-based LLM com-pression baselines under 20% to 80% compression ra-tios.The perplexity values of FWSVD and ASVD are larger than100,thus are not shown in the figure.

<!-- x103 6 2.71x 2.08x S/SuexOL 4 1.63x 1.29x 2 Original 20% 40% 60% 80% Compression Ratio -->
![](https://web-api.textin.com/ocr_image/external/eaee50683b9ce054.jpg)

Figure 5: Throughput (Tokens/s) achieved by original LLaMA-7B and its compressed version by SVD-LLMV2under different compression ratios on a single NVIDIA A100 GPU. We fix the batch size to 4, prefill length to 1024,and decoding length to 256. The speedup over the original LLM is marked in red.

SVD-LLM V2 consistently achieves faster token gen-eration speeds across all the compression ratios. More importantly, the speedup becomes more sig-nificant as the compression ratio increases, result-ing in a speedup of 1.29x under 20% compression ratio, 1.63x under 40% compression ratio, 2.08x under 60% compression ratio, and 2.71x under 80% compression ratio.

Table 4: Perplexity (↓) of compressed LLaMA-7B by SVD-LLM and SVD-LLM V2 with individual /both com-ponents under 20% compression ratio on WikiText-2.

<table border="1" ><tr>
<td>SVD-LLM</td>
<td>SVD-LLM V2(A)</td>
<td>SVD-LLM V2(T)</td>
<td>SVD-LLM V2</td>
</tr><tr>
<td>7.94</td>
<td>7.91(↓1%)</td>
<td>7.43(↓6%)</td>
<td>7.12(↓10%)</td>
</tr></table>

### 4.3 Ablation Study

SVD-LLM V2 has two key components, both of which optimize the truncation loss. In our ablation study, we first evaluate the individual contribution of each of the two components to the compression performance. Next, since both components fully utilize the whitening matrix S, which is calculated with a randomly selected calibration set, we evalu-ate the impacts of different calibration data on the

<!-- 7.24 AIxa 7.20 $\hat {\mathcal {R}}$ 7.16 Per 7.12 256 512 1024 2048 Number of data -->
![](https://web-api.textin.com/ocr_image/external/b4715d628acc222e.jpg)

<!-- 7.15 AaIxeldad 7.12 7.09 7.06 3 17 42 100 Seed for Random Sampling -->
![](https://web-api.textin.com/ocr_image/external/b91a36617fc624b5.jpg)

(a) Various Data Number

(b) Various Samping Seed

Figure 6: Perplexity of LLaMA-7B under 20% compres-sion ratio using calibration data sampled with different numbers or seeds from WikiText-2.

performance of SVD-LLM V2.

**Component Sensitivity Study.** We first evaluate the individual contribution of the two components (i.e., heterogeneous compression ratio allocation and loss-optimized weight truncation) of SVD-LLM V2. Let SVD-LLM V2 (A) denote the version of SVD-LLM V2 with heterogeneous compression ra-tio allocation only; and SVD-LLM V2 (T) denote the version of SVD-LLM V2 with loss-optimized weight truncation only. The results are shown in Table 4. We have two observations. (1) Both SVD-LLM V2(A) and SVD-LLM V2 (T) outperform SVD-LLM, demonstrating the effectiveness of each of these two components alone in achieving superior com-pression performance. (2) SVD-LLM V2 outperforms SVD-LLM V2 (A) and SVD-LLM V2 (T), demonstrat-ing the necessity of having both components.

**Impact of Calibration Set.** Next, we examine the impact of the calibration set on the compres-sion performance of SVD-LLM V2. Specifically,we measure the changes in perplexity of LLaMA-7B compressed by SVD-LLM V2 under 20% compres-sion ratio on WikiText-2 when using the default calibration set but with various numbers of data and sampling seeds. As shown in Figure 6, the changes in both data number and sampling seed in the calibration set incur no more than 1% fluctua-tion in the final performance, demonstrating that SVD-LLM V2 is not sensitive to the calibration set.

### 4.4 Comparison with Other Types of Post-training LLM Compression Methods

SVD-LLM V2 is orthogonal to other post-training LLM compression methods, including quantization and pruning. In this experiment, we compare the performance of SVD-LLM V2 with state-of-the-art structured pruning-based and quantization-based LLM compression methods.

Table 5: Perplexity (↓) of LLaMA-7B compressed by structured pruning methods and SVD-LLM V2 under vari-ous weight memory budgets on WikiText-2.

<table border="1" ><tr>
<td></td>
<td colspan="4">PERPLEXITY (↓) UNDER WEIGHT MEMORY BUDGET</td>
</tr><tr>
<td>METHOD</td>
<td>10 GB</td>
<td>9 GB</td>
<td>8 GB</td>
<td>7 GB</td>
</tr><tr>
<td>LLM-Pruner</td>
<td>9.88</td>
<td>12.21</td>
<td>18.94</td>
<td>21.68</td>
</tr><tr>
<td>SliceGPT</td>
<td>8.78</td>
<td>12.73</td>
<td>16.39</td>
<td>27.41</td>
</tr><tr>
<td>BlockPruner</td>
<td>9.40</td>
<td>12.76</td>
<td>19.78</td>
<td>43.05</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>7.84(↓17%)</td>
<td>8.48(↓34%)</td>
<td>10.17(↓49%)</td>
<td>15.62(↓28%)</td>
</tr></table>

Table 6: Average accuracy (↑) of LLaMA-7B com-pressed by structured pruning methods and SVD-LLM V2 under various weight memory budgets.

<table border="1" ><tr>
<td></td>
<td colspan="4">AVERAGE ACCURACY (↑) UNDER WEIGHT MEMORY BUDGET</td>
</tr><tr>
<td>METHOD</td>
<td>10 GB</td>
<td>9 GB</td>
<td>8 GB</td>
<td>7 GB</td>
</tr><tr>
<td>LLM-Pruner</td>
<td>0.49</td>
<td>0.47</td>
<td>0.35</td>
<td>0.31</td>
</tr><tr>
<td>SliceGPT</td>
<td>0.51</td>
<td>0.46</td>
<td>0.38</td>
<td>0.29</td>
</tr><tr>
<td>BlockPruner</td>
<td>0.48</td>
<td>0.46</td>
<td>0.33</td>
<td>0.20</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>0.52(↑2%)</td>
<td>0.50(↑6%)</td>
<td>0.42(↑11%)</td>
<td>0.35(↑13%)</td>
</tr></table>

**Comparison with Structured Pruning.** First, we compare SVD-LLM V2 with three state-of-the-art post-training structured pruning-based LLM com-pression methods: LLM-Pruner (Ma et al., 2023), SliceGPT (Ashkboos et al., 2024), and Block-Pruner (Zhong et al., 2024) under various weight memory budgets, ranging from 10 GB to 7 GB. The perplexity results are shown in Table 5 and the average accuracy results are shown in Table 6. As shown, SVD-LLM V2 outperforms all three state-of-the-art structured pruning-based LLM compression methods. In particular, under 7 GB memory budget, SVD-LLM V2 achieves 28% reduction in perplexity and 13% higher average accuracy.

**Comparison with Quantization.** Next, we com-pare SVD-LLM V2 with post-training quantization-based LLM compression methods. We first com-pare SVD-LLM V2 with GPTQ (Frantar et al., 2022) under 3-bit quantization. As shown in Table 7, while SVD-LLM V2 achieves worse perplexity com-pared to GPTQ under 3-bit memory budget, com-bining SVD-LLM V2 (30% compression ratio) wvith GPTQ-4-bit achieves superior perplexity compared to GPTQ-3-bit under the same memory budget. In other words, we find that under the same mem-ory budget, by first compressing the original 16-bit LLM with SVD-LLM V2 at 30% compression ratio, then quantizing the compressed LLM to 4-bit us-ing GPTQ, we are able to achieve better perplex-ity compared to directly quantizing the original LLM to 3-bit. Finally, we compare SVD-LLM V2with two state-of-the-art post-training quantization-based LLM compression methods: **BiLLM** (Huang et al., 2024) and PB-LLM (Yuan et al., 2024), which push the frontier to 1-bit quantization. The results are shown in Table 8. We have two ob-servations: (1) Without combining with quanti-zation techniques, SVD-LLM V2 (16-bit) outper-forms PB-LLM with 5% lower perplexity.(2)By combining with quantization techniques, SVD-LLM V2 (2-bit) outperforms state-of-the-art 1-bit post-training quantization method BiLLM. In particular, SVD-LLM V2 (2-bit) achieves 69% lowerperplexity than BiLLM, showing the promise of combining SVD-based and quantization-based compression methods for pushing the frontier of post-training LLM compression forward.

Table 7: Perplexity (↓) of LLaMA-7B compressed by GPTQ and SVD-LLM V2 on WikiText-2.

<table border="1" ><tr>
<td>METHOD</td>
<td>WEIGHT MEMORY</td>
<td>PERPLEXITY</td>
</tr><tr>
<td>GPTQ-3bit</td>
<td>2.8 GB</td>
<td>16.28</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>2.8 GB</td>
<td>119</td>
</tr><tr>
<td>SVD-LLM V2+ GPTQ-4bit</td>
<td>2.8 GB</td>
<td>9.97(↓39%)</td>
</tr></table>

Table 8: Perplexity (↓) of LLaMA-7B compressed by 1-bit post-training quantization methods and SVD-LLM V2 on WikiText-2.

<table border="1" ><tr>
<td>METHOD</td>
<td>DATA TYPE</td>
<td>WEIGHT MEMORY</td>
<td>PERPLEXITY</td>
</tr><tr>
<td>PB-LLM</td>
<td>1-bit</td>
<td>1.9 GB</td>
<td>104.83</td>
</tr><tr>
<td>BiLLM</td>
<td>1-bit</td>
<td>1.5 GB</td>
<td>47.67</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>16-bit</td>
<td>1.5 GB</td>
<td>99.64</td>
</tr><tr>
<td>SVD-LLM V2</td>
<td>2-bit</td>
<td>1.5 GB</td>
<td>14.73(↓69%)</td>
</tr></table>

## 5 Conclusion

In this paper, we present SVD-LLM V2, a SVD-based post-training LLM compression method. SVD-LLM V2 addresses the limitation of existing methods about high truncation loss during compression. Specifically, SVD-LLM V2 first employs a hetero-geneous compression ratio allocation strategy to effectively balance truncation loss across different weight matrices of the LLM. It further introduces a loss-optimized weight truncation to ensure a lower and more stable truncation loss. Our evaluation results demonstrate the superiority of SVD-LLM V2over state-of-the-art SVD-based post-training LLM compression methods.

## 6 Limitations

While SVD-LLM V2 outperforms existing SVD-based LLM compression methods, there is still space for further improvement. For example, under 90% compression ratio, there is a small perfor-mance gapcompared with state-of-the-art quanti-zation methods. We aim to fill this gap in future.

## 7 Acknowledgement

This study is supported in part by NSF Award NeTS-2312675.

## References

Aida Amini, Saadia Gabriel, Shanchuan Lin,Rik Koncel-Kedziorski,Yejin Choi, and Hannaneh Ha-jishirzi. 2019. Mathqa: Towards interpretable math word problem solving with operation-based for-malisms. In NAACL-HLT (1), pages 2357-2367. As-sociation for Computational Linguistics.

Saleh Ashkboos, Maximilian L. Croci, Marcelo Gen-nari Do Nascimento, Torsten Hoefler, and James Hensman.2024. Slicegpt: Compress large language models by deleting rows and columns. In ICLR. OpenReview.net.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: reasoning about physical commonsense in natural language. In AAAI, pages 7432-7439.AAAI Press.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question an-swering? try arc,the AI2 reasoning challenge. CoRR, abs/1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton,Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word prob-lems. CoRR, abs/2110.14168.

Elias Frantar and Dan Alistarh. 2023. Sparsegpot: Mas-sive language models can be accurately pruned in one-shot. In ICML, volume 202 of Proceedings of Machine Learning Research, pages 10323-10337. PMLR.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler,and Dan Alistarh. 2022. GPTQ: accurate post-training quantization for generative pre-trained transformers. CoRR, abs/2210.17323.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black,Anthony DiPofi, Charles Foster, Laurence Golding,Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, An-ish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. A framework for few-shot language model evaluation.

G.H. Golub, Alan Hoffman, and G.W. Stewart. 1987. A generalization of the eckart-young-mirsky matrix approximation theorem. Linear Algebra and its Ap-plications, 88-89:317-327.

Roberto Gozalo-Brizuela and Eduardo C. Garrido-Merchán. 2023. A survey of generative AI appli-cations. CoRR, abs/2306.02781.

Shwai He, Guoheng Sun, Zheyu Shen, and Ang Li. 2024. What matters in transformers? not all attention is needed. CoRR, abs/2406.15786.

Yen-Chang Hsu, Ting Hua, Sungen Chang,Qian Lou, Yilin Shen, and Hongxia Jin. 2022. Language model compression with weighted low-rank factorization. In ICLR. OpenReview.net.

Wei Huang,Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, and Xiaojuan Qi. 2024. Billm: Pushing the limit of post-training quantization for llms. In ICML. OpenRe-view.net.

Yixin Ji,Yang Xiang,Juntao Li,Wei Chen,Zhongyi Liu, Kehai Chen, and Min Zhang. 2024. Feature-based low-rank compression of large language models via bayesian optimization. CoRR, abs/2405.10616.

Jianwei Li,Yijun Dong, and Qi Lei. 2024. Greedy output approximation: Towards efficient struc-tured pruning for llms without retraining. CoRR, abs/2407.19126.

Chi-Heng Lin, Shangqian Gao, James Seale Smith, Ab-hishek Patel,Shikhar Tuli, Yilin Shen, Hongxia Jin, and Yen-Chang Hsu. 2024a. Modegpt: Modular de-composition for large language model compression. CoRR,abs/240)8.09632.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. In ACL (1),pages 3214-3252. Associa-tion for Computational Linguistics.

Yujun Lin,Haotian Tang,Shang Yang,Zhekai Zhang, Guangxuan Xiao, Chuang Gan, and Song Han. 2024b. Qserve: W4A8KV4 quantization and sys-tem co-design for efficient LLM serving. CoRR, abs/2405.04532.

Xinyin Ma,Gongfan Fang, and Xinchao Wang.2023. Llm-pruner: On the structural pruning of large lan-guage models. In NeurIPS.

Stephen Merity, Caiming Xiong,James Bradbury,and Richard Socher. 2017. Pointer sentinel mixture mod-els. In ICLR(Poster). OpenReview.net.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal.2018. Can a suit of armor conduct elec-tricity? A new dataset for open book question an-swering. In EMNLP, pages 2381-2391. Association for Computational Linguistics.

Colin Raffel, Noam Shazeer,Adam Roberts, Katherine Lee,Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text trans-former. J.Mach. Learn. Res., 21:140:1-140:67.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavat-ula, and Yejin Choi. 2020. Winogrande: An adver-sarial winograd schema challenge at scale. In AAAI, pages 8732-8740. AAAI Press.

Hui Shen,Zhongwei Wan, Xin Wang, and Mi Zhang. 2024. Famba-v: Fast vision mamba with cross-layer token fusion. CoRR, abs/2409.09808.

Hui Shen, Jingxuan Zhang,Boning Xiong, Rui Hu, Shoufa Chen, Zhongwei Wan, Xin Wang,Yu Zhang, Zixuan Gong, Guangyin Bao, et al. 2025. Effi-cient diffusion models: A survey. arXiv preprint arXiv:2502.06805.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-bert,Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava,Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu,Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, An-thony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee,Di-ana Liskovich, Yinghai Lu, Yuning Mao, Xavier Mar-tinet, Todor Mihaylov, Pushkar Mishra, Igor Moly-bog, Yixin Nie, Andrew Poulton,Jeremy Reizen-stein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subrama-nian, Xiaoqing Ellen Tan,Binh Tang, Ross Tay-lor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Ro-driguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288.

Zhongwei Wan, Hui Shen, Xin Wang,Che Liu,Zheda Mai, and Mi Zhang. 2025. Meda: Dynamic kv cache allocation for efficient multimodal long-context infer-ence. arXiv preprint arXiv:2502.17599.

Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng,Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu,Quanlu Zhang, Mosharaf Chowdhury, and Mi Zhang. 2024a. Efficient large language models: A survey. Trans. Mach. Learn. Res., 2024.

Zhongwei Wan, Xinjian Wu,Yu Zhang,Yi Xin,Chaofan Tao,Zhihong Zhu, Xin Wang, Siqi Luo, Jing Xiong, and Mi Zhang. 2024b. D2O: dynamic discriminative operations for efficient generative inference of large language models. CoRR, abs/2406.13035.

Xin Wang, Zhongwei Wan, Arvin Hekmati, Mingyu Zong, Samiul Alam, Mi Zhang, and Bhaskar Krish-namachari. 2024a. The internet of things in the era of generative AI: vision and challenges. IEEE Internet Comput.,28(5):57-64.

Xin Wang,Yu Zheng,Zhongwei Wan,and Mi Zhang. 2024b. SVD-LLM: truncation-aware singular value decomposition for large language model compres-sion. CoRR, abs/2403.07378.

Zhihang Yuan, Yuzhang Shang, and Zhen Dong. 2024. PB-LLM:partially binarized large language models. In ICLR. OpenReview.net.

Zhihang Yuan,Yuzhang Shang,Yue Song, Qiang Wu, Yan Yan,and Guangyu Sun. 2023. ASVD:activation-aware singular value decomposition for compressing large language models. CoRR, abs/2312.05821. Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In ACL(1), pages 4791-4800. Association for Computational Linguistics.

Susan Zhang, Stephen Roller,Naman Goyal, Mikel Artetxe,Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shus-ter, Daniel Simig, Punit Singh Koura, Anjali Srid-har, Tianlu Wang, and Luke Zettlemoyer. 2022. OPT: open pre-trained transformer language mod-els.CoRR, abs/2205.01068.

Wayne Xin Zhao,Kun Zhou, Junyi Li,Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Be-ichen Zhang, Junjie Zhang,Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang,Ruiyang Ren, Yifan Li, Xinyu Tang,Zikang Liu,Peiyu Liu,Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models. CoRR, abs/2303.18223.

Weibo Zhao,Yubin Shi, Xinyu Lyu,Wanchen Sui,Shen Li, and Yong Li. 2024. ASER: activation smoothing and error reconstruction for large language model quantization. CoRR, abs/2411.07762.

Longguang Zhong, Fanqi Wan, Ruijun Chen, Xiaojun Quan, and Liangzhi Li. 2024. Blockpruner: Fine-grained pruning for large language models. CoRR, abs/2406.10594.

Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu,Ji-aming Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan,Xiuhong Li, Shengen Yan, Guohao Dai,Xiao-Ping Zhang, Yuhan Dong, and Yu Wang. 2024. A survey on efficient inference for large lan-guage models. Preprint, arXiv:2404.14294.

Xunyu Zhu,Jian Li, Yong Liu, Can Ma, and Weiping Wang. 2023. A survey on model compression for large language models. CoRR, abs/2308.07633.

