# ResSVD: Residual Compensated SVD for Large

# Language Model Compression

**Haolei Bai**

WestlakeUniversity

Nanyang Technological University

**Yu Yin**

Case Western Reserve University

**Siyong Jian**

WestlakeUniversity

NanjingUniversity

**Tuo Liang**

CaseWesternReserveUniversity

**Huan Wang***

Westlake UUniversity

## Abstract

Large language models (LLMs) have demonstrated impressive capabilities in a wide range of downstream natural language processing tasks. Nevertheless, their considerable sizes and memory demands hinder practical deployment, underscor-ing the importance of developing efficient compression strategies. Singular value decomposition (SVD) decomposes a matrix into orthogonal components, enabling efficient low-rank approximation. This is particularly suitable for LLM compres-sion, where weight matrices often exhibit significant redundancy. However, current SVD-based methods neglect the residual matrix from truncation, resuIting in sig-nificant truncation loss. Additionally, compressing all layers of the model results in severe performance degradation. To overcome these limitations, we propose **ResSVD,** a new post-training SVD-based LLM compression method. Specifically, we leverage the residual matrix generated during the truncation process to reduce truncation loss. Moreover, under a fixed overall compression ratio, we selectively compress the last few layers of the model, which mitigates error propagation and significantly improves the performance of compressed models. Comprehensive evaluations of ResSVD on diverse LLM families and multiple benchmark datasets indicate that ResSVD consistently achieves superior performance over existing counterpart methods, demonstrating its practical effectiveness.

## 1 Introduction

Large language models (LLMs) have emerged as powerful tools, delivering state-of-the-art perfor-mance across a wide range of tasks such as text generation, translation, and reasoning. The scaling law [18] has driven a trend toward increasingly large models, exemplified by models such as GPT [4], PaLM [6],LLaMA [38], Qwen [2], and Deepseek [25], which often contain tens to hundreds of billions of parameters. Despite their powerful capabilities, the enormous scale of LLMs poses serious challenges for efficient deployment due to high computational demands [48,40,35].This resource burden not only limits deployment on edge devices and consumer-level hardware but also increases the cost and carbon footprint of serving LLMs in production [36,31].

AAs the scale of LLMs continues to grow, compression techniques, including weight quantization [11, 24,16,23],network pruning [27,10,37,12], knowledge distillation [13,43,42,47],and low-rank decomposition [15,19,44,41,22],have become increasingly important for the practical deployment of LLLMs in resource-constrained environments. The syntactic and semantic correlations acquired during training often induce redundancy in LLM weight matrices, giving rise to a low-

*Corresponding author: wanghuan@westlake.edu.cn

<!-- Preprint. Under review. -->

<!-- S70z AeIA Os[TO.so] CACITOZ.SOsZ.AAIXTg -->

rank structure [33]. As a result, singular value decomposition (SVD) provides a principled and effective approach for compressing these matrices with minimal loss. In particular, post-training approaches are gaining traction, as they can significantly reduce memory and compute requirements without the need for expensive retraining, making them especially suitable for scaling up foundation models. Recent studies on post-training SVD-based LLM compression, including ASVD [44], SVD-LLM [41], and AdaSVD [22], have mmade significant progress in reducing model size while preserving performance, demonstrating the effectiveness of low-rank approximation techniques in compressing LLMs. More precisely, while both ASVD and SVD-LLM apply weight matrix scaling before SVD truncation, SVD-LLM outperforms ASVD by leveraging a data whitening technique that enables a direct mapping between singular values and truncation loss. As a further advancement, AdaSVD compensates for truncation loss by iteratively updating singular matrices. However, existing methods suffer from two major limitations. First, existing methods ignore the importance of the residual matrix generated during SVD truncation, leading to significant truncation loss, as indicated by our theoretical analysis in Section 3.1. Second, compressing all model layers often results in high layer-wise error and severe error propagation, as confirmed by our results in Figure 4.

In this work, we propose **ResSVD,** a new post-training SVD-based compression mnethod for LLMs. Building upon the two key observations outlined above, ResSVD introduces two core technical innovations. ① **Residual** **compensation for SVD truncation:** The residual matrix produced during SVD truncation can be effectively utilized to reduce the truncation loss. Specifically, we perform SVD truncation in two stages: we first truncate the original weight matrix M to obtain its low-rank approximationr1. After that, we compute the residual matrix R between W and 1Second, we apply SVD truncation to R,yielding 2.Finally,we construct the compressed weight matrix $\hat {\boldsymbol {W}}_{r}=\boldsymbol {W}_{r_{1}}+\boldsymbol {R}_{r_{2}}$ where the total rank satisfiesr1+r2=r. Theoretical justification is provided in Section 3.1. $\textcircled {2}$  **Partial-layer** **compression** **for** **SVD:** LLMs consist of a sequence of consecutive layers, where the output of each layer serves as the input to the next. As a result,any error introduced in earlier layers can propagate and accumulate through subsequent layers,leading to severe degradation in performance. To mitigate this, we propose compressing only the last few layers under a fixed overall compression ratio while keeping the earlier layers intact. This strategy ensures that the earlier layers remain error-free,thereby reducing the impact of error propagation.

Our key contributions can be summarized as follows:

·We introduce **residual** **compensation** **for** **SVD** **truncation,** a theoretically grounded com-pensation strategy. By leveraging the residual matrix generated during the SVD truncation, our strategy significantly reduces the overall truncation loss.

·We propose **partial-layer** **compression** **for** SVD, which compresses only the last few layers of the model under a fixed overall compression ratio. This strategy effectively reduces layer-wise error and mitigates error propagation.

·Extensive experiments on multiple LLMs (LLaMA, OPT, Mistral, and Vicuna) and bench-mark datasets (both language modeling and zero-shot reasoning) demonstratethat ResSVD outperforms existing methods across a wide range of compression ratios.

## 2 Related Work

### 2.1 Techniques for Large Language Model Compression

The growing size of large language models (LLMs) has raised increasing concerns over their com-putational and memory demands. To address these challenges, a variety of model compression techniques have been proposed. Conventional approaches often require computationally expensive retraining, which is generally impractical due to the substantial computational cost associated with the massive size of LLMs. Consequently, recent efforts have shifted toward more resource-friendly post-training compression techniques [10,11,49]. Typically employed approaches include net-work pruning and weight quantization. And pruning techniques can be classified into unstructured and structured methods. Unstructured pruning removes individual weights based on importance scores. For example, SparseGPT [10] performs one-shot pruning using second-order approximations without retraining. However, since unstructured pruning retains the original matrix shape,it offers limited inference acceleration and requires specialized hardware. In contrast, structured pruning eliminates entire blocks oor channels, enabling compatibility with conventional hardware platforms.

<!-- 2 -->

<!-- Original LLM Overall Target Compressed LLM Layer Structure Compression Ratio x Layer Structure Layer 1 W Stay Unchanged W Layer 1 Layer 2 Layer 2 SVD Truncation 000 W Wr1 LayerN-k+1 Compressed Wr Wr LayerN-k+1 000 r1+r2=r 0。。 Layer N LayerN R SVD Truncation Rr2 LayerN 1 Partial-layer Compression 2 Residual Compensation Compression ratio 0 Compression ratio (Nx/k) (Nx/k) -->
![](./images/f676ae63a8430b30.jpg)

Figure 1: Framework of **ResSVD.** The lastk layers are selected through **partial-layer** **compression** and compressed using **residual compensation** with calibration data. The first(N-)layers remain intact, while the lastk layers are replaced by low-rank approximations.

LLM-Pruner [27] groups dependent linear projections into coupled structures, assigns each group a loss-aware importance score, prunes the least important groups, and applies LoRA fine-tuning to restore performance. Additionally, ZipLM [20] prioritizes pruning components that yield the worst trade-off between latency and loss, but this often causes notable performance degradation. Quantization provides another mainstream solution. GPTQ [11] gradually quantizes and updates each weight using the Hessian matrix to minimize the quantization error. AWQ [24] preserves important weight channels by selecting reparameterization coefficients via grid search. Furthermore, BiLLM [16] and ARB-LLM [23] push quantization to the 1-bit level, while still delivering impressive accuracy across downstream tasks. However, these techniques still cause significant performance degradation, especially at low bit-widths, due to the lack of weight and activation adaptation.

### 2.2 SVD-based Techniques for Compressing LLMs

Singular value decommposition (SVD) is a commonly employed technique for reducing matrix dimen-sionality by representing the original matrix as the product of two low-rank factor matrices. Recent research has demonstrated the potential of SVD-based LLM compression methods, but comprehen-sive exploration remains limited. The conventional SVD only aims to compress the original weight matrix, without accounting for the relative importance of individual parameters, leading to significant approximation errors. To address this issue, Hsu et al. [15] propose FWSVD, incorporating Fisher information to weight the importance of parameters. However, it relies on computationally intensive training and was originally applied only to small language models (e.g., BERT [9],ALBERT [21]). LoRD [19] first applies SVD to LLMs by grouping layers to improve efficiency. However, similar to conventional SVD, LoRD shares the limnitation of overlooking input activations. To address this problem, Yuan et al. [44] introduce ASVD, which mitigates the impact of activation outliers by reshaping the weight matrix basedon activation distribution, thereby enhancing the precision of the low-rank decomposition. Additionally, SVD-LLM [41] establishes a direct mapping between singular values and truncation loss, which means that truncating the smallest singular values leads to minimal truncation loss using a truncation-aware data whitening technique. More recently, AdaSVD [22] proposes an adaptive compensation method, iteratively updating matrices during truncation. Despite these advances, all existing methods ignore the residual matrix produced during SVD truncation, which can significantly compensate for the SVD truncation loss. Moreover, compressing only the last few layers of the model under a fixed target compression ratio can provide better performance for compressed models. However, existing methods compress all layers, either apply a uniform compression ratio acrooss all layers or assign variable ratios based on layer-wise importance, which often results in sub-optimal performance.

## 3 ResSVD

The framework of **RResSVD** is illustrated in Figure 1. We first perform SVD to compress the last few layers of the model, while ensuring that the overall compression ratio is satisfied, and compute the

<!-- 3 -->

### Algorithm 1 Pseudocode of ResSVD

**Input:** Original LLM: M,weight matrix:W∈Rm×n, number of model layers: N, intermediate rankr1step:S

**Output:** Compressed LLM M′ by ResSVD

1: CD←Randomly collect calibration samples fromthe dataset

2:SW←M,SW′←\emptyset

Initialize the sets of weight matrices

3:k,R1←PARTIAL-LAYER COMPRESSION(M,N,.o,)

4:**for** **Layer** i in original **LLM** **M** **do**

5:

**if**i∈[1,N-k)**then**

6:

$$\text {Set}_{\boldsymbol {W}^{\prime }}(i)\leftarrow \text {Set}_{\boldsymbol {W}}(i)$$

Current weight matrices stay the same

7:

**else**

8:

Wi←SetW(i),Si←WHITENING(Wi,.,CD)Initialize weight matrices to compress

9:

10:

Setw'(i)←RESIDUAL COMPENSATION(Si, ,k.Rl,r1) **end if**

11:

$$\text {Set}_{\boldsymbol {W}^{\prime }}\leftarrow \text {Set}_{\boldsymbol {W}^{\prime }}(i)\cup \text {Set}_{\boldsymbol {W}^{\prime }}$$

Append weight matrices after operation

12: end **for**

13: M′←UPDATE(M,CD,SetW′)

14**: return** M′

corresponding residual matrix. A second SVD is then applied to the residual matrix to obtain its low-rank approximation. The two truncated matrices are subsequently combined to construct the final compressed weight matrices. In Section 3.1, we first provide a theoretical analysis demonstrating how **residual compensation** effectively reduces truncation loss. Subsequently, in Section 3.2,we explain the benefits of **partial-layer** **compression** in mitigating error propagation. The pseudocode of ResSVD is shown in Algorithm 1, and the pseudocodes for residual compensation and partial-layer compression are provided in Appendix A.1.

### 3.1 Residual Compensation for SVD Truncation

**Preliminaries** **for** **SVD.** Typical SVD-based LLM compression methods apoply SVD to the original weight matrixW∈Rm×,and discard the smallest singular values to obtain a compressed low-rank approximationWr:

$$\boldsymbol {W}=\boldsymbol {U}\boldsymbol {Σ}\boldsymbol {V}^{T}\approx \boldsymbol {U}_{r}\boldsymbol {Σ}_{r}\boldsymbol {V}_{r}^{T}=\boldsymbol {W}_{r},r<\min (m,n)\tag{1}$$

whereUr∈Rm×randVrT∈Rrxn are composed of the top-r left and right singular vectors, respectively,andΣr∈Rr×ris a diagonal matrix containing the corresponding singular values. Fol-lowing prior post-training SVD-based works [44,41], the optimization objective for SVD truncation in LLMs can be formulated as:

$$\hat {\boldsymbol {W}}_{r}=\arg \min _{\boldsymbol {W}_{r}}\left\|\boldsymbol {W}\boldsymbol {X}-\boldsymbol {W}_{r}\boldsymbol {X}\right\|_{F}\tag{2}$$

$$\mathcal {L}=\left\|\boldsymbol {W}\boldsymbol {X}-\boldsymbol {W}_{r}\boldsymbol {X}\right\|_{F}=\left\|\left(\boldsymbol {W}-\boldsymbol {W}_{r}\right)\boldsymbol {X}\right\|_{F}\tag{3}$$

where Xdenotes the activation of Mgiven an input, and Cis the truncation loss measured by the Frobenius norm. Although previous works [15,44,41] have made significant progress in minimizing C,they consistently overlook the residual matrix generated during truncation, despite its potential to compensate for the loss introduced by low-rank approximation.

Based on Equation 3, minimizing the truncation loss reduces to minimizing the discrepancy between the original weight matrix Wand its low-rank approximation Wr.Accordingly, we reformulate the optimization objective as:

$$\hat {\boldsymbol {W}}_{r}=\arg \min _{\boldsymbol {W}_{r}}\left\|\boldsymbol {W}-\boldsymbol {W}_{r}\right\|.\tag{4}$$

**Method** **Explanation** **for** **ResSVD.** Given the original weight matrix W and the target rank r we decompose r into two components: an intermediate rank r1and a residual rankr2=r-r1. The compression process consists of two SVD truncation steps. First, we apply SVD to Wand retain top-r1 singular values to obtain a low-rank approximation: Wr1=Ur1Σr1Vr1T.The residual matrix is then computed as the difference between the original matrix and its rank r1

<!-- 4 -->

approximation:R=W-Wr1Weperform a second SVD on R and retain the top-r2singular values:Rr2=Ur2Σr2Vr2TFinally, we obtain the compressed weight matrix by combining the two approximations: $\hat {\boldsymbol {W}}_{r}=\boldsymbol {W}_{r_{1}}+\boldsymbol {R}_{r_{2}}=\boldsymbol {U}_{r}\boldsymbol {Σ}_{r}\boldsymbol {V}_{r}^{T}$ 

In this way, the residual-compensated compressed matrix $\hat {\boldsymbol {W}}_{r}$ provides a closer approximation to the original weight matrix W than the directly truncated matrixW,as proven in the theoretical analysis that follows.

**Lemma 3.1.** Eckart-Young-Mirsky Theorem [14].Given a matrixA∈R×with singular values σ1(A)≥σ2(A)≥σ3(A)≥⋯≥σmin(,)(A). The best rankr approximationA is given by:

$$\boldsymbol {A}_{r}=\sum _{i=1}^{r}σ_{i}(\boldsymbol {A})u_{i}(\boldsymbol {A})v_{i}^{T}(\boldsymbol {A})\tag{5}$$

and the truncation loss satisfies:

$$\left\|\boldsymbol {A}-\boldsymbol {A}_{r}\right\|_{F}^{2}=\sum _{i=r+1}^{\min (m,n)}σ_{i}^{2}(\boldsymbol {A})\tag{6}$$

As defined above, $\hat {\boldsymbol {W}}_{r}=\boldsymbol {W}_{r_{1}}+\boldsymbol {R}_{r_{2}}$  LetLCdenote the truncation loss between the original matrix W and the residual-compensated compressed matrix $\hat {\boldsymbol {W}}_{r}$ . Then,by applying Lemma 3.1, we have:

$$\mathcal {L}_{C}=\left\|\boldsymbol {W}-\hat {\boldsymbol {W}}_{r}\right\|_{F}^{2}=\left\|\boldsymbol {W}-\left(\boldsymbol {W}_{r_{1}}+\boldsymbol {R}_{r_{2}}\right)\right\|_{F}^{2}=\left\|\boldsymbol {R}-\boldsymbol {R}_{r_{2}}\right\|_{F}^{2}=\sum _{i=r_{2}+1}^{\text {rank}(\boldsymbol {R})}σ_{i}^{2}(\boldsymbol {R})\tag{7}$$

$$\text {rank}(\boldsymbol {R})\leq \min (m,n)-r_{1}$$

**Lemma 3.2.**IfAris the best rank-r approximation ofA∈Rm×n,and the residual isB=A-Ar The we have:

$$σ_{i}(\boldsymbol {B})\leq σ_{r+i}(\boldsymbol {A}),1\leq i\leq \min (m,n)-r.\tag{8}$$

**Theorem 3.3.IfV**Wr is the rank r direct truncation matrix form the original matrix W**,** $\hat {\boldsymbol {W}}_{}$ is the residual-compensated compressed matrix,then $\hat {\boldsymbol {W}}_{r}$ is closer toW

Proof. The direct truncation lossLD between Wand Wris given by:

$$\mathcal {L}_{D}=\left\|\boldsymbol {W}-\boldsymbol {W}_{r}\right\|_{F}^{2}=\sum _{i=r+1}^{\min (m,n)}σ_{i}^{2}(\boldsymbol {W})\tag{9}$$

According to Lemma 3.2, the compensated truncation lossLC, as defined in Equation (7),is upper-bounded by:

$$\sum _{i=r_{2}+1}^{\text {rank}(\boldsymbol {R})}σ_{i}^{2}(\boldsymbol {R})\leq \sum _{i=r_{1}+r_{2}+1}^{\min (m,n)}σ_{i}^{2}(\boldsymbol {W})=\sum _{i=r+1}^{\min (m,n)}σ_{i}^{2}(\boldsymbol {W})\tag{10}$$

Therefore, we obtain:

<table border="1" ><tr>
<td>LC≤LD</td>
</tr></table>

(11)

This proves that the compensated truncation lossLCis smaller than the direct truncation lossLD, which means that $\hat {\boldsymbol {W}}_{r}$ is closer to the original weight matrix M. ☐

### 3.2 PartiaI-layer Compression for SVD

Previous works [15,44,41,22] compress all model layers even if they assign layer-specific ratios based on their relative importance, which often leads to a high layer-wise error, resulting in noticeable degradation in the performance of compressed models. We compare the layer-wise error of LLMs across four families with different layer selection strategies, the results are shown in Figure 3.There

<!-- 5 -->

<!-- 1.4 LLaMA-7B Compress Frst & Layers Compress La2 Compress First 16 Layers Compress Last 18 layers 1.2 Compress All layers Compress Last 16 layers Compress Last 30 1 Cor 1.0 Campress Lant 28 layers Compress Last 12 layers Compress Last 26 layers Compress Last 10 layers Compress Last 24 layers Compress Last Blayers A0Tш童こ99 1 0.6 0.2 10 Layer ex -->
![](./images/60d9f8933e191c19.jpg)

<!-- OPT-6.7B 10 Cemees First & Layers Compress Last 20 layers Compress Frat 16 Layers Compress Last 18 layers 10 Compress Compress Last 16 layers Compress Last 30 layers Compress Last 14 Iauace 10 Compress Lost 28 layers Compress Last 12 layers Compress Last 26 layers Compress Last 10 layers 10 Compress Last 24 layers Pa csAiaA Comeress Last 22 layers 10 ■市1 81 10 ■业 合 ■金： 10-1 -工 12 10 15 25 Lay -->
![](./images/08b730c73811ffd9.jpg)

<!-- Kendall Correlation Kendall:-0.9662 p-va value:0.00% 0 0.4 0.5 0.6 0.7 0.0 Final Layer Error -->
![](./images/62351ea97175be8e.jpg)

<!-- Mistral-7B Campress n Layers Campress CompressFirst 16 Layers Compress Last 18 layers 1.2 Compress All layers ■lpers Compress Last 16 layen Compress Last 30 layers Compress Last 14 layers 1.0 Compress Last 28 layers Compress Last 12 layers Compress Last 26 layers Compress Last 10 layers 自030I30Ae 0.8 s Last 24 layers Compress Last 8 layers 0.6 0.4 0.2 10 Layer Index -->
![](./images/01f6393810b72907.jpg)

<!-- 14 Vicuna-7B Campress rat B Layers Compress yer Compress First 16 Layers Compress Last 18 layers Compress A Compress Last 16 layers Compress Last 30 layers -· Compress Last 14 layens 10 Compress Last 20 layers Compress Last 12 layers Compress Last 26 layers Compress Last 10 layers J0ua eSJahe1 00 Compress LLast 24 la Compress Last 8 layers Compress Last 22 layers 0.0 0.2 0.0 15 20 25 30 Layer Index -->
![](./images/45691b7e1565bbc7.jpg)

<!-- Layer-wise Error Comparison SVD-LLM ResSVD 0a03JeAe 15 20 30 Layer Index -->
![](./images/bdbfde6022b603db.jpg)

Figure 3: Layer-wise error comparison between original model Figure 4: (Top) Kendall correla-and multiple LLM structures compressed by ResSVD with dif- tion.(Bottom) Layer-wise error ferent layer selection strategies on WikiText-2 [29].The overall comparison between ResSVD compression ratio is 20%, and different layer selection strategies and SVD-LLM [41] under 20% strictly adhere to the overall compression ratio constraint. compression ratio.

is a significant error propagation across multiple LLM families, where the error progressively accu-mulates layer by layer during the forward pass. This phenomenon is particularly pronounced when compressing the first 8 layers. To satisfy the overall compression ratio, a higher layer compression ratio (80% in this case) must be applied to these layers. Compressing the model at such an early stage introduces substantial approximation errors, which are then propagated through the remaining layers. Although the later layers are left uncompressed, the forward pass carries the accumulated error through the network. Consequently, in all evaluated LLM families, these early compressed layers exhibit the highest layer-wise error across model layers. In contrast, when only compressing the last few layers, the earlier layers remain untouched with zero error. Although the compression ratio for selected layers is relatively high, leading to slightly faster error accumulation,the overall error remains significantly lower than that of compressing all layers.

Interestingly, we observe that although the final-layer error converges to a narrow range regardless of how many of the last layers are compressed, the differences within this range still exhibit a strong influence on model performance. As illustrated in the top figure of Figure 4, there is a strong correlation between the final-layer error and the average zero-shot accuracy. Motivated by this observation, we select the number of last layers to compress by minimizing the final-layer error, and the layer compression ratio R satisfies:

$$R_{l}=\left(N·R_{o}\right)/k,k\in \left\{1,2,\cdots ,N-1|R_{l}<1\right\}\tag{12}$$

where N is the number of model layers,Rl represents the overall compression ratio. As shown in the bottom figure of Figure 4, compared to SVD-LLM [41], our strategy significantly reduces layer-wise error across all model layers and mitigates error propagation.

## 4 Experiments

### 4.1 Setups

**Baselines.** We compare with four baselines in post-training setting, i.e., without re-training of the model parameters: conventional SVD, ASVD [44], SVD-LLM [41], and AdaSVD [22].

**Models** **and** **Datasets.** We evaluate **ResSVD** on ten models spanning four LLM families: LLaMA-7B [38], LLaMA-13B [38], LLaMA-30B [38],LLaMA-2-7B [39], LLaMA-2-13B [39], OPT-6.7B [46],OPT-13B [46], OPT-30B [46], Mistral-7B [17], Vicuna-7B [5]. For language modeling, we use three benchmark datasets: WikiText-2 [29],PTB [28], and C4 [32]. For zero-shot reasoning

<!-- 6 -->

Table 1: Overall performance of LLaMA-7B [38] compressed by **ResSVD** and baselines under 20% to 60% compression ratios ("RATIO"), including performance on three language modeling datasets (measured by perplexity (↓)) and zero-shot performance on seven common sense reasoninng datasets (measured by individual and average accuracy (↑)). The best results are marked in **bold.** Blue arrows within parentheses highlight the relative improvement over the second-best method.

<table border="1" ><tr>
<td>RATIO</td>
<td>METHOD</td>
<td>WikiText-2↓</td>
<td>PTB↓</td>
<td>C4↓</td>
<td>Openb.↑ </td>
<td>ARCe↑</td>
<td>WinoG.↑</td>
<td> HellaS.↑</td>
<td>ARCc↑</td>
<td> PIQA↑</td>
<td> MathQA↑</td>
<td>Average↑</td>
</tr><tr>
<td></td>
<td>Original</td>
<td>5.67</td>
<td>8.80</td>
<td>7.63</td>
<td>0.28</td>
<td>0.67</td>
<td>0.67</td>
<td>0.56</td>
<td>0.38</td>
<td>0.78</td>
<td>0.27</td>
<td>0.52</td>
</tr><tr>
<td rowspan="4">20%</td>
<td>SVD</td>
<td>20082.86</td>
<td>20338.96</td>
<td>18784.20</td>
<td>0.14</td>
<td>0.27</td>
<td>0.51</td>
<td>0.26</td>
<td>0.21</td>
<td>0.53</td>
<td>0.21</td>
<td>0.30</td>
</tr><tr>
<td>ASVD [44]</td>
<td>9.27</td>
<td>15.09</td>
<td>13.68</td>
<td>0.25</td>
<td>0.53</td>
<td>0.60</td>
<td>0.41</td>
<td>0.27</td>
<td>0.68</td>
<td>0.25</td>
<td>0.43</td>
</tr><tr>
<td>SVD-LLM[41]</td>
<td>7.89</td>
<td>16.54</td>
<td>15.92</td>
<td>0.23</td>
<td>0.56</td>
<td>0.62</td>
<td>0.42</td>
<td>0.29</td>
<td>0.69</td>
<td>0.23</td>
<td>0.43</td>
</tr><tr>
<td>ResSVD</td>
<td>7.47(15%)</td>
<td>12.27(↓19%)</td>
<td>12.22(↓11%)</td>
<td>0.25</td>
<td>0.59</td>
<td>0.67</td>
<td>0.47</td>
<td>0.34</td>
<td>0.71</td>
<td>0.25</td>
<td>0.47(↑9%)</td>
</tr><tr>
<td rowspan="4">30%</td>
<td>SVD</td>
<td>13155.97</td>
<td>17354.46</td>
<td>21012.91</td>
<td>0.13</td>
<td>0.26</td>
<td>0.51</td>
<td>0.25</td>
<td>0.21</td>
<td>0.54</td>
<td>0.21</td>
<td>0.30</td>
</tr><tr>
<td>ASVD [44]</td>
<td>222.98</td>
<td>586.79</td>
<td>148.79</td>
<td>0.15</td>
<td>0.32</td>
<td>0.53</td>
<td>0.30</td>
<td>0.21</td>
<td>0.59</td>
<td>0.21</td>
<td>0.33</td>
</tr><tr>
<td>SVD-LLM [41]</td>
<td>9.52</td>
<td>28.97</td>
<td>26.38</td>
<td>0.20</td>
<td>0.49</td>
<td>0.59</td>
<td>0.37</td>
<td>0.27</td>
<td>0.65</td>
<td>0.22</td>
<td>0.40</td>
</tr><tr>
<td>ResSVD</td>
<td>9.52(-)</td>
<td>20.31(↓30%)</td>
<td>18.29(↓31%)</td>
<td>0.23</td>
<td>0.54</td>
<td>0.63</td>
<td>0.41</td>
<td>0.30</td>
<td>0.67</td>
<td>0.23</td>
<td>0.43(↑8%)</td>
</tr><tr>
<td rowspan="4">40%</td>
<td>SVD</td>
<td>52326.99</td>
<td>59859.41</td>
<td>47643.04</td>
<td>0.15</td>
<td>0.25</td>
<td>0.51</td>
<td>0.25</td>
<td>0.21</td>
<td>0.52</td>
<td>0.20</td>
<td>0.30</td>
</tr><tr>
<td>ASVD [44]</td>
<td>5262.11</td>
<td>8806.33</td>
<td>6522.61</td>
<td>0.14</td>
<td>0.26</td>
<td>0.49</td>
<td>0.26</td>
<td>0.22</td>
<td>0.53</td>
<td>0.20</td>
<td>0.30</td>
</tr><tr>
<td>SVD-LLM [41]</td>
<td>13.83</td>
<td>57.07</td>
<td>48.47</td>
<td>0.19</td>
<td>0.41</td>
<td>0.58</td>
<td>0.32</td>
<td>0.22</td>
<td>0.59</td>
<td>0.22</td>
<td>0.36</td>
</tr><tr>
<td>ResSVD</td>
<td>12.92(↓7%)</td>
<td>46.93(↓18%)</td>
<td>30.51(↓37%)</td>
<td>0.19</td>
<td>0.44</td>
<td>0.60</td>
<td>0.35</td>
<td>0.27</td>
<td>0.62</td>
<td>0.24</td>
<td>0.39(↑8%)</td>
</tr><tr>
<td rowspan="4">50%</td>
<td>SVD</td>
<td>130388.72</td>
<td>86721.38</td>
<td>79853.46</td>
<td>0.16</td>
<td>0.26</td>
<td>0.49</td>
<td>0.25</td>
<td>0.22</td>
<td>0.52</td>
<td>0.18</td>
<td>0.30</td>
</tr><tr>
<td>ASVD[44]</td>
<td>62726.86</td>
<td>117959.06</td>
<td>77773.84</td>
<td>0.13</td>
<td>0.25</td>
<td>0.48</td>
<td>0.25</td>
<td>0.22</td>
<td>0.53</td>
<td>0.21</td>
<td>0.30</td>
</tr><tr>
<td>SVD-LLM [41]</td>
<td>24.05</td>
<td>150.58</td>
<td>141.87</td>
<td>0.16</td>
<td>0.34</td>
<td>0.55</td>
<td>0.29</td>
<td>0.21</td>
<td>0.56</td>
<td>0.22</td>
<td>0.33</td>
</tr><tr>
<td>ResSVD</td>
<td>22.41(↓7%)</td>
<td>128.80(↓14%)</td>
<td>83.80(↓41%)</td>
<td>0.16</td>
<td>0.36</td>
<td>0.58</td>
<td>0.32</td>
<td>0.23</td>
<td>0.59</td>
<td>0.22</td>
<td>0.35(↑6%)</td>
</tr><tr>
<td rowspan="4">60%</td>
<td>SVD</td>
<td>52326.99</td>
<td>59859.41</td>
<td>47643.04</td>
<td>0.15</td>
<td>0.25</td>
<td>0.51</td>
<td>0.25</td>
<td>0.21</td>
<td>0.52</td>
<td>0.20</td>
<td>0.30</td>
</tr><tr>
<td>ASVD [44]</td>
<td>16221.43</td>
<td>20119.36</td>
<td>16561.39</td>
<td>0.13</td>
<td>0.26</td>
<td>0.50</td>
<td>0.25</td>
<td>0.23</td>
<td>0.53</td>
<td>0.18</td>
<td>0.30</td>
</tr><tr>
<td>SVD-LLM[41]</td>
<td>53.20</td>
<td>378.19</td>
<td>310.17</td>
<td>0.12</td>
<td>0.29</td>
<td>0.52</td>
<td>0.28</td>
<td>0.20</td>
<td>0.55</td>
<td>0.22</td>
<td>0.31</td>
</tr><tr>
<td>ResSVD</td>
<td>48.67(19%)</td>
<td>323.74(↓14%)</td>
<td>260.06(↓16%)</td>
<td>0.14</td>
<td>0.30</td>
<td>0.53</td>
<td>0.28</td>
<td>0.21</td>
<td>0.55</td>
<td>0.21</td>
<td>0.32(↑3%)</td>
</tr></table>

and understanding, we evaluate on seven tasks within the LM-Evaluation-Harness framework2: Open-bookQA [30],WinoGrande [34], HellaSwag [45], PIQA [3], MathQA [1], ARCe and ARCc[7].

**Implementation** **Details.** To facilitate a fair comparison, we follow the protocols of ASVD, SVD-LLM, and AdaSVD, randomly selecting 256 calibration samples with a sequence length of 2048from WikiText-2. We then apply data whitening prior to performing SVD truncation. All results are reproduced by re-running their respective open-source codebases,except for AdaSVD,whose results are directly taken from the original paper due to the lack of released code. All methods are all implemented with PyTorch³ and Transformers⁴ on NVIDIA A100 GPUs.

### 4.2 Results

We evaluate the overall performance of ResSVD from three aspects: ① **Effectiveness** **under** **different** **compression ratios** (ranging from 20% to 60% in increments of 10%), ② **Scalability when applied to** **larger-scale models,** and ③ **Generalizability across diverse LLM families.** In addition,qualitative results, such as generated contents produced by compressed models, are providedin Appendix A.4, offering a more intuitive and direct comparison.

**Performance under different compression ratios.** We evaluate the performance of LLaMA-7B [38] and LLaMA-2-7B [39] compressed by RResSVD, conventional SVD, and existing post-training baselines under compression ratios ranging from 20% to 60% across 10 benchmark datasets. The results for LLaMA-7B are presented in Table 1, while those for LLaMA-2-7B are provided in Appendix A.2. On the three language mnodeling datasets, WikiText-2 [29], PTB [28],and C4 [32], ResSVD consistently outperforms all baselines across evaluated compression ratios. In particular, on PTB and C4, the improvements are more pronounced, suggesting that ResSVD exhibits stronger generalization capability. More importantly, under a relatively high compression ratio (e.g.,50%), ResSVD still achieves substantial perplexity reductions compared to the existing best-performing baseline SVD-LLM [41]: 7% on WikiText-2, 14% on PTB, and 41% on C4. This demonstrates that ResSVD maintains an obvious performance advantage even under aggressive compression. In addition, on seven common sense reasoning datasets, ResSVD surpasses the existing best baseline on the majority of datasets, with an average accuracy gain of up to 9%, and a minimumn improvement of 3% improvement, further highlighting its robustness and overall effectiveness.

**Performance** **on** **Larger** **Scale** **LLMs.** To examine the scalability and robustness of ResSVD,we evaluate its performance on larger scale LLMs from two representative LLM families: LLaMA

²https://github.com/EleutherAI/1m-evaluation-harness

3https://github.com/pytorch/pytorchandHuggingFace

4https://github.com/huggingface/transformers

<!-- 7 -->

**Performance on Different LLM fam-**ilies. To evaluate the generalization ability of ResSVD,we apply it to four LLMs from different families, including OPT-6.7B [46],LLaMA-2-7B,Mistral-7B [17], and Vicuna-7B [5].As shown in Table 3, under 30% compression ra-tio,ResSVD consistently outperforms all baselines across three language mod-eling benchmarks across these diverse ar-chitectures. The most significant relative improvement reaches 75% on LLaMA-2-7B.Moreover, the best overall improve-ment is achieved on Mistral-7B, withper-plexity reductions of 71% on WikiText-2,45% on PTB, and 46% on C4. We re-produce ASVD [44] and SVD-LLM [41] using their public codebases. While ASVD fails in certain cases due to nu-merical instability (denoted as NaN in the table), ResSVD consistently main-tains stable and reliable performance.

Table 2: Perplexity (↓) of larger-scale LLMs under 20% compression ratio on PTB [28].

<table border="1" ><tr>
<td>METHOD</td>
<td>LLaMA-13B </td>
<td>LLaMA-30B </td>
<td> LLaMA-2-13B</td>
<td>OPT-13B</td>
<td>OPT-30B</td>
</tr><tr>
<td>SVD</td>
<td>1878.04</td>
<td>555.55</td>
<td>5464.57</td>
<td>1552.55</td>
<td>250.49</td>
</tr><tr>
<td>ASVD [44]</td>
<td>12.42</td>
<td>31.36</td>
<td>81.00</td>
<td>36.47</td>
<td>30.95</td>
</tr><tr>
<td>SVD-LLM [41]</td>
<td>12.17</td>
<td>9.10</td>
<td>88.13</td>
<td>14.86</td>
<td>12.94</td>
</tr><tr>
<td>ResSVD</td>
<td>9.70</td>
<td>8.41</td>
<td>66.47</td>
<td>13.22</td>
<td>12.89</td>
</tr></table>

Table 3: Perplexity (↓) of different LLM structures un-der 30% compression ratio.

<table border="1" ><tr>
<td>MODEL</td>
<td>METHOD</td>
<td>WikiText-2↓</td>
<td>PTB↓</td>
<td>C4↓</td>
</tr><tr>
<td rowspan="4">OPT-6.7B</td>
<td>SVD</td>
<td>116067.28</td>
<td>86760.50</td>
<td>168165.89</td>
</tr><tr>
<td>ASVD [44]</td>
<td>26.67</td>
<td rowspan="2">71.36<br>37.46</td>
<td rowspan="2">44.51<br>40.35</td>
</tr><tr>
<td>SVD-LLM[41]</td>
<td>28.03</td>
</tr><tr>
<td>ResSVD</td>
<td>17.10(↓36%)</td>
<td>27.24(↓27%)</td>
<td>38.40(↓5%)</td>
</tr><tr>
<td rowspan="4">LLaMA-2-7B</td>
<td>SVD</td>
<td>30373.39</td>
<td>48930.94</td>
<td>36905.54</td>
</tr><tr>
<td>ASVD [44]</td>
<td>984.03</td>
<td>NaN</td>
<td>NaN</td>
</tr><tr>
<td>SVD-LLM[41]</td>
<td>10.66</td>
<td>292.90</td>
<td>34.96</td>
</tr><tr>
<td>ResSVD</td>
<td>10.32(↓3%)</td>
<td>73.04(↓75%)</td>
<td>23.28(↓33%)</td>
</tr><tr>
<td rowspan="4">Mistral-7B</td>
<td>SVD</td>
<td>59569.54</td>
<td>57830.63</td>
<td>78168.24</td>
</tr><tr>
<td>ASVD [44]</td>
<td>221.66</td>
<td>927.15</td>
<td>266.04</td>
</tr><tr>
<td>SVD-LLM[41]</td>
<td>48.94</td>
<td>193.22</td>
<td>56.55</td>
</tr><tr>
<td>ResSVD</td>
<td>14.09(↓71%)</td>
<td>105.37(↓45%)</td>
<td>30.72(↓46%)</td>
</tr><tr>
<td rowspan="4">Vicuna-7B</td>
<td>SVD</td>
<td>24835.33</td>
<td>24510.90</td>
<td>29368.55</td>
</tr><tr>
<td>ASVD [44]</td>
<td>106.32</td>
<td>NaN</td>
<td>NaN</td>
</tr><tr>
<td>SVD-LLM[41]</td>
<td>12.42</td>
<td>104.27</td>
<td>39.55</td>
</tr><tr>
<td>ResSVD</td>
<td>11.57(↓7%)</td>
<td>69.28(↓34%)</td>
<td>27.24(↓31%)</td>
</tr></table>

(including LLaMA-13B,LLaMA-30B, and LLaMA-2-13B) and OPT (OPT-13B and OPT-30B).As presented in Table 2, ResSVD consistently achieves superior performance over existing baselines under 20% compression ratio, demonstrating robust effectiveness across varying model scales.

### 4.3 Ablation Study

In this section, we present extensive ablation study results to assess the effectiveness and robustness of **ResSVD** from three key perspectives. ① **Effectiveness of residual compensation and partial-layer** **compression:** We evaluate the individual contributions of the residual compensation mechanism and partial-layer compression. ③ **Impact** of **layer** **selection** **strategy:** We show that restricting com-pression to the last few layers mitigates error accumulation and consistently improves performance. ③ **Impact** of **calibration** **data:** We analyze the effects of both calibration sample size and dataset choice on the performance of compressed models.

**Effectiveness of residual compensation and partial-layer compression.** As shown in Table 4, we conduct ablation studies to evaluate the individual contributions of residual compensation (REC) and partial-layer compression (PLC). Specifically, the left table presents the results for REC under multiple compression ratios. Incorporating REC consistently reduces perplexity across all settings, and ResSVD achieves the best performance across all datasets and baselines. The right table reports the results for PLC. Without PLC, perplexity on WikiText-2 [29] increases slightly, indicating the effect of cumulative error. In contrast, applying PLC consistently enhances performance,further amplifying the benefits of REC. These results confirm that PLC is essential for mitigating error propagation. More ablation results are provided in Appendix A.3.

**Impact** of **layer** **selection** **strategy.** We compare the performance of LLaMA-7B [38] under 20% compression ratio with different layer selection strategies. The results are shown in Table 5. We observe that the performance of the compressed model varies significantly depending on which layers are selected for compression. The worst performance is observed when compressing the first 8 layers, primarily because the compression ratio per layer becomes excessively high, and error accumulation begins from the first layer. Across all configurations, compressing the last 16 layers yields the best overall performance. However, as more layers are compressed, the performance gradually declines, which can be attributed to increased cumulative approximation error.

**Impact** of **calibration** **data.** The selection of calibration data plays a critical role in post-training compression. To assess this, we vary both the number of calibration samples and the choice of calibration datasets. Figure 6 shows the perplexity on WikiText-2 [29] and average zero-shot accuracy under 20% compression ratio. While performance slightly improves with more calibration samples,

<!-- 8 -->

Table 4: Ablation results of residual compensation (REC) and partial-layer compression (PLC)on LLaMA-7B [38] under compression ratios 20% and 30%. More ablation results are in Table 8.

(a) REC

<table border="1" ><tr>
<td>COMP. RATIO</td>
<td>METHOD</td>
<td>REC</td>
<td>WikiText-2↓</td>
<td>PTB↓</td>
<td>C4↓</td>
</tr><tr>
<td rowspan="5">20%</td>
<td>SVD</td>
<td>x</td>
<td>20082.86</td>
<td>20338.96</td>
<td>18784.20</td>
</tr><tr>
<td rowspan="2">ASVD [44]SVD-LLM[41]</td>
<td>x</td>
<td>9.27</td>
<td>15.09</td>
<td>13.68</td>
</tr><tr>
<td>x</td>
<td>7.89</td>
<td>16.54</td>
<td>15.92</td>
</tr><tr>
<td>ResSVD</td>
<td>x</td>
<td>7.93</td>
<td>13.68</td>
<td>13.72</td>
</tr><tr>
<td>ResSVD</td>
<td>✓</td>
<td>7.47</td>
<td>12.27</td>
<td>12.22</td>
</tr><tr>
<td rowspan="4">30%</td>
<td>SVD</td>
<td>x</td>
<td>13155.97</td>
<td>17354.46</td>
<td>21012.91</td>
</tr><tr>
<td>ASVD [44]SVD-LLM [41]</td>
<td>x ×</td>
<td>222.98<br>9.52</td>
<td>586.79<br>28.97</td>
<td>148.79<br>26.38</td>
</tr><tr>
<td>ResSVD</td>
<td></td>
<td>9.60</td>
<td>26.12</td>
<td>20.97</td>
</tr><tr>
<td>ResSVD</td>
<td></td>
<td>9.52</td>
<td>20.31</td>
<td>18.29</td>
</tr></table>

Table 5: Performance of compressed LLaMA-7B [38] with various layer selection strategies.

<table border="1" ><tr>
<td>COMP. LAYERS</td>
<td>LAYER COMP. RATIO</td>
<td>WikiText-2↓</td>
<td>PTB↓</td>
<td>C4↓</td>
<td>Average↑</td>
</tr><tr>
<td>First 8</td>
<td>80%</td>
<td>16.56</td>
<td>37.58</td>
<td>46.16</td>
<td>0.33</td>
</tr><tr>
<td>Last 8</td>
<td>80%</td>
<td>10.69</td>
<td>19.26</td>
<td>16.06</td>
<td>0.38</td>
</tr><tr>
<td>Last 12</td>
<td>53%</td>
<td>8.49</td>
<td>14.04</td>
<td>13.50</td>
<td>0.44</td>
</tr><tr>
<td>Last 16</td>
<td>40%</td>
<td>7.47</td>
<td>12.27</td>
<td>12.22</td>
<td>0.47</td>
</tr><tr>
<td>Last 20</td>
<td>32%</td>
<td>8.07</td>
<td>14.19</td>
<td>13.68</td>
<td>0.46</td>
</tr><tr>
<td>Last 24</td>
<td>27%</td>
<td>8.08</td>
<td>14.66</td>
<td>13.94</td>
<td>0.45</td>
</tr><tr>
<td>Last 28</td>
<td>23%</td>
<td>8.21</td>
<td>16.04</td>
<td>14.98</td>
<td>0.45</td>
</tr><tr>
<td>Last 32</td>
<td>20%</td>
<td>8.27</td>
<td>16.00</td>
<td>15.78</td>
<td>0.44</td>
</tr></table>

(b) PLC

<table border="1" ><tr>
<td>COMP. RATIO</td>
<td>METHOD</td>
<td>PLC</td>
<td>WikiText-2↓</td>
<td>PTB↓</td>
<td>C4↓</td>
</tr><tr>
<td rowspan="5">20%</td>
<td>SVD</td>
<td>x</td>
<td>20082.86</td>
<td>20338.96</td>
<td>18784.20</td>
</tr><tr>
<td rowspan="2">ASVD [44]<br>SVD-LLM [41]</td>
<td>x</td>
<td>9.27</td>
<td>15.09</td>
<td>13.68</td>
</tr><tr>
<td>x</td>
<td>7.89</td>
<td>16.54</td>
<td>15.92</td>
</tr><tr>
<td>ResSVD</td>
<td>x</td>
<td>8.27</td>
<td>16.00</td>
<td>15.78</td>
</tr><tr>
<td>ResSVD</td>
<td>✓</td>
<td>7.47</td>
<td>12.27</td>
<td>12.22</td>
</tr><tr>
<td rowspan="5">30%</td>
<td>SVD</td>
<td>x</td>
<td>13155.97</td>
<td>17354.46</td>
<td>21012.91</td>
</tr><tr>
<td rowspan="2">ASVD [44]<br>SVD-LLM [41]</td>
<td>x</td>
<td>222.98</td>
<td>586.79</td>
<td>148.79</td>
</tr><tr>
<td>x</td>
<td>9.52</td>
<td>28.97</td>
<td>26.38</td>
</tr><tr>
<td>ResSVD</td>
<td>X</td>
<td>10.11</td>
<td>24.49</td>
<td>23.82</td>
</tr><tr>
<td>ResSVD</td>
<td></td>
<td>9.52</td>
<td>20.31</td>
<td>18.29</td>
</tr></table>

Table 6: Perplexity (↓) and zero-shot average accuracy (↑) of LLaMA-7B [38] compressed by **ResSVD** under 20% compression ratio while using different calibration datasets.

<table border="1" ><tr>
<td>CALIBRATION<br>DATASET</td>
<td>WikiText-2↓</td>
<td>PTB↓</td>
<td>C4↓</td>
<td>Avg↑</td>
</tr><tr>
<td>WikiText-2[29]</td>
<td>7.47</td>
<td>12.27</td>
<td>12.22</td>
<td>0.47</td>
</tr><tr>
<td>PTB [28]</td>
<td>9.86</td>
<td>9.54</td>
<td>12.70</td>
<td>0.45</td>
</tr><tr>
<td>C4[32]</td>
<td>10.05</td>
<td>12.42</td>
<td>10.94</td>
<td>0.47</td>
</tr></table>

the gains are modest, indicating that ResSVD remains robust even with limited calibration data. Moreover, Table 6 shows that the best results are achieved when the calibration and evaluation datasets align, suggesting that domain consistency enhances compression quality.

### 4.4 Efficiency Results

ResSVD not only preserves competitive model performance but also delivers significant inference speedup on hardware. We evaluate the throughput of compressed models on an NVIDIA A100 GPU, with results shown in Figure 7. Models compressed by ResSVD consistently achieve fasterinference than those compressed by SVD-LLM [41]. Moreover, the speedup becomes increasingly significant as the batch size grows, indicating that ResSVD scales more efficiently under larger workloads. These findings highlight the practical efficiency of ResSVD in enabling faster inference while maintaining accuracy, making it well-suited for deployment in resource-constrained environments.

<!-- 7.56 LLaMA-7B 754 E 7.46 64 128 256 512 1024 Number of calibration samples -->
![](./images/7a2e63fcf56ab1f4.jpg)

Figure 6: Impact of the number of calibration data samples Figure 7: Throughput of LLaMA-on LLaMA-7B [38] under 20% compression ratio. (Left) 7B [38] and its 40% compressed ver-Perplexity (↓). (Right) Average accuracy (↑).

<!-- 042 LLsMA-7B 0.000 62 0.466 o - 0.464 042 64 128 256 512 1024 Number of calibration samples -->
![](./images/7d2ccebb3e34e81c.jpg)

<!-- 4000 ResSVD SVD-LLM Original Mlodel 3000 8s/SUaNO 2000 1000 64 128 256 512 Batch Size -->
![](./images/c266349a7fc895e2.jpg)

sions. The sequence length is 32.

## 5 Conclusion

In this work, we propose **ResSVD,** a novel post-training SVD-based compression method for large language models. ResSVD effectively leverages the residual matrix resulting from SVD truncation to reduce the truncation loss and enhance layer-wise reconstruction accuracy. Furthermore, it selectively compresses only the last few layers of the model under a fixed overall compression ratio, thereby significantly mitigating error propagationacross the model. Extensive experiments across various LLM families and benchmark datasets demonstrate that ResSVD consistently outperforms existing SVD-based baselines across various settings. These results highlight the effectivenness and generalizability of ResSVD in enabling efficient LLM deployment.

<!-- 9 -->

## References

[1] A. Amini, S. Gabriel, S. Lin, R. Koncel-Kedziorski,Y.Choi, and H. Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. In NAACL-HLT, 2019.

[2] J.Bai,S.Bai, Y.Chu,Z.Cui,K.Dang, X. Deng, Y.Fan,W.Ge,Y.Han,F.Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609,2023.

[3]Y.Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi. PIQA: reasoning about physical common-sense in natural language. In AAAI, 2020.

[4] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh,D.M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M.Litwin,S.Gray, B.Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language modeIs are few-shot learners. In NeurIPS, 2020.

[5] W.-L.Chiang,Z.Li,Z.Lin,Y.Sheng,Z. Wu,H. Zhang,L. Zheng, S. Zhuang, Y.Zhuang, J.E. Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna.lmsys.org,2(3):6,2023.

[6] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S.Tsvyashchenko,J. Maynez, A. Rao, P.Barnes,Y.Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson,R.Pope, J. Bradbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin, T. Duke,A.Levskaya, S. Ghemawat, S.Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito,D.Luan, H.Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omernick,A.M.Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J.Dean, S.Petrov, and N. Fiedel. Palm: Scaling language modeling with pathways.Journal of Machine Learning Research, 24(240):1-113,2023.

[7] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answvering? try arc, the AI2 reasoning challenge. In CVPR, 2023.

[8] J.Deng,W.Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

[9] J.Devlin, M. Chang, K. Lee, and K. Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT, 2019.

[10] E. Frantar and D. Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot. In ICML,2023.

[11] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh. GPTQ: accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323,2022.

[12] S. Gao, C.-H.Lin, T.Hua, Z. Tang, Y. Shen, H. Jin, and Y.-C. Hsu. Disp-llm: Dimension-independent structural pruning for large language models. InNeurIPS,2024.

[13] Y.Gu,L.Dong, F. Wei, and M. Huang. Minillm: Knowledge distillation of large language models. In ICLR, 2024.

[14] R.A. Horn and C. R. Johnson. Matrix analysis. Cambridge university press,2012.

[15] Y.Hsu,T. Hua, S. Chang, Q. Lou, Y. Shen, and H. Jin. Language model compression with weighted low-rank factorization. In ICLR, 2022.

[16] W.Huang,Y.Liu,H.Qin, Y.Li,S. Zhang, X. Liu, M. Magno, and X. Qi. Billm: Pushing the limit of post-training quantization for llms. In ICML, 2024.

[17] A.Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de Las Casas, F. Bres-sand, G. Lengyel, G. Lample, L. Saulnier, L.R.Lavaud, M. Lachaux, P. Stock, T. L. Scao, T.Lavril, T. Wang, T. Lacroix, and W.E. Sayed. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

<!-- 10 -->

[18] J. Kaplan, S. McCandlish, T. Henighan, T.B. Brown, B. Chess, R. Child, S. Gray, A. Rad-ford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,2020.

[19] A. Kaushal, T. Vaidhya, and I. Rish. LORD: low rank decomposition of monolingual code llms for one-shot compression. arXiv preprint arXiv:2309.14021,2023.

[20] E. Kurtic, E. Frantar, and D. Alistarh. Ziplm: Inference-aware structured pruning of language models. In NeurIPS,2023.

[21] Z.Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. In ICLR, 2020.

[22] Z.Li,M.Xia,J.Zhang, Z. Hui, L. Kong, Y. Zhang, and X. Yang. Adasvd: Adaptive singular value decomposition for large language models. arXiv preprint arXiv:2502.01403,2025.

[23] Z.Li,X.Yan,T.Zhang, H. Qin,D. Xie, J. Tian, L. Kong, Y. Zhang, X. Yang, et al. ARB-LLM: alternating refined binarizations for large language models. arXiv preprint arXiv:2410.03129, 2024.

[24] J.Lin,J.Tang,H.Tang, S. Yang, W. Chen, W. Wang, G. Xiao, X. Dang, C. Gan, and S. Han. AWQ:activation-aware weight quantization for on-device LLM compression and acceleration. In MLSys,2024.

[25] A.Liu,B.Feng,B. Xue,B.Wang,B. Wu, C.Lu, C. Zhao,C. Deng, C.Zhang,C. Ruan,et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437,2024.

[26]H.Liu,C.Li,Y.Li, and Y. J.Lee. Improved baselines with visual instruction tuning. In CVPR, 2024.

[27] X.Ma, G. Fang, and X. Wang. Llm-pruner: On the structural pruning of large language models. In NeurIPS,2023.

[28] M. P. Marcus, B. Santorini, and M. A. Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313-330,1993.

[29] S. Merity, C. Xiong, J. Bradbury, and R. Socher. Pointer sentinel mixture models. In ICLR, 2017.

[30] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. In EMNLP, 2018.

[31] D. A. Patterson, J. Gonzalez, U. Hölzle,Q.V.Le, C. Liang, L. Munguia, D. Rothchild, D.R. So, M. Texier, and J. Dean. The carbon footprint of machine learning training will plateau, then shrink. Computer, 55(7):18-28,2022.

[32] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou,W.Li, and P.J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(1440):1-67,2020.

[33] R. Saha, N. Sagan, V. Srivastava, A. Goldsmith, and M. Pilanci. Compressing large language models using low rank and low precision decomposition. In NeurIPS, 2024.

[34] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi. Winogrande: an adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99-106,2021.

[35] Y.Sheng,L.Zheng,B. Yuan, Z. Li, M. Ryabinin, B. Chen, P. Liang, C. Ré, I. Stoica,and C.Zhang. Flexgen1: High-throughput generative inference of large language models with a single GPU. In ICML, 2023.

[36] E. Strubell, A. Ganesh, and A. McCallum. Energy and policy considerations for modern deep learning research. In AAAI, 2020.

[37] M. Sun, Z. Liu, A. Bair, and J. Z. Kolter. A simple and effective pruning approach for large language models. In ICLR, 2024.

[38] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière,N.Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971,2023.

<!-- 11 -->

[39] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, NN. Bashlykov, S.Ba-tra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D.Esiobu, J.Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N.Goyal,A.Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardlas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, PP.S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskoovich, Y.Lu,Y.Mao,X.Martinet, T. Mihaylov, P.Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P.Xu,Z.Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang,A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288,2023.

[40]W.Wang,W.Chen,Y.Luo,Y.Long, Z. Lin, L. Zhang, B. Lin, D. Cai, and X. He. Model compression and efficient inference for large language models: A survey. arXiv preprint arXiv:2402.09748,2024.

[41] X.Wang,Y.Zheng,Z. Wan, and M. Zhang. SVD-LLM: truncation-aware singular value decomposition for large language model compression. In ICLR, 2025.

[42] X. Xu,M.Li,C. Tao,T. Shen,R. CCheng, J. Li, C. Xu, D. Tao, and T. Zhou. A survey on knowledge dlistillation of large language models. arXiv preprint arXiv:2402.13116,2024.

[43]C.Yang,W.Lu,Y.Zhu,Y.Wang,Q.Chen,C. Gao,B. Yan, and Y.Chen. Survey on knowledge distillation for large language models: Methods, evaluation, and application. ACM Transactions on Intelligent Systems and Technology, 2024.

[44] Z.Yuan,Y.Shang,Y.Song,Q. Wu, Y. Yan, and G. Sun. ASVD: activation-aware singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821,2023.

[45] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. Hellaswag: Can a machine really finish your sentence? In ACL,2019.

[46] S.Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. T. Diab, X.Li, X. V.Lin,T.Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A.Sridhar, T.Wang,and L. Zettlemoyer. OPT: open pre-trained transformer language models. arXiv preprint arXiv:2205.01068,2022.

[47] S.Zhang,X.Zhang, Z. Sun, Y. Chen, and J. Xu. Dual-space knowledge distillation for large language models. In EMNLP, 2024.

[48] Z.Zhou, X. Ning, K. Hong, T. Fu, J. Xu, S.Li,Y.Lou,L.Wang, Z. Yuan, X.Li,et al. A survey on efficient inference for large language models. arXiv preprint arXiv:2404.14294,2024.

[49] X.Zhu,J.Li,Y.Liu,C.Ma,and W.Wang. A survey on model compression for large language models. In ACL, 2024.

<!-- 12 -->

## A Appendix

### A.1 Pseudocode

Algorithm 2 and Algorithm 3 present the pseudocode for **partial-layer** **compression** and **residual** **compensation,** respectively. Algorithm 2 identifies the optimal number of last layers k★ to compress, along with their corresponding layer compression ratio Rl★ . Specifically, for a model with N layers, we iterate over candidate values ofk using a step size s and compute the associated Rl for each. For each candidate configuration, we invoke Algorithm 3 to perform the compression, after which we compute the final-layer error relative to the original model. The configuration yieldingthe lowest final-layer error is selected as the optimal compression setting.

#### Algorithm 2 Pseudocode of Partial-layer Compression

**Input**: Original LLM: M, number of model layers:N, overall compression ratio:Rointermediate rank:r1,step:s

**Output**: Number of layers to compress:k, layer compression ratio: Rl

1: CD←Randomly collect calibration samples from the dataset

2:SetW←M

Extract the set of weight matrices in M

3:Setk←{k|k=s,2s,⋯,N-s}

Obtain candidate compressed layer numbers

4:LowestErr←+∞

5:**for** each k in Setk**do**

6:

$$R_{l}\leftarrow \left(N·R_{o}\right)/$$

Calculate corresponding layer compression ratio

7:

**if**Rl≥0**then**

8:

$$M_{TMP}\leftarrow \text {DEEP_{CoPY}}(M)$$

9:

**for**i∈[N-k+1,N]do

10:

Wi←SetW(i),Si←WHITENING(Wi,CD) ΔInitialize weight matrices to comnpress

11:

SW′(i)←RESIDUALCOMPENSATION(WiSi.,k,.Rl,r1)

12:

Replace weights of layer i inMTMPwithSetW′(i)

13:

**end for**

14:

Err←CAL ERROR(M,MTMP,CD, Layer(N)) Calculate the last layer output error

15:

ifErr&lt;Lowest\_Errthen

16:

$$\text {Lowest_{Err}}\leftarrow \text {Err}$$

Search for the lowest layer-wise error

17:

$$k^{\star }\leftarrow k$$

18:

$$R_{l}^{\star }\leftarrow R_{l}$$

19:

**end if**

20:

**end if**

21**:end for**

22: **return**k★,Rl★

### A.2 Overall Performance of LLaMA-2-7B

We evaluate the perplexity and zero-shot accuracy of LLaMA-2-7B [39] to assess the effectiveness of ResSVD, with results reported in Table 7. On the three language modeling datasets, ResSVD outperforms all baselines across the evaluated compression ratios, except at the 60% compression ratio on WikiText-2 [29]. In particular, compared to the existing best baseline, the relative improvement ranges from 3% to 75%. For compression ratios of 40%, 50%, and 60%, we also include results from the recent method AdaSVD [22], using values reported in their paper due to the lack of an open-source codebase. ResSVD continues to outperform AdaSVD across these settings. Moreover, on zero-shot accuracy across seven common sense reasoning datasets, ResSVD achieves the best performance on most benchmarks, with an average improvement of up to 16%. These results further demonstrate the robustness and overall effectiveness of ResSVD.

### A.3 More Ablation Results

The ablation results for LLaMA-7B [38] under 40%, 50%, and 60% compression ratios are presented in Table 8. A similar trend to that observed at 20% and 30% compression holds here: incorporating both REC and PLC leads to a substantial reduction in perplexity across all settings. Moreover,

<!-- 13 -->

#### Algorithm 3 Pseudocode of Residual Compensation

**Input**: Original LLM: M, weight matrix: W∈Rm×n, number of model layers: N, overall compression ratio:R,step: S intermediate rank:r1

**Output**: Compressed weight matrix setSetW′(i) of layer i

1: CD← Randomly collect calibration samples from the dataset

2:SW←M

Extract the sets of weight matrices in M to compress

3:Sets←\;WHITENING(SW,CD),SW′←\emptyset

Initialize sets of weight matrices

4: k★,Rt←PARTIAL-LAYER COMPRESSION(M,N,Ro,s.

5: r=m·n·(1-l★)/(m+n),r2=r-r1

6:fori∈[N-k★+1,N]do

7:

$$\boldsymbol {W}_{i}\leftarrow \text {Set}_{\boldsymbol {W}}(i),\boldsymbol {S}_{i}\leftarrow \text {Set}_{\boldsymbol {S}}(i)$$

Extract whitening matrices of current weight matrices

8:

$$\boldsymbol {U}_{i,r_{1}},\boldsymbol {Σ}_{i,r_{1}},\boldsymbol {V}_{i,r_{1}}^{T}\leftarrow \boldsymbol {W}_{i,r_{1}}\leftarrow \text {SVD_{T}}\quad \text {RUNC}\left(\boldsymbol {W}_{i}\boldsymbol {S}_{i}\right)$$

Apply the first SVD truncation

9:

$$\boldsymbol {R}_{i}\leftarrow \text {CAL_{RES}}\left(\boldsymbol {W}_{i},\boldsymbol {W}_{i,r_{1}}\right)$$

Calculate residual matrices

10:

$$\boldsymbol {U}_{i,r_{2}},\boldsymbol {Σ}_{i,r_{2}},\boldsymbol {V}_{i,r_{2}}^{T}\leftarrow \boldsymbol {R}_{i,r_{2}}\leftarrow \text {SVD_{TRUNC}}\left(\boldsymbol {R}_{i}\right)$$

Apply the second SVD truncation

11:

$$\hat {\boldsymbol {U}}_{i,r_{1}}\leftarrow \text {MUL}\left(\boldsymbol {U}_{i,r_{1}},\sqrt {\boldsymbol {Σ}_{i,r_{1}}}\right),\hat {\boldsymbol {V}}_{i,r_{1}}\leftarrow \text {MUL}\left(\sqrt {\boldsymbol {Σ}_{i,r_{1}}},\boldsymbol {V}_{i,r_{1}}^{T}\right)$$

Absorb singular values

12:

$$\hat {\boldsymbol {U}}_{i,r_{2}}\leftarrow \text {MUL}\left(\boldsymbol {U}_{i,r_{2}},\sqrt {\boldsymbol {Σ}_{i,r_{2}}}\right),\hat {\boldsymbol {V}}_{i,r_{2}}\leftarrow \text {MUL}\left(\sqrt {\boldsymbol {Σ}_{i,r_{2}}},\boldsymbol {V}_{i,r_{2}}^{T}\right)$$

Absorb singular values

13:

$$\hat {\boldsymbol {U}}_{i,r}\leftarrow \text {ComB}\left(\hat {\boldsymbol {U}}_{i,r_{1}},\hat {\boldsymbol {U}}_{i,r_{2}}\right),\hat {\boldsymbol {V}}_{i,r}\leftarrow \text {ComB}\left(\hat {\boldsymbol {V}}_{i,r_{1}},\hat {\boldsymbol {V}}_{i,r_{2}}\right)$$

Combine weight matrices

$$\text {Set}_{\boldsymbol {W}^{\prime }}\leftarrow \left(\hat {\boldsymbol {U}}_{i,r},\hat {\boldsymbol {V}}_{i,r}\right)\cup \text {Set}_{\boldsymbol {W}^{\prime }}$$

ΔAppend decomposed weight matrices

14**: end for**

15: **return** SetW′

Table 7: Overall performance of LLaMA-2-7B [39] compressed by **ResSVD** and baselines under 20% to 60% compression ratios ("RATIO"), including performance on three language modeling datasets (measured by perplexity (↓)) and zero-shot performance on seven common sense reasoning datasets (measured by individual and average accuracy (↓)). The best results are marked in bold.Blue arrows within parentheses highlight the relative improvement over the existing second-best baseline. NaN denotes evaluation failure due to numerical instability.

<table border="1" ><tr>
<td>RATIO</td>
<td>METHOD</td>
<td>WikiText-2↓</td>
<td>PTB↓</td>
<td>C4↓</td>
<td>Openb.↑</td>
<td>ARC↑</td>
<td>WinoG.↑</td>
<td>HellaS.↑</td>
<td>ARC↑</td>
<td>PIQA↑</td>
<td>MathQA↑</td>
<td>Average↑</td>
</tr><tr>
<td></td>
<td>Original</td>
<td>5.47</td>
<td>26.84</td>
<td>7.28</td>
<td>0.31</td>
<td>0.69</td>
<td>0.67</td>
<td>0.56</td>
<td>0.40</td>
<td>0.78</td>
<td>0.27</td>
<td>0.53</td>
</tr><tr>
<td rowspan="4">20%</td>
<td>SVD</td>
<td>18208.79</td>
<td>59320.78</td>
<td>27131.56</td>
<td>0.14</td>
<td>0.26</td>
<td>0.49</td>
<td>0.25</td>
<td>0.22</td>
<td>0.52</td>
<td>0.20</td>
<td>0.30</td>
</tr><tr>
<td>ASVD [44]</td>
<td>9.56</td>
<td>120.74</td>
<td>12.85</td>
<td>0.25</td>
<td>0.59</td>
<td>0.61</td>
<td>0.46</td>
<td>0.32</td>
<td>0.72</td>
<td>0.24</td>
<td>0.45</td>
</tr><tr>
<td>SVD-LLM [41]</td>
<td>8.37</td>
<td>139.68</td>
<td>20.18</td>
<td>0.23</td>
<td>0.50</td>
<td>0.59</td>
<td>0.39</td>
<td>0.26</td>
<td>0.65</td>
<td>0.23</td>
<td>0.41</td>
</tr><tr>
<td>ResSVD</td>
<td>7.63(19%)</td>
<td>45.37(↓62%)</td>
<td>14.73(18%)</td>
<td>0.28</td>
<td>0.61</td>
<td>0.65</td>
<td>0.50</td>
<td>0.35</td>
<td>0.73</td>
<td>0.26</td>
<td>0.48(↑7%)</td>
</tr><tr>
<td rowspan="4">30%</td>
<td>SVD</td>
<td>30373.39</td>
<td>48930.94</td>
<td>36905.54</td>
<td>0.12</td>
<td>0.25</td>
<td>0.49</td>
<td>0.25</td>
<td>0.22</td>
<td>0.51</td>
<td>0.21</td>
<td>0.29</td>
</tr><tr>
<td>ASVD [44]</td>
<td>984.03</td>
<td>NaN</td>
<td>NaN</td>
<td>0.15</td>
<td>0.27</td>
<td>0.51</td>
<td>0.26</td>
<td>0.22</td>
<td>0.53</td>
<td>0.20</td>
<td>0.31</td>
</tr><tr>
<td>SVD-LLM [41]</td>
<td>10.66</td>
<td>292.90</td>
<td>34.96</td>
<td>0.21</td>
<td>0.42</td>
<td>0.55</td>
<td>0.34</td>
<td>0.22</td>
<td>0.60</td>
<td>0.23</td>
<td>0.37</td>
</tr><tr>
<td>ResSVD</td>
<td>10.32(↓3%)</td>
<td>73.04(↓75%)</td>
<td>23.28(↓33%)</td>
<td>0.23</td>
<td>0.51</td>
<td>0.62</td>
<td>0.42</td>
<td>0.29</td>
<td>0.68</td>
<td>0.24</td>
<td>0.43(↑16%)</td>
</tr><tr>
<td rowspan="5">40%</td>
<td>SVD</td>
<td>39524.00</td>
<td>68829.98</td>
<td>56518.95</td>
<td>0.13</td>
<td>0.26</td>
<td>0.50</td>
<td>0.25</td>
<td>0.21</td>
<td>0.52</td>
<td>0.18</td>
<td>0.29</td>
</tr><tr>
<td>ASVD [44]</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>0.15</td>
<td>0.25</td>
<td>0.50</td>
<td>0.26</td>
<td>0.22</td>
<td>0.52</td>
<td>0.18</td>
<td>0.30</td>
</tr><tr>
<td>SVD-LLM [41]</td>
<td>16.11</td>
<td>717.34</td>
<td>61.96</td>
<td>0.16</td>
<td>0.35</td>
<td>0.55</td>
<td>0.30</td>
<td>0.20</td>
<td>0.57</td>
<td>0.23</td>
<td>0.34</td>
</tr><tr>
<td>AdaSVD [22]</td>
<td>14.76</td>
<td>304.62</td>
<td>56.98</td>
<td>0.19</td>
<td>0.41</td>
<td>0.58</td>
<td>0.32</td>
<td>0.23</td>
<td>0.58</td>
<td></td>
<td></td>
</tr><tr>
<td>ResSVD</td>
<td>14.17(↓4%)</td>
<td>136.32(↓55%)</td>
<td>43.19(↓24%)</td>
<td>0.20</td>
<td>0.43</td>
<td>0.57</td>
<td>0.35</td>
<td>0.24</td>
<td>0.63</td>
<td>0.23</td>
<td>0.38(↑12%)</td>
</tr><tr>
<td rowspan="5">50%</td>
<td>SVD</td>
<td>53405.48</td>
<td>39023.05</td>
<td>58547.82</td>
<td>0.15</td>
<td>0.25</td>
<td>0.48</td>
<td>0.25</td>
<td>0.22</td>
<td>0.53</td>
<td>0.18</td>
<td>0.29</td>
</tr><tr>
<td>ASVD [44]</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>0.13</td>
<td>0.26</td>
<td>0.50</td>
<td>0.25</td>
<td>0.23</td>
<td>0.50</td>
<td>0.20</td>
<td>0.30</td>
</tr><tr>
<td>SVD-LLM [41]</td>
<td>27.19</td>
<td>1775.52</td>
<td>129.71</td>
<td>0.14</td>
<td>0.30</td>
<td>0.50</td>
<td>0.28</td>
<td>0.20</td>
<td>0.54</td>
<td>0.23</td>
<td>0.31</td>
</tr><tr>
<td>AdaSVD [22]</td>
<td>25.58</td>
<td>593.14</td>
<td>113.84</td>
<td>0.15</td>
<td>0.34</td>
<td>0.54</td>
<td>0.29</td>
<td>0.20</td>
<td>0.56</td>
<td></td>
<td></td>
</tr><tr>
<td>ResSVD</td>
<td>24.26(↓5%)</td>
<td>286.24(↓52%)</td>
<td>100.34(↓12%)</td>
<td>0.14</td>
<td>0.35</td>
<td>0.55</td>
<td>0.31</td>
<td>0.22</td>
<td>0.59</td>
<td>0.22</td>
<td>0.34(↑10%)</td>
</tr><tr>
<td rowspan="5">60%</td>
<td>SVD</td>
<td>65240.23</td>
<td>79002.21</td>
<td>70659.74</td>
<td>0.14</td>
<td>0.25</td>
<td>0.50</td>
<td>0.25</td>
<td>0.23</td>
<td>0.52</td>
<td>0.19</td>
<td>0.30</td>
</tr><tr>
<td>ASVD [44]</td>
<td>NaN</td>
<td>19581.17</td>
<td>NaN</td>
<td>0.15</td>
<td>0.25</td>
<td>0.50</td>
<td>0.25</td>
<td>0.23</td>
<td>0.52</td>
<td>0.12</td>
<td>0.29</td>
</tr><tr>
<td>SVD-LLM [41]</td>
<td>54.19</td>
<td>3442.74</td>
<td>263.02</td>
<td>0.14</td>
<td>0.26</td>
<td>0.50</td>
<td>0.27</td>
<td>0.20</td>
<td>0.53</td>
<td>0.21</td>
<td>0.30</td>
</tr><tr>
<td>AdaSVD [22]</td>
<td>60.08</td>
<td>2137.28</td>
<td>294.26</td>
<td>0.12</td>
<td>0.27</td>
<td>0.50</td>
<td>0.27</td>
<td>0.20</td>
<td>0.53</td>
<td></td>
<td></td>
</tr><tr>
<td>ResSVD</td>
<td>58.88(-)</td>
<td>991.48(↓54%)</td>
<td>255.70(↓3%)</td>
<td>0.13</td>
<td>0.29</td>
<td>0.52</td>
<td>0.28</td>
<td>0.21</td>
<td>0.55</td>
<td>0.22</td>
<td>0.31(↑3%)</td>
</tr></table>

ResSVD continues to outperform all existing baselines, further validating the effectiveness of these two components under higher compression ratios.

### A.4 Demonstration of Generated Contents

Table 9 shows the generation contents of models compressed by ResSVD compared to the original model under zero-shot conditions. The results indicate that across various input questions, compressed models consistently produce fluent, coherent, and highly relevant responses. Even at 40%compression ratio, the compressed model can also deliver accurate and relevant answers to common sense questions.

<!-- 14 -->

Table 8: Ablation results of Residual Compensation (REC) and Partial-layer Compression (PLC) on LLaMA-7B [38] under compression ratios 40%,50% and 60%.

(a) REC

<table border="1" ><tr>
<td>COMP. RATIO</td>
<td>METHOD</td>
<td>RC</td>
<td>WikiText-2↓</td>
<td>PTB↓</td>
<td>C4↓</td>
</tr><tr>
<td rowspan="5">40%</td>
<td>SVD</td>
<td>X</td>
<td>52326.99</td>
<td>59859.41</td>
<td>47643.04</td>
</tr><tr>
<td rowspan="2">ASVD [44]<br>SVD-LLM [41]</td>
<td rowspan="2">x +</td>
<td rowspan="2">5262.11<br>13.83</td>
<td rowspan="2">8806.33<br>57.07</td>
<td>6522.61</td>
</tr><tr>
<td>48.47</td>
</tr><tr>
<td>ResSVD</td>
<td>x</td>
<td>13.32</td>
<td>56.25</td>
<td>36.61</td>
</tr><tr>
<td>ResSVD</td>
<td>✓</td>
<td>12.92</td>
<td>46.93</td>
<td>30.51</td>
</tr><tr>
<td rowspan="4">50%</td>
<td>SVD</td>
<td>x</td>
<td>130388.72</td>
<td>86721.38</td>
<td>79853.46</td>
</tr><tr>
<td>ASVD [44]SVD-LLM [41]</td>
<td>メX</td>
<td>62726.86<br>24.05</td>
<td>117959.06<br>150.58</td>
<td>77773.84<br>141.87</td>
</tr><tr>
<td>ResSVD</td>
<td>X</td>
<td>23.06</td>
<td>134.91</td>
<td>87.31</td>
</tr><tr>
<td>ResSVD</td>
<td></td>
<td>22.41</td>
<td>128.80</td>
<td>83.80</td>
</tr><tr>
<td rowspan="4">60%</td>
<td>SVD</td>
<td></td>
<td>52326.99</td>
<td>59859.41</td>
<td>47643.04</td>
</tr><tr>
<td>ASVD [44]<br>SVD-LLM [41]</td>
<td>×<br>+</td>
<td>16221.43<br>53.20</td>
<td>20119.36<br>378.19</td>
<td>16561.39<br>310.17</td>
</tr><tr>
<td>ResSVD</td>
<td></td>
<td>51.03</td>
<td>357.94</td>
<td>249.36</td>
</tr><tr>
<td>ResSVD</td>
<td></td>
<td>48.67</td>
<td>323.74</td>
<td>260.06</td>
</tr></table>

(b) PLC

<table border="1" ><tr>
<td>COMP. RATIO</td>
<td>METHOD</td>
<td>PLC</td>
<td>WikiText-2↓</td>
<td>PTB↓</td>
<td>C4↓</td>
</tr><tr>
<td rowspan="5">40%</td>
<td>SVD</td>
<td>X</td>
<td>52326.99</td>
<td>59859.41</td>
<td>47643.04</td>
</tr><tr>
<td>ASVD [44]</td>
<td>x</td>
<td rowspan="2">5262.11<br>13.83</td>
<td rowspan="2">8806.33<br>57.07</td>
<td rowspan="2">6522.61<br>48.47</td>
</tr><tr>
<td>SVD-LLM [41]</td>
<td>x</td>
</tr><tr>
<td>ResSVD</td>
<td>x</td>
<td>14.33</td>
<td>50.96</td>
<td>46.40</td>
</tr><tr>
<td>ResSVD</td>
<td>✓</td>
<td>12.92</td>
<td>46.93</td>
<td>30.51</td>
</tr><tr>
<td rowspan="5">50%</td>
<td>SVD</td>
<td>x</td>
<td>130388.72</td>
<td>86721.38</td>
<td>79853.46</td>
</tr><tr>
<td rowspan="2">ASVD [44]<br>SVD-LLM [41]</td>
<td>x</td>
<td>62726.86</td>
<td>117959.06</td>
<td rowspan="2">77773.84<br>141.87</td>
</tr><tr>
<td>x</td>
<td>24.05</td>
<td>150.58</td>
</tr><tr>
<td>ResSVD</td>
<td>x</td>
<td>25.92</td>
<td>147.24</td>
<td>113.76</td>
</tr><tr>
<td>ResSVD</td>
<td></td>
<td>22.41</td>
<td>128.80</td>
<td>83.80</td>
</tr><tr>
<td rowspan="5">60%</td>
<td>SVD</td>
<td>x</td>
<td>52326.99</td>
<td>59859.41</td>
<td>47643.04</td>
</tr><tr>
<td rowspan="2">ASVD [44]<br>SVD-LLM [41]</td>
<td>x</td>
<td>16221.43</td>
<td rowspan="2">20119.36<br>378.19</td>
<td rowspan="2">16561.39<br>310.17</td>
</tr><tr>
<td>x</td>
<td>53.20</td>
</tr><tr>
<td>ResSVD</td>
<td>x</td>
<td>60.12</td>
<td>364.38</td>
<td>304.21</td>
</tr><tr>
<td>ResSVD</td>
<td></td>
<td>48.67</td>
<td>323.74</td>
<td>260.06</td>
</tr></table>

These results highlight the effectiveness of ResSVD in achieving substantial compression without compromising performance.

### A.5 Computation Complexity

ResSVD decomposes the original weight matrixW∈Rm×ninto two low-rank matrices: $\hat {\boldsymbol {W}}_{u}\in$ Rm×rand $\hat {\boldsymbol {W}}_{v}\in \mathbb {R}^{r\times n}$ .The layer compression ratioRlis computed as $R_{l}=1-\frac {(m+n)r}{mn}$ .Under a fixed overall compression ratioRo,the relationship betweenRlandRois given by $R_{l}=\frac {NR_{o}}{k}$ where N is the total number of layers annd k denotes the number of last layers to be compressed.

Given an input activationX∈Rn×m, the original output is computed asY=WX. In the compressed model layer, an intermediate state is first computed as $\boldsymbol {I}=\hat {\boldsymbol {W}}_{v}\boldsymbol {X}$ ,followed byY= $\hat {\boldsymbol {W}}_{u}\boldsymbol {I}.$  The computational complexityof the original model isN·O(m2). For the compressed model,the first(N-k)layers remain uncompressed and retain a complexity of $(N-k)·\hat {\mathcal {O}}\left(m^{2}\right),$ while the compressed k layers incur a cost of k·O(m2r+rm).Thus, the total complexity becomes:

$$(N-k)·\mathcal {O}\left(m^{2}n\right)+k·\mathcal {O}\left(m^{2}r+rnm\right)\tag{13}$$

And the rank r is given by:

$$r=\frac {mn\left(1-R_{l}\right)}{m+n}=\frac {mn\left(k-NR_{o}\right)}{k(m+n)}\tag{14}$$

Substituting the expression for r into Equation 13, we obtain the simplified total complexity:

$$N\left(1-R_{o}\right)·\mathcal {O}\left(m^{2}n\right)\tag{15}$$

Compared to the original computation complexityN·O(m2n). This indicates that the total compu-tation cost is reduced proportionally to the overall compression ratio. For example, withRo=40%, the compressed model requires only 60% of the original computational cost.

### A.6 Limitations

Our strategy is to compress the last few layers of LLMs, which consistently leads to performance improvements. However, for simplicity, we currently apply a uniform compression ratio across all weight matrices within each transformer block. Given that MLP and attention layers may exhibit different sensitivities to compression, assigning module-specific compression ratios in a principled way remains a promising direction for future work.

<!-- 15 -->

Table **9**: Generated contents of LLaMA-7B [38] compressed by **ResSVD.**

<table border="1" ><tr>
<td>QUESTION</td>
<td>COMP. RATIO</td>
<td>ANSWER</td>
</tr><tr>
<td rowspan="4">What is the responsibility of the AI<br>assistant?</td>
<td></td>
<td>The AI assistant is responsible for understanding the context and taking appropriate action. The AI assistant is responsible for understanding the intent,context, environment and taking appro-priate actions. The AI assistant is responsible for interpreting.</td>
</tr><tr>
<td>20%</td>
<td>The AI assistant should be responsible for providing an answer to questions. This means that when asked a question, you should<br>do your best to answer it and not give any information other than what is requested. You should also consider the tone and intent.</td>
</tr><tr>
<td>30%</td>
<td>The AI Assistant's job is to provide assistance tothose who wish for assistance in their lives. It has no duty or obligation to assist those who do not desire assistance, nor does it have any duty to prevent.</td>
</tr><tr>
<td>40%</td>
<td>The AI assistant does not have a specific role. The AI assistant will provide instruactions to follow a path, and will give directions as an instruction to follow the path.</td>
</tr><tr>
<td rowspan="4">What is the universe? Explain this<br>concept.</td>
<td></td>
<td>The Universe is everything that exists. It includes all space, time,matter and energy. It also includes everything which has happened or will happen. The universe began about 13.7 billion years ago (the Big Bang) and will.</td>
</tr><tr>
<td>20%</td>
<td>The universe is the whole of space andI time. It consists of all things in space and all things that happen,or have happened. There are many theories on how it began;there were theories that it had no beginning at all.</td>
</tr><tr>
<td>30%</td>
<td>The universe consists of everything in it, including matter and objects, space, the cosmos or outer space, and the galaxy system, as well as other galaxies that make up the Universe. The universe has its own life cycle.</td>
</tr><tr>
<td>40%</td>
<td>In the universe, life is very diverse and complex. The universe is made up of many different objects. These objects are divided into groups. Each group consists of different objects that have specific properties. This group contains the universe,which is large.</td>
</tr><tr>
<td rowspan="4">Why do cars need fuel to run?</td>
<td></td>
<td>Fuel is the lifeblood of a car. Without it, there's nothing to power the pistons and no way to drive. A car needs a constant supply of fuel.</td>
</tr><tr>
<td>20%</td>
<td>The fuel in the tank is burned to produce heat, which produces steam that drives a turbine, which turns a crankshaft. The crankshaft turns the wheels of the car, propelling it forward.</td>
</tr><tr>
<td>30%</td>
<td>The fuel is the fuel in a car and it burns and goes up and up. It's important because it burns the gasoline, which makes it go faster. It's important because you can get energy out.</td>
</tr><tr>
<td>40%</td>
<td>The fuel gives energy for the cars. The gasoline is used by a motor, which produces heat and air, heating up the engine. This allows the engines to travel faster, allowing them to reach higher speed.</td>
</tr><tr>
<td rowspan="4">Why do people feel hungry after<br>exercise?</td>
<td></td>
<td>The answer is because the body burns more calories when you are exercising. You may not realize that the body uses up energy<br>to digest food and absorb nutrients from it.</td>
</tr><tr>
<td>20%</td>
<td>Because the body is more efficient at converting food into energy when inactive, so when active it needs to convert food into energy faster. This is why people often get hungry after exercise.</td>
</tr><tr>
<td>30%</td>
<td>The reason people feel hungry after an activity is because they burn energy and energy is metabolised in the body.</td>
</tr><tr>
<td>40%</td>
<td>The reason the body is hungry after a long athletic activity was discovered by scientists in 1976.</td>
</tr></table>

### A.7 Broader Impacts.

By enabling the efficient deployment of LLMs on limited hardware, our method has the potential to broaden access to LLMs. However, we also acknowledge that increased accessibility may lower the barrier to misuse, and thus encourage responsibole application in practice.

<!-- 16 -->

### A.8 Performance on VLMs

We also apply **ResSVD** to the language part of vision-language models (VLMs) such as LLaVA [26], and evaluate its performance on image captioning (Section A.8.1) and visual question answering (Section A.8.2) tasks. All images used in these evaluations are selected from the ImageNet [8] dataset.

#### A.8.1 Performance of Image captioning

As shown in Figure 8, LLaVA-1.5-7B compressed with ResSVD under 20% compression ratio is still able to produce image captions that are faithful to the visual content. This indicates that the model retains its ability to interpret and describe images accurately, despite parameter reduction.


![](./images/b3fa729330e56ac0.jpg)

The image features a man pushing a cart filled with children,with four children sitting in the cart. He is walking down a street,with cars nearby. One car can be seen on the right side,and another car is further away on the right. Another car is located further behind the man, near the left edge of the image.The man appears to be taking care of his children by pushing the cart through the street.


![](./images/3c61fd8732fddfad.jpg)

The image features a sailboat sailing through a body of water,possibly a lake.A person is sitting in the boat, steering it with the sail. The boat's sails are white,indicating it's a sailboat. The boat's presence in the water suggests that it's enjoying a calm day, with no storms or rough waters.In addition to the boat, there are severaI small boats floating in the water.


![](./images/332da0a98660145c.jpg)

The image features a beautiful beach with sand and trees surrounding it. A tree stands near the beach and appears to be growing near the water. The tree's branches extend out over the beach, adding to the scene's scenic charm.A pathway leads to the beach, providing access to the sandy shore.There are also some trees along the beach, adding more visual interest to the scene.


![](./images/2b03997c0b9cdda0.jpg)

In this image,there is a wine bottle sitting on a table next to a number of wine glasses. The wine bottle appears to be red, while the glasses are white in color. The glasses are arranged around the bottle, with some positioned closer to the bottle and others further away. These glasses seem to be ready for wine to be poured into them,creating a scene for wine consumption.

Figure 8: Image captioning results of LLaVA-1.5-7B [26] compressed by **ResSVD** under 20% compression ratio. Captions that accurately describe the image content are highlighted in orange.

#### A.8.2 Performance of Visual Question Answering

Figure 9 presents the visual question answering results of LLaVA-1.5-7B compressed by ResSVD. The model retains strong performance across these question types, indicating that its multi-modal reasoning capability remains intact despite the compression.


![](./images/793356526252dd74.jpg)

Q:What should I do if I meet the shark in the ocean?


![](./images/9067449c27ae5262.jpg)

Q:What are the things should be cautious while visiting?

A:When encountering a shark in the ocean,swimming away from it and avoiding close contact is crucial. Sharks are known to be aggressive predators, and their presence can pose a threat to humans.Swimming away from the shark while maintaining distance and swimming at a faster speed can help avoid being targeted.Additionally,...wearing a safety cage or wearing a protective suit can be helpful.Finally.

A:When visiting the lighthouse located on a rocky beach,one should be cautious about the rocky terrain and the rough sea conditions.The rocky beach might be unstable and difficult to navigate,while the rough sea may pose risks to swimming or boating.Additionally, the lighthouse's location atop the rocks may pose potential hazards if the lighthouse's structure is fragile or weak. To avoid potential accidents...


![](./images/f2184f16186d00ba.jpg)

Q:What is interesting about this image?


![](./images/85dbac2ef0873bd4.jpg)

Q:What do you think I can do in this area?

A: The image is interesting because it features two teddy bears dressed in old clothes and painted to look like men, each wearing paint brushes. The teddy bears are placed next to a paint can, which suggests that they might be part of a painting project. This unusual scene creates a unique combination of objects that deviates from the typical depiction of teddy bears and paint brushes, adding a sense of creativity.

A:In this area, you can watch TV, listen to music, and possibly listen to the radio. The TV is located on a wooden shelf,which also houses other items such as books,and there is a potted plant next to it. A small wooden table nearby offers a place to place other items. Additionally, there is a light in the area that illuminates the room,making it sultable for sitting and watching TV or listening to music. The presence...

Figure 9: Visual question answering outputs generated by LLaVA-1.5-7B [26] compressed using **ResSVD** under 20% compression ratio. Questions (Q) and model answers (A) are provided,correct answers are highlighted in orange to emphasize answer quality retention.

<!-- 17 -->

