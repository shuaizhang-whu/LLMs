<!-- Published as a conference paper at ICLR 2025 -->

<!-- SVD -->
![](https://web-api.textin.com/ocr_image/external/7a03fbfb23ad399f.jpg)

# SVD-LLM: TRUNCATION-AWARE SINGULAR

# VALUE DECOMPOSITION FOR LARGELANGUAGE

# MODEL COMPRESSION

**Xin Wang¹ Yu Zheng² Zhongwei Wan¹ Mi Zhang¹**

¹The Ohio State University 2Michigan State University

https://github.com/AIoT-MLSys-Lab/SVD-LLM

## ABSTRACT

The advancements in Large Language Models (LLMs) have been hindered by their substantial sizes, which necessitates LLM compression methods for practical deployment. Singular Value Decomposition (SVD) offers a promising solution for LLM compression. However, state-of-the-art SVD-based LLM compression meth-ods have two key limitations: truncating smaller singular values may lead to higher compression loss, and the lack of update on the compressed weights after SVD truncation. In this work, we propose SVD-LLM, a SVD-based post-training LLM compression method that addresses the limitations of existing methods. SVD-LLM incorporates a truncation-aware data whitening technique to ensure a direct map-ping between singular values and compression loss. Moreover, SVD-LLM adopts a parameter update with sequential low-rank approximation to compensate for the accuracy degradation after SVD compression. We evaluate SVD-LLM on 10datasets and seven models from three different LLM families at three different scales. Our results demonstrate the superiority of SVD-LLM over state-of-the-arts, especially at high model compression ratios.

## 1 INTRODUCTION

Large Language Models (LLMs) have demonstrated remarkable capabilities in a wide range of tasks such as natural language understanding and generation (Zhao et al., 2023; Gozalo-Brizuela and Garrido-Merchán, 2023). Despite such capabilities, the democratization of LLMs is primarily restricted by their substantial resource demands, which motivates the design of LLM compression methods (Wan et al., 2024a; Wang et al., 2024; Zhu et al., 2024; Zhou et al., 2024).These methods are often performed in a post-training manner without requiringretraining from scratch. Post-training LLM compression methods based on quantization (Yuan et al., 2024; Huang et al., 2024), unstructured pruning (Frantar and Alistarh, 2023), and structured pruning (Ma et al., 2023;Ashkboos et al.,2024; Zhong et al., 2024) have been intensively studied. Despite their success, these methods have certain limitations, such as dependence on specific hardware and low inference speedup. In contrast, compression methods based on low-rank approximation, such as Singular Value Decomposition (SVD) are not limited by those constraints. Moreover, the KV cache of LLMs compressed via SVD at runtime can also be reduced.

Despite these advantages, the potential of SVD for LLM compression has not been fully explored. Several SVD-based LLM compression methods, such as FWSVD (Hsu et al., 2022) and ASVD (Yuan et al., 2023) have been proposed. However, these methods exhibit severe performance degradation when model compression ratio¹ increases. Such limitation can be attributed to two fundamental issues involved in their approaches. **❶ Misalignment between SVD truncation and compression** loss: both FWSVD and ASVD fail to establish a direct relationship between singular values and model compression loss. As a consequence, truncating smaller singular values in SVD could lead to higher compression loss. **❷ Lack of model parameter update after SVD truncation:** as model compression ratio increases, the number of singular values that need to be truncated in SVD increases as well. To compensate for the accuracy degradation caused by truncating a larger number of

¹Model compression ratio refers to the percentage of parameter reduction achieved through compression.

<!-- 1 -->

<!-- Published as a conference paper at ICLR 2025 -->

singular values, it becomes necessary to update the remaining parameters of the compressed model. Unfortunately, existing SVD-based LLM compression methods do not incorporate such update in their design, and thus fail to compensate for the accuracy degradation especially under high model compression ratios.

In this paper, we propose a SVD-based post-training LLM compression method,SVD-LLM,which effectively addresses the two fundamental issues of existing SVD-based LLM compression methods. SVD-LLM differs from them in two key aspects. ❶ **Truncation-Aware** **Data** **Whitening:** supported by theoretical proof, SVD-LLM incorporates a truncation-aware data whitening technique that ensures a direct mapping between singular values and model compression loss. In doing so, the proposed truncation-aware data whitening technique is able to identify which singular values should be trun-cated to incur minimal model compression loss. **❷** **Parameter** **Update** **with** **Sequential Low-rank** **Approximation**: to compensate for accuracy degradation after compression, SVD-LLM sequentially fine-tunes the decomposed low-ranking matrices for a global accuracy recovery.

We compare SVD-LLM with both state-of-the-art SVD-based LLM compression methods as well as pruning and quantization-based LLM compression methods. To demonstrate the generability of SVD-LLM, we conduct our evaluation on a total of 10 datasets and seven models from three different LLM families (LLaMA, OPT, and Mistral) at three different scales (7B, 13B, 30B), and evaluate the performance of SVD-LLM on both GPU and CPU. We highlight three of our findings:

·SVD-LLM outperforms state-of-the-art SVD-based LLM compression methods FWSVD and ASVD across all 10 datasets, three LLM families at three scales by a large margin.

·SVD-LLM also outperforms state-of-the-art structured pruning-based LLM compression methods, including LLM-Pruner, SliceGPT, BlockPruner as well as state-of-the-art 1-bit post-training quantization-based LLM compression methods, including PB-LLM and BiLLM. More importantly, when combined with 2-bit post-training quantization, SVD-LLM outperforms state-of-the-art 1-bit training-required quantization-based LLM compression method OneBit, presenting a new way to achieve state-of-the-art compression performance without incurring expensive retraining.

·LLMs compressed by SVD-LLM are able to achieve inference speedup and memory reduc-tion when deployed on real hardware, including both GPU and CPU. At the same time, SVD-LLM is able to reduce runtime KV cache memory without additional accuracy drop.

## 2 RELATED WORK

**Large** **Language** **Model Compression:** LLMs in general contain billion-scale parameters. Applying conventional model compression methods for LLMs is unfeasible as they necessitate resource-intensive retraining. Given that, post-training methods that avoid retraining in the compression process have been proposed. In general, these methods can be grouped into four categories: unstructured pruning, structured pruning, quantization, and low-rank approximation. Specifically, unstructured pruning methods (Frantar and Alistarh, 2023) set the individual weights of an LLM to zero without changing its shape. However, irregular sparsification of unstructured pruning is difficult to achieve the desired speedup or memory saving. Unlike unstructured pruning, structured pruning methods (Ma et al., 2023; Ashkboos et al., 2024; Zhong et al., 2024) remove entire channels or other structured components from LLMs, making them easier to implement on hardware. One notable contribution is LLM-Pruner (Ma et al., 2023), which groups weight matrices based on their dependency and assigns the pruning ratio to each group based on the estimated importance. Quantization methods (Lin et al., 2024) compress models by reducing the precision of weight matrices of the LLM. However, similar to unstructured pruning, quantization is also difficult to achieve the desired inference speedup due to the lack of hardware support and efficient kernels for low-precision computation (Lin et al., 2024). Recent studies including PB-LLM (Yuan et al., 2024) and BiLLM (Huang et al., 2024) push the frontier to 1-bit quantization. Nevertheless, these approaches often lead to severe accuracy degradation.

**SVD** **for** **Language** **Model Compression**: Singular Value Decomposition (SVD) is a widely used low-rank approximation technique to reduce matrix size by approximating a matrix with two smaller low-ranking matrices (Golub et al., 1987). Given that, SVD is commonly used for model compression. For instance, DRONE (Chen et al., 2021) achieves optimal SVD compression for small language models such as BERT. However, during SVD compression, DRONE caches all the input activations, making it challenging for LLM compression due to excessive memory usage. For LLMs,directly

<!-- 2 -->

<!-- Published as a conference paper at ICLR 2025 -->

<!-- Original Compressed LLM Calibration Data LLM Layer 1 Wu′ Input Activation X Layer 1 Update W′ Layer 2 SVD Cholesky decomposition WS Wu′ X W′v w Wu嫌 Wv Layer 2 Trunc. Update 红 Layer M Whitening MatrixS Wu Wv Layer M Original Truncation-Aware Data ① ② Parameter Update with Sequential Compressed Weights W Whitening Low-rank Approximation Weights W' -->
![](https://web-api.textin.com/ocr_image/external/3d43c90859248a76.jpg)

Figure 1: Overview of SVD-LLM.

applying SVD on the weight matrix without considering the importance of the weights leads to a large compression loss. To address this issue, Hsu et al. (2022) propose FWSVD, which introduces Fisher information to weigh the importance of parameters. However, FWSVD requires a complex gradient calculation that demands substantial computing and memory resources for LLM compression. Another issue of directly applying SVD is that the distribution of activation can affect the compression loss. To address it, Yuan et al. (2023) propose ASVD, which scales the weight matrix by a diagonal matrix that normalizes the impact of input channels on the weights. However, both FWSVD and ASVD do not establish a direct relationship between singular values and compression loss. As a consequence, truncating the smaller singular values may lead to higher compression loss. Moreover, as compression ratio increases, it becomes necessary to update the compressed weights for accuracy recovery. However, both FWSVD and ASVD do not take it into consideration, and thus incur severe accuracy degradation under high compression ratios.

## 3 SVD-LLM

Figure 1 provides an overview of SVD-LLM. At a high level, SVD-LLM is a SVD-based post-training LLM compression method. Specifically,following the standard procedure of post-training LLM compression methods (Frantar and Alistarh, 2023; Yuan et al., 2023; Xiao et al., 2023),SVD-LLM uses a random set of sentences as calibration data to generate activation for truncation-aware data whitening. Given the generated activation, SVD-LLM derives the whitening matrix S through Cholesky decomposition, and then performs SVD to truncate the multiplication of weight matrices W and whitening matrix S to compress the LLM. After truncation, SVD-LLM updates the remaining model parameters with sequential low-rank approximation to recover accuracy. In the followving, we describe both truncation-aware data whitening and paramneter update with sequential low-rank approximation in detail. The pseudocode of SVD-LLM is provided in Appendix A.1.

### 3.1 TRUNCATION-AWARE DATA WHITENING

**Motivation:** Due to high variance of input activation, simply applying vanilla SVD for LLM compression leads to severe accuracy degradation (Yuan et al., 2023). To address this issue, existing methods (Yuan et al., 2023; Hsu et al., 2022) formulate LLM compression as an optimization problem with the following objective:

$$O=\min \left(\left\|WX-W^{\prime }X\right\|_{F}\right)\tag{1}$$

where W is the weight matrix of the original LLM,X is the activation ofW,W′is the compressed weight matrix,and\|X-′X\|Fis the compression loss in the form of Frobenius loss.

Although existing methods attempt to reduce this compression loss during their SVD truncation, they all fail to establish a direct relationship between singular values and compression loss. As a consequence, truncating smaller singular values in SVD could lead to significant compression loss. Taking ASVD (Yuan et al., 2023) as an example, ASVD extracts a diagonal matrixS0from X where each element in the diagonal is the absolute mean value of each channel. It then usesS0to normalize Xand converts WX into(WS0)(S0-1X). Subsequently, SVD is performed on WS0to obtain the decomposed matricesU0Σ0andV0. Lastly, ASVD truncates the smallest singular values inΣ0to obtain the compressed weight matrixW0′=U0×Tuc(Σ0)×V0×S0-1

<!-- 3 -->

<!-- Published as aconference paper at ICLR 2025 -->

<!-- y 0 2.4 1.8 0.9 0.1 Trunc. 0 One Singular Value Multiple Singular Values 0.1 0.9 0.9 0.1 2.4 0.1 Loss: 1.1 0.7 1.9 1.7 -->
![](https://web-api.textin.com/ocr_image/external/7e83586416b51197.jpg)

<!-- Σ 2.4 1.8 0.9 0.1 Trunc. One Singular Value Multiple Singular Values 0.1 0.9 0.9 0.1 2.4 0.1 Loss: 0.1 0.9 2 2 0.9 0.1 2.4 + 0.1 -->
![](https://web-api.textin.com/ocr_image/external/67c5b94157ae3f1e.jpg)

(a) Data Normalization (ASVD) (b) Truncation-Aware Data Whitening(SVD-LLM)

Figure 2: Compression loss(L=\|WX-W′X\|F)of different data preprocessing methods.

Although normalizing the activation improves performnance, ASVD does not establish a direct rela-tionship between singular values and compression loss (a detailed proof is included in Appendix A.2). To better illustrate this point, we show two concrete examples in Figure 2(a). In example $\textcircled {1}$  where only one singular value is truncated, truncating the smallest singular value 0.1 results in a higher compression loss(lo=.1)compared to truncating the second smallest singular value 0.9 (losss= 0.7). In example $\textcircled {2}$ where multiple singular values are truncated, truncating the smallest two singular values 0.9 and 0.1 also leads to a higher loss(l=than truncating 2.4 and 0.1((l=17.As such,truncating the smallest singular values does not lead to minimal loSS.

**Key** **Design:** The key idea of SVD-LLM is to incorporate a truncation-aware data whitening tech-nique that ensures a direct mapping between singular values and compression loss. To achieve this,SVD-LLM enforces the whitened activationS-1Xto be orthonormal such that each channel is independent of each other,i.e.,(S-1X)(S-1X)T=S-1XXT(S-1)T=I,where S is derived through Cholesky decomposition (Meyer, 2000). SVD is then performed on W S to obtain the de-composed matrices U Σ,V,whereU=[u1u2u3⋯ur]Σ=iag(σ1σ2σ3⋯σr),andV= [v1,v2,v3,⋯,vr]. Lastly,the smallest singular values in Σ are truncated (denoted by Tunc.(Σ)) to obtain two low-ranking matricesWu′=Ux[Tuc $(Σ)]^{\frac {1}{2}}$ $,W_{v}^{\prime }=[\text {Tuc}(Σ)]^{\frac {1}{2}}xV^{T}xS^{-1}$ ⋯v and the compressed weight matrixW′=Wu′×Wv′=U×Truc.(Σ)×VT×S-1

Figure 2(b) illustrates the effect of the proposed truncation-aware data whitening method. In example $\textcircled {1}$  where only one singular value is truncated, the compression loss equals to the truncated singular value. In example $\textcircled {2}$ , the compression loss of truncating multiple singular values equals to the square root of the sum of their squares. As such, under the proposed truncation-aware data whitening technique, truncating the smallest singular values leads to minimal compression loss.

In the following, we provide a theoretical proof on why the proposed truncation-aware data whitening technique ensures a dlirect mapping between singular values and compression loss in the case of one singular value (Theorem 3.2) and multiple singular values (Corollary 3.3), respectively.

**Lemma** 3.1. The Frobenius norm of matrixAwith dimension m×ncan be deduced into the square root of the trace of its gram matrix,which is:

$$\|A\|_{F}\Delta q\left(\sum _{j=1}^{n}\sum _{i=1}^{m}\left|a_{ij}\right|^{2}\right)^{\frac {1}{2}}=\left[\text {Trace}\left(A^{T}A\right)\right]^{\frac {1}{2}}\tag{2}$$

LetSVD(WS)denote SVD compression on matrix WS. The compressed weight matrixW′can be expressed asW′=SV(WS)S-1. Using Lemma 3.1, we obtain the compression lossLiwhen truncating theithsingular value ofWSto reduce its rank for compression:

$$L_{i}=\left\|WX-W^{\prime }X\right\|_{F}=\left\|WSS^{-1}X-\mathbf {SVD}(WS)S^{-1}X\right\|_{F}=\left\|(WS-\mathbf {SVD}(WS))S^{-1}X\right\|_{F}\quad =\left\|σ_{i}u_{i}v_{i}^{T}S^{-1}X\right\|_{F}=σ_{i}\text {Trace}\left(u_{i}v_{i}^{T}S^{-1}XX^{T}\left(S^{-1}\right)^{T}v_{i}u_{i}^{T}\right)^{\frac {1}{2}}\tag{3}$$

Since bothU=[u1,u2,u3,⋯,ur]andV=[v1,v2,v3,⋯,vr]are orthonormal matrices, we have:

$$v_{i}^{T}v_{i}=u_{i}^{T}u_{i}=1;v_{i}^{T}v_{j}=u_{i}^{T}u_{j}=0,\forall i\neq j;\text {Trace}\left(v_{i}v_{i}^{T}\right)=\text {Trace}\left(u_{i}u_{i}^{T}\right)=1\tag{4}$$

**Theorem** 3.2. IfS is the Cholesky decompositionoXXT,the compression lossLiequals toσi

<!-- 4 -->

<!-- Published as a conference paper at ICLR 2025 -->

Proof. Since the whitening matrix S is the Cholesky decomposition ofXXT,we haveSST=XXT We can further infer Equation (3) to obtain:

$$L_{i}=σ_{i}\text {Trace}\left(u_{i}v_{i}^{T}v_{i}u_{i}^{T}\right)^{\frac {1}{2}}=σ_{i}\text {Trace}\left(u_{i}\left(v_{i}^{T}v_{i}\right)u_{i}^{T}\right)^{\frac {1}{2}}=σ_{i}\text {Trace}\left(u_{i}u_{i}^{T}\right)^{\frac {1}{2}}=σ_{i}\tag{5}$$

Therefore,Liof truncatingσiequals to the singular valueσiitself.

**Corollary** 3.3. IfS is the Cholesky decomposition oofXXT,truncating the smallest singular values leads to the lowest loss Lcompared to truncating others.

Proof. If we truncateσm+1,σm+2,σm+3,⋯,σrinΣfor compression, the square of the loss L is:

$$=\sum _{i=m+1}^{r}σ_{i}^{2}\text {Trace}\left(u_{i}v_{i}^{T}S^{-1}XX^{T}\left(S^{-1}\right)^{T}v_{i}u_{i}^{T}\right)=\sum _{i=m+1}^{r}\left(L_{i}\right)^{2}=\sum _{i=m+1}^{r}\left(σ_{i}\right)^{2}\quad L^{2}=\left\|\sum _{i=m+1}^{r}σ_{i}u_{i}v_{i}^{T}S^{-1}X\right\|_{F}^{2}=\sum _{j=m+1}^{r}\sum _{i=m+1}^{r}σ_{i}σ_{j}\mathbf {Trace}\left(u_{i}v_{i}^{T}S^{-1}XX^{T}\left(S^{-1}\right)^{T}v_{j}u_{j}^{T}\right)\tag{6}$$

The squared lossL2equals to the sum of the squared singular values (More detailed derivation is in Appendix A.3). Truncating the smallest singular values achieves the lowest compression loss. ☐

Given that our proposed truncation-aware data whitening technique is built upon SVD, whose applicability depends on certain singular value distribution, wve further conduct a spectrum analysis of the singular values obtained by our technique. We refer the readers to Appendix A.4 for details.

It should also be noted that our proposed truncation-aware data whitening technique not only ensures a direct mapping between singular values and compression loss, but is also able to achieve the same theoretical optimal compression loss as DRONE (Chen et al., 2021). However, during SVD compression, DRONE stores all input activations in memory, which poses a challenge for LLM compression due to high memory consumption. In contrast, SVD-LLM incrementally updates its XXTmatrix by adding theTof each new input x, which is considerably smaller than the full input activation. In Appendix A.5, we provide our proof on SVD-LLM achieving the same theoretical optimal compression loss as DRONE and discuss the advantages of SVD-LLM over DRONE in memory saving, compression speed, and numerical stability in details.

### 3.2 PARAMETER UPDATE WITH SEQUENTIAL LOW-RANK APPROXIMATION

**Motivation:** Although the proposed truncation-aware data whitening technique helps preserve the accuracy during compression, as the compression ratio increases, the accuracy of the compressed LLM may degrade given that more larger singular values will be truncated by SVD compression. Therefore, it is necessary to update the remaining parameters in the compressed LLM.

**Key** **Design:** SVD-LLM proposes a variant of LoRA fine-tuning to update the remaining weight parameters of the compressed LLM for accuracy recovery. Specifically, given the two low-ranking matricesWu′Wv′generated by truncation-aware data whitening, instead of directly applying LoRA fine-tuning to the compressed weight matrixW′=Wu′×Wv′as standard LoRA does, we propose to apply LoRA on top ofWu′andWv′separately to preserve their low-rank structures as follows:

$$W_{u}^{\prime }\leftarrow W_{u}^{\prime }+B_{u}A_{u},W_{v}^{\prime }\leftarrow W_{v}^{\prime }+B_{v}A_{v}\tag{7}$$

whereAu,Bu,Av,andBvare matrices used for LoRA fine-tuning.

Simultaneously fine-tuningWu′andWv′will not guarantee a decrease in fine-tuning loss. This is because the derivatives ofWu′andWv′are interdependent during the fine-tuning process,where optimization of one matrix may interfere with the optimization of the other. Therefore, as shown in Figure 1, we propose a sequential low-rank approximation strategy to fine-tuneWu′and Wv′in a sequential manner. Specifically, we first freeze matrixWv′and fine-tuneWu′with LoRA.We then perform the second-round LoRRA fine-tuning on matrixWv′while freezing the updated weight matrix Wu′Finally,we addBu×uandBv×vmatrices toWu′andWv′to obtain the final compressed weight matrices.

<!-- 5 -->

<!-- PublishedI as a conference paper at ICLR 2025 -->

Table 1: Performance of LLaMA-7B compressed by SVD-LLM,SVD-LLM (W),and baselines under different compression ratio (corresponding weight memory is listed inside bracket) on two language modeling datasets (measured by perplexity (↓)), eight common sense reasoning datasets (six measured by both individual and average accuracy (↑), TruthfulQA measured by BLEU score (↑), and GSM8K measured by Exact Match Accuracy (↑)).The best performance is marked in bold. The relative performance gain compared to the best-performing baseline is marked in green inside bracket.

<table border="1" ><tr>
<td>RATIO(MEM.)</td>
<td>METHOD</td>
<td>WikiText-2↓</td>
<td>C4↓</td>
<td>Openb.</td>
<td> ARCc </td>
<td>WinoG.</td>
<td> HellaS.</td>
<td> PIQA </td>
<td>MathQA</td>
<td>Average↑</td>
<td>TruthfulQA↑</td>
<td>GSM8K↑</td>
</tr><tr>
<td>0%(13.5 GB)</td>
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
<td rowspan="5">20%(10.2 GB)</td>
<td>SVD</td>
<td>20061</td>
<td>18800</td>
<td>0.05</td>
<td>0.04</td>
<td>0.01</td>
<td>0.03</td>
<td>0.02</td>
<td>0.03</td>
<td>0.03</td>
<td>0.00</td>
<td>0.00</td>
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
<td>SVD-LLM(W)</td>
<td>7.94(↓29%)</td>
<td>15.84(↓1%)</td>
<td>0.31</td>
<td>0.62</td>
<td>0.61</td>
<td>0.45</td>
<td>0.71</td>
<td>0.21</td>
<td>0.49(19%)</td>
<td>0.26(+0.05)</td>
<td>0.05(+0.01)</td>
</tr><tr>
<td>SVD-LLM</td>
<td>7.73(↓31%)</td>
<td>12.23(↓23%)</td>
<td>0.33</td>
<td>0.67</td>
<td>0.69</td>
<td>0.55</td>
<td>0.79</td>
<td>0.26</td>
<td>0.55(↑22%)</td>
<td>0.28(+0.07)</td>
<td>0.08(+0.04)</td>
</tr><tr>
<td rowspan="5">40%(7.76 GB)</td>
<td>SVD</td>
<td>52489</td>
<td>47774</td>
<td>0.04</td>
<td>0.04</td>
<td>0.05</td>
<td>0.01</td>
<td>0.03</td>
<td>0.02</td>
<td>0.03</td>
<td>0.00</td>
<td>0.00</td>
</tr><tr>
<td>FWSVD</td>
<td>18156</td>
<td>12847</td>
<td>0.06</td>
<td>0.05</td>
<td>0.02</td>
<td>0.00</td>
<td>0.05</td>
<td>0.03</td>
<td>0.04</td>
<td>0.00</td>
<td>0.00</td>
</tr><tr>
<td>ASVD</td>
<td>1407</td>
<td>1109</td>
<td>0.08</td>
<td>0.11</td>
<td>0.09</td>
<td>0.08</td>
<td>0.13</td>
<td>0.08</td>
<td>0.10</td>
<td>0.01</td>
<td>0.00</td>
</tr><tr>
<td>SVD-LLM (W)</td>
<td>13.73(↓99%)</td>
<td>75.42(↓93%)</td>
<td>0.25</td>
<td>0.33</td>
<td>0.55</td>
<td>0.40</td>
<td>0.63</td>
<td>0.12</td>
<td>0.38(↑280%)</td>
<td>0.17(+0.17)</td>
<td>0.02(+0.02)</td>
</tr><tr>
<td>SVD-LLM</td>
<td>9.27(↓99%)</td>
<td>15.63(↓99%)</td>
<td>0.29</td>
<td>0.59</td>
<td>0.68</td>
<td>0.52</td>
<td>0.69</td>
<td>0.20</td>
<td>0.50(↑400%)</td>
<td>0.24(+0.23)</td>
<td>0.07(+0.07)</td>
</tr><tr>
<td rowspan="5">60%(5.35 GB)</td>
<td>SVD</td>
<td>105474</td>
<td>106976</td>
<td>0.01</td>
<td>0.03</td>
<td>0.01</td>
<td>0.00</td>
<td>0.01</td>
<td>0.02</td>
<td>0.01</td>
<td>0.00</td>
<td>0.00</td>
</tr><tr>
<td>FWSVD</td>
<td>32194</td>
<td>29292</td>
<td>0.06</td>
<td>0.02</td>
<td>0.01</td>
<td>0.01</td>
<td>0.02</td>
<td>0.03</td>
<td>0.03</td>
<td>0.00</td>
<td>0.00</td>
</tr><tr>
<td>ASVD</td>
<td>57057</td>
<td>43036</td>
<td>0.05</td>
<td>0.04</td>
<td>0.06</td>
<td>0.09</td>
<td>0.08</td>
<td>0.05</td>
<td>0.06</td>
<td>0.00</td>
<td>0.00</td>
</tr><tr>
<td>SVD-LLM(W)</td>
<td>66.62(↓99%)</td>
<td>471.83(↓99%)</td>
<td>0.10</td>
<td>0.05</td>
<td>0.17</td>
<td>0.10</td>
<td>0.21</td>
<td>0.04</td>
<td>0.11(↑83%)</td>
<td>0.01(+0.01)</td>
<td>0.00(+0.00)</td>
</tr><tr>
<td>SVD-LLM</td>
<td>15.00(↓99%)</td>
<td>26.26(↓99%)</td>
<td>0.18</td>
<td>0.42</td>
<td>0.44</td>
<td>0.31</td>
<td>0.35</td>
<td>0.12</td>
<td>0.30(↑400%)</td>
<td>0.14(+0.14)</td>
<td>0.04(+0.04)</td>
</tr><tr>
<td rowspan="5">80%(2.58 GB)</td>
<td>SVD</td>
<td>687291</td>
<td>708243</td>
<td>0.00</td>
<td>0.01</td>
<td>0.02</td>
<td>0.01</td>
<td>0.01</td>
<td>0.00</td>
<td>0.01</td>
<td>0.00</td>
<td>0.00</td>
</tr><tr>
<td>FWSVD</td>
<td>96872</td>
<td>89243</td>
<td>0.01</td>
<td>0.02</td>
<td>0.00</td>
<td>0.01</td>
<td>0.01</td>
<td>0.00</td>
<td>0.01</td>
<td>0.00</td>
<td>0.00</td>
</tr><tr>
<td>ASVD</td>
<td>80425</td>
<td>67927</td>
<td>0.04</td>
<td>0.03</td>
<td>0.03</td>
<td>0.02</td>
<td>0.01</td>
<td>0.01</td>
<td>0.03</td>
<td>0.00</td>
<td>0.00</td>
</tr><tr>
<td>SVD-LLM(W)</td>
<td>1349(198%)</td>
<td>6224(↓91%)</td>
<td>0.07</td>
<td>0.03</td>
<td>0.04</td>
<td>0.02</td>
<td>0.07</td>
<td>0.01</td>
<td>0.04(↑33%)</td>
<td>0.00(+0.00)</td>
<td>0.00(+0.00)</td>
</tr><tr>
<td>SVD-LLM</td>
<td>31.79(↓99%)</td>
<td>43.71(↓99%)</td>
<td>0.11</td>
<td>0.23</td>
<td>0.21</td>
<td>0.14</td>
<td>0.17</td>
<td>0.08</td>
<td>0.16(↑433%)</td>
<td>0.04(+0.04)</td>
<td>0.02(+0.02)</td>
</tr></table>

## 4 EXPERIMENTS AND ANALYSIS

**Baselines.** We compare SVD-LLM against three groups of methods: (1) Vanilla SVD and state-of-the-art SVD-based LLM compression methods: FWSVD (Hsu et al., 2022), ASVD (Yuan et al., 2023) (Section 4.1) and FLAP (Appendix A.6). (2) Other types of LLM compression methods. These include three state-of-the-art pruning-based LLM compression methods: LLM-Pruner (Ma et al., 2023),SliceGPT (Ashkboos et al., 2024), and BlockPruner (Zhong et al., 2024), and three state-of-the-art quantization-based LLM compression methods: PB-LLM (Yuan et al., 2024),BiLLM (Huang et al., 2024), and OneBit (Xu et al., 2024) (Section 4.4). (3) Smaller LLM StableLM-3B (Tow et al.) pre-trained from scratch (Appendix A.7).

**Models** **and** **Datasets.** To demonstrate the generability of SVD-LLM, we evaluate the performance of SVD-LLM on seven models from three different LLM families at three different scales (LLaMA-7B, 13B,30B,LLaMA2-7B (Touvron et al., 2023), OPT-6.7B (Zhang et al., 2022), Vicuna-7B(Chiang et al., 2023) and Mistral-7B (Jiang et al., 2023)) and 10 datasets including two language modeling datasets (WikiText-2 (Merity et al., 2017), and C4 (Raffel et al., 2020)), six classification datasets (OpenbookQA (Mihaylov et al., 2018), WinoGrande (Sakaguchi et al., 2020), HellaSwag (Zellers et al.,2019),Arce (Clark et al., 2018), PIQA (Bisk et al., 2020), MathQA (Amini et al.,2019)),and two generation datasets (TruthfulQA (Lin et al., 2022) and GSM8K (Cobbe et al., 2021)) with the LM-Evaluation-Harness framework (Gao et al., 2023).

**Implementation** **Details.** To ensure a fair comparison, we followed ASVD (Yuan et al., 2023) to randomly select 256 samples from WikiText-2 as the calibration data. We followed the same configuration used in LLM-Pruner (Ma et al., 2023) to use Alpaca (Taori et al., 2023) dataset with 50K samples for parameter update in SVD-LLM. The inference efficiency experiment is conducted on both NVIDIA A100 GPU and AMD EPYC 7643 CPU while the other experiments are conducted on NVIDIA A100 GPUs.

### 4.1 COMPARISON WITH STATE-OF-THE-ART SVD-BASED LLM COMPRESSION METHODS

First, we compare the performance of SVD-LLM with state-of-the-art SVD-based LLM compression methods from four aspects: (1) performance under different compression ratios, (2) performance on different LLMs, (3) performance on LLMs with larger scales, and (4) compression speed (Ap-pendix A.8). Given that all the SVD-based baselines do not incorporate LoRa fine-tuning, to ensure a fair comparison, we also compare to SVD-LLM with truncation-aware data whitening only(denoted as SVD-LLM (W)).

**Performance under Different Compression Ratios.** We first evaluate the performance of LLaMA-7B compressed by SVD-LLM, SVD-LLM (W), and the SVD-based baselines under compression

<!-- 6 -->

<!-- Published as a conference paper at ICLR 2025 -->

Table 2: Perplexity (↓) of SVD-LLM, SVD-LLM (W), and baselines on WikiText-2 and the average accuracy (↑) of the six common sense reasoning datasets of four different LLMs-OPT-6.7B,LLaMA 2-7B, Mistral-7B, and Vicuna-7B- under 20% compression ratio. The relative performance gain compared to the best-performing baseline is marked in green color inside bracket.

<table border="1" ><tr>
<td></td>
<td colspan="2">OPT-6.7B</td>
<td colspan="2">LLAMA 2-7B</td>
<td colspan="2">MISTRAL-7B</td>
<td colspan="2">VICUNA-7B</td>
</tr><tr>
<td>METHOD</td>
<td>Perplexity↓</td>
<td>Accuracy↑</td>
<td>Perplexity↓</td>
<td>Accuracy↑</td>
<td>Perplexity↓</td>
<td>Accuracy↑</td>
<td>Perplexity↓</td>
<td>Accuracy↑</td>
</tr><tr>
<td>Original</td>
<td>10.86</td>
<td>0.52</td>
<td>5.47</td>
<td>0.57</td>
<td>5.25</td>
<td>0.61</td>
<td>6.78</td>
<td>0.56</td>
</tr><tr>
<td>SVD</td>
<td>66275</td>
<td>0.03</td>
<td>18192</td>
<td>0.09</td>
<td>159627</td>
<td>0.03</td>
<td>18644</td>
<td>0.05</td>
</tr><tr>
<td>FWSVD</td>
<td>14559</td>
<td>0.06</td>
<td>2360</td>
<td>0.12</td>
<td>6357</td>
<td>0.08</td>
<td>2758</td>
<td>0.09</td>
</tr><tr>
<td>ASVD</td>
<td>82.00</td>
<td>0.32</td>
<td>10.10</td>
<td>0.36</td>
<td>13.72</td>
<td>0.32</td>
<td>16.23</td>
<td>0.33</td>
</tr><tr>
<td>SVD-LLM(W)</td>
<td>16.04(↓80%)</td>
<td>0.41(↑28%)</td>
<td>8.50(↓16%)</td>
<td>0.53(↑47%)</td>
<td>10.21(↓26%)</td>
<td>0.42(↑24%)</td>
<td>8.41(↓48%)</td>
<td>0.51(↑55%)</td>
</tr><tr>
<td>SVD-LLM</td>
<td>14.47(↓82%)</td>
<td>0.49(↑53%)</td>
<td>7.73(↓23%)</td>
<td>0.54(↑50%)</td>
<td>7.47(↓45%)</td>
<td>0.55(↑72%)</td>
<td>7.43(↓54%)</td>
<td>0.54(↑64%)</td>
</tr></table>

ratios ranging from 20% to 80% on all 10 datasets. The results are summarized inTable 1. Both SVD-LLM and SVD-LLM (W) consistently outperform vanilla SVD, FWSVD and ASVD across all the compression ratios. In particular, when the compression ratio is 40% and above, SVD-LLM reduces the perplexity by more than 99% on two language modeling datasets and achieves over 400% higher average accuracy on six classification datasets. More importantly, the results on two generation datasets (TruthfulQA, GSM8K) of all three baselines when compression ratios are 60% and above are zero, meaning that the compressed LLMs totally lose their generation ability. In contrast, SVD-LLM still outputs good generation even under 80% compression ratio. Example contents generated by the compressed LLMs are included in Appendix A.9.

**Performance** on **Different LLMs.** To examine Table 3: Perplexity (↓) of SVD-LLM,SVD-LLM the generability of SVD-LLM and SVD-LLM (W) (W),and baselines on WikiText-2 and the aver-across different LLMs, we compare the perfor- age accuracy (↑) of the six classification datasets mance between SVD-LLM and the baselines on of LLaMA-13B and LLaMA-30B under 20% four different models from three different LLM compression ratio. The relative performance families-OPT-6.7B (OPT family),LLaMA 2-7B gain compared to the best-performing baseline (LLaMA family), Mistral-7B (Mistral family),and is marked in green color inside bracket. Vicuna-7B (LLaMA family) - under 20% com-pression ratio on WikiText-2 and six classification datasets. As shown in Table 2, both SVD-LLM and SVD-LLM (W) consistently outperform base-lines on all four LLMs, and exhibits more stable performance across different LLMs, especially compared to vanilla SVD and FWSVD.

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
<td>SVD</td>
<td>946.31</td>
<td>0.21</td>
<td>54.11</td>
<td>0.33</td>
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
<td>SVD-LLM(W)</td>
<td>6.61(↓2%)</td>
<td>0.54(↑0%)</td>
<td>5.63(↓73%)</td>
<td>0.57(↑30%)</td>
</tr><tr>
<td>SVD-LLM</td>
<td>6.43(↓5%)</td>
<td>0.55(↑2%)</td>
<td>5.14(↓75%)</td>
<td>0.59(↑34%)</td>
</tr></table>

**Performance** **on** **LLMs** **with** **Larger** **Scales.** To examine the generability of SVD-LLM and SVD-LLM (W) on LLMs with larger scales, we compare the performance between SVD-LLM and the baselines on LLaMA-13B and LLaMA-30B under 20% compression ratio. As shown in Ta-ble 3, both SVD-LLM and SVD-LLM (W)consistently outperform vanilla SVD, FWSVD, and ASVD on both model sizes.

### 4.2 INFERENCE EFFICIENCY OF SVD-LLM

**Theoretical Analysis of Inference Efficiency.** Assume SVD-LLM compresses the weight matrix W∈Rd×ninto two low-ranking matricesWu′∈Rd×r,Wv′∈Rr×n. The compression ratio is then calculatedas $R_{w}=1-\frac {(d+n)r}{dn}$ 

(1) Compute Complexity Analysis: Given inputX∈R×d,instead of recalculating the full weight matrixW′=Wu′×Wv′and then computing the outputW′×X SVD-LLMcalculates the intermediate stateM=Wv′×Xand then computes the outputY=Wu′×M.In this way, the computation complexity is reduced fromO(2)to ⋅O(2r+r). Taking compression ratioRw=50%as an example,since $R_{w}=1-\frac {(d+n)r}{dn}$ ,we have $r=\frac {dn}{2(d+n)}$ . Then the computation complexity is O $\left(d^{2}r+rnd\right)=O(rd(d+n))=O\left(\frac {d^{2}n}{2}\right)=\frac {1}{2}O\left(d^{2}n\right)$ ,which reduces 50%. In general, given any compression ratioRw,the computation complexity is reduced to1-Rwtimes of the original.

(2) Inference Memory Analysis: Since SVD-LLMdoes not recalculate the full weightW′=Wu′×Wv′,the weight memory is reduced to1-Rwtimes of the original during inference. As another

<!-- 7 -->

<!-- Published as a conference paper at ICLR 2025 -->

<!-- Ratio=80% Ratio=60% Ratio=40% Ratio=20% Original Sas/ 4000 2400 3200 CeS/ 2000 2400 1600 SUeX은L 1600 SUaXQ 1200 800 64 128 256 512 800 32 64 128 256 Batch Size Sequence Length (a) Varying Batch Size on GPU (b)Varying Sequence Length on GPU -->
![](https://web-api.textin.com/ocr_image/external/e3619d6e6f7f41b7.jpg)

<!-- 350 SaS/SueXOL 250 150 50 64 128 256 512 Batch Size -->
![](https://web-api.textin.com/ocr_image/external/ee3da70553886f2a.jpg)

<!-- 120 SaS/SueXOL 80 40 32 64 128 256 Sequence Length -->
![](https://web-api.textin.com/ocr_image/external/97bef5ad340d1949.jpg)

(c) Varying Batch Size on CPU

(d) Varying Sequence Length on CPU

Figure 3: Throughput (tokens/sec) of LLaMA-7B and its compressed version by SVD-LLM under different compression ratios on a single A100 GPU under different batch sizes (a) and different sequence lengths (b) and on a single AMD EPYC 7643 CCPU under different batch sizes (c) and different sequence lengths (d). For (a) and (c), sequence length is 32. For (b) and (d), batch size is 64.

advantage,SVD-LLM is able to reduce the runtime KV cache memory as well (Wan et al., 2024b; 2025; Shen et al., 2024). Specifically, instead of keepingWu′×Wv′×Xin the KV cache, SVD-LLM provides the option to store the intermediate resultM=Wv′×Xin the KV cache and recomputes the original key and value states with the decomposed wveight matrixWu′if required. As such,the runtime KV cache is reduced to $\frac {r}{}=\left(1-R_{w}\right)\times \frac {}{n+}$ times of the original. Moreover, sinceWu′is already stored as the weight matrix in the decomposed LLM, the original intermediate state matrix can still be recovered byWu′without accuracy drop. Therefore, SVD-LLM provides a unified solution that enables simultaneous model compression and KV cache compression.

**Inference** **Speedup** **on** **Hardware.** To quantify inference speedup achieved by SVD-LLM on real hardware, we measure the numbers of tokens that LLaMA-7B and its compressed version by SVD-LLM generate per second (i.e., throughput) under different batch sizes and sequence lengths on a single NVIDIA A100 GPU and a single AMD EPYC 7643 CPU, respectively.The results are shown in Figure 3. We have three observations. (1) Under a specific batch size or sequence length, the speedup achieved by SVD-LLM related to the original model increases as the compression ratio increases. (2) Under a specific compression ratio, the speedup achieved by SVD-LLM related to the original model becomes more significant as the batch size increases or as the sequence length decreases. (3) The above two observations are valid for both GPU and CPU.

**Inference** **Memory** **Reduction** **on** **Hardware.** Lastly,we evaluate the inference memory saving, including both weight memory and runtime KV cache memory saving on a single A100 GPU. Specifically, we measure the peak memory footprint during inference when generating 128 tokens with batch size of 32 using LLaMA-7B compressed by SVD-LLM under different compression ratios w/and w/o considering KV cache reduction. The results are illustrated in Figure 4 where the memory reduction from the dotted line to the blue bars comes mainly from model weight compression and the memory reduction from the blue bars to the yellow bars comes mainly from KV cache compression. As shown, both weight memory saving and runtime KV cache memory saving brought by SVD-LLM are near linear to the compression ratio.

### 4.3 ABLATION STUDY

**Modular** **Sensitivityy Study.** We conduct ablation studies to evaluate the separate contributions of the two keycomponents (i.e., truncation-aware data whitening and parameter update with sequential low-rank approximation) of SVD-LLM. Let SVD-LLM (W) denote the version of SVD-LLM with truncation-aware data whitening only; SVD-LLM (U) denote the version of SVD-LLM with normal

<!-- 8 -->

<!-- Published as a conference paper at ICLR 2025 -->

<!-- SVD-LLM w/o KV cache reduction SVD-LLM w/KV cache reduction 17.7 14.4 3.3GB 5.7GB 8.2GB (a9).uaW Xead 12.0 ↓2.4GB 12.0 10.6GB 2.8GB 9.0 9.5 3.2GB 7.1 6.0 3.6GB 3.0 20% 40% 60% 80% Compression Ratio -->
![](https://web-api.textin.com/ocr_image/external/fed71734dd1ccc2f.jpg)

Figure 4: Peak memory to generate 128 tokens with batch size of 32 using LLaMA-7B com-pressed by SVD-LLM w/ and w/o KV-cache reduc-tion. The dotted line indicates the peak memory of the original LLaMA-7B. The memory reduction from the dotted line to the blue bars mainly comes from the model compression. The memory reduc-tion from the blue to the yellow bars mainly comes from the reduced footprint of the KV cache.

Table 4: Perplexity (↓) of compressed LLaMA-7B on WikiText-2 under different compression ratios. SVD-LLM (W) denotes the version of SVD-LLM with truncation-aware data whiten-ing only; SVD-LLM (U) denotes the version of SVD-LLM with parameter update with sequential low-rank approximation only; The relative per-formance gain compared to ASVD is marked in green color inside bracket.

<table border="1" ><tr>
<td>METHOD</td>
<td>20%</td>
<td>40%</td>
<td>60%</td>
</tr><tr>
<td>ASVD</td>
<td>11.14</td>
<td>1407</td>
<td>57057</td>
</tr><tr>
<td>SVD-LLM(W)</td>
<td>7.94(↓29%)</td>
<td>13.11(↓99%)</td>
<td>42.30(↓99%)</td>
</tr><tr>
<td>SVD-LLM (U)</td>
<td>10.12(↓9%)</td>
<td>19.28(↓99%)</td>
<td>49.88(↓99%)</td>
</tr><tr>
<td>SVD-LLM</td>
<td>7.73(↓31%)</td>
<td>9.27(↓99%)</td>
<td>15.00(↓99%)</td>
</tr></table>

SVD truncation and parameter update with sequential low-rank approximation. As shown in Ta-bole 4, we have three observations. (1) SVD-LLM (W),SVD-LLM (U) and SVD-LLM consistently outperform ASVD across all the compression ratios. Notably, when the compression ratio is at and above 40%, all of them reduce the perplexity by more than 99% compared to ASVD. (2) SVD-LLM consistently outperforms both SVD-LLM (U) and SVD-LLM (W) across all compression ratios. This result demonstrates the unique contribution from each of the two key components and the importance of combining both components to achieve the best performance. (3) Comparing between SVD-LLM (W) and SVD-LLM (U),SVD-LLM (W) achieves a lower perplexity compared to SVD-LLM (U) across all compression ratios. This result indicates that truncation-aware data whitening plays a more significant role than parameter update with sequential low-rank approximation.

**Impact** **of** **Calibration** **Data.** Next, we examine the impact of calibration data on SVD-LLM. Figure 5 and Table 6 summarize the performance of compressed LLaMA-7B when changing three key characteristics of the calibration data: (1) number of calibration data, (2) the seed used to randomly sample the calibration data, and (3) dataset from which calibration data is sampled. As shown, the changes of the three key characteristics on calibration data incur no more than 3% to the final performance, indicating that the sensitivity of SVD-LLM on calibration data is limited.

**Impact** of **Updating** **Order.** Lastly, we examine the impact of uupdating order in the parameter update with sequential low-rank approximation component to the final performance of the compressed LLM. Table 5 shows the performance of compressed LLaMA-7B under 20% to 80% compression ratios on WikiText-2 with different updating orders. As shown, there is only a small difference of the final performance between updating matrixWu′first and updating matrixWv′first. This result indicates SVD-LLM is not sensitive to the updlating order.

More ablation studies are included in Appendix A.10.

### 4.4 COMPARISON WITH OTHER TYPES OF LLM COMPRESSION METHODS

SVD-LLM is orthogonal to other post-training LLM compression methods including pruning and quantization (Wan et al., 2024a; Shen et al., 2025). Lastly, we compare the performance of SVD-LLM with state-of-the-art structured pruning-based and quantization-based LLM compression methods. As discussed in Section 2, since unstructured pruning methods are difficult to achieve its efficiency on hardware, we do not make a comparison with them in this experiment.

**Comparison** **with** **Structured** **Pruning.** First, we compare SVD-LLM with three state-of-the-art structured pruning-based LLM compression methods: LLM-Pruner (Ma et al., 2023), SliceGPT (Ashkboos et al., 2024) and BlockPruner (Zhong et al., 2024) under the same mem-ory budget, ranging from 10 GB to 7 GB on LLaMA-7B using WikiText-2 dataset. As shown in Table 7, SVD-LLM outperformms all three structured pruning-based LLM compression methods. In particular, SVD-LLM achieves up to 56% reduction in perplexity under 7G memory budget.

<!-- 9 -->

<!-- Published as a conference paper at ICLR 2025 -->

<!-- 6.93 MIxeldJad 6.78 6.70 32 64 128 256 512 Number of data (x2048 tokens) -->
![](https://web-api.textin.com/ocr_image/external/b61df381d1b92e45.jpg)

(a) Change of Number

<!-- 1 6.70 3 10 42 57 100 Seed for Random Sampling -->
![](https://web-api.textin.com/ocr_image/external/601b90f1d265ba99.jpg)

(b) Change of Seed

Table 6: Performance of LLaMA-7B compressed by SVD-LLM under 20% compression ratio us-ing calibration data sampled from WikiText-2(by default in our paper) and C4 datasets. The performance on WikiText-2 and C4 are reported by perplexity (↓), while the performance on six Figure 5: Perplexity of LLaMA-7B under 20%downstream datasets are reported by average ac-compression ratio using calibration dlata sampled curacy (↑).The performance on TruthfulQA and with different number or seeds from WikiText-2. GSM8K are reported by BLEU score(↑) and Ex-act Match Accuracy (↑)respectively. The rela-tive pperformance gain for data sampled from one dataset compared to another is mnarked in green color inside bracket.

Table 5: Perplexity of **LLaMA-7B** compressed by SVD-LLM under 20% to 80% compression ratio on WikiText-2 with different updating orders.

<table border="1" ><tr>
<td>UPDATING ORDER</td>
<td>20%</td>
<td>40%</td>
<td>60%</td>
<td>80%</td>
</tr><tr>
<td>Wu′first,then Wv′</td>
<td>7.85</td>
<td>9.32</td>
<td>13.20(↓1%)</td>
<td>31.67(↓1%)</td>
</tr><tr>
<td>Wv′first,thenWu′</td>
<td>7.73(↓2%)</td>
<td>9.27(↓1%)</td>
<td>15.00</td>
<td>31.79</td>
</tr></table>

<table border="1" ><tr>
<td>WikiText-2↓</td>
<td>C4↓</td>
<td>Average↑</td>
<td> |TruthfulQA↑</td>
<td> GSM8K↑</td>
</tr><tr>
<td colspan="5">Calibration data sampled from WikiText-2</td>
</tr><tr>
<td>7.73(↓1%)</td>
<td>12.23</td>
<td>0.55(↑2%)</td>
<td>0.28</td>
<td>0.08</td>
</tr><tr>
<td colspan="5">Calibration data sampled from C4</td>
</tr><tr>
<td>7.79</td>
<td>11.97(↓1%)</td>
<td>0.54</td>
<td>0.28</td>
<td>0.08</td>
</tr></table>

Table 7: Perplexity (↓) of LLaMA-7B compressedTable 8: Perplexity (↓) of LLaMA-7B com-by structured pruning methods and SVD-LLM underpressed by 1-bit quantization methods and various memory budget on WikiText-2.The relative SVD-LLM on WikiText-2. The relative perfor-performance gain compared to the best-performing mance gain compared to the best-performing baseline is marked in green. baseline is marked in green.

<table border="1" ><tr>
<td></td>
<td colspan="4">PERPLEXITY UNDER VARIOUS MEMORY BUDGET</td>
</tr><tr>
<td>METHOD</td>
<td>10 GB</td>
<td>9 GB</td>
<td>8 GB</td>
<td>7GB</td>
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
<td>SVD-LLM</td>
<td>7.92(↓10%)</td>
<td>8.18(↓33%)</td>
<td>8.33(↓49%)</td>
<td>9.63(↓56%)</td>
</tr></table>

<table border="1" ><tr>
<td>METHOD</td>
<td>TYPE</td>
<td>MEMORY</td>
<td>PERPLEXITY</td>
</tr><tr>
<td>PB-LLM</td>
<td>Post-training</td>
<td>1.9 GB</td>
<td>104.83</td>
</tr><tr>
<td>BiLLM</td>
<td>Post-training</td>
<td>1.5 GB</td>
<td>47.67</td>
</tr><tr>
<td>SVD-LLM</td>
<td>Post-training</td>
<td>1.5 GB</td>
<td>47.21(↓1%)</td>
</tr><tr>
<td>OneBit</td>
<td>Training-required</td>
<td>1.3 GB</td>
<td>10.20</td>
</tr><tr>
<td>SVD-LLM(2-bit)</td>
<td>Post-training</td>
<td>1.3 GB</td>
<td>9.83(↓4%)</td>
</tr></table>

**Comparison** **with** **Quantization.** Finally, we compare SVD-LLM with three state-of-the-art quantization-based LLM compression methods: BiLLM (Huang et al., 2024), PB-LLM (Yuan et al., 2024), and OneBit (Xu et al., 2024), which push the frontier to 1-bit quantization.Among them, both BiLLM and PB-LLM are post-training methods, and OneBit is training-required. The results on LLaMA-7B using WikiText-2 dataset are shown in Table 8. We have three observations. (1)Com-pared to post-training methods PB-LLM and BiLLM,SVD-LLM achieves the best performance.(2) Training-required method OneBit outperforms SVD-LLM. This result is expected because OneBit in-volves resource-intensive retraining using large-scale datasets to boost the accuracy after compression. (3) Lastly, we combine SVD-LLM with a 2-bit quantization-based post-training LLM compression method QuIP# (Tseng et al., 2024). This is achieved by first applying SVD-LLM to the LLM under 40% compression ratio, and then applying QuIP# for 2-bit quantization on the compressed model Wu′and.Wv′)generated from SVD-LLM. As shown in Table 8, SVD-LLM outperformns state-of-the-art 1-bit training-required method OneBit without involving resource-intensive retraining. This result demonstrates the potential of a hybrid approach - integrating SVD-based and quantization-based compression techniques - to push the boundaries of post-training LLM compression.

## 5 CONCLUSION

In this work, we present SVD-LLM, a SVD-based post-training LLM compression method. SVD-LLM proposes a truncation-aware data whitening technique to guide which singular values to be truncated with minimal compression loss. It also introduces a sequential low-rank approximation strategy to compensate for accuracy degradation caused by singular value truncation. We evaluate SVD-LLM on 10 datasets and seven models from three LLM families at three scales. Our results demonstrate the superiority of SVD-LLM over state-of-the-arts, especially at high model compression ratios.

## 6 ACKNOWLEDGEMENT

This work is supported in part by NSF Award NeTS-2312675.

<!-- 10 -->

<!-- Published as a conference paper at ICLR 2025 -->

## A APPENDIX.

### A.1 PSEUDOCODE OF SVD-LLM

Algorithm 1 shows the pseudocode of SVD-LLM. Before compression, SVD-LLM randomly collects a small amount of sentences as calibration data C, then runs the truncation-aware data whitening process as shown in Algorithm 2 to obtain the set of whitening matrix SetS for the weight to compress. After that, it runs the SVD and truncation with SetSon each weight matrix in the LLM. Instead of directly finishing the whole compression, it stores the decomposed matrices and further utilizes these matrices to run the parameter update with sequential low-rank approximation as shown in Algorithm 3.

#### Algorithm 1 Pseudocode of SVD-LLM

1: **Input:** M: Original LLM

2**: Output:**M′′: Compressed LLM by SVD-LLM

3: **procedure** SVD-LLM(M)

4:

Randomly collect several sentences as the calibration data C

5:

SetS←TRUNCATION-AWARE DATA WHITENING(M,C)

6:

$$\text {Set}_{W}\leftarrow $$

Obtain the set of weights in M to compress

7:

**for** W**in** Setw **do**

8:

$$S\leftarrow \text {Set}_{S}()$$

Extract the whitening matrix of current weight W

9:

$$U,Σ,V\leftarrow \text {SVD}(WS)$$

Apply singular value decomposition on W

10:

$$Σ_{1}\leftarrow \text {Trunc.}(Σ)$$

$$\begin{array}{c}W_{u}^{\prime }\leftarrow U\left(Σ_{1}\right)^{1/2},W_{v}^{\prime }\leftarrow \left(Σ_{1}\right)^{1/2}V^{T}S^{-1}\\ M^{\prime }(W)\leftarrow W_{u}^{\prime },W_{v}^{\prime }\\ \text {dfor}\end{array}$$

Truncate the smallest singular values in Σ

11:

Obtain two low-rank matrices

12:

ΔReplace W with Wu′andWv′in L

13:

**en**

14:

M′′← PARAMETER UPDATE WITH SEQUENTIAL LOW-RANK APPROXIMATION (M′)

15:

returnM′′

16**: end procedure**

#### Algorithm 2 Pseudocode of Truncation-Aware Data Whitening

1: **Input**: M: Original LLM

2: **Input**: C: Calibration Data

3: **Outpuit**:SetS: Set of whitening matrices for the weight to compress in M

4: **procedure** TRUNCATION-AWARE DATA WHITENING(M,C)

5:

$$\text {St}_{S}\leftarrow \emptyset$$

Initialize the set of whitening matrices

6:

$$\text {Set}_{W}\leftarrow M$$

Obtain the set of weights in M to compress

7:

**for** W**in** Setw **do**

8:

$$X\leftarrow M(W,C)$$

Obtain the input activation of the weight matrix W

9:

S←CholeskyDecomposition(XXT) Apply cholesky decomposition on XXT

10:

$$\text {Set}_{S}\leftarrow S\cup \text {Set}_{S}$$

Store the whitening weight matrix in the set

11:

**end for**

12:

**return** Sets

13**: end procedure**

## Algorithm 3 Pseudocode of PParameter Update with Sequential Low-rank Approximation

1: **Input:** M′: Compressed LLM by Truncation-aware Data Whitening

2: **Output:**′′: Compressed LLM with Parameter Update with Sequential Low-rank Approxima-tion

3: **procedure** PARAMETER UPDATE WITH SEQUENTIAL LOW-RANK APPROXIMATION(M')

4:

$$M_{u}^{\prime }\leftarrow \text {oA}_{\mathrm {u}}\left(M^{\prime }\right)$$

$$M^{\prime \prime }\leftarrow \text {LoRA}_{\mathrm {v}}\left(M_{u}^{\prime }\right)$$

Fix all WWv′,fine-tune allWu′

5:

Fix allWu′,fine-tune all W

6:

**return**M′′

7**: end procedure**

<!-- 14 -->

<!-- Published as a conference paper at ICLR 2025 -->

### A.2 COMPRESSION Loss OF ASVD

ASVD introduces a diagonal scaling matrix S0 that modifies the weight matrix to reflect the varying significance of different input channels. The linear layer is formnulated asY=(WS0)S0-1X.The compression is made by keeping the largest m singular value of W S0

$$WS_{0}\approx \sum _{i=1}^{m}σ_{i}^{\prime }u_{i}^{\prime }v_{i}^{\prime T}$$

The resulting activation is expressed as:

$$Y\approx \sum _{i=1}^{m}σ_{i}^{\prime }u_{i}^{\prime }v_{i}^{\prime T}S_{0}^{-1}X$$

The compression errorL=\|(WS0-∑i=1mσi′ui′vi′T)S0-1X\|Fis demonstrated below:

$$L^{2}=\left\|\left(WS_{0}-\sum _{i=1}^{m}σ_{i}^{\prime }u_{i}^{\prime }v_{i}^{\prime T}\right)S_{0}^{-1}X\right\|_{F}^{2}$$

$$=\left\|\sum _{i=m+1}^{r}σ_{i}^{\prime }u_{i}^{\prime }v_{i}^{\prime T}S_{0}^{-1}X\right\|_{F}^{2}$$

$$=\sum _{j=m+1}^{r}\sum _{i=m+1}^{r}σ_{i}^{\prime }σ_{j}^{\prime }\text {Trace}\left(u_{i}^{\prime }v_{i}^{\prime T}XX^{T}v_{j}^{\prime }u_{j}^{\prime T}\right)$$

$$=\sum _{j=m+1}^{r}\sum _{i=m+1}^{r}σ_{i}^{\prime }σ_{j}^{\prime }\text {Trace}\left(u_{j}^{\prime T}u_{i}^{\prime }v_{i}^{\prime T}S_{0}^{-1}XX^{T}\left(S_{0}^{-1}\right)^{T}v_{j}^{\prime }\right)$$

$$=\sum _{i=m+1}^{r}σ_{i}^{\prime 2}\text {Trace}\left(v_{i}^{\prime T}S_{0}^{-1}XX^{T}\left(S_{0}^{-1}\right)^{T}v_{i}^{\prime }\right)$$

$$=\sum _{i=m+1}^{r}σ_{i}^{2}\left\|v_{i}^{\prime T}S_{0}^{-1}X\right\|_{F}^{2}$$

which is still a complex function that involves the activation X, the diagonal matrixS0,the singular vectorvi′and the singular valueσi′. As a result, compression error is not directly related to the singular value, and the conventional SVD compression by truncating the smallest singular values may lead to suboptimal compression error.

### A.3 COMPRESSION LOSs OF SVD-LLM

In SVD-LLM,we also formulate the linear layer asY=(WS)S-1X,whereS-1XXT(S-1)T=I. The compression is made by keeping the largest m out of total r singular values of WS. The compression loss L is demonstrated as:

<!-- 15 -->

<!-- Published as a conference paper at ICLR 2025 -->

$$L^{2}=\left\|WX-W^{\prime }X\right\|_{F}^{2}=\left\|WSS^{-1}X-\mathbf {SVD}(WS)S^{-1}X\right\|_{F}^{2}$$

$$=\left\|(WS-\mathbf {SVD}(WS))S^{-1}X\right\|_{F}^{2}$$

$$=\left\|\left(WS-\sum _{i=1}^{m}σ_{i}u_{i}v_{i}^{T}\right)S^{-1}X\right\|_{F}^{2}$$

$$=\left\|\sum _{i=m+1}^{r}σ_{i}u_{i}v_{i}^{T}S^{-1}X\right\|_{F}^{2}$$

$$=\sum _{j=m+1}^{r}\sum _{i=m+1}^{r}σ_{i}σ_{j}\text {Trace}\left(u_{i}v_{i}^{T}S^{-1}XX^{T}\left(S^{-1}\right)^{T}v_{j}u_{j}^{T}\right)$$

$$=\sum _{j=m+1}^{r}\sum _{i=m+1}^{r}σ_{i}σ_{j}\text {Trace}\left(u_{i}v_{i}^{T}\left(S^{-1}XX^{T}\left(S^{-1}\right)^{T}\right)v_{j}u_{j}^{T}\right)$$

$$=\sum _{j=m+1}^{r}\sum _{i=m+1}^{r}σ_{i}σ_{j}\text {Trace}\left(u_{i}v_{i}^{T}v_{j}u_{j}^{T}\right)$$

$$\because v_{i}^{T}v_{i}=u_{i}^{T}u_{i}=1;v_{i}^{T}v_{j}=u_{i}^{T}u_{j}=0,\text {Trace}\left(v_{i}v_{i}^{T}\right)=\text {Trace}\left(u_{i}u_{i}^{T}\right)=1,\forall i\neq j$$

$$\therefore L^{2}=\sum _{j=m+1}^{r}\sum _{i=m+1}^{r}σ_{i}σ_{j}\mathbf {Trace}\left(u_{i}v_{i}^{T}v_{j}u_{j}^{T}\right)=\sum _{i=m+1}^{r}σ_{i}^{2}\mathbf {Trace}\left(u_{i}v_{i}^{T}v_{i}u_{i}^{T}\right)=\sum _{i=m+1}^{r}σ_{i}^{2}$$

Therefore, the squared lossL2isequal to the sum of the squared singular values. Therefore, truncating the smallest singular values achieves the lowest compression loss.

### A.4 SPECTRUM ANALYSIS OF SINGULAR VALUES DECOMPOSED BY SVD-LLM

In general, SVD is useful for compression when the matrixto be compressed shows a sharp decay of the singular values. Since SVD-LLM decomposes the multiplication of the weight matrix W and its corresponding whitening matrix S instead of the original weight matrix M, which is different from the weight decomposition in the previous work (Yuan et al., 2023; Hsu et al., 2022), to study whether SVD comnpression is also applicable in SVD-LLM, we select the Query (WQ))and Key((WK) weight matrices and show the spectrum of singular values of their multiplication with corresponding whitening matricesSQandSK. As shown in Figure 6, most of the single values are less than or around 100 with only a few extremely large values, indicating that SVD is applicable in SVD-LLM.

### A.5 COMPARISON WITH DRONE

Previous work DRONE (Chen et al., 2021) also proposes their data-aware method for SVD compres-sion. They even provide a theoretical analysis to prove the optimal solution that their method achieves. Specifically, DRONE represents the low-rank compressed weight matrixW′byWM.It performs SVD on both weight matrixW=UwSwVwTand the transpose of input activationXT=UxSxVxT and then split these decomposed matrices as follows:

$$U_{W}=\left[\begin{array}{cc}U_{W,r}&\bar {U}_{W,r}\end{array}\right],S_{W}=\left[\begin{array}{cc}S_{W,r}&0\\ 0&0\end{array}\right],V_{W}=\left[\begin{array}{cc}V_{W,r}&\bar {V}_{W,r}\end{array}\right]\quad U_{X}=\left[\begin{array}{cc}U_{X,t}&\bar {U}_{X,t}\end{array}\right],S_{X}=\left[\begin{array}{cc}S_{X,t}&0\\ 0&0\end{array}\right],V_{X}=\left[\begin{array}{cc}V_{X,t}&\bar {V}_{X,t}\end{array}\right]$$

where r and k are the rank of the Wand X. UW,r,VW,r,UX,t,VX,tdenote corresponding row spaces and column spaces and $\bar {U}_{,r},\bar {V}_{,r},\bar {U}_{X,t},\bar {V}_{X,t}$ are null spaces. Through theoretical deduction,

<!-- 16 -->

<!-- Published as a conference paper at ICLR 2025 -->

<!-- 104 104 o o O O O 。 o 。θ 。8 09 09 09 OT O 8 09 O 09 0- O 09 。目 08- 08- 。目 ο目 9 ο우 ® 0目 日 自O 102 102 SenlIen eIn6uIS4 unJtadS ° 100 §0O 00。 目8o。 e0Co0。 8 Tc 8O。 0e 10-2 10-2 冒 00 自0。 。 8。 000 月0目o 0000 000 O 晶800 。 O 0。 。 。 。 O o O 10-4 O o O o O o 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 Layer Index -->
![](https://web-api.textin.com/ocr_image/external/eee89852a02a2622.jpg)

(3)WQxSQ

<!-- 。 O 。 。 。 O 。 O 。 O O o 104 8 自 自 目 自 。目 SenlIen eln6uIS4 unJtedS 102 10° 100 电o80 8 n000 OO 8 0目 10-2 10-2 o O O P00 O o O o 10-4 自导aID000 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 Layer Index -->
![](https://web-api.textin.com/ocr_image/external/1b6d337866976d9d.jpg)

(0) WK×SK

Figure 6: The singular value spectrum of the decomposed matrices across layers inLLaMA-7B.

DRONE converts the minimization of compression loOsS\|WX-W′X\|F=\|WX-WMX\|F10the minimization of\|SW,rVW,rTVX,tSX,t-SW,rVW,rTMVX,tSX,t\|F,whose optimal valueLmin15the rank-k truncated SVD ofZ=SW,rVW,rTVX,tSX,tby the fundamental property of SVD decom-position. To achieve the optimal value, DRONE formulates a solutionM=VW,rSW,r-1ZkSX,t-1VX,tT, whereZkis the rank-k SVD truncation of Z.

In short, compared with DRONE, SVD-LLM is also optimal with the same theoretical compression loss as DRONE. Moreover, SVD-LLM has **three** key advantages. Below is our detailed explanation.

**SVD-LLM is also optimal with the same theoretical compression loss as DRONE.** The theoretical minimum compression lossLminis the F-norm loss of rank-k SVD truncation of WX,which has also been achieved by DRONE in their paper. Unlike DRONE, SVD-LLM constructs the whitening matrix S so thatS-1Xis orthonormal. Therefore, we have\|AS-1X\|F=\|A\|F. Suppose that we decompose S with SVD toUSV,we can haveS=SxU=UxU=UxV=QVx,where Q is an orthogonal matrix. The matrix WS to which SVD-LLM applies SVD could be represented

<!-- 17 -->

<!-- Published as a conference paper at ICLR 2025 -->

Table 9: Compression loss of the randomly generated weight and activation matrices with different shapes under 50% compression ratio using SVD-LLM, Drone, and the theoretical minimum.

<table border="1" ><tr>
<td>Loss</td>
<td>[128×128]×[128×128]|[2048×2048]×[2048×2048]|[4096×4096]×[4096×4096]</td>
<td></td>
<td></td>
</tr><tr>
<td>MINIMUM</td>
<td>276.1130</td>
<td>17784.2637</td>
<td>50321.9141</td>
</tr><tr>
<td>DRONE</td>
<td>276.1130</td>
<td>17785.6992</td>
<td>50337.2148</td>
</tr><tr>
<td>SVD-LLM</td>
<td>276.1130</td>
<td>17784.2676</td>
<td>50321.9727</td>
</tr></table>

Table 10: Compression Time of the randomly generated weight and activation matrices with different shapes using SVD-LLM and Drone. The compression time is measured for 5 times' compression.

<table border="1" ><tr>
<td>TIME</td>
<td>[128x128]x[128x128]|[2048x2048]x[2048x2048]|[4096x4096]x[4096x4096]</td>
<td></td>
<td></td>
</tr><tr>
<td>DRONE</td>
<td>0.07 seconds</td>
<td>5.81 seconds</td>
<td>30.35 seconds</td>
</tr><tr>
<td>SVD-LLM</td>
<td>0.02 seconds</td>
<td>1.98 seconds</td>
<td>10.37 seconds</td>
</tr></table>

byUwSwVwTUsSsVsT. Suppose that we use **Trunc.**(C) to represent the rank-k truncation of the matrix C during SVD compression, the compression loss L is derived as follows:

$$L=\left\|WX-W^{\prime }X\right\|_{F}=\left\|\left(WSS^{-1}X-\mathbf {SVD}(WS)S^{-1}X\right)\right\|_{F}=\left\|(WS-SVD(WS))S^{-1}X\right\|_{F}\quad =\left\|\text {Trunc.}(WS)S^{-1}X\right\|_{F}=\|\text {Trunc.}(WS)\|_{F}=\left\|\text {Trunc.}\left(U_{w}S_{w}V_{w}^{T}U_{s}S_{s}V_{s}^{T}\right)\right\|_{F}=\left\|\text {Trunc.}\left(WXQ^{T}\right)\right\|_{F}=L_{\min }$$

Therefore, SVD-LLM shares the same theoretical compression loss as DRONE.

**Advantage #1: DRONE incurs out-of-memory when compressing LLMs due to its requirement** of **storing the full large-scale activations, whereas SVD-LLM is feasible.** To achieve data-awareness during compression, DRONE caches all input activationsXand spans them to calculate the corre-sponding singular vectors and singular values. In the DRONE paper, the authors apply DRONE to small LMs such as BERT. However, the activations generated by LLMs are often extremely large and are much larger than BERT. For example, to compress LLaMA-7B with 5,000 calibration data by DRONE, the total memory to cache the activation for a single weight matrix at a time is 5000 (data umb)×256(q\_)×118(dim)×32(fp32)÷124÷124÷124=419GBwhichi more than 5 times larger than the memory provided by the NVIDIA A100 GPU, which has 80GB memory. Therefore, applying DRONE for LLM compression is infeasbile.

In contrast, SVD-LLM incrementally updates itsXXTmatrix by adding theTof each new input x. As such, SVD-LLM eliminates the need to store the full activations, which requires only the storage of the matrixXXT, which is considerably smaller than the full input activation. To compress LLaMA-7B with 5,000 calibration data, SVD-LLM requires only 1118×11008×32÷1024÷1024÷1024=3.6GB. Compared to DRONE,SVD-LLM achieves 116.38 times less memory reduction than DRONE. In our paper, we use WikiText-2 as a dataset. If we use DRONE,the total memory to cache the activation1 of a single weight matrix is larger than 24,600GB. In contrast, SVD-LLM1 still requires 3.6GB of memory, which is more than 6,000 times less than DRONE. Due to this advantage, SVD-LLM is much more practical to compress LLMs of size 7B or larger compared to DRONE.

**Advantage #2: SVD-LLM incurs much shorter compression time compared to DRONE.** DRONE involves more complex matrix operations, leading to longer compression time compared to SVD-LLM. To illustrate this, we measured the time required by DRONE and SVD-LLM to compress randomly generated weight and activation matrices of varying shapes under 50% compression ratio. The results show that SVD-LLM is approximately three times faster than DRONE.

**Advantage #3: SVD-LLM has better numerical stability, wvhich leads to superior empirical** **compression** **loss.** While SVD-LLM shares the same theoretical compression loss as DRONE, DRONE's higher complexity-stemming from additional SVD operations and inverse calculations on large-scale matrices-makes it less numerically stable compared to SVD-LLM. This often results in higher empirical compression losses. To illustrate this, we compare SVD-LLM and DRONE in terms of empirical compression losses for randomly generated matrices of various shapes. We also include the theoretical minimum value, represented by the rank-k Frobenius norm loss of WX. The results are summarized in the following table. As shown, we observe that SVD-LLM achieves lower empirical compression losses than DRONE, underscoring its superior numerical stability.

<!-- 18 -->

<!-- Published as a conference paper at ICLR 2025 -->

Table 11: Perplexity (↓) of SVD-LLM and FLAP on WikiText-2 to compress LLaMA-7B under different compression ratios. The better performance is marked in bold. The relative performance gain of SVD-LLM compared to FLAP is marked in green inside bracket.

<table border="1" ><tr>
<td>RATIO (MEM.)</td>
<td>20%(10.2GB)</td>
<td>40%(7.76GB)</td>
<td>60%(5.35GB)</td>
<td>80%(2.58GB)</td>
</tr><tr>
<td>FLAP</td>
<td>7.99</td>
<td>14.43</td>
<td>106.87</td>
<td>15023</td>
</tr><tr>
<td>SVD-LLM</td>
<td>7.73(↓3%)</td>
<td>9.27(↓36%)</td>
<td>15.00(↓86%)</td>
<td>31.79(↓99%)</td>
</tr></table>

Table 12: Comparison of LLaMA-3B (compressed from LLaMA-7B by SVD-LLM) and original StableLM-3B (Tow et al.) trained from scratch. Both the throughput and the peak memory footprint during the inference are measured with batch size=32,,sequence length=128 on single A100 GPU.

<table border="1" ><tr>
<td>MODEL</td>
<td>Throughput</td>
<td>Peak Mem.</td>
<td>Openb.</td>
<td>Arce</td>
<td>WinoG.</td>
<td>HellaS.</td>
<td>PIQA</td>
<td>MathQA</td>
<td>Average↑</td>
<td>TruthfulQA↑</td>
<td>GSM8K↑</td>
</tr><tr>
<td>StableLM-3B</td>
<td>8463 Tokens/sec</td>
<td>9.41 GB</td>
<td>0.19</td>
<td>0.51</td>
<td>0.55</td>
<td>0.47</td>
<td>0.69</td>
<td>0.21</td>
<td>0.44</td>
<td>0.22</td>
<td>0.02</td>
</tr><tr>
<td>LLaMA-3B</td>
<td>9254 Tokens/sec</td>
<td>7.43 GB</td>
<td>0.27</td>
<td>0.54</td>
<td>0.60</td>
<td>0.49</td>
<td>0.68</td>
<td>0.19</td>
<td>0.46(↑5%)</td>
<td>0.23(+0.01)</td>
<td>0.04(+0.02)</td>
</tr></table>

### A.6 COMPARISON WITH FLAP

Recent work FLAP (An et al., 2024) is also a post-training structured-pruning method. Below we compare the perplexity of SVD-LLM and FLAP on WikiText-2 under different compression ratios when compressing LLaMA-7B. As shown in Table 11, SVD-LLM consistently outperforms FLAP, especially under high compression ratios.

### A.7 COMPARISON WITH SMALLER LLMS PRE-TRAINED FROM SCRATCH

To compare the performance between SVD-LLM and scratch training, following the previous ex-perimental design (Ma et al., 2023), we compress LLaMA-7B to the size of the 3B parameter with SVD-LLM and select StableLM-3B (Tow et al.) as the baseline for comparison. As shown in Table 12. LLaMA-3B compressed from LLaMA-7B by SVD-LLM achieves better accuracy in all datasets, indicating that SVD-LLM could even achieve better accuracy than some scratch training methods. Fur-thermore, SVD-LLM ensures higher throughput and lower memory consumption than StableLM-3B as shown in the table, which also meets our efficiency analysis and discussion in Section 4.2.

Table 13: Compression time of SVD-LLM and ASVD on LLaMA-7B under 20% compression ratio. The relative speedup is marked in green color inside bracket.

<table border="1" ><tr>
<td colspan="3">SVD-LLM</td>
<td colspan="3">ASVD</td>
</tr><tr>
<td>Truncation-Aware Data Whitening</td>
<td>Parameter Update with Sequential Low-rank Approximation</td>
<td>Total</td>
<td>Normalize</td>
<td>Search</td>
<td>Total</td>
</tr><tr>
<td>10min</td>
<td>3.5h</td>
<td>3.5h(↓36%)</td>
<td>5min</td>
<td>5.5h</td>
<td>5.5h</td>
</tr></table>

### A.8 COMPRESSION SPEED EVALUATION

In addition to compression performance, we also evaluate the compression speed of SVD-LLM and the baselines. Specifically, we measured the GPU hours used for SVD-LLM and ASVD when compressing LLaMA-7B under 20% compression ratio on an A100 GPU. Thhe results are shown in Table 13. As shown, ASVD takes about 5.5 hours, while SVD-LLM completes the compression process in 3.5 hours, which is 36% times faster. When breaking down the time, most of the time consumed by ASVD is dedicated to searching for the specific compression ratio for each weight matrix based on its calculated importance score. In contrast, SVD-LLM maintains a consistent compression ratio in all weight matrices and thus eliminates the time-consuming search process.

<!-- 19 -->

<!-- Published as a conference paper at ICLR 2025 -->

### A.9 CONTENTS GENERATED FROM LLMS COMPRESSED BY SVD-LLM AND ASVD

Some examples of sentences generated by LLaMA-7B compressed with SVD-LLM and ASVD are shown in Table 14. The sentences generated by the model compressed by SVD-LLM exhibit better fluency, relevance, and informativeness compared to those compressed by ASVD.More importantly,when the compression ratio is increased to 40%, the previous state-of-the-art method ASVD completely loses its generation ability. In contrast, even when the compression ratio is up to 80%, SVD-LLM is still capable of generating complete sentences.

Table 14: An example of contents generated by the fine-tuned LLaMA-7B with LoRA after being compressed by SVD-LLM and ASVD under different compression ratios. The input is marked in bold and thenormal texts are the generated sentences.

<table border="1" ><tr>
<td>RATIO</td>
<td>SVD-LLM</td>
<td>ASVD</td>
</tr><tr>
<td>Original</td>
<td colspan="2">What is the responsibility of the AI assistant? The responsibility of an AI assistant is to complete the job without any<br>hesitation. Also, the AI assistant has no problem with taking a decision and executing it. An AI assistant is responsible to complete the task and it needs to learn from his past experience to take the best possible decision without any hesitation...</td>
</tr><tr>
<td>20%</td>
<td>What is the responsibility of the AI assistant? The AI assistant's role and responsibilities are often ambiguous and<br>can vary depending on the context and the type of assistant<br>being used. Some AI assistants may have preset responses to common queries, but their behavior can also depend on the user's interaction and expectations...</td>
<td>What is the responsibility of the AI assistant? 1. To act in the same way as the programmer intended. 2. To make<br>a value assessment of the user's intentions and offer an<br>appropriate response. 3. To help the user carry out specific tasks or follow a strict script. 4. To make the system as independent of the user as possible...</td>
</tr><tr>
<td>40%</td>
<td>What is the responsibility of the AI assistant? 1. The AI assistant is responsible for performing tasks that can only be done by AI technology such as scheduling appointments, setting reminders and making suggestions, while also pro-viding quick and accurate responses. 2. The AI assistant<br>is also responsible for learning and updating its knowledge<br>base through continuous improvement, thus making it more<br>efficient and effective. 3. The AI assistant is responsible<br>for handling and managing user-generated and customer-oriented input...</td>
<td>What is the responsibility of the AI assistant? 1.Define the definition of the job specification. 2. Define the role of the AI assistant for the job. 3. Define the skills necessary in order to carryout the tasks assigned to the job to do the work of the same capacity in the market. 4. Determine the evaluation of the job competence of the AI assistant to ensure their professional integrity. 5. What type of AI assistant jobs are available? 6. What industries require technology assistance work to complete the tasks that it own personnel cannot perform..</td>
</tr><tr>
<td>60%</td>
<td>What is the responsibility of the AI assistant? In a nut-shell, the responsibility of the AI assistant could vary depend-ing on the task, but generally, the focus would be on auto-matic tasks, without the need for human intervention. Some<br>common tasks could include setting reminders, scheduling<br>appointments, and making routine household or productivity tasks. The AI assistant also serves as a backup or a relief system,taking on responsibilities when the user is not avail-able...</td>
<td>What is the responsibility of the AI assistant? 2.3. ?? Brush for a discussion I wonder is it worth doing is important.2- It isn't useful just reducing labor costs;ithelps employees feel a sense of connected to your atten-tion which leads to better workplace values among staffers and leads to long relationships among org...</td>
</tr><tr>
<td>80%</td>
<td>What is the responsibility of the AI assistant? Our Design is based on our understanding of the world, and we are actively learning, adapting and adapting, so we're always<br>evolving new ideas, which we see to be most unique and relevant in our community...</td>
<td>What is the responsibility of the AI assistant? ygua Aleltemperaturen/2, (64mbz/.3/.1/, 7.kbld.org.0/2/Inthese puthebout les bnvols n merginels...</td>
</tr></table>

### A.10 MORE ABLATION STUDIES

**SVD-LLM+ Normal LoRA Fine-tuning v.s. SVD-LLM + Sequential LoRA fine-tuning.** To illustrate the superiority of the designed parameter update with the sequential low-rank approximation in SVD-LLM,which is a kind of sequential LoRA fine-tuning strategy over the normal LoRA fine-tuning strategy, we compare the compression performance of SVD-LLM by applying either of these two fine-tuning strategies. Let's denote SVD-LLM **(SFT)** as SVD-LLM by applying sequential LoRA fine-tuning and SVD-LLM (NFT) as SVD-LLM by applying normal LoRA fine-tuning. As shown in Table 15, SVD-LLM (SFT) consistently outperforms SVD-LLM (NFT), which also reaffirms our analysis in Section 3.2 that optimizing both low-rank matricesWuWvat the same time is not stable and may lead to poor fine-tuning performance.

ASVD+SequentialLoRA**Fine-tuning v.s.**SVD-LLM+SequentialLoRAFine-tuning. Although the designed sequential LoRA fine-tuning strategy could also be applied in other SVD-based LLM compression methods, other methods' performance is still poorer than SVD-LLM even when inte-grated with this strategy for enhancement. To illustrate this, we compare the performance of the

<!-- 20 -->

<!-- Published as a conference paper at ICLR 2025 -->

Table 15: Perplexity (↓) of SVD-LLM with original LoRA fine-tuning (denoted as SVD-LLM(SFT)), ASVD with sequential LoRA fine-tuning (denoted as ASVD (SFT)), and SVD-LLM with sequential LoRA fine-tuning (denoted as SVD-LLM (SFT)) on WikiText-2 to compress LLaMA-7B under different compression ratios.

<table border="1" ><tr>
<td>RATIO (MEM.)</td>
<td>20%(10.2GB)</td>
<td>40%(7.76GB)</td>
<td>60%(5.35GB)</td>
<td>80%(2.58GB)</td>
</tr><tr>
<td>SVD-LLM(NFT)</td>
<td>7.87</td>
<td>11.98</td>
<td>16.30</td>
<td>80.23</td>
</tr><tr>
<td>ASVD (SFT)</td>
<td>8.37</td>
<td>14.86</td>
<td>44.81</td>
<td>271</td>
</tr><tr>
<td>SVD-LLM(SFT)</td>
<td>7.73</td>
<td>9.27</td>
<td>15.00</td>
<td>31.79</td>
</tr></table>

previous state-of-the-art method ASVD when applied with the sequential LoRA finetuning with SVD-LLM.Let's denote SVD-LLM(SFT) as SVD-LLM by applying sequential LoRA fine-tuning and ASVD (SFT) as ASVD by applying sequential LoRA fine-tuning. As showvn in Table 15, SVD-LLM (SFT) consistently outperforms ASVD (SFT) under various compression ratios.

### A.11 LIMITATIONS

There are three limitations of SVD-LLM,which are left for future work.

**(1) The compression accuracy still needs to be improved under the high compression ratio.** Although SVD-LLM has achieved the state-of-the-art performance compared to previous works such as FWSVD and ASVD, its compression accuracy still suffers from degradation, especially under the high compression ratio. To enhance the practicability of SVD-LLM in real-world scenario, its accuracy should be at least comparable to that of the quantization method, including both low-bit and high-bit quantization, rather than being combined with the quantization methods for usage.

**(2) The latency of SVD-LLM needs to be optimized while being used to compress the KV cache.** As discussed in Section 4.2, SVD-LLM can also be applied to compress the KV cache for memory saving during inference. However, this benefit does not come free. In fact, due to the additional calculation caused by recovering the original key and value states, the inference speed will be impacted by compressing the KV cache with SVD-LLM, which should be optimized in the future.

**(3) SVD-LLM should be better guided for high-quality generation.** Although SVD-LLM achieves low perplexity, as demonstrated in Table 1, it is still possible to generate low-quality content, such1 as repeated words even under low compression ratios. This phenomenon of compressed LLM has also been mentioned in previous work (Ma et al., 2023) and should be eliminated in the future by guiding the compressed LLM for outputting high-quality generation.

<!-- 21 -->

