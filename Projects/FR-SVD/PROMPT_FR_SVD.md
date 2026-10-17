# FR-SVD 完整压缩代码实现任务

## 任务目标
基于 FR-SVD.tex 论文，在 `/data/zhangs/LLMs/SVD-LLM/` 目录下创建 `FR-SVD.py`，实现完整的逐层两阶段压缩框架。代码必须能在现有框架中直接运行。

---

## 第一步：阅读参考文件（按顺序）

### 1.1 论文（仅需读 §3-4 和 Appendix A/B）
- `FR-SVD.tex`：方法的精确数学定义

### 1.2 现有代码（理解接口规范）
按以下顺序阅读，每个文件只需关注标注的部分：

| 文件 | 关注内容 |
|------|----------|
| `utils/model_utils.py` | `get_model_from_huggingface`、`get_model_from_local`、`find_layers` 的签名和返回值 |
| `utils/data_utils.py` | `get_calib_train_data` 的返回格式（list of dict，含 input_ids 和 attention_mask） |
| `evaluater.py` | `ppl_eval` 的签名 |
| `component/svd_llama.py` | `SVD_LlamaAttention` 和 `SVD_LlamaMLP` 的构造函数参数、低秩因子命名（`q_u_proj`/`q_v_proj` 等）、forward 的调用方式 |
| `SAES-SVD_v2.1.4.py` 第 213-703 行 | **重点参考**：`saes_svd_compress_layerwise` 函数的完整流程——teacher/student 双路径激活收集、逐层 hook 统计 H 和 Δ、SVD 替身模块的写入模式、逐层激活传播更新 |
| `SAES-SVD_v2.1.4.py` 第 83-210 行 | `aces_beta_select_unified` 的 FS-FOA 求解逻辑（FR-SVD 需复用，仅替换输入矩阵为 G^F(β)） |

**确认点**：读完后向我确认你理解了以下接口：
- 激活收集的 Catcher 模式
- H/Δ 的 hook 统计方式
- SVD 替身模块的写入和层替换模式
- 逐层闭环传播（压缩后重新前向更新下游激活）

---

## 第二步：实现 FR-SVD

### 架构概述
```
逐层循环 for ℓ = 1..L:
  ├── 数据准备：收集 X_ℓ（student）和 X_ℓ^f（teacher），统计 H_ℓ 和 Δ_ℓ
  ├── Fisher 因子估计：需要梯度，独立于 @torch.no_grad 的统计收集
  ├── Stage 1：Fisher-Guided CEALC（rank r₁ 截断）
  ├── Stage 2：Eigenspace Residual Compensation（rank r₂ 补偿）
  ├── 组合 W' = W^(1) + W^comp，写入 SVD 替身模块
  └── 更新下游激活
```

### 2.1 Fisher Kronecker 因子估计（论文 §3.3，式 P7-P9）

这是 FR-SVD 相比 SAES-SVD 的**新增核心模块**，项目内无现成实现。

```python
def estimate_kronecker_fisher(model, layer, layer_idx, calib_loader, dev,
                               lambda_phi=1e-4, lambda_psi=1e-4):
    """
    对当前层的每个线性子层估计 Kronecker-factored Fisher 因子。

    数学定义（论文 §3.3）：
      I_F ≈ Φ ⊗ Ψ
      其中 Φ ∈ R^{n×n}（输入维度），Ψ ∈ R^{m×m}（输出维度）

    GFWSVD Rank-1 估计法：
      对每个 mini-batch i：
        梯度 G_i = ∇_{W} L(batch_i)  # shape: (m, n)
        输入激活 a_i ∈ R^n（线性层的输入）
        输出梯度 b_i ∈ R^m（loss 对线性层输出的梯度）
        由于 G_i = b_i ⊗ a_i^T（外积），所以：
        Φ += a_i a_i^T    # 输入侧
        Ψ += b_i b_i^T    # 输出侧
      归一化后加阻尼：
        Φ = Φ/N + λ_Φ I
        Ψ = Ψ/N + λ_Ψ I

    注意：
    - 需要开启梯度计算（不能在 @torch.no_grad 下）
    - 对每个 batch 做一次前向+反向，收集各线性子层的输入和输出梯度
    - 复杂度 O(m²n + mn²) per layer

    Returns:
        dict: {sublayer_name: (L_Phi, L_Psi)}  # Cholesky 因子
    """
```

**实现要点**：
- 用 `register_forward_hook` 捕获线性层输入 a_i
- 用 `register_full_backward_hook` 捕获输出梯度 b_i
- 每个 batch 前向+反向后累加 Φ 和 Ψ
- Cholesky 分解时处理非正定情况（参考 SVDLLM.py 的处理方式）

### 2.2 Stage 1：Fisher-Guided CEALC（论文 §4.2）

```
输入：W, H, Δ, L_Φ, L_Ψ, rank r₁
步骤：
  1. 构造 Fisher 空间目标矩阵（式1）：
     G^F(β) = L_Ψ^T · W(H + βΔ)H^{-1/2} · L_Φ
     分解为 G^F(β) = S^F + β·D^F，其中：
       S^F = L_Ψ^T · W · H^{1/2} · L_Φ
       D^F = L_Ψ^T · W · Δ · H^{-1/2} · L_Φ

  2. FS-FOA 优化 β（式2）：
     ★ 直接复用 aces_beta_select_unified，但将输入的 S 和 D 替换为 S^F 和 D^F
     注意：原函数接收 (W, H, Delta, L, rank)，
     你需要适配为接收预计算的 (S^F, D^F, rank)，
     或者创建一个新的 wrapper 函数

  3. 构造 G^F(β*) 并做 rank-r₁ 截断 SVD

  4. 映回权重空间（式4）：
     W^(1) = L_Ψ^{-T} Û Σ̂ V̂^T L_Φ^{-1} H^{-1/2}
     低秩因子：
       B^(1) = L_Ψ^{-T} Û Σ̂^{1/2}  ∈ R^{m×r₁}
       A^(1) = Σ̂^{1/2} V̂^T L_Φ^{-1} H^{-1/2}  ∈ R^{r₁×n}
```

### 2.3 Stage 2：Eigenspace Residual Compensation（论文 §4.3）

```
输入：W, W^(1), X, X^f, H, rank r₂
步骤：
  1. 全局输出对齐误差（式5）：
     E = W X^f - W^(1) X        # 注意：W^(1)X 需要用 B^(1)A^(1)X 计算

  2. 岭回归闭式解（式7）：
     ΔW^LS = E X^T H^{-1}

  3. H 的特征分解：
     H = Q Λ Q^T
     截断接近零的特征值（< eps），避免 Λ^{-1/2} 爆炸

  4. 白化并截断 SVD（式8）：
     ΔW_white = ΔW^LS · Q · Λ^{1/2}
     [ΔW_white]_{r₂} = Ũ Σ̃ Ṽ^T

  5. 映回原空间：
     W^comp = Ũ Σ̃ Ṽ^T Λ^{-1/2} Q^T
     低秩因子：
       B^(2) = Ũ Σ̃^{1/2}  ∈ R^{m×r₂}
       A^(2) = Σ̃^{1/2} Ṽ^T Λ^{-1/2} Q^T  ∈ R^{r₂×n}
```

### 2.4 组合与部署

FR-SVD 的每个子层产出 **4 个矩阵**：B^(1), A^(1), B^(2), A^(2)。

**关键设计决策**：现有的 `SVD_LlamaAttention` / `SVD_LlamaMLP` 每个子层只有 2 个 slot（`*_u_proj` 和 `*_v_proj`）。FR-SVD 需要 4 个 slot（两个分支）。

**推荐方案**：将两个分支合并为等效的单个低秩因子后写入现有 SVD 替身模块：
```python
# 合并方案：将 B^(1)A^(1) + B^(2)A^(2) 合并为 [B^(1)|B^(2)] · [A^(1); A^(2)]
# 其中 [B^(1)|B^(2)] ∈ R^{m×r}, [A^(1); A^(2)] ∈ R^{r×n}, r = r₁ + r₂
B_merged = torch.cat([B1, B2], dim=1)  # (m, r)
A_merged = torch.cat([A1, A2], dim=0)  # (r, n)
# 然后按照 SAES-SVD 的模式写入 svd_u = B_merged, svd_v = A_merged
```

这样**完全复用**现有的 SVD 替身模块，无需创建新的 component 类。注意：
- 现有 SVD 替身模块的 `low_rank` 是用 `ratio` 算出来的
- 合并后的秩 r = r₁ + r₂ 必须等于该 `low_rank`
- 即：`r = int(m * n * ratio / (m + n))`，然后 `r₁ = int(r * r1_ratio)`，`r₂ = r - r₁`

---

## 第三步：超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_H` | 1e-4 | 输入协方差阻尼（Stage 1 和 2 共用） |
| `lambda_phi` | 1e-4 | Fisher Φ 因子阻尼 |
| `lambda_psi` | 1e-4 | Fisher Ψ 因子阻尼 |
| `r1_ratio` | 0.8 | r₁/r 的比例（默认 4:1） |
| `alpha_min` | 0.4 | ACES α 下界（复用 SAES-SVD v2.1.4 的设置） |
| `alpha_max` | 0.6 | ACES α 上界 |
| `rho` | 0.95 | ACES 收缩因子 |
| `eps_eigen` | 1e-6 | 特征值截断阈值（Stage 2 的 Λ^{-1/2}） |

---

## 第四步：代码结构要求

```python
# FR_SVD.py 的顶层结构
def estimate_kronecker_fisher(...)     # Fisher 因子估计
def fisher_beta_select(...)            # Fisher 空间 β 优化（可复用/适配 aces_beta_select_unified）
def fr_svd_compress_layerwise(...)     # 主压缩函数（对标 saes_svd_compress_layerwise）
def smoke_test()                       # 单元测试

if __name__ == '__main__':
    parser = argparse.ArgumentParser(...)
    # 命令行参数与 SAES-SVD_v2.1.4.py 保持一致的风格
    # 新增：--r1_ratio, --lambda_phi, --lambda_psi
```

**主压缩函数 `fr_svd_compress_layerwise` 的签名**必须兼容现有调用模式：
```python
def fr_svd_compress_layerwise(
    model_name, model, calib_loader, ratio, dev,
    eps=1e-4, r1_ratio=0.8,
    lambda_phi=1e-4, lambda_psi=1e-4,
    rho=0.95, alpha_min=0.4, alpha_max=0.6,
    verbose=True
):
```

---

## 第五步：Smoke Test（验收条件）

在文件末尾实现 `smoke_test()` 函数，验证：

```python
def smoke_test():
    """用小型线性层验证 FR-SVD 的正确性"""
    torch.manual_seed(42)
    m, n, p = 128, 64, 256  # 输出维度, 输入维度, token数
    r = 16  # 总秩预算
    r1 = 12  # Stage 1
    r2 = 4   # Stage 2

    W = torch.randn(m, n)
    X = torch.randn(n, p)      # student 激活
    X_f = X + 0.1 * torch.randn(n, p)  # teacher 激活（模拟累积误差）

    # 验证项：
    # 1. 维度正确：B^(1) ∈ R^{m×r₁}, A^(1) ∈ R^{r₁×n}, B^(2) ∈ R^{m×r₂}, A^(2) ∈ R^{r₂×n}
    # 2. 合并后 rank(W') ≤ r
    # 3. Lemma 1：||E - W^comp X||_F² ≤ ||E||_F²
    # 4. Corollary 1：eigenspace compensation ≤ naive weight compensation
```

---

## 关键注意事项

1. **Fisher 估计需要梯度**：与 H/Δ 的统计收集分开。先在 `@torch.no_grad` 下收集 H/Δ，再单独做 Fisher 估计（需要前向+反向）
2. **显存管理**：参考 SAES-SVD 的模式——大矩阵存 CPU fp16，计算时转 GPU fp32
3. **数值稳定性**：
   - H^{-1/2} 用特征分解 + 特征值截断（参考 `compute_H_inv_sqrt`）
   - Cholesky 非正定时加阻尼（参考 SVDLLM.py 的处理）
   - `torch.linalg.solve_triangular` 替代显式求逆
4. **OPT 模型支持**：与 SAES-SVD 保持一致的分支处理
