#coding:utf8
"""
版本历史：
  v1.0  初始版本。Fisher Phase B（Kronecker Φ/Ψ 估计 + Cholesky 分解）因
        solve_triangular 逆变换放大噪声（W_err = 0.3-0.7）而禁用，退化为
        纯 H-weighted CEALC + 旧版白化残差补偿。
        实验结果：rho=0.6 时 PPL=7.894，优于 SAES-SVD 基线（8.054）。

  v1.1  Plan A 尝试：Stage 1 使用完整秩 r，Stage 2 使用额外秩 r2（r_total > r），
        SVD 模块以扩大的 ratio_eff 创建。Stage 2 沿用旧版双重 Λ^{-1/2}
        白化路径，导致 PPL 严重劣化（14.4-14.7），实验失败。

  v1.2  （当前版本）针对 v1.1 的两个失败原因分别修复：
        ① 创新点A（Psi 加权，Fisher 空间变换）：
           - 放弃 Cholesky + solve_triangular，改为对 Psi 做特征分解并 clamp
             小特征值，通过 Psi^{1/2} / Psi^{-1/2} 实现稳定的左侧 Fisher 变换
           - 逆变换仅一次矩阵乘法：B1 = Psi^{-1/2} @ U @ Σ^{1/2}，无数值爆炸风险
           - 通过 --use_fisher 开关控制启用/禁用，便于消融对比
        ② 创新点B（H-weighted 残差补偿，Stage 2）：
           - 放弃旧版双重 Λ^{-1/2} 白化路径（等效 H^{-1}，小特征值方向噪声被平方放大）
           - 改用与 Stage 1 完全相同的稳定路径：R = E_w @ H^{1/2}，SVD 后
             A2 = ... @ H^{-1/2}，单次逆变换，W_err 可控
        ③ 参数预算恢复同参数设置：r1 + r2 = r（不再扩充总秩），消融实验公平对比
        
  消融实验设计（rho=0.6, ratio=0.2, LLaMA-7B, wikitext2）：
        Exp1  Baseline：创新点A OFF + 创新点B OFF（r1_ratio=1.0）
        Exp2  A only  ：创新点A ON  + 创新点B OFF（r1_ratio=1.0）
        Exp3  B only  ：创新点A OFF + 创新点B ON （r1_ratio=0.9）
        Exp4  A + B   ：创新点A ON  + 创新点B ON （r1_ratio=0.9）

FR-SVD：基于 Fisher 信息引导的累积误差感知 SVD + 特征空间残差补偿

两阶段低秩压缩框架，用于大语言模型压缩：

  第一阶段（Stage 1）：Fisher-Guided CEALC（累积误差感知低秩压缩）
    - 在 Kronecker 分解的 Fisher 度量空间中执行 SVD 截断
    - 利用 Φ（输入端）和 Ψ（输出端）两个 Fisher 因子构成黎曼度量
    - 通过 FS-FOA（Fisher空间自适应 β 优化）选择最优截断方向

  第二阶段（Stage 2）：Eigenspace Residual Compensation（特征空间残差补偿）
    - 利用 H 的特征空间（即输入激活的主成分方向）构造残差补偿项
    - 通过岭回归 shortcut 将补偿投影到激活特征空间，避免双重SVD开销

论文参考：FR-SVD paper §3-4，Algorithm 1
"""

import os
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

# 将当前脚本目录加入模块搜索路径，确保 utils/component 等子包可导入
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

from utils.data_utils import *
from utils.model_utils import *
from evaluater import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# ==================== 辅助函数 ====================

def compute_H_decomposition(H_nodamp, device, eps=1e-4):
    """
    对输入协方差矩阵 H 进行阻尼处理并特征分解，返回 H^{1/2} 和 H^{-1/2}。

    论文 §3.1 中，H = X X^T 是未阻尼的输入协方差矩阵（即 H_nodamp),
    加上阻尼项 ε·I 后得到正定矩阵 H_reg = H_nodamp + ε·I,
    对其做特征分解 H_reg = Q Λ Q^T，再构造：
        H^{1/2}  = Q diag(√λ) Q^T
        H^{-1/2} = Q diag(1/√λ) Q^T
    这两个矩阵在 Stage 1 的 Fisher 空间变换和 Stage 2 的白化投影中都需要用到。

    参数：
        H_nodamp : 未加阻尼的输入协方差矩阵，形状 (n, n)，即 XX^T 的累加
        device   : 计算设备（如 'cuda:0'）
        eps      : H 的阻尼系数 λ_H，防止特征值为零导致数值不稳定

    返回：
        Q          : 特征向量矩阵，形状 (n, n)，列向量为 H 的正交基
        Lambda     : 阻尼后的特征值向量，形状 (n,)，已 clamp 到 >= eps
        H_sqrt     : H^{1/2}，形状 (n, n)
        H_inv_sqrt : H^{-1/2}，形状 (n, n)
    """
    d = H_nodamp.shape[0]

    # 当矩阵维度超过 4096 时（如 MLP 层的大权重矩阵），在 CPU 上做特征分解避免 GPU OOM
    if d > 4096:
        H_reg = H_nodamp.cpu().float() + eps * torch.eye(d, dtype=torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)
    else:
        H_reg = H_nodamp.to(device).float() + eps * torch.eye(d, device=device, dtype=torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)

    # 将特征值 clamp 到 eps 以下，防止极小负特征值（数值误差）导致开方失败
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    sqrt_lam = torch.sqrt(eigenvalues)
    inv_sqrt_lam = 1.0 / sqrt_lam

    Q = eigenvectors.to(device)
    Lambda = eigenvalues.to(device)

    # H^{1/2} = Q diag(√λ) Q^T，利用广播优化，等价于 Q @ diag(sqrt_lam) @ Q^T
    # (Q * sqrt_lam.unsqueeze(0)) 对每列乘以对应特征值的平方根，再右乘 Q^T
    H_sqrt = (Q * sqrt_lam.to(device).unsqueeze(0)) @ Q.T
    H_inv_sqrt = (Q * inv_sqrt_lam.to(device).unsqueeze(0)) @ Q.T

    return Q, Lambda, H_sqrt, H_inv_sqrt


def estimate_kronecker_fisher(layer, teacher_layer, inps_student, inps_teacher,
                               attention_masks, position_ids, model_name, dev,
                               num_samples, lambda_phi=1e-4, lambda_psi=1e-4,
                               max_fisher_samples=32):
    """
    估计每个线性子层的 Kronecker 分解 Fisher 信息矩阵（Φ, Ψ），返回其 Cholesky 因子。

    论文 §3.2 中，针对线性层 W ∈ R^{m×n}，真实 Fisher 信息矩阵 F(W) 是 (mn)×(mn)
    的大矩阵，存储和求逆均不现实。采用 Kronecker 近似：
        F(W) ≈ Ψ ⊗ Φ
    其中：
        Φ ∈ R^{n×n}（输入端 Fisher 因子）= E[a a^T]，a 为输入激活向量
        Ψ ∈ R^{m×m}（输出端 Fisher 因子）= E[g g^T]，g 为输出梯度向量

    具体估计方式：以教师模型输出为参考，对学生模型输出计算 MSE proxy loss，
    触发反向传播，从 forward/backward hook 中收集 a 和 g，累加得到 Φ 和 Ψ 的估计。

    最后加阻尼 λ_Φ·I 和 λ_Ψ·I 后做 Cholesky 分解：
        Φ + λ_Φ·I = L_Φ L_Φ^T
        Ψ + λ_Ψ·I = L_Ψ L_Ψ^T

    这两个下三角矩阵 L_Φ, L_Ψ 即为后续 Stage 1 Fisher 空间变换所需。

    参数：
        layer              : 当前学生 Transformer 层（已在 GPU 上，float32）
        teacher_layer      : 对应教师层（fp16，用于生成参考输出）
        inps_student       : 所有样本的学生层输入，形状 (N, seq_len, hidden)，CPU fp16
        inps_teacher       : 所有样本的教师层输入，形状同上
        attention_masks    : 每个样本的注意力掩码列表
        position_ids       : 每个样本的位置编码列表（OPT 模型为 None）
        model_name         : 模型名称字符串，用于区分 LLaMA/Mistral/OPT 的 API 差异
        dev                : 计算设备
        num_samples        : 总样本数
        lambda_phi         : Φ 的阻尼系数
        lambda_psi         : Ψ 的阻尼系数
        max_fisher_samples : 用于 Fisher 估计的最大样本数（过多样本会增加显存压力）

    返回：
        dict: {子层名称: (L_Phi, L_Psi)} —— Cholesky 因子，存储在 CPU
    """
    subset = find_layers(layer)

    # 初始化 Φ 和 Ψ 的累加器（CPU float32，避免显存不足）
    phi_accum = {}
    psi_accum = {}
    input_cache = {}  # 临时存储 forward hook 捕获的输入激活

    for name in subset:
        n_in = subset[name].in_features
        n_out = subset[name].out_features
        phi_accum[name] = torch.zeros(n_in, n_in, dtype=torch.float32)
        psi_accum[name] = torch.zeros(n_out, n_out, dtype=torch.float32)

    def make_hooks(name):
        """
        为子层 `name` 创建一对 forward/backward hook，用于收集激活和梯度。

        forward_hook：捕获输入激活 a（形状 batch*seq × n_in），
                      存入 input_cache[name]，供 backward_hook 使用。
        backward_hook：捕获输出梯度 g（形状 batch*seq × n_out），
                       累加 Φ += a^T a，Ψ += g^T g。
        """
        def forward_hook(module, inp, out):
            a = inp[0].detach().float()
            if a.dim() == 3:
                # 将 (batch, seq, n_in) 展平为 (batch*seq, n_in)
                a = a.reshape(-1, a.shape[-1])
            input_cache[name] = a

        def backward_hook(module, grad_input, grad_output):
            b = grad_output[0].detach().float()
            if b.dim() == 3:
                # 将 (batch, seq, n_out) 展平为 (batch*seq, n_out)
                b = b.reshape(-1, b.shape[-1])
            a = input_cache.get(name)
            if a is not None:
                # Φ 累加：Φ += a^T a （输入激活的外积期望值）
                # Ψ 累加：Ψ += g^T g （输出梯度的外积期望值）
                phi_accum[name] += (a.T @ a).cpu()
                psi_accum[name] += (b.T @ b).cpu()

        return forward_hook, backward_hook

    # 注册所有子层的 hook
    handles = []
    for name in subset:
        fh, bh = make_hooks(name)
        handles.append(subset[name].register_forward_hook(fh))
        handles.append(subset[name].register_full_backward_hook(bh))

    # 开启梯度，Fisher 估计需要反向传播
    for p in layer.parameters():
        p.requires_grad_(True)

    # 实际使用的 Fisher 样本数（不超过 max_fisher_samples）
    n_fisher = min(num_samples, max_fisher_samples)

    for j in range(n_fisher):
        inp_j = inps_student[j].unsqueeze(0).to(dev).float()
        teacher_inp_j = inps_teacher[j].unsqueeze(0).to(dev).float()
        attn_mask = attention_masks[j].to(dev)
        pos_ids = position_ids[j].to(dev) if position_ids is not None else None

        layer.zero_grad()
        # 需要 requires_grad=True 才能让 backward 流过 layer 的参数
        inp_for_grad = inp_j.detach().requires_grad_(True)

        # 学生模型前向（触发 forward hook 收集 a）
        if "opt" not in model_name:
            out_student = layer(inp_for_grad, attention_mask=attn_mask, position_ids=pos_ids)[0]
        else:
            out_student = layer(inp_for_grad, attention_mask=attn_mask)[0]

        # 教师模型前向（无梯度，仅生成参考输出）
        with torch.no_grad():
            out_teacher = teacher_layer.float()(
                teacher_inp_j,
                attention_mask=attn_mask,
                **({"position_ids": pos_ids} if "opt" not in model_name and pos_ids is not None else {})
            )[0]
            teacher_layer.half()  # 教师模型用完立即转回 fp16 节省显存

        # MSE proxy loss：以教师输出为目标，触发反向传播（收集 g）
        loss = nn.functional.mse_loss(out_student, out_teacher.detach())
        loss.backward()  # 此处触发 backward hook 累加 Ψ

        del inp_j, teacher_inp_j, out_student, out_teacher, loss, inp_for_grad
        input_cache.clear()  # 清空当前样本的激活缓存
        if j % 16 == 0:
            torch.cuda.empty_cache()

    # 摘除所有 hook
    for h in handles:
        h.remove()

    # 关闭梯度（后续压缩不再需要反向传播）
    for p in layer.parameters():
        p.requires_grad_(False)

    # 对 Φ 和 Ψ 做归一化、加阻尼、Cholesky 分解
    result = {}
    for name in subset:
        # 按样本数归一化，得到期望值
        # 仅使用 Psi（输出端 Fisher 因子），Phi ≈ H 已由 H_matrix 提供
        psi = psi_accum[name] / n_fisher
        psi += lambda_psi * torch.eye(psi.shape[0])

        # 对 Psi 做特征分解（比 Cholesky 更稳定，可 clamp 小特征值）
        # Psi = P_psi @ diag(eigvals_psi) @ P_psi^T
        m = psi.shape[0]
        if m > 4096:
            eigvals_psi, eigvecs_psi = torch.linalg.eigh(psi.float())
        else:
            eigvals_psi, eigvecs_psi = torch.linalg.eigh(psi.float())
        eigvals_psi = torch.clamp(eigvals_psi, min=lambda_psi)

        result[name] = (eigvecs_psi, eigvals_psi)  # 均存在 CPU

    del phi_accum, psi_accum
    torch.cuda.empty_cache()
    return result


def fisher_beta_select(S_F, D_F, rank, alpha_min=0.4, alpha_max=0.6,
                        rho=0.95, beta_cap=0.375):
    """
    在 Fisher 度量空间中通过 FS-FOA（Fisher Space First-Order Approximation）
    选择最优的累积误差混合系数 β*。

    背景（论文 §3.3）：
    CEALC 将两个信号矩阵合并为 G^F(β) = S^F + β·D^F，其中：
        S^F = L_Ψ^T W H^{1/2} L_Φ   （当前层权重在 Fisher 空间的映射）
        D^F = L_Ψ^T W Δ H^{-1/2} L_Φ （累积误差信号在 Fisher 空间的映射）
    对 G^F(β) 做秩-r₁ SVD 截断，β 控制对累积误差的关注程度。

    最优 β* 通过最小化 RER（Rank-r₁ Relative Error Ratio）确定：
        RER(β) = ||G^F_⊥(β)||_F² / ||G^F(β)||_F²
    其中 G^F_⊥ 是 G^F 在秩-r₁ 子空间正交补上的投影（即被截断掉的部分）。

    RER(β) 是 β 的有理函数，对 β 求导令其为零得到二次方程，
    解出候选点后与边界 [β_min, β_max] 一同评估，取最小值。
    最后乘以收缩因子 ρ 并约束上界 beta_cap，防止 β 过大导致误差信号主导。

    α 与 β 的关系：α = β/(1-β)（论文将 α 作为超参数，内部转为 β 处理）

    参数：
        S_F      : Fisher 空间的权重信号矩阵，形状 (m, n)
        D_F      : Fisher 空间的累积误差信号矩阵，形状 (m, n)
        rank     : Stage 1 的目标秩 r₁
        alpha_min: ACES α 参数的下界（对应 β_min = α_min/(1+α_min)）
        alpha_max: ACES α 参数的上界（对应 β_max = α_max/(1+α_max)）
        rho      : 收缩因子，将最优 β 缩小以避免过激调整（默认 0.95）
        beta_cap : β 的硬上限安全阀（默认 0.375，对应 α=0.6）

    返回：
        beta_star : 最优 β 值
        info      : 诊断字典，包含 alpha_star、RER 系数等
    """
    # α → β 变换：β = α/(1+α)，单调递增，保证 β ∈ (0,1)
    beta_min = alpha_min / (1 + alpha_min)
    beta_max = alpha_max / (1 + alpha_max)

    S_F = S_F.float()
    D_F = D_F.float()

    # 对 S^F 做完整 SVD，取前 r₁ 个左/右奇异向量，构成秩-r₁ 子空间的基
    U, Sigma, Vt = torch.linalg.svd(S_F, full_matrices=False)
    U_r = U[:, :rank]    # (m, r₁)，左奇异向量
    V_r = Vt[:rank, :].T # (n, r₁)，右奇异向量

    # 计算 G^F 在秩-r₁ 子空间正交补上的投影（即截断掉的部分）
    # 正交补投影 = I - U_r U_r^T（左）和 I - V_r V_r^T（右）的组合
    # 利用恒等式展开以避免显式构建 (m×m) 的投影矩阵：
    #   P_⊥(M) = M - U_r(U_r^T M) - (M V_r)V_r^T + U_r(U_r^T M V_r)V_r^T
    UrTS = U_r.T @ S_F          # (r₁, n)
    SVr = S_F @ V_r             # (m, r₁)
    UrTSVr = U_r.T @ SVr       # (r₁, r₁)
    S_perp = S_F - U_r @ UrTS - SVr @ V_r.T + U_r @ UrTSVr @ V_r.T  # S^F 的正交补分量

    UrTD = U_r.T @ D_F
    DVr = D_F @ V_r
    UrTDVr = U_r.T @ DVr
    D_perp = D_F - U_r @ UrTD - DVr @ V_r.T + U_r @ UrTDVr @ V_r.T  # D^F 的正交补分量

    # RER(β) 的分子 = ||S^F_⊥ + β D^F_⊥||_F² = a + 2bβ + cβ²
    # RER(β) 的分母 = ||S^F + β D^F||_F²    = A + 2Bβ + Cβ²
    a = torch.norm(S_perp, 'fro') ** 2
    b = torch.sum(S_perp * D_perp)   # 内积 <S_⊥, D_⊥>
    c = torch.norm(D_perp, 'fro') ** 2
    A_coef = torch.norm(S_F, 'fro') ** 2
    B_coef = torch.sum(S_F * D_F)    # 内积 <S^F, D^F>
    C_coef = torch.norm(D_F, 'fro') ** 2

    # 对 RER(β) 求导并令其为零，化简得二次方程：
    # coef_2·β² + coef_1·β + coef_0 = 0
    coef_2 = c * B_coef - b * C_coef
    coef_1 = c * A_coef - a * C_coef
    coef_0 = b * A_coef - a * B_coef

    # 候选点初始化为边界值
    candidates = [beta_min, beta_max]

    # 若二次项系数不为零，求解极值点（二次公式）
    if abs(coef_2.item()) > 1e-10:
        discriminant = coef_1 ** 2 - 4 * coef_2 * coef_0
        if discriminant >= 0:
            sqrt_disc = torch.sqrt(discriminant)
            candidates.append((-coef_1 + sqrt_disc).item() / (2 * coef_2.item()))
            candidates.append((-coef_1 - sqrt_disc).item() / (2 * coef_2.item()))

    # 在所有候选点中（约束到 [β_min, β_max]）选 RER 最小的
    best_beta = beta_min
    best_score = float('inf')

    for beta_cand in candidates:
        beta_val = max(beta_min, min(beta_cand, beta_max))
        if beta_val < 0 or beta_val >= 1:
            continue
        numerator = a + 2 * b * beta_val + c * beta_val ** 2
        denominator = A_coef + 2 * B_coef * beta_val + C_coef * beta_val ** 2 + 1e-10
        score = (numerator / denominator).item()
        if score < best_score:
            best_score = score
            best_beta = beta_val

    # 收缩保护：ρ 收缩 + beta_cap 硬约束，防止误差信号过度主导
    best_beta = rho * min(best_beta, beta_cap)
    # 将 β 反转回 α（论文报告 α*）
    alpha_star = best_beta / (1 - best_beta) if best_beta < 1 else float('inf')

    return best_beta, {
        'alpha_star': alpha_star,
        'best_score': best_score,
        'a': a.item(), 'b': b.item(), 'c': c.item(),
        'A': A_coef.item(), 'B': B_coef.item(), 'C': C_coef.item()
    }


# ==================== 主压缩函数 ====================

@torch.no_grad()
def fr_svd_compress_layerwise(model_name, model, calib_loader, ratio, dev,
                                eps=1e-4, r1_ratio=0.8,
                                use_fisher=True, max_fisher_samples=32,
                                lambda_psi=1e-4,
                                rho=0.95, alpha_min=0.4, alpha_max=0.6,
                                verbose=True):
    """
    FR-SVD 逐层闭环压缩主函数。

    对模型的每一个 Transformer 层 ℓ 依次执行以下六个阶段：

      Phase A（H/Δ 收集）：
          - 用校准数据对学生层和教师层分别做前向，
            通过 hook 收集输入激活 X 和教师激活 X^f
          - 计算输入协方差 H = Σ X X^T（即 X X^T 的样本均值）
          - 计算累积误差矩阵 Δ = Σ (X^f - X) X^T
            Δ 量化了当前层相比未压缩教师模型在输入分布上的漂移

      Phase B（Fisher 估计）：
          - 开启梯度，以教师输出为目标做 MSE 反向传播
          - 从 forward/backward hook 累加 Φ = E[aa^T] 和 Ψ = E[gg^T]
          - Cholesky 分解得到 L_Φ, L_Ψ

      Phase C（创建 SVD 替换模块）：
          - 根据模型类型实例化 SVD_LlamaAttention/SVD_LlamaMLP 等 SVD 替换层
          - 这些层持有 u_proj 和 v_proj 两组权重（W ≈ B·A）

      Phase D（两阶段压缩）：对每个子层（q/k/v/o_proj, gate/up/down_proj）：
          Stage 1 — Fisher-Guided CEALC（论文 §3.1-3.3）：
            1. 计算 S = W H^{1/2}，D = W Δ H^{-1/2}（权重在H度量空间的表示）
            2. Fisher 空间变换：S^F = L_Ψ^T S L_Φ，D^F = L_Ψ^T D L_Φ
            3. FS-FOA 选 β*，构造 G^F(β*) = S^F + β* D^F
            4. 对 G^F 做秩-r₁ SVD 截断，得 U_{r1}, Σ_{r1}, V_{r1}^T
            5. 逆变换回权重空间：
                B^(1) = L_Ψ^{-T} U_{r1} Σ^{1/2}
                A^(1) = Σ^{1/2} V_{r1}^T L_Φ^{-1} H^{-1/2}

          Stage 2 — Eigenspace Residual Compensation（论文 §3.4，Lemma 1, Corollary 1）：
            1. 计算全局输出对齐误差 E·X^T = W(H + Δ) - B^(1)A^(1)·H
            2. 白化投影：ΔW_white = E·X^T · Q · Λ^{-1/2}（岭回归 shortcut）
            3. 对 ΔW_white 做秩-r₂ SVD，得 B^(2), A^(2)

          合并：W' = [B^(1)|B^(2)] · [A^(1); A^(2)]（总秩 r = r₁ + r₂）
          数值安全：若重建误差 W_err > 5.0 则回退到 H-weighted SVD

      Phase E（替换层）：
          - 将压缩后的 B_merged/A_merged 写入 SVD 替换模块
          - 用 SVD 替换模块替换原始 Transformer 层的 self_attn/mlp

      Phase F（闭环 re-forward）：
          - 用压缩后的层重新前向校准数据，更新 outs_student
          - 将其作为下一层的输入 inps_student（闭环传播）
          - 教师模型的输出同步更新 inps_teacher（维持累积误差估计的正确性）

    参数：
        model_name         : 模型名称字符串（用于区分 LLaMA/Mistral/OPT 结构差异）
        model              : 待压缩的学生模型（就地修改）
        calib_loader       : 校准数据加载器（list of dict，含 input_ids/attention_mask）
        ratio              : 保留比例（如 0.8 表示保留 80% 参数量）
        dev                : 计算设备字符串（如 'cuda:0'）
        eps                : H 矩阵阻尼系数 λ_H
        r1_ratio           : Stage 1 秩占比 r₁/r（默认 0.8）
        lambda_phi         : Fisher Φ 阻尼系数
        lambda_psi         : Fisher Ψ 阻尼系数
        rho                : ACES β 收缩因子
        alpha_min/max      : ACES α 参数范围
        max_fisher_samples : Fisher 估计使用的最大样本数
        verbose            : 是否打印每层详细信息

    返回：
        compression_stats : 每层压缩统计信息的列表
    """
    print("=" * 60)
    print("FR-SVD v1.2: Psi-weighted CEALC + H-weighted Residual Compensation")
    print("=" * 60)
    print(f"  保留比例: {ratio} ({ratio*100:.1f}%)")
    print(f"  秩分配: r1={r1_ratio*100:.0f}% of r / r2={100-r1_ratio*100:.0f}% of r (r1+r2=r, 同参数预算)")
    print(f"  创新点A (Fisher Psi 加权): {'ON' if use_fisher else 'OFF'}")
    print(f"  创新点B (残差补偿 Stage 2): {'ON (r2>0)' if r1_ratio < 1.0 else 'OFF (r2=0)'}")
    print(f"  H 阻尼: eps={eps}")
    print(f"  Alpha 范围: [{alpha_min}, {alpha_max}], rho={rho}")
    print("=" * 60)

    # ========== Step 1：加载教师模型 ==========
    # 教师模型保持原始精度（fp16 存储，forward 时转 fp32 保证数值精度），
    # 用于提供各层未压缩的激活作为参考，计算累积误差 Δ
    print("\n加载教师模型（fp16 存储，fp32 计算）...")
    from transformers import AutoModelForCausalLM
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model.config._name_or_path,
        torch_dtype=torch.float16,
        device_map="cpu"   # 初始放在 CPU，逐层搬运到 GPU 以节省显存
    )
    teacher_model.eval()

    # ========== Step 2：获取模型的层列表 ==========
    # 将 embedding 和 norm 等共享模块移到 GPU，仅在校准阶段使用
    if "opt" in model_name:
        layers = model.model.decoder.layers
        teacher_layers = teacher_model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        # LLaMA / Mistral 及其变种
        layers = model.model.layers
        teacher_layers = teacher_model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

    # ========== Step 3：捕获第 0 层的输入激活 ==========
    # 用 Catcher 模块拦截第 0 层的输入，避免完整前向传播消耗额外显存
    layers[0] = layers[0].to(dev)

    num_samples = len(calib_loader)

    # 预分配存储所有样本的第 0 层输入激活（CPU fp16，节省显存）
    inps_student = torch.zeros(
        (num_samples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16, device='cpu'
    )
    inps_teacher = torch.zeros_like(inps_student)

    cache = {'i': 0}          # 样本计数器（闭包共享）
    attention_masks_list = []  # 收集每个样本的 attention_mask
    position_ids_list = []     # 收集每个样本的 position_ids（OPT 不需要）

    class StudentCatcher(nn.Module):
        """
        临时替换第 0 层，在 forward 时拦截输入并保存，然后抛出 ValueError
        中断模型前向（不需要计算实际输出，节省时间）。
        """
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps_student[cache['i']] = inp.detach().half().cpu()
            attention_masks_list.append(kwargs['attention_mask'].detach().cpu())
            if "opt" not in model_name:
                pos_ids = kwargs.get('position_ids')
                position_ids_list.append(pos_ids.detach().cpu() if pos_ids is not None else None)
            cache['i'] += 1
            raise ValueError  # 中断前向，无需计算后续层

    class TeacherCatcher(nn.Module):
        """同 StudentCatcher，但拦截教师模型第 0 层的输入。"""
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.idx = 0
        def forward(self, inp, **kwargs):
            inps_teacher[self.idx] = inp.detach().half().cpu()
            self.idx += 1
            raise ValueError

    layers[0] = StudentCatcher(layers[0])
    teacher_layers[0] = TeacherCatcher(teacher_layers[0])

    print("收集第 0 层输入激活...")
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass  # 被 Catcher 中断，属于正常流程
        try:
            teacher_model.to(dev)
            teacher_model(**batch)
        except ValueError:
            pass
        finally:
            teacher_model.to("cpu")  # 立即移回 CPU，防止显存泄漏

    # 恢复第 0 层并移回 CPU
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    teacher_layers[0] = teacher_layers[0].module

    # 将 embedding/norm 移回 CPU，为后续逐层压缩腾出显存
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()

    torch.cuda.empty_cache()

    # 分配存储各层输出激活的缓冲区（逐层更新，用作下一层的输入）
    outs_student = torch.zeros_like(inps_student)
    outs_teacher = torch.zeros(
        (num_samples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16, device='cpu'
    )
    attention_masks = attention_masks_list
    position_ids = position_ids_list if position_ids_list and position_ids_list[0] is not None else None

    # ========== Step 4：逐层压缩主循环 ==========
    compression_stats = []

    print("\n开始逐层 FR-SVD 压缩...")

    for layer_idx in tqdm(range(len(layers)), desc="FR-SVD Compression"):
        layer_stats = {'layer': layer_idx, 'sublayers': {}}

        # 将当前学生层和教师层移到 GPU 进行计算
        # 学生层转 float32 保证数值精度；教师层保持 fp16 后续 forward 时再转 fp32
        layer = layers[layer_idx].to(dev).float()
        teacher_layer = teacher_layers[layer_idx].to(dev)

        # 获取当前层的所有线性子层（q/k/v/o_proj, gate/up/down_proj 等）
        subset = find_layers(layer)
        teacher_subset = find_layers(teacher_layer)

        # ────────── Phase A：收集 H 和 Δ（无梯度）──────────
        # 为每个子层初始化 H 矩阵累加器和输入缓存
        for name in subset:
            subset[name].H_matrix = 0            # 累加 H = Σ X X^T
            subset[name].inp_cache = []           # 保存每个样本的学生输入 X
            teacher_subset[name].inp_cache = []   # 保存每个样本的教师输入 X^f

        def make_hook_student(module_name):
            """
            学生层 forward hook：
            - 累加 H = Σ (X^T X) 的批次贡献（即 X·X^T，注意 Linear 的输入是行向量）
            - 将输入缓存下来供后续计算 Δ 使用
            """
            def hook(module, input, output):
                inp = input[0].detach().float()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)  # (batch, seq, n_in)
                # H_batch[k] = X_k^T X_k，对 batch 维度求和
                H_batch = torch.matmul(inp.transpose(1, 2), inp)
                module.H_matrix = module.H_matrix + torch.sum(H_batch, dim=0)
                module.inp_cache.append(inp.half().cpu())  # fp16 存 CPU 节省显存
            return hook

        def make_hook_teacher(module_name):
            """
            教师层 forward hook：仅缓存输入激活 X^f，用于计算 Δ = Σ (X^f - X) X^T
            """
            def hook(module, input, output):
                inp = input[0].detach()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                module.inp_cache.append(inp.half().cpu())
            return hook

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_hook_student(name)))
            handles.append(teacher_subset[name].register_forward_hook(make_hook_teacher(name)))

        # 用 calib 数据对当前层做前向，收集 H 和 inp_cache
        for j in range(num_samples):
            inp_j = inps_student[j].unsqueeze(0).to(dev).float()
            attn_mask_j = attention_masks[j].to(dev)
            pos_ids_j = position_ids[j].to(dev) if position_ids is not None else None

            if "opt" not in model_name:
                out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)[0]
                outs_student[j] = out_j.half().cpu()

                teacher_inp = inps_teacher[j].unsqueeze(0).to(dev).float()
                teacher_out = teacher_layer.float()(
                    teacher_inp, attention_mask=attn_mask_j, position_ids=pos_ids_j
                )[0]
                outs_teacher[j] = teacher_out.half().cpu()
                teacher_layer.half()  # 及时转回 fp16
            else:
                out_j = layer(inp_j, attention_mask=attn_mask_j)[0]
                outs_student[j] = out_j.half().cpu()

                teacher_inp = inps_teacher[j].unsqueeze(0).to(dev).float()
                teacher_out = teacher_layer.float()(teacher_inp, attention_mask=attn_mask_j)[0]
                outs_teacher[j] = teacher_out.half().cpu()
                teacher_layer.half()

            del inp_j, out_j, teacher_inp, teacher_out
            if j % 32 == 0:
                torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        # 计算累积误差矩阵 Δ = Σ (X^f - X) X^T
        # 其中 X^f 是教师激活，X 是学生激活
        # Δ 反映了当前层输入分布相对教师模型的系统性漂移（累积压缩误差）
        for name in subset:
            subset[name].Delta_matrix = 0
            for idx in range(len(subset[name].inp_cache)):
                X = subset[name].inp_cache[idx].to(dev).float()
                X_f = teacher_subset[name].inp_cache[idx].to(dev).float()
                diff = X_f - X  # 教师与学生激活之差（形状 batch × seq × n_in）
                # Delta_batch[k] = diff_k^T @ X_k，对 batch 维求和
                Delta_batch = torch.matmul(diff.transpose(1, 2), X)
                subset[name].Delta_matrix = subset[name].Delta_matrix + torch.sum(Delta_batch, dim=0)
                del X, X_f, diff, Delta_batch

            # 移回 CPU，释放 GPU 显存
            subset[name].H_matrix = subset[name].H_matrix.cpu()
            subset[name].Delta_matrix = subset[name].Delta_matrix.cpu()
            del subset[name].inp_cache
            del teacher_subset[name].inp_cache

        torch.cuda.empty_cache()

        # ────────── Phase B：Fisher Psi 估计（创新点A）──────────
        psi_decomp = {}
        if use_fisher:
            # 开启梯度（主函数 @torch.no_grad 下需要显式 enable_grad）
            with torch.enable_grad():
                fisher_result = estimate_kronecker_fisher(
                    layer, teacher_layer, inps_student, inps_teacher,
                    attention_masks, position_ids, model_name, dev,
                    num_samples, lambda_psi=lambda_psi,
                    max_fisher_samples=max_fisher_samples
                )
            # fisher_result: {name: (eigvecs_psi, eigvals_psi)}
            for name in fisher_result:
                eigvecs_psi, eigvals_psi = fisher_result[name]
                P_psi = eigvecs_psi.to(dev).float()
                lam_psi = eigvals_psi.to(dev).float()
                sqrt_psi = torch.sqrt(lam_psi)
                inv_sqrt_psi = 1.0 / sqrt_psi
                # Psi^{1/2} = P @ diag(√λ) @ P^T, Psi^{-1/2} = P @ diag(1/√λ) @ P^T
                Psi_half = (P_psi * sqrt_psi.unsqueeze(0)) @ P_psi.T
                Psi_inv_half = (P_psi * inv_sqrt_psi.unsqueeze(0)) @ P_psi.T
                psi_decomp[name] = (Psi_half, Psi_inv_half)
                del P_psi, lam_psi, sqrt_psi, inv_sqrt_psi
            del fisher_result
            torch.cuda.empty_cache()

        # ────────── Phase C：创建 SVD 替换模块 ──────────
        # 使用原始 ratio（同参数预算），r1 + r2 = r
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio).to(dev)
            svd_mlp = SVD_LlamaMLP(
                hidden_size=layer.hidden_size,
                intermediate_size=model.config.intermediate_size,
                hidden_act=model.config.hidden_act,
                ratio=ratio
            ).to(dev)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio).to(dev)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio).to(dev)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio).to(dev)

        # ────────── Phase D：逐子层两阶段压缩 ──────────
        for name in subset:
            # 获取子层权重 W（形状 m×n）及保留原始精度类型
            W = subset[name].weight.data.float().to(dev)
            original_dtype = subset[name].weight.data.dtype
            m, n = W.shape

            # 将 H 和 Δ 移到 GPU 进行计算
            H_nodamp = subset[name].H_matrix.float().to(dev)
            Delta = subset[name].Delta_matrix.float().to(dev)

            # 计算秩预算（同参数预算：r1 + r2 = r）
            r = int(m * n * ratio / (m + n))
            r = max(1, min(r, min(m, n)))
            r1 = int(r * r1_ratio) if r1_ratio < 1.0 else r
            r2 = r - r1  # 同预算分配，r2 从 r 中分出

            # ── NaN 级联防护 ──
            if not torch.isfinite(H_nodamp).all() or not torch.isfinite(Delta).all():
                print(f"  ⚠ L{layer_idx} {name}: H/Delta 含 NaN（级联误差），"
                      f"替换为 eps*I / 零矩阵")
                H_nodamp = eps * torch.eye(n, device=dev, dtype=torch.float32)
                Delta = torch.zeros(n, n, device=dev, dtype=torch.float32)

            # 对 H 做特征分解，得到 Q, Λ, H^{1/2}, H^{-1/2}
            Q, Lambda, H_sqrt, H_inv_sqrt = compute_H_decomposition(H_nodamp, dev, eps)

            # ===== Stage 1：Psi-weighted CEALC（创新点A）=====
            # 构造 H-weighted 空间中的信号矩阵：
            #   S = W @ H^{1/2}，D = W @ Δ @ H^{-1/2}
            S = W @ H_sqrt
            D = W @ Delta @ H_inv_sqrt

            # 创新点A：Psi 加权（输出端 Fisher 信息）
            # S_F = Ψ^{1/2} @ S，D_F = Ψ^{1/2} @ D
            # 这使得 SVD 截断在 Fisher 度量空间中进行，优先保留高 Fisher 信息方向
            if use_fisher and name in psi_decomp:
                Psi_half, Psi_inv_half = psi_decomp[name]
                S_F = Psi_half @ S
                D_F = Psi_half @ D
            else:
                S_F = S
                D_F = D

            # ACES β 优化选择最优混合系数
            beta, aces_info = fisher_beta_select(
                S_F, D_F, r1,
                alpha_min=alpha_min, alpha_max=alpha_max, rho=rho
            )

            # 构造目标矩阵 G_F = S_F + β*D_F，做秩-r1 SVD 截断
            G_F = S_F + beta * D_F
            U_g, S_g, Vt_g = torch.linalg.svd(G_F, full_matrices=False)
            U_r1 = U_g[:, :r1]
            S_r1 = S_g[:r1]
            Vt_r1 = Vt_g[:r1, :]

            sqrt_s1 = torch.sqrt(S_r1)

            # 逆变换回原始权重空间
            # 创新点A启用时：B^(1) = Ψ^{-1/2} @ U_{r1} @ Σ^{1/2}（撤销左侧 Psi 变换）
            # 创新点A关闭时：B^(1) = U_{r1} @ Σ^{1/2}（与 SAES-SVD 一致）
            if use_fisher and name in psi_decomp:
                B1 = Psi_inv_half @ (U_r1 * sqrt_s1.unsqueeze(0))
            else:
                B1 = U_r1 * sqrt_s1.unsqueeze(0)
            # A^(1) = Σ^{1/2} @ V_{r1}^T @ H^{-1/2}（右侧 H 逆变换，单次乘法，稳定）
            A1 = (sqrt_s1.unsqueeze(1) * Vt_r1) @ H_inv_sqrt

            # 记录 Stage 1 保留的能量比例（诊断指标）
            total_energy = torch.sum(S_g ** 2).item()
            retained_energy = torch.sum(S_r1 ** 2).item()
            energy_ratio = retained_energy / (total_energy + 1e-10)

            # ===== Stage 2：H-weighted 残差补偿（创新点B）=====
            # 使用稳定的 H-weighted SVD 路径（与 Stage 1 相同的数值管道）
            # 避免旧版本中 Λ^{-1/2} 双重反转导致的数值不稳定
            if r2 > 0:
                # 权重残差
                E_w = W - B1 @ A1

                # H-weighted 残差（与 Stage 1 相同的稳定路径）
                R = E_w @ H_sqrt

                # 对 H-weighted 残差做秩-r2 SVD
                U_r, S_r, Vt_r = torch.linalg.svd(R, full_matrices=False)
                U_r2 = U_r[:, :r2]
                S_r2 = S_r[:r2]
                Vt_r2 = Vt_r[:r2, :]

                sqrt_s2 = torch.sqrt(S_r2)
                B2 = U_r2 * sqrt_s2.unsqueeze(0)
                # 单次 H^{-1/2} 逆变换（与 Stage 1 完全相同的路径，稳定！）
                A2 = (sqrt_s2.unsqueeze(1) * Vt_r2) @ H_inv_sqrt

                del E_w, R, U_r, S_r, Vt_r
            else:
                B2 = torch.empty(m, 0, device=dev, dtype=torch.float32)
                A2 = torch.empty(0, n, device=dev, dtype=torch.float32)

            # ===== 合并两阶段低秩因子 =====
            B_merged = torch.cat([B1, B2], dim=1)  # (m, r)
            A_merged = torch.cat([A1, A2], dim=0)  # (r, n)

            # 计算权重重建误差
            W_recon = B_merged.float() @ A_merged.float()
            weight_err = (torch.norm(W - W_recon) / torch.norm(W)).item()

            # 数值安全：若 W_err 异常或含 NaN，退化到纯截断 SVD
            if weight_err > 1.0 or not torch.isfinite(B_merged).all() or not torch.isfinite(A_merged).all():
                print(f"  警告：L{layer_idx} {name} 重建不稳定 (W_err={weight_err:.2f})，回退到纯截断 SVD")
                U_fb, S_fb, Vt_fb = torch.linalg.svd(W, full_matrices=False)
                sqrt_s_fb = torch.sqrt(torch.clamp(S_fb[:r], min=0))
                B_merged = U_fb[:, :r] * sqrt_s_fb.unsqueeze(0)
                A_merged = sqrt_s_fb.unsqueeze(1) * Vt_fb[:r, :]
                W_recon = B_merged.float() @ A_merged.float()
                weight_err = (torch.norm(W - W_recon) / torch.norm(W)).item()
                print(f"    回退后 W_err={weight_err:.4f}")

            B_merged = B_merged.to(original_dtype)
            A_merged = A_merged.to(original_dtype)

            delta_norm = torch.norm(Delta).item()
            h_norm = torch.norm(H_nodamp).item()

            # 记录本子层的统计信息
            layer_stats['sublayers'][name] = {
                'beta': beta,
                'r': r, 'r1': r1, 'r2': r2,
                'weight_err': weight_err,
                'energy_ratio': energy_ratio,
                'delta_norm': delta_norm,
                'h_norm': h_norm,
                'delta_ratio': delta_norm / (h_norm + 1e-10),
                'aces_info': aces_info
            }

            # 打印详细信息（第 0 层、最后一层、每隔 8 层）
            if verbose and (layer_idx == 0 or layer_idx == len(layers) - 1 or layer_idx % 8 == 0):
                print(f"  L{layer_idx} {name}: beta={beta:.4f}, alpha*={aces_info['alpha_star']:.4f}, "
                      f"r={r}(r1={r1}+r2={r2}), E_kept={energy_ratio:.4f}, "
                      f"W_err={weight_err:.4f}, D/H={delta_norm/h_norm:.4f}")

            # 将压缩后的低秩因子写入对应的 SVD 替换模块
            # u_proj.weight = B_merged（(m, r)），v_proj.weight = A_merged（(r, n)）
            # W' ≈ u_proj(v_proj(x)) = B_merged · A_merged · x
            if 'opt' in model_name:
                # OPT 模型使用 q/k/v/out_proj（注意力）和 fc1/fc2（MLP）命名
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data.copy_(B_merged)
                    svd_decoder.self_attn.q_v_proj.weight.data.copy_(A_merged)
                    svd_decoder.self_attn.q_u_proj.bias.data.copy_(layer.self_attn.q_proj.bias.data)
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data.copy_(B_merged)
                    svd_decoder.self_attn.k_v_proj.weight.data.copy_(A_merged)
                    svd_decoder.self_attn.k_u_proj.bias.data.copy_(layer.self_attn.k_proj.bias.data)
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data.copy_(B_merged)
                    svd_decoder.self_attn.v_v_proj.weight.data.copy_(A_merged)
                    svd_decoder.self_attn.v_u_proj.bias.data.copy_(layer.self_attn.v_proj.bias.data)
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data.copy_(B_merged)
                    svd_decoder.self_attn.out_v_proj.weight.data.copy_(A_merged)
                    svd_decoder.self_attn.out_u_proj.bias.data.copy_(layer.self_attn.out_proj.bias.data)
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data.copy_(B_merged)
                    svd_decoder.fc1_v_proj.weight.data.copy_(A_merged)
                    svd_decoder.fc1_u_proj.bias.data.copy_(layer.fc1.bias.data)
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data.copy_(B_merged)
                    svd_decoder.fc2_v_proj.weight.data.copy_(A_merged)
                    svd_decoder.fc2_u_proj.bias.data.copy_(layer.fc2.bias.data)
                    # OPT 层归一化不做 SVD 分解，直接复制
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
            else:
                # LLaMA / Mistral：注意力层使用 q/k/v/o_proj，MLP 使用 gate/up/down_proj
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data.copy_(B_merged)
                    svd_attn.q_v_proj.weight.data.copy_(A_merged)
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data.copy_(B_merged)
                    svd_attn.k_v_proj.weight.data.copy_(A_merged)
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data.copy_(B_merged)
                    svd_attn.v_v_proj.weight.data.copy_(A_merged)
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data.copy_(B_merged)
                    svd_attn.o_v_proj.weight.data.copy_(A_merged)
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data.copy_(B_merged)
                    svd_mlp.gate_v_proj.weight.data.copy_(A_merged)
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data.copy_(B_merged)
                    svd_mlp.down_v_proj.weight.data.copy_(A_merged)
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data.copy_(B_merged)
                    svd_mlp.up_v_proj.weight.data.copy_(A_merged)

            # 释放该子层用到的所有中间张量，防止显存积累
            del W, H_nodamp, Delta
            del Q, Lambda, H_sqrt, H_inv_sqrt
            del S, D, S_F, D_F, G_F, U_g, S_g, Vt_g, U_r1, S_r1, Vt_r1
            del B1, A1, B_merged, A_merged, W_recon
            if r2 > 0:
                del B2, A2
            del subset[name].H_matrix, subset[name].Delta_matrix
            torch.cuda.empty_cache()

        # 释放本层的 Psi 分解结果
        if use_fisher:
            for name in list(psi_decomp.keys()):
                del psi_decomp[name]
            psi_decomp.clear()
            torch.cuda.empty_cache()

        # ────────── Phase E：将原始层替换为 SVD 替换模块 ──────────
        if 'opt' in model_name:
            # OPT：整个 decoder layer 替换为 SVDOPTDecoderLayer
            layers[layer_idx] = svd_decoder
        else:
            # LLaMA/Mistral：仅替换 self_attn 和 mlp 子模块
            layer.self_attn = svd_attn
            layer.mlp = svd_mlp

        # 记录层级统计汇总
        avg_beta = sum(s['beta'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        avg_delta_ratio = sum(s['delta_ratio'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        layer_stats['avg_beta'] = avg_beta
        layer_stats['avg_delta_ratio'] = avg_delta_ratio

        if verbose:
            avg_alpha_star = sum(s['aces_info']['alpha_star'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
            print(f"Layer {layer_idx} Summary: avg_beta={avg_beta:.4f}, avg_alpha*={avg_alpha_star:.4f}, "
                  f"avg_D/H={avg_delta_ratio:.4f}")

        compression_stats.append(layer_stats)

        # ────────── Phase F：闭环重新前向（更新下一层的输入）──────────
        # 用压缩后的层重新前向校准数据，得到压缩后的输出激活
        # 这保证下一层看到的是压缩后模型的真实输出，而非压缩前的（闭环误差修正）
        # 注意：outs_student[j] 在 Phase A 中已由未压缩层的输出初始化。
        # 若压缩后 out_j 含 NaN（说明本层压缩严重失败），保留 Phase A 的有效输出，
        # 避免 NaN 作为下一层的输入引发级联崩溃。
        _phase_f_nan_count = 0
        for j in range(num_samples):
            inp_j = inps_student[j].unsqueeze(0).to(dev).float()
            attn_mask_j = attention_masks[j].to(dev)
            pos_ids_j = position_ids[j].to(dev) if position_ids is not None else None

            if "opt" not in model_name:
                out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)[0]
            else:
                out_j = layers[layer_idx](inp_j, attention_mask=attn_mask_j)[0]

            if torch.isfinite(out_j).all():
                outs_student[j] = out_j.half().cpu()
            else:
                # 保留 Phase A 中未压缩层的输出（不覆盖），阻断 NaN 级联
                _phase_f_nan_count += 1
            del inp_j, out_j
        if _phase_f_nan_count > 0:
            print(f"  ⚠ L{layer_idx}: Phase F 有 {_phase_f_nan_count}/{num_samples} 个样本输出含 NaN，"
                  f"保留 Phase A 输出（未压缩层输出）作为下一层输入")

        torch.cuda.empty_cache()

        # 将本层的输出作为下一层的输入（逐层串联）
        inps_student = outs_student.clone()
        inps_teacher = outs_teacher.clone()

        # 将已压缩的层移回 CPU（节省 GPU 显存供下一层使用）
        layers[layer_idx] = layers[layer_idx].half().cpu()
        teacher_layers[layer_idx] = teacher_layers[layer_idx].cpu()
        torch.cuda.empty_cache()

    # 清理教师模型（压缩完成后不再需要）
    del teacher_model
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("FR-SVD 压缩完成！")
    print("=" * 60)

    all_betas = [s['avg_beta'] for s in compression_stats]
    all_delta_ratios = [s['avg_delta_ratio'] for s in compression_stats]
    print(f"  平均 beta: {sum(all_betas)/len(all_betas):.4f}")
    print(f"  平均 D/H:  {sum(all_delta_ratios)/len(all_delta_ratios):.4f}")
    print(f"  Beta 范围: [{min(all_betas):.4f}, {max(all_betas):.4f}]")

    return compression_stats


# ==================== 冒烟测试 ====================

def smoke_test():
    """
    单元测试：在小型随机矩阵上验证 FR-SVD 核心数学的正确性。

    检验以下三项：
      1. Stage 1 和 Stage 2 输出因子 B^(1), A^(1), B^(2), A^(2) 的维度正确性
      2. Lemma 1：残差补偿不增加误差
         ||E - W^comp X||_F² ≤ ||E||_F²
      3. Corollary 1：特征空间补偿优于朴素权重空间补偿
         ||E - B^(2)A^(2)X||_F² ≤ ||E - W^naive X||_F²
         其中 W^naive 是对 (W - W^(1)) 的秩-r₂ SVD 截断
    """
    print("=" * 50)
    print("FR-SVD Smoke Test")
    print("=" * 50)

    torch.manual_seed(42)
    m, n, p = 128, 64, 256  # 输出维度、输入维度、token 数（等效样本数）
    r = 16   # 总秩预算
    r1 = 12  # Stage 1 秩
    r2 = 4   # Stage 2 秩
    eps = 1e-4

    W = torch.randn(m, n)          # 模拟原始权重矩阵
    X = torch.randn(n, p)          # 模拟学生激活
    X_f = X + 0.1 * torch.randn(n, p)  # 模拟教师激活（含累积误差偏移）

    # ── 计算统计量 ──
    H_nodamp = X @ X.T              # 输入协方差 H = X X^T，形状 (n, n)
    Delta = (X_f - X) @ X.T        # 累积误差矩阵 Δ = (X^f - X) X^T
    H = H_nodamp + eps * torch.eye(n)  # 加阻尼后的 H

    # H 的特征分解
    eigenvalues, Q = torch.linalg.eigh(H)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    sqrt_lam = torch.sqrt(eigenvalues)
    inv_sqrt_lam = 1.0 / sqrt_lam

    H_sqrt = (Q * sqrt_lam.unsqueeze(0)) @ Q.T       # H^{1/2}
    H_inv_sqrt = (Q * inv_sqrt_lam.unsqueeze(0)) @ Q.T  # H^{-1/2}

    # ── Stage 1：H-weighted CEALC（与主压缩逻辑一致）──
    S = W @ H_sqrt                 # 权重在 H 度量空间的信号
    D = W @ Delta @ H_inv_sqrt     # 累积误差在 H 度量空间的信号

    # 选择最优 β*
    beta, info = fisher_beta_select(S, D, r1)
    print(f"  beta*={beta:.4f}, alpha*={info['alpha_star']:.4f}")

    # 在 H-weighted 空间做秩-r₁ SVD 截断
    G = S + beta * D
    U, Sigma, Vt = torch.linalg.svd(G, full_matrices=False)
    U_r1 = U[:, :r1]
    S_r1 = Sigma[:r1]
    Vt_r1 = Vt[:r1, :]
    sqrt_s1 = torch.sqrt(S_r1)

    # 逆变换到权重空间（仅需 H^{-1/2}，无 solve_triangular）
    B1 = U_r1 * sqrt_s1.unsqueeze(0)                        # B^(1)，形状 (m, r₁)
    A1 = (sqrt_s1.unsqueeze(1) * Vt_r1) @ H_inv_sqrt        # A^(1)，形状 (r₁, n)

    assert B1.shape == (m, r1), f"B1 维度错误: {B1.shape}"
    assert A1.shape == (r1, n), f"A1 维度错误: {A1.shape}"
    print("[PASS] Stage 1 维度正确")

    # ── Stage 2：H-weighted 残差补偿（v1.2 稳定版）──
    E_w = W - B1 @ A1  # 权重残差
    R = E_w @ H_sqrt    # H-weighted 残差（与 Stage 1 相同的稳定路径）

    U2, S2, Vt2 = torch.linalg.svd(R, full_matrices=False)
    U_r2 = U2[:, :r2]
    S_r2 = S2[:r2]
    Vt_r2 = Vt2[:r2, :]
    sqrt_s2 = torch.sqrt(S_r2)

    B2 = U_r2 * sqrt_s2.unsqueeze(0)
    A2 = (sqrt_s2.unsqueeze(1) * Vt_r2) @ H_inv_sqrt  # 单次 H^{-1/2}，稳定

    assert B2.shape == (m, r2), f"B2 维度错误: {B2.shape}"
    assert A2.shape == (r2, n), f"A2 维度错误: {A2.shape}"
    print("[PASS] Stage 2 维度正确")

    # ── 合并因子 ──
    B_merged = torch.cat([B1, B2], dim=1)   # (m, r)
    A_merged = torch.cat([A1, A2], dim=0)   # (r, n)
    assert B_merged.shape == (m, r), f"B_merged 维度: {B_merged.shape}"
    assert A_merged.shape == (r, n), f"A_merged 维度: {A_merged.shape}"
    print("[PASS] 合并维度正确 (r=%d)" % r)

    # ── 验证：H-weighted 残差补偿减小重建误差 ──
    W1 = B1 @ A1
    W_comp = B2 @ A2
    # H-weighted 误差：||(W - W1) @ H^{1/2}||_F vs ||(W - W1 - W_comp) @ H^{1/2}||_F
    err_no_comp = torch.norm((W - W1) @ H_sqrt, 'fro') ** 2
    err_with_comp = torch.norm((W - W1 - W_comp) @ H_sqrt, 'fro') ** 2

    print(f"  ||(W-W1) H^1/2||^2 = {err_no_comp.item():.4f}")
    print(f"  ||(W-W1-W2) H^1/2||^2 = {err_with_comp.item():.4f}")
    assert err_with_comp <= err_no_comp * 1.001, \
        f"残差补偿应减小误差: {err_with_comp.item():.6f} > {err_no_comp.item():.6f}"
    print("[PASS] Stage 2 残差补偿减小 H-weighted 重建误差")

    # ── 验证：H-weighted SVD 补偿优于朴素 SVD 补偿 ──
    W_diff = W - W1
    U_n, S_n, Vt_n = torch.linalg.svd(W_diff, full_matrices=False)
    W_naive = U_n[:, :r2] @ torch.diag(S_n[:r2]) @ Vt_n[:r2, :]
    err_naive = torch.norm((W - W1 - W_naive) @ H_sqrt, 'fro') ** 2

    print(f"  ||(W-W1-W_naive) H^1/2||^2 = {err_naive.item():.4f}")
    assert err_with_comp <= err_naive * 1.001, \
        f"H-weighted 补偿应优于朴素 SVD: {err_with_comp.item():.6f} > {err_naive.item():.6f}"
    print("[PASS] H-weighted 残差补偿优于朴素 SVD 补偿")

    print("\n所有冒烟测试通过！")
    print("=" * 50)


# ==================== 命令行入口 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FR-SVD：Fisher 引导的累积误差感知 SVD + 特征空间残差补偿'
    )

    # 基础参数
    parser.add_argument('--model', type=str, default=None, help='模型路径（HuggingFace ID 或本地路径）')
    parser.add_argument('--ratio', type=float, default=0.2, help='压缩率（0.2 = 去掉 20%% 参数）')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='校准数据集名称')
    parser.add_argument('--nsamples', type=int, default=128, help='校准样本数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--DEV', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--save_path', type=str, default='./result/FR_SVD_v1.2', help='模型保存路径')
    parser.add_argument('--result_dir', type=str, default=None, help='结果目录（覆盖 save_path）')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='校准序列长度')

    # FR-SVD 专有参数
    parser.add_argument('--eps', type=float, default=1e-4, help='H 矩阵阻尼系数 λ_H')
    parser.add_argument('--r1_ratio', type=float, default=0.8, help='Stage 1 秩占比 r₁/r')
    parser.add_argument('--rho', type=float, default=0.95, help='ACES β 收缩因子')
    parser.add_argument('--alpha_min', type=float, default=0.4, help='ACES α 下界')
    parser.add_argument('--alpha_max', type=float, default=0.6, help='ACES α 上界')
    parser.add_argument('--use_fisher', action='store_true', default=False,
                        help='启用创新点A：Psi（输出端Fisher因子）加权')
    parser.add_argument('--max_fisher_samples', type=int, default=32, help='Fisher 估计最大样本数')
    parser.add_argument('--lambda_psi', type=float, default=1e-4, help='Psi 阻尼系数')

    # 功能模式
    parser.add_argument('--eval_only', action='store_true', help='仅运行评估（不压缩）')
    parser.add_argument('--model_path', type=str, default=None, help='--eval_only 模式下的压缩模型路径')
    parser.add_argument('--smoke_test', action='store_true', help='运行冒烟测试并退出')

    args = parser.parse_args()

    # result_dir 优先级高于 save_path
    if args.result_dir:
        args.save_path = args.result_dir

    # 冒烟测试模式：验证数学正确性后退出
    if args.smoke_test:
        smoke_test()
        sys.exit(0)

    if args.model is None and not args.eval_only:
        parser.error("非 --eval_only 模式下必须指定 --model")

    # 压缩率 ratio 表示"去掉多少"，keep_ratio = 1 - ratio 表示"保留多少"
    keep_ratio = 1 - args.ratio

    print("=" * 60)
    print("FR-SVD 配置")
    print("=" * 60)
    print(f"  模型: {args.model}")
    print(f"  压缩率: {args.ratio} (保留 {keep_ratio*100:.1f}%)")
    print(f"  数据集: {args.dataset}, 样本数: {args.nsamples}")
    print(f"  设备: {args.DEV}")
    print(f"  r1_ratio: {args.r1_ratio}")
    print(f"  创新点A (Fisher Psi): {'ON' if args.use_fisher else 'OFF'}")
    print(f"  创新点B (Stage 2): {'ON' if args.r1_ratio < 1.0 else 'OFF'}")
    print(f"  eps={args.eps}")
    print(f"  Alpha: [{args.alpha_min}, {args.alpha_max}], rho={args.rho}")
    print("=" * 60)

    if args.eval_only and args.model_path:
        # 仅评估模式：加载已压缩模型直接评估 PPL
        print("\n仅评估模式...")
        model, tokenizer = get_model_from_local(args.model_path)
        model.eval()
        model = model.float().to(args.DEV)

        print("\n运行困惑度评估...")
        ppl_eval(model, tokenizer, datasets=['wikitext2'],
                 model_seq_len=args.model_seq_len, batch_size=4, device=args.DEV)
    else:
        # 完整压缩流程
        print("\n加载模型...")
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval().half()  # fp16 存储，节省显存

        print("\n加载校准数据...")
        calib_data = get_calib_train_data(
            args.dataset, tokenizer, args.nsamples, seqlen=args.model_seq_len
        )

        print("\n开始 FR-SVD 压缩...")
        stats = fr_svd_compress_layerwise(
            args.model, model, calib_data, keep_ratio, args.DEV,
            eps=args.eps, r1_ratio=args.r1_ratio,
            use_fisher=args.use_fisher, max_fisher_samples=args.max_fisher_samples,
            lambda_psi=args.lambda_psi,
            rho=args.rho, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
            verbose=True
        )

        # 保存压缩模型、tokenizer 及压缩统计信息
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        save_file = os.path.join(
            args.save_path,
            f"{args.model.replace('/', '_').replace('-', '_')}_fr_svd_{keep_ratio}.pt"
        )
        print(f"\n保存压缩模型到: {save_file}")
        model = model.float()  # 保存前转回 float32，方便后续加载评估
        torch.save({'model': model, 'tokenizer': tokenizer, 'stats': stats}, save_file)

        # 释放显存后再评估（Fisher 模式下显存紧张）
        del stats, calib_data
        import gc; gc.collect()
        torch.cuda.empty_cache()

        # 评估压缩后模型的困惑度（PPL）
        print("\n运行困惑度评估...")
        model = model.half().to(args.DEV)  # half 评估节省显存
        ppl_eval(model, tokenizer, datasets=['wikitext2'],
                 model_seq_len=args.model_seq_len, batch_size=4, device=args.DEV)

        print("\n全部完成！")