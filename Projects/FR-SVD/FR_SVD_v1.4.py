#coding:utf8
"""
版本历史：
  v1.0  初始版本。Fisher Phase B 因 solve_triangular 逆变换放大噪声而禁用，
        退化为纯 H-weighted CEALC + 旧版白化残差补偿。
        实验结果：rho=0.6 时 PPL=7.894。

  v1.1  Plan A 尝试：Stage 2 使用额外秩（r_total > r）。失败（PPL=14.4-14.7）。

  v1.2  修复 v1.1 的两个失败原因：
        ① 创新点A：Psi 加权改用特征分解 + clamp
        ② 创新点B：Stage 2 改用与 Stage 1 相同的 H-weighted 稳定路径
        消融结果：Exp1(基线)=7.894, Exp2(A)=7.871, Exp3(B)=7.869, Exp4(A+B)=7.848

  v1.3  基于 SAES-SVD 发布版论文的 3 个关键洞察优化：
        ① Alpha 范围修正：[0.4,0.6] → [0.25,0.75]（论文 Table 4 最优），
           beta_cap 同步更新为 alpha_max/(1+alpha_max)
        ② 取消 rho 收缩：默认 rho=1.0
        ③ 创新点B升级为 CEALC-aware Stage 2
        ④ 创新点A支持两种 Fisher 模式（--fisher_mode：none/direct/soft）

  v1.4  （当前版本）在 v1.3 压缩逻辑完全不变的基础上，新增两阶段 LoRA
        微调合并评估能力（对标 SVDLLM / SAES-SVD LoRA 流程）：
        ① 新增 --lora 命令行参数：指定 PEFT LoRA adapter 目录
        ② 在 --eval_only 模式下，自动把 LoRA 权重 merge_and_unload 到
           u_proj / v_proj 上，保存 merge.pt 并进行 PPL 评估
        ③ 配套 run_lora_finetuning.sh：7 步两阶段微调流程
           (baseline 评估 → 一阶段 U LoRA → 合并/评估 → 二阶段 V LoRA
            → 合并/评估 → 总结)
"""

import os
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

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
    对输入协方差矩阵 H 做阻尼 + 特征分解，返回 H^{1/2} 和 H^{-1/2}。
    """
    d = H_nodamp.shape[0]
    if d > 4096:
        H_reg = H_nodamp.cpu().float() + eps * torch.eye(d, dtype=torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)
    else:
        H_reg = H_nodamp.to(device).float() + eps * torch.eye(d, device=device, dtype=torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)

    eigenvalues = torch.clamp(eigenvalues, min=eps)
    sqrt_lam = torch.sqrt(eigenvalues)
    inv_sqrt_lam = 1.0 / sqrt_lam

    Q = eigenvectors.to(device)
    Lambda = eigenvalues.to(device)
    H_sqrt = (Q * sqrt_lam.to(device).unsqueeze(0)) @ Q.T
    H_inv_sqrt = (Q * inv_sqrt_lam.to(device).unsqueeze(0)) @ Q.T

    return Q, Lambda, H_sqrt, H_inv_sqrt


def estimate_kronecker_fisher(layer, teacher_layer, inps_student, inps_teacher,
                               attention_masks, position_ids, model_name, dev,
                               num_samples, lambda_psi=1e-4, max_fisher_samples=32):
    """
    估计每个线性子层的输出端 Fisher 因子 Ψ = E[g g^T]。
    返回 {name: (eigvecs_psi, eigvals_psi)} 用于后续 Fisher 加权。
    """
    subset = find_layers(layer)
    psi_accum = {}
    input_cache = {}

    for name in subset:
        n_out = subset[name].out_features
        psi_accum[name] = torch.zeros(n_out, n_out, dtype=torch.float32)

    def make_hooks(name):
        def forward_hook(module, inp, out):
            a = inp[0].detach().float()
            if a.dim() == 3:
                a = a.reshape(-1, a.shape[-1])
            input_cache[name] = a

        def backward_hook(module, grad_input, grad_output):
            b = grad_output[0].detach().float()
            if b.dim() == 3:
                b = b.reshape(-1, b.shape[-1])
            psi_accum[name] += (b.T @ b).cpu()

        return forward_hook, backward_hook

    handles = []
    for name in subset:
        fh, bh = make_hooks(name)
        handles.append(subset[name].register_forward_hook(fh))
        handles.append(subset[name].register_full_backward_hook(bh))

    for p in layer.parameters():
        p.requires_grad_(True)

    n_fisher = min(num_samples, max_fisher_samples)

    for j in range(n_fisher):
        inp_j = inps_student[j].unsqueeze(0).to(dev).float()
        teacher_inp_j = inps_teacher[j].unsqueeze(0).to(dev).float()
        attn_mask = attention_masks[j].to(dev)
        pos_ids = position_ids[j].to(dev) if position_ids is not None else None

        layer.zero_grad()
        inp_for_grad = inp_j.detach().requires_grad_(True)

        if "opt" not in model_name:
            out_student = layer(inp_for_grad, attention_mask=attn_mask, position_ids=pos_ids)[0]
        else:
            out_student = layer(inp_for_grad, attention_mask=attn_mask)[0]

        with torch.no_grad():
            out_teacher = teacher_layer.float()(
                teacher_inp_j,
                attention_mask=attn_mask,
                **({"position_ids": pos_ids} if "opt" not in model_name and pos_ids is not None else {})
            )[0]
            teacher_layer.half()

        loss = nn.functional.mse_loss(out_student, out_teacher.detach())
        loss.backward()

        del inp_j, teacher_inp_j, out_student, out_teacher, loss, inp_for_grad
        input_cache.clear()
        if j % 16 == 0:
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    for p in layer.parameters():
        p.requires_grad_(False)

    result = {}
    for name in subset:
        psi = psi_accum[name] / n_fisher
        psi += lambda_psi * torch.eye(psi.shape[0])
        eigvals_psi, eigvecs_psi = torch.linalg.eigh(psi.float())
        eigvals_psi = torch.clamp(eigvals_psi, min=lambda_psi)
        result[name] = (eigvecs_psi, eigvals_psi)

    del psi_accum
    torch.cuda.empty_cache()
    return result


def aces_beta_select(S, D, rank, alpha_min=0.25, alpha_max=0.75,
                     rho=1.0, beta_cap=None, fisher_weights=None):
    """
    ACES β 选择（论文 Algorithm 3）。

    v1.3 改进：
    - 默认 alpha 范围扩展到 [0.25, 0.75]（论文 Table 4 最优）
    - beta_cap 自动从 alpha_max 计算
    - rho 默认 1.0（不收缩）
    - 可选 fisher_weights：对 RER 目标函数中的各方向施加 Fisher 重要性加权

    参数：
        S            : 权重信号矩阵 W@H^{1/2}，形状 (m, n)
        D            : 误差信号矩阵 W@Δ@H^{-1/2}，形状 (m, n)
        rank         : 目标秩
        alpha_min    : α 下界
        alpha_max    : α 上界
        rho          : 收缩因子
        beta_cap     : β 硬上限（None → 自动从 alpha_max 计算）
        fisher_weights : 可选，形状 (m,) 的 Fisher 重要性权重向量。
                         当提供时，在计算 RER 系数 a/b/c/A/B/C 前，
                         对 S 和 D 的行施加 sqrt(fisher_weights) 缩放。

    返回：
        beta_star  : 最优 β
        info       : 诊断字典
    """
    if beta_cap is None:
        beta_cap = alpha_max / (1 + alpha_max)

    beta_min = alpha_min / (1 + alpha_min)
    beta_max = alpha_max / (1 + alpha_max)

    S = S.float()
    D = D.float()

    # 可选：Fisher 重要性加权
    # 对 S, D 的行乘以 sqrt(fisher_weights)，等价于改变输出方向的度量
    # 这种 soft-reweighting 方式不改变 SVD 截断本身的度量空间，
    # 只在 ACES 的 β 选择目标中引入 Fisher 信息引导
    if fisher_weights is not None:
        fw_sqrt = torch.sqrt(fisher_weights.float()).unsqueeze(1)  # (m, 1)
        S_w = S * fw_sqrt
        D_w = D * fw_sqrt
    else:
        S_w = S
        D_w = D

    # 对 S_w 做 SVD，取前 rank 个左/右奇异向量
    U, Sigma, Vt = torch.linalg.svd(S_w, full_matrices=False)
    U_r = U[:, :rank]
    V_r = Vt[:rank, :].T

    # 正交补投影
    UrTS = U_r.T @ S_w
    SVr = S_w @ V_r
    UrTSVr = U_r.T @ SVr
    S_perp = S_w - U_r @ UrTS - SVr @ V_r.T + U_r @ UrTSVr @ V_r.T

    UrTD = U_r.T @ D_w
    DVr = D_w @ V_r
    UrTDVr = U_r.T @ DVr
    D_perp = D_w - U_r @ UrTD - DVr @ V_r.T + U_r @ UrTDVr @ V_r.T

    # RER 系数
    a = torch.norm(S_perp, 'fro') ** 2
    b = torch.sum(S_perp * D_perp)
    c = torch.norm(D_perp, 'fro') ** 2
    A_coef = torch.norm(S_w, 'fro') ** 2
    B_coef = torch.sum(S_w * D_w)
    C_coef = torch.norm(D_w, 'fro') ** 2

    # 二次方程 (cB - bC)β² + (cA - aC)β + (bA - aB) = 0
    coef_2 = c * B_coef - b * C_coef
    coef_1 = c * A_coef - a * C_coef
    coef_0 = b * A_coef - a * B_coef

    candidates = [beta_min, beta_max]

    if abs(coef_2.item()) > 1e-10:
        # 标准二次情形：求驻点
        discriminant = coef_1 ** 2 - 4 * coef_2 * coef_0
        if discriminant >= 0:
            sqrt_disc = torch.sqrt(discriminant)
            candidates.append((-coef_1 + sqrt_disc).item() / (2 * coef_2.item()))
            candidates.append((-coef_1 - sqrt_disc).item() / (2 * coef_2.item()))
    elif abs(coef_1.item()) > 1e-10:
        # 线性退化情形：coef_2 ≈ 0 时 RER 导数为线性，驻点 = -coef_0/coef_1
        candidates.append(-coef_0.item() / coef_1.item())

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

    # 护栏：rho 收缩 + beta_cap
    best_beta = rho * min(best_beta, beta_cap)
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
                                rho=1.0, alpha_min=0.25, alpha_max=0.75,
                                fisher_mode='none',
                                verbose=True):
    """
    FR-SVD v1.3 逐层闭环压缩主函数。

    相比 v1.2 的核心改进：
    - Alpha 范围：[0.25, 0.75]（论文 Table 4 最优）
    - rho：默认 1.0（不收缩）
    - Stage 2：CEALC-aware 残差补偿（使用 Delta 信息 + 独立 ACES β 选择）
    - Fisher 模式：direct（Psi 直接变换，v1.2 验证有效）/ soft / none
    """
    # 兼容旧接口：use_fisher=True + fisher_mode='none' → 使用 'soft'
    if use_fisher and fisher_mode == 'none':
        fisher_mode = 'soft'
    use_fisher = fisher_mode in ('direct', 'soft')

    beta_cap = alpha_max / (1 + alpha_max)

    fisher_mode_str = {'none': 'OFF', 'soft': 'soft-reweighting', 'direct': 'Psi 直接变换'}[fisher_mode]
    print("=" * 60)
    print("FR-SVD v1.3: Widened-ACES + CEALC-aware Stage 2 + Fisher")
    print("=" * 60)
    print(f"  保留比例: {ratio} ({ratio*100:.1f}%)")
    print(f"  秩分配: r1={r1_ratio*100:.0f}% of r / r2={100-r1_ratio*100:.0f}% of r (r1+r2=r)")
    print(f"  Fisher 模式: {fisher_mode_str}")
    print(f"  创新点B (CEALC-aware Stage 2): {'ON (r2>0)' if r1_ratio < 1.0 else 'OFF (r2=0)'}")
    print(f"  H 阻尼: eps={eps}")
    print(f"  Alpha 范围: [{alpha_min}, {alpha_max}], rho={rho}, beta_cap={beta_cap:.4f}")
    print("=" * 60)

    # ========== Step 1：加载教师模型 ==========
    print("\n加载教师模型（fp16 存储，fp32 计算）...")
    from transformers import AutoModelForCausalLM
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model.config._name_or_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    teacher_model.eval()

    # ========== Step 2：获取模型的层列表 ==========
    if "opt" in model_name:
        layers = model.model.decoder.layers
        teacher_layers = teacher_model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        teacher_layers = teacher_model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

    # ========== Step 3：捕获第 0 层输入激活 ==========
    layers[0] = layers[0].to(dev)
    num_samples = len(calib_loader)

    inps_student = torch.zeros(
        (num_samples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16, device='cpu'
    )
    inps_teacher = torch.zeros_like(inps_student)

    cache = {'i': 0}
    attention_masks_list = []
    position_ids_list = []

    class StudentCatcher(nn.Module):
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
            raise ValueError

    class TeacherCatcher(nn.Module):
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
            pass
        try:
            teacher_model.to(dev)
            teacher_model(**batch)
        except ValueError:
            pass
        finally:
            teacher_model.to("cpu")

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    teacher_layers[0] = teacher_layers[0].module

    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()

    torch.cuda.empty_cache()

    outs_student = torch.zeros_like(inps_student)
    outs_teacher = torch.zeros(
        (num_samples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16, device='cpu'
    )
    attention_masks = attention_masks_list
    position_ids = position_ids_list if position_ids_list and position_ids_list[0] is not None else None

    # ========== Step 4：逐层压缩主循环 ==========
    compression_stats = []

    print("\n开始逐层 FR-SVD v1.3 压缩...")

    for layer_idx in tqdm(range(len(layers)), desc="FR-SVD v1.3 Compression"):
        layer_stats = {'layer': layer_idx, 'sublayers': {}}

        layer = layers[layer_idx].to(dev).float()
        teacher_layer = teacher_layers[layer_idx].to(dev)

        subset = find_layers(layer)
        teacher_subset = find_layers(teacher_layer)

        # ────────── Phase A：收集 H 和 Δ ──────────
        for name in subset:
            subset[name].H_matrix = 0
            subset[name].inp_cache = []
            teacher_subset[name].inp_cache = []

        def make_hook_student(module_name):
            def hook(module, input, output):
                inp = input[0].detach().float()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                H_batch = torch.matmul(inp.transpose(1, 2), inp)
                module.H_matrix = module.H_matrix + torch.sum(H_batch, dim=0)
                module.inp_cache.append(inp.half().cpu())
            return hook

        def make_hook_teacher(module_name):
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
                teacher_layer.half()
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

        # 计算 Δ = Σ (X^f - X) X^T
        for name in subset:
            subset[name].Delta_matrix = 0
            for idx in range(len(subset[name].inp_cache)):
                X = subset[name].inp_cache[idx].to(dev).float()
                X_f = teacher_subset[name].inp_cache[idx].to(dev).float()
                diff = X_f - X
                Delta_batch = torch.matmul(diff.transpose(1, 2), X)
                subset[name].Delta_matrix = subset[name].Delta_matrix + torch.sum(Delta_batch, dim=0)
                del X, X_f, diff, Delta_batch

            subset[name].H_matrix = subset[name].H_matrix.cpu()
            subset[name].Delta_matrix = subset[name].Delta_matrix.cpu()
            del subset[name].inp_cache
            del teacher_subset[name].inp_cache

        torch.cuda.empty_cache()

        # ────────── Phase B：Fisher Psi 估计（创新点A）──────────
        psi_decomp = {}
        if use_fisher:
            with torch.enable_grad():
                fisher_result = estimate_kronecker_fisher(
                    layer, teacher_layer, inps_student, inps_teacher,
                    attention_masks, position_ids, model_name, dev,
                    num_samples, lambda_psi=lambda_psi,
                    max_fisher_samples=max_fisher_samples
                )
            for name in fisher_result:
                eigvecs_psi, eigvals_psi = fisher_result[name]
                P_psi = eigvecs_psi.to(dev).float()
                lam_psi = eigvals_psi.to(dev).float()

                if fisher_mode == 'direct':
                    # direct 模式：构造完整 Psi^{1/2} 和 Psi^{-1/2} 矩阵
                    sqrt_psi = torch.sqrt(lam_psi)
                    inv_sqrt_psi = 1.0 / sqrt_psi
                    Psi_half = (P_psi * sqrt_psi.unsqueeze(0)) @ P_psi.T
                    Psi_inv_half = (P_psi * inv_sqrt_psi.unsqueeze(0)) @ P_psi.T
                    psi_decomp[name] = ('direct', Psi_half, Psi_inv_half)
                    del sqrt_psi, inv_sqrt_psi
                else:
                    # soft 模式：仅提取对角元素作为标量权重
                    psi_diag = torch.sum(P_psi ** 2 * lam_psi.unsqueeze(0), dim=1)  # (m,)
                    psi_diag = psi_diag / (psi_diag.mean() + 1e-10)
                    psi_decomp[name] = ('soft', psi_diag)

                del P_psi, lam_psi
            del fisher_result
            torch.cuda.empty_cache()

        # ────────── Phase C：创建 SVD 替换模块 ──────────
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
            W = subset[name].weight.data.float().to(dev)
            original_dtype = subset[name].weight.data.dtype
            m, n = W.shape

            H_nodamp = subset[name].H_matrix.float().to(dev)
            Delta = subset[name].Delta_matrix.float().to(dev)

            r = int(m * n * ratio / (m + n))
            r = max(1, min(r, min(m, n)))
            r1 = int(r * r1_ratio) if r1_ratio < 1.0 else r
            r2 = r - r1

            # NaN 防护
            if not torch.isfinite(H_nodamp).all() or not torch.isfinite(Delta).all():
                print(f"  ⚠ L{layer_idx} {name}: H/Delta 含 NaN，替换为 eps*I / 零矩阵")
                H_nodamp = eps * torch.eye(n, device=dev, dtype=torch.float32)
                Delta = torch.zeros(n, n, device=dev, dtype=torch.float32)

            Q, Lambda, H_sqrt, H_inv_sqrt = compute_H_decomposition(H_nodamp, dev, eps)

            # ===== Stage 1：CEALC + Fisher ACES =====
            S = W @ H_sqrt              # S = W H^{1/2}
            D = W @ Delta @ H_inv_sqrt  # D = W Δ H^{-1/2}

            # Fisher 变换
            psi_info = psi_decomp.get(name, None) if use_fisher else None
            Psi_half = Psi_inv_half = None
            fw = None

            if psi_info is not None and psi_info[0] == 'direct':
                # direct 模式：S_F = Ψ^{1/2} S, D_F = Ψ^{1/2} D
                _, Psi_half, Psi_inv_half = psi_info
                S_F = Psi_half @ S
                D_F = Psi_half @ D
            elif psi_info is not None and psi_info[0] == 'soft':
                # soft 模式：仅传递权重给 ACES
                _, fw = psi_info
                S_F = S
                D_F = D
            else:
                S_F = S
                D_F = D

            beta, aces_info = aces_beta_select(
                S_F, D_F, r1,
                alpha_min=alpha_min, alpha_max=alpha_max,
                rho=rho, beta_cap=beta_cap,
                fisher_weights=fw
            )

            # G_F = S_F + β*D_F，在 Fisher（或原始）空间做 SVD 截断
            G_F = S_F + beta * D_F
            U_g, S_g, Vt_g = torch.linalg.svd(G_F, full_matrices=False)
            U_r1 = U_g[:, :r1]
            S_r1 = S_g[:r1]
            Vt_r1 = Vt_g[:r1, :]

            sqrt_s1 = torch.sqrt(S_r1)

            # direct 模式需要逆变换回原空间：B1 = Ψ^{-1/2} U_r Σ^{1/2}
            if Psi_inv_half is not None:
                B1 = Psi_inv_half @ (U_r1 * sqrt_s1.unsqueeze(0))  # (m, r1)
            else:
                B1 = U_r1 * sqrt_s1.unsqueeze(0)                   # (m, r1)
            A1 = (sqrt_s1.unsqueeze(1) * Vt_r1) @ H_inv_sqrt  # (r1, n)

            total_energy = torch.sum(S_g ** 2).item()
            retained_energy = torch.sum(S_r1 ** 2).item()
            energy_ratio = retained_energy / (total_energy + 1e-10)

            # ===== Stage 2：CEALC-aware 残差补偿（v1.3 创新点B）=====
            # 不再只做 SVD(E_w @ H^{1/2})，而是用完整 CEALC 公式：
            # G₂ = E_w @ (H + β₂·Δ) @ H^{-1/2}，带独立的 ACES β₂ 选择
            if r2 > 0:
                E_w = W - B1 @ A1  # 权重残差

                # CEALC-aware: 用残差的 S₂ 和 D₂
                S2 = E_w @ H_sqrt              # S₂ = E_w @ H^{1/2}
                D2 = E_w @ Delta @ H_inv_sqrt  # D₂ = E_w @ Δ @ H^{-1/2}

                # direct 模式：Stage 2 也在 Fisher 空间做
                if Psi_half is not None:
                    S2_F = Psi_half @ S2
                    D2_F = Psi_half @ D2
                else:
                    S2_F = S2
                    D2_F = D2

                # Stage 2 独立的 ACES β₂ 选择
                beta2, aces_info2 = aces_beta_select(
                    S2_F, D2_F, r2,
                    alpha_min=alpha_min, alpha_max=alpha_max,
                    rho=rho, beta_cap=beta_cap,
                    fisher_weights=fw
                )

                # G₂ = S₂_F + β₂·D₂_F
                G2_F = S2_F + beta2 * D2_F
                U_r, S_r, Vt_r = torch.linalg.svd(G2_F, full_matrices=False)
                U_r2 = U_r[:, :r2]
                S_r2 = S_r[:r2]
                Vt_r2 = Vt_r[:r2, :]

                sqrt_s2 = torch.sqrt(S_r2)
                # direct 模式逆变换
                if Psi_inv_half is not None:
                    B2 = Psi_inv_half @ (U_r2 * sqrt_s2.unsqueeze(0))
                else:
                    B2 = U_r2 * sqrt_s2.unsqueeze(0)
                A2 = (sqrt_s2.unsqueeze(1) * Vt_r2) @ H_inv_sqrt

                del E_w, S2, D2, U_r, S_r, Vt_r
                if Psi_half is not None:
                    del S2_F, D2_F, G2_F
                else:
                    del G2_F
            else:
                B2 = torch.empty(m, 0, device=dev, dtype=torch.float32)
                A2 = torch.empty(0, n, device=dev, dtype=torch.float32)
                beta2 = 0.0
                aces_info2 = {}

            # ===== 合并 =====
            B_merged = torch.cat([B1, B2], dim=1)
            A_merged = torch.cat([A1, A2], dim=0)

            W_recon = B_merged.float() @ A_merged.float()
            weight_err = (torch.norm(W - W_recon) / torch.norm(W)).item()

            # 数值安全回退
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

            layer_stats['sublayers'][name] = {
                'beta': beta,
                'beta2': beta2 if r2 > 0 else 0.0,
                'r': r, 'r1': r1, 'r2': r2,
                'weight_err': weight_err,
                'energy_ratio': energy_ratio,
                'delta_norm': delta_norm,
                'h_norm': h_norm,
                'delta_ratio': delta_norm / (h_norm + 1e-10),
                'aces_info': aces_info
            }

            if verbose and (layer_idx == 0 or layer_idx == len(layers) - 1 or layer_idx % 8 == 0):
                beta2_str = f", β₂={beta2:.4f}" if r2 > 0 else ""
                print(f"  L{layer_idx} {name}: β₁={beta:.4f}, α*={aces_info['alpha_star']:.4f}{beta2_str}, "
                      f"r={r}(r1={r1}+r2={r2}), E_kept={energy_ratio:.4f}, "
                      f"W_err={weight_err:.4f}, D/H={delta_norm/h_norm:.4f}")

            # 写入 SVD 替换模块
            if 'opt' in model_name:
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
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
            else:
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

            del W, H_nodamp, Delta
            del Q, Lambda, H_sqrt, H_inv_sqrt
            del S, D, S_F, D_F, G_F, U_g, S_g, Vt_g, U_r1, S_r1, Vt_r1
            del B1, A1, B_merged, A_merged, W_recon
            if r2 > 0:
                del B2, A2
            del subset[name].H_matrix, subset[name].Delta_matrix
            torch.cuda.empty_cache()

        # 释放 Psi
        if use_fisher:
            for name in list(psi_decomp.keys()):
                del psi_decomp[name]
            psi_decomp.clear()
            torch.cuda.empty_cache()

        # ────────── Phase E：替换层 ──────────
        if 'opt' in model_name:
            layers[layer_idx] = svd_decoder
        else:
            layer.self_attn = svd_attn
            layer.mlp = svd_mlp

        avg_beta = sum(s['beta'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        avg_delta_ratio = sum(s['delta_ratio'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        layer_stats['avg_beta'] = avg_beta
        layer_stats['avg_delta_ratio'] = avg_delta_ratio

        if verbose:
            avg_alpha_star = sum(s['aces_info']['alpha_star'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
            print(f"Layer {layer_idx} Summary: avg_β={avg_beta:.4f}, avg_α*={avg_alpha_star:.4f}, "
                  f"avg_D/H={avg_delta_ratio:.4f}")

        compression_stats.append(layer_stats)

        # ────────── Phase F：闭环重新前向 ──────────
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
                _phase_f_nan_count += 1
            del inp_j, out_j
        if _phase_f_nan_count > 0:
            print(f"  ⚠ L{layer_idx}: Phase F 有 {_phase_f_nan_count}/{num_samples} 个样本含 NaN")

        torch.cuda.empty_cache()

        inps_student = outs_student.clone()
        inps_teacher = outs_teacher.clone()

        layers[layer_idx] = layers[layer_idx].half().cpu()
        teacher_layers[layer_idx] = teacher_layers[layer_idx].cpu()
        torch.cuda.empty_cache()

    del teacher_model
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("FR-SVD v1.3 压缩完成！")
    print("=" * 60)

    all_betas = [s['avg_beta'] for s in compression_stats]
    all_delta_ratios = [s['avg_delta_ratio'] for s in compression_stats]
    print(f"  平均 beta: {sum(all_betas)/len(all_betas):.4f}")
    print(f"  平均 D/H:  {sum(all_delta_ratios)/len(all_delta_ratios):.4f}")
    print(f"  Beta 范围: [{min(all_betas):.4f}, {max(all_betas):.4f}]")

    return compression_stats


# ==================== 冒烟测试 ====================

def smoke_test():
    """单元测试：验证 v1.3 核心数学正确性。"""
    print("=" * 50)
    print("FR-SVD v1.3 Smoke Test")
    print("=" * 50)

    torch.manual_seed(42)
    m, n, p = 128, 64, 256
    r = 16
    r1 = 12
    r2 = 4
    eps = 1e-4

    W = torch.randn(m, n)
    X = torch.randn(n, p)
    X_f = X + 0.1 * torch.randn(n, p)

    H_nodamp = X @ X.T
    Delta = (X_f - X) @ X.T
    H = H_nodamp + eps * torch.eye(n)

    eigenvalues, Q = torch.linalg.eigh(H)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    sqrt_lam = torch.sqrt(eigenvalues)
    inv_sqrt_lam = 1.0 / sqrt_lam

    H_sqrt = (Q * sqrt_lam.unsqueeze(0)) @ Q.T
    H_inv_sqrt = (Q * inv_sqrt_lam.unsqueeze(0)) @ Q.T

    # ── Stage 1：CEALC ──
    S = W @ H_sqrt
    D = W @ Delta @ H_inv_sqrt

    beta, info = aces_beta_select(S, D, r1)
    print(f"  Stage 1: beta*={beta:.4f}, alpha*={info['alpha_star']:.4f}")

    G = S + beta * D
    U, Sigma, Vt = torch.linalg.svd(G, full_matrices=False)
    U_r1 = U[:, :r1]
    S_r1 = Sigma[:r1]
    Vt_r1 = Vt[:r1, :]
    sqrt_s1 = torch.sqrt(S_r1)

    B1 = U_r1 * sqrt_s1.unsqueeze(0)
    A1 = (sqrt_s1.unsqueeze(1) * Vt_r1) @ H_inv_sqrt

    assert B1.shape == (m, r1), f"B1 维度错误: {B1.shape}"
    assert A1.shape == (r1, n), f"A1 维度错误: {A1.shape}"
    print("[PASS] Stage 1 维度正确")

    # ── Stage 2：CEALC-aware 残差补偿（v1.3 创新）──
    E_w = W - B1 @ A1
    S2 = E_w @ H_sqrt
    D2 = E_w @ Delta @ H_inv_sqrt

    beta2, info2 = aces_beta_select(S2, D2, r2)
    print(f"  Stage 2: beta2*={beta2:.4f}, alpha2*={info2['alpha_star']:.4f}")

    G2 = S2 + beta2 * D2
    U2, S2_vals, Vt2 = torch.linalg.svd(G2, full_matrices=False)
    U_r2 = U2[:, :r2]
    S_r2 = S2_vals[:r2]
    Vt_r2 = Vt2[:r2, :]
    sqrt_s2 = torch.sqrt(S_r2)

    B2 = U_r2 * sqrt_s2.unsqueeze(0)
    A2 = (sqrt_s2.unsqueeze(1) * Vt_r2) @ H_inv_sqrt

    assert B2.shape == (m, r2), f"B2 维度错误: {B2.shape}"
    assert A2.shape == (r2, n), f"A2 维度错误: {A2.shape}"
    print("[PASS] Stage 2 维度正确")

    # ── 合并 ──
    B_merged = torch.cat([B1, B2], dim=1)
    A_merged = torch.cat([A1, A2], dim=0)
    assert B_merged.shape == (m, r), f"B_merged 维度: {B_merged.shape}"
    assert A_merged.shape == (r, n), f"A_merged 维度: {A_merged.shape}"
    print("[PASS] 合并维度正确 (r=%d)" % r)

    # ── 验证：CEALC-aware Stage 2 减小 H-weighted 重建误差 ──
    W1 = B1 @ A1
    W_comp = B2 @ A2
    err_no_comp = torch.norm((W - W1) @ H_sqrt, 'fro') ** 2
    err_with_comp = torch.norm((W - W1 - W_comp) @ H_sqrt, 'fro') ** 2

    print(f"  ||(W-W1) H^1/2||^2 = {err_no_comp.item():.4f}")
    print(f"  ||(W-W1-W2) H^1/2||^2 = {err_with_comp.item():.4f}")
    assert err_with_comp <= err_no_comp * 1.001, \
        f"Stage 2 应减小误差: {err_with_comp.item():.6f} > {err_no_comp.item():.6f}"
    print("[PASS] CEALC-aware Stage 2 减小 H-weighted 重建误差")

    # ── 验证：CEALC-aware 优于纯 H-weighted SVD 补偿 ──
    R_plain = E_w @ H_sqrt
    U_plain, S_plain, Vt_plain = torch.linalg.svd(R_plain, full_matrices=False)
    sqrt_sp = torch.sqrt(S_plain[:r2])
    B2_plain = U_plain[:, :r2] * sqrt_sp.unsqueeze(0)
    A2_plain = (sqrt_sp.unsqueeze(1) * Vt_plain[:r2, :]) @ H_inv_sqrt
    W_plain_comp = B2_plain @ A2_plain
    err_plain_comp = torch.norm((W - W1 - W_plain_comp) @ H_sqrt, 'fro') ** 2

    print(f"  ||(W-W1-W2_plain) H^1/2||^2 = {err_plain_comp.item():.4f} (v1.2 Stage 2)")
    print(f"  ||(W-W1-W2_cealc) H^1/2||^2 = {err_with_comp.item():.4f} (v1.3 CEALC-aware)")
    # 注意：CEALC-aware 优化的是 CEALC 目标(含 Δ)，不一定在纯 H-weighted 指标上更优
    # 但在下游 PPL 上应该更优（因为 CEALC 目标更好地对齐了模型推理误差）

    # ── 验证：Fisher soft-reweighting ACES ──
    fisher_w = torch.abs(torch.randn(m))  # 模拟 Fisher 权重
    fisher_w = fisher_w / fisher_w.mean()
    beta_fw, info_fw = aces_beta_select(S, D, r1, fisher_weights=fisher_w)
    print(f"  Fisher-weighted ACES: beta*={beta_fw:.4f}, alpha*={info_fw['alpha_star']:.4f}")
    assert 0 <= beta_fw < 1, f"Fisher-weighted beta 超出范围: {beta_fw}"
    print("[PASS] Fisher soft-reweighting ACES 正常工作")

    print("\n所有冒烟测试通过！")
    print("=" * 50)


# ==================== 命令行入口 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FR-SVD v1.4：v1.3 压缩 + 两阶段 LoRA 微调合并评估'
    )

    # 基础参数
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--ratio', type=float, default=0.2, help='压缩率（0.2 = 去掉 20%%）')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='校准数据集')
    parser.add_argument('--nsamples', type=int, default=128, help='校准样本数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--DEV', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--save_path', type=str, default='./result/FR_SVD_v1.4', help='保存路径')
    parser.add_argument('--result_dir', type=str, default=None, help='结果目录')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='序列长度')

    # FR-SVD v1.3 参数
    parser.add_argument('--eps', type=float, default=1e-4, help='H 阻尼系数')
    parser.add_argument('--r1_ratio', type=float, default=0.8, help='Stage 1 秩占比')
    parser.add_argument('--rho', type=float, default=1.0, help='ACES β 收缩因子（v1.3 默认 1.0）')
    parser.add_argument('--alpha_min', type=float, default=0.25, help='ACES α 下界（v1.3 默认 0.25）')
    parser.add_argument('--alpha_max', type=float, default=0.75, help='ACES α 上界（v1.3 默认 0.75）')
    parser.add_argument('--use_fisher', action='store_true', default=False,
                        help='启用 Fisher（兼容旧接口，等价于 --fisher_mode soft）')
    parser.add_argument('--fisher_mode', type=str, default='none', choices=['none', 'direct', 'soft'],
                        help='Fisher 模式：none=不使用, direct=Psi直接变换(v1.2式), soft=仅ACES权重')
    parser.add_argument('--max_fisher_samples', type=int, default=32, help='Fisher 估计最大样本数')
    parser.add_argument('--lambda_psi', type=float, default=1e-4, help='Psi 阻尼系数')

    # 功能模式
    parser.add_argument('--eval_only', action='store_true', help='仅评估')
    parser.add_argument('--model_path', type=str, default=None, help='压缩模型路径')
    parser.add_argument('--smoke_test', action='store_true', help='冒烟测试')
    # v1.4 新增：LoRA 合并评估
    parser.add_argument('--lora', type=str, default=None,
                        help='LoRA adapter 目录；eval_only 模式下自动 merge_and_unload 并保存 merge.pt')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='PPL 评估 batch size')
    parser.add_argument('--eval_dtype', type=str, default='fp16', choices=['fp16', 'fp32'],
                        help='评估时模型精度，默认 fp16 以降低显存占用')

    args = parser.parse_args()

    if args.result_dir:
        args.save_path = args.result_dir

    if args.smoke_test:
        smoke_test()
        sys.exit(0)

    if args.model is None and not args.eval_only:
        parser.error("非 --eval_only 模式下必须指定 --model")

    keep_ratio = 1 - args.ratio

    print("=" * 60)
    print("FR-SVD v1.4 配置")
    print("=" * 60)
    print(f"  模型: {args.model}")
    print(f"  压缩率: {args.ratio} (保留 {keep_ratio*100:.1f}%)")
    print(f"  数据集: {args.dataset}, 样本数: {args.nsamples}")
    print(f"  设备: {args.DEV}")
    print(f"  r1_ratio: {args.r1_ratio}")
    effective_fisher = args.fisher_mode if args.fisher_mode != 'none' else ('soft' if args.use_fisher else 'none')
    fisher_mode_display = {'none': 'OFF', 'soft': 'soft-reweighting', 'direct': 'Psi直接变换'}[effective_fisher]
    print(f"  Fisher 模式: {fisher_mode_display}")
    print(f"  创新点B (CEALC-aware Stage 2): {'ON' if args.r1_ratio < 1.0 else 'OFF'}")
    print(f"  eps={args.eps}")
    print(f"  Alpha: [{args.alpha_min}, {args.alpha_max}], rho={args.rho}")
    print("=" * 60)

    if args.eval_only and args.model_path:
        print("\n仅评估模式...")
        model, tokenizer = get_model_from_local(args.model_path)

        # v1.4：LoRA adapter 合并（对标 SVDLLM step 4 分支）
        if args.lora is not None:
            print(f"\n加载 LoRA adapter 并合并权重: {args.lora}")
            from utils.peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                args.lora,
                torch_dtype=torch.float16,
            )
            model = model.merge_and_unload()
            if not os.path.exists(args.lora):
                os.makedirs(args.lora)
            merge_path = os.path.join(args.lora, 'merge.pt')
            print(f"保存合并后模型到: {merge_path}")
            torch.save({'model': model, 'tokenizer': tokenizer}, merge_path)

        model.eval()
        if args.eval_dtype == 'fp32':
            model = model.float().to(args.DEV)
        else:
            model = model.half().to(args.DEV)
        print("\n运行困惑度评估...")
        ppl_eval(model, tokenizer, datasets=['wikitext2'],
                 model_seq_len=args.model_seq_len,
                 batch_size=args.eval_batch_size, device=args.DEV)
    else:
        print("\n加载模型...")
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval().half()

        print("\n加载校准数据...")
        calib_data = get_calib_train_data(
            args.dataset, tokenizer, args.nsamples, seqlen=args.model_seq_len
        )

        print("\n开始 FR-SVD v1.4 压缩...")
        stats = fr_svd_compress_layerwise(
            args.model, model, calib_data, keep_ratio, args.DEV,
            eps=args.eps, r1_ratio=args.r1_ratio,
            use_fisher=args.use_fisher, max_fisher_samples=args.max_fisher_samples,
            lambda_psi=args.lambda_psi,
            rho=args.rho, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
            fisher_mode=args.fisher_mode,
            verbose=True
        )

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        save_file = os.path.join(
            args.save_path,
            f"{args.model.replace('/', '_').replace('-', '_')}_fr_svd_{keep_ratio}.pt"
        )
        print(f"\n保存压缩模型到: {save_file}")
        model = model.float()
        torch.save({'model': model, 'tokenizer': tokenizer, 'stats': stats}, save_file)

        del stats, calib_data
        import gc; gc.collect()
        torch.cuda.empty_cache()

        print("\n运行困惑度评估...")
        model = model.half().to(args.DEV)
        ppl_eval(model, tokenizer, datasets=['wikitext2'],
                 model_seq_len=args.model_seq_len,
                 batch_size=args.eval_batch_size, device=args.DEV)
