#coding:utf8
"""
版本历史：
  v1.0-v1.3  见 v1.4 注释。
  v1.4       v1.3 压缩 + 两阶段 LoRA 微调合并评估。

  v1.5  在 v1.4 基础上扩展多模型架构支持：
        ① 采用模型无关的 FactorizedLinear：以原位替换方式把任意 nn.Linear
           替换为 u_proj·v_proj 两层级联，不依赖 SVD_LlamaAttention 等
           硬编码组件，保留原模型的 Attention / MLP 前向逻辑。
        ② 适配模型：
           - LLaMA-7B / Llama-2-7b / Llama-13b / Llama-30b（标准 MHA）
           - Qwen2.5-7B（GQA + q/k/v bias）
           - Qwen3.5-35B-A3B（混合注意力 + MoE，256 experts）
        ③ Tokenizer：非 OPT/Mistral/Qwen 用 LlamaTokenizer，其余用 AutoTokenizer。
        ④ 新增 --skip_moe_experts：MoE 模型可选择跳过 experts，仅压缩共享层。
        ⑤ LoRA target 子模块名：self_attn.<name>.u_proj / .v_proj。

  v1.6  工程稳定版本：
        ① fisher_mode=none 且关闭 innovation2 时，退化为 SAES-SVD 单阶段
           CEALC/ACES。
        ② innovation1 使用保守 Fisher-guided CEALC 候选：Fisher 信息进入实际
           SVD 目标空间，但只有校准目标优于 baseline 时才接受。
        ③ innovation2 默认使用稳定 CEALC-aware residual Stage2；论文版
           eigenspace residual 保留为实验模式。
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# ==================== 模型无关的低秩线性层 ====================

class FactorizedLinear(nn.Module):
    """
    通用低秩分解 Linear 层：y = u_proj(v_proj(x)) (+ bias)
    - v_proj: (in_features → rank), 无 bias
    - u_proj: (rank → out_features), 可选 bias（继承自原 Linear）
    原位替换任意 nn.Linear 而不触动外层 Attention/MLP 前向逻辑，
    天然支持 LLaMA / Qwen2 (GQA+bias) / Qwen3 MoE 等各种架构。
    """
    def __init__(self, in_features, out_features, rank, bias=False,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.v_proj = nn.Linear(in_features, rank, bias=False,
                                 dtype=dtype, device=device)
        self.u_proj = nn.Linear(rank, out_features, bias=bias,
                                 dtype=dtype, device=device)

    def forward(self, x):
        return self.u_proj(self.v_proj(x))


# ==================== 模型适配器 ====================

def _detect_arch_family(model):
    """从模型配置推断架构家族。"""
    cfg = model.config
    mt = (getattr(cfg, 'model_type', '') or '').lower()
    arch = ''
    if getattr(cfg, 'architectures', None):
        arch = (cfg.architectures[0] or '').lower()
    key = f"{mt}|{arch}|{type(model).__name__.lower()}"

    if 'opt' in key:
        return 'opt'
    if 'mistral' in key:
        return 'mistral'
    if 'qwen3' in key and ('moe' in key or 'a3b' in key):
        return 'qwen3_moe'
    if 'qwen' in key:
        return 'qwen'
    if 'llama' in key or 'vicuna' in key:
        return 'llama'
    return 'llama'  # 默认按 LLaMA 结构处理


def get_model_access(model, arch_family):
    """
    统一获取 model.layers / 嵌入 / 归一化等入口。
    OPT 与 LLaMA 家族结构不同，其他（LLaMA/Qwen2/Qwen3）都走 model.model.layers。
    """
    if arch_family == 'opt':
        dec = model.model.decoder
        return {
            'layers': dec.layers,
            'to_dev': [
                ('model.model.decoder.embed_tokens', dec.embed_tokens),
                ('model.model.decoder.final_layer_norm', dec.final_layer_norm),
                ('model.model.decoder.embed_positions', dec.embed_positions),
            ],
        }
    base = model.model
    to_dev = [
        ('model.model.embed_tokens', base.embed_tokens),
        ('model.model.norm', base.norm),
    ]
    # Qwen3 MoE 有 rotary_emb 等额外子模块，如果存在也搬到设备上
    if hasattr(base, 'rotary_emb') and base.rotary_emb is not None:
        to_dev.append(('model.model.rotary_emb', base.rotary_emb))
    return {
        'layers': base.layers,
        'to_dev': to_dev,
    }


def filter_linears_for_compression(linears, skip_moe_experts=False, min_dim=64):
    """
    过滤不适合压缩的 Linear：
    - 维度太小（min(in, out) < min_dim）：压缩后秩过小，收益低甚至劣化。
      典型如 Qwen3 MoE 的 `mlp.shared_expert_gate` (out=1)、
      `linear_attn.in_proj_a/b` (out=32)、MoE router (out=num_experts)。
    - skip_moe_experts=True 时跳过所有 `experts.<i>.*` 子模块。
    """
    filtered = {}
    for name, mod in linears.items():
        if min(mod.in_features, mod.out_features) < min_dim:
            continue
        if skip_moe_experts and 'experts.' in name:
            continue
        filtered[name] = mod
    return filtered


def get_parent_and_attr(root, dotted_name):
    """根据 dotted_name（如 'self_attn.q_proj'）返回 (parent_module, final_attr_name)。"""
    parts = dotted_name.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def load_model_v15(model_id, seq_len=2048):
    """
    v1.5 的 HuggingFace 模型加载器：按模型名自动选择正确的 tokenizer。
    - 有 tokenizer.json（Qwen / Mistral / OPT 等）：AutoTokenizer
    - 只有 tokenizer.model（旧 LLaMA-7B / LLaMA-13b / LLaMA-30b / Llama-2）：
      LlamaTokenizer（transformers v5 的 AutoTokenizer 加载纯 sentencepiece
      文件时存在递归 bug，必须显式指定）
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

    # 检测是否存在 tokenizer.json；没有则必须用 LlamaTokenizer 避免 transformers v5 递归 bug
    has_tok_json = os.path.isfile(os.path.join(model_id, 'tokenizer.json'))
    force_llama_tok = (not has_tok_json)

    tok_cls = LlamaTokenizer if force_llama_tok else AutoTokenizer
    print(f"  Tokenizer: {tok_cls.__name__} (tokenizer.json={'found' if has_tok_json else 'not found'})")
    # use_fast=False：transformers v5 的 fast tokenizer 在纯 sentencepiece 模型上
    # 存在 unk_token_id 递归 bug，slow tokenizer 无此问题
    tok_kwargs = {"use_fast": False} if force_llama_tok else {}
    try:
        tokenizer = tok_cls.from_pretrained(model_id, trust_remote_code=True, **tok_kwargs)
    except Exception as e:
        print(f"  {tok_cls.__name__} 加载失败 ({e})，回退到 AutoTokenizer use_fast=False")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float16,
        trust_remote_code=True, cache_dir=None
    )
    model.seqlen = seq_len
    return model, tokenizer


def replace_linear_with_factorized(root, dotted_name, B, A, bias_data,
                                    device, dtype=torch.float32):
    """
    用 FactorizedLinear 原位替换 root 下 dotted_name 处的 nn.Linear。
    B: (m, r), A: (r, n), bias_data: (m,) 或 None
    v_proj.weight ← A （形状 (rank=r, in=n)）
    u_proj.weight ← B （形状 (out=m, rank=r)）
    """
    parent, attr = get_parent_and_attr(root, dotted_name)
    old_linear = getattr(parent, attr)
    in_features = old_linear.in_features
    out_features = old_linear.out_features
    rank = A.shape[0]
    has_bias = (bias_data is not None)

    new_mod = FactorizedLinear(
        in_features=in_features, out_features=out_features, rank=rank,
        bias=has_bias, dtype=dtype, device=device
    )
    new_mod.v_proj.weight.data.copy_(A.to(dtype))
    new_mod.u_proj.weight.data.copy_(B.to(dtype))
    if has_bias:
        new_mod.u_proj.bias.data.copy_(bias_data.to(dtype))

    setattr(parent, attr, new_mod)
    return new_mod


# ==================== 辅助函数 ====================

def compute_H_decomposition(H_nodamp, device, eps=1e-4):
    """
    对输入协方差矩阵 H 做阻尼 + 特征分解，返回 H^{1/2} 和 H^{-1/2}。
    d <= 4096 在 GPU 上做 eigh；d > 4096 在 CPU 上做（eigh 工作空间 ~O(d^2)，
    大矩阵在共享 GPU 上易 OOM），结果搬回 GPU。
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


def truncated_svd_factorize(W, rank):
    """
    纯截断 SVD 基线分解，用作数值安全护栏。
    当 FR-SVD/CEALC 结果在某些层上虽然“有限”但重建误差明显劣于
    plain truncated SVD 时，优先采用后者，避免保存语义已崩坏的层。
    """
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    sqrt_s = torch.sqrt(torch.clamp(S[:rank], min=0))
    B = U[:, :rank] * sqrt_s.unsqueeze(0)
    A = sqrt_s.unsqueeze(1) * Vt[:rank, :]
    W_recon = B @ A
    weight_err = (torch.norm(W - W_recon) / torch.norm(W)).item()
    return B, A, weight_err


def estimate_kronecker_fisher(layer, teacher_layer, inps_student, inps_teacher,
                               attention_masks, position_ids, model_name, dev,
                               num_samples, lambda_psi=1e-4, max_fisher_samples=32,
                               position_embeddings=None, fisher_mode='guide',
                               max_fisher_dim=4096):
    """
    估计每个线性子层的输出端 Fisher 因子 Psi。

    v1.6 只在 direct 小矩阵调试路径保留完整 Psi；默认 guide 路径只累积
    diag(Psi)，避免 LLaMA/Qwen MLP 大矩阵上的 O(m^2) 显存/内存开销。
    """
    subset = find_layers(layer)
    psi_diag_accum = {}
    psi_full_accum = {}
    token_counts = {}
    want_direct = (fisher_mode == 'direct')

    for name in subset:
        n_out = subset[name].out_features
        psi_diag_accum[name] = torch.zeros(n_out, dtype=torch.float32)
        token_counts[name] = 0
        if want_direct and n_out <= max_fisher_dim:
            psi_full_accum[name] = torch.zeros(n_out, n_out, dtype=torch.float32)
        else:
            psi_full_accum[name] = None

    def make_hooks(name):
        def backward_hook(module, grad_input, grad_output):
            b = grad_output[0].detach().float()
            if b.dim() == 3:
                b = b.reshape(-1, b.shape[-1])
            psi_diag_accum[name] += torch.sum(b * b, dim=0).cpu()
            token_counts[name] += b.shape[0]
            if psi_full_accum[name] is not None:
                psi_full_accum[name] += (b.T @ b).cpu()

        return backward_hook

    handles = []
    for name in subset:
        bh = make_hooks(name)
        handles.append(subset[name].register_full_backward_hook(bh))

    for p in layer.parameters():
        p.requires_grad_(True)

    n_fisher = min(num_samples, max_fisher_samples)

    for j in range(n_fisher):
        inp_j = inps_student[j].unsqueeze(0).to(dev).float()
        teacher_inp_j = inps_teacher[j].unsqueeze(0).to(dev).float()
        attn_mask = attention_masks[j].to(dev) if attention_masks[j] is not None else None
        pos_ids = position_ids[j].to(dev) if position_ids is not None else None

        layer.zero_grad()
        inp_for_grad = inp_j.detach().requires_grad_(True)

        if "opt" not in model_name:
            fw_extra = {}
            if pos_ids is not None:
                fw_extra['position_ids'] = pos_ids
            if position_embeddings is not None:
                fw_extra['position_embeddings'] = tuple(t.to(dev) for t in position_embeddings)
            out_student = layer(inp_for_grad, attention_mask=attn_mask, **fw_extra)[0]
        else:
            out_student = layer(inp_for_grad, attention_mask=attn_mask)[0]

        with torch.no_grad():
            out_teacher = teacher_layer.float()(
                teacher_inp_j,
                attention_mask=attn_mask,
                **(fw_extra if "opt" not in model_name else {})
            )[0]
            teacher_layer.half()

        loss = nn.functional.mse_loss(out_student, out_teacher.detach())
        loss.backward()

        del inp_j, teacher_inp_j, out_student, out_teacher, loss, inp_for_grad
        if j % 16 == 0:
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    for p in layer.parameters():
        p.requires_grad_(False)

    result = {}
    for name in subset:
        denom = max(int(token_counts[name]), 1)
        psi_diag = psi_diag_accum[name] / denom
        psi_diag = torch.clamp(psi_diag + lambda_psi, min=lambda_psi)

        psi_cov = None
        if psi_full_accum[name] is not None:
            psi_cov = psi_full_accum[name] / denom
            psi_cov += lambda_psi * torch.eye(psi_cov.shape[0], dtype=torch.float32)

        result[name] = {
            'psi_diag': psi_diag,
            'psi_cov': psi_cov,
            'tokens': denom,
            'direct_available': psi_cov is not None,
        }

    del psi_diag_accum, psi_full_accum
    torch.cuda.empty_cache()
    return result


def aces_beta_select(S, D, rank, alpha_min=0.25, alpha_max=0.75,
                     rho=1.0, beta_cap=None, fisher_weights=None):
    """
    ACES β 选择（论文 Algorithm 3）。

    v1.6 仍保留 legacy fisher_weights 参数用于兼容旧实验；新的
    Fisher-guided CEALC 会先把 S/D 变换到 Fisher metric 空间，再调用本函数。

    ACES 改进：
    - alpha 范围由调用方控制
    - beta_cap 自动从 alpha_max 计算
    - rho 由调用方控制
    - 可选 fisher_weights：对 RER 目标函数中的各方向施加 legacy 加权

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

    # 可选 legacy 加权：只影响 ACES beta 选择，不用于 v1.6 主 Fisher 路径。
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


def _normalize_positive_diag(diag, eps=1e-8, clamp_min=0.5, clamp_max=2.0, gamma=0.5):
    """把 Fisher/KFAC 对角缩放到均值 1，并做保守裁剪和平滑。"""
    diag = torch.nan_to_num(diag.float(), nan=1.0, posinf=clamp_max, neginf=clamp_min)
    diag = torch.clamp(diag, min=eps)
    diag = diag / (diag.mean() + eps)
    diag = torch.clamp(diag, min=clamp_min, max=clamp_max)
    diag = diag / (diag.mean() + eps)
    if gamma != 1.0:
        diag = torch.pow(torch.clamp(diag, min=eps), gamma)
        diag = diag / (diag.mean() + eps)
    return torch.clamp(diag, min=clamp_min, max=clamp_max)


def _spd_sqrt_and_inv_sqrt(mat, dev, damping=1e-4):
    """小矩阵 direct Fisher 路径使用的 SPD sqrt / inv sqrt。"""
    d = mat.shape[0]
    mat = mat.float().cpu()
    diag_mean = torch.diag(mat).mean().clamp(min=damping)
    mat = mat / diag_mean
    mat = mat + damping * torch.eye(d, dtype=torch.float32)
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=damping)
    sqrt_vals = torch.sqrt(eigvals)
    inv_sqrt_vals = 1.0 / sqrt_vals
    Q = eigvecs.to(dev)
    half = (Q * sqrt_vals.to(dev).unsqueeze(0)) @ Q.T
    inv_half = (Q * inv_sqrt_vals.to(dev).unsqueeze(0)) @ Q.T
    return half, inv_half


def build_fisher_transform(name, fisher_info, H_nodamp, fisher_mode, dev,
                           lambda_phi=1e-4, lambda_psi=1e-4,
                           weight_min=0.5, weight_max=2.0,
                           fisher_gamma=0.5,
                           max_fisher_dim=4096):
    """
    构造 FR-SVD v1.6 的 Fisher metric 变换。

    返回的 transform 会作用到实际 SVD 目标：
      G_F = L_psi^T G L_phi
    并在因子回写时反变换，而不是只改变 ACES 打分。
    """
    if fisher_mode == 'none' or fisher_info is None:
        return {'mode': 'none', 'reason': 'disabled'}

    n = H_nodamp.shape[0]
    psi_diag = fisher_info['psi_diag'].to(dev)
    phi_diag = torch.diag(H_nodamp).to(dev) + lambda_phi

    psi_diag = _normalize_positive_diag(
        psi_diag, eps=lambda_psi, clamp_min=weight_min, clamp_max=weight_max,
        gamma=fisher_gamma
    )
    phi_diag = _normalize_positive_diag(
        phi_diag, eps=lambda_phi, clamp_min=weight_min, clamp_max=weight_max,
        gamma=fisher_gamma
    )

    transform = {
        'mode': 'guide',
        'psi_sqrt': torch.sqrt(psi_diag),
        'psi_inv_sqrt': 1.0 / torch.sqrt(psi_diag),
        'phi_sqrt': torch.sqrt(phi_diag),
        'phi_inv_sqrt': 1.0 / torch.sqrt(phi_diag),
        'psi_range': (psi_diag.min().item(), psi_diag.mean().item(), psi_diag.max().item()),
        'phi_range': (phi_diag.min().item(), phi_diag.mean().item(), phi_diag.max().item()),
        'reason': 'diagonal_kfac',
    }

    if fisher_mode != 'direct':
        return transform

    psi_cov = fisher_info.get('psi_cov')
    m = psi_diag.numel()
    if psi_cov is None or m > max_fisher_dim or n > max_fisher_dim:
        transform['reason'] = (
            f"direct_fallback_to_guide(m={m}, n={n}, max={max_fisher_dim})"
        )
        return transform

    try:
        psi_half, psi_inv_half = _spd_sqrt_and_inv_sqrt(psi_cov, dev, lambda_psi)
        phi_half, phi_inv_half = _spd_sqrt_and_inv_sqrt(H_nodamp, dev, lambda_phi)
        transform.update({
            'mode': 'direct',
            'psi_half': psi_half,
            'psi_inv_half': psi_inv_half,
            'phi_half': phi_half,
            'phi_inv_half': phi_inv_half,
            'reason': 'full_kfac',
        })
    except RuntimeError as exc:
        transform['reason'] = f"direct_fallback_to_guide({type(exc).__name__})"
        torch.cuda.empty_cache()

    return transform


def apply_fisher_to_target(S, D, transform):
    """把 S/D 变换到 Fisher metric 空间。"""
    mode = transform['mode']
    if mode == 'direct':
        return transform['psi_half'] @ S @ transform['phi_half'], \
               transform['psi_half'] @ D @ transform['phi_half']
    if mode == 'guide':
        psi = transform['psi_sqrt'].unsqueeze(1)
        phi = transform['phi_sqrt'].unsqueeze(0)
        return S * psi * phi, D * psi * phi
    return S, D


def factors_from_metric_svd(U_r, S_r, Vt_r, H_inv_sqrt, transform):
    """把 Fisher metric 空间的截断 SVD 因子反变换回 Linear 权重空间。"""
    sqrt_s = torch.sqrt(torch.clamp(S_r, min=0))
    B_metric = U_r * sqrt_s.unsqueeze(0)
    A_metric = sqrt_s.unsqueeze(1) * Vt_r

    mode = transform['mode']
    if mode == 'direct':
        B = transform['psi_inv_half'] @ B_metric
        A = A_metric @ transform['phi_inv_half'] @ H_inv_sqrt
    elif mode == 'guide':
        B = B_metric * transform['psi_inv_sqrt'].unsqueeze(1)
        A = (A_metric * transform['phi_inv_sqrt'].unsqueeze(0)) @ H_inv_sqrt
    else:
        B = B_metric
        A = A_metric @ H_inv_sqrt
    return B, A


def eigenspace_residual_compensation(W, W1, H_nodamp, Delta, H_inv_sqrt, rank):
    """
    实验版 eigenspace Stage2（非默认）：
      EX^T = W(H + Delta) - W1H
      DeltaW_LS H^{1/2} = EX^T H^{-1/2}
    对 whitened ridge 解做 rank-r 截断，再乘 H^{-1/2} 反白化。
    """
    if rank <= 0:
        m, n = W.shape
        empty_B = torch.empty(m, 0, device=W.device, dtype=torch.float32)
        empty_A = torch.empty(0, n, device=W.device, dtype=torch.float32)
        return empty_B, empty_A, {'before': 0.0, 'after': 0.0, 'reduction': 0.0}

    EXt = W @ (H_nodamp + Delta) - W1 @ H_nodamp
    DeltaW_white = EXt @ H_inv_sqrt
    U, S, Vt = torch.linalg.svd(DeltaW_white, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    sqrt_s = torch.sqrt(torch.clamp(S_r, min=0))
    B2 = U_r * sqrt_s.unsqueeze(0)
    A2 = (sqrt_s.unsqueeze(1) * Vt_r) @ H_inv_sqrt

    total = torch.sum(S ** 2).item()
    kept = torch.sum(S_r ** 2).item()
    after = max(total - kept, 0.0)
    reduction = kept / (total + 1e-10)
    info = {'before': total, 'after': after, 'reduction': reduction}
    return B2, A2, info


def cealc_lowrank_candidate(name, S, D, rank, H_inv_sqrt, transform,
                            alpha_min, alpha_max, rho, beta_cap):
    """构造一个 CEALC/ACES 低秩候选。"""
    S_F, D_F = apply_fisher_to_target(S, D, transform)
    beta, aces_info = aces_beta_select(
        S_F, D_F, rank,
        alpha_min=alpha_min, alpha_max=alpha_max,
        rho=rho, beta_cap=beta_cap,
    )
    G_F = S_F + beta * D_F
    U, Sigma, Vt = torch.linalg.svd(G_F, full_matrices=False)
    B, A = factors_from_metric_svd(
        U[:, :rank], Sigma[:rank], Vt[:rank, :], H_inv_sqrt, transform
    )
    total_energy = torch.sum(Sigma ** 2).item()
    retained_energy = torch.sum(Sigma[:rank] ** 2).item()
    return {
        'name': name,
        'B': B,
        'A': A,
        'beta': beta,
        'beta2': 0.0,
        'aces_info': aces_info,
        'energy_ratio': retained_energy / (total_energy + 1e-10),
        'stage2_info': {'mode': 'none', 'before': 0.0, 'after': 0.0, 'reduction': 0.0},
        'fisher_mode': transform.get('mode', 'none'),
        'fisher_reason': transform.get('reason', ''),
        'psi_range': transform.get('psi_range'),
        'phi_range': transform.get('phi_range'),
    }


def cealc_residual_compensation(W, W1, H_sqrt, H_inv_sqrt, Delta, rank,
                                alpha_min, alpha_max, rho, beta_cap):
    """v1.4/v1.5 中更稳定的 CEALC-aware residual Stage2。"""
    if rank <= 0:
        m, n = W.shape
        empty_B = torch.empty(m, 0, device=W.device, dtype=torch.float32)
        empty_A = torch.empty(0, n, device=W.device, dtype=torch.float32)
        return empty_B, empty_A, {
            'mode': 'cealc_residual', 'beta2': 0.0,
            'before': 0.0, 'after': 0.0, 'reduction': 0.0
        }

    E_w = W - W1
    S2 = E_w @ H_sqrt
    D2 = E_w @ Delta @ H_inv_sqrt
    beta2, aces_info2 = aces_beta_select(
        S2, D2, rank,
        alpha_min=alpha_min, alpha_max=alpha_max,
        rho=rho, beta_cap=beta_cap,
    )
    G2 = S2 + beta2 * D2
    U, Sigma, Vt = torch.linalg.svd(G2, full_matrices=False)
    sqrt_s = torch.sqrt(torch.clamp(Sigma[:rank], min=0))
    B2 = U[:, :rank] * sqrt_s.unsqueeze(0)
    A2 = (sqrt_s.unsqueeze(1) * Vt[:rank, :]) @ H_inv_sqrt

    total = torch.sum(Sigma ** 2).item()
    kept = torch.sum(Sigma[:rank] ** 2).item()
    return B2, A2, {
        'mode': 'cealc_residual',
        'beta2': beta2,
        'aces_info2': aces_info2,
        'before': total,
        'after': max(total - kept, 0.0),
        'reduction': kept / (total + 1e-10),
    }


def build_stage2_candidate(name, stage1, W, H_sqrt, H_inv_sqrt, H_nodamp, Delta,
                           rank, alpha_min, alpha_max, rho, beta_cap,
                           innovation2_mode='cealc_residual'):
    """基于一个 Stage1 候选追加稳定的 Stage2 补偿分支。"""
    W1 = stage1['B'] @ stage1['A']
    if innovation2_mode == 'eigenspace':
        B2, A2, stage2_info = eigenspace_residual_compensation(
            W, W1, H_nodamp, Delta, H_inv_sqrt, rank
        )
        stage2_info['mode'] = 'eigenspace'
        stage2_info['beta2'] = 0.0
    else:
        B2, A2, stage2_info = cealc_residual_compensation(
            W, W1, H_sqrt, H_inv_sqrt, Delta, rank,
            alpha_min=alpha_min, alpha_max=alpha_max,
            rho=rho, beta_cap=beta_cap,
        )

    B = torch.cat([stage1['B'], B2], dim=1)
    A = torch.cat([stage1['A'], A2], dim=0)
    candidate = {
        'name': name,
        'B': B,
        'A': A,
        'beta': stage1['beta'],
        'beta2': stage2_info.get('beta2', 0.0),
        'aces_info': stage1['aces_info'],
        'energy_ratio': stage1['energy_ratio'],
        'stage2_info': stage2_info,
        'fisher_mode': stage1.get('fisher_mode', 'none'),
        'fisher_reason': stage1.get('fisher_reason', ''),
        'psi_range': stage1.get('psi_range'),
        'phi_range': stage1.get('phi_range'),
    }
    del W1, B2, A2
    return candidate


def evaluate_lowrank_candidate(candidate, W, H_sqrt, global_target):
    """用校准激活目标和普通 W_err 同时评估候选。"""
    B = candidate['B'].float()
    A = candidate['A'].float()
    finite = torch.isfinite(B).all() and torch.isfinite(A).all()
    if not finite:
        candidate.update({
            'finite': False,
            'weight_err': float('inf'),
            'global_score': float('inf'),
            'global_improve': 0.0,
        })
        return candidate

    W_recon = B @ A
    weight_err = (torch.norm(W - W_recon) / (torch.norm(W) + 1e-10)).item()
    residual = global_target - W_recon @ H_sqrt
    global_score = (torch.norm(residual, 'fro') / (torch.norm(global_target, 'fro') + 1e-10)).item()
    candidate.update({
        'finite': True,
        'weight_err': weight_err,
        'global_score': global_score,
        'global_improve': 0.0,
    })
    del W_recon, residual
    return candidate


def candidate_acceptance(candidate, baseline, min_improve=1e-5,
                         weight_rel_limit=1.25, weight_abs_limit=0.05):
    """只有校准目标优于 baseline 且权重误差没有明显恶化时才接受候选。"""
    if not candidate.get('finite', False):
        return False, 'non_finite'
    if candidate['global_score'] >= baseline['global_score'] * (1.0 - min_improve):
        return False, 'no_global_score_improvement'

    weight_limit = max(
        baseline['weight_err'] * weight_rel_limit,
        baseline['weight_err'] + weight_abs_limit,
    )
    if candidate['weight_err'] > weight_limit:
        return False, f"weight_err_too_high(limit={weight_limit:.4f})"
    return True, 'accepted'


# ==================== 主压缩函数 ====================

@torch.no_grad()
def fr_svd_compress_layerwise(model_name, model, calib_loader, ratio, dev,
                                eps=1e-4, r1_ratio=0.92,
                                use_fisher=False, max_fisher_samples=32,
                                lambda_psi=1e-4, lambda_phi=1e-4,
                                rho=0.95, alpha_min=0.4, alpha_max=0.6,
                                fisher_mode='none',
                                enable_innovation2=False,
                                innovation2_mode='cealc_residual',
                                fisher_weight_min=0.5,
                                fisher_weight_max=2.0,
                                fisher_gamma=0.5,
                                max_fisher_dim=4096,
                                skip_moe_experts=False,
                                verbose=True):
    """
    FR-SVD v1.6 逐层闭环压缩主函数（模型无关版）。

    v1.6 开关语义：
    - fisher_mode=none 且 enable_innovation2=False：SAES-SVD 单阶段 CEALC/ACES。
    - fisher_mode=guide/direct：innovation1 作为保守候选参与选择。
    - enable_innovation2=True：默认使用稳定的 CEALC-aware residual Stage2。
    """
    # 兼容旧接口：use_fisher=True + fisher_mode='none' → 使用 guide
    if use_fisher and fisher_mode == 'none':
        fisher_mode = 'guide'
    if fisher_mode == 'soft':
        print("  [兼容提示] fisher_mode=soft 在 v1.6 中映射为 guide：Fisher 会进入实际 SVD metric。")
        fisher_mode = 'guide'
    use_fisher = fisher_mode in ('direct', 'guide')
    if use_fisher and (fisher_weight_min < 0.5 or fisher_weight_max > 2.0):
        print(
            f"  [稳定性提示] Fisher clamp [{fisher_weight_min}, {fisher_weight_max}] "
            "过宽，v1.6 工程模式自动收紧到 [0.5, 2.0]"
        )
        fisher_weight_min = max(fisher_weight_min, 0.5)
        fisher_weight_max = min(fisher_weight_max, 2.0)

    beta_cap = alpha_max / (1 + alpha_max)

    if enable_innovation2:
        mode_name = (
            "full FR-SVD" if use_fisher else
            "innovation2 only / residual compensation"
        )
    else:
        mode_name = (
            "innovation1 only / Fisher-guided CEALC" if use_fisher else
            "SAES baseline"
        )

    fisher_mode_str = {
        'none': 'OFF',
        'guide': 'diagonal KFAC Fisher metric',
        'direct': 'full KFAC Fisher metric when small, otherwise guide fallback'
    }[fisher_mode]
    print("=" * 60)
    print("FR-SVD v1.6: Candidate-selected Fisher CEALC + Stable Residual")
    print("=" * 60)
    print(f"  模式: {mode_name}")
    print(f"  保留比例: {ratio} ({ratio*100:.1f}%)")
    if enable_innovation2:
        print(f"  秩分配: r1={r1_ratio*100:.0f}% of r / r2={100-r1_ratio*100:.0f}% of r (r1+r2=r)")
    else:
        print("  秩分配: single-stage，全部 rank 用于 Stage1（SAES 等价路径）")
    print(f"  Fisher 模式: {fisher_mode_str}")
    print(f"  Innovation2 residual: {'ON' if enable_innovation2 else 'OFF'} ({innovation2_mode})")
    print(f"  H 阻尼: eps={eps}")
    print(f"  Fisher 阻尼: lambda_phi={lambda_phi}, lambda_psi={lambda_psi}, gamma={fisher_gamma}")
    print(f"  Alpha 范围: [{alpha_min}, {alpha_max}], rho={rho}, beta_cap={beta_cap:.4f}")
    print("=" * 60)

    # ========== Step 1：省内存模式 ==========
    # 不加载完整教师模型，改为逐层 deepcopy 作为教师参考
    # 对 30B+ 模型可节省 60-70 GB CPU 内存
    import copy

    # ========== Step 2：获取模型的层列表（架构无关）==========
    arch_family = _detect_arch_family(model)
    is_opt = (arch_family == 'opt')
    is_moe = (arch_family == 'qwen3_moe')
    print(f"  检测到架构家族: {arch_family}")
    if is_moe:
        print(f"  MoE skip_experts: {skip_moe_experts}")

    student_access = get_model_access(model, arch_family)
    layers = student_access['layers']

    # 把 embed / norm / rotary_emb 等预处理模块搬到 dev
    for _path, mod in student_access['to_dev']:
        mod.to(dev)
    _preamble_mods_student = [m for _, m in student_access['to_dev']]

    # ========== Step 3：捕获第 0 层输入激活 ==========
    layers[0] = layers[0].to(dev)
    num_samples = len(calib_loader)

    inps_student = torch.zeros(
        (num_samples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16, device='cpu'
    )

    cache = {'i': 0}
    attention_masks_list = []
    position_ids_list = []
    position_embeddings_cache = [None]  # 只需保存一份 (cos, sin)，所有样本共享

    class StudentCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps_student[cache['i']] = inp.detach().half().cpu()
            attn_mask = kwargs.get('attention_mask')
            attention_masks_list.append(attn_mask.detach().cpu() if attn_mask is not None else None)
            if not is_opt:
                pos_ids = kwargs.get('position_ids')
                position_ids_list.append(pos_ids.detach().cpu() if pos_ids is not None else None)
                # 捕获 position_embeddings（Qwen2/Qwen3 等需要预计算的 cos/sin）
                if position_embeddings_cache[0] is None:
                    pos_emb = kwargs.get('position_embeddings')
                    if pos_emb is not None:
                        position_embeddings_cache[0] = tuple(t.detach().cpu() for t in pos_emb)
            cache['i'] += 1
            raise ValueError

    layers[0] = StudentCatcher(layers[0])

    print("收集第 0 层输入激活...")
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    # 把预处理模块搬回 CPU
    for mod in _preamble_mods_student:
        mod.cpu()

    torch.cuda.empty_cache()

    # 初始时 student 和 teacher 输入相同（都来自同一嵌入层）
    inps_teacher = inps_student.clone()
    outs_student = torch.zeros_like(inps_student)
    outs_teacher = torch.zeros(
        (num_samples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16, device='cpu'
    )
    attention_masks = attention_masks_list
    position_ids = position_ids_list if position_ids_list and position_ids_list[0] is not None else None
    # position_embeddings: (cos, sin) 元组，Qwen2/Qwen3 等需要；LLaMA/OPT 为 None
    pos_embed_cached = position_embeddings_cache[0]

    # ========== Step 4：逐层压缩主循环 ==========
    compression_stats = []
    selection_counts = {
        'baseline': 0,
        'fisher': 0,
        'stage2': 0,
        'fisher_stage2': 0,
        'tsvd': 0,
    }
    rejected_counts = {}

    print("\n开始逐层 FR-SVD v1.6 压缩...")

    for layer_idx in tqdm(range(len(layers)), desc="FR-SVD v1.6 Compression"):
        layer_stats = {'layer': layer_idx, 'sublayers': {}}

        # 逐层 deepcopy：仅克隆当前层作为教师参考，而非加载整个教师模型
        layer = layers[layer_idx].to(dev)
        teacher_layer = copy.deepcopy(layer)  # fp16 克隆，~1-2GB/层
        layer = layer.float()

        subset = find_layers(layer)
        teacher_subset = find_layers(teacher_layer)
        # 过滤 MoE router / 可选跳过 experts
        subset = filter_linears_for_compression(subset, skip_moe_experts=skip_moe_experts)
        teacher_subset = {k: v for k, v in teacher_subset.items() if k in subset}

        # ────────── Phase A：收集 H 和 Δ（增量式，不存全量 inp_cache）──────────
        for name in subset:
            subset[name].H_matrix = 0
            subset[name].Delta_matrix = 0
            subset[name]._tmp_inp = None
            teacher_subset[name]._tmp_inp = None

        def make_hook_student(module_name):
            def hook(module, input, output):
                inp = input[0].detach().float()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                H_batch = torch.matmul(inp.transpose(1, 2), inp)
                module.H_matrix = module.H_matrix + torch.sum(H_batch, dim=0)
                module._tmp_inp = inp  # 暂存当前样本在 GPU，用于 Delta 计算
            return hook

        def make_hook_teacher(module_name):
            def hook(module, input, output):
                inp = input[0].detach().float()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                module._tmp_inp = inp  # 暂存当前样本在 GPU
            return hook

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_hook_student(name)))
            handles.append(teacher_subset[name].register_forward_hook(make_hook_teacher(name)))

        for j in range(num_samples):
            inp_j = inps_student[j].unsqueeze(0).to(dev).float()
            attn_mask_j = attention_masks[j].to(dev) if attention_masks[j] is not None else None
            pos_ids_j = position_ids[j].to(dev) if position_ids is not None else None

            fw_kwargs = {}
            if attn_mask_j is not None:
                fw_kwargs['attention_mask'] = attn_mask_j
            if not is_opt and pos_ids_j is not None:
                fw_kwargs['position_ids'] = pos_ids_j
            if pos_embed_cached is not None:
                fw_kwargs['position_embeddings'] = tuple(t.to(dev) for t in pos_embed_cached)

            out_j = layer(inp_j, **fw_kwargs)[0]
            outs_student[j] = out_j.half().cpu()

            teacher_inp = inps_teacher[j].unsqueeze(0).to(dev).float()
            teacher_out = teacher_layer.float()(teacher_inp, **fw_kwargs)[0]
            outs_teacher[j] = teacher_out.half().cpu()
            teacher_layer.half()

            # 增量计算 Δ = Σ (X^f - X) X^T（逐样本累加，不存全量 cache）
            for name in subset:
                X = subset[name]._tmp_inp
                X_f = teacher_subset[name]._tmp_inp
                diff = X_f - X
                Delta_batch = torch.matmul(diff.transpose(1, 2), X)
                subset[name].Delta_matrix = subset[name].Delta_matrix + torch.sum(Delta_batch, dim=0)
                subset[name]._tmp_inp = None
                teacher_subset[name]._tmp_inp = None
                del X, X_f, diff, Delta_batch

            del inp_j, out_j, teacher_inp, teacher_out
            if j % 32 == 0:
                torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        for name in subset:
            subset[name].H_matrix = subset[name].H_matrix.cpu()
            subset[name].Delta_matrix = subset[name].Delta_matrix.cpu()

        torch.cuda.empty_cache()

        # ────────── Phase B：Fisher Psi 估计（innovation1）──────────
        fisher_cache = {}
        if use_fisher:
            with torch.enable_grad():
                fisher_cache = estimate_kronecker_fisher(
                    layer, teacher_layer, inps_student, inps_teacher,
                    attention_masks, position_ids, model_name, dev,
                    num_samples, lambda_psi=lambda_psi,
                    max_fisher_samples=max_fisher_samples,
                    position_embeddings=pos_embed_cached,
                    fisher_mode=fisher_mode,
                    max_fisher_dim=max_fisher_dim
                )
            torch.cuda.empty_cache()

        # Phase B 结束后释放 teacher_layer，后续 Phase D 不再需要
        del teacher_subset, teacher_layer
        torch.cuda.empty_cache()

        # ────────── Phase C（v1.5 简化）：无需预先创建专用 SVD 模块 ──────────
        # FactorizedLinear 会在 Phase D 逐子层原位替换原 nn.Linear，
        # 保留原层的 Attention/MLP 前向逻辑（包括 GQA、bias、MoE 路由等）。

        # ────────── Phase D：逐子层两阶段压缩 ──────────
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            original_dtype = subset[name].weight.data.dtype
            m, n = W.shape

            H_nodamp = subset[name].H_matrix.float().to(dev)
            Delta = subset[name].Delta_matrix.float().to(dev)

            r = int(m * n * ratio / (m + n))
            r = max(1, min(r, min(m, n)))
            if enable_innovation2 and r > 1:
                r1 = max(1, min(r - 1, int(r * r1_ratio)))
            else:
                r1 = r
            r2 = r - r1

            # NaN 防护
            if not torch.isfinite(H_nodamp).all() or not torch.isfinite(Delta).all():
                print(f"  ⚠ L{layer_idx} {name}: H/Delta 含 NaN，替换为 eps*I / 零矩阵")
                H_nodamp = eps * torch.eye(n, device=dev, dtype=torch.float32)
                Delta = torch.zeros(n, n, device=dev, dtype=torch.float32)

            Q, Lambda, H_sqrt, H_inv_sqrt = compute_H_decomposition(H_nodamp, dev, eps)
            S = W @ H_sqrt              # S = W H^{1/2}
            D = W @ Delta @ H_inv_sqrt  # D = W Δ H^{-1/2}
            global_target = S + D       # W(H+Δ)H^{-1/2}，用于候选选择

            none_transform = {'mode': 'none', 'reason': 'baseline'}
            baseline = cealc_lowrank_candidate(
                'baseline', S, D, r, H_inv_sqrt, none_transform,
                alpha_min, alpha_max, rho, beta_cap
            )
            baseline = evaluate_lowrank_candidate(baseline, W, H_sqrt, global_target)
            selected = baseline
            candidate_logs = []

            def consider_candidate(candidate):
                candidate = evaluate_lowrank_candidate(candidate, W, H_sqrt, global_target)
                candidate['global_improve'] = (
                    (baseline['global_score'] - candidate['global_score']) /
                    (baseline['global_score'] + 1e-10)
                )
                accepted, reason = candidate_acceptance(candidate, baseline)
                candidate_logs.append({
                    'name': candidate['name'],
                    'score': candidate['global_score'],
                    'weight_err': candidate['weight_err'],
                    'improve': candidate['global_improve'],
                    'accepted': accepted,
                    'reason': reason,
                })
                if not accepted:
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                    del candidate['B'], candidate['A']
                    return selected
                if candidate['global_score'] < selected['global_score']:
                    old_selected = selected
                    selected_new = candidate
                    if old_selected is not baseline:
                        del old_selected['B'], old_selected['A']
                    else:
                        del baseline['B'], baseline['A']
                    return selected_new
                del candidate['B'], candidate['A']
                return selected

            fisher_transform = build_fisher_transform(
                name,
                fisher_cache.get(name) if use_fisher else None,
                H_nodamp,
                fisher_mode,
                dev,
                lambda_phi=lambda_phi,
                lambda_psi=lambda_psi,
                weight_min=fisher_weight_min,
                weight_max=fisher_weight_max,
                fisher_gamma=fisher_gamma,
                max_fisher_dim=max_fisher_dim
            )

            if use_fisher:
                fisher_candidate = cealc_lowrank_candidate(
                    'fisher', S, D, r, H_inv_sqrt, fisher_transform,
                    alpha_min, alpha_max, rho, beta_cap
                )
                selected = consider_candidate(fisher_candidate)

            if enable_innovation2 and r2 > 0:
                stage1 = cealc_lowrank_candidate(
                    'stage2_stage1', S, D, r1, H_inv_sqrt, none_transform,
                    alpha_min, alpha_max, rho, beta_cap
                )
                stage2_candidate = build_stage2_candidate(
                    'stage2', stage1, W, H_sqrt, H_inv_sqrt, H_nodamp, Delta,
                    r2, alpha_min, alpha_max, rho, beta_cap,
                    innovation2_mode=innovation2_mode
                )
                del stage1['B'], stage1['A']
                selected = consider_candidate(stage2_candidate)

                if use_fisher:
                    fisher_stage1 = cealc_lowrank_candidate(
                        'fisher_stage2_stage1', S, D, r1, H_inv_sqrt, fisher_transform,
                        alpha_min, alpha_max, rho, beta_cap
                    )
                    fisher_stage2_candidate = build_stage2_candidate(
                        'fisher_stage2', fisher_stage1, W, H_sqrt, H_inv_sqrt, H_nodamp, Delta,
                        r2, alpha_min, alpha_max, rho, beta_cap,
                        innovation2_mode=innovation2_mode
                    )
                    del fisher_stage1['B'], fisher_stage1['A']
                    selected = consider_candidate(fisher_stage2_candidate)

            tsvd_err = None
            fallback_reason = None
            if (not selected.get('finite', False)) or selected['weight_err'] > 1.5:
                B_tsvd, A_tsvd, tsvd_err = truncated_svd_factorize(W, r)
                if 'B' in selected:
                    del selected['B'], selected['A']
                selected = {
                    'name': 'tsvd',
                    'B': B_tsvd,
                    'A': A_tsvd,
                    'beta': 0.0,
                    'beta2': 0.0,
                    'aces_info': {'alpha_star': 0.0, 'best_score': 0.0},
                    'energy_ratio': 0.0,
                    'stage2_info': {'mode': 'tsvd', 'before': 0.0, 'after': 0.0, 'reduction': 0.0},
                    'fisher_mode': 'none',
                    'fisher_reason': 'tsvd_fallback',
                    'psi_range': None,
                    'phi_range': None,
                    'finite': True,
                    'weight_err': tsvd_err,
                    'global_score': float('inf'),
                    'global_improve': 0.0,
                }
                fallback_reason = 'tsvd_last_resort'

            B_merged = selected['B']
            A_merged = selected['A']
            beta = selected['beta']
            beta2 = selected.get('beta2', 0.0)
            aces_info = selected['aces_info']
            energy_ratio = selected['energy_ratio']
            stage2_info = selected['stage2_info']
            weight_err = selected['weight_err']
            selected_name = selected['name']
            selection_counts[selected_name] = selection_counts.get(selected_name, 0) + 1

            B_merged = B_merged.to(original_dtype)
            A_merged = A_merged.to(original_dtype)

            delta_norm = torch.norm(Delta).item()
            h_norm = torch.norm(H_nodamp).item()

            layer_stats['sublayers'][name] = {
                'beta': beta,
                'beta2': beta2 if r2 > 0 else 0.0,
                'r': r, 'r1': r1, 'r2': r2,
                'weight_err': weight_err,
                'tsvd_err': tsvd_err,
                'fallback_reason': fallback_reason,
                'energy_ratio': energy_ratio,
                'delta_norm': delta_norm,
                'h_norm': h_norm,
                'delta_ratio': delta_norm / (h_norm + 1e-10),
                'aces_info': aces_info,
                'selected_candidate': selected_name,
                'global_score': selected.get('global_score'),
                'global_improve': selected.get('global_improve', 0.0),
                'baseline_global_score': baseline.get('global_score'),
                'candidate_logs': candidate_logs,
                'fisher_mode': selected.get('fisher_mode', 'none'),
                'fisher_reason': selected.get('fisher_reason', ''),
                'psi_range': selected.get('psi_range'),
                'phi_range': selected.get('phi_range'),
                'stage2_info': stage2_info,
            }

            should_print = verbose and (
                use_fisher or enable_innovation2 or
                layer_idx == 0 or layer_idx == len(layers) - 1 or layer_idx % 8 == 0
            )
            if should_print:
                stage2_str = (
                    f", Stage2_red={stage2_info['reduction']:.4f}"
                    if r2 > 0 else ""
                )
                fisher_str = f", Fisher={selected.get('fisher_mode', 'none')}:{selected.get('fisher_reason', '')}"
                if selected.get('psi_range') is not None:
                    pmin, pmean, pmax = selected['psi_range']
                    qmin, qmean, qmax = selected['phi_range']
                    fisher_str += (
                        f", Psi[{pmin:.3g}/{pmean:.3g}/{pmax:.3g}]"
                        f", Phi[{qmin:.3g}/{qmean:.3g}/{qmax:.3g}]"
                    )
                reject_str = ""
                if candidate_logs:
                    compact = []
                    for c in candidate_logs:
                        state = "A" if c['accepted'] else f"R:{c['reason']}"
                        compact.append(
                            f"{c['name']} {state} score={c['score']:.4g} "
                            f"imp={c['improve']:.3%} W={c['weight_err']:.4f}"
                        )
                    reject_str = " | candidates: " + "; ".join(compact)
                print(f"  L{layer_idx} {name}: β={beta:.4f}, α*={aces_info['alpha_star']:.4f}, "
                      f"r={r}(r1={r1}+r2={r2}), E_kept={energy_ratio:.4f}, "
                      f"W_err={weight_err:.4f}, score={selected.get('global_score', 0.0):.4g}, "
                      f"base_score={baseline.get('global_score', 0.0):.4g}, "
                      f"chosen={selected_name}, D/H={delta_norm/(h_norm+1e-10):.4f}"
                      f"{stage2_str}{fisher_str}{reject_str}")

            # 写入：原位用 FactorizedLinear 替换当前 Linear（架构无关）
            bias_data = subset[name].bias.data if subset[name].bias is not None else None
            replace_linear_with_factorized(
                layer, name, B_merged, A_merged, bias_data,
                device=dev, dtype=original_dtype
            )

            del W, H_nodamp, Delta
            del Q, Lambda, H_sqrt, H_inv_sqrt
            del S, D, global_target
            if 'B' in selected:
                del selected['B'], selected['A']
            del B_merged, A_merged, selected, baseline, fisher_transform
            del subset[name].H_matrix, subset[name].Delta_matrix
            torch.cuda.empty_cache()

        # 释放 Fisher 缓存
        if use_fisher:
            for name in list(fisher_cache.keys()):
                del fisher_cache[name]
            fisher_cache.clear()
            torch.cuda.empty_cache()

        # ────────── Phase E（v1.5 已并入 Phase D）──────────
        # 所有 Linear 已在 Phase D 原位被 FactorizedLinear 替换，layer 结构未变。

        avg_beta = sum(s['beta'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        avg_delta_ratio = sum(s['delta_ratio'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        avg_stage2_reduction = sum(s['stage2_info']['reduction'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        layer_selection_counts = {}
        for s in layer_stats['sublayers'].values():
            cname = s.get('selected_candidate', 'unknown')
            layer_selection_counts[cname] = layer_selection_counts.get(cname, 0) + 1
        layer_stats['avg_beta'] = avg_beta
        layer_stats['avg_delta_ratio'] = avg_delta_ratio
        layer_stats['avg_stage2_reduction'] = avg_stage2_reduction
        layer_stats['selection_counts'] = layer_selection_counts

        if verbose:
            avg_alpha_star = sum(s['aces_info']['alpha_star'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
            print(f"Layer {layer_idx} Summary: avg_β={avg_beta:.4f}, avg_α*={avg_alpha_star:.4f}, "
                  f"avg_D/H={avg_delta_ratio:.4f}, avg_Stage2_red={avg_stage2_reduction:.4f}, "
                  f"selected={layer_selection_counts}")

        compression_stats.append(layer_stats)

        # ────────── Phase F：闭环重新前向 ──────────
        _phase_f_nan_count = 0
        for j in range(num_samples):
            inp_j = inps_student[j].unsqueeze(0).to(dev).float()
            attn_mask_j = attention_masks[j].to(dev) if attention_masks[j] is not None else None
            pos_ids_j = position_ids[j].to(dev) if position_ids is not None else None

            fw_kwargs = {}
            if attn_mask_j is not None:
                fw_kwargs['attention_mask'] = attn_mask_j
            if not is_opt and pos_ids_j is not None:
                fw_kwargs['position_ids'] = pos_ids_j
            if pos_embed_cached is not None:
                fw_kwargs['position_embeddings'] = tuple(t.to(dev) for t in pos_embed_cached)

            out_j = layer(inp_j, **fw_kwargs)[0]

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
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("FR-SVD v1.6 压缩完成！")
    print("=" * 60)

    all_betas = [s['avg_beta'] for s in compression_stats]
    all_delta_ratios = [s['avg_delta_ratio'] for s in compression_stats]
    print(f"  平均 beta: {sum(all_betas)/len(all_betas):.4f}")
    print(f"  平均 D/H:  {sum(all_delta_ratios)/len(all_delta_ratios):.4f}")
    print(f"  Beta 范围: [{min(all_betas):.4f}, {max(all_betas):.4f}]")
    total_selected = sum(selection_counts.values()) or 1
    print("  候选选择统计:")
    for cname, count in selection_counts.items():
        print(f"    {cname}: {count} ({count / total_selected:.2%})")
    if rejected_counts:
        print("  候选拒绝原因:")
        for reason, count in sorted(rejected_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {reason}: {count}")

    return compression_stats


# ==================== 冒烟测试 ====================

def smoke_test():
    """单元测试：验证 v1.6 开关语义和核心数学路径。"""
    print("=" * 50)
    print("FR-SVD v1.6 Smoke Test")
    print("=" * 50)

    torch.manual_seed(42)
    m, n, p = 96, 48, 192
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

    # ── off/off：SAES 单阶段 CEALC/ACES 路径 ──
    S = W @ H_sqrt
    D = W @ Delta @ H_inv_sqrt

    beta, info = aces_beta_select(S, D, r1)
    print(f"  SAES Stage1: beta*={beta:.4f}, alpha*={info['alpha_star']:.4f}")

    none_transform = {'mode': 'none'}
    S_base, D_base = apply_fisher_to_target(S, D, none_transform)
    assert torch.allclose(S_base, S) and torch.allclose(D_base, D)
    G = S + beta * D
    U, Sigma, Vt = torch.linalg.svd(G, full_matrices=False)
    U_r1 = U[:, :r1]
    S_r1 = Sigma[:r1]
    Vt_r1 = Vt[:r1, :]
    B1, A1 = factors_from_metric_svd(U_r1, S_r1, Vt_r1, H_inv_sqrt, none_transform)

    assert B1.shape == (m, r1), f"B1 维度错误: {B1.shape}"
    assert A1.shape == (r1, n), f"A1 维度错误: {A1.shape}"
    print("[PASS] off/off 等价 SAES 单阶段目标")

    # ── innovation1：Fisher guide 改变实际 SVD metric ──
    fisher_info = {
        'psi_diag': torch.linspace(0.25, 4.0, m),
        'psi_cov': None,
        'tokens': p,
        'direct_available': False,
    }
    fisher_transform = build_fisher_transform(
        'toy', fisher_info, H_nodamp, 'guide', 'cpu',
        lambda_phi=1e-4, lambda_psi=1e-4
    )
    assert abs(fisher_transform['psi_range'][1] - 1.0) < 0.25
    assert abs(fisher_transform['phi_range'][1] - 1.0) < 0.25
    S_F, D_F = apply_fisher_to_target(S, D, fisher_transform)
    assert torch.norm(S_F - S).item() > 1e-4, "Fisher guide 未改变实际 SVD 目标"
    beta_f, info_f = aces_beta_select(S_F, D_F, r1, alpha_min=0.4, alpha_max=0.6, rho=0.95)
    G_F = S_F + beta_f * D_F
    U_f, Sigma_f, Vt_f = torch.linalg.svd(G_F, full_matrices=False)
    B_f, A_f = factors_from_metric_svd(
        U_f[:, :r1], Sigma_f[:r1], Vt_f[:r1, :], H_inv_sqrt, fisher_transform
    )
    assert torch.isfinite(B_f).all() and torch.isfinite(A_f).all()
    assert torch.norm((B_f @ A_f) - (B1 @ A1)).item() > 1e-4
    print(f"[PASS] Fisher guide 进入实际 SVD metric: beta={beta_f:.4f}, alpha={info_f['alpha_star']:.4f}")

    # ── innovation2：稳定 CEALC-aware residual compensation ──
    W1 = B1 @ A1
    B2, A2, stage2_info = cealc_residual_compensation(
        W, W1, H_sqrt, H_inv_sqrt, Delta, r2,
        alpha_min=0.4, alpha_max=0.6, rho=0.95, beta_cap=0.6 / 1.6
    )

    assert B2.shape == (m, r2), f"B2 维度错误: {B2.shape}"
    assert A2.shape == (r2, n), f"A2 维度错误: {A2.shape}"
    assert torch.isfinite(B2).all() and torch.isfinite(A2).all()
    assert stage2_info['mode'] == 'cealc_residual'
    print(
        f"[PASS] CEALC residual Stage2 有限且可诊断: "
        f"{stage2_info['before']:.4f} -> {stage2_info['after']:.4f}, "
        f"reduction={stage2_info['reduction']:.4f}"
    )

    global_target = S + D
    baseline = cealc_lowrank_candidate(
        'baseline', S, D, r, H_inv_sqrt, none_transform,
        alpha_min=0.4, alpha_max=0.6, rho=0.95, beta_cap=0.6 / 1.6
    )
    baseline = evaluate_lowrank_candidate(baseline, W, H_sqrt, global_target)
    bad = {
        'name': 'bad',
        'B': 1000.0 * baseline['B'],
        'A': baseline['A'],
    }
    bad = evaluate_lowrank_candidate(bad, W, H_sqrt, global_target)
    ok, reason = candidate_acceptance(bad, baseline)
    assert (not ok) and (reason == 'no_global_score_improvement' or reason.startswith('weight_err_too_high'))
    del baseline['B'], baseline['A'], bad['B'], bad['A']
    print("[PASS] 候选选择会拒绝明显坏候选")

    # ── 合并 ──
    B_merged = torch.cat([B1, B2], dim=1)
    A_merged = torch.cat([A1, A2], dim=0)
    assert B_merged.shape == (m, r), f"B_merged 维度: {B_merged.shape}"
    assert A_merged.shape == (r, n), f"A_merged 维度: {A_merged.shape}"
    print("[PASS] 并联低秩分支可合并为单个 FactorizedLinear (r=%d)" % r)

    print("\n所有冒烟测试通过！")
    print("=" * 50)


# ==================== 命令行入口 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FR-SVD v1.6：SAES baseline + candidate-selected Fisher CEALC + stable residual'
    )

    # 基础参数
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--ratio', type=float, default=0.2, help='压缩率（0.2 = 去掉 20%%）')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='校准数据集')
    parser.add_argument('--nsamples', type=int, default=128, help='校准样本数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--DEV', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--save_path', type=str, default='./result/FR_SVD_v1.6', help='保存路径')
    parser.add_argument('--result_dir', type=str, default=None, help='结果目录')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='序列长度')

    # FR-SVD v1.6 参数
    parser.add_argument('--eps', type=float, default=1e-4, help='H 阻尼系数')
    parser.add_argument('--r1_ratio', type=float, default=0.92,
                        help='启用 innovation2 时 Stage 1 秩占比；关闭 innovation2 时忽略')
    parser.add_argument('--rho', type=float, default=0.95, help='ACES β 收缩因子（SAES 常用默认 0.95）')
    parser.add_argument('--alpha_min', type=float, default=0.4, help='ACES α 下界（SAES 常用默认 0.4）')
    parser.add_argument('--alpha_max', type=float, default=0.6, help='ACES α 上界（SAES 常用默认 0.6）')
    parser.add_argument('--use_fisher', action='store_true', default=False,
                        help='启用 Fisher（兼容旧接口，等价于 --fisher_mode guide）')
    parser.add_argument('--fisher_mode', type=str, default='none', choices=['none', 'guide', 'direct', 'soft'],
                        help='Fisher 模式：none=SAES, guide=diagonal KFAC, direct=小矩阵完整 KFAC, soft=兼容映射到 guide')
    parser.add_argument('--enable_innovation2', action='store_true', default=False,
                        help='启用工程稳定版 residual compensation 候选')
    parser.add_argument('--innovation2_mode', type=str, default='cealc_residual',
                        choices=['cealc_residual', 'eigenspace'],
                        help='Innovation2 实现：cealc_residual=稳定默认, eigenspace=实验路径')
    parser.add_argument('--max_fisher_samples', type=int, default=32, help='Fisher 估计最大样本数')
    parser.add_argument('--lambda_psi', type=float, default=1e-4, help='Psi 阻尼系数')
    parser.add_argument('--lambda_phi', type=float, default=1e-4, help='Phi 阻尼系数')
    parser.add_argument('--fisher_weight_min', type=float, default=0.5, help='diagonal Fisher 权重裁剪下界')
    parser.add_argument('--fisher_weight_max', type=float, default=2.0, help='diagonal Fisher 权重裁剪上界')
    parser.add_argument('--fisher_gamma', type=float, default=0.5, help='Fisher 权重平滑指数，越小越保守')
    parser.add_argument('--max_fisher_dim', type=int, default=4096, help='direct Fisher 允许的最大输入/输出维度')

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
    parser.add_argument('--eval_large', action='store_true', default=False,
                        help='超大模型评估模式：模型常驻 CPU，逐层搬上 GPU（适用于 30B+ 等超过单卡显存的模型）')
    parser.add_argument('--no_eval_after_compress', action='store_true', default=False,
                        help='压缩并保存后立即退出，不运行压缩后的 PPL 评估')
    # v1.5 新增：多模型 / MoE
    parser.add_argument('--skip_moe_experts', action='store_true', default=False,
                        help='MoE 模型（如 Qwen3.5-35B-A3B）跳过所有 experts 的 gate/up/down_proj 压缩，'
                             '仅压缩 attention + shared_expert（推荐：experts 每个只看到少量 token）')

    args = parser.parse_args()

    if args.result_dir:
        args.save_path = args.result_dir

    if args.smoke_test:
        smoke_test()
        sys.exit(0)

    if args.model is None and not args.eval_only:
        parser.error("非 --eval_only 模式下必须指定 --model")

    keep_ratio = 1 - args.ratio
    effective_fisher = args.fisher_mode
    if args.use_fisher and effective_fisher == 'none':
        effective_fisher = 'guide'
    if effective_fisher == 'soft':
        effective_fisher = 'guide'
    if args.enable_innovation2:
        mode_name = 'full FR-SVD' if effective_fisher != 'none' else 'innovation2 only'
    else:
        mode_name = 'innovation1 only' if effective_fisher != 'none' else 'SAES baseline'

    print("=" * 60)
    print("FR-SVD v1.6 配置")
    print("=" * 60)
    print(f"  模式: {mode_name}")
    print(f"  模型: {args.model}")
    print(f"  压缩率: {args.ratio} (保留 {keep_ratio*100:.1f}%)")
    print(f"  数据集: {args.dataset}, 样本数: {args.nsamples}")
    print(f"  设备: {args.DEV}")
    print(f"  r1_ratio: {args.r1_ratio} ({'启用' if args.enable_innovation2 else '关闭 innovation2 时忽略'})")
    fisher_mode_display = {
        'none': 'OFF',
        'guide': 'diagonal KFAC Fisher metric',
        'direct': 'full KFAC if small else guide fallback',
    }[effective_fisher]
    print(f"  Fisher 模式: {fisher_mode_display}")
    print(f"  Innovation2 residual: {'ON' if args.enable_innovation2 else 'OFF'} ({args.innovation2_mode})")
    print(f"  eps={args.eps}")
    print(f"  lambda_phi={args.lambda_phi}, lambda_psi={args.lambda_psi}, fisher_gamma={args.fisher_gamma}")
    print(f"  Alpha: [{args.alpha_min}, {args.alpha_max}], rho={args.rho}")
    print(f"  skip_moe_experts: {args.skip_moe_experts}")
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
        print("\n运行困惑度评估...")
        if args.eval_large:
            # 逐层评估：不整体搬 GPU，适用于超过单卡显存的大模型
            ppl_eval_large(model, tokenizer, datasets=['wikitext2'],
                           seq_len=args.model_seq_len,
                           batch_size=args.eval_batch_size, device=args.DEV)
        else:
            if args.eval_dtype == 'fp32':
                model = model.float().to(args.DEV)
            else:
                model = model.half().to(args.DEV)
            ppl_eval(model, tokenizer, datasets=['wikitext2'],
                     model_seq_len=args.model_seq_len,
                     batch_size=args.eval_batch_size, device=args.DEV)
    else:
        print("\n加载模型...")
        model, tokenizer = load_model_v15(args.model, seq_len=args.model_seq_len)
        model = model.eval().half()

        print("\n加载校准数据...")
        calib_data = get_calib_train_data(
            args.dataset, tokenizer, args.nsamples, seqlen=args.model_seq_len
        )

        print("\n开始 FR-SVD v1.6 压缩...")
        stats = fr_svd_compress_layerwise(
            args.model, model, calib_data, keep_ratio, args.DEV,
            eps=args.eps, r1_ratio=args.r1_ratio,
            use_fisher=args.use_fisher, max_fisher_samples=args.max_fisher_samples,
            lambda_psi=args.lambda_psi, lambda_phi=args.lambda_phi,
            rho=args.rho, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
            fisher_mode=args.fisher_mode,
            enable_innovation2=args.enable_innovation2,
            innovation2_mode=args.innovation2_mode,
            fisher_weight_min=args.fisher_weight_min,
            fisher_weight_max=args.fisher_weight_max,
            fisher_gamma=args.fisher_gamma,
            max_fisher_dim=args.max_fisher_dim,
            skip_moe_experts=args.skip_moe_experts,
            verbose=True
        )

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        save_file = os.path.join(
            args.save_path,
            f"{args.model.replace('/', '_').replace('-', '_')}_fr_svd_v1.6_{keep_ratio}.pt"
        )
        print(f"\n保存压缩模型到: {save_file}")
        model = model.float()
        torch.save({'model': model, 'tokenizer': tokenizer, 'stats': stats}, save_file)

        del stats, calib_data
        import gc; gc.collect()
        torch.cuda.empty_cache()

        if args.no_eval_after_compress:
            print("\n跳过压缩后的困惑度评估（--no_eval_after_compress）")
            sys.exit(0)

        print("\n运行困惑度评估...")
        if args.eval_large:
            ppl_eval_large(model, tokenizer, datasets=['wikitext2'],
                           seq_len=args.model_seq_len,
                           batch_size=args.eval_batch_size, device=args.DEV)
        else:
            model = model.half().to(args.DEV)
            ppl_eval(model, tokenizer, datasets=['wikitext2'],
                     model_seq_len=args.model_seq_len,
                     batch_size=args.eval_batch_size, device=args.DEV)
