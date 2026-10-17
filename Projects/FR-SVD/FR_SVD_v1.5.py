#coding:utf8
"""
版本历史：
  v1.0-v1.3  见 v1.4 注释。
  v1.4       v1.3 压缩 + 两阶段 LoRA 微调合并评估。

  v1.5  （当前版本）在 v1.4 基础上扩展多模型架构支持：
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
                               position_embeddings=None):
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
                                skip_moe_experts=False,
                                verbose=True):
    """
    FR-SVD v1.5 逐层闭环压缩主函数（模型无关版）。

    相比 v1.4：
    - 用 FactorizedLinear 原位替换任意 nn.Linear，不再依赖
      SVD_LlamaAttention / SVD_MistralAttention / SVDOPTDecoderLayer。
    - 通过 _detect_arch_family + get_model_access 统一入口，支持
      LLaMA / Llama-2 / Llama-13b / Llama-30b / Qwen2.5 / Qwen3.5 MoE。
    - MoE 可选 skip_moe_experts 跳过 256 个 experts 的 gate/up/down_proj。
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
    # Qwen 家族在当前超参下更容易出现“数值有限但重建已失真”的层；
    # 这时仅靠 W_err > 1.0 的硬回退太迟，需要更积极地对比 plain truncated SVD。
    tsvd_guard_threshold = 0.30 if arch_family in ('qwen', 'qwen3_moe') else 0.60

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

    print("\n开始逐层 FR-SVD v1.3 压缩...")

    for layer_idx in tqdm(range(len(layers)), desc="FR-SVD v1.3 Compression"):
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

        # ────────── Phase B：Fisher Psi 估计（创新点A）──────────
        psi_decomp = {}
        if use_fisher:
            with torch.enable_grad():
                fisher_result = estimate_kronecker_fisher(
                    layer, teacher_layer, inps_student, inps_teacher,
                    attention_masks, position_ids, model_name, dev,
                    num_samples, lambda_psi=lambda_psi,
                    max_fisher_samples=max_fisher_samples,
                    position_embeddings=pos_embed_cached
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
            tsvd_err = None
            fallback_reason = None

            # 数值安全 + plain truncated SVD 质量护栏
            non_finite = (not torch.isfinite(B_merged).all()) or (not torch.isfinite(A_merged).all())
            should_compare_tsvd = non_finite or weight_err > tsvd_guard_threshold
            if should_compare_tsvd:
                B_tsvd, A_tsvd, tsvd_err = truncated_svd_factorize(W, r)

                use_tsvd = False
                if non_finite or weight_err > 1.0:
                    use_tsvd = True
                    fallback_reason = "numerical_instability"
                elif tsvd_err < weight_err * 0.90:
                    use_tsvd = True
                    fallback_reason = "tsvd_guard"

                if use_tsvd:
                    print(
                        f"  警告：L{layer_idx} {name} 当前重建较差 "
                        f"(W_err={weight_err:.4f}, TSVD={tsvd_err:.4f})，回退到纯截断 SVD"
                    )
                    B_merged = B_tsvd
                    A_merged = A_tsvd
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
                'tsvd_err': tsvd_err,
                'fallback_reason': fallback_reason,
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

            # 写入：原位用 FactorizedLinear 替换当前 Linear（架构无关）
            bias_data = subset[name].bias.data if subset[name].bias is not None else None
            replace_linear_with_factorized(
                layer, name, B_merged, A_merged, bias_data,
                device=dev, dtype=original_dtype
            )

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

        # ────────── Phase E（v1.5 已并入 Phase D）──────────
        # 所有 Linear 已在 Phase D 原位被 FactorizedLinear 替换，layer 结构未变。

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
        description='FR-SVD v1.5：v1.3 压缩 + 两阶段 LoRA 微调 + 多模型支持'
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

    print("=" * 60)
    print("FR-SVD v1.5 配置")
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

        print("\n开始 FR-SVD v1.5 压缩...")
        stats = fr_svd_compress_layerwise(
            args.model, model, calib_data, keep_ratio, args.DEV,
            eps=args.eps, r1_ratio=args.r1_ratio,
            use_fisher=args.use_fisher, max_fisher_samples=args.max_fisher_samples,
            lambda_psi=args.lambda_psi,
            rho=args.rho, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
            fisher_mode=args.fisher_mode,
            skip_moe_experts=args.skip_moe_experts,
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
