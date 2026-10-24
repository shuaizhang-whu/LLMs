#coding:utf8
"""
SAES-SVD v2.1.1: 逐层闭环实现 (参数优化版)

基于v2.1修复：
★ 关键修复: 根据论文Section 5.3设置α∈[0.25, 0.75]，即β∈[0.2, 0.4286]
★ 论文指出: "we set the lower bound to 0.25 and the upper bound to 0.75 in practice"

核心修复：
1. profiling 和压缩合并为逐层闭环流程
2. 每层统计 Δ 时，student model 已经是压缩过上游层的
3. ACES 和 Whitening 使用同一个 L
4. 当层立刻压缩并替换权重
5. ★ 恢复压缩后重新前向传播，保证闭环语义正确

修复内容：
- ★ 恢复第575-588行的重复前向传播（闭环关键！）
- ★ 显存优化：teacher存储fp16，计算时转fp32
- 添加rank的clamp防止为0
- ACES使用隐式投影避免OOM
- 修复设备漂移问题

作者：重构版 v2.1
"""

import os
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

# 添加路径
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

from utils.data_utils import *
from utils.model_utils import *
from evaluater import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer

# ==================== Wandb Configuration ====================
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: wandb not installed.")
    WANDB_AVAILABLE = False
    wandb = None


def compute_H_inv_sqrt(H, device, eps=1e-4):
    """
    使用特征分解计算 H^{-1/2}
    
    Args:
        H: 输入协方差矩阵 (n, n)
        device: 计算设备
        eps: 正则化参数
    
    Returns:
        L: H^{-1/2} (对称矩阵)
    """
    d = H.shape[0]
    
    # ★ 对于大矩阵(>4096)，在CPU上计算特征分解更安全
    if d > 4096:
        H_cpu = H.cpu().float()
        H_reg = H_cpu + eps * torch.eye(d, device='cpu', dtype=torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        L = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T
        return L.to(device)
    else:
        H_reg = H + eps * torch.eye(d, device=device, dtype=H.dtype)
        eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        L = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T
        return L


def aces_beta_select_unified(W, H, Delta, L, rank, beta_min=0.2, beta_max=0.4286):
    """
    SAES-SVD Algorithm 3: ACES Beta Selection (统一版)
    护栏设置为: rho = 0.9, beta_cap = 0.8
    关键修复：
    1. 直接使用传入的 L，不再重新计算！
    2. 使用隐式投影避免OOM（不显式构造P_L和P_R）
    
    Args:
        W: 权重矩阵 (m, n)
        H: 协方差矩阵 (n, n)
        Delta: 差分协方差 (n, n)
        L: 白化矩阵 H^{-1/2}，与压缩阶段使用同一个！
        rank: 目标秩
        beta_min, beta_max: β 搜索范围
    
    Returns:
        beta_star: 最优 β
    """
    device = W.device
    m, n = W.shape
    
    # 确保数据类型一致
    W = W.float()
    H = H.float()
    Delta = Delta.float()
    L = L.float()
    
    # 构造 S = W H L 和 D = W Δ L
    S = W @ H @ L
    D = W @ Delta @ L
    
    # Step 2: 对 S 做 SVD 获取秩-r 主子空间
    U, Sigma, Vt = torch.linalg.svd(S, full_matrices=False)
    U_r = U[:, :rank]
    V_r = Vt[:rank, :].T
    
    # ★ 修复: 使用隐式投影避免OOM，不显式构造P_L和P_R
    # S_perp = (I - U_r U_r^T) S (I - V_r V_r^T)
    #        = S - U_r(U_r^T S) - (S V_r)V_r^T + U_r(U_r^T S V_r)V_r^T
    UrTS = U_r.T @ S
    SVr = S @ V_r
    UrTSVr = U_r.T @ SVr
    S_perp = S - U_r @ UrTS - SVr @ V_r.T + U_r @ UrTSVr @ V_r.T
    
    # D_perp同理
    UrTD = U_r.T @ D
    DVr = D @ V_r
    UrTDVr = U_r.T @ DVr
    D_perp = D - U_r @ UrTD - DVr @ V_r.T + U_r @ UrTDVr @ V_r.T
    
    # 计算系数 a, b, c, A, B, C (论文公式 18)
    a = torch.norm(S_perp, 'fro')**2
    b = torch.sum(S_perp * D_perp)
    c = torch.norm(D_perp, 'fro')**2
    
    A = torch.norm(S, 'fro')**2
    B = torch.sum(S * D)
    C = torch.norm(D, 'fro')**2
    
    # Step 4: 求解二次方程 (公式 19)
    coef_2 = c * B - b * C
    coef_1 = c * A - a * C
    coef_0 = b * A - a * B
    
    candidates = [beta_min, beta_max]
    
    if abs(coef_2.item()) > 1e-10:
        discriminant = coef_1**2 - 4 * coef_2 * coef_0
        if discriminant >= 0:
            sqrt_disc = torch.sqrt(discriminant)
            root1 = (-coef_1 + sqrt_disc) / (2 * coef_2)
            root2 = (-coef_1 - sqrt_disc) / (2 * coef_2)
            candidates.extend([root1.item(), root2.item()])
    
    # Step 5: 评分选择最优 beta
    best_beta = beta_min
    best_score = float('inf')
    
    for beta_candidate in candidates:
        beta = max(beta_min, min(beta_candidate, beta_max))
        
        if beta < 0 or beta >= 1:
            continue
        
        # 能量比目标函数 (公式 17)
        numerator = a + 2 * b * beta + c * beta**2
        denominator = A + 2 * B * beta + C * beta**2 + 1e-10
        score = numerator / denominator
        
        if score < best_score:
            best_score = score
            best_beta = beta
    
    # 应用护栏 (论文建议)
    rho = 0.9
    beta_cap = 0.8
    best_beta = rho * min(best_beta, beta_cap)
    
    return best_beta, {
        'a': a.item(), 'b': b.item(), 'c': c.item(),
        'A': A.item(), 'B': B.item(), 'C': C.item(),
        'best_score': best_score
    }


@torch.no_grad()
def saes_svd_compress_layerwise(model_name, model, calib_loader, ratio, dev, 
                                 eps=1e-4, verbose=True):
    """
    SAES-SVD 逐层闭环压缩
    
    核心流程：
    1. 创建 teacher model (存储fp16，计算fp32)
    2. 逐层处理：
       a. 用【已压缩的 student】前向得到 X
       b. 用【全精度 teacher】前向得到 X^f
       c. 统计 H = X X^T, Δ = (X^f - X) X^T
       d. 计算 L = H^{-1/2}
       e. ACES 选择 β* (使用同一个 L)
       f. 构造 G = W(H + β*Δ)L, SVD 压缩
       g. 立刻替换本层权重
       h. ★ 重新前向传播获取压缩后输出（闭环关键！）
       i. 进入下一层
    
    Args:
        model_name: 模型名称
        model: 待压缩模型 (将被原地修改)
        calib_loader: 校准数据
        ratio: 保留比例 (如 0.8 表示保留 80%)
        dev: 设备
        eps: 正则化参数
        verbose: 是否打印详细信息
    
    Returns:
        compression_stats: 压缩统计信息
    """
    print("=" * 60)
    print("🚀 SAES-SVD v2.1.1: 逐层闭环压缩 (参数优化版)")
    print("=" * 60)
    print(f"📊 压缩比例: {ratio} (保留 {ratio*100:.1f}%)")
    print(f"📊 正则化参数 eps: {eps}")
    print("=" * 60)
    
    # ========== Step 1: 创建 teacher model ==========
    # ★ 显存优化: teacher存储fp16，计算时转fp32
    print("\n📚 Loading teacher model (fp16 storage, fp32 compute)...")
    from transformers import AutoModelForCausalLM
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model.config._name_or_path,
        torch_dtype=torch.float16,  # ★ 存储用fp16节省显存
        device_map="cpu"
    )
    teacher_model.eval()
    
    # ========== Step 2: 获取层结构 ==========
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
    
    # ========== Step 3: 捕获第 0 层输入 ==========
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    num_samples = len(calib_loader)
    
    # ★ 显存优化: 所有大型缓存都放CPU，用fp16存储
    # Student 输入缓存 (放CPU！)
    inps_student = torch.zeros(
        (num_samples, model.seqlen, model.config.hidden_size), 
        dtype=torch.float16, device='cpu'  # ★ 改为CPU + fp16
    )
    # Teacher 输入缓存 (fp16存储节省显存)
    inps_teacher = torch.zeros(
        (num_samples, model.seqlen, model.config.hidden_size), 
        dtype=torch.float16, device='cpu'
    )
    
    cache = {'i': 0}
    attention_masks_list = []
    position_ids_list = []
    
    class StudentCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps_student[cache['i']] = inp.detach().half().cpu()  # ★ 存到CPU+fp16
            # 为每个样本收集 attention_mask 和 position_ids
            attention_masks_list.append(kwargs['attention_mask'].detach().cpu())  # ★ 也存CPU
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
            inps_teacher[self.idx] = inp.detach().half().cpu()  # ★ 存储为fp16
            self.idx += 1
            raise ValueError
    
    # 捕获输入
    layers[0] = StudentCatcher(layers[0])
    teacher_layers[0] = TeacherCatcher(teacher_layers[0])
    
    print("📊 Collecting layer-0 inputs...")
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
        
        try:
            teacher_model.to(dev)
            teacher_model(**batch)
            teacher_model.to("cpu")
        except ValueError:
            pass
    
    # 恢复第 0 层
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    teacher_layers[0] = teacher_layers[0].module
    
    # 清理嵌入层
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    
    torch.cuda.empty_cache()
    
    # 输出缓存 (★ 全部放CPU)
    outs_student = torch.zeros_like(inps_student)  # CPU + fp16
    outs_teacher = torch.zeros(
        (num_samples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16, device='cpu'  # ★ 存储为fp16
    )
    # 将列表转换为张量或保持为列表（每个元素可能形状不同）
    attention_masks = attention_masks_list
    position_ids = position_ids_list if position_ids_list and position_ids_list[0] is not None else None
    
    # ========== Step 4: 逐层闭环处理 ==========
    compression_stats = []
    
    print("\n🔧 Starting layer-wise compression with cumulative error tracking...")
    
    for layer_idx in tqdm(range(len(layers)), desc="SAES-SVD Compression"):
        layer_stats = {'layer': layer_idx, 'sublayers': {}}
        
        # 将当前层移到 GPU (★ 转fp32计算)
        layer = layers[layer_idx].to(dev).float()
        teacher_layer = teacher_layers[layer_idx].to(dev)  # teacher保持fp16
        
        subset = find_layers(layer)
        teacher_subset = find_layers(teacher_layer)
        
        # ---------- 4.1 统计 H 和 Δ ----------
        # 为每个子层初始化统计量
        for name in subset:
            subset[name].H_matrix = 0
            subset[name].inp_cache = []
            teacher_subset[name].inp_cache = []
        
        # 定义钩子
        def make_hook_student(module_name):
            def hook(module, input, output):
                inp = input[0].detach().float()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                # 累加 H = X X^T
                H_batch = torch.matmul(inp.transpose(1, 2), inp)
                module.H_matrix = module.H_matrix + torch.sum(H_batch, dim=0)
                # 缓存输入用于计算 Δ
                module.inp_cache.append(inp.half().cpu())  # ★ 存储为fp16
            return hook
        
        def make_hook_teacher(module_name):
            def hook(module, input, output):
                inp = input[0].detach()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                module.inp_cache.append(inp.half().cpu())  # ★ 存储为fp16
            return hook
        
        # 注册钩子
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_hook_student(name)))
            handles.append(teacher_subset[name].register_forward_hook(make_hook_teacher(name)))
        
        # 前向传播收集统计量（用压缩前的layer）
        for j in range(num_samples):
            # ★ 从CPU加载到GPU，转fp32计算
            inp_j = inps_student[j].unsqueeze(0).to(dev).float()
            attn_mask_j = attention_masks[j].to(dev)
            pos_ids_j = position_ids[j].to(dev) if position_ids is not None else None
            
            if "opt" not in model_name:
                out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)[0]
                outs_student[j] = out_j.half().cpu()  # ★ 存回CPU fp16
                
                # Teacher 前向（fp16加载，转fp32计算）
                teacher_inp = inps_teacher[j].unsqueeze(0).to(dev).float()
                teacher_out = teacher_layer.float()(
                    teacher_inp,
                    attention_mask=attn_mask_j,
                    position_ids=pos_ids_j
                )[0]
                outs_teacher[j] = teacher_out.half().cpu()
                teacher_layer.half()  # 转回fp16
            else:
                out_j = layer(inp_j, attention_mask=attn_mask_j)[0]
                outs_student[j] = out_j.half().cpu()
                
                teacher_inp = inps_teacher[j].unsqueeze(0).to(dev).float()
                teacher_out = teacher_layer.float()(teacher_inp, attention_mask=attn_mask_j)[0]
                outs_teacher[j] = teacher_out.half().cpu()
                teacher_layer.half()
            
            # ★ 及时清理GPU临时变量
            del inp_j, out_j, teacher_inp, teacher_out
            if j % 32 == 0:
                torch.cuda.empty_cache()
        
        # 移除钩子
        for h in handles:
            h.remove()
        
        # 计算 Δ = (X^f - X) X^T
        for name in subset:
            subset[name].Delta_matrix = 0
            for idx in range(len(subset[name].inp_cache)):
                # ★ 从fp16加载并转fp32计算
                X = subset[name].inp_cache[idx].to(dev).float()  # Student 输入
                X_f = teacher_subset[name].inp_cache[idx].to(dev).float()  # Teacher 输入
                diff = X_f - X  # 累积误差！
                Delta_batch = torch.matmul(diff.transpose(1, 2), X)
                subset[name].Delta_matrix = subset[name].Delta_matrix + torch.sum(Delta_batch, dim=0)
                del X, X_f, diff, Delta_batch
            
            # 移到 CPU 释放显存
            subset[name].H_matrix = subset[name].H_matrix.cpu()
            subset[name].Delta_matrix = subset[name].Delta_matrix.cpu()
            del subset[name].inp_cache
            del teacher_subset[name].inp_cache
        
        torch.cuda.empty_cache()
        
        # ---------- 4.2 创建 SVD 替身模块 ----------
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
        
        # ---------- 4.3 逐个子层压缩 ----------
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            original_dtype = subset[name].weight.data.dtype
            
            H = subset[name].H_matrix.float().to(dev)
            Delta = subset[name].Delta_matrix.float().to(dev)
            
            # 计算截断秩
            m, n = W.shape
            num_s_after_trunc = int(m * n * ratio / (m + n))
            # ★ 修复: 防止rank为0或超过矩阵秩
            num_s_after_trunc = max(1, min(num_s_after_trunc, min(m, n)))
            
            # ★ 关键：计算 L = H^{-1/2}，后续 ACES 和压缩都用这个 L
            L = compute_H_inv_sqrt(H, dev, eps=eps)
            
            # ★ ACES 选择 β（使用同一个 L）
            # ★ 论文Section 5.3推荐: α∈[0.25, 0.75] → β∈[0.2, 0.4286]
            beta, aces_info = aces_beta_select_unified(
                W, H, Delta, L, num_s_after_trunc,
                beta_min=0.2, beta_max=0.4286
            )
            
            # ★ 构造白化目标矩阵 G = W(H + βΔ)L
            G = W @ (H + beta * Delta) @ L
            
            # 截断 SVD
            U, S, Vt = torch.linalg.svd(G, full_matrices=False)
            truc_u = U[:, :num_s_after_trunc]
            truc_s = S[:num_s_after_trunc]
            Vt_r = Vt[:num_s_after_trunc, :]
            
            truc_sigma = torch.diag(truc_s)
            sqrtSigma = torch.sqrt(truc_sigma)
            
            # 计算低秩因子
            # A = U_r Σ^{1/2}
            # B = Σ^{1/2} V_r^T L
            # ★ 修复: 先保持在dev上，避免设备漂移
            svd_u = torch.matmul(truc_u, sqrtSigma).to(original_dtype)
            svd_v = torch.matmul(sqrtSigma, Vt_r @ L).to(original_dtype)
            
            # 计算重建误差(权重空间 - 仅供参考，不是优化目标)
            W_reconstructed = svd_u.float() @ svd_v.float()
            weight_recon_error = (torch.norm(W - W_reconstructed) / torch.norm(W)).item()
            
            # ★ 计算更有意义的指标: G矩阵的truncation error
            G_reconstructed = truc_u @ torch.diag(truc_s) @ Vt_r
            G_trunc_error = (torch.norm(G - G_reconstructed) / torch.norm(G)).item()
            
            # ★ 计算保留能量比 (retained energy ratio)
            total_energy = torch.sum(S**2).item()
            retained_energy = torch.sum(truc_s**2).item()
            energy_ratio = retained_energy / (total_energy + 1e-10)
            
            # 计算 Delta 范数（用于诊断）
            delta_norm = torch.norm(Delta).item()
            h_norm = torch.norm(H).item()
            
            # 记录统计
            layer_stats['sublayers'][name] = {
                'beta': beta,
                'rank': num_s_after_trunc,
                'G_trunc_error': G_trunc_error,
                'energy_ratio': energy_ratio,
                'delta_norm': delta_norm,
                'h_norm': h_norm,
                'delta_ratio': delta_norm / (h_norm + 1e-10),
                'aces_info': aces_info
            }
            
            # 打印信息
            if verbose and (layer_idx == 0 or layer_idx == len(layers) - 1 or layer_idx % 8 == 0):
                print(f"  Layer {layer_idx}, {name}: β={beta:.4f}, rank={num_s_after_trunc}, "
                      f"G_err={G_trunc_error:.4f}, E_kept={energy_ratio:.4f}, Δ/H={delta_norm/h_norm:.4f}")
            
            # ★ 立刻写入 SVD 替身模块 (使用copy_避免设备漂移)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data.copy_(svd_u)
                    svd_decoder.self_attn.q_v_proj.weight.data.copy_(svd_v)
                    svd_decoder.self_attn.q_u_proj.bias.data.copy_(layer.self_attn.q_proj.bias.data)
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data.copy_(svd_u)
                    svd_decoder.self_attn.k_v_proj.weight.data.copy_(svd_v)
                    svd_decoder.self_attn.k_u_proj.bias.data.copy_(layer.self.attn.k_proj.bias.data)
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data.copy_(svd_u)
                    svd_decoder.self_attn.v_v_proj.weight.data.copy_(svd_v)
                    svd_decoder.self_attn.v_u_proj.bias.data.copy_(layer.self.attn.v_proj.bias.data)
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data.copy_(svd_u)
                    svd_decoder.self_attn.out_v_proj.weight.data.copy_(svd_v)
                    svd_decoder.self_attn.out_u_proj.bias.data.copy_(layer.self.attn.out_proj.bias.data)
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data.copy_(svd_u)
                    svd_decoder.fc1_v_proj.weight.data.copy_(svd_v)
                    svd_decoder.fc1_u_proj.bias.data.copy_(layer.fc1.bias.data)
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data.copy_(svd_u)
                    svd_decoder.fc2_v_proj.weight.data.copy_(svd_v)
                    svd_decoder.fc2_u_proj.bias.data.copy_(layer.fc2.bias.data)
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data.copy_(svd_u)
                    svd_attn.q_v_proj.weight.data.copy_(svd_v)
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data.copy_(svd_u)
                    svd_attn.k_v_proj.weight.data.copy_(svd_v)
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data.copy_(svd_u)
                    svd_attn.v_v_proj.weight.data.copy_(svd_v)
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data.copy_(svd_u)
                    svd_attn.o_v_proj.weight.data.copy_(svd_v)
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data.copy_(svd_u)
                    svd_mlp.gate_v_proj.weight.data.copy_(svd_v)
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data.copy_(svd_u)
                    svd_mlp.down_v_proj.weight.data.copy_(svd_v)
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data.copy_(svd_u)
                    svd_mlp.up_v_proj.weight.data.copy_(svd_v)
            
            # 清理
            del W, H, Delta, L, G, U, S, Vt, svd_u, svd_v, W_reconstructed
            del subset[name].H_matrix, subset[name].Delta_matrix
            torch.cuda.empty_cache()
        
        # ---------- 4.4 ★ 立刻替换本层权重 ----------
        if 'opt' in model_name:
            layers[layer_idx] = svd_decoder
        else:
            layer.self_attn = svd_attn
            layer.mlp = svd_mlp
        
        # 计算层级统计
        avg_beta = sum(s['beta'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        avg_delta_ratio = sum(s['delta_ratio'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        layer_stats['avg_beta'] = avg_beta
        layer_stats['avg_delta_ratio'] = avg_delta_ratio
        
        if verbose:
            print(f"Layer {layer_idx}: avg_β={avg_beta:.4f}, avg_Δ/H={avg_delta_ratio:.4f}")
        
        compression_stats.append(layer_stats)
        
        # ========== ★★★ 关键修复：重新前向传播获取压缩后输出 ★★★ ==========
        # 闭环语义：下一层的输入 = 当前层【压缩后】的实际输出
        # 这样才能正确累积压缩误差到 Δ 中
        for j in range(num_samples):
            inp_j = inps_student[j].unsqueeze(0).to(dev).float()  # ★ CPU->GPU
            attn_mask_j = attention_masks[j].to(dev)
            pos_ids_j = position_ids[j].to(dev) if position_ids is not None else None
            
            if "opt" not in model_name:
                out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)[0]
            else:
                out_j = layers[layer_idx](inp_j, attention_mask=attn_mask_j)[0]
            
            outs_student[j] = out_j.half().cpu()  # ★ 存回CPU fp16
            del inp_j, out_j
        
        torch.cuda.empty_cache()
        
        # 更新输入为当前层【压缩后】的输出
        inps_student = outs_student.clone()
        inps_teacher = outs_teacher.clone()  # Teacher 输出不变（全精度参考）
        
        # 移回 CPU (★ 转回fp16节省内存)
        layers[layer_idx] = layers[layer_idx].half().cpu()
        teacher_layers[layer_idx] = teacher_layers[layer_idx].cpu()
        torch.cuda.empty_cache()
    
    # 清理 teacher model
    del teacher_model
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("✅ SAES-SVD v2.1.1 compression complete!")
    print("=" * 60)
    
    # 打印总结
    all_betas = [s['avg_beta'] for s in compression_stats]
    all_delta_ratios = [s['avg_delta_ratio'] for s in compression_stats]
    print(f"📊 Average β across all layers: {sum(all_betas)/len(all_betas):.4f}")
    print(f"📊 Average Δ/H ratio: {sum(all_delta_ratios)/len(all_delta_ratios):.4f}")
    print(f"📊 β range: [{min(all_betas):.4f}, {max(all_betas):.4f}]")
    
    return compression_stats


# ==================== Main ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAES-SVD v2.1: Layer-wise Closed-loop Compression (Memory Optimized)')
    
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Compression ratio (0.2 = remove 20%)')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Calibration dataset')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--DEV', type=str, default='cuda:0', help='Device')
    parser.add_argument('--save_path', type=str, default='./saes_svd_v2.1_output', help='Save path')
    parser.add_argument('--result_dir', type=str, default=None, help='Result directory')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--eps', type=float, default=1e-4, help='Regularization epsilon')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--model_path', type=str, default=None, help='Path to compressed model for evaluation')
    
    args = parser.parse_args()

    # 在 args 解析后添加
    if args.result_dir:
        args.save_path = args.result_dir
    
    # 转换压缩比
    keep_ratio = 1 - args.ratio  # 0.2 -> 0.8 (保留 80%)
    
    print(f"=" * 60)
    print(f"SAES-SVD v2.1.1 Configuration (Parameter Optimized)")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Compression ratio: {args.ratio} (keeping {keep_ratio*100:.1f}%)")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.nsamples}")
    print(f"Device: {args.DEV}")
    print(f"Epsilon: {args.eps}")
    print(f"=" * 60)
    
    if args.eval_only and args.model_path:
        # 仅评估模式
        print("\n📊 Evaluation only mode...")
        model, tokenizer = get_model_from_local(args.model_path)
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        
        print("\n🔍 Running perplexity evaluation...")
        ppl_eval(model, tokenizer, datasets=['wikitext2'], 
                 model_seq_len=args.model_seq_len, batch_size=4, device=args.DEV)
    else:
        # 压缩模式
        print("\n📦 Loading model...")
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        # ★ 显存优化: student也用fp16存储，计算时转fp32
        model = model.half()  # fp16存储
        
        print("\n📊 Loading calibration data...")
        calib_data = get_calib_train_data(
            args.dataset, tokenizer, args.nsamples, seqlen=args.model_seq_len
        )
        
        print("\n🔧 Starting compression...")
        stats = saes_svd_compress_layerwise(
            args.model, model, calib_data, keep_ratio, args.DEV,
            eps=args.eps, verbose=True
        )
        
        # 保存模型
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        save_file = os.path.join(
            args.save_path,
            f"{args.model.replace('/', '_').replace('-', '_')}_saes_svd_v2.1.1_{keep_ratio}.pt"
        )
        print(f"\n💾 Saving compressed model to: {save_file}")
        torch.save({'model': model, 'tokenizer': tokenizer, 'stats': stats}, save_file)
        
        # 评估
        print("\n🔍 Running perplexity evaluation...")
        model = model.float().to(args.DEV)  # ★ 评估时转fp32
        ppl_eval(model, tokenizer, datasets=['wikitext2'],
                 model_seq_len=args.model_seq_len, batch_size=4, device=args.DEV)
        
        print("\n✅ All done!")
