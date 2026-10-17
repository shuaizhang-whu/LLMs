#!/usr/bin/env python3
"""
FR-SVD 诊断脚本：隔离 Stage 1 vs Stage 2 的影响

核心问题：FR-SVD (PPL=22.61) 为什么比 SAES-SVD (PPL=8.05) 差 3 倍？

本脚本对 Layer 0 的一个子层做如下对比：
  ① SAES-SVD：单阶段 CEALC (rank=r)
  ② FR-SVD Stage 1 only：CEALC (rank=r, 即 r1_ratio=1.0)
  ③ FR-SVD 完整两阶段：Stage 1 (rank=r1) + Stage 2 (rank=r2)

比较指标：
  - G_err: 白化空间截断误差 ||G - [G]_r||/||G||
  - W_err: 权重重建误差 ||W - W'||/||W||
  - output_err: 输出误差 ||W'X - WX||_F / ||WX||_F
  - output_align_err: 教师对齐误差 ||W'X - WX^f||_F / ||WX^f||_F

如果 ②≈① 且 ③>>①，则问题在 Stage 2。
如果 ②>>①，则问题在 Stage 1 的 G 构造差异。
"""

import torch
import sys
import os
import argparse

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FR_SVD import compute_H_decomposition, fisher_beta_select


def saes_svd_compress(W, H_raw, Delta, rank, eps=1e-4, rho=0.95,
                      alpha_min=0.4, alpha_max=0.6):
    """SAES-SVD 单阶段压缩（复现 SAES-SVD_v2.1.4 的核心逻辑）"""
    dev = W.device
    m, n = W.shape

    # 计算 L = H^{-1/2}（与 SAES-SVD 完全一致的方式）
    d = H_raw.shape[0]
    H_reg = H_raw + eps * torch.eye(d, device=dev, dtype=torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    L = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T

    # ACES beta selection（SAES-SVD 风格：内部构造 S, D）
    S = W @ H_raw @ L
    D = W @ Delta @ L

    # 简化 ACES（与 FR-SVD 的 fisher_beta_select 相同算法）
    beta, info = fisher_beta_select(S, D, rank,
                                     alpha_min=alpha_min, alpha_max=alpha_max,
                                     rho=rho)

    # 构造 G = W(H + βΔ)L
    G = W @ (H_raw + beta * Delta) @ L

    # SVD 截断
    U, Sigma, Vt = torch.linalg.svd(G, full_matrices=False)
    U_r = U[:, :rank]
    S_r = Sigma[:rank]
    Vt_r = Vt[:rank, :]

    # G_err
    total_energy = torch.sum(Sigma ** 2).item()
    retained_energy = torch.sum(S_r ** 2).item()
    G_err = torch.norm(G - U_r @ torch.diag(S_r) @ Vt_r).item() / torch.norm(G).item()

    # 低秩因子（SAES-SVD 风格：A = √Σ V^T L）
    sqrtSigma = torch.diag(torch.sqrt(S_r))
    B = U_r @ sqrtSigma
    A = sqrtSigma @ Vt_r @ L

    return B, A, beta, G_err, info


def fr_svd_stage1_only(W, H_nodamp, Delta, rank, eps=1e-4, rho=0.95,
                        alpha_min=0.4, alpha_max=0.6):
    """FR-SVD Stage 1 only (r1_ratio=1.0)"""
    dev = W.device
    m, n = W.shape

    Q, Lambda, H_sqrt, H_inv_sqrt = compute_H_decomposition(H_nodamp, dev, eps)

    S = W @ H_sqrt
    D = W @ Delta @ H_inv_sqrt
    beta, info = fisher_beta_select(S, D, rank,
                                     alpha_min=alpha_min, alpha_max=alpha_max,
                                     rho=rho)

    G = S + beta * D
    U, Sigma, Vt = torch.linalg.svd(G, full_matrices=False)
    U_r = U[:, :rank]
    S_r = Sigma[:rank]
    Vt_r = Vt[:rank, :]

    G_err = torch.norm(G - U_r @ torch.diag(S_r) @ Vt_r).item() / torch.norm(G).item()

    sqrt_s = torch.sqrt(S_r)
    B = U_r * sqrt_s.unsqueeze(0)
    A = (sqrt_s.unsqueeze(1) * Vt_r) @ H_inv_sqrt

    return B, A, beta, G_err, info


def fr_svd_two_stage(W, H_nodamp, Delta, r, r1_ratio=0.8, eps=1e-4, rho=0.95,
                      alpha_min=0.4, alpha_max=0.6):
    """FR-SVD 完整两阶段压缩"""
    dev = W.device
    m, n = W.shape
    r1 = max(1, int(r * r1_ratio))
    r2 = r - r1

    Q, Lambda, H_sqrt, H_inv_sqrt = compute_H_decomposition(H_nodamp, dev, eps)

    # Stage 1
    S = W @ H_sqrt
    D = W @ Delta @ H_inv_sqrt
    beta, info = fisher_beta_select(S, D, r1,
                                     alpha_min=alpha_min, alpha_max=alpha_max,
                                     rho=rho)

    G = S + beta * D
    U, Sigma, Vt = torch.linalg.svd(G, full_matrices=False)
    U_r1 = U[:, :r1]
    S_r1 = Sigma[:r1]
    Vt_r1 = Vt[:r1, :]

    G_err_s1 = torch.norm(G - U_r1 @ torch.diag(S_r1) @ Vt_r1).item() / torch.norm(G).item()
    energy_s1 = torch.sum(S_r1 ** 2).item() / (torch.sum(Sigma ** 2).item() + 1e-10)

    sqrt_s1 = torch.sqrt(S_r1)
    B1 = U_r1 * sqrt_s1.unsqueeze(0)
    A1 = (sqrt_s1.unsqueeze(1) * Vt_r1) @ H_inv_sqrt

    # W_err Stage 1 only
    W_recon_s1 = B1 @ A1
    W_err_s1 = (torch.norm(W - W_recon_s1) / torch.norm(W)).item()

    # Stage 2
    if r2 > 0:
        EXt = W @ (H_nodamp + Delta) - B1 @ A1 @ H_nodamp
        inv_sqrt_lam = 1.0 / torch.sqrt(Lambda)
        DeltaW_white = EXt @ (Q * inv_sqrt_lam.unsqueeze(0))

        U2, S2, Vt2 = torch.linalg.svd(DeltaW_white, full_matrices=False)
        U_r2 = U2[:, :r2]
        S_r2 = S2[:r2]
        Vt_r2 = Vt2[:r2, :]
        sqrt_s2 = torch.sqrt(S_r2)
        B2 = U_r2 * sqrt_s2.unsqueeze(0)
        A2 = sqrt_s2.unsqueeze(1) * ((Vt_r2 * inv_sqrt_lam.unsqueeze(0)) @ Q.T)
    else:
        B2 = torch.empty(m, 0, device=dev, dtype=torch.float32)
        A2 = torch.empty(0, n, device=dev, dtype=torch.float32)

    B_merged = torch.cat([B1, B2], dim=1)
    A_merged = torch.cat([A1, A2], dim=0)

    return B_merged, A_merged, beta, G_err_s1, energy_s1, W_err_s1, info, r1, r2


def evaluate_compression(W, B, A, X, X_f, label=""):
    """计算各种误差指标"""
    W_recon = B.float() @ A.float()
    W_err = (torch.norm(W - W_recon) / torch.norm(W)).item()

    # 输出误差（学生空间）
    Y_orig = W @ X       # (m, T)
    Y_comp = W_recon @ X  # (m, T)
    output_err = (torch.norm(Y_comp - Y_orig) / torch.norm(Y_orig)).item()

    # 教师对齐误差
    Y_teacher = W @ X_f
    align_err = (torch.norm(Y_comp - Y_teacher) / torch.norm(Y_teacher)).item()

    # fp16 误差
    B_fp16 = B.half()
    A_fp16 = A.half()
    W_recon_fp16 = B_fp16.float() @ A_fp16.float()
    W_err_fp16 = (torch.norm(W - W_recon_fp16) / torch.norm(W)).item()
    Y_comp_fp16 = W_recon_fp16 @ X
    output_err_fp16 = (torch.norm(Y_comp_fp16 - Y_orig) / torch.norm(Y_orig)).item()

    return {
        'W_err_fp32': W_err,
        'W_err_fp16': W_err_fp16,
        'output_err_fp32': output_err,
        'output_err_fp16': output_err_fp16,
        'align_err_fp32': align_err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--DEV', type=str, default='cuda:0')
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--rho', type=float, default=0.95)
    parser.add_argument('--sublayer', type=str, default='v_proj',
                        help='测试哪个子层 (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj)')
    args = parser.parse_args()

    dev = args.DEV
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("FR-SVD 诊断：隔离 Stage 1 vs Stage 2")
    print("=" * 70)

    # 加载模型
    from utils.model_utils import get_model_from_huggingface, find_layers
    from utils.data_utils import get_calib_train_data

    print("加载模型...")
    model, tokenizer = get_model_from_huggingface(args.model)
    model = model.eval().half()
    model_name = args.model.lower()

    # 加载校准数据
    print("加载校准数据...")
    trainloader = get_calib_train_data(
        'wikitext2', tokenizer, args.nsamples, seqlen=2048
    )

    # 获取层列表
    if "llama" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    # 收集 Layer 0 的输入激活
    print("收集 Layer 0 输入激活...")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # 收集输入
    inps = []
    attention_masks = []
    position_ids = []

    class InputCatcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp.detach().cpu())
            if 'attention_mask' in kwargs:
                attention_masks.append(kwargs['attention_mask'].detach().cpu())
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                position_ids.append(kwargs['position_ids'].detach().cpu())
            raise ValueError("stop")

    layers[0] = InputCatcher(layers[0])
    for batch in trainloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.config.use_cache = use_cache

    num_samples = len(inps)
    print(f"  收集了 {num_samples} 个样本的输入")

    # 获取 Layer 0 的子层
    layer = layers[0].to(dev).float()
    subset = find_layers(layer)

    target_name = None
    for name in subset:
        if args.sublayer in name:
            target_name = name
            break

    if target_name is None:
        print(f"错误：找不到子层 '{args.sublayer}'，可选: {list(subset.keys())}")
        return

    print(f"\n目标子层: {target_name}")
    W = subset[target_name].weight.data.float().to(dev)
    m, n = W.shape
    r = int(m * n * args.ratio / (m + n))
    r = max(1, min(r, min(m, n)))
    print(f"  W 形状: {m} × {n}, 目标秩: {r}")

    # 收集 H 和输入激活
    print("收集 H 矩阵和输入激活...")
    H_accum = 0
    all_X = []

    def hook_fn(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        nonlocal H_accum
        H_batch = torch.matmul(inp.transpose(1, 2), inp)
        H_accum = H_accum + torch.sum(H_batch, dim=0)
        all_X.append(inp.cpu())

    handle = subset[target_name].register_forward_hook(hook_fn)

    for j in range(num_samples):
        inp_j = inps[j].unsqueeze(0).to(dev).float() if inps[j].dim() == 3 else inps[j].to(dev).float()
        attn_mask_j = attention_masks[j].to(dev)
        pos_ids_j = position_ids[j].to(dev) if position_ids else None

        if "opt" not in model_name:
            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
        else:
            layer(inp_j, attention_mask=attn_mask_j)

    handle.remove()

    H = H_accum.float().to(dev)
    # Layer 0 没有累积误差，Delta = 0
    Delta = torch.zeros_like(H)

    # 拼接所有输入激活用于输出误差评估
    # 取前 4 个样本避免 OOM
    X_eval_list = []
    for j in range(min(4, len(all_X))):
        x = all_X[j].to(dev).float()
        X_eval_list.append(x.reshape(-1, n).T)  # (n, T_per_sample)
    X_eval = torch.cat(X_eval_list, dim=1)  # (n, total_T)
    X_f = X_eval  # Layer 0 教师和学生输入相同

    print(f"  H 形状: {H.shape}, 条件数指标: max_eig/min_eig = ", end="")
    eigs = torch.linalg.eigvalsh(H + args.eps * torch.eye(n, device=dev))
    print(f"{eigs[-1].item():.2e} / {eigs[0].item():.2e} = {(eigs[-1]/eigs[0]).item():.2e}")

    # ======= 对比压缩 =======
    print("\n" + "=" * 70)
    print("开始对比压缩")
    print("=" * 70)

    # ① SAES-SVD（基线）
    print("\n[1] SAES-SVD (单阶段, rank={})".format(r))
    B_saes, A_saes, beta_saes, G_err_saes, _ = saes_svd_compress(
        W, H, Delta, r, eps=args.eps, rho=args.rho)
    metrics_saes = evaluate_compression(W, B_saes, A_saes, X_eval, X_f, "SAES-SVD")
    print(f"  β={beta_saes:.4f}, G_err={G_err_saes:.6f}")
    print(f"  W_err(fp32)={metrics_saes['W_err_fp32']:.4f}, W_err(fp16)={metrics_saes['W_err_fp16']:.4f}")
    print(f"  output_err(fp32)={metrics_saes['output_err_fp32']:.6f}, output_err(fp16)={metrics_saes['output_err_fp16']:.6f}")
    print(f"  align_err(fp32)={metrics_saes['align_err_fp32']:.6f}")

    # ② FR-SVD Stage 1 only (r1_ratio=1.0)
    print("\n[2] FR-SVD Stage 1 only (rank={}, r1_ratio=1.0)".format(r))
    B_s1, A_s1, beta_s1, G_err_s1, _ = fr_svd_stage1_only(
        W, H, Delta, r, eps=args.eps, rho=args.rho)
    metrics_s1 = evaluate_compression(W, B_s1, A_s1, X_eval, X_f, "FR Stage1")
    print(f"  β={beta_s1:.4f}, G_err={G_err_s1:.6f}")
    print(f"  W_err(fp32)={metrics_s1['W_err_fp32']:.4f}, W_err(fp16)={metrics_s1['W_err_fp16']:.4f}")
    print(f"  output_err(fp32)={metrics_s1['output_err_fp32']:.6f}, output_err(fp16)={metrics_s1['output_err_fp16']:.6f}")
    print(f"  align_err(fp32)={metrics_s1['align_err_fp32']:.6f}")

    # ③ FR-SVD 两阶段 (r1_ratio=0.8)
    print("\n[3] FR-SVD 两阶段 (total_rank={}, r1_ratio=0.8)".format(r))
    (B_fr, A_fr, beta_fr, G_err_fr_s1, energy_s1, W_err_s1,
     info_fr, r1, r2) = fr_svd_two_stage(
        W, H, Delta, r, r1_ratio=0.8, eps=args.eps, rho=args.rho)
    metrics_fr = evaluate_compression(W, B_fr, A_fr, X_eval, X_f, "FR 2-stage")
    print(f"  β={beta_fr:.4f}, r1={r1}, r2={r2}")
    print(f"  Stage1: G_err={G_err_fr_s1:.6f}, E_kept={energy_s1:.4f}, W_err(s1 only)={W_err_s1:.4f}")
    print(f"  Combined: W_err(fp32)={metrics_fr['W_err_fp32']:.4f}, W_err(fp16)={metrics_fr['W_err_fp16']:.4f}")
    print(f"  output_err(fp32)={metrics_fr['output_err_fp32']:.6f}, output_err(fp16)={metrics_fr['output_err_fp16']:.6f}")
    print(f"  align_err(fp32)={metrics_fr['align_err_fp32']:.6f}")

    # ④ FR-SVD 两阶段 (r1_ratio=0.5, 极端测试)
    print("\n[4] FR-SVD 两阶段 (total_rank={}, r1_ratio=0.5, 极端)".format(r))
    (B_fr5, A_fr5, beta_fr5, G_err_fr5, energy_s5, W_err_s5,
     info_fr5, r1_5, r2_5) = fr_svd_two_stage(
        W, H, Delta, r, r1_ratio=0.5, eps=args.eps, rho=args.rho)
    metrics_fr5 = evaluate_compression(W, B_fr5, A_fr5, X_eval, X_f, "FR 2-stage 0.5")
    print(f"  β={beta_fr5:.4f}, r1={r1_5}, r2={r2_5}")
    print(f"  Stage1: G_err={G_err_fr5:.6f}, E_kept={energy_s5:.4f}, W_err(s1 only)={W_err_s5:.4f}")
    print(f"  Combined: W_err(fp32)={metrics_fr5['W_err_fp32']:.4f}, W_err(fp16)={metrics_fr5['W_err_fp16']:.4f}")
    print(f"  output_err(fp32)={metrics_fr5['output_err_fp32']:.6f}, output_err(fp16)={metrics_fr5['output_err_fp16']:.6f}")

    # ⑤ 纯截断 SVD（无白化）作为下界
    print("\n[5] 纯截断 SVD（无白化，下界参考）")
    U_w, S_w, Vt_w = torch.linalg.svd(W, full_matrices=False)
    sqrt_sw = torch.sqrt(S_w[:r])
    B_vanilla = U_w[:, :r] * sqrt_sw.unsqueeze(0)
    A_vanilla = sqrt_sw.unsqueeze(1) * Vt_w[:r, :]
    metrics_vanilla = evaluate_compression(W, B_vanilla, A_vanilla, X_eval, X_f, "Vanilla SVD")
    print(f"  W_err(fp32)={metrics_vanilla['W_err_fp32']:.4f}")
    print(f"  output_err(fp32)={metrics_vanilla['output_err_fp32']:.6f}")

    # ======= 汇总 =======
    print("\n" + "=" * 70)
    print("汇总比较")
    print("=" * 70)
    print(f"{'方法':<30} {'W_err(fp32)':<14} {'W_err(fp16)':<14} {'out_err(fp32)':<14} {'out_err(fp16)':<14}")
    print("-" * 86)

    results = [
        ("SAES-SVD (baseline)", metrics_saes),
        ("FR Stage1 (r1_ratio=1.0)", metrics_s1),
        ("FR 2-stage (r1_ratio=0.8)", metrics_fr),
        ("FR 2-stage (r1_ratio=0.5)", metrics_fr5),
        ("Vanilla SVD (lower bound)", metrics_vanilla),
    ]
    for name, m in results:
        print(f"{name:<30} {m['W_err_fp32']:<14.6f} {m['W_err_fp16']:<14.6f} "
              f"{m['output_err_fp32']:<14.6f} {m['output_err_fp16']:<14.6f}")

    # ======= 差异分析 =======
    print("\n" + "=" * 70)
    print("差异分析")
    print("=" * 70)

    # W' 矩阵之间的差异
    W_saes = B_saes @ A_saes
    W_s1 = B_s1 @ A_s1
    W_fr = B_fr @ A_fr

    diff_s1_saes = (torch.norm(W_s1 - W_saes) / torch.norm(W_saes)).item()
    diff_fr_saes = (torch.norm(W_fr - W_saes) / torch.norm(W_saes)).item()
    diff_fr_s1 = (torch.norm(W_fr - W_s1) / torch.norm(W_s1)).item()

    print(f"||W'_FR_S1 - W'_SAES|| / ||W'_SAES|| = {diff_s1_saes:.6f}  (Stage1 vs SAES差异)")
    print(f"||W'_FR_2s - W'_SAES|| / ||W'_SAES|| = {diff_fr_saes:.6f}  (两阶段 vs SAES差异)")
    print(f"||W'_FR_2s - W'_FR_S1|| / ||W'_FR_S1|| = {diff_fr_s1:.6f}  (Stage2 贡献)")

    # Stage 2 对输出的具体影响
    Y_orig = W @ X_eval
    Y_s1 = W_s1 @ X_eval
    Y_fr = W_fr @ X_eval
    Y_saes = W_saes @ X_eval

    s2_effect = torch.norm(Y_fr - Y_s1) / torch.norm(Y_orig)
    print(f"\nStage 2 对输出的影响: ||Y_FR_2s - Y_FR_S1||/||Y_orig|| = {s2_effect.item():.6f}")
    print(f"Stage 2 是否改善输出: ", end="")
    err_s1 = torch.norm(Y_s1 - Y_orig).item()
    err_fr = torch.norm(Y_fr - Y_orig).item()
    if err_fr < err_s1:
        print(f"✓ 改善 ({err_s1:.4f} → {err_fr:.4f}, 降低 {(err_s1-err_fr)/err_s1*100:.1f}%)")
    else:
        print(f"✗ 恶化 ({err_s1:.4f} → {err_fr:.4f}, 增加 {(err_fr-err_s1)/err_s1*100:.1f}%)")

    # A 矩阵的数值范围（检查 H^{-1/2} 放大）
    print(f"\n数值范围分析:")
    print(f"  A_saes: range=[{A_saes.min().item():.4f}, {A_saes.max().item():.4f}], "
          f"std={A_saes.std().item():.4f}")
    A1_fr = A_fr[:r1, :]
    A2_fr = A_fr[r1:, :]
    print(f"  A1_FR:  range=[{A1_fr.min().item():.4f}, {A1_fr.max().item():.4f}], "
          f"std={A1_fr.std().item():.4f}")
    if A2_fr.numel() > 0:
        print(f"  A2_FR:  range=[{A2_fr.min().item():.4f}, {A2_fr.max().item():.4f}], "
              f"std={A2_fr.std().item():.4f}")

    print("\n诊断完成！")


if __name__ == '__main__':
    main()
