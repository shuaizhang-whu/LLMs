"""
SAES-SVD 调试脚本
检查 beta=0 和误差爆炸的原因
"""
import torch
import sys
sys.path.append('/data/zhangs/LLMs/Projects/SAES-SVD')

# 加载 profiling 数据
profiling_path = '/data/zhangs/LLMs/Projects/SAES-SVD/result/SAES_SVD/llama-7b-hf/ratio_0.2/dataset_wikitext2_samples_128_seed_0/compress_output'
import glob
pt_files = glob.glob(f'{profiling_path}/*profiling*.pt')
if pt_files:
    print(f"找到 profiling 文件: {pt_files[0]}")
    profiling_mat = torch.load(pt_files[0], weights_only=False)
else:
    print("未找到 profiling 文件，尝试其他路径...")
    pt_files = glob.glob('/data/zhangs/LLMs/Projects/SAES-SVD/result/**/*profiling*.pt', recursive=True)
    if pt_files:
        print(f"找到 profiling 文件: {pt_files[0]}")
        profiling_mat = torch.load(pt_files[0], weights_only=False)
    else:
        print("未找到任何 profiling 文件!")
        exit(1)

print("\n" + "="*70)
print("分析 Layer 0 的统计量")
print("="*70)

layer_0 = profiling_mat[0]
for name in list(layer_0.keys())[:4]:  # 只看前4个子层
    print(f"\n--- {name} ---")
    H_chol = layer_0[name]['H_chol']
    Delta = layer_0[name]['Delta']
    
    # 重建 H
    H = H_chol @ H_chol.T
    
    # 检查 H 的特征值
    eigenvalues = torch.linalg.eigvalsh(H.float())
    print(f"H 特征值范围: [{eigenvalues.min().item():.2e}, {eigenvalues.max().item():.2e}]")
    print(f"H 条件数: {(eigenvalues.max() / eigenvalues.min()).item():.2e}")
    
    # 检查 Delta 的范数
    print(f"Delta Frobenius 范数: {torch.norm(Delta).item():.2e}")
    print(f"H Frobenius 范数: {torch.norm(H).item():.2e}")
    print(f"Delta/H 比值: {(torch.norm(Delta) / torch.norm(H)).item():.4f}")
    
    # 计算 H^{-1/2}
    eps = 1e-5
    H_reg = H.float() + eps * torch.eye(H.shape[0])
    eigenvalues_reg, eigenvectors = torch.linalg.eigh(H_reg)
    L = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues_reg)) @ eigenvectors.T
    print(f"L (H^{{-1/2}}) 范数: {torch.norm(L).item():.2e}")
    print(f"L 最大元素: {L.abs().max().item():.2e}")

print("\n" + "="*70)
print("检查 ACES 为何总是选择 beta=0")
print("="*70)

# 模拟 ACES 的计算
name = 'self_attn.q_proj'
H_chol = layer_0[name]['H_chol'].float()
Delta = layer_0[name]['Delta'].float()
H = H_chol @ H_chol.T

# 随机模拟一个 W (实际应该从模型中读取)
W = torch.randn(4096, 4096)  # q_proj 的形状
rank = 1638

# 计算 L = H^{-1/2}
eps = 1e-5
H_reg = H + eps * torch.eye(H.shape[0])
eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)
eigenvalues = torch.clamp(eigenvalues, min=eps)
L = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T

# 构造 S 和 D
S = W @ H @ L
D = W @ Delta @ L

print(f"\nS 范数: {torch.norm(S).item():.2e}")
print(f"D 范数: {torch.norm(D).item():.2e}")
print(f"D/S 比值: {(torch.norm(D) / torch.norm(S)).item():.4f}")

# SVD
U, Sigma, Vt = torch.linalg.svd(S, full_matrices=False)
U_r = U[:, :rank]
V_r = Vt[:rank, :].T

m = W.shape[0]
P_L = torch.eye(m) - U_r @ U_r.T
P_R = torch.eye(S.shape[1]) - V_r @ V_r.T

S_perp = P_L @ S @ P_R
D_perp = P_L @ D @ P_R

a = torch.norm(S_perp, 'fro')**2
b = torch.sum(S_perp * D_perp)
c = torch.norm(D_perp, 'fro')**2
A = torch.norm(S, 'fro')**2
B = torch.sum(S * D)
C = torch.norm(D, 'fro')**2

print(f"\n系数:")
print(f"  a (||S_perp||²) = {a.item():.2e}")
print(f"  b (⟨S_perp, D_perp⟩) = {b.item():.2e}")
print(f"  c (||D_perp||²) = {c.item():.2e}")
print(f"  A (||S||²) = {A.item():.2e}")
print(f"  B (⟨S, D⟩) = {B.item():.2e}")
print(f"  C (||D||²) = {C.item():.2e}")

# 计算不同 beta 的 score
print(f"\n不同 beta 的 score (越小越好):")
for beta in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75]:
    score = (a + 2*b*beta + c*beta**2) / (A + 2*B*beta + C*beta**2 + 1e-10)
    print(f"  beta={beta:.2f}: score={score.item():.6f}")

# 检查二次方程的解
coef_2 = c*B - b*C
coef_1 = c*A - a*C
coef_0 = b*A - a*B
print(f"\n二次方程系数:")
print(f"  coef_2 = {coef_2.item():.2e}")
print(f"  coef_1 = {coef_1.item():.2e}")
print(f"  coef_0 = {coef_0.item():.2e}")

if abs(coef_2.item()) > 1e-10:
    discriminant = coef_1**2 - 4*coef_2*coef_0
    print(f"  判别式 = {discriminant.item():.2e}")
    if discriminant >= 0:
        sqrt_disc = torch.sqrt(discriminant)
        root1 = (-coef_1 + sqrt_disc) / (2 * coef_2)
        root2 = (-coef_1 - sqrt_disc) / (2 * coef_2)
        print(f"  根1 = {root1.item():.4f}")
        print(f"  根2 = {root2.item():.4f}")
