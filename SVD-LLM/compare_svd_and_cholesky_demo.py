import numpy as np
import time
import torch

# 生成一个随机的 4096x4096 矩阵 X
n = 4096
X = np.random.randn(n, n)

# 测试 Cholesky 分解
# 计算 X*X^T
X_XT = np.dot(X, X.T)

print(f"Generating random matrix X is {X_XT},shape is {X_XT.shape}")

# 测量 Cholesky 分解的时间
start_time = time.time()
print("Starting Cholesky decomposition...")
L = np.linalg.cholesky(X_XT)  # Cholesky 分解
cholesky_time = time.time() - start_time
print(f"Cholesky 分解时间: {cholesky_time:.6f} 秒")

L_tensor = torch.tensor(L)
start_time = time.time()
L_inv = torch.linalg.inv(L_tensor)
inv_time = time.time() - start_time
print(f"Inverse time: {inv_time:.6f} 秒")

# 测试 SVD 分解
# 测量 SVD 分解的时间
start_time = time.time()
print("Starting SVD decomposition...")
U, S, Vt = np.linalg.svd(X_XT)  # SVD 分解
svd_time = time.time() - start_time

# 输出时间

print(f"SVD 分解时间: {svd_time:.6f} 秒")
