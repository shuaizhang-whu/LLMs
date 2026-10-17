import os

# ============================================================
# Qwen3 模型下载脚本
# 推荐使用 ModelScope 下载，国内速度更快
# ============================================================

# 模型保存基础目录
BASE_DIR = "/mnt/data/zhangshuai/LLMs/Models"

# 方式1：使用 ModelScope 下载（推荐，速度更快）
# print("=" * 60)
# print("使用 ModelScope 下载 Qwen3-4B")
# print("=" * 60)

# 下载 Qwen3-4B
# os.system(f'modelscope download --model Qwen/Qwen3-4B --local_dir {BASE_DIR}/Qwen3-4B')

print("=" * 60)
print("使用 ModelScope 下载 Qwen3.5-35B-A3B")
print("=" * 60)

# 下载 Qwen3-32B
# os.system(f'modelscope download --model Qwen/Qwen3-32B --local_dir {BASE_DIR}/Qwen3-32B')

# 下载 Qwen3.5-35B-A3B
os.system(f'modelscope download --model Qwen/Qwen3.5-35B-A3B --local_dir {BASE_DIR}/Qwen3.5-35B-A3B')

# ============================================================
# 方式2：使用 HuggingFace 镜像下载（备选方案）
# 如果 ModelScope 不可用，可以使用以下命令
# ============================================================
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.system(f'hf download --repo-type model Qwen/Qwen3-4B --local-dir {BASE_DIR}/Qwen3-4B')
# os.system(f'hf download --repo-type model Qwen/Qwen3-32B --local-dir {BASE_DIR}/Qwen3-32B')

print("\n下载完成！")