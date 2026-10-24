import torch

# 加载你保存的压缩模型
# model_path = "/mnt/share/zhangshuai/LLMs/SVD-LLM/compress_output/_mnt_data_zhangshuai_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.5_2025_10_17_06_53_44.pt"

model_path = "/data/zhangs/LLMs/SVD-LLM/compress_output_v2/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.8_2025_10_24.pt"

checkpoint = torch.load(model_path, weights_only=False)

if isinstance(checkpoint, dict) and 'model' in checkpoint:
    model = checkpoint['model']
else:
    model = checkpoint

# 检查注意力层的权重形状
print("检查 SVD 模型的权重形状:")
for name, module in model.named_modules():
    if 'q_u_proj' in name or 'q_v_proj' in name:
        print(f"{name}: {module.weight.shape}")
        
# 预期输出应该是:
# q_u_proj: torch.Size([4096, 1565])  或其他合理的压缩比例
# q_v_proj: torch.Size([1565, 4096])
