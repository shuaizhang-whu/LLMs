#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复已保存的SVD压缩模型，添加 in_features 和 out_features 属性
以便 LoRA 能够正确识别每层的输入输出维度
"""

import torch
import argparse

def fix_svd_model(model_path, output_path=None):
    """
    修复SVD压缩模型，为每个SVD分解的线性层添加真实的输入输出维度
    
    参数:
        model_path: 原始压缩模型路径
        output_path: 输出路径，如果为None则覆盖原文件
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']
    
    print("Fixing SVD layers...")
    
    # 遍历所有层
    if hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model.model, 'decoder'):
        layers = model.model.decoder.layers
    else:
        raise ValueError("Unknown model structure")
    
    for i, layer in enumerate(layers):
        # 检查是否有 self_attn
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            
            # 修复 q_proj
            if hasattr(attn, 'q_v_proj') and hasattr(attn, 'q_u_proj'):
                # q_v_proj: [hidden_size] -> [low_rank]
                # q_u_proj: [low_rank] -> [num_heads * head_dim]
                q_v_out_dim = attn.q_v_proj.weight.shape[0]  # low_rank
                q_v_in_dim = attn.q_v_proj.weight.shape[1]   # hidden_size
                q_u_out_dim = attn.q_u_proj.weight.shape[0]  # num_heads * head_dim
                q_u_in_dim = attn.q_u_proj.weight.shape[1]   # low_rank
                
                attn.q_v_proj.in_features = q_v_in_dim
                attn.q_v_proj.out_features = q_v_out_dim
                attn.q_u_proj.in_features = q_u_in_dim
                attn.q_u_proj.out_features = q_u_out_dim
                
                print(f"Layer {i} q_v_proj: in={q_v_in_dim}, out={q_v_out_dim}")
                print(f"Layer {i} q_u_proj: in={q_u_in_dim}, out={q_u_out_dim}")
            
            # 修复 k_proj
            if hasattr(attn, 'k_v_proj') and hasattr(attn, 'k_u_proj'):
                k_v_out_dim = attn.k_v_proj.weight.shape[0]
                k_v_in_dim = attn.k_v_proj.weight.shape[1]
                k_u_out_dim = attn.k_u_proj.weight.shape[0]
                k_u_in_dim = attn.k_u_proj.weight.shape[1]
                
                attn.k_v_proj.in_features = k_v_in_dim
                attn.k_v_proj.out_features = k_v_out_dim
                attn.k_u_proj.in_features = k_u_in_dim
                attn.k_u_proj.out_features = k_u_out_dim
            
            # 修复 v_proj
            if hasattr(attn, 'v_v_proj') and hasattr(attn, 'v_u_proj'):
                v_v_out_dim = attn.v_v_proj.weight.shape[0]
                v_v_in_dim = attn.v_v_proj.weight.shape[1]
                v_u_out_dim = attn.v_u_proj.weight.shape[0]
                v_u_in_dim = attn.v_u_proj.weight.shape[1]
                
                attn.v_v_proj.in_features = v_v_in_dim
                attn.v_v_proj.out_features = v_v_out_dim
                attn.v_u_proj.in_features = v_u_in_dim
                attn.v_u_proj.out_features = v_u_out_dim
            
            # 修复 o_proj
            if hasattr(attn, 'o_v_proj') and hasattr(attn, 'o_u_proj'):
                o_v_out_dim = attn.o_v_proj.weight.shape[0]
                o_v_in_dim = attn.o_v_proj.weight.shape[1]
                o_u_out_dim = attn.o_u_proj.weight.shape[0]
                o_u_in_dim = attn.o_u_proj.weight.shape[1]
                
                attn.o_v_proj.in_features = o_v_in_dim
                attn.o_v_proj.out_features = o_v_out_dim
                attn.o_u_proj.in_features = o_u_in_dim
                attn.o_u_proj.out_features = o_u_out_dim
        
        # 检查是否有 mlp
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            
            # 修复 gate_proj
            if hasattr(mlp, 'gate_v_proj') and hasattr(mlp, 'gate_u_proj'):
                gate_v_out_dim = mlp.gate_v_proj.weight.shape[0]
                gate_v_in_dim = mlp.gate_v_proj.weight.shape[1]
                gate_u_out_dim = mlp.gate_u_proj.weight.shape[0]
                gate_u_in_dim = mlp.gate_u_proj.weight.shape[1]
                
                mlp.gate_v_proj.in_features = gate_v_in_dim
                mlp.gate_v_proj.out_features = gate_v_out_dim
                mlp.gate_u_proj.in_features = gate_u_in_dim
                mlp.gate_u_proj.out_features = gate_u_out_dim
            
            # 修复 up_proj
            if hasattr(mlp, 'up_v_proj') and hasattr(mlp, 'up_u_proj'):
                up_v_out_dim = mlp.up_v_proj.weight.shape[0]
                up_v_in_dim = mlp.up_v_proj.weight.shape[1]
                up_u_out_dim = mlp.up_u_proj.weight.shape[0]
                up_u_in_dim = mlp.up_u_proj.weight.shape[1]
                
                mlp.up_v_proj.in_features = up_v_in_dim
                mlp.up_v_proj.out_features = up_v_out_dim
                mlp.up_u_proj.in_features = up_u_in_dim
                mlp.up_u_proj.out_features = up_u_out_dim
            
            # 修复 down_proj
            if hasattr(mlp, 'down_v_proj') and hasattr(mlp, 'down_u_proj'):
                down_v_out_dim = mlp.down_v_proj.weight.shape[0]
                down_v_in_dim = mlp.down_v_proj.weight.shape[1]
                down_u_out_dim = mlp.down_u_proj.weight.shape[0]
                down_u_in_dim = mlp.down_u_proj.weight.shape[1]
                
                mlp.down_v_proj.in_features = down_v_in_dim
                mlp.down_v_proj.out_features = down_v_out_dim
                mlp.down_u_proj.in_features = down_u_in_dim
                mlp.down_u_proj.out_features = down_u_out_dim
    
    # 保存修复后的模型
    if output_path is None:
        output_path = model_path
    
    print(f"Saving fixed model to {output_path}...")
    torch.save({'model': model, 'tokenizer': tokenizer}, output_path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='修复SVD压缩模型以支持LoRA微调')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='原始压缩模型路径')
    parser.add_argument('--output_path', type=str, default=None,
                        help='输出路径（默认覆盖原文件）')
    
    args = parser.parse_args()
    fix_svd_model(args.model_path, args.output_path)

