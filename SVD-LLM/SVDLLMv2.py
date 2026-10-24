#coding:utf8
import os
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import * 
from datetime import datetime

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

@torch.no_grad() # 关闭梯度计算
def profle_svdllm(name, model, calib_loader, dev):
    """
    对给定模型进行SVDLLM分析,计算每层线性层的白化矩阵(Whitening Matrix)并返回。

    参数:
        name (str): 模型名称,用于判断模型结构类型。
        model (torch.nn.Module): 待分析的模型。
        calib_loader (DataLoader): 用于校准的数据加载器。
        dev (torch.device): 模型和数据运行所在的设备。

    返回:
        dict: 每一层的线性层对应的Cholesky分解后的白化矩阵。
    """

    # 根据模型名称确定模型中层的结构
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers

    model = model.to(dev)
    print("Start obtaining the whitening matrix...")

    # 定义前向传播钩子函数,用于收集输入特征的协方差信息
    def hook(module, input, output):
        # 捕获模型在前向传播时的中间结果。
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        # 并计算其协方差矩阵
        adds_sum = torch.sum(adds, dim=0)
        # 累加到每个线性层上
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()

    # 注册钩子到所有线性层上
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 对这些线性层,我们将 hook 函数注册为它们的前向钩子
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    # 使用校准数据运行模型,触发钩子收集统计信息
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)

    # 清除注册的钩子
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()

    torch.cuda.empty_cache()
    model = model.cpu()

    # 将统计矩阵移动到CPU上
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()

    profiling_mat = {}
    print("Start Cholesky Decomposition...")

    # 对每一层进行Cholesky分解以获取白化矩阵
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        # 一个字典,键是子层的名称(例如 "q_proj"、"k_proj" 等),值是对应的子层对象
        subset = find_layers(layers[i])
        for name in subset:
            # 获取每个子层的协方差矩阵,将矩阵转换为 double 精度(64 位浮点数),并转移到GPU设备
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                # 尝试对 raw_scaling_diag_matrix 进行 Cholesky 分解。Cholesky 分解将一个正定矩阵分解为一个下三角矩阵和它的转置的乘积,目的是为了得到白化矩阵(使得数据分布变得更加规范化)。torch.linalg.cholesky()：这是 PyTorch 中用于计算 Cholesky 分解的函数。
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                # 打印警告,表示协方差矩阵不是正定的
                print("Warning: eigen scaling_diag_matrix is not positive!")
                # ：使用 torch.linalg.eigvalsh() 计算矩阵的特征值。eigvalsh 是一个用于计算 Hermitian 或对称矩阵特征值的函数。
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                # 对矩阵进行调整,使得它变得正定。调整的方法是将最小特征值(eigenvalues[0])加上一个小的常数(1e-6),并将其加到对角线矩阵上
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                # 重新进行 Cholesky 分解：对调整后的矩阵重新进行 Cholesky 分解。
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                # 清除特征值变量：eigenvalues 用完后被删除
                del eigenvalues
            # 将分解后的白化矩阵 scaling_diag_matrix 存储到 layer_profile 字典中。矩阵存储到 CPU 中,以便后续使用和节省 GPU 内存。
            layer_profile[name] = scaling_diag_matrix.cpu()
            # 释放这些变量,因为它们已经被存储在 layer_profile 中
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            # 清理
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        # 更新字典,将 layer_profile 字典(其中包含了当前层的白化矩阵)添加到 profiling_mat 中。profiling_mat 是一个字典,键是层的索引 i,值是当前层的白化矩阵(存储在 layer_profile 中)
        profiling_mat[i] = layer_profile

    return profiling_mat


@torch.no_grad() # 关闭梯度计算，后续仅做推理/权重重排，节省显存与加速
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    """
    对低资源环境下的SVDLLM模型进行性能分析(profiling),计算每层中各子模块的scaling diagonal矩阵,
    用于后续量化或压缩等优化操作。

    参数:
        model_name (str): 模型名称,用于判断模型结构(如是否为OPT模型)。
        model (torch.nn.Module): 待分析的模型实例。
        calib_loader (DataLoader): 校准数据加载器,用于提供输入样本。
        dev (torch.device): 模型运行设备(如cuda或cpu)。

    返回:
        dict: 每一层的性能分析结果,键为层索引,值为该层各子模块的scaling diagonal矩阵。
    """

    # 根据模型名称选择不同的层结构和嵌入层
    # 判断模型是否是 OPT 系列（OPT 的模块命名和 Embedding 结构与 LLaMA/ Mistral 略有不同，需要分支处理）。
    if "opt" in model_name:
        layers = model.model.decoder.layers
        # 把 OPT 的嵌入层、位置编码层、最终归一化层搬到 dev（GPU）。
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        # 非 OPT：直接拿 model.model.layers（Transformer block 列表）。
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    
    # 将第一层移动到指定设备
    layers[0] = layers[0].to(dev)

    # 初始化输入张量,用于存储校准数据的中间激活值
    # 不一次把所有层都搬上去，而是用到哪一层再上 GPU，减少显存占用。
    dtype = next(iter(model.parameters())).dtype
    # 取模型参数的数据类型（如 float16/bfloat16/float32），保证我们新建张量和模型精度一致。
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    
    # 缓存用于存储中间变量
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    # 定义一个包装模块，替换第一层用，拦截第 0 层的输入张量。
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        # 定义前向：一旦第 0 层要执行，这里就先“截获”输入 inp 和关键的 kw 参数。
        def forward(self, inp, **kwargs):
            # 存储输入激活值,把 inp先 .cpu()：避免在 GPU 上长期占用；赋值时会触发跨设备拷贝到 inps（inps 在 dev 上）
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            
            # 第一次批次：把 attention_mask（以及非 OPT 的 position_ids）缓存下来（放 CPU 内存，省 GPU）。
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                # 把 mask / position_ids 沿 batch 维拼接起来（形成 [num_batches, T] 或多维形状
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            
            # 抛出异常以中断前向传播
            raise ValueError

    # 替换第一层为Catcher钩子,用于捕获输入
    layers[0] = Catcher(layers[0])
    
    # 遍历校准数据,捕获输入激活值
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass  # 捕获ValueError以继续处理下一个batch

    # 恢复原始第一层并将其移回CPU
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    
    # 把第 0 层移回 CPU，后面我们会按逐层的方式上 GPU。
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        # 对 OPT：把前面暂时放到 GPU 的 Embedding / LN / 位置编码，移回 CPU，释放显存。
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    
    # 清空GPU缓存
    torch.cuda.empty_cache()

    # 创建一个与输入张量 inps 形状相同的零张量，用于存储模型每层的输出
    outs = torch.zeros_like(inps)
    # 从缓存中取出注意力掩码，供后续前向传播使用
    attention_masks = cache['attention_mask']
    # 如果不是 OPT 模型，则也从缓存中取出位置编码信息
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    
    # 存储每层的性能分析结果
    profiling_mat = {}

    # 遍历每一层进行性能分析
    for i in tqdm(range(len(layers))):
        # 本层的子层统计容器：键是子层名（如 "q_proj"），值是该子层的 Cholesky 结果（白化矩阵）
        layer_profile = {}
        # 把第 i 层搬到 GPU（逐层上/下 GPU，节约显存）
        layer = layers[i].to(dev)
        subset = find_layers(layer)  # 获取该层中的子模块

        # 定义前向钩子：对每个线性层的输入激活做 X^T * X 的累加
        def hook(module, input, output):
            # 获取模块输入并转为浮点型
            inp = input[0].detach().float()
            if inp.dim() == 2:  # 对于OPT模型,增加一个维度
                inp = inp.unsqueeze(0)
            
            # 计算输入的自乘矩阵并累加
            adds = torch.matmul(inp.transpose(1,2), inp) # 计算 X^T * X
            # 将每个批次的统计结果累加到 scaling_diag_matrix 中，用于后续分析
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            
            # 清理临时变量和GPU缓存
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()

        # 为模型每一层的子模块（如线性层）注册前向传播钩子（hook），用于收集输入激活值的统计信息
        handles = []
        for name in subset:
            # 初始化统计变量：将每个子模块的 scaling_diag_matrix 设为0，用于累计输入的协方差矩阵
            subset[name].scaling_diag_matrix = 0
            # register_forward_hook 将自定义的 hook 函数绑定到每个子模块上，前向传播时会调用该函数进行统计
            handles.append(subset[name].register_forward_hook(hook))

        # 遍历所有输入样本,执行前向传播以收集统计信息
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                # 遍历我们收集到的“第 0 层输入激活”的每个样本步
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]

        # 移除钩子,避免对后续层重复生效
        for h in handles:
            h.remove()

        # 将层移回CPU并清理缓存
        layer = layer.cpu()
        test_value = 1
        # 将本层每个线性子层累计得到的 Gram 矩阵移到 CPU，释放 GPU
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        # 清缓存
        torch.cuda.empty_cache()

        # 计算每层子模块的scaling diagonal矩阵
        for name in subset:
            # 逐个子层：把累计矩阵拿回来（转为 double 精度，搬到 GPU），准备做SVD。
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)

            # 对 scaling_diag_matrix 进行 SVD 分解
            U_s, S_s, VT_s = torch.linalg.svd(raw_scaling_diag_matrix, full_matrices=False)

            # 计算 S = U_s * sqrt(S_s)
            sqrt_S_s = torch.sqrt(S_s)  # 计算 S_s 的平方根
            scaling_diag_matrix = U_s @ torch.diag(sqrt_S_s)  # S = U_s * sqrt(S_s)

            # 存储结果,把该子层的 Cholesky 结果放到 layer_profile（移回 CPU）
            layer_profile[name] = scaling_diag_matrix.cpu()
            # 显式删除临时变量，释放引用。
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()

        # 更新层和输入张量
        layers[i] = layer.cpu()
        # 本层的所有子层的白化矩阵打包进总字典 profiling_mat（键为层索引 i）
        profiling_mat[i] = layer_profile
        # 关键一步：把当前层的输出 outs，作为下一层的输入 inps。
        inps = outs
        # 清缓存，继续下一层
        torch.cuda.empty_cache()

    # 返回每层的性能分析结果
    return profiling_mat


@torch.no_grad()  # 关闭梯度计算，后续仅做推理/权重重排，节省显存与加速
def whitening(model_name, model, profiling_mat, target_ratio, dev):  # 定义函数：基于白化统计进行SVD分解并替换模块
    """
    带自适应压缩比率分配的白化SVD压缩函数
    
    参数:
        model_name: 模型名称
        model: 待压缩模型
        profiling_mat: 白化统计矩阵
        target_ratio: 目标全局压缩比率
        dev: 计算设备
    """

    # 对每一层内部的权重矩阵进行分组
    def group_weights_by_layer(layers):
        """
        对每一层内部的权重矩阵进行分组
        
        返回:
            layer_groups: list of dict, 每个元素对应一层的分组信息
                        每层的dict: key为权重名称, value为权重信息
        """
        layer_groups = [] # 用于存储每一层的分组信息。每个元素将代表一层的所有权重信息
        
        for i, layer in enumerate(layers): # 遍历所有层
            subset = find_layers(layer) # 查找当前层的所有线性层
            layer_group = {} # 初始化当前层的分组字典
            
            for name in subset: # 遍历当前层的所有线性层
                W = subset[name].weight.data # 获取权重数据
                layer_group[name] = { # 存储权重信息
                    'layer_idx': i, # 记录这是第几层
                    'name': name, # 记录权重的名称
                    'shape': W.shape, # 记录权重矩阵的形状（通常是 [输出维度, 输入维度]）
                    'num_params': W.shape[0] * W.shape[1] # 计算参数总数（矩阵元素个数）
                }
            
            layer_groups.append(layer_group) # 将当前层信息添加到结果列表
        
        return layer_groups

    def compute_theoretical_loss(W, scaling_matrix, ratio, dev):
        """
        计算给定压缩比率下的理论截断损失（基于参数量的压缩比）
        
        参数:
            W: 权重矩阵
            scaling_matrix: 白化缩放矩阵 L
            ratio: 压缩比率（基于参数量）
            dev: 设备
            
        返回:
            loss: 截断损失（被截断的奇异值平方和）
        """
        W = W.float().to(dev)
        scaling_matrix = scaling_matrix.float().to(dev)
        scaling_matrix_inv = torch.linalg.inv(scaling_matrix) # 经过SVD分解计算的白化矩阵L可以直接计算逆矩阵
        
        # 白化变换
        W_scale = torch.matmul(W, scaling_matrix)
        
        # 二次SVD分解
        U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
        
        # 使用基于参数量的压缩比率计算（与 local_update 一致）
        # ratio 控制压缩后的参数量占原始参数量的比例
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        num_s_after_trunc = max(1, num_s_after_trunc)  # 至少保留1个
        
        # 计算被截断部分的Frobenius范数（奇异值平方和的平方根）
        if num_s_after_trunc < len(S):
            truncated_singular_values = S[num_s_after_trunc:]
            loss = torch.sqrt(torch.sum(truncated_singular_values ** 2)).item()
        else:
            loss = 0.0
        
        return loss

    def ratio_allocation(model_name, model, profiling_mat, target_ratio, dev):
        """
        异构压缩比率分配算法（Algorithm 1）
        每层内部独立进行比率分配
        
        参数:
            model_name: 模型名称
            model: 原始模型
            profiling_mat: 白化统计矩阵
            target_ratio: 目标全局压缩比率 R
            dev: 计算设备
            
        返回:
            allocated_ratios: dict, key为(layer_idx, weight_name), value为分配的压缩比率
        """
        if 'opt' in model_name:
            layers = model.model.decoder.layers
        else:
            layers = model.model.layers
        
        print("Starting heterogeneous compression ratio allocation...")
        
        # Step 1: 对每一层进行分组（每层内部独立分组）
        layer_groups = group_weights_by_layer(layers)
        
        allocated_ratios = {}
        
        # 对每一层独立处理
        for layer_idx in tqdm(range(len(layers)), desc="Allocating ratios per layer"):
            layer = layers[layer_idx]
            layer_group = layer_groups[layer_idx]  # 当前层的所有权重
            
            # Step 4: 初始化该层的损失列表
            loss_list = []
            weight_names = []
            
            # Step 5-8: 计算该层每个权重矩阵的理论损失
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                scaling_matrix = profiling_mat[layer_idx][name].to(dev)
                
                # 计算理论损失
                loss = compute_theoretical_loss(W, scaling_matrix, target_ratio, dev)
                loss_list.append(loss)
                weight_names.append(name)
            
            # Step 9: 归一化损失
            loss_array = np.array(loss_list)

            print(f"Layer {layer_idx} loss list: {loss_array}")
            
            # 避免除零
            if loss_array.sum() == 0:
                loss_array = np.ones_like(loss_array)
            
            # 使用log归一化（防止极端值）
            normalized_loss = 1.0 / np.log(loss_array)  # 使用e作为底数避免log(0)
            
            # Step 10-11: 根据损失分配压缩比率
            # 论文公式: r = Len(L_G) × R × L_min / Sum(L_G)
            # 其中 Len(L_G) 是该层权重数量（通常是7或6）
            num_weights_in_layer = len(weight_names)
            
            for idx, name in enumerate(weight_names):
                # 使用原始损失作为分子，归一化损失和作为分母
                allocated_ratio = num_weights_in_layer * target_ratio * normalized_loss[idx] / normalized_loss.sum()
                
                # 确保比率在合理范围内 [0.05, 1.0]
                allocated_ratio = max(0.01, min(allocated_ratio, 1.0))
                
                # Step 12: 添加到分配结果
                allocated_ratios[(layer_idx, name)] = allocated_ratio
        
        # 打印分配结果统计
        ratios_values = list(allocated_ratios.values())
        return allocated_ratios

    model.eval()

    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    
    # Step 1: 执行压缩比率分配算法
    allocated_ratios = ratio_allocation(model_name, model, profiling_mat, target_ratio, dev)

    # print(f"Ratio allocation completion: \n{allocated_ratios}\n")
    
    print("\nStarting SVD decomposition with adaptive compression ratios...")
    
    # Step 2: 对每一层进行SVD分解（使用分配的压缩比率）
    for i in tqdm(range(len(layers)), desc="SVD Decomposition"):
        layer = layers[i]
        subset = find_layers(layer)
        
        # 创建SVD替换模块（ratio参数仅用于初始化，实际使用分配的ratio）
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=target_ratio)
            svd_mlp = SVD_LlamaMLP(
                hidden_size=layer.hidden_size,
                intermediate_size=model.config.intermediate_size,
                hidden_act=model.config.hidden_act,
                ratio=target_ratio
            )
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=target_ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=target_ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=target_ratio)
        
        # 对每个权重矩阵进行SVD分解
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev) #拿到经过SVD分解计算的白化矩阵 S = U_s * sqrt(S_s)
            
            # 使用该权重矩阵的专属压缩比率
            ratio = allocated_ratios.get((i, name), target_ratio)

            print(f"Layer {i}, {name} ratio: {ratio:.4f}")
            
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix) # 经过SVD分解计算的白化矩阵可以直接计算逆矩阵
            
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix) # 计算白化空间的权重矩阵
            
            # 二次SVD分解
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            
            # 使用自适应压缩比率计算截断逻辑
            # ratio 表示基于参数量的压缩比，确保压缩后参数量 = 原始参数量 × ratio
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            num_s_after_trunc = max(1, num_s_after_trunc)  # 至少保留1个
            
            truc_s = S[:num_s_after_trunc] # 取前 r 个奇异值
            truc_u = U[:, :num_s_after_trunc] # U_r
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv) # 还原到原坐标系 V^T_r · S^{-1} → 相当于把白化消掉
            truc_sigma = torch.diag(truc_s) # 将 r 个奇异值组成对角矩阵 Σ_r,将一维奇异值向量转换为对角矩阵
            
            # 分解为两个低秩矩阵
            sqrtSigma = torch.sqrt(truc_sigma) # √Σ_r 
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype) # U_r · √Σ_r
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype) # √Σ_r · (V^T_r · S^{-1})
            
            # 替换权重（与原始代码相同的替换逻辑）
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data
                    # 为 LoRA 添加真实的输入输出维度标记
                    svd_decoder.self_attn.q_v_proj.in_features = W.shape[1]  # 原始输入维度
                    svd_decoder.self_attn.q_v_proj.out_features = svd_v.shape[0]  # SVD后的中间维度
                    svd_decoder.self_attn.q_u_proj.in_features = svd_u.shape[1]  # SVD后的中间维度  
                    svd_decoder.self_attn.q_u_proj.out_features = W.shape[0]  # 原始输出维度
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                    svd_decoder.self_attn.k_v_proj.in_features = W.shape[1]
                    svd_decoder.self_attn.k_v_proj.out_features = svd_v.shape[0]
                    svd_decoder.self_attn.k_u_proj.in_features = svd_u.shape[1]
                    svd_decoder.self_attn.k_u_proj.out_features = W.shape[0]
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                    svd_decoder.self_attn.v_v_proj.in_features = W.shape[1]
                    svd_decoder.self_attn.v_v_proj.out_features = svd_v.shape[0]
                    svd_decoder.self_attn.v_u_proj.in_features = svd_u.shape[1]
                    svd_decoder.self_attn.v_u_proj.out_features = W.shape[0]
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                    svd_decoder.self_attn.out_v_proj.in_features = W.shape[1]
                    svd_decoder.self_attn.out_v_proj.out_features = svd_v.shape[0]
                    svd_decoder.self_attn.out_u_proj.in_features = svd_u.shape[1]
                    svd_decoder.self_attn.out_u_proj.out_features = W.shape[0]
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                    svd_decoder.fc1_v_proj.in_features = W.shape[1]
                    svd_decoder.fc1_v_proj.out_features = svd_v.shape[0]
                    svd_decoder.fc1_u_proj.in_features = svd_u.shape[1]
                    svd_decoder.fc1_u_proj.out_features = W.shape[0]
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.fc2_v_proj.in_features = W.shape[1]
                    svd_decoder.fc2_v_proj.out_features = svd_v.shape[0]
                    svd_decoder.fc2_u_proj.in_features = svd_u.shape[1]
                    svd_decoder.fc2_u_proj.out_features = W.shape[0]
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                    # 为 LoRA 添加真实的输入输出维度标记
                    svd_attn.q_v_proj.in_features = W.shape[1]
                    svd_attn.q_v_proj.out_features = svd_v.shape[0]
                    svd_attn.q_u_proj.in_features = svd_u.shape[1]
                    svd_attn.q_u_proj.out_features = W.shape[0]
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                    svd_attn.k_v_proj.in_features = W.shape[1]
                    svd_attn.k_v_proj.out_features = svd_v.shape[0]
                    svd_attn.k_u_proj.in_features = svd_u.shape[1]
                    svd_attn.k_u_proj.out_features = W.shape[0]
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                    svd_attn.v_v_proj.in_features = W.shape[1]
                    svd_attn.v_v_proj.out_features = svd_v.shape[0]
                    svd_attn.v_u_proj.in_features = svd_u.shape[1]
                    svd_attn.v_u_proj.out_features = W.shape[0]
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    svd_attn.o_v_proj.in_features = W.shape[1]
                    svd_attn.o_v_proj.out_features = svd_v.shape[0]
                    svd_attn.o_u_proj.in_features = svd_u.shape[1]
                    svd_attn.o_u_proj.out_features = W.shape[0]
                    layer.self_attn = svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                    svd_mlp.gate_v_proj.in_features = W.shape[1]
                    svd_mlp.gate_v_proj.out_features = svd_v.shape[0]
                    svd_mlp.gate_u_proj.in_features = svd_u.shape[1]
                    svd_mlp.gate_u_proj.out_features = W.shape[0]
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                    svd_mlp.down_v_proj.in_features = W.shape[1]
                    svd_mlp.down_v_proj.out_features = svd_v.shape[0]
                    svd_mlp.down_u_proj.in_features = svd_u.shape[1]
                    svd_mlp.down_u_proj.out_features = W.shape[0]
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    svd_mlp.up_v_proj.in_features = W.shape[1]
                    svd_mlp.up_v_proj.out_features = svd_v.shape[0]
                    svd_mlp.up_u_proj.in_features = svd_u.shape[1]
                    svd_mlp.up_u_proj.out_features = W.shape[0]
                    layer.mlp = svd_mlp
            
            # 清理内存
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = None
            U = S = VT = truc_s = truc_u = truc_v = sqrtSigma = None
            del W, W_scale, scaling_matrix_inv, scaling_diag_matrix
            del U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        
        del layer
        torch.cuda.empty_cache()
    
    print("SVD decomposition with adaptive ratios completed!")


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False, use_dynamic_ratio=False):
     # 定义函数：白化+SVD后做局部更新（可选直达SVD）
    """
    对模型的每一层进行白化（whitening）处理并更新参数。该函数通过SVD分解对注意力和MLP层的权重进行压缩和重构。

    参数:
        model_name (str): 模型名称，用于判断模型结构（如 "opt", "llama", "vicuna", "mistral" 等）。
        model (torch.nn.Module): 待处理的模型对象。
        dataloader (DataLoader): 用于获取输入数据的数据加载器。
        profiling_mat (dict or None): 包含每层每个模块的缩放矩阵（scaling matrix）的字典，用于白化处理。
        ratio (float): 压缩比例，用于控制SVD保留的奇异值数量。
        dev (torch.device): 指定模型运行的设备（如 CPU 或 GPU）。
        direct_update (bool, optional): 是否直接更新权重而不进行额外的SVD处理。默认为 False。
        use_dynamic_ratio (bool, optional): 是否使用动态压缩比分配。默认为 False。

    返回:
        无返回值。函数会直接修改传入的 model 对象。
    """
    print("Start SVD decomposition then update...")  # 提示开始SVD分解并执行局部更新
    use_cache = model.config.use_cache  # 备份原来的use_cache配置
    model.config.use_cache = False  # 关闭KV cache，避免截获与逐层前向时干扰
    
    # 导入必要的工具函数（无论是否使用动态压缩比都需要）
    from utils.model_utils import find_layers
    import numpy as np

    if "opt" in model_name:  # OPT结构分支（其层与嵌入模块路径不同）
        layers = model.model.decoder.layers  # 取OPT的decoder层列表
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)  # 嵌入搬到GPU，便于前向
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)  # 末端LN搬到GPU
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)  # 位置编码搬到GPU
    else:  # LLaMA/Mistral/Vicuna分支
        layers = model.model.layers  # 取通用的层列表
        model.model.embed_tokens = model.model.embed_tokens.to(dev)  # 嵌入搬到GPU
        model.model.norm = model.model.norm.to(dev)  # 末端RMSNorm/LN搬到GPU
    
    # 如果启用动态压缩比且提供了白化矩阵，则计算每层的专属压缩比
    allocated_ratios = {}
    if use_dynamic_ratio and profiling_mat is not None:
        print("Computing dynamic compression ratios for local update...")
        # 调用 ratio_allocation 函数（定义在 whitening 函数内部）
        # 需要导入或重新定义相关函数
        
        def group_weights_by_layer(layers):
            """对每一层内部的权重矩阵进行分组"""
            layer_groups = []
            for i, layer in enumerate(layers):
                subset = find_layers(layer)
                layer_group = {}
                for name in subset:
                    W = subset[name].weight.data
                    layer_group[name] = {
                        'layer_idx': i,
                        'name': name,
                        'shape': W.shape,
                        'num_params': W.shape[0] * W.shape[1]
                    }
                layer_groups.append(layer_group)
            return layer_groups
        
        def compute_theoretical_loss(W, scaling_matrix, ratio, dev):
            """计算给定压缩比率下的理论截断损失（基于参数量的压缩比）"""
            W = W.float().to(dev)
            scaling_matrix = scaling_matrix.float().to(dev)
            scaling_matrix_inv = torch.linalg.inv(scaling_matrix)
            W_scale = torch.matmul(W, scaling_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            # 使用与 local_update 相同的公式：基于参数量控制
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            num_s_after_trunc = max(1, num_s_after_trunc)
            if num_s_after_trunc < len(S):
                truncated_singular_values = S[num_s_after_trunc:]
                loss = torch.sqrt(torch.sum(truncated_singular_values ** 2)).item()
            else:
                loss = 0.0
            return loss
        
        # 执行压缩比分配
        layer_groups = group_weights_by_layer(layers)
        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            layer_group = layer_groups[layer_idx]
            loss_list = []
            weight_names = []
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                scaling_matrix = profiling_mat[layer_idx][name].to(dev)
                loss = compute_theoretical_loss(W, scaling_matrix, ratio, dev)
                loss_list.append(loss)
                weight_names.append(name)
            
            loss_array = np.array(loss_list)
            if loss_array.sum() == 0:
                loss_array = np.ones_like(loss_array)
            normalized_loss = 1.0 / np.log(loss_array + 1e-6)  # 加小常数避免log(0)
            num_weights_in_layer = len(weight_names)
            
            for idx, name in enumerate(weight_names):
                allocated_ratio = num_weights_in_layer * ratio * normalized_loss[idx] / normalized_loss.sum()
                allocated_ratio = max(0.01, min(allocated_ratio, 1.0))
                allocated_ratios[(layer_idx, name)] = allocated_ratio
        
        print(f"Dynamic ratio allocation completed. Ratio range: [{min(allocated_ratios.values()):.4f}, {max(allocated_ratios.values()):.4f}]")
    else:
        print(f"Using fixed compression ratio: {ratio}")
        # 如果不使用动态压缩比，所有层都使用相同的ratio
        for i in range(len(layers)):
            subset = find_layers(layers[i])
            for name in subset:
                allocated_ratios[(i, name)] = ratio

    model.model.norm = model.model.norm.to(dev)  # 冗余防御：再次确保norm在GPU（上一分支已做，此行重复）
    layers[0] = layers[0].to(dev)  # 仅将第0层先搬到GPU，后续逐层上/下GPU以省显存

    dtype = next(iter(model.parameters())).dtype  # 读取模型参数的dtype（float16/bfloat16/float32）
    inps = torch.zeros(  # 预分配第0层输入激活缓存（逐样本步存放）
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev  # 形状=[批次数, 序列长, 隐藏维]
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}  # 缓存游标与mask/pos_ids

    class Catcher(nn.Module):  # 包装第0层以“截获”第0层输入激活
        def __init__(self, module):
            super().__init__()
            self.module = module  # 保存被包装的原第0层
        def forward(self, inp, **kwargs):  # 前向时截获进入第0层的输入
            inps[cache['i']] = inp  # 记录本步输入激活（直接存GPU版inps）
            cache['i'] += 1  # 样本步自增
            if cache['attention_mask'] is None:  # 首次记录mask/pos_ids
                cache['attention_mask'] = kwargs['attention_mask']  # 记录attention_mask（保持设备一致）
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']  # 非OPT记录position_ids
            else:  # 后续样本步拼接mask/pos_ids到batch维
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError  # 抛异常中断后续前向（仅需截获第0层输入）

    layers[0] = Catcher(layers[0])  # 用Catcher替换第0层，拦截第0层输入
    for batch in dataloader:  # 遍历校准/更新用数据
        try:
            model(batch[0].to(dev))  # 前向一次以触发Catcher（注意：此处假设dataloader返回tuple，取batch[0]为输入）
        except ValueError:
            pass  # 捕获并吞掉Catcher抛出的异常，继续下一个batch

    layers[0] = layers[0].module  # 恢复第0层为原模块
    layers[0] = layers[0].cpu()  # 第0层移回CPU以释放显存（后面逐层上GPU）
    model.model.embed_tokens = model.model.embed_tokens.cpu()  # 嵌入也移回CPU
    model.model.norm = model.model.norm.cpu()  # 末端norm移回CPU
    torch.cuda.empty_cache()  # 清空CUDA缓存，降低显存占用峰值

    outs = torch.zeros_like(inps)  # 创建一个与 inps 同形状的张量，并且初始化为全零。这个张量将用于存储每一层的输出激活
    attention_masks = cache['attention_mask']  # 取出记录的attention_mask（当前设备与inps一致）
    if "opt" not in model_name:
        position_ids = cache['position_ids']  # 非OPT取出position_ids

    for i in tqdm(range(len(layers))):  # 逐层处理（进度条）
        layer = layers[i].to(dev)  # 当前层上GPU
        subset = find_layers(layer)  # 找到本层内所有Linear子层（如q/k/v/o_proj、up/down/gate等）
        gpts = {}  # 存放每个子层的local_update对象

        if "llama" in model_name or "vicuna" in model_name:  # LLaMA/Vicuna：构建SVD替身模块
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)  # SVD注意力替身（两层低秩线性）
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)  # SVD-MLP替身
        elif "mistral" in model_name:  # Mistral：构建对应替身
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:  # OPT：整层SVD替身
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)

        for name in subset:  # 为本层每个Linear子层初始化local_update
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)  # 若提供白化矩阵，则取对应Cholesky因子L
            else:
                scaling_diag_matrix = None  # 否则直接对原W做SVD（direct_update控制）
            # 使用该层该子层的专属压缩比
            layer_ratio = allocated_ratios.get((i, name), ratio)
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=layer_ratio, name=name, direct_update=direct_update)  # 构造local_update（完成白化→SVD→截断的初始化）

        # 计算与真实输出的误差
        def add_batch(name):  # 构造hook闭包：把当前子层的inp/out喂给local_update以更新U
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)  # 传入子层的输入/输出张量，做最小二乘更新（仅更新U侧）
            return tmp

        handles = []  # 保存hook句柄便于稍后移除
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))  # 给每个子层挂前向hook以收集一次层级前向的inp/out

        if "opt" not in model_name:  # 执行一次本层前向（用捕获的inps和mask/pos_ids）
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]  # 得到本层输出hidden_states（触发hook收集）
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]  # OPT无position_ids参数

        for h in handles:
            h.remove()  # 移除全部hook，避免影响后续

        for name in gpts:  # 从每个local_update取出更新后的低秩因子
            svd_u, svd_v = gpts[name].fasterprune()  # 返回 U√Σ 与 √ΣV（已做近似误差最小化的更新）
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)  # 转回模型dtype（如float16）

            if 'opt' in model_name:  # 写回到OPT的SVD替身层（保留原bias）
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # OPT线性层带bias
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm  # LN直接复用
                    svd_decoder.final_layer_norm = layer.final_layer_norm  # LN直接复用
                    layers[i] = svd_decoder  # 用SVD版整层替换原层
            else:  # LLaMA/Mistral/Vicuna 写回到SVD注意力/MLP替身
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn  # 当走到o_proj时，替换整块self_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp  # 当走到up_proj时，替换整块MLP

        layer = layer.to(dev)  # 确保替换后的层在GPU（有的写回操作可能在CPU上）
        if "opt" not in model_name:  # 用“更新后的层”再前向一次，得到新的outs，作为下一层的inps
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]

        layers[i] = layer.cpu()  # 当前层下放CPU，省显存
        del gpts  # 释放本层local_update容器（里面可能持有较大张量）
        torch.cuda.empty_cache()  # 清理CUDA缓存
        inps = outs  # 滚动：将本层输出作为下一层输入
        outs = None  # 释放引用
        del outs  # 显式删除变量，利于内存回收

    model.config.use_cache = use_cache  # 恢复模型原use_cache设置



class local_update:
    """
    用于对神经网络层进行局部更新的类，基于SVD（奇异值分解）实现低秩近似和参数更新。

    参数:
        layer: 神经网络层（如 nn.Linear），包含权重 weight。
        scaling_diag_matrix: 缩放矩阵，用于在非 direct_update 模式下对权重进行预处理。
        ratio: 控制截断SVD保留的奇异值比例，影响低秩近似的精度。
        name: 层的名称标识符，用于调试或日志记录。
        direct_update: 是否直接对权重进行SVD分解，若为False则先与缩放矩阵相乘再分解。
    """

    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False):
        self.layer = layer # 保存传入的层对象
        self.name = name # 保存层的名称。
        self.dev = self.layer.weight.device # 获取该层权重所在的设备（GPU 或 CPU），用于后续的计算
        # 克隆权重以避免修改原始参数
        W = layer.weight.data.clone()
        # 获取权重矩阵的行数和列数，这些信息在后续处理中会用到
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # 如果为True，那么直接对权重矩阵 W 进行 SVD 分解，得到 U、S（奇异值）和 VT（右奇异向量）
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            #  # 先使用 scaling_diag_matrix 对权重矩阵进行缩放，并通过 SVD 分解获得低秩近似
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix) # 对缩放的白化矩阵进行求逆操作，以便后续使用
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                # 对白化矩阵进行数值稳定性调整
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix) # 将原始的权重矩阵与缩放矩阵相乘，得到一个调整后的矩阵
            # 对调整后的权重矩阵进行 SVD 分解，得到左奇异矩阵 U、奇异值 S 和右奇异矩阵 VT。
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  

        # 截断SVD：根据 ratio 计算保留的奇异值数量
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda() # 保留前 num_s_after_trunc 个奇异值，并将其转移到 GPU 上。
        self.truc_u = self.U[:, :num_s_after_trunc].cuda() # 保留左奇异向量的前 num_s_after_trunc 列，并将其转移到 GPU 上。
        # 直接使用原始的右奇异向量；否则，先对其进行缩放处理。
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda() # 保留右奇异向量的前 num_s_after_trunc 行，并将其转移到 GPU 上。
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s) # 将保留的奇异值构造为对角矩阵。

        # 构建新的低秩权重矩阵
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))

        # 初始化误差项
        self.updated_err = self.error = 0


    def add_batch_update_u(self, inp, out):
        """
        使用输入输出数据更新左奇异向量 U，并计算重构误差。
        用于在每个批次的前向传播中进行低秩矩阵的更新

        参数:
            inp: 输入张量，形状为 [batch_size, seq_len, input_dim]。
            out: 输出张量，形状为 [batch_size, seq_len, output_dim]。
        """
        # 将输入输出展平为二维张量
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2])

        # 当前低秩权重重构输出并计算误差
        # 这是低秩近似的标准操作，类似于 U * Σ * V^T，它将原始权重矩阵的低秩表示恢复到一个近似的权重矩阵
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        # 使用重构后的低秩权重 new_w 计算新的输出
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item() # 计算重构后的输出 new_output 和真实输出 outs 之间的误差。torch.sum((outs - new_output)**2) 是求误差平方和，torch.sqrt 计算平方和的平方根，即 欧几里得距离（L2 范数）。torch.norm(outs, p='fro').item() 计算 Frobenius 范数（矩阵的 Frobenius 范数），即真实输出 outs 的平方和的平方根。

        # 构造用于最小二乘求解的输入特征
        x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)

        # 通过最小二乘法更新 U 的转置,这里做的就是解方程,用 最小二乘法（least squares）找到一个新的UrT使得重构输出Y和真实输出Y 的误差最小。
        self.updated_uT = torch.linalg.lstsq(x, outs).solution

        # 使用更新后的 U 重新计算输出并评估误差
        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()

        # 清理临时变量并释放GPU内存
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()


    def fasterprune(self):
        """
        构造用于剪枝或压缩的左右因子矩阵。

        返回:
            appendU: 更新后的左因子矩阵。
            appendV: 更新后的右因子矩阵。
        """
        sqrtSigma = torch.sqrt(self.truc_sigma) # 对奇异值矩阵Σ进行平方根处理
        self.appendU = self.updated_uT.t().matmul(sqrtSigma) # 经过最小二乘法更新后的U矩阵来计算更新后的左因子矩阵
        self.appendV = sqrtSigma.matmul(self.truc_v) # 经过最小二乘法更新后的V矩阵来计算更新后的右因子矩阵
        return self.appendU, self.appendV

# 如果该文件是主程序执行的文件（而不是被导入为模块），则执行下面的代码
if __name__ == '__main__':

    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加命令行参数：模型路径（默认为指定的LLaMA模型）
    parser.add_argument('--model', type=str, default='/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    
    # 添加命令行参数：本地压缩模型路径或白化信息路径
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    
    # 添加命令行参数：目标压缩比率，默认为0.2，表示压缩约20%的参数
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    
    # 添加命令行参数：是否在低资源环境下运行白化，尤其是压缩LLaMA-7B到15G以下的GPU
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    
    # 添加命令行参数：用于提取校准数据集的来源，默认为 'wikitext2'
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Where to extract calibration data from [wikitext2, ptb, c4]')
    
    # 添加命令行参数：白化的校准数据样本数量，默认为256
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    
    # 添加命令行参数：用于更新的校准数据样本数量，默认为16
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    
    # 添加命令行参数：保存压缩后的模型检查点的路径
    parser.add_argument('--save_path', type=str, default='compress_output', help='the path to save the compressed model checkpoints.`')
    
    # 添加命令行参数：本地路径，用于加载配置文件或配置矩阵
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    
    # 添加命令行参数：设置种子，以便用于校准数据采样
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    
    # 添加命令行参数：设置设备（默认值为 'cuda'，表示使用GPU）
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    
    # 添加命令行参数：设置LLaMA模型的默认序列长度
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    
    # 添加命令行参数：设置推理时的批处理大小
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference batch size')
    
    # 添加命令行参数：生成序列的长度，用于效率评估
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    
    # 添加命令行参数：压缩步骤的频率（默认是每4步）
    parser.add_argument('--step', type=int, default=None, help='the step to run the compression')
    
    # 添加命令行参数：LoRA更新的权重路径，用于精度评估
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    
    # 添加命令行参数：是否在 step 2/3 中使用动态压缩比
    parser.add_argument('--use_dynamic_ratio', action='store_true', help='use dynamic compression ratio allocation in step 2/3')

    # 解析命令行参数
    args = parser.parse_args()

    # 反转压缩比率，因为传入的比率是目标截断的比例，而代码实现时使用的是需要保留的比例
    args.ratio = 1 - args.ratio

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"current_time is: {current_time.replace('-', '_')}")

    # 根据步骤数选择不同的操作流程
    if args.step == 1:
        # 获取模型和分词器
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()  # 设置为评估模式

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        # 如果没有指定profiling矩阵路径，则进行白化前的准备工作
        if args.profiling_mat_path is None:
            # 获取校准数据
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            # 生成profiling矩阵
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            
            # 保存profiling矩阵
            if args.save_path is not None:
                # 创建目录（如果不存在）
                os.makedirs(args.save_path, exist_ok=True)
                
                torch.save(profiling_mat, f"{args.save_path}/{args.model.replace('/', '_').replace('-', '_')}_profiling_{args.dataset}_{args.whitening_nsamples}_{args.seed}.pt")
                # 打印保存路径
                print(f"Saved profiling matrix to {args.save_path}/{args.model.replace('/', '_').replace('-', '_')}_profiling_{args.dataset}_{args.whitening_nsamples}_{args.seed}.pt")
        else:
            # 如果指定了profiling矩阵路径，则直接加载
            profiling_mat = torch.load(args.profiling_mat_path, weights_only=False)

        # print(f"Profiling Mat: {profiling_mat}")


        print(f"start svd whtening:")

        # 执行白化操作,SVD低秩近似
        whitening(args.model, model, profiling_mat, args.ratio, args.DEV)
        
        # 如果指定了保存路径，则保存白化后的模型和tokenizer
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, f"{args.save_path}/{args.model.replace('/', '_').replace('-', '_')}_whitening_only_{args.ratio}.pt")

            # 打印保存路径
            print(f"Saved whitening model to {args.save_path}/{args.model.replace('/', '_').replace('-', '_')}_whitening_only_{args.ratio}.pt")

    elif args.step == 2:
        # 如果步骤是2，执行模型更新和白化操作
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        # 获取数据加载器，用于更新模型
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()  # 设置为评估模式
        model = model.float()  # 设置为float精度，白化和更新时需要保持高精度
        
        # 检查并创建保存路径
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        # 如果没有指定profiling矩阵路径，则重新生成profiling矩阵
        if args.profiling_mat_path is None:
            # 获取校准数据
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            
            # 保存profiling矩阵
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            # 如果指定了profiling矩阵路径，则直接加载
            profiling_mat = torch.load(args.profiling_mat_path, weights_only=False)
        
        # 执行白化和本地更新操作
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV, use_dynamic_ratio=args.use_dynamic_ratio)
        
        # 如果指定了保存路径，则保存更新后的模型和tokenizer
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')  # fp32

    elif args.step == 3:
        # 如果步骤是3，执行模型更新操作
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()  # 设置为评估模式
        model = model.float()  # 设置为float精度
        
        # 检查并创建保存路径
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        # 获取更新数据加载器
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        
        # 执行本地更新操作，直接更新模型
        # Step 3 不使用白化矩阵，所以 use_dynamic_ratio 对 step 3 无效（因为动态压缩比依赖白化矩阵）
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True, use_dynamic_ratio=False)
        
        # 如果指定了保存路径，则保存更新后的模型和tokenizer
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')   # fp32

    elif args.step >= 4:
        # 如果步骤大于或等于4，开始模型评估
        print(f"evaluating {args.model_path}...")
        
        # 如果模型路径是“original”，从Hugging Face加载模型，否则加载本地模型
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            
            # 如果指定了LoRA权重，则合并LoRA模型权重
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                # 检查并创建LoRA保存路径
                if not os.path.exists(args.lora):
                    os.makedirs(args.lora)
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        
        model.eval()  # 设置为评估模式
        model = model.float()  # 设置为float精度
        model = model.to(args.DEV)  # 将模型转移到指定设备
        
        # 根据步骤执行不同的评估
        if args.step == 4:
            # 执行困惑度评估
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            # 执行效率评估
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)

