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
            adds = torch.matmul(inp.transpose(1,2), inp)
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
            # 逐个子层：把累计矩阵拿回来（转为 double 精度，搬到 GPU），准备做 Cholesky。
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
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            
            # 使用该权重矩阵的专属压缩比率
            ratio = allocated_ratios.get((i, name), target_ratio)

            print(f"Layer {i}, {name} ratio: {ratio:.4f}")
            
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix) # 经过SVD分解计算的白化矩阵L可以直接计算逆矩阵
            
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            
            # 二次SVD分解
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            
            # 使用自适应压缩比率计算截断逻辑
            # ratio 表示基于参数量的压缩比，确保压缩后参数量 = 原始参数量 × ratio
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            num_s_after_trunc = max(1, num_s_after_trunc)  # 至少保留1个
            
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            
            # 分解为两个低秩矩阵
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            
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

    