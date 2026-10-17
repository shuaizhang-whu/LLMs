import torch
import numpy as np
from tqdm import tqdm
import time
import itertools
from utils.data_utils import get_test_data
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

@torch.no_grad()
def ppl_eval(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()
    ppls = {}
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size = batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            output = model(batch, use_cache=False)
            lm_logits = output.logits
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
        if not nlls:
            print(f"Warning: all batches had non-finite logits for {dataset}! PPL = NaN")
            ppls[dataset] = float('nan')
            continue
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
    print("PPL after pruning: {}".format(ppls))
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

# only call this function when for 65b or more model    
@torch.no_grad()
def ppl_eval_large(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], seq_len=2048, batch_size=1, device="cuda"):
    """逐层 PPL 评估：模型常驻 CPU，每次只将一层搬上 GPU，适用于超过单卡显存的大模型。"""
    import torch.nn as nn
    # 将 lm_head 搬到 GPU 并保持（参数量小，约几百 MB）
    lm_head = model.lm_head.to(device)
    model.eval()
    ppls = {}
    layers = model.model.layers
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=seq_len, batch_size=batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            # ── 阶段1：用 Catcher 捕获第0层输入 ──
            model.model.embed_tokens = model.model.embed_tokens.to(device)
            layers[0] = layers[0].to(device)

            dtype = next(iter(model.parameters())).dtype
            inps = torch.zeros(
                (batch.shape[0], model.seqlen, model.config.hidden_size), dtype=dtype, device=device
            )
            cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
            class Catcher(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, **kwargs):
                    inps[cache['i']] = inp
                    cache['i'] += 1
                    if cache['attention_mask'] is None:
                        cache['attention_mask'] = kwargs.get('attention_mask')
                        cache['position_ids'] = kwargs.get('position_ids')
                    raise ValueError
            layers[0] = Catcher(layers[0])
            for j in range(batch.shape[0]):
                try:
                    model(batch[j].unsqueeze(0).to(device))
                except ValueError:
                    pass
            layers[0] = layers[0].module
            layers[0] = layers[0].cpu()
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            torch.cuda.empty_cache()

            attention_masks = cache['attention_mask']
            position_ids = cache['position_ids']

            # ── 阶段2：逐层前向传播 ──
            for i in range(len(layers)):
                layer = layers[i].to(device)
                fw_kwargs = {}
                if attention_masks is not None:
                    fw_kwargs['attention_mask'] = attention_masks
                if position_ids is not None:
                    fw_kwargs['position_ids'] = position_ids
                outs = layer(inps, **fw_kwargs)[0]
                layers[i] = layer.cpu()
                inps = outs
                torch.cuda.empty_cache()

            # ── 阶段3：norm + lm_head，使用真实模型 norm 权重 ──
            model.model.norm = model.model.norm.to(device)
            hidden_states = model.model.norm(outs)
            model.model.norm = model.model.norm.cpu()
            lm_logits = lm_head(hidden_states)

            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous().to(device)
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
            else:
                print("warning: nan or inf in lm_logits")
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
    # 将 lm_head 归还 CPU
    model.lm_head = lm_head.cpu()
    print("PPL after pruning: {}".format(ppls))
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda"):
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    weight_memory = torch.cuda.memory_allocated()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
            throughput += end_time - start_time
            print("time: {}".format(end_time - start_time))
    print("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
    print("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
    print("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
    print("Throughput: {} tokens/sec".format(token_num / throughput))

