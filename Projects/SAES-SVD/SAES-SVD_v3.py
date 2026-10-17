#coding:utf8
"""
SAES-SVD v3：多模型适配版。

在 SAES-SVD v2.1.4 的单阶段闭环压缩算法基础上，借鉴 FR-SVD v1.5 的
模型无关适配层：
  1. 使用 FactorizedLinear 原位替换任意 nn.Linear，不再依赖硬编码的
     SVD_LlamaAttention / SVD_MistralAttention / SVDOPTDecoderLayer。
  2. 支持 Qwen2.5-7B 的 GQA、q/k/v bias，以及 Qwen3 系列的
     position_embeddings / rotary_emb 传参。
  3. 支持可选跳过 MoE experts，用于 Qwen3.5-35B-A3B 这类 MoE 模型。
  4. 保留 SAES-SVD v2.1.4 的默认参数：rho=0.95, alpha=[0.4, 0.6]。
"""

import os
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
import sys
import argparse
import inspect
import hashlib
import math
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

from utils.data_utils import *
from utils.model_utils import *
from evaluater import *

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# ==================== 模型无关的低秩线性层 ====================

class FactorizedLinear(nn.Module):
    """
    通用低秩分解 Linear 层：y = u_proj(v_proj(x)) (+ bias)
    - v_proj: (in_features → rank), 无 bias
    - u_proj: (rank → out_features), 可选 bias（继承自原 Linear）
    原位替换任意 nn.Linear 而不触动外层 Attention/MLP 前向逻辑，
    天然支持 LLaMA / Qwen2 (GQA+bias) / Qwen3 MoE 等各种架构。
    """
    def __init__(self, in_features, out_features, rank, bias=False,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.v_proj = nn.Linear(in_features, rank, bias=False,
                                 dtype=dtype, device=device)
        self.u_proj = nn.Linear(rank, out_features, bias=bias,
                                 dtype=dtype, device=device)

    def forward(self, x):
        return self.u_proj(self.v_proj(x))


# ==================== 模型适配器 ====================

# 缓存每个 layer 类型的 forward 签名，避免每次调用都做 inspect。
# key: layer 类对象；value: (allowed_param_names_set 或 None, has_var_kwargs_bool)
_FW_SIG_CACHE = {}


def _filter_fw_kwargs(layer, fw_kwargs):
    """
    按 layer.forward 的签名过滤 fw_kwargs。

    场景：Qwen3.5-35B-A3B 的 hybrid attention 中，不同层（full-attn / linear-attn /
    mlp）接受的 kwargs 集合不同。统一捕获的 attention_mask / position_ids /
    position_embeddings 一股脑传进去时，部分实现会抛
    `TypeError: forward() got an unexpected keyword argument 'X'`。

    策略：
    - 若 forward 接受 **kwargs，原样返回（让层自己消化）。
    - 否则只保留出现在签名里的 key。
    - 若 inspect 失败（C 实现等），保守起见原样返回。
    """
    cls = type(layer)
    info = _FW_SIG_CACHE.get(cls)
    if info is None:
        try:
            sig = inspect.signature(layer.forward)
            params = sig.parameters
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            allowed = set(params.keys())
            info = (allowed, has_var_kw)
        except (ValueError, TypeError):
            info = (None, True)
        _FW_SIG_CACHE[cls] = info
    allowed, has_var_kw = info
    if has_var_kw or allowed is None:
        return fw_kwargs
    return {k: v for k, v in fw_kwargs.items() if k in allowed}


def _scan_layer_signatures(layers, captured_kwargs):
    """
    扫描每种 layer 类型的 forward 签名，打印出来用于人工诊断。
    captured_kwargs: StudentCatcher 抓到、后续会回放的 kwarg 名集合。

    输出三类信息：
    - 每种 layer 的位置参数 / VAR_KEYWORD 情况
    - 我们打算回放的 kwargs 中，哪些不在该层签名里（且该层没有 **kwargs 兜底）
    - 该层是否声明了我们没捕获的"必填"参数（强信号：会触发 TypeError）
    """
    print("  扫描 layer.forward 签名（hybrid-attention 兼容性诊断）...")
    seen = {}
    for i, lyr in enumerate(layers):
        cls = type(lyr)
        if cls in seen:
            continue
        seen[cls] = i
        try:
            sig = inspect.signature(lyr.forward)
        except (ValueError, TypeError) as e:
            print(f"    [L{i}] {cls.__name__}: 无法 inspect ({e})")
            continue
        params = sig.parameters
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        allowed = set(params.keys())
        param_str = ", ".join(params.keys())
        print(f"    [L{i}] {cls.__name__}: forward({param_str})")

        # 我们将回放但该层不接受的 kwargs（仅在没有 **kwargs 时才会真崩）
        unaccepted = [k for k in captured_kwargs if k not in allowed]
        if unaccepted and not has_var_kw:
            print(f"        ⚠ 不接受我们回放的 kwargs: {unaccepted} —— 已自动按签名过滤")

        # 该层声明了"必填"且不在我们捕获范围里的参数（强信号）
        required_missing = [
            name for name, p in params.items()
            if name not in ('self', 'hidden_states', 'inp', 'x')
            and p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           inspect.Parameter.KEYWORD_ONLY)
            and name not in captured_kwargs
        ]
        if required_missing:
            print(
                f"        ⚠ 必填但未捕获: {required_missing} —— "
                f"若 forward 内部确实使用，会抛 TypeError"
            )


def _detect_arch_family(model):
    """从模型配置推断架构家族。"""
    cfg = model.config
    mt = (getattr(cfg, 'model_type', '') or '').lower()
    arch = ''
    if getattr(cfg, 'architectures', None):
        arch = (cfg.architectures[0] or '').lower()
    key = f"{mt}|{arch}|{type(model).__name__.lower()}"

    if 'opt' in key:
        return 'opt'
    if 'mistral' in key:
        return 'mistral'
    if 'qwen3' in key and ('moe' in key or 'a3b' in key):
        return 'qwen3_moe'
    if 'qwen' in key:
        return 'qwen'
    if 'llama' in key or 'vicuna' in key:
        return 'llama'
    return 'llama'  # 默认按 LLaMA 结构处理


def _default_runtime_dtype_for_arch(arch_family):
    """Qwen 系列默认用 bf16；其他模型沿用 fp16 路径。"""
    if arch_family in ('qwen', 'qwen3_moe'):
        return torch.bfloat16
    return torch.float16


def _dtype_name(dtype):
    if dtype == torch.bfloat16:
        return 'bf16'
    if dtype == torch.float16:
        return 'fp16'
    if dtype == torch.float32:
        return 'fp32'
    return str(dtype)


def _resolve_eval_dtype(model, requested):
    """解析评估 dtype；auto 时按模型架构选择默认 dtype。"""
    if requested == 'fp32':
        return torch.float32
    if requested == 'bf16':
        return torch.bfloat16
    if requested == 'fp16':
        return torch.float16
    return _default_runtime_dtype_for_arch(_detect_arch_family(model))


def _init_cast_stats(label, target_dtype):
    return {
        'label': label,
        'target_dtype': target_dtype,
        'count': 0,
        'pre_nonfinite': 0,
        'post_nonfinite': 0,
        'pre_min': None,
        'pre_max': None,
        'pre_max_abs': 0.0,
        'post_min': None,
        'post_max': None,
        'post_max_abs': 0.0,
        'fp16_overflow_risk': 0,
    }


def _finite_range(tensor):
    """返回有限元素的 (min, max, max_abs, all_finite)。"""
    t = tensor.detach()
    finite = torch.isfinite(t)
    all_finite = bool(finite.all().item())
    if not bool(finite.any().item()):
        return None, None, float('inf'), all_finite
    vals = t[finite].float()
    return vals.min().item(), vals.max().item(), vals.abs().max().item(), all_finite


def _update_cast_stats(stats, before, after):
    pre_min, pre_max, pre_abs, pre_finite = _finite_range(before)
    post_min, post_max, post_abs, post_finite = _finite_range(after)

    stats['count'] += 1
    if not pre_finite:
        stats['pre_nonfinite'] += 1
    if not post_finite:
        stats['post_nonfinite'] += 1
    if stats['pre_min'] is None or (pre_min is not None and pre_min < stats['pre_min']):
        stats['pre_min'] = pre_min
    if stats['pre_max'] is None or (pre_max is not None and pre_max > stats['pre_max']):
        stats['pre_max'] = pre_max
    if stats['post_min'] is None or (post_min is not None and post_min < stats['post_min']):
        stats['post_min'] = post_min
    if stats['post_max'] is None or (post_max is not None and post_max > stats['post_max']):
        stats['post_max'] = post_max
    stats['pre_max_abs'] = max(stats['pre_max_abs'], pre_abs)
    stats['post_max_abs'] = max(stats['post_max_abs'], post_abs)
    if stats['target_dtype'] == torch.float16 and pre_abs > 60000:
        stats['fp16_overflow_risk'] += 1
    return pre_finite, post_finite


def _cast_activation_for_cache(tensor, target_dtype, stats):
    """写入闭环 CPU cache 前统一 cast，并记录 cast 前后数值范围。"""
    casted = tensor.detach().to(target_dtype)
    pre_finite, post_finite = _update_cast_stats(stats, tensor, casted)
    return casted.cpu(), pre_finite, post_finite


def _format_stat_value(value):
    if value is None:
        return 'None'
    return f'{value:.4e}'


def _print_cast_stats(stats, layer_idx):
    print(
        f"  L{layer_idx} {stats['label']} cast诊断: "
        f"target={_dtype_name(stats['target_dtype'])}, samples={stats['count']}, "
        f"pre_nonfinite={stats['pre_nonfinite']}, post_nonfinite={stats['post_nonfinite']}, "
        f"pre_min={_format_stat_value(stats['pre_min'])}, "
        f"pre_max={_format_stat_value(stats['pre_max'])}, "
        f"pre_max_abs={_format_stat_value(stats['pre_max_abs'])}, "
        f"post_min={_format_stat_value(stats['post_min'])}, "
        f"post_max={_format_stat_value(stats['post_max'])}, "
        f"post_max_abs={_format_stat_value(stats['post_max_abs'])}"
    )
    if stats['fp16_overflow_risk'] > 0:
        print(
            f"  警告: L{layer_idx} {stats['label']} fp16_overflow_risk="
            f"{stats['fp16_overflow_risk']}/{stats['count']} "
            f"(pre_max_abs>60000 且 cache dtype=fp16)"
        )


def get_model_access(model, arch_family):
    """
    统一获取 model.layers / 嵌入 / 归一化等入口。
    OPT 与 LLaMA 家族结构不同，其他（LLaMA/Qwen2/Qwen3）都走 model.model.layers。
    """
    if arch_family == 'opt':
        dec = model.model.decoder
        return {
            'layers': dec.layers,
            'to_dev': [
                ('model.model.decoder.embed_tokens', dec.embed_tokens),
                ('model.model.decoder.final_layer_norm', dec.final_layer_norm),
                ('model.model.decoder.embed_positions', dec.embed_positions),
            ],
        }
    base = model.model
    to_dev = [
        ('model.model.embed_tokens', base.embed_tokens),
        ('model.model.norm', base.norm),
    ]
    # Qwen3 MoE 有 rotary_emb 等额外子模块，如果存在也搬到设备上
    if hasattr(base, 'rotary_emb') and base.rotary_emb is not None:
        to_dev.append(('model.model.rotary_emb', base.rotary_emb))
    return {
        'layers': base.layers,
        'to_dev': to_dev,
    }


def filter_linears_for_compression(linears, skip_moe_experts=False, min_dim=64):
    """
    过滤不适合压缩的 Linear：
    - 维度太小（min(in, out) < min_dim）：压缩后秩过小，收益低甚至劣化。
      典型如 Qwen3 MoE 的 `mlp.shared_expert_gate` (out=1)、
      `linear_attn.in_proj_a/b` (out=32)、MoE router (out=num_experts)。
    - skip_moe_experts=True 时跳过所有 `experts.<i>.*` 子模块。
    """
    filtered = {}
    for name, mod in linears.items():
        if min(mod.in_features, mod.out_features) < min_dim:
            continue
        if skip_moe_experts and 'experts.' in name:
            continue
        filtered[name] = mod
    return filtered


def get_parent_and_attr(root, dotted_name):
    """根据 dotted_name（如 'self_attn.q_proj'）返回 (parent_module, final_attr_name)。"""
    parts = dotted_name.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _load_fast_tokenizer_from_json(model_id):
    """
    旧版 transformers 可能没有 Qwen2Tokenizer/Qwen2TokenizerFast 类。
    对带 tokenizer.json 的模型，直接用 PreTrainedTokenizerFast 构造本地 fast tokenizer。
    """
    from transformers import PreTrainedTokenizerFast

    tokenizer_file = os.path.join(model_id, 'tokenizer.json')
    tokenizer_config_file = os.path.join(model_id, 'tokenizer_config.json')
    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(f"未找到 tokenizer.json: {tokenizer_file}")

    tok_cfg = {}
    if os.path.isfile(tokenizer_config_file):
        with open(tokenizer_config_file, 'r', encoding='utf-8') as f:
            tok_cfg = json.load(f)

    tok_kwargs = {
        'tokenizer_file': tokenizer_file,
        'eos_token': tok_cfg.get('eos_token'),
        'pad_token': tok_cfg.get('pad_token') or tok_cfg.get('eos_token'),
        'unk_token': tok_cfg.get('unk_token'),
        'bos_token': tok_cfg.get('bos_token'),
        'additional_special_tokens': tok_cfg.get('additional_special_tokens', []),
        'model_max_length': tok_cfg.get('model_max_length'),
    }
    tok_kwargs = {k: v for k, v in tok_kwargs.items() if v is not None}
    tokenizer = PreTrainedTokenizerFast(**tok_kwargs)

    for attr in ('padding_side', 'truncation_side'):
        value = tok_cfg.get(attr)
        if value is not None:
            setattr(tokenizer, attr, value)
    if tok_cfg.get('chat_template') is not None:
        tokenizer.chat_template = tok_cfg['chat_template']

    return tokenizer


def load_model_v15(model_id, seq_len=2048):
    """
    v1.5 的 HuggingFace 模型加载器：按模型名自动选择正确的 tokenizer。
    - 有 tokenizer.json（Qwen / Mistral / OPT 等）：AutoTokenizer
    - 只有 tokenizer.model（旧 LLaMA-7B / LLaMA-13b / LLaMA-30b / Llama-2）：
      LlamaTokenizer（transformers v5 的 AutoTokenizer 加载纯 sentencepiece
      文件时存在递归 bug，必须显式指定）
    """
    import os
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
    import transformers

    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        cfg_key = (
            f"{getattr(cfg, 'model_type', '')}|"
            f"{','.join(getattr(cfg, 'architectures', []) or [])}"
        ).lower()
        load_dtype = torch.bfloat16 if 'qwen' in cfg_key else torch.float16
    except Exception as e:
        local_model_type = None
        local_config = os.path.join(model_id, 'config.json')
        if os.path.isfile(local_config):
            try:
                with open(local_config, 'r', encoding='utf-8') as f:
                    local_model_type = json.load(f).get('model_type')
            except Exception:
                local_model_type = None
        if local_model_type and 'qwen' in local_model_type.lower():
            raise RuntimeError(
                "当前 Python/Transformers 环境不支持 Qwen 配置加载："
                f"model_type={local_model_type!r}, transformers={transformers.__version__}, "
                f"python={sys.executable}。请使用 "
                "/data/zhangs/anaconda3/envs/compress_Qwen3.5/bin/python 运行。"
            ) from e
        print(f"  配置读取失败 ({e})，默认使用 fp16 加载")
        load_dtype = torch.float16

    # 检测是否存在 tokenizer.json；没有则必须用 LlamaTokenizer 避免 transformers v5 递归 bug
    has_tok_json = os.path.isfile(os.path.join(model_id, 'tokenizer.json'))
    force_llama_tok = (not has_tok_json)

    tok_cls = LlamaTokenizer if force_llama_tok else AutoTokenizer
    print(f"  Tokenizer: {tok_cls.__name__} (tokenizer.json={'found' if has_tok_json else 'not found'})")
    print(f"  模型加载 dtype: {_dtype_name(load_dtype)}")
    # use_fast=False：transformers v5 的 fast tokenizer 在纯 sentencepiece 模型上
    # 存在 unk_token_id 递归 bug，slow tokenizer 无此问题
    tok_kwargs = {"use_fast": False} if force_llama_tok else {}
    try:
        tokenizer = tok_cls.from_pretrained(model_id, trust_remote_code=True, **tok_kwargs)
    except Exception as e:
        print(f"  {tok_cls.__name__} 加载失败 ({e})")
        if has_tok_json:
            print("  回退到 PreTrainedTokenizerFast(tokenizer.json)")
            tokenizer = _load_fast_tokenizer_from_json(model_id)
        else:
            print("  回退到 AutoTokenizer use_fast=False")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    print(
        f"  Tokenizer loaded: {tokenizer.__class__.__name__}, "
        f"vocab_size={getattr(tokenizer, 'vocab_size', 'unknown')}, len={len(tokenizer)}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", torch_dtype=load_dtype,
        trust_remote_code=True, cache_dir=None
    )
    model.seqlen = seq_len
    return model, tokenizer


def _safe_cache_component(value, max_len=80):
    """构造适合放进缓存文件名的短标识。"""
    text = str(value)
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text)
    return safe[:max_len] if safe else 'unknown'


def _tokenizer_vocab_size(tokenizer):
    """优先使用 tokenizer.vocab_size，失败时回退到 len(tokenizer)。"""
    vocab_size = getattr(tokenizer, 'vocab_size', None)
    if vocab_size is None:
        try:
            vocab_size = len(tokenizer)
        except TypeError:
            vocab_size = 'unknown'
    return vocab_size


def _validate_calib_token_ids(calib_data, model_vocab_size, source):
    """在进入 CUDA embedding 前检查校准 token id，避免 device-side assert。"""
    if not calib_data:
        raise ValueError(f"校准数据为空: {source}")

    min_id = None
    max_id = None
    for batch in calib_data:
        input_ids = batch.get('input_ids')
        if input_ids is None:
            raise ValueError(f"校准 batch 缺少 input_ids: {source}")
        cur_min = int(input_ids.min().item())
        cur_max = int(input_ids.max().item())
        min_id = cur_min if min_id is None else min(min_id, cur_min)
        max_id = cur_max if max_id is None else max(max_id, cur_max)

    print(f"  校准 token id 范围: min={min_id}, max={max_id}, model_vocab_size={model_vocab_size}")
    if min_id < 0 or (model_vocab_size is not None and max_id >= model_vocab_size):
        raise ValueError(
            "校准数据 token id 超出当前模型 embedding 范围，"
            f"source={source}, min_id={min_id}, max_id={max_id}, "
            f"model_vocab_size={model_vocab_size}. "
            "这通常是不同 tokenizer 共用了旧缓存导致的。"
        )
    return min_id, max_id


def get_calib_train_data_v3_safe(name, tokenizer, nsamples, seqlen=2048,
                                 seed=3, batch_size=1, dataset_cache_dir=None,
                                 model_id=None, model_vocab_size=None):
    """
    SAES-SVD v3 专用校准数据加载器。

    与 utils.data_utils.get_calib_train_data 的生成逻辑保持一致，但缓存文件名
    加入 tokenizer/model 指纹，避免 Qwen 与 LLaMA 等模型误读彼此的缓存。
    """
    import random
    from datasets import load_dataset

    random.seed(seed)
    tokenizer_cls = tokenizer.__class__.__name__
    tokenizer_vocab = _tokenizer_vocab_size(tokenizer)
    model_hash = hashlib.sha1(str(model_id or '').encode('utf-8')).hexdigest()[:10]
    cache_name = (
        f"saes_svd_v3_{_safe_cache_component(name)}_{nsamples}_{seqlen}_{seed}_{batch_size}_"
        f"{_safe_cache_component(tokenizer_cls)}_tokv{_safe_cache_component(tokenizer_vocab)}_"
        f"modelv{_safe_cache_component(model_vocab_size)}_m{model_hash}.pt"
    )
    cache_dir = os.path.join(current_path, "cache")
    cache_file = os.path.join(cache_dir, cache_name)
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_file):
        print(f"  读取 SAES-SVD v3 专属校准缓存: {cache_file}")
        traindataset = torch.load(cache_file, weights_only=False)
        _validate_calib_token_ids(traindataset, model_vocab_size, cache_file)
        return traindataset

    print(f"  生成 SAES-SVD v3 专属校准缓存: {cache_file}")
    original_nsamples = nsamples
    nsamples += 1

    if name == "c4":
        traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        traindata = load_dataset(
            'ptb_text_only', 'penn_treebank', split='train',
            cache_dir=dataset_cache_dir
        )
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "wikitext2":
        traindata = load_dataset(
            "/data/zhangs/LLMs/Datasets/wikitext", "wikitext-2-raw-v1",
            split="train", cache_dir=dataset_cache_dir
        )
        tot_text = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError

    traindataset = []
    for s in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if s % batch_size == 0:
            if s != 0:
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)

    if len(traindataset) != original_nsamples:
        print(f"  Warning: 期望 {original_nsamples} 个校准 batch，实际生成 {len(traindataset)} 个。")

    _validate_calib_token_ids(traindataset, model_vocab_size, cache_file)
    torch.save(traindataset, cache_file)
    return traindataset


def replace_linear_with_factorized(root, dotted_name, B, A, bias_data,
                                    device, dtype=torch.float32):
    """
    用 FactorizedLinear 原位替换 root 下 dotted_name 处的 nn.Linear。
    B: (m, r), A: (r, n), bias_data: (m,) 或 None
    v_proj.weight ← A （形状 (rank=r, in=n)）
    u_proj.weight ← B （形状 (out=m, rank=r)）
    """
    parent, attr = get_parent_and_attr(root, dotted_name)
    old_linear = getattr(parent, attr)
    in_features = old_linear.in_features
    out_features = old_linear.out_features
    rank = A.shape[0]
    has_bias = (bias_data is not None)

    new_mod = FactorizedLinear(
        in_features=in_features, out_features=out_features, rank=rank,
        bias=has_bias, dtype=dtype, device=device
    )
    new_mod.v_proj.weight.data.copy_(A.to(dtype))
    new_mod.u_proj.weight.data.copy_(B.to(dtype))
    if has_bias:
        new_mod.u_proj.bias.data.copy_(bias_data.to(dtype))

    setattr(parent, attr, new_mod)
    return new_mod


# ==================== 辅助函数 ====================

def compute_H_decomposition(H_nodamp, device, eps=1e-4):
    """
    对输入协方差矩阵 H 做阻尼 + 特征分解，返回 H^{1/2} 和 H^{-1/2}。
    d <= 4096 在 GPU 上做 eigh；d > 4096 在 CPU 上做（eigh 工作空间 ~O(d^2)，
    大矩阵在共享 GPU 上易 OOM），结果搬回 GPU。
    """
    d = H_nodamp.shape[0]
    if d > 4096:
        H_reg = H_nodamp.cpu().float() + eps * torch.eye(d, dtype=torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)
    else:
        H_reg = H_nodamp.to(device).float() + eps * torch.eye(d, device=device, dtype=torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)

    eigenvalues = torch.clamp(eigenvalues, min=eps)
    sqrt_lam = torch.sqrt(eigenvalues)
    inv_sqrt_lam = 1.0 / sqrt_lam

    Q = eigenvectors.to(device)
    Lambda = eigenvalues.to(device)
    H_sqrt = (Q * sqrt_lam.to(device).unsqueeze(0)) @ Q.T
    H_inv_sqrt = (Q * inv_sqrt_lam.to(device).unsqueeze(0)) @ Q.T

    return Q, Lambda, H_sqrt, H_inv_sqrt


def truncated_svd_factorize(W, rank):
    """
    纯截断 SVD 基线分解，用作数值安全护栏。
    当 FR-SVD/CEALC 结果在某些层上虽然“有限”但重建误差明显劣于
    plain truncated SVD 时，优先采用后者，避免保存语义已崩坏的层。
    """
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    sqrt_s = torch.sqrt(torch.clamp(S[:rank], min=0))
    B = U[:, :rank] * sqrt_s.unsqueeze(0)
    A = sqrt_s.unsqueeze(1) * Vt[:rank, :]
    W_recon = B @ A
    weight_err = (torch.norm(W - W_recon) / torch.norm(W)).item()
    return B, A, weight_err


def estimate_kronecker_fisher(layer, teacher_layer, inps_student, inps_teacher,
                               attention_masks, position_ids, model_name, dev,
                               num_samples, lambda_psi=1e-4, max_fisher_samples=32,
                               position_embeddings=None):
    """
    估计每个线性子层的输出端 Fisher 因子 Ψ = E[g g^T]。
    返回 {name: (eigvecs_psi, eigvals_psi)} 用于后续 Fisher 加权。
    """
    subset = find_layers(layer)
    psi_accum = {}
    input_cache = {}

    for name in subset:
        n_out = subset[name].out_features
        psi_accum[name] = torch.zeros(n_out, n_out, dtype=torch.float32)

    def make_hooks(name):
        def forward_hook(module, inp, out):
            a = inp[0].detach().float()
            if a.dim() == 3:
                a = a.reshape(-1, a.shape[-1])
            input_cache[name] = a

        def backward_hook(module, grad_input, grad_output):
            b = grad_output[0].detach().float()
            if b.dim() == 3:
                b = b.reshape(-1, b.shape[-1])
            psi_accum[name] += (b.T @ b).cpu()

        return forward_hook, backward_hook

    handles = []
    for name in subset:
        fh, bh = make_hooks(name)
        handles.append(subset[name].register_forward_hook(fh))
        handles.append(subset[name].register_full_backward_hook(bh))

    for p in layer.parameters():
        p.requires_grad_(True)

    n_fisher = min(num_samples, max_fisher_samples)

    for j in range(n_fisher):
        inp_j = inps_student[j].unsqueeze(0).to(dev).float()
        teacher_inp_j = inps_teacher[j].unsqueeze(0).to(dev).float()
        attn_mask = attention_masks[j].to(dev) if attention_masks[j] is not None else None
        pos_ids = position_ids[j].to(dev) if position_ids is not None else None

        layer.zero_grad()
        inp_for_grad = inp_j.detach().requires_grad_(True)

        if "opt" not in model_name:
            fw_extra = {}
            if pos_ids is not None:
                fw_extra['position_ids'] = pos_ids
            if position_embeddings is not None:
                fw_extra['position_embeddings'] = tuple(t.to(dev) for t in position_embeddings)
            out_student = layer(inp_for_grad, attention_mask=attn_mask, **fw_extra)[0]
        else:
            out_student = layer(inp_for_grad, attention_mask=attn_mask)[0]

        with torch.no_grad():
            out_teacher = teacher_layer.float()(
                teacher_inp_j,
                attention_mask=attn_mask,
                **(fw_extra if "opt" not in model_name else {})
            )[0]
            teacher_layer.half()

        loss = nn.functional.mse_loss(out_student, out_teacher.detach())
        loss.backward()

        del inp_j, teacher_inp_j, out_student, out_teacher, loss, inp_for_grad
        input_cache.clear()
        if j % 16 == 0:
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    for p in layer.parameters():
        p.requires_grad_(False)

    result = {}
    for name in subset:
        psi = psi_accum[name] / n_fisher
        psi += lambda_psi * torch.eye(psi.shape[0])
        eigvals_psi, eigvecs_psi = torch.linalg.eigh(psi.float())
        eigvals_psi = torch.clamp(eigvals_psi, min=lambda_psi)
        result[name] = (eigvecs_psi, eigvals_psi)

    del psi_accum
    torch.cuda.empty_cache()
    return result


def aces_beta_select(S, D, rank, alpha_min=0.25, alpha_max=0.75,
                     rho=1.0, beta_cap=None, fisher_weights=None):
    """
    ACES β 选择（论文 Algorithm 3）。

    v1.3 改进：
    - 默认 alpha 范围扩展到 [0.25, 0.75]（论文 Table 4 最优）
    - beta_cap 自动从 alpha_max 计算
    - rho 默认 1.0（不收缩）
    - 可选 fisher_weights：对 RER 目标函数中的各方向施加 Fisher 重要性加权

    参数：
        S            : 权重信号矩阵 W@H^{1/2}，形状 (m, n)
        D            : 误差信号矩阵 W@Δ@H^{-1/2}，形状 (m, n)
        rank         : 目标秩
        alpha_min    : α 下界
        alpha_max    : α 上界
        rho          : 收缩因子
        beta_cap     : β 硬上限（None → 自动从 alpha_max 计算）
        fisher_weights : 可选，形状 (m,) 的 Fisher 重要性权重向量。
                         当提供时，在计算 RER 系数 a/b/c/A/B/C 前，
                         对 S 和 D 的行施加 sqrt(fisher_weights) 缩放。

    返回：
        beta_star  : 最优 β
        info       : 诊断字典
    """
    if beta_cap is None:
        beta_cap = alpha_max / (1 + alpha_max)

    beta_min = alpha_min / (1 + alpha_min)
    beta_max = alpha_max / (1 + alpha_max)

    S = S.float()
    D = D.float()

    # 可选：Fisher 重要性加权
    # 对 S, D 的行乘以 sqrt(fisher_weights)，等价于改变输出方向的度量
    # 这种 soft-reweighting 方式不改变 SVD 截断本身的度量空间，
    # 只在 ACES 的 β 选择目标中引入 Fisher 信息引导
    if fisher_weights is not None:
        fw_sqrt = torch.sqrt(fisher_weights.float()).unsqueeze(1)  # (m, 1)
        S_w = S * fw_sqrt
        D_w = D * fw_sqrt
    else:
        S_w = S
        D_w = D

    # 对 S_w 做 SVD，取前 rank 个左/右奇异向量
    U, Sigma, Vt = torch.linalg.svd(S_w, full_matrices=False)
    U_r = U[:, :rank]
    V_r = Vt[:rank, :].T

    # 正交补投影
    UrTS = U_r.T @ S_w
    SVr = S_w @ V_r
    UrTSVr = U_r.T @ SVr
    S_perp = S_w - U_r @ UrTS - SVr @ V_r.T + U_r @ UrTSVr @ V_r.T

    UrTD = U_r.T @ D_w
    DVr = D_w @ V_r
    UrTDVr = U_r.T @ DVr
    D_perp = D_w - U_r @ UrTD - DVr @ V_r.T + U_r @ UrTDVr @ V_r.T

    # RER 系数
    a = torch.norm(S_perp, 'fro') ** 2
    b = torch.sum(S_perp * D_perp)
    c = torch.norm(D_perp, 'fro') ** 2
    A_coef = torch.norm(S_w, 'fro') ** 2
    B_coef = torch.sum(S_w * D_w)
    C_coef = torch.norm(D_w, 'fro') ** 2

    # 二次方程 (cB - bC)β² + (cA - aC)β + (bA - aB) = 0
    coef_2 = c * B_coef - b * C_coef
    coef_1 = c * A_coef - a * C_coef
    coef_0 = b * A_coef - a * B_coef

    candidates = [beta_min, beta_max]

    if abs(coef_2.item()) > 1e-10:
        # 标准二次情形：求驻点
        discriminant = coef_1 ** 2 - 4 * coef_2 * coef_0
        if discriminant >= 0:
            sqrt_disc = torch.sqrt(discriminant)
            candidates.append((-coef_1 + sqrt_disc).item() / (2 * coef_2.item()))
            candidates.append((-coef_1 - sqrt_disc).item() / (2 * coef_2.item()))
    elif abs(coef_1.item()) > 1e-10:
        # 线性退化情形：coef_2 ≈ 0 时 RER 导数为线性，驻点 = -coef_0/coef_1
        candidates.append(-coef_0.item() / coef_1.item())

    best_beta = beta_min
    best_score = float('inf')

    for beta_cand in candidates:
        beta_val = max(beta_min, min(beta_cand, beta_max))
        if beta_val < 0 or beta_val >= 1:
            continue
        numerator = a + 2 * b * beta_val + c * beta_val ** 2
        denominator = A_coef + 2 * B_coef * beta_val + C_coef * beta_val ** 2 + 1e-10
        score = (numerator / denominator).item()
        if score < best_score:
            best_score = score
            best_beta = beta_val

    # 护栏：rho 收缩 + beta_cap
    best_beta = rho * min(best_beta, beta_cap)
    alpha_star = best_beta / (1 - best_beta) if best_beta < 1 else float('inf')

    return best_beta, {
        'alpha_star': alpha_star,
        'best_score': best_score,
        'a': a.item(), 'b': b.item(), 'c': c.item(),
        'A': A_coef.item(), 'B': B_coef.item(), 'C': C_coef.item()
    }


# ==================== 主压缩函数 ====================

@torch.no_grad()
def saes_svd_v3_compress_layerwise(model_name, model, calib_loader, ratio, dev,
                                   eps=1e-4, rho=0.95,
                                   alpha_min=0.4, alpha_max=0.6,
                                   skip_moe_experts=False, min_dim=64,
                                   verbose=True):
    """
    SAES-SVD v3 逐层闭环压缩。

    算法保持 v2.1.4 的单阶段 SAES：
      G = W @ (H + beta * Delta) @ L, L=(H+epsI)^(-1/2)
    模型替换方式改为 FR-SVD v1.5 的 FactorizedLinear 原位替换，从而兼容
    Qwen2.5/Qwen3 的 GQA、bias、position_embeddings 和 MoE 结构。
    """
    beta_cap = alpha_max / (1 + alpha_max)

    print("=" * 60)
    print("SAES-SVD v3: model-agnostic Qwen-compatible compression")
    print("=" * 60)
    print(f"  保留比例: {ratio} ({ratio*100:.1f}%)")
    print(f"  H 阻尼: eps={eps}")
    print(f"  Alpha 范围: [{alpha_min}, {alpha_max}], rho={rho}, beta_cap={beta_cap:.4f}")
    print(f"  skip_moe_experts: {skip_moe_experts}, min_dim={min_dim}")
    print("=" * 60)

    import copy

    arch_family = _detect_arch_family(model)
    is_opt = (arch_family == 'opt')
    is_moe = (arch_family == 'qwen3_moe')
    activation_cache_dtype = _default_runtime_dtype_for_arch(arch_family)
    print(f"  检测到架构家族: {arch_family}")
    print(f"  闭环 activation cache dtype: {_dtype_name(activation_cache_dtype)}")
    if is_moe:
        print("  Qwen3 MoE: 可通过 --skip_moe_experts 跳过 experts.<i>.*")

    # Qwen2.5 的日志显示：SAES 因子可以保持有限，但把层输出从几十放大到 1e4~1e6。
    # 因此 Qwen 系列恢复 FR-SVD v1.5 的 TSVD 质量护栏；LLaMA 等仍只对极端数值崩坏回退。
    qwen_safety_guard = arch_family in ('qwen', 'qwen3_moe')
    tsvd_guard_margin = 0.98 if qwen_safety_guard else 0.90
    tsvd_guard_min_weight_err = 0.10 if qwen_safety_guard else 0.60
    extreme_blowup_threshold = 1.0e4
    activation_guard_ratio = 10.0
    activation_guard_abs = 1.0e4
    output_rel_l2_guard = 0.50
    output_rms_ratio_min = 0.50
    output_rms_ratio_max = 2.00
    if qwen_safety_guard:
        print(
            "  Qwen safety guard: "
            f"TSVD min_W_err={tsvd_guard_min_weight_err}, margin={tsvd_guard_margin}, "
            f"layer激活阈值=max({activation_guard_ratio}x teacher, {activation_guard_abs:.1e}), "
            f"output_rel_l2阈值={output_rel_l2_guard}, "
            f"output_rms_ratio范围=[{output_rms_ratio_min}, {output_rms_ratio_max}]"
        )

    access = get_model_access(model, arch_family)
    layers = access['layers']

    for _path, mod in access['to_dev']:
        mod.to(dev)
    preamble_mods = [m for _, m in access['to_dev']]

    layers[0] = layers[0].to(dev)
    num_samples = len(calib_loader)

    inps_student = torch.zeros(
        (num_samples, model.seqlen, model.config.hidden_size),
        dtype=activation_cache_dtype, device='cpu'
    )

    cache = {'i': 0}
    attention_masks_list = []
    position_ids_list = []
    position_embeddings_cache = [None]

    class StudentCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps_student[cache['i']] = inp.detach().to(activation_cache_dtype).cpu()
            attn_mask = kwargs.get('attention_mask')
            attention_masks_list.append(attn_mask.detach().cpu() if attn_mask is not None else None)
            if not is_opt:
                pos_ids = kwargs.get('position_ids')
                position_ids_list.append(pos_ids.detach().cpu() if pos_ids is not None else None)
                if position_embeddings_cache[0] is None:
                    pos_emb = kwargs.get('position_embeddings')
                    if pos_emb is not None:
                        position_embeddings_cache[0] = tuple(t.detach().cpu() for t in pos_emb)
            cache['i'] += 1
            raise ValueError

    layers[0] = StudentCatcher(layers[0])

    print("收集第 0 层输入激活...")
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    for mod in preamble_mods:
        mod.cpu()
    torch.cuda.empty_cache()

    inps_teacher = inps_student.clone()
    outs_student = torch.zeros_like(inps_student)
    outs_teacher = torch.zeros_like(inps_student)
    attention_masks = attention_masks_list
    position_ids = position_ids_list if position_ids_list and position_ids_list[0] is not None else None
    pos_embed_cached = position_embeddings_cache[0]

    compression_stats = []

    # ── 一次性诊断：扫描所有 layer 类型的 forward 签名 ──
    # 为 hybrid attention（如 Qwen3.5-35B-A3B 的 full-attn + linear-attn 混合）
    # 提前暴露 kwargs 不匹配。Phase A/F 的 forward 调用统一使用 _filter_fw_kwargs
    # 按签名过滤，避免不接受 'position_embeddings' 等的层报 TypeError。
    _captured = {'attention_mask'}
    if not is_opt:
        _captured.add('position_ids')
        if pos_embed_cached is not None:
            _captured.add('position_embeddings')
    _scan_layer_signatures(layers, _captured)

    print("\n开始逐层 SAES-SVD v3 压缩...")
    for layer_idx in tqdm(range(len(layers)), desc="SAES-SVD v3 Compression"):
        layer_stats = {'layer': layer_idx, 'sublayers': {}}

        layer = layers[layer_idx].to(dev)
        teacher_layer = copy.deepcopy(layer)
        layer = layer.float()

        subset = find_layers(layer)
        teacher_subset = find_layers(teacher_layer)
        subset = filter_linears_for_compression(
            subset, skip_moe_experts=skip_moe_experts, min_dim=min_dim
        )
        teacher_subset = {k: v for k, v in teacher_subset.items() if k in subset}

        if not subset:
            student_cast_stats = _init_cast_stats('student_precompress_no_subset', activation_cache_dtype)
            teacher_cast_stats = _init_cast_stats('teacher_precompress_no_subset', activation_cache_dtype)
            for j in range(num_samples):
                inp_j = inps_student[j].unsqueeze(0).to(dev).float()
                teacher_inp = inps_teacher[j].unsqueeze(0).to(dev).float()
                attn_mask_j = attention_masks[j].to(dev) if attention_masks[j] is not None else None
                pos_ids_j = position_ids[j].to(dev) if position_ids is not None else None

                fw_kwargs = {}
                if attn_mask_j is not None:
                    fw_kwargs['attention_mask'] = attn_mask_j
                if not is_opt and pos_ids_j is not None:
                    fw_kwargs['position_ids'] = pos_ids_j
                if pos_embed_cached is not None:
                    fw_kwargs['position_embeddings'] = tuple(t.to(dev) for t in pos_embed_cached)

                out_student = layer(inp_j, **_filter_fw_kwargs(layer, fw_kwargs))[0]
                out_teacher = teacher_layer.float()(teacher_inp, **_filter_fw_kwargs(teacher_layer, fw_kwargs))[0]
                cached_student, _, _ = _cast_activation_for_cache(
                    out_student, activation_cache_dtype, student_cast_stats
                )
                cached_teacher, _, _ = _cast_activation_for_cache(
                    out_teacher, activation_cache_dtype, teacher_cast_stats
                )
                outs_student[j] = cached_student
                outs_teacher[j] = cached_teacher
                teacher_layer.to(dtype=activation_cache_dtype)
                del inp_j, teacher_inp, out_student, out_teacher, cached_student, cached_teacher

            _print_cast_stats(student_cast_stats, layer_idx)
            _print_cast_stats(teacher_cast_stats, layer_idx)

            inps_student = outs_student.clone()
            inps_teacher = outs_teacher.clone()
            layers[layer_idx] = layer.to(dtype=activation_cache_dtype).cpu()
            del teacher_layer, teacher_subset
            torch.cuda.empty_cache()
            continue

        for name in subset:
            subset[name].H_matrix = 0
            subset[name].Delta_matrix = 0
            subset[name]._tmp_inp = None
            teacher_subset[name]._tmp_inp = None

        def make_hook_student(_module_name):
            def hook(module, input, output):
                inp = input[0].detach().float()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                H_batch = torch.matmul(inp.transpose(1, 2), inp)
                module.H_matrix = module.H_matrix + torch.sum(H_batch, dim=0)
                module._tmp_inp = inp
            return hook

        def make_hook_teacher(_module_name):
            def hook(module, input, output):
                inp = input[0].detach().float()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                module._tmp_inp = inp
            return hook

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_hook_student(name)))
            handles.append(teacher_subset[name].register_forward_hook(make_hook_teacher(name)))

        student_pre_cast_stats = _init_cast_stats('student_precompress', activation_cache_dtype)
        teacher_pre_cast_stats = _init_cast_stats('teacher_precompress', activation_cache_dtype)
        for j in range(num_samples):
            inp_j = inps_student[j].unsqueeze(0).to(dev).float()
            teacher_inp = inps_teacher[j].unsqueeze(0).to(dev).float()
            attn_mask_j = attention_masks[j].to(dev) if attention_masks[j] is not None else None
            pos_ids_j = position_ids[j].to(dev) if position_ids is not None else None

            fw_kwargs = {}
            if attn_mask_j is not None:
                fw_kwargs['attention_mask'] = attn_mask_j
            if not is_opt and pos_ids_j is not None:
                fw_kwargs['position_ids'] = pos_ids_j
            if pos_embed_cached is not None:
                fw_kwargs['position_embeddings'] = tuple(t.to(dev) for t in pos_embed_cached)

            out_j = layer(inp_j, **_filter_fw_kwargs(layer, fw_kwargs))[0]
            teacher_out = teacher_layer.float()(teacher_inp, **_filter_fw_kwargs(teacher_layer, fw_kwargs))[0]
            teacher_layer.to(dtype=activation_cache_dtype)

            cached_student, _, _ = _cast_activation_for_cache(
                out_j, activation_cache_dtype, student_pre_cast_stats
            )
            cached_teacher, _, _ = _cast_activation_for_cache(
                teacher_out, activation_cache_dtype, teacher_pre_cast_stats
            )
            outs_student[j] = cached_student
            outs_teacher[j] = cached_teacher

            for name in subset:
                X = subset[name]._tmp_inp
                X_f = teacher_subset[name]._tmp_inp
                diff = X_f - X
                Delta_batch = torch.matmul(diff.transpose(1, 2), X)
                subset[name].Delta_matrix = subset[name].Delta_matrix + torch.sum(Delta_batch, dim=0)
                subset[name]._tmp_inp = None
                teacher_subset[name]._tmp_inp = None
                del X, X_f, diff, Delta_batch

            del inp_j, teacher_inp, out_j, teacher_out, cached_student, cached_teacher
            if j % 32 == 0:
                torch.cuda.empty_cache()

        _print_cast_stats(student_pre_cast_stats, layer_idx)
        _print_cast_stats(teacher_pre_cast_stats, layer_idx)

        for h in handles:
            h.remove()

        for name in subset:
            subset[name].H_matrix = subset[name].H_matrix.cpu()
            subset[name].Delta_matrix = subset[name].Delta_matrix.cpu()

        del teacher_subset, teacher_layer
        torch.cuda.empty_cache()

        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            original_dtype = subset[name].weight.data.dtype
            m, n = W.shape

            H = subset[name].H_matrix.float().to(dev)
            Delta = subset[name].Delta_matrix.float().to(dev)

            rank = int(m * n * ratio / (m + n))
            rank = max(1, min(rank, min(m, n)))

            if not torch.isfinite(H).all() or not torch.isfinite(Delta).all():
                print(f"  警告: L{layer_idx} {name} H/Delta 含 NaN，使用安全回退")
                H = eps * torch.eye(n, device=dev, dtype=torch.float32)
                Delta = torch.zeros(n, n, device=dev, dtype=torch.float32)

            _Q, _Lambda, _H_sqrt, L = compute_H_decomposition(H, dev, eps)
            del _Q, _Lambda, _H_sqrt

            S = W @ H @ L
            D = W @ Delta @ L

            beta, aces_info = aces_beta_select(
                S, D, rank,
                alpha_min=alpha_min, alpha_max=alpha_max,
                rho=rho, beta_cap=beta_cap
            )

            G = S + beta * D
            U, Sigma, Vt = torch.linalg.svd(G, full_matrices=False)
            U_r = U[:, :rank]
            Sigma_r = Sigma[:rank]
            Vt_r = Vt[:rank, :]

            sqrt_sigma = torch.sqrt(torch.clamp(Sigma_r, min=0))
            B = U_r * sqrt_sigma.unsqueeze(0)
            A = (sqrt_sigma.unsqueeze(1) * Vt_r) @ L

            W_recon = B.float() @ A.float()
            weight_err = (torch.norm(W - W_recon) / torch.norm(W)).item()
            G_recon = U_r @ torch.diag(Sigma_r) @ Vt_r
            G_err = (torch.norm(G - G_recon) / torch.norm(G)).item()
            energy_ratio = (torch.sum(Sigma_r ** 2) / (torch.sum(Sigma ** 2) + 1e-10)).item()

            tsvd_err = None
            fallback_reason = None
            non_finite = (
                (not torch.isfinite(B).all())
                or (not torch.isfinite(A).all())
                or (not math.isfinite(weight_err))
            )
            extreme_blowup = weight_err > extreme_blowup_threshold
            should_compare_tsvd = qwen_safety_guard or non_finite or extreme_blowup

            if should_compare_tsvd:
                B_tsvd, A_tsvd, tsvd_err = truncated_svd_factorize(W, rank)

                if non_finite:
                    fallback_reason = "numerical_instability"
                elif qwen_safety_guard and weight_err > 1.0:
                    fallback_reason = "qwen_weight_err_gt_1"
                elif (
                    qwen_safety_guard
                    and weight_err > tsvd_guard_min_weight_err
                    and tsvd_err < weight_err * tsvd_guard_margin
                ):
                    fallback_reason = "qwen_tsvd_guard"
                elif extreme_blowup:
                    fallback_reason = "extreme_blowup"

            if fallback_reason is not None:
                print(
                    f"  警告: L{layer_idx} {name} SAES 重建触发回退 "
                    f"(W_err={weight_err:.4e}, TSVD={tsvd_err:.4f}, "
                    f"non_finite={non_finite})，回退到截断 SVD [{fallback_reason}]"
                )
                B, A = B_tsvd, A_tsvd
                W_recon = B.float() @ A.float()
                weight_err = (torch.norm(W - W_recon) / torch.norm(W)).item()

            B = B.to(original_dtype)
            A = A.to(original_dtype)

            delta_norm = torch.norm(Delta).item()
            h_norm = torch.norm(H).item()

            layer_stats['sublayers'][name] = {
                'beta': beta,
                'rank': rank,
                'G_trunc_error': G_err,
                'weight_err': weight_err,
                'tsvd_err': tsvd_err,
                'fallback_reason': fallback_reason,
                'energy_ratio': energy_ratio,
                'delta_norm': delta_norm,
                'h_norm': h_norm,
                'delta_ratio': delta_norm / (h_norm + 1e-10),
                'aces_info': aces_info,
            }

            log_sublayer = verbose and (
                qwen_safety_guard
                or layer_idx == 0
                or layer_idx == len(layers) - 1
                or layer_idx % 8 == 0
            )
            if log_sublayer:
                tsvd_str = f"{tsvd_err:.4f}" if tsvd_err is not None else "not_checked"
                fallback_str = fallback_reason if fallback_reason is not None else "no"
                print(
                    f"  L{layer_idx} {name}: β={beta:.4f}, α*={aces_info['alpha_star']:.4f}, "
                    f"rank={rank}, G_err={G_err:.4f}, E_kept={energy_ratio:.4f}, "
                    f"W_err={weight_err:.4f}, TSVD={tsvd_str}, fallback={fallback_str}, "
                    f"Δ/H={delta_norm/(h_norm+1e-10):.4f}"
                )

            bias_data = subset[name].bias.data if subset[name].bias is not None else None
            replace_linear_with_factorized(
                layer, name, B, A, bias_data,
                device=dev, dtype=original_dtype
            )

            if should_compare_tsvd:
                del B_tsvd, A_tsvd
            del W, H, Delta, L, S, D, G, U, Sigma, Vt
            del U_r, Sigma_r, Vt_r, B, A, W_recon, G_recon
            del subset[name].H_matrix, subset[name].Delta_matrix
            torch.cuda.empty_cache()

        avg_beta = sum(s['beta'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        avg_delta_ratio = sum(s['delta_ratio'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        avg_alpha_star = sum(s['aces_info']['alpha_star'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        avg_best_score = sum(s['aces_info']['best_score'] for s in layer_stats['sublayers'].values()) / len(layer_stats['sublayers'])
        layer_stats['avg_beta'] = avg_beta
        layer_stats['avg_delta_ratio'] = avg_delta_ratio
        compression_stats.append(layer_stats)

        if verbose:
            print(
                f"Layer {layer_idx} Summary: avg_β={avg_beta:.4f}, "
                f"avg_α*={avg_alpha_star:.4f}, avg_ρₑ={avg_best_score:.6f}, "
                f"avg_Δ/H={avg_delta_ratio:.4f}"
            )

        def run_student_postcompress(post_label):
            nan_count_local = 0
            post_cast_stats = _init_cast_stats(post_label, activation_cache_dtype)
            output_err_sq = 0.0
            output_teacher_sq = 0.0
            output_student_sq = 0.0
            output_numel = 0
            for j in range(num_samples):
                inp_j = inps_student[j].unsqueeze(0).to(dev).float()
                attn_mask_j = attention_masks[j].to(dev) if attention_masks[j] is not None else None
                pos_ids_j = position_ids[j].to(dev) if position_ids is not None else None

                fw_kwargs = {}
                if attn_mask_j is not None:
                    fw_kwargs['attention_mask'] = attn_mask_j
                if not is_opt and pos_ids_j is not None:
                    fw_kwargs['position_ids'] = pos_ids_j
                if pos_embed_cached is not None:
                    fw_kwargs['position_embeddings'] = tuple(t.to(dev) for t in pos_embed_cached)

                out_j = layer(inp_j, **_filter_fw_kwargs(layer, fw_kwargs))[0]
                teacher_ref = outs_teacher[j].unsqueeze(0).to(dev).float()
                out_float = out_j.float()
                diff = out_float - teacher_ref
                output_err_sq += torch.sum(diff * diff).item()
                output_teacher_sq += torch.sum(teacher_ref * teacher_ref).item()
                output_student_sq += torch.sum(out_float * out_float).item()
                output_numel += out_float.numel()
                cached_student, pre_finite, post_finite = _cast_activation_for_cache(
                    out_j, activation_cache_dtype, post_cast_stats
                )
                if pre_finite and post_finite:
                    outs_student[j] = cached_student
                else:
                    nan_count_local += 1
                del inp_j, out_j, teacher_ref, out_float, diff, cached_student
            _print_cast_stats(post_cast_stats, layer_idx)
            rel_l2 = math.sqrt(output_err_sq / (output_teacher_sq + 1e-12))
            teacher_rms = math.sqrt(output_teacher_sq / max(output_numel, 1))
            student_rms = math.sqrt(output_student_sq / max(output_numel, 1))
            rms_ratio = student_rms / (teacher_rms + 1e-12)
            post_cast_stats['output_rel_l2'] = rel_l2
            post_cast_stats['output_teacher_rms'] = teacher_rms
            post_cast_stats['output_student_rms'] = student_rms
            post_cast_stats['output_rms_ratio'] = rms_ratio
            print(
                f"  L{layer_idx} {post_label} teacher对齐诊断: "
                f"rel_l2={rel_l2:.4f}, student_rms={student_rms:.4e}, "
                f"teacher_rms={teacher_rms:.4e}, rms_ratio={rms_ratio:.4f}"
            )
            return post_cast_stats, nan_count_local

        def force_layer_tsvd_fallback(reason):
            forced = 0
            print(f"  警告: L{layer_idx} 触发层级激活保护，强制本层 Linear 回退 TSVD [{reason}]")
            for sub_name, sub_stats in layer_stats['sublayers'].items():
                original_linear = subset[sub_name]
                W_force = original_linear.weight.data.float().to(dev)
                rank_force = sub_stats['rank']
                B_force, A_force, tsvd_err_force = truncated_svd_factorize(W_force, rank_force)
                bias_force = original_linear.bias.data if original_linear.bias is not None else None
                replace_linear_with_factorized(
                    layer, sub_name, B_force, A_force, bias_force,
                    device=dev, dtype=original_linear.weight.data.dtype
                )
                sub_stats['tsvd_err'] = tsvd_err_force
                sub_stats['weight_err'] = tsvd_err_force
                sub_stats['layer_guard_forced_tsvd'] = True
                if sub_stats.get('fallback_reason') is None:
                    sub_stats['fallback_reason'] = reason
                forced += 1
                del W_force, B_force, A_force
                torch.cuda.empty_cache()
            return forced

        student_post_cast_stats, nan_count = run_student_postcompress('student_postcompress')

        layer_stats['student_post_max_abs'] = student_post_cast_stats['pre_max_abs']
        layer_stats['teacher_pre_max_abs'] = teacher_pre_cast_stats['pre_max_abs']
        if qwen_safety_guard:
            teacher_abs = max(teacher_pre_cast_stats['pre_max_abs'], 1e-6)
            activation_threshold = max(activation_guard_abs, activation_guard_ratio * teacher_abs)
            activation_guard_failed = (
                student_post_cast_stats['pre_max_abs'] > activation_threshold
                or student_post_cast_stats['output_rel_l2'] > output_rel_l2_guard
                or student_post_cast_stats['output_rms_ratio'] < output_rms_ratio_min
                or student_post_cast_stats['output_rms_ratio'] > output_rms_ratio_max
            )
            layer_stats['activation_guard_threshold'] = activation_threshold
            layer_stats['activation_guard_failed'] = activation_guard_failed
            layer_stats['output_rel_l2'] = student_post_cast_stats['output_rel_l2']
            layer_stats['output_rms_ratio'] = student_post_cast_stats['output_rms_ratio']

            if activation_guard_failed:
                forced_count = force_layer_tsvd_fallback("qwen_layer_activation_guard")
                layer_stats['layer_guard_forced_count'] = forced_count
                print(
                    f"    L{layer_idx} teacher对齐保护触发: student_max_abs="
                    f"{student_post_cast_stats['pre_max_abs']:.4e}, teacher_max_abs="
                    f"{teacher_abs:.4e}, threshold={activation_threshold:.4e}; "
                    f"rel_l2={student_post_cast_stats['output_rel_l2']:.4f}, "
                    f"rms_ratio={student_post_cast_stats['output_rms_ratio']:.4f}; "
                    f"已强制回退 {forced_count} 个 Linear，重新前向"
                )
                student_post_cast_stats, nan_count = run_student_postcompress(
                    'student_postcompress_after_layer_tsvd'
                )
                layer_stats['student_post_max_abs_after_guard'] = student_post_cast_stats['pre_max_abs']
                layer_stats['output_rel_l2_after_guard'] = student_post_cast_stats['output_rel_l2']
                layer_stats['output_rms_ratio_after_guard'] = student_post_cast_stats['output_rms_ratio']
                still_failed = (
                    student_post_cast_stats['pre_max_abs'] > activation_threshold
                    or student_post_cast_stats['output_rel_l2'] > output_rel_l2_guard
                    or student_post_cast_stats['output_rms_ratio'] < output_rms_ratio_min
                    or student_post_cast_stats['output_rms_ratio'] > output_rms_ratio_max
                )
                layer_stats['activation_guard_still_failed'] = still_failed
                if still_failed:
                    print(
                        f"  警告: L{layer_idx} 强制 TSVD 后仍未通过 teacher 对齐保护 "
                        f"(student_max_abs={student_post_cast_stats['pre_max_abs']:.4e}, "
                        f"threshold={activation_threshold:.4e}, "
                        f"rel_l2={student_post_cast_stats['output_rel_l2']:.4f}, "
                        f"rms_ratio={student_post_cast_stats['output_rms_ratio']:.4f})"
                    )

        if nan_count > 0:
            print(f"  警告: L{layer_idx} 闭环前向有 {nan_count}/{num_samples} 个样本含 NaN")

        inps_student = outs_student.clone()
        inps_teacher = outs_teacher.clone()
        layers[layer_idx] = layer.to(dtype=activation_cache_dtype).cpu()
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SAES-SVD v3 压缩完成！")
    print("=" * 60)
    if compression_stats:
        all_betas = [s['avg_beta'] for s in compression_stats]
        all_delta_ratios = [s['avg_delta_ratio'] for s in compression_stats]
        print(f"  平均 beta: {sum(all_betas)/len(all_betas):.4f}")
        print(f"  平均 Δ/H:  {sum(all_delta_ratios)/len(all_delta_ratios):.4f}")
        print(f"  Beta 范围: [{min(all_betas):.4f}, {max(all_betas):.4f}]")

    return compression_stats


# ==================== 冒烟测试 ====================

def smoke_test():
    """单元测试：验证 SAES-SVD v3 单阶段核心数学正确性。"""
    print("=" * 50)
    print("SAES-SVD v3 Smoke Test")
    print("=" * 50)

    torch.manual_seed(42)
    m, n, p = 128, 64, 256
    rank = 16
    eps = 1e-4

    W = torch.randn(m, n)
    X = torch.randn(n, p)
    X_f = X + 0.1 * torch.randn(n, p)

    H = X @ X.T
    Delta = (X_f - X) @ X.T
    _Q, _Lambda, _H_sqrt, L = compute_H_decomposition(H, "cpu", eps)
    del _Q, _Lambda, _H_sqrt

    S = W @ H @ L
    D = W @ Delta @ L
    beta, info = aces_beta_select(
        S, D, rank,
        alpha_min=0.4, alpha_max=0.6,
        rho=0.95, beta_cap=0.6 / 1.6
    )
    print(f"  SAES beta*={beta:.4f}, alpha*={info['alpha_star']:.4f}")

    G = S + beta * D
    U, Sigma, Vt = torch.linalg.svd(G, full_matrices=False)
    sqrt_sigma = torch.sqrt(torch.clamp(Sigma[:rank], min=0))
    B = U[:, :rank] * sqrt_sigma.unsqueeze(0)
    A = (sqrt_sigma.unsqueeze(1) * Vt[:rank, :]) @ L

    assert B.shape == (m, rank), f"B 维度错误: {B.shape}"
    assert A.shape == (rank, n), f"A 维度错误: {A.shape}"
    assert torch.isfinite(B).all() and torch.isfinite(A).all(), "低秩因子包含非有限值"

    G_recon = U[:, :rank] @ torch.diag(Sigma[:rank]) @ Vt[:rank, :]
    g_err = torch.norm(G - G_recon) / torch.norm(G)
    w_err = torch.norm(W - B @ A) / torch.norm(W)
    print(f"  G_err={g_err.item():.4f}, W_err={w_err.item():.4f}")
    print("[PASS] SAES-SVD v3 单阶段维度与数值检查通过")
    print("=" * 50)


# ==================== 命令行入口 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SAES-SVD v3：单阶段闭环压缩 + Qwen 多模型适配'
    )

    # 基础参数
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--model_tag', type=str, default=None,
                        help='模型预设：qwen2_5_7b, qwen3_5_35b_a3b, qwen3_moe')
    parser.add_argument('--ratio', type=float, default=0.2, help='压缩率（0.2 = 去掉 20%%）')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='校准数据集')
    parser.add_argument('--nsamples', type=int, default=128, help='校准样本数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--DEV', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--save_path', type=str, default='./result/SAES_SVD_v3', help='保存路径')
    parser.add_argument('--result_dir', type=str, default=None, help='结果目录')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='序列长度')

    # SAES-SVD v3 参数
    parser.add_argument('--eps', type=float, default=1e-4, help='H 阻尼系数')
    parser.add_argument('--rho', type=float, default=0.95, help='ACES β 收缩因子（v2.1.4 默认 0.95）')
    parser.add_argument('--alpha_min', type=float, default=0.4, help='ACES α 下界（v2.1.4 默认 0.4）')
    parser.add_argument('--alpha_max', type=float, default=0.6, help='ACES α 上界（v2.1.4 默认 0.6）')
    parser.add_argument('--min_dim', type=int, default=64, help='跳过 min(in,out) 小于该值的 Linear')
    parser.add_argument('--skip_moe_experts', action='store_true', default=False,
                        help='MoE 模型跳过 experts.<i>.*，只压缩 attention/shared 层')

    # 功能模式
    parser.add_argument('--eval_only', action='store_true', help='仅评估')
    parser.add_argument('--model_path', type=str, default=None, help='压缩模型路径')
    parser.add_argument('--smoke_test', action='store_true', help='冒烟测试')
    parser.add_argument('--lora', type=str, default=None,
                        help='LoRA adapter 目录；eval_only 模式下自动 merge_and_unload 并保存 merge.pt')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='PPL 评估 batch size')
    parser.add_argument('--eval_dtype', type=str, default='auto',
                        choices=['auto', 'fp16', 'bf16', 'fp32'],
                        help='评估时模型精度；auto 时 Qwen 使用 bf16，其他模型使用 fp16')
    parser.add_argument('--eval_large', action='store_true', default=False,
                        help='超大模型评估模式：模型常驻 CPU，逐层搬上 GPU（适用于 30B+ 等超过单卡显存的模型）')
    parser.add_argument('--no_eval_after_compress', action='store_true', default=False,
                        help='压缩并保存后立即退出，不运行压缩后的 PPL 评估')

    args = parser.parse_args()

    model_presets = {
        'qwen2_5_7b': '/data/zhangs/LLMs/Models/Qwen/Qwen2.5-7B',
        'qwen3_5_35b_a3b': '/data/zhangs/LLMs/Models/Qwen/Qwen3.5-35B-A3B',
        'qwen3_moe': '/data/zhangs/LLMs/Models/Qwen/Qwen3.5-35B-A3B',
    }
    if args.model_tag:
        if args.model_tag not in model_presets:
            parser.error(f"未知 model_tag: {args.model_tag}，支持: {', '.join(model_presets)}")
        args.model = model_presets[args.model_tag]
        if args.model_tag in ('qwen3_5_35b_a3b', 'qwen3_moe'):
            args.skip_moe_experts = True

    if args.result_dir:
        args.save_path = args.result_dir

    if args.smoke_test:
        smoke_test()
        sys.exit(0)

    if args.model is None and not args.eval_only:
        parser.error("非 --eval_only 模式下必须指定 --model 或 --model_tag")

    keep_ratio = 1 - args.ratio

    print("=" * 60)
    print("SAES-SVD v3 配置")
    print("=" * 60)
    print(f"  模型: {args.model}")
    print(f"  压缩率: {args.ratio} (保留 {keep_ratio*100:.1f}%)")
    print(f"  数据集: {args.dataset}, 样本数: {args.nsamples}")
    print(f"  设备: {args.DEV}")
    print(f"  eps={args.eps}")
    print(f"  Alpha: [{args.alpha_min}, {args.alpha_max}], rho={args.rho}")
    print(f"  min_dim: {args.min_dim}, skip_moe_experts: {args.skip_moe_experts}")
    print("=" * 60)

    if args.eval_only and args.model_path:
        print("\n仅评估模式...")
        model, tokenizer = get_model_from_local(args.model_path)
        eval_dtype = _resolve_eval_dtype(model, args.eval_dtype)
        print(f"  评估 dtype: {_dtype_name(eval_dtype)} (requested={args.eval_dtype})")

        # v1.4：LoRA adapter 合并（对标 SVDLLM step 4 分支）
        if args.lora is not None:
            print(f"\n加载 LoRA adapter 并合并权重: {args.lora}")
            from utils.peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                args.lora,
                torch_dtype=eval_dtype,
            )
            model = model.merge_and_unload()
            if not os.path.exists(args.lora):
                os.makedirs(args.lora)
            merge_path = os.path.join(args.lora, 'merge.pt')
            print(f"保存合并后模型到: {merge_path}")
            torch.save({'model': model, 'tokenizer': tokenizer}, merge_path)

        model.eval()
        print("\n运行困惑度评估...")
        if args.eval_large:
            # 逐层评估：不整体搬 GPU，适用于超过单卡显存的大模型
            ppl_eval_large(model, tokenizer, datasets=['wikitext2'],
                           seq_len=args.model_seq_len,
                           batch_size=args.eval_batch_size, device=args.DEV)
        else:
            model = model.to(dtype=eval_dtype, device=args.DEV)
            ppl_eval(model, tokenizer, datasets=['wikitext2'],
                     model_seq_len=args.model_seq_len,
                     batch_size=args.eval_batch_size, device=args.DEV)
    else:
        print("\n加载模型...")
        model, tokenizer = load_model_v15(args.model, seq_len=args.model_seq_len)
        compression_dtype = _default_runtime_dtype_for_arch(_detect_arch_family(model))
        print(f"  压缩运行 dtype: {_dtype_name(compression_dtype)}")
        model = model.eval().to(dtype=compression_dtype)

        print("\n加载校准数据...")
        calib_data = get_calib_train_data_v3_safe(
            args.dataset, tokenizer, args.nsamples, seqlen=args.model_seq_len,
            model_id=args.model, model_vocab_size=model.config.vocab_size
        )

        print("\n开始 SAES-SVD v3 压缩...")
        stats = saes_svd_v3_compress_layerwise(
            args.model, model, calib_data, keep_ratio, args.DEV,
            rho=args.rho, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
            skip_moe_experts=args.skip_moe_experts,
            min_dim=args.min_dim,
            eps=args.eps,
            verbose=True
        )

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        save_file = os.path.join(
            args.save_path,
            f"{args.model.replace('/', '_').replace('-', '_')}_saes_svd_v3_{keep_ratio}.pt"
        )
        print(f"\n保存压缩模型到: {save_file}")
        save_dtype = _default_runtime_dtype_for_arch(_detect_arch_family(model))
        print(f"  保存模型 dtype: {_dtype_name(save_dtype)}")
        model = model.to(dtype=save_dtype)
        torch.save({'model': model, 'tokenizer': tokenizer, 'stats': stats}, save_file)

        del stats, calib_data
        import gc; gc.collect()
        torch.cuda.empty_cache()

        if args.no_eval_after_compress:
            print("\n跳过压缩后的困惑度评估（--no_eval_after_compress）")
            sys.exit(0)

        print("\n运行困惑度评估...")
        if args.eval_large:
            ppl_eval_large(model, tokenizer, datasets=['wikitext2'],
                           seq_len=args.model_seq_len,
                           batch_size=args.eval_batch_size, device=args.DEV)
        else:
            eval_dtype = _resolve_eval_dtype(model, args.eval_dtype)
            print(f"  压缩后评估 dtype: {_dtype_name(eval_dtype)} (requested={args.eval_dtype})")
            model = model.to(dtype=eval_dtype, device=args.DEV)
            ppl_eval(model, tokenizer, datasets=['wikitext2'],
                     model_seq_len=args.model_seq_len,
                     batch_size=args.eval_batch_size, device=args.DEV)
