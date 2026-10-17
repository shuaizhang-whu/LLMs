import subprocess
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GenerationConfig

# FR-SVD 项目根目录需要在 sys.path 中，因为 .pt 文件保存时引用了
# 项目内的自定义模块 `component`（component/svd_llama.py 等）
FR_SVD_ROOT = "/data/zhangs/LLMs/Projects/FR-SVD"
if FR_SVD_ROOT not in sys.path:
    sys.path.insert(0, FR_SVD_ROOT)

def _patch_model_for_new_transformers(model):
    """修复旧版 transformers 序列化的模型与新版之间的属性不兼容问题。

    torch.save/load 只序列化实例的 __dict__，不重跑 __init__，
    导致新版 transformers 在 __init__ 中新增的属性全部缺失。
    在这里统一补全，避免推理时逐个 AttributeError。
    """
    # 确保 config 有 _attn_implementation（新版 PretrainedConfig 的 property）
    if hasattr(model, "config"):
        cfg = model.config
        if not hasattr(cfg, "_attn_implementation_internal"):
            cfg._attn_implementation_internal = "eager"
    for module in model.modules():
        # LlamaModel / LlamaDecoderLayer 等需要的新版属性
        if not hasattr(module, "_use_flash_attention_2"):
            module._use_flash_attention_2 = False
        if not hasattr(module, "_use_sdpa"):
            module._use_sdpa = False
        if not hasattr(module, "_attn_implementation"):
            module._attn_implementation = "eager"
        # SVDLinear 旧版保存时无 activation 属性，新版 forward 里加了该调用
        if type(module).__name__ == "SVDLinear" and not hasattr(module, "activation"):
            module.activation = nn.Identity()
        # SiLUActivation 旧版无 inplace 属性，新版 torch F.silu 需要它
        if type(module).__name__ == "SiLUActivation" and not hasattr(module, "inplace"):
            module.inplace = False


def _pick_best_gpu_via_nvidia_smi(min_free_gib: float = 25.0):
    """通过 nvidia-smi 选择空闲显存最多的 GPU，不创建 CUDA Context。
    
    min_free_gib: 要求 GPU 空闲显存至少这么多 GiB，不足则报错提示。
    LLaMA 7B 压缩模型约占 20 GiB，推理 KV cache 等需要额外缓冲。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free_mib = [int(x.strip()) for x in result.stdout.strip().splitlines()]
        best = max(range(len(free_mib)), key=lambda i: free_mib[i])
        best_free_gib = free_mib[best] / 1024.0
        if best_free_gib < min_free_gib:
            print(f"[Warning] 所有 GPU 空闲显存均不足 {min_free_gib:.0f} GiB，"
                  f"最多只有 cuda:{best}（{best_free_gib:.1f} GiB）。"
                  f"推理时可能 OOM，请等待其他进程释放显存后重试。")
        return best, best_free_gib
    except Exception as exc:
        print(f"[Warning] nvidia-smi 查询失败（{exc}），回退到 cuda:0")
        return 0, 0.0


def generate_text(model, tokenizer, prompt, device, max_new_tokens=200,
                  do_sample=True, top_p=0.95, temperature=0.7):
    """用 model.generate 推理，手动控制设备，返回新生成的文本（不含 prompt）。"""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,  # 新版 transformers 的 DynamicCache 与旧版 svd_llama.py 不兼容
        )

    # 只返回新生成的部分
    new_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main():
    # FR-SVD 压缩后的模型文件（torch.save 保存的 dict，含 model/tokenizer/stats）
    compressed_model_path = "/data/zhangs/LLMs/Projects/FR-SVD/result/phase2/ns256_rho0.42_a045_055_r1_0.95_lam1e5/_data_zhangs_LLMs_Models_jeffwan_llama_7b_hf_fr_svd_0.8.pt"
    # Tokenizer 备用路径（若 .pt 文件中无 tokenizer）
    base_model_path = "/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf"

    # 选择空闲显存最多的 GPU
    best_gpu, free_gb = _pick_best_gpu_via_nvidia_smi()
    device = f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[Auto device] 选择 {device}（空闲显存 {free_gb:.1f} GiB）")

    # 用 torch.load 加载整个压缩模型对象
    print("正在加载压缩模型...")
    ckpt = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model = ckpt["model"]
    model.to(device)
    model.eval()
    # 用新版 GenerationConfig 替换模型中可能保存的旧版对象，避免 transformers 版本属性不兼容
    model.generation_config = GenerationConfig()
    # 旧版 transformers 序列化的模型不会重跑 __init__，新版里新增的属性全部缺失。
    # 一次性补全所有已知属性，避免逐个报错。
    _patch_model_for_new_transformers(model)

    # tokenizer 从 .pt 文件中取出，否则从基础模型目录加载
    tokenizer = ckpt.get("tokenizer") or AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("模型加载完成。")

    # 简单测试
    test_out = generate_text(model, tokenizer, "The meaning of life is", device,
                             max_new_tokens=50, do_sample=False, top_p=1.0, temperature=1.0)
    print("测试输出:", test_out)

    # 交互式对话循环（使用 Alpaca 指令模板，引导基础模型按问答格式续写）
    def build_prompt(instruction):
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

    print("\n=== 交互模式（输入 exit 退出）===")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue
        prompt = build_prompt(user_input)
        answer = generate_text(model, tokenizer, prompt, device,
                               max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.7)
        # 截取 ### Response: 之后、下一个 ### 之前的内容
        answer = answer.split("###")[0].strip()
        print("Model:", answer)

if __name__ == "__main__":
    main()

   