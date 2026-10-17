#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# transformers 在当前环境会打印 torch pytree 的 FutureWarning，这不是推理错误。
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


# 默认模型路径。这里必须使用绝对路径，并且目录名要和磁盘上的目录完全一致：
# /data/zhangs/LLMs/Models/llama-7b
DEFAULT_MODEL_PATH = "/data/zhangs/LLMs/Models/llama-7b"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这部分看起来比你原来的脚本长，主要是为了让同一个脚本支持：
    1. 直接输入一个 prompt 后退出；
    2. 进入交互模式反复输入；
    3. 自动选择 GPU/CPU；
    4. 调整生成长度、温度、top-p 等常用推理参数。

    如果你只想最简单运行，只需要传 --prompt 即可。
    """
    parser = argparse.ArgumentParser(
        description="使用本地 Hugging Face 格式的 LLaMA 7B 模型做推理。"
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"本地模型目录，默认：{DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="单次推理输入。如果不传该参数，脚本会进入交互模式。",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启动交互模式，可以连续输入 prompt。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备：auto、cpu、cuda、cuda:0、cuda:1 或 mps，默认 auto。",
    )
    parser.add_argument(
        "--device-map-auto",
        action="store_true",
        help="使用 accelerate 的 device_map='auto' 自动切分模型。7B 单卡能放下时不建议打开。",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="模型精度。默认 auto：CPU 用 float32，GPU 用模型配置中的精度。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="最多生成多少个新 token。",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1800,
        help="输入 prompt 最多保留多少个 token。LLaMA 7B 上下文通常是 2048。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度。值越高越随机；设为 0 表示贪心解码。",
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="核采样 top-p。")
    parser.add_argument("--top-k", type=int, default=50, help="top-k 采样。")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="重复惩罚，值大于 1 时会降低重复输出概率。",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="使用 tokenizer 的聊天模板包装 prompt。base LLaMA 通常不要打开。",
    )
    parser.add_argument(
        "--use-fast-tokenizer",
        action="store_true",
        help="使用 fast tokenizer。默认使用 slow tokenizer，更贴近原始 LLaMA tokenizer。",
    )
    parser.add_argument(
        "--use-bin-weights",
        action="store_true",
        help="强制使用 .bin 权重，而不是优先使用 .safetensors。",
    )
    parser.add_argument(
        "--system-message",
        default=None,
        help="系统提示词，只在 --chat-template 打开时生效。",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="关闭流式输出，等全部生成完成后一次性打印。",
    )
    return parser.parse_args()


def validate_model_path(model_path: Path) -> None:
    """检查模型目录是否像一个完整的本地 Hugging Face 模型目录。

    你的模型目录里应该至少有：
    - config.json：模型结构配置；
    - tokenizer.model：LLaMA tokenizer 文件；
    - .safetensors 或 .bin：模型权重文件。

    提前检查的好处是路径写错时能给出清楚错误，而不是在 transformers
    内部报一大段不直观的异常。
    """
    required_files = ("config.json", "tokenizer.model")
    missing = [name for name in required_files if not (model_path / name).exists()]
    has_weights = any(model_path.glob("*.safetensors")) or any(model_path.glob("*.bin"))

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if missing:
        raise FileNotFoundError(
            f"Model path is missing required file(s): {', '.join(missing)}"
        )
    if not has_weights:
        raise FileNotFoundError(f"No .safetensors or .bin weight files found in {model_path}")


def resolve_dtype(dtype_name: str, device: str):
    """根据命令行参数和设备选择模型加载精度。

    - GPU 上默认使用模型 config.json 里的 torch_dtype。你的 llama-7b 配置是 float16；
    - CPU 上默认改成 float32，因为很多 CPU 对 float16 推理支持不好；
    - 如果你明确传 --dtype float16 / bfloat16 / float32，就按你指定的来。
    """
    if dtype_name == "auto":
        if device == "cpu":
            return torch.float32
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def _pick_best_gpu_via_nvidia_smi() -> tuple[int, float]:
    """通过 nvidia-smi 查询各 GPU 空闲显存，返回 (最优 GPU 索引, 空闲显存 GiB)。

    使用 nvidia-smi 而非 torch.cuda.mem_get_info，是因为后者在查询每块 GPU 时
    都会创建一个 CUDA Context（~260 MiB），导致进程在每张卡上都有占用。
    nvidia-smi 是纯系统调用，不会接触 CUDA 运行时。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        free_mib = [int(x.strip()) for x in result.stdout.strip().splitlines()]
        best = max(range(len(free_mib)), key=lambda i: free_mib[i])
        return best, free_mib[best] / 1024.0
    except Exception as exc:
        print(f"[Warning] nvidia-smi 查询失败（{exc}），回退到 cuda:0")
        return 0, 0.0


def load_model_and_tokenizer(args: argparse.Namespace):
    """加载 tokenizer 和模型权重。

    这里没有使用 pipeline，是因为 pipeline 对简单 demo 很方便，但对本地大模型
    不够透明。直接用 model.generate 可以更明确地控制：
    - 是否只读本地文件；
    - 模型放到哪块 GPU；
    - dtype 用什么；
    - 是否流式输出。
    """
    model_path = Path(args.model_path).expanduser().resolve()
    validate_model_path(model_path)

    # local_files_only=True 表示只读本地目录，不会尝试联网下载。
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=args.use_fast_tokenizer,
    )

    # 有些 LLaMA tokenizer 没有 pad token。generate 需要 pad_token_id 时，
    # 通常把 pad token 设成 eos token 即可，避免 padding 相关报错。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 自动选择设备：
    # - 有 CUDA 就优先用 GPU，并自动选择空闲显存最多的那块；
    # - macOS 可用 mps；
    # - 都没有就退回 CPU。CPU 能跑但会非常慢。
    #
    # 注意：不能用 torch.cuda.mem_get_info(i) 来遍历 GPU，因为每次调用都会在
    # 对应 GPU 上创建一个 CUDA Context（约占 260 MiB），导致每张卡上都出现进程。
    # 改用 nvidia-smi 子进程查询，完全不会触碰 CUDA。
    device = args.device.lower()
    if device == "auto":
        if torch.cuda.is_available():
            best_gpu, free_gb = _pick_best_gpu_via_nvidia_smi()
            print(f"[Auto device] 选择 cuda:{best_gpu}（空闲显存 {free_gb:.1f} GiB）")
            device = f"cuda:{best_gpu}"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # from_pretrained 的加载参数。
    # low_cpu_mem_usage=True 可以减少加载大模型时的 CPU 内存峰值。
    load_kwargs = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": resolve_dtype(args.dtype, device),
    }
    if args.use_bin_weights:
        load_kwargs["use_safetensors"] = False

    # 默认不使用 device_map="auto"。对 7B 模型和 A40 这类显卡，单卡直接推理更简单稳定。
    # 只有显存不够、确实需要多卡切分时，才通过 --device-map-auto 打开。
    if args.device_map_auto:
        load_kwargs["device_map"] = "auto"
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    elif device not in {"cpu", "mps"}:
        raise ValueError(f"Unsupported device: {args.device}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    # 没用 device_map 时，手动把模型移动到目标设备。
    if "device_map" not in load_kwargs:
        model.to(device)

    # 明确同步 tokenizer 和 generation_config 的特殊 token。
    # 这里把 pad 设成 eos，可以避免 LLaMA 没有 pad token 时 generate 报警或走错默认值。
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # 推理模式：关闭 dropout 等训练行为。
    model.eval()
    check_tokenizer_and_model_vocab(tokenizer, model)
    return model, tokenizer


def check_tokenizer_and_model_vocab(tokenizer, model) -> None:
    """检查 tokenizer 和模型 embedding/lm_head 的词表大小是否一致。

    CUDA 的 device-side assert 很多时候不是 CUDA 本身坏了，而是某个 token id
    超过了模型 logits/embedding 的范围。这里提前检查，能把问题定位清楚。
    """
    input_vocab_size = model.get_input_embeddings().num_embeddings
    output_vocab_size = getattr(model.get_output_embeddings(), "out_features", input_vocab_size)
    tokenizer_size = len(tokenizer)

    if tokenizer_size > input_vocab_size or tokenizer_size > output_vocab_size:
        raise ValueError(
            "Tokenizer 词表大于模型词表，可能是 tokenizer 和权重不匹配："
            f"len(tokenizer)={tokenizer_size}, "
            f"input_embeddings={input_vocab_size}, lm_head={output_vocab_size}。"
        )


def build_prompt(tokenizer, text: str, args: argparse.Namespace) -> str:
    """构造真正送进模型的 prompt。

    当前模型看起来是 base LLaMA，不是专门的 chat/instruct 模型，所以默认直接
    使用原始 prompt。只有你传了 --chat-template 时，才会使用 tokenizer 里的
    chat_template 把输入包装成类似 [INST] ... [/INST] 的格式。
    """
    if not args.chat_template:
        return text

    messages = []
    if args.system_message:
        messages.append({"role": "system", "content": args.system_message})
    messages.append({"role": "user", "content": text})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_generation_kwargs(args: argparse.Namespace) -> dict:
    """整理 model.generate 需要的生成参数。

    temperature=0 时使用贪心解码，不采样；
    temperature>0 时打开采样，并使用 top-p / top-k 控制随机性。
    """
    do_sample = args.temperature > 0
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": None,
        "eos_token_id": None,
    }
    if args.repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = args.repetition_penalty
    if do_sample:
        kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    return kwargs


def generate_once(model, tokenizer, prompt: str, args: argparse.Namespace) -> str:
    """对单条 prompt 做一次生成。"""
    prompt_text = build_prompt(tokenizer, prompt, args)

    # 把文本变成 token id。truncation=True 表示如果输入太长，就从左侧截断到
    # max_input_tokens，避免超过 LLaMA 7B 的上下文长度。
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )

    validate_input_token_ids(inputs["input_ids"], model)

    # 输入 tensor 必须和模型在同一个设备上，否则 generate 会报 device mismatch。
    input_device = next(model.parameters()).device
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

    generation_kwargs = build_generation_kwargs(args)
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    # LLaMA 7B 的上下文窗口通常是 2048。这里做一道保护：
    # 如果 prompt token 数 + 要生成的新 token 数超过上下文窗口，就自动缩短生成长度。
    context_window = getattr(model.config, "max_position_embeddings", None)
    input_tokens = inputs["input_ids"].shape[-1]
    if context_window and input_tokens + generation_kwargs["max_new_tokens"] > context_window:
        allowed_new_tokens = max(context_window - input_tokens, 1)
        print(
            "Warning: prompt plus max_new_tokens exceeds model context; "
            f"using max_new_tokens={allowed_new_tokens}.",
            file=sys.stderr,
        )
        generation_kwargs["max_new_tokens"] = allowed_new_tokens

    # torch.inference_mode() 会关闭梯度计算，减少显存/内存占用。
    with torch.inference_mode():
        if args.no_stream:
            # 非流式输出：等全部 token 生成完，再 decode 成字符串。
            output_ids = model.generate(**inputs, **generation_kwargs)

            # output_ids 里包含“输入 prompt + 新生成内容”，这里切掉 prompt，
            # 只返回模型新生成的部分。
            new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 流式输出：模型边生成边打印，长文本时能更快看到结果。
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        model.generate(**inputs, streamer=streamer, **generation_kwargs)
        return ""


def validate_input_token_ids(input_ids: torch.Tensor, model) -> None:
    """推理前检查 input_ids 是否超过模型词表范围。"""
    vocab_size = model.get_input_embeddings().num_embeddings
    max_token_id = int(input_ids.max().item())
    min_token_id = int(input_ids.min().item())

    if min_token_id < 0 or max_token_id >= vocab_size:
        raise ValueError(
            "输入 token id 超出模型词表范围，说明 tokenizer 和模型权重不匹配："
            f"min_token_id={min_token_id}, max_token_id={max_token_id}, "
            f"model_vocab_size={vocab_size}。"
        )


def run_interactive(model, tokenizer, args: argparse.Namespace) -> None:
    """交互模式：加载一次模型，然后可以连续输入多条 prompt。"""
    print("交互模式。输入 /exit 或按 Ctrl-D 退出。")
    while True:
        try:
            prompt = input("\n>>> ").strip()
        except EOFError:
            print()
            break

        if prompt in {"/exit", "/quit"}:
            break
        if not prompt:
            continue

        answer = generate_once(model, tokenizer, prompt, args)
        if answer:
            print(answer)
        elif not args.no_stream:
            print()


def main() -> int:
    """程序入口。"""
    args = parse_args()

    try:
        model, tokenizer = load_model_and_tokenizer(args)

        # 如果传了 --prompt，就只生成一次后退出。
        if args.prompt is not None:
            answer = generate_once(model, tokenizer, args.prompt, args)
            if answer:
                print(answer)
            elif not args.no_stream:
                print()
        else:
            # 没传 --prompt 时，默认进入交互模式。
            args.interactive = True
            run_interactive(model, tokenizer, args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if "device-side assert" in str(exc):
            print(
                "提示：CUDA device-side assert 通常需要退出当前进程后重新运行。"
                "建议先用单卡命令：CUDA_VISIBLE_DEVICES=0 python "
                "Inference/infer_llama7b.py --device cuda:0 --interactive。"
                "如果仍报错，再加 CUDA_LAUNCH_BLOCKING=1 定位，或尝试 --use-bin-weights。",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
