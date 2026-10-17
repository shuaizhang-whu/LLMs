#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

# 阻断 transformers 联网，强制只用本地文件
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from PIL import Image


# 这个脚本用于测试原版 LLaVA v1.5 7B 的图片描述能力。
# 注意：/data/zhangs/LLMs/Models/llava-v1.5-7b 是原版 LLaVA 权重格式，
# 不是 transformers 新版 llava-hf 转换格式，所以需要官方 LLaVA Python 包。
DEFAULT_MODEL_PATH = "/data/zhangs/LLMs/Models/llava-v1.5-7b"
DEFAULT_PROMPT = "Describe this image in detail."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用本地 LLaVA v1.5 7B 模型做图片描述。")
    parser.add_argument(
        "--image",
        required=True,
        help="输入图片路径，例如 /data/zhangs/LLMs/test.jpg。",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"本地 LLaVA 模型目录，默认：{DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"给图片的文字指令，默认：{DEFAULT_PROMPT}",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备：auto 自动选空闲显存最多的 GPU，或指定 cuda:0 / cuda:1 等，默认 auto。",
    )
    parser.add_argument(
        "--llava-repo",
        default=None,
        help="如果没有 pip install llava，可以指定官方 LLaVA 仓库路径，例如 /data/zhangs/LLMs/Projects/LLaVA。",
    )
    parser.add_argument(
        "--conv-mode",
        default="llava_v1",
        help="LLaVA 对话模板，v1.5 通常使用 llava_v1。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="最多生成多少个新 token。")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度，0 表示贪心解码。")
    parser.add_argument("--top-p", type=float, default=None, help="top-p 采样参数。")
    parser.add_argument("--num-beams", type=int, default=1, help="beam search 数量，默认 1。")
    return parser.parse_args()


def _pick_best_gpu_via_nvidia_smi(min_free_gib: float = 25.0) -> str:
    """通过 nvidia-smi 选择空闲显存最多的 GPU，不创建 CUDA Context。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free_mib = [int(x.strip()) for x in result.stdout.strip().splitlines()]
        best = max(range(len(free_mib)), key=lambda i: free_mib[i])
        free_gib = free_mib[best] / 1024.0
        if free_gib < min_free_gib:
            print(f"[Warning] 所有 GPU 空闲显存均不足 {min_free_gib:.0f} GiB，"
                  f"最多只有 cuda:{best}（{free_gib:.1f} GiB），可能 OOM。")
        else:
            print(f"[Auto device] 选择 cuda:{best}（空闲显存 {free_gib:.1f} GiB）")
        return f"cuda:{best}"
    except Exception as exc:
        print(f"[Warning] nvidia-smi 查询失败（{exc}），回退到 cuda:0")
        return "cuda:0"


def add_llava_repo_to_path(llava_repo: Optional[str]) -> None:
    """把本地 LLaVA 仓库加入 Python import 路径。

    如果你已经执行过 `pip install -e /path/to/LLaVA`，这个参数可以不传。
    如果只是 clone 了仓库但没安装，可以传 `--llava-repo /path/to/LLaVA`。
    """
    if llava_repo is None:
        return

    repo_path = Path(llava_repo).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"LLaVA repo path does not exist: {repo_path}")
    sys.path.insert(0, str(repo_path))


def import_llava_modules():
    """导入官方 LLaVA 推理需要的模块，并在缺依赖时给出明确提示。"""
    try:
        from llava.constants import (  # type: ignore
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates  # type: ignore
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # type: ignore
        from llava.model.builder import load_pretrained_model  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name != "llava":
            raise
        raise ModuleNotFoundError(
            "当前 Python 环境没有找到官方 LLaVA 包。请先安装或指定本地仓库：\n"
            "  方式 1：cd /data/zhangs/LLMs/Projects/LLaVA && pip install -e .\n"
            "  方式 2：运行脚本时加 --llava-repo /data/zhangs/LLMs/Projects/LLaVA\n"
            "如果还没有仓库，需要先 clone 官方 LLaVA 项目。"
        ) from exc

    return {
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
        "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "conv_templates": conv_templates,
        "get_model_name_from_path": get_model_name_from_path,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "load_pretrained_model": load_pretrained_model,
    }


def validate_inputs(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser()
    image_path = Path(args.image).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"模型目录缺少 config.json：{model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    if args.device != "auto" and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("你指定了 CUDA，但当前 torch.cuda.is_available() 为 False。")


def build_llava_prompt(model, args: argparse.Namespace, llava_modules: dict) -> str:
    """构造 LLaVA 的图文 prompt。

    LLaVA 使用特殊的 <image> token 表示图片位置。v1.5 通常不需要
    <im_start>/<im_end>，但这里按模型 config 自动判断。
    """
    image_token = llava_modules["DEFAULT_IMAGE_TOKEN"]
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = (
            llava_modules["DEFAULT_IM_START_TOKEN"]
            + image_token
            + llava_modules["DEFAULT_IM_END_TOKEN"]
        )

    question = image_token + "\n" + args.prompt
    conv_templates = llava_modules["conv_templates"]
    if args.conv_mode not in conv_templates:
        raise ValueError(f"未知 conv-mode：{args.conv_mode}，可用模板：{list(conv_templates.keys())}")

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_image_tensor(image_path: str, image_processor, model, device: str, llava_modules: dict):
    """读取图片并转换成 LLaVA 需要的 tensor。"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = llava_modules["process_images"]([image], image_processor, model.config)[0]
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return image_tensor.to(device=device, dtype=dtype)


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    try:
        validate_inputs(args)
        add_llava_repo_to_path(args.llava_repo)
        llava_modules = import_llava_modules()

        # 解析设备
        device = _pick_best_gpu_via_nvidia_smi() if args.device == "auto" else args.device

        model_path = str(Path(args.model_path).expanduser().resolve())
        model_name = llava_modules["get_model_name_from_path"](model_path)

        print(f"Loading model: {model_path}")
        # device != 'cuda' 时，builder 会用 device_map={'': device}，强制整个模型到单卡
        tokenizer, model, image_processor, _ = llava_modules["load_pretrained_model"](
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device=device,
        )
        model.eval()

        prompt = build_llava_prompt(model, args, llava_modules)
        image_tensor = load_image_tensor(args.image, image_processor, model, device, llava_modules)
        input_ids = llava_modules["tokenizer_image_token"](
            prompt,
            tokenizer,
            llava_modules["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        ).unsqueeze(0).to(device)

        generation_kwargs = {
            "do_sample": args.temperature > 0,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            generation_kwargs["top_p"] = args.top_p

        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=image_tensor, **generation_kwargs)

        # 解码全部 output（含 prompt），再按 ASSISTANT: 分割提取答案。
        # 不用 output_ids[0, input_ids.shape[1]:] 切片，因为 LLaVA 内部把
        # 1 个图片占位符展开成 576 个 patch 嵌入，会导致切片偏移，截掉生成文本开头。
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in full_text:
            answer = full_text.split("ASSISTANT:")[-1].strip()
        else:
            answer = full_text.strip()
        print("\n=== 图片描述 ===")
        print(answer)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if "clip-vit-large-patch14-336" in str(exc):
            print(
                "提示：LLaVA v1.5 还需要视觉塔 openai/clip-vit-large-patch14-336。"
                "如果本地没有缓存，需要先下载该视觉塔，或保证运行时可以从 Hugging Face 读取。",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
