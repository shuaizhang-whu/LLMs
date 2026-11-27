"""使用 ModelScope 下载模型到当前仓库的 Models 目录。"""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help="ModelScope 模型 ID（默认：%(default)s）",
    )
    args = parser.parse_args()

    model_dir = Path(__file__).resolve().parents[1] / "Models" / args.model.rsplit("/", 1)[-1]
    print(f"使用 ModelScope 下载 {args.model} 到 {model_dir}", flush=True)
    subprocess.run(
        ["modelscope", "download", "--model", args.model, "--local_dir", str(model_dir)],
        check=True,
    )
    print("下载完成！")


# lab 原默认模型：--model Qwen/Qwen3.5-35B-A3B
# 其他模型示例：--model Qwen/Qwen3.5-9B 或 --model google/gemma-4-31B-it
# Hugging Face 镜像备选：HF_ENDPOINT=https://hf-mirror.com hf download MODEL_ID --local-dir MODEL_DIR

if __name__ == "__main__":
    main()
