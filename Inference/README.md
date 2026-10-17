# LLaMA 7B 本地推理

本目录用于推理本地模型：

```text
/data/zhangs/LLMs/Models/llama-7b
```

## 最简单用法

在项目根目录运行：

```bash
cd /data/zhangs/LLMs
python Inference/infer_llama7b.py --prompt "The meaning of life is" --max-new-tokens 100
```

## 交互模式

模型只加载一次，然后可以连续输入 prompt：

```bash
cd /data/zhangs/LLMs
python Inference/infer_llama7b.py --interactive
```

退出方式：

```text
/exit
```

或者按 `Ctrl-D`。

## 指定 GPU

使用第 0 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python Inference/infer_llama7b.py --device cuda:0 --prompt "Explain transformers in simple terms."
```

如果你已经通过 `CUDA_VISIBLE_DEVICES=1` 只暴露了一张卡，那么脚本里仍然写 `cuda:0` 即可，因为程序看到的是“当前可见的第 0 张卡”。

如果直接运行 `--interactive` 报 CUDA device-side assert，优先这样运行，避免自动多卡切分：

```bash
CUDA_VISIBLE_DEVICES=0 python Inference/infer_llama7b.py --device cuda:0 --interactive
```

## 常用参数

- `--prompt`：单次输入文本。
- `--interactive`：交互模式。
- `--max-new-tokens`：最多生成多少个新 token。
- `--temperature`：采样温度，越高越随机；设为 `0` 表示贪心解码。
- `--top-p`：核采样参数，常用 `0.8` 到 `0.95`。
- `--device`：选择设备，常用 `auto`、`cuda:0`、`cpu`。
- `--device-map-auto`：使用 accelerate 自动多卡切分。7B 单卡能放下时不建议打开。
- `--dtype`：模型精度，默认 `auto`。GPU 上通常用模型配置里的 `float16`，CPU 上默认用 `float32`。
- `--no-stream`：关闭流式输出，等生成结束后一次性打印。
- `--use-bin-weights`：强制加载 `.bin` 权重。如果 `.safetensors` 权重推理异常，可以试这个参数。

## 说明

- 脚本使用 `local_files_only=True`，只读取本地模型，不会联网下载。
- 你的模型是 base LLaMA，不是 chat/instruct 模型，所以默认不要加 `--chat-template`。
- LLaMA 7B FP16 推理通常需要约 14 GB 显存，还要额外留出 KV cache 显存。
- 没有 CUDA 时脚本会退回 CPU，但 7B 模型在 CPU 上会非常慢。
