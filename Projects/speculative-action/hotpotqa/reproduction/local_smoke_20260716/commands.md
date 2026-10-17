# HotPotQA 本地 smoke test 命令

所有 HotPotQA 命令均从以下目录执行：

```bash
cd /mnt/data/zhangshuai/LLMs/Projects/speculative-action/hotpotqa
```

## vLLM 环境与模型

```bash
conda create -n agent-action-vllm --override-channels -c conda-forge python=3.12 -y
/mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/python -m pip install --upgrade uv
UV_HTTP_RETRIES=12 UV_HTTP_TIMEOUT=180 \
  /mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/uv pip install \
  'vllm==0.25.1' --torch-backend=auto \
  --python /mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/python

HF_HUB_DOWNLOAD_TIMEOUT=180 HF_HUB_ETAG_TIMEOUT=180 \
  /mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/hf download \
  Qwen/Qwen2.5-7B-Instruct \
  --local-dir /mnt/data/zhangshuai/LLMs/Models/Qwen2.5-7B-Instruct \
  --max-workers 2
```

## 启动本地服务

GPU1 在启动前为空闲状态。`VLLM_USE_FLASHINFER_SAMPLER=0` 避免 FlashInfer sampler
在该机器上因系统级 `/usr/local/cuda` 不存在而执行失败；注意力后端仍为
FlashAttention 2。

```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_HOME=/mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/lib/python3.12/site-packages/nvidia/cu13 \
PATH=/mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH \
/mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/vllm serve \
  /mnt/data/zhangshuai/LLMs/Models/Qwen2.5-7B-Instruct \
  --served-model-name local-qwen2.5-7b-instruct \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

## 健康检查与项目 smoke test

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models

NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
LOCAL_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
LOCAL_OPENAI_API_KEY=local \
MPLCONFIGDIR=/tmp/speculative-action-mpl \
/mnt/data/zhangshuai/anaconda3/envs/agent-action/bin/python run.py \
  --modelname local/local-qwen2.5-7b-instruct \
  --guessmodelname local/local-qwen2.5-7b-instruct \
  --num-samples 1 --num-steps 3
```

`local` 只是本机 OpenAI-compatible 服务的占位 API Key，不是秘密信息。
实际运行不能添加 `--norun`；该参数在官方 CLI 中表示跳过在线实验。
