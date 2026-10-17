# HotPotQA 本地 20 样本复现命令

工作目录：

```bash
cd /mnt/data/zhangshuai/LLMs/Projects/speculative-action/hotpotqa
```

公共环境变量：

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export LOCAL_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_OPENAI_API_KEY=local
export MPLCONFIGDIR=/tmp/speculative-action-mpl
export SAMPLES=7107,5619,373,6904,1267,4202,2316,6,1248,7356,1477,1354,7090,3021,2038,4349,2885,2861,4948,2527
```

vLLM 服务：

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

三组固定 20 样本实验：

```bash
python run.py \
  --execution-mode sequential \
  --modelname local/local-qwen2.5-7b-instruct \
  --guessmodelname local/local-qwen2.5-7b-instruct \
  --sample-ids "$SAMPLES" --num-steps 8 --top-k 1 \
  --seed 248 --temperature 0 --run-id local_20_20260716

python run.py \
  --execution-mode paired \
  --modelname local/local-qwen2.5-7b-instruct \
  --guessmodelname local/local-qwen2.5-7b-instruct \
  --sample-ids "$SAMPLES" --num-steps 8 --top-k 1 \
  --seed 248 --temperature 0 --run-id local_20_20260716

python run.py \
  --execution-mode paired \
  --modelname local/local-qwen2.5-7b-instruct \
  --guessmodelname local/local-qwen2.5-7b-instruct \
  --sample-ids "$SAMPLES" --num-steps 8 --top-k 3 \
  --seed 248 --temperature 0 --run-id local_20_20260716
```

图表：

```bash
python scripts/summarize_experiments.py \
  --run-dirs "$SEQ,$PAIR1,$PAIR3" \
  --output-dir reproduction/experiments/local_20_20260716/figures
```

`local` 是本机 vLLM 协议占位值，不是付费 API Key。
