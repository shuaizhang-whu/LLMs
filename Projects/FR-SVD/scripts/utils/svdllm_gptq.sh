#!/bin/bash

# run data whitening with 20% compression ratio
python SVDLLM.py --model /mnt/data/zhangshuai/LLMs/Models/jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path SVD_GPTQ

# further compress the model with GPTQ-4bit
CUDA_VISIBLE_DEVICES=2 python quant_llama.py --model_path compress_output/_mnt_data_zhangshuai_LLMs_Models_jeffwan_llama_7b_hf_whitening_only_0.8.pt --dataset wikitext2 --wbits 4 --true-sequential --act-order --new-eval  --save output_SVDLLM_GPTQ/svdllm_gptq_4.pt

# CUDA_VISIBLE_DEVICES=0 python SVDLLM.py --model /mnt/data/zhangshuai/LLMs/Models/jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path compress_output

