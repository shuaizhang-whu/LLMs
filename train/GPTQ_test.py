from transformers import AutoTokenizer,AutoModelForCausalLM,GPTQConfig
from datasets import load_dataset

en = load_dataset("allenai/c4","en")

mode_id = "/mnt/data/zhangshuai/LLMs/Models/Llama-3.2-1B"
tokennizer = AutoTokenizer.from_pretrained(mode_id)
gptq_config = GPTQConfig(bits=4, dataset=en, tokennizer = tokennizer)

quantized_model = AutoModelForCausalLM.from_pretrained(mode_id,device_map="auto",quantization_config=gptq_config)

quantized_model.save_pretrained("/mnt/data/zhangshuai/LLMs/Models/quantization/Llama-3.2-1B-gptq")

tokennizer.save_pretrained("/mnt/data/zhangshuai/LLMs/Models/quantization/Llama-3.2-1B-gptq")