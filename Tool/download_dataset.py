import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载数据集
# os.system('huggingface-cli download --repo-type dataset --resume-download yahma/alpaca-cleaned --local-dir /data/zhangs/LLMs/Datasets/alpaca-cleaned')

# os.system('huggingface-cli download --repo-type dataset --resume-download yahma/alpaca-cleaned --local-dir /mnt/data/zhangshuai/LLMs/Datasets/alpaca-cleaned')

os.system('huggingface-cli download --repo-type dataset --resume-download Salesforce/wikitext --local-dir /mnt/data/zhangshuai/LLMs/Datasets/wikitext')

