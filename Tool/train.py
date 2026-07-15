# 选择模型
from unsloth import FastLanguageModel 
#  是 Unsloth 提供的优化版模型加载类，可实现：4bit 量化加载（降低显存占用）自动 RoPE 扩展（支持超长上下文）Flash Attention 加速（提升训练速度
import torch


max_seq_length = 2048 # 设置最大序列长度为 2048 token。Unsloth 内部自动启用 RoPE 扩展技术，即使原模型仅支持 2048 长度，也可通过插值扩展到更长序列（如 8k）。
dtype = None # 数据类型自动选择策略：Tesla T4/V100 等旧显卡：自动使用 float16,Ampere 架构以上显卡（如 RTX 30/40 系列）：自动使用 bfloat16
load_in_4bit = True #启用 4bit 量化加载：显存占用减少 50-75%,支持动态反量化（训练时自动恢复高精度计算）,与普通 4bit 量化（如 bitsandbytes）相比，Unsloth 的实现支持梯度检查点等高级特性

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/mnt/data/zhangshuai/LLMs/Models/Llama-3.2-1B", #指定模型路径，可为本地路径或 Hugging Face 上的模型名称；
    max_seq_length = max_seq_length,  # 传入之前定义的最大序列长度；
    dtype = dtype, # 使用之前定义的自动精度；
    load_in_4bit = load_in_4bit, # 是否启用 4bit 量化；
    # token = "hf_...", # 如果加载的是 HuggingFace 上 gated 模型（如 Meta 的 LLaMA3），需填写授权 token
)


print(f"\n模型加载完成")
'---------------------'


print(f"模型加载中...")
print(model) # 最后一条语句是表达式时，Jupyter会回显它的值

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template( # 会把一个结构化的对话转成 模型真正看到的字符串形式
    tokenizer, # 设置 tokenizer 的对话格式模板为 llama-3.1 风格
    chat_template = "llama-3.1",
)

print(tokenizer.chat_template)


# 加载数据集
from datasets import load_dataset
dataset = load_dataset("/mnt/data/zhangshuai/LLMs/Datasets/FineTome-100k", split = "train") # 加载你本地路径下的训练数据，返回 Hugging Face Dataset 对象 dataset
print(dataset[0])

# 定义一个用于数据集批处理的函数
def formatting_prompts_func(examples):
    convos = examples["conversations"] # 提取数据集中名为 "conversations" 的字段
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos] #将格式化后的纯文本作为新的 "text" 字段返回；
    return { "text" : texts, }
# 为 ShareGPT 格式的数据集做预处理和 prompt 构建，使其适配 LLaMA 3 的聊天模板格式

from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset) # 将原始数据集转换为 Unsloth 推荐的结构
dataset = dataset.map(formatting_prompts_func, batched = True,) # 将上一步标准化的 "conversations" 字段，格式化为训练用的 prompt 字符串
dataset # 自动输出处理后的数据结构，方便你检查格式是否正确
print(dataset[0])



# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# 训练模型

# 对已经插入 LoRA 的 LLaMA 模型进行 有监督微调（SFT）
from trl import SFTConfig, SFTTrainer # TRL 提供的轻量级微调 Trainer，专门用于对指令数据集进行监督学习微调，SFTConfig是Trainer 的配置参数类
from transformers import DataCollatorForSeq2Seq # 用于在训练时对样本进行打包（如 padding、动态 batch 编码等）。
trainer = SFTTrainer(
    model = model, # 你的 LoRA 插入后的 LLaMA 模型
    tokenizer = tokenizer, # 与模型配套的 tokenizer
    train_dataset = dataset, # 已格式化并包含 "text" 字段的训练数据
    dataset_text_field = "text", # 明确指定数据集中用于训练的字段名称
    max_seq_length = max_seq_length, # 每个样本最大 token 长度（训练时会 truncate）
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer), # 自动处理 padding、构建 input_ids 和 labels 的工具
    dataset_num_proc = 8, # 多进程加速数据处理
    packing = False, # 不开启样本“合并填充”，适合长文本训练（否则会训练更快但可能打断逻辑）

    args = SFTConfig(
        per_device_train_batch_size = 2, # 单张显卡上每次前向的 batch size
        gradient_accumulation_steps = 4, # 累积 4 次再更新梯度
        warmup_steps = 5, # 学习率预热，前5步线性增加到设置值，避免初始震荡
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps =30, # 最多训练 60 步（如测试运行用）
        learning_rate = 2e-4, # 微调推荐学习率（对于 LoRA 非常合适）
        logging_steps = 1, # 每一步打印日志，便于观察 loss
        optim = "adamw_8bit", # 使用 bitsandbytes 提供的 8bit 版本的 AdamW，节省显存
        weight_decay = 0.01, # 权重衰减（防止过拟合）
        lr_scheduler_type = "linear", # 学习率调度器
        seed = 3407, # 固定随机种子，结果可复现
        output_dir = "outputs", # 保存模型的目录
        report_to = "none", # 不连接到 WandB、TensorBoard 等日志平台，可设为 "wandb" 开启
    ),
)

# 检查训练字段

trainer.train_dataset[0].keys()  # 查看 SFTTrainer 内部 train_dataset 的 第一个样本包含哪些字段。这对验证数据是否被正确预处理


# 开始 LoRA 微调过程

trainer.train()



from transformers import AutoModelForCausalLM, AutoTokenizer # 引入基础 Hugging Face 模型和 tokenizer 加载器
from peft import PeftModel # 用于加载 LoRA 插件权重
import torch


tokenizer = AutoTokenizer.from_pretrained( # 加载你之前使用的 tokenizer
    "/mnt/data/zhangshuai/LLMs/Models/Llama-3.2-1B",
    torch_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained( # 加载未微调的原始 LLaMA 模型
    "/mnt/data/zhangshuai/LLMs/Models/Llama-3.2-1B",
    torch_dtype=torch.bfloat16
)

merged_model = PeftModel.from_pretrained( # 将训练好的 LoRA 权重 加载进 base_model，返回的是 PeftModel 对象
    base_model,
    "/mnt/data/zhangshuai/LLMs/outputs/checkpoint-60",
    torch_dtype=torch.bfloat16
)


messages = [
    {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize = True,
#     add_generation_prompt = True, # Must add for generation
#     return_tensors = "pt",
# ).to("cuda")
# inputs

from transformers import pipeline

base_model.model.max_seq_length = 2048 # 设置模型最大序列长度


pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer) # 构建一个推理管道（pipeline），让模型可以直接接收文本并输出生成结果

pipe("Compare two strings and find which one is lexicographically greater. The length of each string will not exceed 100 characters. Both strings will consist of lowercase alphabets only.")


merge_pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)

# 对相同的 prompt（任务描述）进行生成推理。
merge_pipe("Compare two strings and find which one is lexicographically greater. The length of each string will not exceed 100 characters. Both strings will consist of lowercase alphabets only.") 
