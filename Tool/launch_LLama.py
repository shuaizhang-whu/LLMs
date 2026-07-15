import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def main():
    model_path = "/mnt/data/zhangshuai/LLMs/Models/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.eos_token_id
    )

    print(generator("The meaning of life is", max_new_tokens=50)[0]["generated_text"])

    # 示例输入
    prompt = "The secret to happiness is"

    # 推理生成
    outputs = generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )

    # 打印结果
    print("=== 模型生成内容 ===")
    print(outputs[0]['generated_text'])

if __name__ == "__main__":
    main()
