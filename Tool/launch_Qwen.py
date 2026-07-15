import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_path ="/mnt/data/zhangshuai/LLMs/Models/Qwen3-4B"):
        assert os.path.isdir(model_path ), f"❌ 模型路径不存在：{model_path }"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path ,
                                                       trust_remote_code=True,
                                                       local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path ,
                                                          torch_dtype="auto",
                                                          device_map="auto",
                                                          trust_remote_code=True,
                                                          local_files_only=True)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=32768,
            )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

        # 分离 <think> 和正常回答
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

        # 更新历史记录（只加入最终回答）
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": content})

        return thinking_content, content

# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()
    print("🤖 Qwen Chatbot (输入 'exit' 可退出对话)\n")

    while True:
        user_input = input("你：")
        if user_input.strip().lower() == "exit":
            print("🛑 结束对话")
            break

        thinking, answer = chatbot.generate_response(user_input)
        if thinking:
            print("\n🧠 思维链:\n" + thinking)
        print("\n🤖 回答:\n" + answer)
        print("-" * 40)
