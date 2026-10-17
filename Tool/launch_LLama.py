import subprocess
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _pick_best_gpu_via_nvidia_smi(min_free_gib: float = 15.0):
    """通过 nvidia-smi 选择空闲显存最多的 GPU，不创建 CUDA Context。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free_mib = [int(x.strip()) for x in result.stdout.strip().splitlines()]
        best = max(range(len(free_mib)), key=lambda i: free_mib[i])
        best_free_gib = free_mib[best] / 1024.0
        if best_free_gib < min_free_gib:
            print(f"[Warning] 所有 GPU 空闲显存均不足 {min_free_gib:.0f} GiB，"
                  f"最多只有 cuda:{best}（{best_free_gib:.1f} GiB）。"
                  f"推理时可能 OOM，请等待其他进程释放显存后重试。")
        return best, best_free_gib
    except Exception as exc:
        print(f"[Warning] nvidia-smi 查询失败（{exc}），回退到 cuda:0")
        return 0, 0.0


def generate_text(model, tokenizer, prompt, device, max_new_tokens=200,
                  do_sample=True, top_p=0.95, temperature=0.7):
    """推理并返回新生成的文本（不含 prompt）。"""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main():
    model_path = "/data/zhangs/LLMs/Models/llama-7b"

    best_gpu, free_gb = _pick_best_gpu_via_nvidia_smi()
    device = f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[Auto device] 选择 {device}（空闲显存 {free_gb:.1f} GiB）")

    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("正在加载模型（fp16）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    print("模型加载完成。")

    # 简单测试
    test_out = generate_text(model, tokenizer, "The meaning of life is", device,
                             max_new_tokens=50, do_sample=False, top_p=1.0, temperature=1.0)
    print("测试输出:", test_out)

    # 交互式对话循环（Alpaca 指令模板）
    def build_prompt(instruction):
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

    print("\n=== 交互模式（输入 exit 退出）===")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue
        prompt = build_prompt(user_input)
        answer = generate_text(model, tokenizer, prompt, device,
                               max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.7)
        answer = answer.split("###")[0].strip()
        print("Model:", answer)


if __name__ == "__main__":
    main()
