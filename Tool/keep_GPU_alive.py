# 文件名：keep_gpu_alive.py
import os, time, argparse, torch, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
def print_mem(tag: str) -> None:
    mem_mb = torch.cuda.memory_allocated() / 1024 ** 2
    print(f"[{tag}] 显存占用：{mem_mb:.2f} MB")

def tiny_op(device: torch.device) -> None:
    a = torch.randn(2, 2, dtype=torch.float16, device=device)
    b = torch.randn(2, 2, dtype=torch.float16, device=device)
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    del a, b, _
    torch.cuda.empty_cache()

from tqdm import tqdm          # 新增
import math, sys, time, torch  # 原有

# … 上面内容保持不变 …

def main(seconds: int, interval: float) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 GPU")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    print(f"\n脚本将保持 GPU 活跃 {seconds} 秒，每 {interval} 秒运行一次小矩阵运算。")
    print_mem("启动")

    # 预热一次
    with torch.no_grad():
        tiny_op(device)
    print_mem("第一次运算后")

    # ——► 进度条区域 ◄——
    num_iters = math.ceil(seconds / interval)        # 预计总次数
    start_time = time.time()

    with tqdm(
        total=num_iters,
        desc="GPU keep-alive",
        unit="step",
        ncols=100,
        bar_format=(
            "{desc}: [{percentage:6.2f}%] |{bar}| "
            "{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}"
        )
    ) as pbar:
        for _ in range(num_iters):
            with torch.no_grad():
                tiny_op(device)

            mem_mb = torch.cuda.memory_allocated() / 1024 ** 2
            remaining = max(0, int(seconds - (time.time() - start_time)))
            pbar.set_postfix(mem=f"{mem_mb:.2f}MB", left=f"{remaining}s")
            pbar.update(1)

            if remaining <= 0:
                break
            time.sleep(min(interval, remaining))

    print_mem("结束")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="低显存 GPU 常驻脚本")
    parser.add_argument("--seconds", type=int, default=36000,
                        help="持续运行总秒数（默认 3600 秒）")
    parser.add_argument("--interval", type=float, default=1,
                        help="两次 tiny_op 之间的休眠秒数（默认 1 秒）")
    args = parser.parse_args()
    main(args.seconds, args.interval)
