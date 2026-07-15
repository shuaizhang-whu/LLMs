import os
# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
#
# 下载模型

# os.system('huggingface-cli download --resume-download huggyllama/llama-7b --local-dir /mnt/data/zhangshuai/LLMs/Models/llama-7b')

os.system('hf download jeffwan/llama-7b-hf --local-dir /mnt/data/zhangshuai/LLMs/Models/jeffwan/llama-7b-hf')

# os.system('hf download  circulus/llama-13b --local-dir /data/zhangs/LLMs/Models/llama-13b')

# os.system('huggingface-cli download --resume-download huggyllama/llama-30b --local-dir /data/zhangs/LLMs/Models/llama-30b')

# os.system('huggingface-cli download --resume-download Qwen/Qwen2.5-7B --local-dir /mnt/data/zhangshuai/LLMs/Models/Qwen2.5-7B')

# 下载需要令牌的模型，添加token
# os.system('huggingface-cli download --token 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 --resume-download meta-llama/Llama-2-7b --local-dir /data/zhangs/LLMs/Models/Llama-2-7b')

# 删除模型
# rm -rf /data/zhangs/LLMs/Models/Llama-2-7b

# 使用modelscope下载llama2
# 下载完整模型库
# modelscope download --model LLM-Research/llama-2-7b --local_dir /data/zhangs/LLMs/Models/Llama-2-7b
#modelscope download --model skyline2006/llama-30b --local_dir /data/zhangs/LLMs/Models/llama-30b
# 下载单个文件到指定文件夹
# modelscope download --model shakechen/Llama-2-7b README.md --local_dir /data/zhangs/LLMs/Models/Llama-2-7b
# SDK下载模型
# from modelscope import snapshot_download
# model_dir = snapshot_download('shakechen/Llama-2-7b')
# Git下载，请确保 lfs 已经被正确安装
# git lfs install
# git clone https://www.modelscope.cn/shakechen/Llama-2-7b.git
# 如果您希望跳过 lfs 大文件下载，可以使用如下命令
# GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/shakechen/Llama-2-7b.git






# 检查模型是否完整
# import os
# import json
# import subprocess
# import sys
# from pathlib import Path


# REPO_ID = "Qwen/Qwen3.5-35B-A3B"
# LOCAL_DIR = Path("/data/zhangs/LLMs/Models/Qwen/Qwen3.5-35B-A3B")

# # 至少先校验这些关键文件，避免出现权重在但 tokenizer/processor 缺失的情况。
# REQUIRED_FILES = [
#     "config.json",
#     "generation_config.json",
#     "model.safetensors.index.json",
#     "merges.txt",
#     "tokenizer.json",
#     "tokenizer_config.json",
#     "preprocessor_config.json",
# ]


# def list_missing_files(local_dir: Path, required_files: list[str]) -> list[str]:
#     return [name for name in required_files if not (local_dir / name).exists()]


# def bytes_to_human_readable(size: int) -> str:
#     units = ["B", "KB", "MB", "GB", "TB"]
#     value = float(size)
#     for unit in units:
#         if value < 1024 or unit == units[-1]:
#             return f"{value:.2f} {unit}"
#         value /= 1024
#     return f"{size} B"


# def load_index_file(index_path: Path) -> dict:
#     with index_path.open("r", encoding="utf-8") as f:
#         return json.load(f)


# def validate_shards_from_index(local_dir: Path) -> dict:
#     index_path = local_dir / "model.safetensors.index.json"
#     result = {
#         "index_exists": index_path.exists(),
#         "index_error": None,
#         "expected_shards": [],
#         "missing_shards": [],
#         "extra_shards": [],
#         "actual_total_size": 0,
#         "expected_total_size": None,
#     }

#     if not index_path.exists():
#         return result

#     try:
#         index_data = load_index_file(index_path)
#     except Exception as exc:
#         result["index_error"] = str(exc)
#         return result

#     expected_shards = sorted(set(index_data.get("weight_map", {}).values()))
#     actual_shards = sorted(p.name for p in local_dir.glob("model.safetensors-*.safetensors"))

#     missing_shards = [name for name in expected_shards if not (local_dir / name).exists()]
#     extra_shards = [name for name in actual_shards if name not in expected_shards]
#     actual_total_size = sum((local_dir / name).stat().st_size for name in expected_shards if (local_dir / name).exists())
#     expected_total_size = index_data.get("metadata", {}).get("total_size")

#     result.update(
#         {
#             "expected_shards": expected_shards,
#             "missing_shards": missing_shards,
#             "extra_shards": extra_shards,
#             "actual_total_size": actual_total_size,
#             "expected_total_size": expected_total_size,
#         }
#     )
#     return result


# def print_validation_report(local_dir: Path) -> bool:
#     missing_required = list_missing_files(local_dir, REQUIRED_FILES)
#     shard_report = validate_shards_from_index(local_dir)

#     print("\nValidation report:")
#     print(f"  - Local directory: {local_dir}")

#     if missing_required:
#         print("  - Missing required files:")
#         for name in missing_required:
#             print(f"    - {name}")
#     else:
#         print("  - Required files: OK")

#     if not shard_report["index_exists"]:
#         print("  - Shard index: missing")
#         return False

#     if shard_report["index_error"] is not None:
#         print(f"  - Shard index: unreadable ({shard_report['index_error']})")
#         return False

#     print(f"  - Expected shard count: {len(shard_report['expected_shards'])}")

#     if shard_report["missing_shards"]:
#         print("  - Missing weight shards:")
#         for name in shard_report["missing_shards"]:
#             print(f"    - {name}")
#     else:
#         print("  - Weight shards: OK")

#     if shard_report["extra_shards"]:
#         print("  - Extra weight shards:")
#         for name in shard_report["extra_shards"]:
#             print(f"    - {name}")
#     else:
#         print("  - Extra shards: none")

#     expected_total_size = shard_report["expected_total_size"]
#     actual_total_size = shard_report["actual_total_size"]
#     if expected_total_size is not None:
#         print(
#             "  - Indexed shard size: "
#             f"{bytes_to_human_readable(expected_total_size)} "
#             f"({expected_total_size} bytes)"
#         )
#         print(
#             "  - Local shard size: "
#             f"{bytes_to_human_readable(actual_total_size)} "
#             f"({actual_total_size} bytes)"
#         )
#     else:
#         print(
#             "  - Local shard size: "
#             f"{bytes_to_human_readable(actual_total_size)} "
#             f"({actual_total_size} bytes)"
#         )

#     return not missing_required and not shard_report["missing_shards"] and shard_report["index_error"] is None


# def run_download(repo_id: str, local_dir: Path) -> int:
#     os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#     local_dir.mkdir(parents=True, exist_ok=True)

#     cmd = [
#         "huggingface-cli",
#         "download",
#         "--resume-download",
#         repo_id,
#         "--local-dir",
#         str(local_dir),
#     ]

#     print("Running:", " ".join(cmd))
#     result = subprocess.run(cmd)
#     return result.returncode


# def main() -> int:
#     missing_before = list_missing_files(LOCAL_DIR, REQUIRED_FILES)
#     if missing_before:
#         print("Missing files before download:")
#         for name in missing_before:
#             print("  -", name)
#     else:
#         print("Required files already exist before download.")

#     return_code = run_download(REPO_ID, LOCAL_DIR)
#     validation_ok = print_validation_report(LOCAL_DIR)
#     missing_after = list_missing_files(LOCAL_DIR, REQUIRED_FILES)

#     if not validation_ok:
#         if missing_after:
#             print("\nDownload finished, but required files are still missing.")
#         else:
#             print("\nDownload finished, but shard validation did not pass.")
#         print("You can rerun this script to resume the download.")
#         return 1

#     if return_code != 0:
#         print("\nDownload command returned a non-zero exit code, but required files are present.")
#         print("Weights or auxiliary files may still need manual inspection.")
#         return return_code

#     print("\nDownload and validation completed successfully.")
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())
