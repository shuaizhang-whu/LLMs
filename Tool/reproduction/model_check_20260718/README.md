# 模型静态校验记录（2026-07-18）

本目录只记录匿名、无 GPU 的模型静态校验。执行时显式移除了常见 Hugging Face
令牌环境变量，并设置 `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`；未读取或修改
`Tool/download_model.py`。

## 工具环境

- Python: `/mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/python`
  (`3.12.13`)
- Hugging Face CLI: `/mnt/data/zhangshuai/anaconda3/envs/agent-action-vllm/bin/hf`
  (`1.23.0`)
- Hub endpoint: `https://huggingface.co`

## 固定 revision

- `Qwen/Qwen3.5-4B`: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- `Qwen/Qwen3.5-9B`: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- `Qwen/Qwen3.6-27B-FP8`: `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`

## 结果

- 4B：远端 14 个文件均存在；7 个 JSON、2 个 safetensors header、3 个
  LFS SHA-256 已通过模型管理器的逐文件检查。
- 9B：远端 16 个文件均存在；7 个 JSON、4 个 safetensors header、5 个
  LFS SHA-256 已通过模型管理器的逐文件检查。
- 初版两条 `verify` 最终均返回 1。根因不是模型内容，而是 `hf cache
  verify --fail-on-extra-files` 将 `hf download --local-dir` 自己生成的
  `.cache/huggingface/**` 元数据判定为额外文件。修复为由模型管理器自身
  过滤该缓存并继续严格检查其他额外文件后，最终 17 项单元测试全部通过。
- 4B 与 9B 的修复后验证均退出 0，状态为 `STATIC_VERIFIED`。
- 27B 首轮 Xet 续传在只剩两个文件时连续超过 10 分钟没有任何 I/O，且连接
  出现 `CLOSE_WAIT`。只向该下载进程组发送 SIGINT 后 2 秒内退出，退出码
  130；没有使用 SIGTERM 或 SIGKILL，断点文件保留。
- 27B 随后在相同固定 revision 上禁用 Xet、通过官方 Hub 普通 HTTP 原地续传。
  HTTP 读取超时由 CLI 自动恢复，最终退出 0：远端 80 个文件、本地可见总计
  30,890,040,900 bytes；9 个 JSON、66 个 safetensors header、67 个 LFS
  SHA-256 与 `hf cache verify` 均通过，状态为 `STATIC_VERIFIED`。
- `.cache/huggingface` 中仍保留两个共 68,032,404 bytes 的旧断点临时文件；
  它们不属于远端模型清单且不影响严格校验，本轮按“不自动删除”约束保留。
- 所有模型仍未进行 vLLM 加载与生成，不能标记为 `READY`。

初版失败输出见 `verify_qwen3.5-4b.log` 与 `verify_qwen3.5-9b.log`，修复后
成功输出见同名 `_fixed.log`；两次 27B 续传日志分别见 `_xet` 对应的主日志
与 `_http_retry.log`，退出码单独保存在 `.exitcode` 文件。本地最终一级文件名
及字节数见 `local_inventory_final.log`。
