# 实际执行命令

工作目录均为：

```text
/mnt/data/zhangshuai/LLMs/Projects/speculative-action/hotpotqa
```

公共执行约束：

```bash
env -u OPENAI_API_KEY -u GOOGLE_API_KEY -u GEMINI_API_KEY -u OPENROUTER_API_KEY \
  PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/speculative-action-mpl \
  HTTP_PROXY=http://127.0.0.1:9 \
  HTTPS_PROXY=http://127.0.0.1:9 \
  ALL_PROXY=http://127.0.0.1:9 \
  NO_PROXY= \
  /mnt/data/zhangshuai/anaconda3/envs/agent-action/bin/python <command>
```

执行顺序：

```bash
# 1. 对全部附带逐样本源文件生成修改前 SHA-256 清单
find run_metrics -type f \
  \( -name log.txt -o -name normalobs.json -o -name simobs.json -o -name metrics.json \) \
  -print0 | sort -z | xargs -0 sha256sum \
  > reproduction/offline_20260714/source_sha256_before.txt

# 2. 语法与标准库 unittest
python -m py_compile run.py src/grapher.py
python -m unittest discover -s tests -v

# 3. 无保存离线验收
python run.py --norun --getmetric
python run.py --norun --getmetric2

# 4. 保存聚合指标
python run.py --norun --getmetric --savemetrics
python run.py --norun --getmetric2 --savemetrics

# 5. README 支持的两类绘图
python run.py --norun --graph
python run.py --norun --graph3

# 6. 同时执行两类指标并跟踪全部网络系统调用
strace -f -e trace=network \
  -o reproduction/offline_20260714/network_trace.log \
  python run.py --norun --getmetric --getmetric2

# 7. 生成修改后哈希清单并逐字节比较
find run_metrics -type f \
  \( -name log.txt -o -name normalobs.json -o -name simobs.json -o -name metrics.json \) \
  -print0 | sort -z | xargs -0 sha256sum \
  > reproduction/offline_20260714/source_sha256_after.txt
cmp -s reproduction/offline_20260714/source_sha256_before.txt \
  reproduction/offline_20260714/source_sha256_after.txt

# 8. 独立解析并交叉核验 JSON、CSV、逐样本指标、PNG、网络跟踪和哈希
python reproduction/offline_20260714/validate_artifacts.py

# 9. 环境、diff 和秘密信息终检
python -m pip check
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
rg 'BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}' \
  hotpotqa CODEBASE_GUIDE_ZH.md
```

实际执行时，上述 Python 命令均使用绝对解释器路径和公共执行约束；标准输出分别保存在本目录的 `tests.log`、`getmetric_nosave.log`、`getmetric2_nosave.log`、`getmetric_save.log`、`getmetric2_save.log`、`graph.log` 与 `graph3.log`。

明确未执行：`python run.py`、`--cleanuptrajs`、`--graph2`、`Metrics.recalculate_metrics`、任何模型/API 调用。
