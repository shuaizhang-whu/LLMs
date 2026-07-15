"""HotPotQA 实验的集中配置常量。

这些值在模块导入时直接生效，当前 CLI 只允许覆盖 Actor 和 Guess 模型名称，
样本数、步数、随机种子和 Top-K 都需要从本模块读取。不要把真实 API Key
写入受 Git 跟踪的源码；后续本地服务适配应改为从环境变量读取。
"""

# OpenRouter 的占位配置。模型名称中的 provider 前缀也会进入结果目录命名。
openrouter_api_key = "your-openrouter-api-key"
openrouter_model_name = "openai/gpt-4"
openrouter_guess_model_name = "openai/gpt-5-nano"

# Prompt 文件和 Actor 的文字角色说明。所有路径都相对 hotpotqa/ 工作目录。
prompts_folder = "./prompts/"
prompt_file = "prompts_naive.json"
agent_role = "Question Answering Agent"

# 数据采样与运行控制。
# random_seed 只用于 runner 中打乱候选索引，不等同于 LLM 生成随机种子。
random_seed = 248
# dev split 的预期样本总数；runner 先构造 range(num)，再截取 n_samples_to_run。
num = 7405
# webthink 使用 range(1, n_steps_to_run)，所以最多记录 n_steps_to_run - 1 轮。
n_steps_to_run = 8
n_samples_to_run = 20
# Gemini ClientError/ServerError 的退避时间；其他 provider 异常不走这两个分支。
client_error_sleep_time = 60
server_error_sleep_time = 60
# cleanup_trajs 用这四个文件判断一个样本目录是否完整。
trajectory_filenames = ["log.txt", "normalobs.json", "simobs.json", "metrics.json"]
# 模拟轨迹每步要求 Actor 给出的候选 action 数量。
guess_num_actions = 3
max_agent_retries = 1
max_guess_retries = 3

# Actor LLM 生成设置：既用于真实 Top-1 action，也用于模拟历史上的 Top-K action。
max_output_tokens = 1000
top_p = 1
temperature = 1

# Observation Guess LLM 设置：该模型预测 Wikipedia 风格的页面内容，而非直接猜 action。
max_guess_output_tokens = 100
guess_top_p = 0.9
guess_temperature = 0.1
