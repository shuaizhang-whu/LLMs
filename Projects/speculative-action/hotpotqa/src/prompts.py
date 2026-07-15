"""HotPotQA ReAct 与模拟 observation 使用的提示词模板。

模板本身属于实验协议的一部分。为保持 baseline 行为，本轮只解释用途，不翻译或
改写英文提示词；任何措辞变化都可能改变 action 格式、候选顺序和最终指标。
"""


class PromptTemplates:
    """集中保存 Actor、重试和 Guess LLM 的文本模板。"""

    # 同一个 Actor 在真实历史上要求 1 个 action，在模拟历史上要求 Top-K 个 action。
    ACTION_GUESS_PROMPT = (
        "Reason very very succinctly about the {i}th action, return {num_guesses} candidates for the next action. IMPORTANT FORMAT: "
        "Your response must be a first a thought based on the current situation and then the {num_guesses} options for actions as a list."
        "(exact syntax with square brackets and commas between actions!). For example: Action {i}: Search[Barack Obama], Search[Obama], Lookup[Barack] "
        "Ensure all actions are from the list of actions - Search, Lookup and Finish. Even if uncertain, return exactly {num_guesses} candidates. "
        "Reason very very quickly, no need to truly think about each of the actions, just return the most likely candidates and return precisely {num_guesses} valid actions - no more, no less."
    )

    # 正则未提取到合法 Search/Lookup/Finish 时追加的短重试说明。
    RETRY_PROMPT = (
        "Attempt {attempt} failed, please remember that you are acting as {role}"
        "and you need to output {num_guesses} valid actions out of the three possible actions - Search, Lookup and Finish. "
    )

    # ReAct 系统协议：声明三种可用动作及其文本格式。
    REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
        (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
        (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
        (3) Finish[answer], which returns the answer and finishes the task.
        Here are some examples.
        """

    # few-shot 示例之后、当前问题之前的最终格式约束。
    PROMPT_INSTRUCTION = """Now answer the question. Start with the thought always. Do not under any circumstances give me the actions first."""

    # 将当前 thought/action/observation 追加到下一轮 Actor 上下文。
    NEXT_STEP_PROMPT = """Thought {i}: {thought}
        Action {i}: {action}
        Observation {i}: {obs}"
        """

    # Guess LLM 根据 Search entity 生成拟 Wikipedia 页面；环境随后只保留前五句作为 observation。
    GUESS_STEP_PROMPT = "Give me information about {} in wikipedia style. If wikipedia information is not available still give me the information. Do not give me any error messages. Just give me the information and nothing else"
