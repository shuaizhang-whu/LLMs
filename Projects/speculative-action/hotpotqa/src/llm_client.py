"""Gemini、OpenAI 与 OpenRouter 的同步文本生成客户端封装。

依赖包在导入模块时即加载，三个 provider 客户端则在每个 ``LLMClient`` 实例
初始化时全部创建，不会根据当前模型延迟构造。因此，即便只走一个 provider，
另外两个客户端所需的配置字段和初始化条件也必须可用。当前实现没有本地
OpenAI-compatible 服务的可配置 ``base_url`` 参数。
"""

import openai
from google import genai
from google.genai import types

from . import constants


class LLMClient:
    """根据模型名前缀路由到 Gemini、OpenAI 或 OpenRouter 的统一客户端。"""

    def __init__(self, model_name, temperature, max_tokens, top_p):
        """保存生成参数，并立即构造三个 provider 的客户端。

        客户端构造是 eager 的：它不依据 ``model_name`` 只初始化目标 provider。
        这里直接读取 ``constants`` 中的三个 API Key 字段，且 OpenRouter 地址
        固定为其公开 API 端点。
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        # 即使最终只调用 OpenAI 或 OpenRouter，Gemini 客户端仍会在这里创建。
        self.gemini_client = genai.Client(api_key=constants.gemini_api_key)
        self.openai_client = openai.OpenAI(api_key=constants.openai_api_key)
        self.openrouter_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=constants.openrouter_api_key,
        )

    def call(self, prompt, stop=None):
        """按模型名前缀执行一次同步请求并返回文本。

        ``gemini*`` 路由到 Gemini，``gpt*`` 路由到 OpenAI，其余所有模型名均
        落入 OpenRouter；这里没有显式 provider 参数或本地服务分支。
        """
        if self.model_name.startswith("gemini"):
            return self._gemini_call(prompt, stop)
        elif self.model_name.startswith("gpt"):
            return self._openai_call(prompt, stop)
        else:
            return self._openrouter_call(prompt, stop)

    def _gemini_call(self, prompt, stop):
        """调用 Gemini generate_content，并仅将 stop 写入生成配置。

        构造函数保存的 temperature、max_tokens 和 top_p 没有在此请求中传入。
        """
        if stop is not None:
            config = types.GenerateContentConfig(stop_sequences=stop)
        else:
            config = types.GenerateContentConfig()
        response = self.gemini_client.models.generate_content(
            model=self.model_name, contents=prompt, config=config
        )
        return str(response.text)

    def _openai_call(self, prompt, stop):
        """调用 OpenAI Chat Completions 并返回首个 choice 的正文。

        ``stop`` 参数虽然由上层传入，但当前请求体没有使用它。
        """
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content

    def _openrouter_call(self, prompt, stop):
        """通过 OpenAI SDK 调用 OpenRouter，并返回首个 choice 的正文。

        与 OpenAI 分支相同，当前请求体不会转发 ``stop`` 参数。
        """
        response = self.openrouter_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content
