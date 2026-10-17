"""Gemini、OpenAI、OpenRouter 与本地兼容服务的同步文本生成客户端封装。

使用 ``local/<served-model-name>`` 可把请求发给 OpenAI-compatible 本地服务。
各 provider 客户端按模型路由延迟构造，因此本地 smoke test 不需要任何云 API Key。
"""

import os
import threading
import time

import openai
from google import genai
from google.genai import types

from . import constants


class LLMClient:
    """根据模型名前缀路由到 Gemini、OpenAI、OpenRouter 或本地服务。"""

    def __init__(self, model_name, temperature, max_tokens, top_p, seed=None):
        """保存生成参数，并且只构造当前模型实际需要的 provider 客户端。"""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed
        self.provider = self._get_provider(model_name)
        self._stats_lock = threading.Lock()
        self._stats = {
            "model": model_name,
            "provider": self.provider,
            "calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
        }

        if self.provider == "local":
            base_url = os.environ.get("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
            api_key = os.environ.get("LOCAL_OPENAI_API_KEY", "local")
            self.local_client = openai.OpenAI(base_url=base_url, api_key=api_key)
            # local/ 是客户端路由标记，不是 vLLM 暴露的真实模型名称。
            self.request_model_name = model_name.removeprefix("local/")
        elif self.provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", getattr(constants, "gemini_api_key", None))
            self.gemini_client = genai.Client(api_key=api_key)
            self.request_model_name = model_name
        elif self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", getattr(constants, "openai_api_key", None))
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.request_model_name = model_name
        else:
            api_key = os.environ.get("OPENROUTER_API_KEY", constants.openrouter_api_key)
            self.openrouter_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            self.request_model_name = model_name

    @staticmethod
    def _get_provider(model_name):
        """把模型名映射到 provider；``local/`` 是本仓库新增的显式本地前缀。"""
        if model_name.startswith("local/"):
            return "local"
        if model_name.startswith("gemini"):
            return "gemini"
        if model_name.startswith("gpt"):
            return "openai"
        return "openrouter"

    def call(self, prompt, stop=None):
        """按模型名前缀执行一次同步请求并返回文本。

        ``local/*`` 路由到本地兼容服务，``gemini*`` 路由到 Gemini，``gpt*``
        路由到 OpenAI，其余模型名落入 OpenRouter。
        """
        started = time.perf_counter()
        with self._stats_lock:
            self._stats["calls"] += 1
        try:
            if self.provider == "local":
                text, usage = self._local_call(prompt, stop)
            elif self.provider == "gemini":
                text, usage = self._gemini_call(prompt, stop)
            elif self.provider == "openai":
                text, usage = self._openai_call(prompt, stop)
            else:
                text, usage = self._openrouter_call(prompt, stop)
        except Exception:
            with self._stats_lock:
                self._stats["failed_calls"] += 1
                self._stats["latency_seconds"] += time.perf_counter() - started
            raise

        with self._stats_lock:
            self._stats["prompt_tokens"] += usage["prompt_tokens"]
            self._stats["output_tokens"] += usage["output_tokens"]
            self._stats["reasoning_tokens"] += usage["reasoning_tokens"]
            self._stats["total_tokens"] += usage["total_tokens"]
            self._stats["latency_seconds"] += time.perf_counter() - started
        return text

    def get_stats(self):
        """返回线程安全的累计调用统计快照。"""
        with self._stats_lock:
            return dict(self._stats)

    def stats_since(self, before):
        """用两个累计快照计算单样本增量。"""
        after = self.get_stats()
        result = {"model": after["model"], "provider": after["provider"]}
        for key in (
            "calls",
            "failed_calls",
            "prompt_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "latency_seconds",
        ):
            result[key] = after[key] - before.get(key, 0)
        return result

    @staticmethod
    def _openai_usage(response):
        """从 OpenAI-compatible 响应中提取 token；缺失时显式记为 0。"""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = int(
            getattr(completion_details, "reasoning_tokens", 0) or 0
        )
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + output_tokens)
            or prompt_tokens + output_tokens
        )
        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _gemini_usage(response):
        """从 Gemini usage_metadata 中提取 token；字段缺失时记为 0。"""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        visible_output_tokens = int(
            getattr(usage, "candidates_token_count", 0) or 0
        )
        reasoning_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
        output_tokens = visible_output_tokens + reasoning_tokens
        total_tokens = int(
            getattr(usage, "total_token_count", prompt_tokens + output_tokens)
            or prompt_tokens + output_tokens
        )
        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
        }

    def _local_call(self, prompt, stop):
        """通过 OpenAI Chat Completions 协议调用本地 vLLM 服务。"""
        request = {
            "model": self.request_model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": stop,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        if self.seed is not None:
            request["seed"] = self.seed
        response = self.local_client.chat.completions.create(
            **request,
        )
        return response.choices[0].message.content, self._openai_usage(response)

    def _gemini_call(self, prompt, stop):
        """调用 Gemini generate_content，并仅将 stop 写入生成配置。

        显式传入实验温度、Top-p、输出上限和随机种子，避免 provider 默认值
        随时间变化后悄悄改变复现协议。
        """
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
            stop_sequences=stop,
            seed=self.seed,
        )
        response = self.gemini_client.models.generate_content(
            model=self.request_model_name, contents=prompt, config=config
        )
        return str(response.text), self._gemini_usage(response)

    def _openai_call(self, prompt, stop):
        """调用 OpenAI Chat Completions 并返回首个 choice 的正文。

        Chat Completions 的 stop 参数与本地 OpenAI-compatible 路径保持一致。
        """
        request = {
            "model": self.request_model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": stop,
        }
        if self.seed is not None:
            request["seed"] = self.seed
        response = self.openai_client.chat.completions.create(**request)
        return response.choices[0].message.content, self._openai_usage(response)

    def _openrouter_call(self, prompt, stop):
        """通过 OpenAI SDK 调用 OpenRouter，并返回首个 choice 的正文。

        与 OpenAI 分支相同，转发 stop 和可选 seed。
        """
        request = {
            "model": self.request_model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": stop,
        }
        if self.seed is not None:
            request["seed"] = self.seed
        response = self.openrouter_client.chat.completions.create(**request)
        return response.choices[0].message.content, self._openai_usage(response)
