"""本地 OpenAI-compatible 路由的最小回归测试。"""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.llm_client import LLMClient


class LocalLLMClientTest(unittest.TestCase):
    """确保本地 smoke test 不初始化或调用任何云 provider。"""

    @patch("src.llm_client.openai.OpenAI")
    def test_local_prefix_uses_configurable_base_url_and_strips_prefix(self, openai_cls):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Search[test]"))],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )
        local_client = MagicMock()
        local_client.chat.completions.create.return_value = completion
        openai_cls.return_value = local_client

        with patch.dict(
            os.environ,
            {
                "LOCAL_OPENAI_BASE_URL": "http://127.0.0.1:8123/v1",
                "LOCAL_OPENAI_API_KEY": "local-test-key",
            },
            clear=False,
        ):
            client = LLMClient(
                "local/test-model",
                temperature=0.1,
                max_tokens=32,
                top_p=0.9,
                seed=248,
            )
            result = client.call("prompt", stop=["Observation"])

        openai_cls.assert_called_once_with(
            base_url="http://127.0.0.1:8123/v1", api_key="local-test-key"
        )
        self.assertEqual(result, "Search[test]")
        kwargs = local_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["stop"], ["Observation"])
        self.assertEqual(kwargs["seed"], 248)
        self.assertEqual(
            client.get_stats(),
            {
                "model": "local/test-model",
                "provider": "local",
                "calls": 1,
                "failed_calls": 0,
                "prompt_tokens": 11,
                "output_tokens": 3,
                "reasoning_tokens": 0,
                "total_tokens": 14,
                "latency_seconds": client.get_stats()["latency_seconds"],
            },
        )


if __name__ == "__main__":
    unittest.main()
