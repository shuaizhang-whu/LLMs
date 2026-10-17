"""在线 Search 预取缓存的无网络单元测试。"""

import unittest
from unittest.mock import patch

from src.environment import WikiEnv


class SearchPrefetchTest(unittest.TestCase):
    def setUp(self):
        with patch("src.environment.LLMClient"):
            self.env = WikiEnv("local/dummy")
        self.env.reset()

    def test_strict_matching_prefetch_is_consumed_by_normal_step(self):
        result = {
            "requested_entity": "Alpha",
            "resolved_entity": "Alpha",
            "observation": "Alpha is a test.",
            "page": "Alpha is a test.",
            "result_titles": [],
            "fetch_seconds": 1.25,
            "num_requests": 1,
        }
        self.env.install_prefetched_search(
            "search[Alpha]",
            result,
            {"hidden_seconds": 1.0, "wait_seconds": 0.25},
        )

        with patch.object(self.env, "fetch_search_result") as fetch:
            observation, _, _, _ = self.env.step("search[Alpha]")

        fetch.assert_not_called()
        self.assertEqual(observation, "Alpha is a test.")
        self.assertTrue(self.env.last_search_metadata["prefetched"])
        self.assertEqual(self.env.last_search_metadata["hidden_seconds"], 1.0)

    def test_parameter_case_mismatch_does_not_consume_prefetch(self):
        prefetched = {
            "requested_entity": "Alpha",
            "resolved_entity": "Alpha",
            "observation": "prefetched",
            "page": "prefetched",
            "result_titles": [],
            "fetch_seconds": 1.0,
            "num_requests": 1,
        }
        fetched = {
            **prefetched,
            "requested_entity": "alpha",
            "resolved_entity": "alpha",
            "observation": "fresh",
            "page": "fresh",
        }
        self.env.install_prefetched_search("search[Alpha]", prefetched)

        with patch.object(self.env, "fetch_search_result", return_value=fetched) as fetch:
            observation, _, _, _ = self.env.step("search[alpha]")

        fetch.assert_called_once_with("alpha")
        self.assertEqual(observation, "fresh")
        self.assertFalse(self.env.last_search_metadata["prefetched"])


if __name__ == "__main__":
    unittest.main()
