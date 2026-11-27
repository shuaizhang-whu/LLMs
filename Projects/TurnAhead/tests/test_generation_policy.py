"""Generation policy must survive parent/child and probe transport copies."""
from types import SimpleNamespace
import unittest

from spork_core.prefix_replay import PrefixHttpClient
from spork_core.successor_prefetch import TrackedHTTP, generation_key, rejection_reason
from spork_core.vllm_client import VllmClient


class Tests(unittest.TestCase):
    def test_policy_is_shared_by_parent_child_and_probe_requests(self):
        client = VllmClient("http://local/v1", "model", presence_penalty=1.5)
        for transport in [TrackedHTTP(client), TrackedHTTP(client),
                          PrefixHttpClient(client.base_url, client.model,
                                           presence_penalty=client.presence_penalty)]:
            for stream in [False, True]:
                body = transport.body([1, 2], max_tokens=32, seed=42, stream=stream)
                self.assertEqual(body["presence_penalty"], 1.5)
                self.assertEqual(body["temperature"], 0)
        self.assertEqual(client.request_body("prompt", max_tokens=32)["presence_penalty"], 1.5)

    def test_policy_change_invalidates_child_reuse(self):
        client = VllmClient("http://local/v1", "model")
        cfg = SimpleNamespace(seed=42, baseline_max_tokens=32,
                              context_window_tokens=128, enable_thinking=True)
        adapter = SimpleNamespace(main_stop=("</tool_call>",))
        old = generation_key(cfg, client, adapter)
        client.presence_penalty = 1.5
        new = generation_key(cfg, client, adapter)
        self.assertNotEqual(old, new)
        self.assertEqual(new["presence_penalty"], 1.5)


if __name__ == "__main__":
    unittest.main()
