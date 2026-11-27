import asyncio
import json
import unittest

from scripts.run_qwen35_mechanism import ChunkHTTP, token_boundary
from spork_core.prefix_replay import StreamState


class Tests(unittest.IsolatedAsyncioTestCase):
    def test_boundary_comparison_preserves_suffix_and_requires_valid_tool(self):
        self.assertEqual(token_boundary([1, 2, 3], 2, 3, True),
            {"through_stop": [1, 2], "suffix": [3], "suffix_is_only_eos": True})
        self.assertFalse(token_boundary([1, 2, 9], 2, 3, True)["suffix_is_only_eos"])
        self.assertEqual(token_boundary([1, 2, 3], 2, 3, False)["through_stop"], [1, 2, 3])
        self.assertEqual(token_boundary([1, 3], 2, 3, True)["through_stop"], [1, 3])

    async def test_stream_retains_original_stop_reason_text_and_extra_eos(self):
        payloads = [
            {"choices": [{"text": "call", "token_ids": [1], "finish_reason": None}]},
            {"choices": [{"text": "", "token_ids": [2, 3], "finish_reason": "stop", "stop_reason": "</tool_call>"}]},
            {"choices": [], "usage": {"completion_tokens": 3}},
        ]
        class Response:
            def raise_for_status(self): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            @property
            def content(self):
                async def stream():
                    for payload in payloads:
                        yield ("data: " + json.dumps(payload) + "\n").encode()
                    yield b"data: [DONE]\n"
                return stream()
        class Session:
            def post(self, *args, **kwargs): return Response()
        http = ChunkHTTP("http://unused/v1", "cpu-mock", 1)
        state = StreamState()
        await http.stream(Session(), [9], state, max_tokens=10, seed=42)
        self.assertEqual(state.token_ids, [1, 2, 3])
        self.assertEqual(state.text, "call")
        self.assertEqual(state.finish_reason, "stop")
        self.assertEqual([x["payload"] for x in http.chunks], payloads)
        self.assertEqual(state.usage["completion_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
