import asyncio
import json
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.run_qwen35_mechanism import ChunkHTTP, token_boundary, seal, summarize
from spork_core.prefix_replay import StreamState


class Tests(unittest.IsolatedAsyncioTestCase):
    def test_timing_summary_excludes_changed_outputs_and_cross_session_pairs(self):
        base = {"id":"base", "kind":"pair", "case":"gain", "config":"spork", "repeat":0,
            "session_id":"s", "errors":[], "wall_s":10., "parent_finished_s":6., "post_parent_s":4.,
            "trace":{}, "turns":[{"main_token_ids":[1], "baseline_text":"one"},
                                    {"main_token_ids":[2], "baseline_text":"two"}]}
        for variant in ("valid", "text", "tokens", "session"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
                other=copy.deepcopy(base)
                other.update(id="raw",config="raw",wall_s=8.)
                if variant=="text": other["turns"][1]["baseline_text"]="different"
                if variant=="tokens": other["turns"][1]["main_token_ids"]=[9]
                if variant=="session": other["session_id"]="other"
                out=Path(folder)
                for row in (base,other): seal(out/"records"/(row["id"]+".json"),row)
                report=summarize(out,{"schedule":[{"id":"base"},{"id":"raw"}],"scope":"test"},None,2,3)
                metric=next(r for r in report["timing_by_case"] if r["case"]=="gain" and r["config"]=="raw")
                self.assertEqual(metric["strict_pairs"],int(variant=="valid"))
                self.assertEqual(metric["mean_saved_s"],2. if variant=="valid" else None)

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
