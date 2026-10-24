"""CPU transport/lifecycle tests, not a simulated GPU performance claim."""
import asyncio
import time
import unittest

from scripts.check_actor_overlap import measure


class HTTP:
    def __init__(self, fail_first=False):
        self.fail_first = fail_first
        self.closed = set()

    async def stream(self, session, prompt, state, **kwargs):
        marker = prompt[0]
        try:
            await asyncio.sleep(.005)
            state.first_output_at = time.perf_counter()
            state.token_ids.append(marker)
            state.changed.set()
            for _ in range(5):
                await asyncio.sleep(.003)
                if self.fail_first and marker == 11:
                    raise RuntimeError("first request failed")
                state.token_ids.append(marker)
                state.changed.set()
            state.finish_reason = "length"
        finally:
            self.closed.add(marker)
            state.finished_at = time.perf_counter()
            state.changed.set()


class Checks(unittest.IsolatedAsyncioTestCase):
    pair = {"prompts": [{"token_ids": [11]}, {"token_ids": [22]}]}

    async def test_parallel_starts_second_after_first_token_and_keeps_both_outputs(self):
        http = HTTP()
        out = await measure(None, http, self.pair, "parallel", 6)
        first, second = out["responses"]
        self.assertGreaterEqual(second["start_offset_s"], first["start_offset_s"] + first["first_token_s"])
        self.assertGreater(out["overlap_s"], 0)
        self.assertEqual(first["token_ids"], [11] * 6)
        self.assertEqual(second["token_ids"], [22] * 6)
        self.assertEqual(http.closed, {11, 22})

    async def test_serial_condition_has_no_overlap(self):
        out = await measure(None, HTTP(), self.pair, "serial", 6)
        self.assertEqual(out["overlap_s"], 0)
        self.assertGreaterEqual(out["responses"][1]["start_offset_s"], out["responses"][0]["finish_offset_s"])

    async def test_transport_failure_closes_the_other_request(self):
        http = HTTP(fail_first=True)
        with self.assertRaisesRegex(RuntimeError, "first request failed"):
            await measure(None, http, self.pair, "parallel", 6)
        self.assertEqual(http.closed, {11, 22})


if __name__ == "__main__":
    unittest.main()
