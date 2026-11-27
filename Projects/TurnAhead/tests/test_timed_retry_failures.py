"""CPU regressions for cancellation, incomplete output and latency accounting."""
import asyncio
from contextlib import contextmanager
import json
import threading
import time
import unittest
from unittest.mock import patch

from test_launch_timing import Adapter, CALL, Client, Controller, fake_stream
from spork_core.executors import SimulatedToolExecutor
from spork_core.turn_runner import RetryTurnConfig, run_spork_retry_turn


class SlowTool(SimulatedToolExecutor):
    def __init__(self):
        super().__init__(latency_s=.04)
        self.started = threading.Event()
        self.finished = threading.Event()

    def execute(self, call):
        self.started.set()
        try:
            return super().execute(call)
        finally:
            self.finished.set()


async def finished_stream(client, session, prompt, state, **kwargs):
    state.text = '<think>done</think><tool_call>' + json.dumps(CALL)
    state.token_ids = list(state.text.encode())
    state.first_output_at = time.perf_counter()
    state.finish_reason = "stop"
    state.finished_at = time.perf_counter()
    state.changed.set()


@contextmanager
def transport(stream, complete=None):
    """Patch only the HTTP boundary; no model/server requests are made."""
    with patch("spork_core.prefix_replay.PrefixHttpClient.stream", new=stream):
        if complete is None:
            yield
        else:
            async def legacy(client, session, prompt, **kwargs):
                result = await complete(client, session, prompt, **kwargs)
                return result, result["wall_s"]

            with patch("spork_core.prefix_replay.PrefixHttpClient.complete", new=complete), \
                    patch.object(Client, "complete", new=legacy):
                yield


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def run_turn(self, cfg, executor=None):
        return await run_spork_retry_turn(
            None, Client(), Adapter(), [{"role": "user", "content": "lookup a"}],
            executor or SimulatedToolExecutor(latency_s=0), 1, cfg,
        )

    async def test_truncated_actor_never_executes_or_commits_a_tool(self):
        tool = SlowTool()

        async def truncated(*args, **kwargs):
            await finished_stream(*args, **kwargs)
            args[3].finish_reason = "length"

        with transport(truncated):
            with self.assertRaisesRegex(RuntimeError, "[Ii]ncomplete Actor"):
                await self.run_turn(RetryTurnConfig(launch_mode="off"), tool)
        self.assertFalse(tool.started.is_set())

    async def test_malformed_tool_call_is_not_a_valid_final_answer(self):
        with transport(finished_stream), patch.object(Adapter, "parse_main_tool_call", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "unparsable tool call"):
                await self.run_turn(RetryTurnConfig(launch_mode="off"))

    async def test_cancelling_turn_drains_canonical_tool_thread(self):
        tool = SlowTool()
        with transport(finished_stream):
            task = asyncio.create_task(self.run_turn(RetryTurnConfig(launch_mode="off"), tool))
            while not tool.started.is_set():
                await asyncio.sleep(.001)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(tool.finished.is_set(), "turn returned while the real tool thread still ran")

    async def test_clear_failure_is_not_a_successful_turn(self):
        async def clear():
            return {"error": "sidecar unavailable"}

        with transport(finished_stream):
            with self.assertRaisesRegex(RuntimeError, "sidecar unavailable"):
                await self.run_turn(RetryTurnConfig(launch_mode="off", d3_clear=clear))

    async def test_actor_end_time_does_not_include_waiting_for_a_late_prediction(self):
        prediction_started = asyncio.Event()

        async def stream(client, session, prompt, state, **kwargs):
            state.text = '<think>\n' + 'x' * 24
            state.token_ids = list(state.text.encode())
            state.first_output_at = time.perf_counter()
            state.changed.set()
            await prediction_started.wait()
            await finished_stream(client, session, prompt, state, **kwargs)

        async def complete(*args, **kwargs):
            prediction_started.set()
            await asyncio.sleep(.04)
            return {"text": json.dumps(CALL), "logprobs": {"confidence": 1.}, "wall_s": .04}

        with transport(stream, complete):
            result = await self.run_turn(RetryTurnConfig(launch_mode="now"))
        self.assertGreater(result["tool_result_ready_ms"] - result["main_end_ms"], 20)

    async def test_actor_failure_drains_speculative_tool_and_clears_d3(self):
        tool = SlowTool()
        clears = []

        async def stream(client, session, prompt, state, **kwargs):
            try:
                state.text = '<think>\n' + 'x' * 24
                state.token_ids = list(state.text.encode())
                state.first_output_at = time.perf_counter()
                state.changed.set()
                while not tool.started.is_set():
                    await asyncio.sleep(.001)
                raise RuntimeError("broken Actor stream")
            finally:
                state.finished_at = time.perf_counter()
                state.changed.set()

        async def complete(*args, **kwargs):
            return {"text": json.dumps(CALL), "logprobs": {"confidence": 1.}, "wall_s": 0.}

        async def clear():
            clears.append(True)
            return {"clear_status": "cleared"}

        with transport(stream, complete):
            with self.assertRaisesRegex(RuntimeError, "broken Actor stream"):
                await self.run_turn(RetryTurnConfig(launch_mode="now", d3_clear=clear), tool)
        self.assertTrue(tool.finished.is_set())
        self.assertEqual(len(clears), 1)

    async def test_invalid_configuration_starts_no_actor_request(self):
        calls = []

        async def stream(*args, **kwargs):
            calls.append(True)
            await finished_stream(*args, **kwargs)

        for cfg in (RetryTurnConfig(launch_mode="unknown"),
                    RetryTurnConfig(launch_mode="bert"),
                    RetryTurnConfig(launch_mode="now", continue_after_dispatch=True)):
            with transport(stream):
                with self.assertRaises(ValueError):
                    await self.run_turn(cfg)
                await asyncio.sleep(0)
        self.assertEqual(calls, [])

    async def test_cancel_during_bert_drains_worker_and_stops_actor(self):
        started, finished = threading.Event(), threading.Event()
        actor_stopped = asyncio.Event()

        class SlowController(Controller):
            def predict(self, inp):
                started.set()
                try:
                    return super().predict(inp)
                finally:
                    finished.set()

        async def stream(client, session, prompt, state, **kwargs):
            try:
                state.text = '<think>\n' + 'x' * 24
                state.token_ids = list(state.text.encode())
                state.changed.set()
                await asyncio.Event().wait()
            finally:
                state.finished_at = time.perf_counter()
                state.changed.set()
                actor_stopped.set()

        cfg = RetryTurnConfig(launch_mode="bert", launch_controller=SlowController(delay=.04))
        with transport(stream):
            task = asyncio.create_task(self.run_turn(cfg))
            while not started.is_set():
                await asyncio.sleep(.001)
            task.cancel()
            await asyncio.sleep(.002)
            task.cancel()  # Repeated cancellation must not abandon cleanup.
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(finished.is_set())
        self.assertTrue(actor_stopped.is_set())

    async def test_http_failure_is_visible_without_hidden_transport_retries(self):
        from aiohttp import ClientSession, web

        requests = []

        async def failed_completion(request):
            requests.append(await request.json())
            return web.Response(status=503, text="test unavailable")

        app = web.Application()
        app.router.add_post("/v1/completions", failed_completion)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = Client()
        client.base_url = f"http://127.0.0.1:{port}/v1"
        try:
            async with ClientSession(trust_env=False) as session:
                with transport(fake_stream):
                    result = await run_spork_retry_turn(
                        session, client, Adapter(), [], SimulatedToolExecutor(latency_s=0), 1,
                        RetryTurnConfig(launch_mode="now", max_retries=1),
                    )
            self.assertEqual(len(requests), 1)
            self.assertIn("503", result["probes"][0]["probe_error"])
            self.assertFalse(result["spec_used"])
            self.assertTrue(result["completion_valid"])
        finally:
            await runner.cleanup()

    async def test_legacy_d2_does_not_duplicate_first_sse_chunk(self):
        prediction_started = asyncio.Event()

        class LegacyAdapter(Adapter):
            def build_probe_prompt(self, messages, *, enable_thinking, observed_main_prefix):
                return observed_main_prefix

        class LegacyClient(Client):
            async def stream(self, session, prompt, *, on_first_token, on_token, **kwargs):
                await on_first_token(0., "abc")
                await asyncio.sleep(0)
                await on_token("abc", 1)
                await prediction_started.wait()
                return {"text": '<tool_call>' + json.dumps(CALL), "first_token_s": 0.}

            async def complete(self, *args, **kwargs):
                prediction_started.set()
                return await super().complete(*args, **kwargs)

        client = LegacyClient()
        await run_spork_retry_turn(None, client, LegacyAdapter(), [], SimulatedToolExecutor(latency_s=0),
                                   1, RetryTurnConfig())
        self.assertEqual(client.prompts[0], "abc")

    async def test_wasted_tool_cost_is_recorded_after_draining(self):
        async def complete(*args, **kwargs):
            return {"text": json.dumps({"name": "wrong", "arguments": {}}),
                    "logprobs": {"confidence": 1.}, "wall_s": 0.}

        with transport(fake_stream, complete):
            result = await self.run_turn(RetryTurnConfig(launch_mode="now"), SlowTool())
        self.assertTrue(result["spec_wasted"])
        self.assertFalse(result["spec_used"])
        self.assertGreaterEqual(result["spec_wall_s"], .035)
        self.assertLessEqual(result["tool_result_ready_ms"], result["turn_end_ms"])


if __name__ == "__main__":
    unittest.main()
