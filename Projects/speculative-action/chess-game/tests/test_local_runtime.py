import threading
import unittest
from types import SimpleNamespace

from local_runtime import (
    ACTOR_ROUTE,
    LOCAL_DUMMY_API_KEY,
    SPECULATOR_ROUTE,
    ChatResult,
    ChessLocalRuntime,
    EndpointConfig,
    OpenAICompatibleChatClient,
    RequestContext,
    Trace,
    TraceRecorder,
    build_actor_messages,
    canonical_actor_prompt,
    parse_actor_action,
    parse_prediction_actions,
    stable_fingerprint,
    stable_seed,
)


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
START_MOVES = [
    "a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3", "d2d4",
    "e2e3", "e2e4", "f2f3", "f2f4", "g2g3", "g2g4", "h2h3", "h2h4",
    "b1a3", "b1c3", "g1f3", "g1h3",
]


class FakeClock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        current = self.value
        self.value += 0.1
        return current


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


class FakeCompletions:
    def __init__(self, streams):
        self.streams = list(streams)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.streams.pop(0)


class FakeOpenAI:
    def __init__(self, streams):
        self.chat = SimpleNamespace(completions=FakeCompletions(streams))


def chunk(content=None, reasoning=None, usage=None):
    choices = []
    if content is not None or reasoning is not None:
        choices = [SimpleNamespace(
            delta=SimpleNamespace(content=content, reasoning_content=reasoning)
        )]
    return SimpleNamespace(choices=choices, usage=usage)


class StableHelpersTests(unittest.TestCase):
    def test_fingerprint_and_seed_are_stable_and_mapping_order_independent(self):
        first = stable_fingerprint("position", {"b": 2, "a": 1}, [3, 4])
        second = stable_fingerprint("position", {"a": 1, "b": 2}, [3, 4])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(stable_seed("run", START_FEN, 248), stable_seed("run", START_FEN, 248))
        self.assertNotEqual(stable_seed("run", START_FEN, 248), stable_seed("run", START_FEN, 249))
        self.assertLessEqual(stable_seed("run", START_FEN, 248), 0x7FFFFFFF)

    def test_position_context_is_stable(self):
        one = RequestContext.for_position(ACTOR_ROUTE, START_FEN, 248)
        two = RequestContext.for_position(ACTOR_ROUTE, START_FEN, 248)
        self.assertEqual(one, two)
        self.assertNotEqual(
            one,
            RequestContext.for_position(SPECULATOR_ROUTE, START_FEN, 248),
        )

    def test_branch_label_does_not_change_generation_identity(self):
        authority = RequestContext.for_position(
            ACTOR_ROUTE,
            START_FEN,
            248,
            branch_id="authority-step-0",
            identity_parts=("white", "actor-model", "prompt", {"max_tokens": 2048}),
        )
        branch = RequestContext.for_position(
            ACTOR_ROUTE,
            START_FEN,
            248,
            branch_id="candidate-e2e4",
            identity_parts=("white", "actor-model", "prompt", {"max_tokens": 2048}),
        )
        self.assertEqual(authority.seed, branch.seed)
        self.assertEqual(authority.fingerprint, branch.fingerprint)
        self.assertNotEqual(authority.request_id, branch.request_id)

    def test_canonical_actor_prompt_ignores_legal_move_order(self):
        prompt_a = canonical_actor_prompt(START_FEN, START_MOVES)
        prompt_b = canonical_actor_prompt(f"  {START_FEN}  ", reversed(START_MOVES))
        self.assertEqual(prompt_a, prompt_b)
        messages_a = build_actor_messages(START_FEN, START_MOVES)
        messages_b = build_actor_messages(START_FEN, reversed(START_MOVES))
        self.assertEqual(messages_a, messages_b)
        self.assertIn("[GAME] FEN:", messages_a[1]["content"])


class ActionParsingTests(unittest.TestCase):
    def test_actor_returns_last_legal_bracketed_move(self):
        response = "Maybe [a1a8], then [E2E4], but final answer [d2d4]."
        self.assertEqual(parse_actor_action(response, START_MOVES), "[d2d4]")
        self.assertIsNone(parse_actor_action("[a1a8]", START_MOVES))

    def test_predictions_are_legal_unique_and_strictly_top_k(self):
        response = "[E2E4], [a1a8], [e2e4], [d2d4], [g1f3], [c2c4]"
        self.assertEqual(
            parse_prediction_actions(response, START_MOVES, top_k=3),
            ["[e2e4]", "[d2d4]", "[g1f3]"],
        )
        self.assertEqual(
            parse_prediction_actions("[a1a8], [e2e4]", START_MOVES, top_k=3),
            ["[e2e4]"],
        )
        with self.assertRaises(ValueError):
            parse_prediction_actions(response, START_MOVES, top_k=0)


class StreamingClientTests(unittest.TestCase):
    def make_context(self, route=ACTOR_ROUTE):
        return RequestContext.for_position(route, START_FEN, 248)

    def test_explicit_endpoint_and_thinking_parameters_are_forwarded(self):
        usage = SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=5,
            total_tokens=17,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
        stream = FakeStream([
            chunk(reasoning="think "),
            chunk(content="[e2"),
            chunk(content="e4]"),
            chunk(usage=usage),
        ])
        fake = FakeOpenAI([stream])
        endpoint = EndpointConfig("http://127.0.0.1:8100/v1/", "actor-model", api_key="")
        client = OpenAICompatibleChatClient(endpoint, client=fake, clock=FakeClock())

        result = client.chat(
            build_actor_messages(START_FEN, START_MOVES),
            self.make_context(),
            enable_thinking=True,
            max_tokens=2048,
            temperature=0.0,
        )

        self.assertIsInstance(result, ChatResult)
        self.assertEqual(result.content, "[e2e4]")
        self.assertEqual(result.reasoning_content, "think ")
        self.assertEqual(endpoint.api_key, LOCAL_DUMMY_API_KEY)
        call = fake.chat.completions.calls[0]
        self.assertEqual(call["model"], "actor-model")
        self.assertTrue(call["stream"])
        self.assertEqual(call["stream_options"], {"include_usage": True})
        self.assertEqual(call["seed"], self.make_context().seed)
        self.assertEqual(
            call["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True}},
        )
        self.assertEqual(result.trace.prompt_tokens, 12)
        self.assertEqual(result.trace.completion_tokens, 5)
        self.assertEqual(result.trace.total_tokens, 17)
        self.assertEqual(result.trace.reasoning_tokens, 2)
        self.assertAlmostEqual(result.trace.first_chunk_seconds, 0.1)
        self.assertAlmostEqual(result.trace.ttft_seconds, 0.1)
        self.assertAlmostEqual(result.trace.first_content_seconds, 0.2)
        self.assertTrue(stream.closed)

    def test_stream_without_usage_preserves_nulls(self):
        stream = FakeStream([chunk(), chunk(content="[d2d4]")])
        recorder = TraceRecorder()
        client = OpenAICompatibleChatClient(
            EndpointConfig("http://127.0.0.1:8100/v1", "actor-model"),
            recorder,
            client=FakeOpenAI([stream]),
            clock=FakeClock(),
        )
        result = client.chat(
            [{"role": "user", "content": "move"}],
            self.make_context(),
            enable_thinking=False,
            max_tokens=32,
        )
        self.assertEqual(result.content, "[d2d4]")
        self.assertIsNone(result.trace.prompt_tokens)
        self.assertIsNone(result.trace.completion_tokens)
        self.assertIsNone(result.trace.total_tokens)
        self.assertIsNone(result.trace.reasoning_tokens)
        self.assertEqual(recorder.snapshot(), (result.trace,))

    def test_runtime_routes_actor_and_speculator_to_distinct_endpoints(self):
        created = []

        def factory(**kwargs):
            fake = FakeOpenAI([FakeStream([chunk(content="[e2e4]")])])
            created.append((kwargs, fake))
            return fake

        runtime = ChessLocalRuntime(
            EndpointConfig("http://actor.local/v1", "actor-model"),
            EndpointConfig("http://spec.local/v1", "spec-model"),
            client_factory=factory,
            clock=FakeClock(),
        )
        actor_result = runtime.chat(
            [{"role": "user", "content": "actor"}],
            self.make_context(ACTOR_ROUTE),
            enable_thinking=True,
            max_tokens=64,
        )
        spec_result = runtime.chat(
            [{"role": "user", "content": "spec"}],
            self.make_context(SPECULATOR_ROUTE),
            enable_thinking=False,
            max_tokens=32,
        )

        self.assertEqual(created[0][0]["base_url"], "http://actor.local/v1")
        self.assertEqual(created[1][0]["base_url"], "http://spec.local/v1")
        self.assertEqual(created[0][0]["api_key"], LOCAL_DUMMY_API_KEY)
        self.assertEqual(created[1][0]["api_key"], LOCAL_DUMMY_API_KEY)
        self.assertEqual(created[0][1].chat.completions.calls[0]["model"], "actor-model")
        self.assertEqual(created[1][1].chat.completions.calls[0]["model"], "spec-model")
        self.assertTrue(
            created[0][1].chat.completions.calls[0]["extra_body"]
            ["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertFalse(
            created[1][1].chat.completions.calls[0]["extra_body"]
            ["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertEqual(actor_result.trace.route, ACTOR_ROUTE)
        self.assertEqual(spec_result.trace.route, SPECULATOR_ROUTE)


class TraceRecorderTests(unittest.TestCase):
    def test_recorder_is_thread_safe(self):
        recorder = TraceRecorder()
        trace = Trace(
            request_id="request",
            route=ACTOR_ROUTE,
            fingerprint="0" * 64,
            base_url="http://local/v1",
            model="model",
            seed=1,
            started_at=0.0,
            finished_at=1.0,
            latency_seconds=1.0,
            first_chunk_seconds=None,
            ttft_seconds=None,
            first_content_seconds=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            reasoning_tokens=None,
        )
        threads = [threading.Thread(target=recorder.record, args=(trace,)) for _ in range(50)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(recorder), 50)
        self.assertEqual(len(recorder.snapshot()), 50)


if __name__ == "__main__":
    unittest.main()
