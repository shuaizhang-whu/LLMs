import time
import unittest
from concurrent.futures import Future

import chess

from chess_runner import (
    ActorActionError,
    BranchRecord,
    CacheIdentity,
    ChessExperimentRunner,
    validate_branch_cache,
)
from local_runtime import (
    ACTOR_ROUTE,
    SPECULATOR_ROUTE,
    ChatResult,
    Trace,
    TraceRecorder,
)


class FakeRuntime:
    """Thread-safe deterministic runtime with controllable delays/failures."""

    def __init__(
        self,
        *,
        prediction="hit",
        actor_delay=0.02,
        speculator_delay=0.003,
        fail_branch_prefix=None,
        invalid_actor=False,
    ):
        self.recorder = TraceRecorder()
        self.prediction = prediction
        self.actor_delay = actor_delay
        self.speculator_delay = speculator_delay
        self.fail_branch_prefix = fail_branch_prefix
        self.invalid_actor = invalid_actor

    def chat(
        self,
        messages,
        context,
        *,
        enable_thinking,
        max_tokens,
        temperature=0.0,
    ):
        started = time.perf_counter()
        route = context.route
        delay = self.actor_delay if route == ACTOR_ROUTE else self.speculator_delay
        time.sleep(delay)

        if (
            route == ACTOR_ROUTE
            and self.fail_branch_prefix
            and self.fail_branch_prefix in context.branch_id
        ):
            raise RuntimeError("injected branch failure")

        board = chess.Board(context.fen)
        legal = sorted(move.uci() for move in board.legal_moves)
        if route == ACTOR_ROUTE:
            content = "not a chess move" if self.invalid_actor else f"[{legal[0]}]"
        elif self.prediction == "hit":
            content = f"[{legal[0]}]"
        elif self.prediction == "miss":
            content = f"[{legal[-1]}]"
        elif self.prediction == "wrong_then_hit":
            content = f"[{legal[-1]}] [{legal[0]}]"
        else:
            raise AssertionError(f"unknown prediction mode {self.prediction}")

        finished = time.perf_counter()
        trace = Trace(
            request_id=context.request_id,
            route=route,
            fingerprint=context.fingerprint,
            base_url="http://fake.local/v1",
            model="actor" if route == ACTOR_ROUTE else "speculator",
            seed=context.seed,
            started_at=started,
            finished_at=finished,
            latency_seconds=finished - started,
            first_chunk_seconds=delay / 2,
            ttft_seconds=delay / 2,
            first_content_seconds=delay / 2,
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11,
            reasoning_tokens=0,
        )
        self.recorder.record(trace)
        return ChatResult(content=content, reasoning_content="", trace=trace)


def make_runner(runtime, mode, *, guesses=1, stop=4):
    return ChessExperimentRunner(
        runtime,
        actor_model="fake-actor",
        speculator_model="fake-speculator",
        execution_mode=mode,
        num_guesses=guesses,
        run_seed=248,
        stop_after=stop,
    )


class ConcurrencyAndHitTests(unittest.TestCase):
    def test_actor_and_speculator_really_overlap(self):
        result = make_runner(
            FakeRuntime(actor_delay=0.04, speculator_delay=0.008),
            "speculative",
            stop=3,
        ).run()
        overlaps = result.metrics["actor_speculator_overlap_intervals"]
        self.assertTrue(overlaps)
        self.assertGreater(sum(item["seconds"] for item in overlaps), 0)
        self.assertEqual(result.steps[0].hit_completion_status, "unfinished_hit")
        self.assertEqual(result.steps[1].authority_source, "speculative_cache")
        self.assertEqual(result.steps[1].speculator_attempts, 0)

    def test_hit_and_miss_are_strict(self):
        hit = make_runner(FakeRuntime(prediction="hit"), "speculative", stop=3).run()
        miss = make_runner(FakeRuntime(prediction="miss"), "speculative", stop=3).run()
        self.assertTrue(hit.steps[0].predicted_hit)
        self.assertGreaterEqual(hit.metrics["speculative"]["cache_uses"], 1)
        self.assertFalse(miss.steps[0].predicted_hit)
        self.assertEqual(miss.metrics["speculative"]["cache_uses"], 0)

    def test_failed_branch_does_not_shift_candidate_association(self):
        runtime = FakeRuntime(
            prediction="wrong_then_hit",
            fail_branch_prefix="candidate-0",
            actor_delay=0.005,
            speculator_delay=0.001,
        )
        result = make_runner(runtime, "speculative", guesses=2, stop=3).run()
        first = result.steps[0]
        self.assertEqual(first.matching_branch_id.split("-")[-1], first.authority_action[1:-1])
        self.assertEqual(result.steps[1].authority_source, "speculative_cache")
        failed = [branch for branch in first.branches if branch["error"]]
        successful = [branch for branch in first.branches if not branch["error"]]
        self.assertEqual(len(failed), 1)
        self.assertIn("candidate-0", failed[0]["branch_id"])
        self.assertTrue(any(branch["candidate_action"] == first.authority_action for branch in successful))

    def test_failed_matching_branch_falls_back_without_changing_semantics(self):
        failed = make_runner(
            FakeRuntime(
                prediction="hit",
                fail_branch_prefix="candidate-0",
                actor_delay=0.001,
                speculator_delay=0.0001,
            ),
            "speculative",
            stop=3,
        ).run()
        reference = make_runner(
            FakeRuntime(actor_delay=0.001), "sequential", stop=3
        ).run()
        self.assertEqual(failed.actions, reference.actions)
        self.assertEqual(failed.steps[1].authority_source, "fallback_after_cache_failure")
        self.assertTrue(failed.steps[1].cache_from_previous["fallback"])


class LosslessAndValidationTests(unittest.TestCase):
    def test_sequential_and_speculative_actions_are_identical(self):
        sequential = make_runner(
            FakeRuntime(actor_delay=0.001), "sequential", stop=6
        ).run()
        speculative = make_runner(
            FakeRuntime(actor_delay=0.001, speculator_delay=0.0002),
            "speculative",
            stop=6,
        ).run()
        self.assertEqual(sequential.actions, speculative.actions)
        self.assertEqual(sequential.final_fen, speculative.final_fen)
        self.assertEqual(
            sequential.action_sequence_sha256, speculative.action_sequence_sha256
        )

    def test_stale_cache_is_rejected_by_fen_and_identity(self):
        board = chess.Board()
        board.push_uci("e2e4")
        expected = CacheIdentity(
            target_fen=board.fen(),
            player="black",
            actor_model="actor",
            generation_fingerprint="generation",
            prompt_fingerprint="prompt",
        )
        future = Future()
        future.set_result(None)
        stale_identity = CacheIdentity(
            target_fen=board.fen(),
            player="black",
            actor_model="different-actor",
            generation_fingerprint="generation",
            prompt_fingerprint="prompt",
        )
        branch = BranchRecord(
            candidate_action="[e2e4]",
            source_fen=chess.STARTING_FEN,
            target_fen=board.fen(),
            cache_identity=stale_identity,
            submitted_at=0.0,
            future=future,
            branch_id="stale",
        )
        self.assertEqual(
            validate_branch_cache(branch, board, expected),
            (False, "cache_identity_mismatch"),
        )
        branch.target_fen = chess.STARTING_FEN
        self.assertEqual(
            validate_branch_cache(branch, board, expected),
            (False, "target_fen_mismatch"),
        )

    def test_illegal_actor_output_fails_after_bounded_retries(self):
        runner = make_runner(
            FakeRuntime(invalid_actor=True, actor_delay=0.0001),
            "sequential",
            stop=1,
        )
        with self.assertRaises(ActorActionError) as caught:
            runner.run()
        self.assertEqual(caught.exception.attempts, 4)
        self.assertEqual(caught.exception.invalid_attempts, 4)


class MetricsTests(unittest.TestCase):
    def test_game_and_background_drain_are_separate(self):
        result = make_runner(
            FakeRuntime(actor_delay=0.02, speculator_delay=0.002),
            "speculative",
            guesses=3,
            stop=1,
        ).run()
        self.assertGreater(result.game_wall_clock_seconds, 0)
        self.assertGreater(result.background_drain_wall_clock_seconds, 0)
        self.assertAlmostEqual(
            result.total_wall_clock_seconds,
            result.game_wall_clock_seconds + result.background_drain_wall_clock_seconds,
            places=5,
        )
        self.assertEqual(result.steps[0].hit_completion_status, "terminal_not_consumed")
        self.assertIn("definition", result.metrics["idealized_official_style_timing"])


if __name__ == "__main__":
    unittest.main()
