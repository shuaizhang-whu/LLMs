"""Deterministic local sequential/speculative Chess baseline runner.

The official workflows are intentionally left untouched.  This module is a
separate, auditable execution path for local OpenAI-compatible endpoints.  It
uses ``python-chess`` as the sole game-state authority and never substitutes a
speculator prediction for the authoritative actor move.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
import threading
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import chess

from local_runtime import (
    ACTOR_ROUTE,
    SPECULATOR_ROUTE,
    ChatResult,
    RequestContext,
    Trace,
    TraceRecorder,
    build_actor_messages,
    canonical_actor_prompt,
    parse_actor_action,
    parse_prediction_actions,
    stable_fingerprint,
)


ACTOR_GENERATION = {
    "enable_thinking": True,
    "max_tokens": 2048,
    "temperature": 0.0,
}
SPECULATOR_GENERATION = {
    "enable_thinking": False,
    "max_tokens": 256,
    "temperature": 0.0,
}
MAX_ACTOR_RETRIES = 3
MAX_SPECULATOR_RETRIES = 2


class Runtime(Protocol):
    recorder: TraceRecorder

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        context: RequestContext,
        *,
        enable_thinking: bool,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> ChatResult: ...


class ActorActionError(RuntimeError):
    """Raised when the actor cannot produce a legal move after bounded retries."""

    def __init__(self, message: str, *, attempts: int, invalid_attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.invalid_attempts = invalid_attempts


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    target_fen: str
    player: str
    actor_model: str
    generation_fingerprint: str
    prompt_fingerprint: str


@dataclass(frozen=True, slots=True)
class ActorOutcome:
    action: str
    cache_identity: CacheIdentity
    attempts: int
    retries: int
    invalid_attempts: int
    request_ids: tuple[str, ...]
    trace_ids: tuple[str, ...]
    result: ChatResult


@dataclass(frozen=True, slots=True)
class SpeculatorOutcome:
    predictions: tuple[str, ...]
    attempts: int
    retries: int
    request_ids: tuple[str, ...]
    trace_ids: tuple[str, ...]
    result: Optional[ChatResult]
    error: Optional[str] = None


@dataclass(slots=True)
class BranchRecord:
    """One candidate and its actor future, kept together to prevent index drift."""

    candidate_action: str
    source_fen: str
    target_fen: str
    cache_identity: CacheIdentity
    submitted_at: float
    future: Future[ActorOutcome]
    branch_id: str

    def snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_action": self.candidate_action,
            "source_fen": self.source_fen,
            "target_fen": self.target_fen,
            "cache_identity": asdict(self.cache_identity),
            "submitted_at": self.submitted_at,
            "branch_id": self.branch_id,
            "done": self.future.done(),
            "cancelled": self.future.cancelled(),
            "error": None,
            "actor_action": None,
            "actor_request_ids": [],
            "actor_trace_ids": [],
        }
        if self.future.done() and not self.future.cancelled():
            try:
                outcome = self.future.result()
            except Exception as exc:  # The error is data; it must not shift another branch.
                payload["error"] = f"{type(exc).__name__}: {exc}"
            else:
                payload["actor_action"] = outcome.action
                payload["actor_request_ids"] = list(outcome.request_ids)
                payload["actor_trace_ids"] = list(outcome.trace_ids)
        return payload


@dataclass(frozen=True, slots=True)
class SpeculationBatch:
    source_fen: str
    speculator: SpeculatorOutcome
    branches: tuple[BranchRecord, ...]
    prediction_started_at: float
    prediction_finished_at: float


@dataclass(slots=True)
class StepRecord:
    step: int
    player: str
    fen_before: str
    fen_after: str
    legal_move_count: int
    started_at: float
    finished_at: float
    wall_clock_seconds: float
    authority_action: str
    authority_source: str
    authority_wait_seconds: float
    authority_latency_seconds: float
    authority_attempts: int
    authority_retries: int
    authority_invalid_attempts: int
    authority_request_ids: list[str]
    authority_trace_ids: list[str]
    cache_from_previous: Optional[dict[str, Any]]
    prediction_wait_seconds: float = 0.0
    speculator_request_ids: list[str] = field(default_factory=list)
    speculator_trace_ids: list[str] = field(default_factory=list)
    speculator_attempts: int = 0
    speculator_retries: int = 0
    speculator_error: Optional[str] = None
    predictions: list[str] = field(default_factory=list)
    predicted_hit: bool = False
    hit_completion_status: Optional[str] = None
    extra_branch_count: int = 0
    matching_branch_id: Optional[str] = None
    branches: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RunResult:
    execution_mode: str
    final_fen: str
    actions: tuple[str, ...]
    action_sequence_sha256: str
    game_over: bool
    termination_reason: str
    game_wall_clock_seconds: float
    background_drain_wall_clock_seconds: float
    total_wall_clock_seconds: float
    steps: tuple[StepRecord, ...]
    traces: tuple[Trace, ...]
    metrics: Mapping[str, Any]


def _player(board: chess.Board) -> str:
    return "white" if board.turn == chess.WHITE else "black"


def _legal_uci(board: chess.Board) -> tuple[str, ...]:
    return tuple(sorted(move.uci() for move in board.legal_moves))


def _strip_action(action: str) -> str:
    return action[1:-1]


def build_speculator_messages(
    board: chess.Board,
    actor_model: str,
    num_guesses: int,
) -> list[dict[str, str]]:
    """Build a canonical prompt for predicting the authoritative actor action."""

    if num_guesses not in {1, 2, 3}:
        raise ValueError("num_guesses must be 1, 2, or 3")
    legal_moves = _legal_uci(board)
    actor_prompt = canonical_actor_prompt(board.fen(), legal_moves)
    system = (
        "You predict the move returned by a deterministic chess actor. "
        f"Return at most {num_guesses} distinct legal UCI moves, ordered most "
        "likely first, each enclosed in square brackets. Return no prose."
    )
    user = (
        f"Actor model: {actor_model}\n"
        f"Number of predictions: {num_guesses}\n"
        "The actor receives this canonical position prompt:\n"
        f"{actor_prompt}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_branch_cache(
    branch: BranchRecord,
    board: chess.Board,
    expected_identity: CacheIdentity,
) -> tuple[bool, str]:
    """Strictly validate branch metadata before its actor result can be reused."""

    if branch.target_fen != board.fen():
        return False, "target_fen_mismatch"
    if branch.cache_identity != expected_identity:
        return False, "cache_identity_mismatch"
    return True, "valid"


class ChessExperimentRunner:
    """Run one local Chess experiment without modifying official workflows."""

    def __init__(
        self,
        runtime: Runtime,
        *,
        actor_model: str,
        speculator_model: str,
        execution_mode: str,
        num_guesses: int,
        run_seed: int,
        stop_after: int,
        max_workers: Optional[int] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if execution_mode not in {"sequential", "speculative"}:
            raise ValueError("execution_mode must be sequential or speculative")
        if num_guesses not in {1, 2, 3}:
            raise ValueError("num_guesses must be 1, 2, or 3")
        if stop_after <= 0:
            raise ValueError("stop_after must be positive")
        if not actor_model.strip() or not speculator_model.strip():
            raise ValueError("actor_model and speculator_model must be non-empty")

        self.runtime = runtime
        self.actor_model = actor_model.strip()
        self.speculator_model = speculator_model.strip()
        self.execution_mode = execution_mode
        self.num_guesses = num_guesses
        self.run_seed = run_seed
        self.stop_after = stop_after
        self.max_workers = max_workers or max(8, num_guesses + 4)
        self.clock = clock
        self._executor: Optional[ThreadPoolExecutor] = None
        self._branch_lock = threading.Lock()
        self._all_branches: list[BranchRecord] = []
        self._batch_futures: list[Future[SpeculationBatch]] = []

    def _base_actor_messages(self, board: chess.Board) -> list[dict[str, str]]:
        return build_actor_messages(board.fen(), _legal_uci(board))

    def _cache_identity(self, board: chess.Board) -> CacheIdentity:
        messages = self._base_actor_messages(board)
        return CacheIdentity(
            target_fen=board.fen(),
            player=_player(board),
            actor_model=self.actor_model,
            generation_fingerprint=stable_fingerprint(ACTOR_GENERATION),
            prompt_fingerprint=stable_fingerprint(messages),
        )

    def _context(
        self,
        *,
        route: str,
        board: chess.Board,
        model: str,
        messages: Sequence[Mapping[str, str]],
        generation: Mapping[str, Any],
        label: str,
    ) -> RequestContext:
        return RequestContext.for_position(
            route,
            board.fen(),
            self.run_seed,
            branch_id=label,
            identity_parts=(
                _player(board),
                model,
                stable_fingerprint(messages),
                stable_fingerprint(generation),
            ),
        )

    def _actor_call(self, board: chess.Board, label: str) -> ActorOutcome:
        legal_moves = _legal_uci(board)
        base_messages = self._base_actor_messages(board)
        identity = self._cache_identity(board)
        invalid_attempts = 0
        request_ids: list[str] = []
        trace_ids: list[str] = []
        errors: list[str] = []

        for retry in range(MAX_ACTOR_RETRIES + 1):
            messages = list(base_messages)
            if retry:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Retry attempt {retry}: the previous response did not contain "
                            "exactly one usable legal "
                            "move. Return one move from the legal list in [uci] format."
                        ),
                    }
                )
            context = self._context(
                route=ACTOR_ROUTE,
                board=board,
                model=self.actor_model,
                messages=messages,
                generation=ACTOR_GENERATION,
                label=f"{label}-retry-{retry}",
            )
            request_ids.append(context.request_id)
            try:
                result = self.runtime.chat(
                    messages,
                    context,
                    enable_thinking=True,
                    max_tokens=2048,
                    temperature=0.0,
                )
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue

            trace_ids.append(result.trace.request_id)
            action = parse_actor_action(result.content, legal_moves)
            if action is None:
                invalid_attempts += 1
                errors.append("illegal_or_missing_uci")
                continue
            return ActorOutcome(
                action=action,
                cache_identity=identity,
                attempts=retry + 1,
                retries=retry,
                invalid_attempts=invalid_attempts,
                request_ids=tuple(request_ids),
                trace_ids=tuple(trace_ids),
                result=result,
            )

        summary = "; ".join(errors[-4:]) or "unknown actor failure"
        raise ActorActionError(
            f"actor failed after {MAX_ACTOR_RETRIES + 1} attempts: {summary}",
            attempts=MAX_ACTOR_RETRIES + 1,
            invalid_attempts=invalid_attempts,
        )

    def _speculator_call(self, board: chess.Board, label: str) -> SpeculatorOutcome:
        messages = build_speculator_messages(board, self.actor_model, self.num_guesses)
        request_ids: list[str] = []
        trace_ids: list[str] = []
        last_error: Optional[str] = None

        for retry in range(MAX_SPECULATOR_RETRIES + 1):
            context = self._context(
                route=SPECULATOR_ROUTE,
                board=board,
                model=self.speculator_model,
                messages=messages,
                generation=SPECULATOR_GENERATION,
                label=f"{label}-retry-{retry}",
            )
            request_ids.append(context.request_id)
            try:
                result = self.runtime.chat(
                    messages,
                    context,
                    enable_thinking=False,
                    max_tokens=256,
                    temperature=0.0,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            trace_ids.append(result.trace.request_id)
            predictions = parse_prediction_actions(
                result.content, _legal_uci(board), self.num_guesses
            )
            return SpeculatorOutcome(
                predictions=tuple(predictions),
                attempts=retry + 1,
                retries=retry,
                request_ids=tuple(request_ids),
                trace_ids=tuple(trace_ids),
                result=result,
            )

        return SpeculatorOutcome(
            predictions=(),
            attempts=MAX_SPECULATOR_RETRIES + 1,
            retries=MAX_SPECULATOR_RETRIES,
            request_ids=tuple(request_ids),
            trace_ids=tuple(trace_ids),
            result=None,
            error=last_error or "speculator failed",
        )

    def _prepare_speculation(self, board: chess.Board, step: int) -> SpeculationBatch:
        if self._executor is None:  # pragma: no cover - internal invariant
            raise RuntimeError("executor is not initialized")
        started = self.clock()
        speculator = self._speculator_call(board, f"step-{step}-speculator")
        branches: list[BranchRecord] = []
        for candidate_index, candidate in enumerate(speculator.predictions):
            move = chess.Move.from_uci(_strip_action(candidate))
            if move not in board.legal_moves:  # Defense in depth after strict parser.
                continue
            target = board.copy(stack=False)
            target.push(move)
            identity = self._cache_identity(target)
            branch_id = f"step-{step}-candidate-{candidate_index}-{_strip_action(candidate)}"
            submitted = self.clock()
            future = self._executor.submit(self._actor_call, target, branch_id)
            record = BranchRecord(
                candidate_action=candidate,
                source_fen=board.fen(),
                target_fen=target.fen(),
                cache_identity=identity,
                submitted_at=submitted,
                future=future,
                branch_id=branch_id,
            )
            branches.append(record)
            with self._branch_lock:
                self._all_branches.append(record)
        return SpeculationBatch(
            source_fen=board.fen(),
            speculator=speculator,
            branches=tuple(branches),
            prediction_started_at=started,
            prediction_finished_at=self.clock(),
        )

    def _attach_batch(
        self,
        step_record: StepRecord,
        batch: SpeculationBatch,
        *,
        terminal: bool,
    ) -> Optional[BranchRecord]:
        spec = batch.speculator
        step_record.speculator_request_ids = list(spec.request_ids)
        step_record.speculator_trace_ids = list(spec.trace_ids)
        step_record.speculator_attempts = spec.attempts
        step_record.speculator_retries = spec.retries
        step_record.speculator_error = spec.error
        step_record.predictions = list(spec.predictions)
        step_record.extra_branch_count = len(batch.branches)

        matching: Optional[BranchRecord] = None
        for branch in batch.branches:
            if (
                branch.candidate_action == step_record.authority_action
                and branch.source_fen == step_record.fen_before
                and branch.target_fen == step_record.fen_after
            ):
                matching = branch
                break
        step_record.predicted_hit = matching is not None
        if matching is not None:
            step_record.matching_branch_id = matching.branch_id
            if terminal:
                step_record.hit_completion_status = "terminal_not_consumed"
            elif matching.future.done() and matching.future.exception() is not None:
                step_record.hit_completion_status = "completed_failed_hit"
            else:
                step_record.hit_completion_status = (
                    "completed_hit" if matching.future.done() else "unfinished_hit"
                )
        step_record.branches = [branch.snapshot() for branch in batch.branches]
        return matching

    def _consume_cached_or_fresh(
        self,
        board: chess.Board,
        pending: Optional[BranchRecord],
        step: int,
    ) -> tuple[ActorOutcome, str, float, Optional[dict[str, Any]]]:
        if self._executor is None:  # pragma: no cover - internal invariant
            raise RuntimeError("executor is not initialized")
        expected = self._cache_identity(board)
        wait_started = self.clock()
        cache_info: Optional[dict[str, Any]] = None

        if pending is not None:
            valid, reason = validate_branch_cache(pending, board, expected)
            cache_info = {
                "branch_id": pending.branch_id,
                "candidate_action": pending.candidate_action,
                "validation": reason,
                "completed_at_entry": pending.future.done(),
                "fallback": False,
            }
            if valid:
                try:
                    outcome = pending.future.result()
                    if outcome.cache_identity != expected:
                        raise RuntimeError("completed branch identity mismatch")
                    if chess.Move.from_uci(_strip_action(outcome.action)) not in board.legal_moves:
                        raise RuntimeError("completed branch action is no longer legal")
                    return outcome, "speculative_cache", self.clock() - wait_started, cache_info
                except Exception as exc:
                    cache_info["fallback"] = True
                    cache_info["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            else:
                cache_info["fallback"] = True
                cache_info["fallback_reason"] = reason

        future = self._executor.submit(self._actor_call, board.copy(stack=False), f"step-{step}-authority")
        outcome = future.result()
        source = "fallback_after_cache_failure" if pending is not None else "authoritative_actor"
        return outcome, source, self.clock() - wait_started, cache_info

    def run(self) -> RunResult:
        board = chess.Board()
        actions: list[str] = []
        steps: list[StepRecord] = []
        pending: Optional[BranchRecord] = None
        trace_start_index = len(self.runtime.recorder.snapshot())
        run_started = self.clock()

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="chess-local"
        )
        terminal_batch: Optional[tuple[StepRecord, Future[SpeculationBatch]]] = None
        try:
            for step in range(self.stop_after):
                if board.is_game_over(claim_draw=True):
                    break
                step_started = self.clock()
                fen_before = board.fen()
                player = _player(board)
                legal_count = board.legal_moves.count()

                batch_future: Optional[Future[SpeculationBatch]] = None
                # 与官方 speculative pipeline 保持一致：上一窗口命中后，本步
                # 直接消费预计算 Actor 响应，不再同时开启新的投机窗口。
                # 即使该分支最终失败并回退，当前步也不追溯创建额外分支。
                if self.execution_mode == "speculative" and pending is None:
                    batch_future = self._executor.submit(
                        self._prepare_speculation, board.copy(stack=False), step
                    )
                    self._batch_futures.append(batch_future)

                outcome, source, authority_wait, cache_info = self._consume_cached_or_fresh(
                    board, pending, step
                )
                action = outcome.action
                move = chess.Move.from_uci(_strip_action(action))
                if move not in board.legal_moves:  # Never trust cached/model output blindly.
                    raise ActorActionError(
                        f"authoritative actor produced illegal action {action}",
                        attempts=outcome.attempts,
                        invalid_attempts=outcome.invalid_attempts + 1,
                    )
                board.push(move)
                actions.append(action)
                step_finished = self.clock()
                record = StepRecord(
                    step=step,
                    player=player,
                    fen_before=fen_before,
                    fen_after=board.fen(),
                    legal_move_count=legal_count,
                    started_at=step_started,
                    finished_at=step_finished,
                    wall_clock_seconds=step_finished - step_started,
                    authority_action=action,
                    authority_source=source,
                    authority_wait_seconds=authority_wait,
                    authority_latency_seconds=outcome.result.trace.latency_seconds,
                    authority_attempts=outcome.attempts,
                    authority_retries=outcome.retries,
                    authority_invalid_attempts=outcome.invalid_attempts,
                    authority_request_ids=list(outcome.request_ids),
                    authority_trace_ids=list(outcome.trace_ids),
                    cache_from_previous=cache_info,
                )
                steps.append(record)
                pending = None

                terminal = (
                    step + 1 >= self.stop_after or board.is_game_over(claim_draw=True)
                )
                if batch_future is not None:
                    if terminal:
                        terminal_batch = (record, batch_future)
                    else:
                        prediction_wait_started = self.clock()
                        batch = batch_future.result()
                        record.prediction_wait_seconds = self.clock() - prediction_wait_started
                        pending = self._attach_batch(record, batch, terminal=False)
                        record.finished_at = self.clock()
                        record.wall_clock_seconds = record.finished_at - record.started_at

            game_finished = self.clock()

            # The game critical path ends above.  Only now do we drain unused work.
            if self._batch_futures:
                wait(self._batch_futures)
            with self._branch_lock:
                branch_futures = [branch.future for branch in self._all_branches]
            if branch_futures:
                wait(branch_futures)
            drain_finished = self.clock()

            if terminal_batch is not None:
                record, future = terminal_batch
                batch = future.result()
                self._attach_batch(record, batch, terminal=True)

            # Refresh branch snapshots now that background work has drained.
            batch_by_source: dict[str, SpeculationBatch] = {}
            for future in self._batch_futures:
                batch = future.result()
                batch_by_source[batch.source_fen] = batch
            for record in steps:
                batch = batch_by_source.get(record.fen_before)
                if batch is not None:
                    record.branches = [branch.snapshot() for branch in batch.branches]

            traces = self.runtime.recorder.snapshot()[trace_start_index:]
            termination = "game_over" if board.is_game_over(claim_draw=True) else "stop_after"
            metrics = self._build_metrics(
                steps=steps,
                traces=traces,
                actions=actions,
                final_fen=board.fen(),
                run_started=run_started,
                game_finished=game_finished,
                drain_finished=drain_finished,
                termination=termination,
            )
            action_hash = metrics["action_sequence_sha256"]
            return RunResult(
                execution_mode=self.execution_mode,
                final_fen=board.fen(),
                actions=tuple(actions),
                action_sequence_sha256=action_hash,
                game_over=board.is_game_over(claim_draw=True),
                termination_reason=termination,
                game_wall_clock_seconds=game_finished - run_started,
                background_drain_wall_clock_seconds=drain_finished - game_finished,
                total_wall_clock_seconds=drain_finished - run_started,
                steps=tuple(steps),
                traces=tuple(traces),
                metrics=metrics,
            )
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    def _build_metrics(
        self,
        *,
        steps: Sequence[StepRecord],
        traces: Sequence[Trace],
        actions: Sequence[str],
        final_fen: str,
        run_started: float,
        game_finished: float,
        drain_finished: float,
        termination: str,
    ) -> dict[str, Any]:
        hits = sum(step.predicted_hit for step in steps)
        opportunities = sum(step.speculator_attempts > 0 for step in steps)
        completed = sum(step.hit_completion_status == "completed_hit" for step in steps)
        unfinished = sum(step.hit_completion_status == "unfinished_hit" for step in steps)
        failed_hits = sum(
            step.hit_completion_status == "completed_failed_hit" for step in steps
        )
        cache_uses = sum(step.authority_source == "speculative_cache" for step in steps)
        illegal = sum(step.authority_invalid_attempts for step in steps)
        retries = sum(step.authority_retries + step.speculator_retries for step in steps)
        extra_branches = sum(step.extra_branch_count for step in steps)

        tokens: dict[str, dict[str, int]] = {
            ACTOR_ROUTE: {"calls": 0, "prompt": 0, "completion": 0, "total": 0},
            SPECULATOR_ROUTE: {"calls": 0, "prompt": 0, "completion": 0, "total": 0},
        }
        for trace in traces:
            bucket = tokens[trace.route]
            bucket["calls"] += 1
            bucket["prompt"] += trace.prompt_tokens or 0
            bucket["completion"] += trace.completion_tokens or 0
            bucket["total"] += trace.total_tokens or 0

        overlaps: list[dict[str, Any]] = []
        actor_traces = [trace for trace in traces if trace.route == ACTOR_ROUTE]
        spec_traces = [trace for trace in traces if trace.route == SPECULATOR_ROUTE]
        for actor in actor_traces:
            for spec in spec_traces:
                start = max(actor.started_at, spec.started_at)
                end = min(actor.finished_at, spec.finished_at)
                if end > start:
                    overlaps.append(
                        {
                            "actor_request_id": actor.request_id,
                            "speculator_request_id": spec.request_id,
                            "started_at": start,
                            "finished_at": end,
                            "seconds": end - start,
                        }
                    )

        sequential_estimate = sum(step.authority_latency_seconds for step in steps)
        actor_only_critical = sum(step.authority_wait_seconds for step in steps)
        estimated_saved = sequential_estimate - actor_only_critical
        estimated_pct = (
            100.0 * estimated_saved / sequential_estimate if sequential_estimate else 0.0
        )
        action_payload = json.dumps(list(actions), separators=(",", ":")).encode("utf-8")
        return {
            "execution_mode": self.execution_mode,
            "steps_completed": len(steps),
            "termination_reason": termination,
            "final_fen": final_fen,
            "actions": list(actions),
            "action_sequence_sha256": sha256(action_payload).hexdigest(),
            "game_wall_clock_seconds": game_finished - run_started,
            "background_drain_wall_clock_seconds": drain_finished - game_finished,
            "total_wall_clock_seconds": drain_finished - run_started,
            "speculative": {
                "opportunities_with_predictions": opportunities,
                "hits": hits,
                "hit_rate": hits / opportunities if opportunities else 0.0,
                "completed_hits": completed,
                "unfinished_hits": unfinished,
                "completed_failed_hits": failed_hits,
                "cache_uses": cache_uses,
                "extra_branches": extra_branches,
            },
            "requests_and_tokens": tokens,
            "retries": retries,
            "illegal_actor_responses": illegal,
            "actor_speculator_overlap_intervals": overlaps,
            "idealized_official_style_timing": {
                "sequential_actor_sum_seconds": sequential_estimate,
                "speculative_actor_only_critical_path_seconds": actor_only_critical,
                "estimated_time_saved_seconds": estimated_saved,
                "estimated_time_saved_percent": estimated_pct,
                "definition": (
                    "Actor-only idealization: sum consumed actor request latencies versus "
                    "the waits observed for fresh or matching cached actor futures. It "
                    "excludes speculator waits, scheduling overhead, and background drain; "
                    "use game_wall_clock_seconds for real end-to-end latency."
                ),
            },
        }


def trace_to_dict(trace: Trace) -> dict[str, Any]:
    payload = asdict(trace)
    if trace.ttft_seconds is None:
        payload["generation_seconds"] = None
    else:
        payload["generation_seconds"] = max(0.0, trace.latency_seconds - trace.ttft_seconds)
    # OpenAI-compatible streaming does not expose server queue residence time.
    payload["queue_seconds"] = None
    payload["queue_measurement_available"] = False
    return payload


def sanitize_json(value: Any) -> Any:
    """Recursively replace non-finite floats before strict JSON serialization."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    return value


__all__ = [
    "ACTOR_GENERATION",
    "SPECULATOR_GENERATION",
    "ActorActionError",
    "ActorOutcome",
    "BranchRecord",
    "CacheIdentity",
    "ChessExperimentRunner",
    "RunResult",
    "StepRecord",
    "build_speculator_messages",
    "sanitize_json",
    "trace_to_dict",
    "validate_branch_cache",
]
