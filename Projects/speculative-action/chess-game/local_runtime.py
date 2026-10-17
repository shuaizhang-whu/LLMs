"""Local OpenAI-compatible runtime primitives for the Chess experiments.

This module deliberately contains no workflow or process-management code.  It
provides deterministic prompt/seed helpers, one shared UCI parser, request
tracing, and a small streaming client that can be composed by the regular and
speculative runners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
import re
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


LOCAL_DUMMY_API_KEY = "local-openai-compatible"
ACTOR_ROUTE = "actor"
SPECULATOR_ROUTE = "speculator"
VALID_ROUTES = frozenset({ACTOR_ROUTE, SPECULATOR_ROUTE})

DEFAULT_ACTOR_SYSTEM_PROMPT = (
    "You are a competitive chess player. Choose exactly one legal move and "
    "return it in square-bracketed UCI format, for example [e2e4]."
)

_UCI_TOKEN = r"[a-h][1-8][a-h][1-8][qrbn]?"
_BRACKETED_UCI_RE = re.compile(rf"\[\s*({_UCI_TOKEN})\s*\]", re.IGNORECASE)
_BARE_UCI_RE = re.compile(rf"({_UCI_TOKEN})", re.IGNORECASE)


def _jsonable(value: Any) -> Any:
    """Convert common values to a canonical JSON-compatible representation."""

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_jsonable(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported value for a stable fingerprint: {type(value).__name__}")


def stable_fingerprint(*parts: Any) -> str:
    """Return a process-independent SHA-256 fingerprint for ``parts``."""

    payload = json.dumps(
        _jsonable(parts), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def stable_seed(*parts: Any) -> int:
    """Return a deterministic non-negative 31-bit seed derived with SHA-256."""

    digest = bytes.fromhex(stable_fingerprint(*parts))
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def canonicalize_fen(fen: str) -> str:
    """Normalize FEN whitespace and reject malformed six-field FEN strings."""

    fields = fen.strip().split()
    if len(fields) != 6:
        raise ValueError("FEN must contain exactly six whitespace-separated fields")
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError("FEN board placement must contain eight ranks")
    for rank in ranks:
        width = 0
        for symbol in rank:
            if symbol.isdigit():
                width += int(symbol)
            elif symbol in "prnbqkPRNBQK":
                width += 1
            else:
                raise ValueError(f"Invalid FEN board symbol: {symbol!r}")
        if width != 8:
            raise ValueError("Every FEN rank must describe exactly eight squares")
    if fields[1] not in {"w", "b"}:
        raise ValueError("FEN active color must be 'w' or 'b'")
    return " ".join(fields)


def _normalize_uci(value: Any) -> Optional[str]:
    text = str(value).strip()
    match = _BARE_UCI_RE.fullmatch(text) or _BRACKETED_UCI_RE.fullmatch(text)
    return match.group(1).lower() if match else None


def _canonical_legal_moves(legal_moves: Iterable[Any]) -> tuple[str, ...]:
    moves = {_normalize_uci(move) for move in legal_moves}
    moves.discard(None)
    return tuple(sorted(moves))  # type: ignore[arg-type]


def extract_uci_moves(text: Optional[str]) -> list[str]:
    """Extract square-bracketed UCI moves in response order, normalized to lower case."""

    if not text:
        return []
    return [match.lower() for match in _BRACKETED_UCI_RE.findall(text)]


def parse_actor_action(text: Optional[str], legal_moves: Iterable[Any]) -> Optional[str]:
    """Return the *last* emitted legal move, in the workflow's ``[uci]`` format."""

    legal = set(_canonical_legal_moves(legal_moves))
    selected: Optional[str] = None
    for move in extract_uci_moves(text):
        if move in legal:
            selected = move
    return f"[{selected}]" if selected is not None else None


def parse_prediction_actions(
    text: Optional[str], legal_moves: Iterable[Any], top_k: int
) -> list[str]:
    """Return legal, stable-deduplicated predictions, strictly truncated to ``top_k``.

    Missing or invalid candidates are never fabricated to fill the requested K.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    legal = set(_canonical_legal_moves(legal_moves))
    selected: list[str] = []
    seen: set[str] = set()
    for move in extract_uci_moves(text):
        if move not in legal or move in seen:
            continue
        selected.append(f"[{move}]")
        seen.add(move)
        if len(selected) == top_k:
            break
    return selected


def _render_fen_board(fen: str) -> str:
    rows: list[str] = []
    for rank in fen.split()[0].split("/"):
        squares: list[str] = []
        for symbol in rank:
            if symbol.isdigit():
                squares.extend("." for _ in range(int(symbol)))
            else:
                squares.append(symbol)
        rows.append(" ".join(squares))

    inner_width = len(rows[0])
    border = f"   +{'-' * (inner_width + 2)}+"
    body = [f" {rank} | {row} |" for rank, row in zip(range(8, 0, -1), rows)]
    files = "   " + " ".join("abcdefgh").center(inner_width + 2)
    return "\n".join([border, *body, border, files])


def canonical_actor_prompt(fen: str, legal_moves: Iterable[Any]) -> str:
    """Build a deterministic actor prompt from a position.

    Legal moves are normalized, deduplicated, and sorted, so callers using the
    same FEN cannot accidentally change the prompt through iterator order.
    """

    normalized_fen = canonicalize_fen(fen)
    role = "White" if normalized_fen.split()[1] == "w" else "Black"
    moves = ", ".join(f"[{move}]" for move in _canonical_legal_moves(legal_moves))
    return (
        f"[GAME] You are playing as {role} in a game of Chess. Make your move in "
        "UCI format enclosed in square brackets (e.g., [e2e4]).\n"
        f"[GAME] FEN: {normalized_fen}\n"
        f"[GAME] The current board is:\n{_render_fen_board(normalized_fen)}\n"
        f"[GAME] The valid moves are: {moves}."
    )


def build_actor_messages(
    fen: str,
    legal_moves: Iterable[Any],
    system_prompt: str = DEFAULT_ACTOR_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Return canonical OpenAI chat messages for an authoritative actor call."""

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": canonical_actor_prompt(fen, legal_moves)},
    ]


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    """An explicit OpenAI-compatible endpoint; no environment lookup is performed."""

    base_url: str
    model: str
    api_key: str = LOCAL_DUMMY_API_KEY
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        base_url = self.base_url.strip().rstrip("/")
        model = self.model.strip()
        if not base_url:
            raise ValueError("base_url must be explicit and non-empty")
        if not model:
            raise ValueError("model must be explicit and non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "api_key", self.api_key or LOCAL_DUMMY_API_KEY)


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Stable identity and seed for one actor or speculator request."""

    request_id: str
    route: str
    fen: str
    seed: int
    fingerprint: str
    branch_id: str = "authority"

    def __post_init__(self) -> None:
        if self.route not in VALID_ROUTES:
            raise ValueError(f"route must be one of {sorted(VALID_ROUTES)}")
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "fen", canonicalize_fen(self.fen))

    @classmethod
    def for_position(
        cls,
        route: str,
        fen: str,
        run_seed: int,
        branch_id: str = "authority",
        *,
        identity_parts: Sequence[Any] = (),
    ) -> "RequestContext":
        """Build a deterministic request context for one position.

        ``branch_id`` is deliberately *not* part of the generation identity.
        It is only a trace label.  Consequently an authoritative call and a
        speculative branch with the same FEN/model/prompt/generation settings
        receive the same seed.  Callers should pass those settings through
        ``identity_parts``; this keeps lossless sequential/speculative replay
        independent of scheduling details such as mode, step, or branch index.
        """

        normalized_fen = canonicalize_fen(fen)
        identity = (route, normalized_fen, run_seed, tuple(identity_parts))
        fingerprint = stable_fingerprint("chess-request-v2", identity)
        trace_suffix = stable_fingerprint("trace-label", branch_id)[:8]
        return cls(
            request_id=f"{route}-{fingerprint[:16]}-{trace_suffix}",
            route=route,
            fen=normalized_fen,
            seed=stable_seed("chess-request-v2", identity),
            fingerprint=fingerprint,
            branch_id=branch_id,
        )


@dataclass(frozen=True, slots=True)
class Trace:
    request_id: str
    route: str
    fingerprint: str
    base_url: str
    model: str
    seed: int
    started_at: float
    finished_at: float
    latency_seconds: float
    first_chunk_seconds: Optional[float]
    ttft_seconds: Optional[float]
    first_content_seconds: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    reasoning_content: str
    trace: Trace


class TraceRecorder:
    """A small thread-safe in-memory trace sink."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._traces: list[Trace] = []

    def record(self, trace: Trace) -> None:
        if not isinstance(trace, Trace):
            raise TypeError("trace must be a Trace instance")
        with self._lock:
            self._traces.append(trace)

    def snapshot(self) -> tuple[Trace, ...]:
        with self._lock:
            return tuple(self._traces)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._traces)


def _member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_delta(chunk: Any) -> Any:
    choices = _member(chunk, "choices", ()) or ()
    if not choices:
        return None
    return _member(choices[0], "delta")


def _usage_from_chunk(chunk: Any) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    usage = _member(chunk, "usage")
    if usage is None:
        return None, None, None, None
    prompt = _member(usage, "prompt_tokens")
    completion = _member(usage, "completion_tokens")
    total = _member(usage, "total_tokens")
    details = _member(usage, "completion_tokens_details")
    reasoning = _member(details, "reasoning_tokens") if details is not None else None
    return prompt, completion, total, reasoning


class OpenAICompatibleChatClient:
    """Streaming client for one explicit local OpenAI-compatible endpoint."""

    def __init__(
        self,
        endpoint: EndpointConfig,
        recorder: Optional[TraceRecorder] = None,
        *,
        client: Any = None,
        client_factory: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if client is not None and client_factory is not None:
            raise ValueError("Pass either client or client_factory, not both")
        self.endpoint = endpoint
        self.recorder = recorder if recorder is not None else TraceRecorder()
        self._clock = clock

        if client is not None:
            self._client = client
        else:
            if client_factory is None:
                from openai import OpenAI  # Lazy import keeps pure helpers dependency-light.

                client_factory = OpenAI
            self._client = client_factory(
                base_url=endpoint.base_url,
                api_key=endpoint.api_key,
                timeout=endpoint.timeout_seconds,
            )

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        context: RequestContext,
        *,
        enable_thinking: bool,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> ChatResult:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not messages:
            raise ValueError("messages must not be empty")

        request = {
            "model": self.endpoint.model,
            "messages": [dict(message) for message in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": context.seed,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            },
        }

        started_at = self._clock()
        first_chunk_at: Optional[float] = None
        first_token_at: Optional[float] = None
        first_content_at: Optional[float] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        reasoning_tokens: Optional[int] = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        stream: Any = None
        error: Optional[str] = None

        try:
            stream = self._client.chat.completions.create(**request)
            for chunk in stream:
                now = self._clock()
                if first_chunk_at is None:
                    first_chunk_at = now

                delta = _first_delta(chunk)
                content = _member(delta, "content") if delta is not None else None
                reasoning = _member(delta, "reasoning_content") if delta is not None else None

                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)
                    if first_token_at is None:
                        first_token_at = now
                if isinstance(content, str) and content:
                    content_parts.append(content)
                    if first_token_at is None:
                        first_token_at = now
                    if first_content_at is None:
                        first_content_at = now

                usage = _usage_from_chunk(chunk)
                if any(value is not None for value in usage):
                    prompt_tokens, completion_tokens, total_tokens, reasoning_tokens = usage
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            finished_at = self._clock()
            trace = Trace(
                request_id=context.request_id,
                route=context.route,
                fingerprint=context.fingerprint,
                base_url=self.endpoint.base_url,
                model=self.endpoint.model,
                seed=context.seed,
                started_at=started_at,
                finished_at=finished_at,
                latency_seconds=finished_at - started_at,
                first_chunk_seconds=(
                    first_chunk_at - started_at if first_chunk_at is not None else None
                ),
                ttft_seconds=(
                    first_token_at - started_at if first_token_at is not None else None
                ),
                first_content_seconds=(
                    first_content_at - started_at if first_content_at is not None else None
                ),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
                error=error,
            )
            self.recorder.record(trace)

        return ChatResult(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            trace=trace,
        )


class ChessLocalRuntime:
    """Route actor/speculator calls to their separately configured endpoints."""

    def __init__(
        self,
        actor_endpoint: EndpointConfig,
        speculator_endpoint: EndpointConfig,
        recorder: Optional[TraceRecorder] = None,
        *,
        client_factory: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.recorder = recorder if recorder is not None else TraceRecorder()
        self._clients = {
            ACTOR_ROUTE: OpenAICompatibleChatClient(
                actor_endpoint,
                self.recorder,
                client_factory=client_factory,
                clock=clock,
            ),
            SPECULATOR_ROUTE: OpenAICompatibleChatClient(
                speculator_endpoint,
                self.recorder,
                client_factory=client_factory,
                clock=clock,
            ),
        }

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        context: RequestContext,
        *,
        enable_thinking: bool,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> ChatResult:
        return self._clients[context.route].chat(
            messages,
            context,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
        )


__all__ = [
    "ACTOR_ROUTE",
    "SPECULATOR_ROUTE",
    "LOCAL_DUMMY_API_KEY",
    "EndpointConfig",
    "RequestContext",
    "Trace",
    "ChatResult",
    "TraceRecorder",
    "OpenAICompatibleChatClient",
    "ChessLocalRuntime",
    "stable_fingerprint",
    "stable_seed",
    "canonicalize_fen",
    "canonical_actor_prompt",
    "build_actor_messages",
    "extract_uci_moves",
    "parse_actor_action",
    "parse_prediction_actions",
]
