"""Native Gemma 4 tool calls for the exact-prefix SPORK runtime.

History policy, shared with the Qwen adapters: the reasoning of earlier turns
(the thought channel) is dropped, visible text is kept, and every parsed call
is rendered by the model's own chat template as a structured tool call. The
Gemma 4 model card recommends keeping thoughts in tool-call turns; like the
Qwen3 harness, this one drops them for every method, which is what makes the
next input known before a turn ends. Papers must state this deviation.

All Gemma markers are special tokens. Requests must therefore keep them in the
returned text, without spaces inserted around them (``REQUEST_FIELDS``), or stop
strings and parsing cannot work: argument strings are delimited by the special token <|"|>.
The confidence gate deliberately reuses the Qwen3 implementation (first 16
prediction tokens, first one skipped) so that both families share one D2 rule.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from .qwen import QwenToolCallAdapter

THOUGHT_OPEN = "<|channel>"
THOUGHT_CLOSE = "<channel|>"
THOUGHT_HEADER = "thought\n"
CALL_OPEN = "<|tool_call>"
CALL_CLOSE = "<tool_call|>"
CALL_PREFIX = "call:"
QUOTE = '<|"|>'
CALL_OPEN_TOKEN_ID = 48
CALL_CLOSE_TOKEN_ID = 49
REQUEST_FIELDS = {"skip_special_tokens": False, "spaces_between_special_tokens": False}

_NAME = re.compile(r"\s*call:([A-Za-z_][\w.-]*)(?=\{)")
_KEY = re.compile(r"[A-Za-z_][\w.-]*")
_SCALAR = re.compile(r"null|true|false|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


class GemmaSyntaxError(ValueError):
    """The text is not a single well-formed Gemma 4 call."""


def format_value(value: Any) -> str:
    """Serialize like the template's format_argument, keeping the given key order."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return QUOTE + value + QUOTE
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{format_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_value(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return str(value)
    raise GemmaSyntaxError(f"Unsupported argument type: {type(value).__name__}")


class _Reader:
    def __init__(self, text: str, position: int):
        self.text, self.i = text, position

    def _skip(self):
        while self.i < len(self.text) and self.text[self.i] in " \t\n":
            self.i += 1

    def _take(self, token: str) -> bool:
        self._skip()
        if self.text.startswith(token, self.i):
            self.i += len(token)
            return True
        return False

    def _string(self) -> str:
        end = self.text.find(QUOTE, self.i)
        if end < 0:
            raise GemmaSyntaxError("Unterminated string")
        value, self.i = self.text[self.i:end], end + len(QUOTE)
        return value

    def value(self) -> Any:
        self._skip()
        if self._take(QUOTE):
            return self._string()
        if self._take("{"):
            return self.mapping()
        if self._take("["):
            items = []
            if self._take("]"):
                return items
            while True:
                items.append(self.value())
                if self._take(","):
                    continue
                if self._take("]"):
                    return items
                raise GemmaSyntaxError(f"Expected ',' or ']' at {self.i}")
        match = _SCALAR.match(self.text, self.i)
        if not match:
            raise GemmaSyntaxError(f"Unexpected value at {self.i}")
        self.i = match.end()
        token = match.group()
        if token == "null":
            return None
        if token in ("true", "false"):
            return token == "true"
        return float(token) if any(c in token for c in ".eE") else int(token)

    def mapping(self) -> dict:
        """Read after an opening brace; keys may be bare or quoted."""
        result = {}
        if self._take("}"):
            return result
        while True:
            self._skip()
            if self._take(QUOTE):
                key = self._string()
            else:
                match = _KEY.match(self.text, self.i)
                if not match:
                    raise GemmaSyntaxError(f"Expected a key at {self.i}")
                key, self.i = match.group(), match.end()
            if not self._take(":"):
                raise GemmaSyntaxError(f"Expected ':' after {key!r}")
            if key in result:
                raise GemmaSyntaxError(f"Duplicate key {key!r}")
            result[key] = self.value()
            if self._take(","):
                continue
            if self._take("}"):
                return result
            raise GemmaSyntaxError(f"Expected ',' or '}}' at {self.i}")


def parse_call(body: str) -> dict | None:
    """Parse 'call:NAME{...}' with nothing but an optional closing marker after it."""
    match = _NAME.match(body or "")
    if not match:
        return None
    reader = _Reader(body, match.end() + 1)
    try:
        arguments = reader.mapping()
    except GemmaSyntaxError:
        return None
    rest = body[reader.i:].strip()
    if rest and rest != CALL_CLOSE:
        return None
    return {"name": match.group(1), "arguments": arguments}


def split_output(text: str) -> tuple[str, str]:
    """(reasoning, visible) parts of one generated turn.

    The last thought-channel close ends the reasoning. Without one, a call
    marker is an implicit end, as in the Qwen3.5 adapter. Otherwise nothing is
    visible: an unclosed thought channel is still reasoning.
    """
    text = text or ""
    if THOUGHT_CLOSE in text:
        head, _, tail = text.rpartition(THOUGHT_CLOSE)
        return head, tail
    if CALL_OPEN in text:
        k = text.index(CALL_OPEN)
        return text[:k], text[k:]
    if THOUGHT_OPEN in text:
        return text, ""
    return "", text


def visible_call(visible: str) -> tuple[str, dict | None, bool]:
    """(text before the call, parsed call, whether a call marker is present)."""
    if CALL_OPEN not in visible:
        return visible, None, False
    if visible.count(CALL_OPEN) > 1:
        return visible, None, True
    k = visible.index(CALL_OPEN)
    return visible[:k], parse_call(visible[k + len(CALL_OPEN):]), True


class Gemma4Adapter(QwenToolCallAdapter):
    name = "gemma4_native_tool_call"
    history_policy = "drop_thought_channel_keep_visible_text_native_tools"
    main_stop = (CALL_CLOSE,)
    probe_stop = (CALL_CLOSE,)
    tool_open_token_id = CALL_OPEN_TOKEN_ID
    request_fields = REQUEST_FIELDS

    def __init__(self, tokenizer: Any, tools: list[dict]):
        super().__init__(tokenizer, tools)
        for marker, expected in ((CALL_OPEN, CALL_OPEN_TOKEN_ID), (CALL_CLOSE, CALL_CLOSE_TOKEN_ID)):
            if tokenizer.encode(marker, add_special_tokens=False) != [expected]:
                raise ValueError(f"Unexpected Gemma 4 marker {marker!r}")
        self._opened_thought_ids = tokenizer.encode(THOUGHT_OPEN + THOUGHT_HEADER, add_special_tokens=False)

    # ----- parsing -------------------------------------------------------
    def parse_main_tool_call(self, text):
        _, call, _ = visible_call(split_output(text)[1])
        return call

    def parse_probe_tool_call(self, text):
        return parse_call(CALL_PREFIX + (text or ""))

    def formal_action_started(self, text):
        return CALL_OPEN in (text or "")

    def reasoning_closed(self, text):
        return THOUGHT_CLOSE in (text or "")

    def visible_answer(self, text):
        answer = split_output(text)[1].strip()
        return answer.removesuffix("<turn|>").removesuffix("<eos>").strip()

    def validate_actor_completion(self, text):
        reasoning_end = THOUGHT_CLOSE in (text or "")
        _, call, marker = visible_call(split_output(text)[1])
        if marker and call is None:
            raise RuntimeError("Gemma 4 produced an unparsable tool call")
        if not reasoning_end and call is None:
            raise RuntimeError("Gemma 4 ended without closing its thought channel or calling a tool")

    # ----- prompts --------------------------------------------------------
    def render_main_prompt(self, messages, *, enable_thinking):
        normalized = copy.deepcopy(messages)
        pending = None
        for index, message in enumerate(normalized):
            role = message.get("role")
            if role == "assistant":
                message.pop("reasoning_content", None)
                message.pop("reasoning", None)
                if message.get("tool_calls"):
                    continue
                before, call, marker = visible_call(split_output(message.get("content") or "")[1])
                if marker and call is None:
                    raise ValueError("History holds an unparsable Gemma 4 tool call")
                message["content"] = before.strip()
                if call is not None:
                    pending = (f"call_{index}", call["name"])
                    message["tool_calls"] = [{"id": pending[0], "type": "function", "function": {
                        "name": call["name"], "arguments": call["arguments"]}}]
            elif role == "tool" and pending is not None:
                message.setdefault("tool_call_id", pending[0])
                message.setdefault("name", pending[1])
                pending = None
        return self.tokenizer.apply_chat_template(normalized, tools=self.tools, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=enable_thinking)

    def prediction_prompt_ids(self, prompt_ids, visible_ids, text):
        """Close the thought channel where it stands, then force 'call:'.

        ':' is a standalone token after 'call', so the forced prefix does not
        split a token that the model would otherwise merge with the name.
        """
        text = text or ""
        force = CALL_OPEN + CALL_PREFIX
        if THOUGHT_CLOSE in text:
            suffix = force
        elif THOUGHT_OPEN in text:
            tail = text.rsplit(THOUGHT_OPEN, 1)[1]
            if THOUGHT_HEADER.startswith(tail):
                suffix = THOUGHT_HEADER[len(tail):] + THOUGHT_CLOSE + force
            else:
                suffix = ("" if text.endswith("\n") else "\n") + THOUGHT_CLOSE + force
        elif list(prompt_ids[-len(self._opened_thought_ids):]) == self._opened_thought_ids:
            suffix = ("" if not text or text.endswith("\n") else "\n") + THOUGHT_CLOSE + force
        else:
            suffix = force
        return list(prompt_ids) + list(visible_ids) + self.tokenizer.encode(suffix, add_special_tokens=False)

    # ----- the successor's anchor and D3 ------------------------------------
    def speculative_tool_message(self, call):
        return CALL_OPEN + self.draft_body(call)

    def draft_body(self, call):
        return CALL_PREFIX + call["name"] + format_value(call.get("arguments") or {})

    # ----- accounting ------------------------------------------------------
    def token_metrics(self, text):
        ids = self.tokenizer.encode(text or "", add_special_tokens=False)
        pos = ids.index(CALL_OPEN_TOKEN_ID) if CALL_OPEN_TOKEN_ID in ids else len(ids)
        return {"cot_tokens": pos, "tool_call_tokens": len(ids) - pos,
                "post_tool_call_tokens": 0, "total_decode_tokens": len(ids),
                "tool_call_closed": CALL_CLOSE in (text or ""),
                "tool_call_parse_ok": self.parse_main_tool_call(text) is not None}
