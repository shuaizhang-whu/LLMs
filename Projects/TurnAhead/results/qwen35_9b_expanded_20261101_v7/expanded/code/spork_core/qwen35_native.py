"""Native Qwen3.5 XML tools for the exact-prefix SPORK runtime.

All methods use the same history policy: completed reasoning is removed, visible
assistant text is retained, and parsed calls use the native structured template.
This is an explicit benchmark adapter, not preservation of a different harness
that retains historical reasoning.
"""
from __future__ import annotations

import copy
import json
import math
import re

from .qwen import QwenToolCallAdapter

FUNCTION = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>", re.S)
PARAMETER = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.S)


def visible_text(text):
    text = text or ""
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    # vLLM's Qwen parser treats the tool-start special token as an implicit
    # reasoning end. Do not put the preceding reasoning into tool history.
    if "<tool_call>" in text:
        return text[text.index("<tool_call>"):].strip()
    return text.strip()


def typed_value(text, schema):
    kind = schema.get("type")
    if not kind and schema.get("anyOf"):
        choices = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if len(choices) == 1:
            schema, kind = choices[0], choices[0].get("type")
    if isinstance(kind, list):
        if text == "null" and "null" in kind:
            return None
        kind = next((t for t in kind if t != "null"), None)
    if kind == "string":
        return text  # Numeric-looking IDs and leading zeros remain strings.
    if kind in {"array", "object", "integer", "number", "boolean", "null"}:
        value = json.loads(text)
        if kind == "array" and not isinstance(value, list):
            raise ValueError("Expected array")
        if kind == "object" and not isinstance(value, dict):
            raise ValueError("Expected object")
        if kind == "integer" and type(value) is not int:
            raise ValueError("Expected integer")
        if kind == "number" and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("Expected finite number")
        if kind == "boolean" and type(value) is not bool:
            raise ValueError("Expected boolean")
        if kind == "null" and value is not None:
            raise ValueError("Expected null")
        return value
    # No schema type: preserve text instead of guessing a potentially wrong ID.
    return text


class NativeQwen35Adapter(QwenToolCallAdapter):
    name = "qwen3.5_native_tool_call"
    history_policy = "drop_explicit_or_tool_terminated_reasoning_keep_visible_text_native_tools"
    tool_open_token_id = 248058

    def __init__(self, tokenizer, tools):
        super().__init__(tokenizer, tools)
        self.schemas = {t.get("function", t)["name"]: t.get("function", t).get("parameters", {})
                        for t in tools}
        if tokenizer.encode("<tool_call>", add_special_tokens=False) != [self.tool_open_token_id]:
            raise ValueError("Unexpected Qwen3.5 tool-call marker")
        from vllm.parser.qwen3 import Qwen3Parser
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        self._syntax_parser = Qwen3Parser(tokenizer)
        # Parser configuration only; this request is never sent to the model.
        self._syntax_request = ChatCompletionRequest(
            model="qwen3.5", messages=[{"role": "user", "content": ""}])

    def _parse(self, text):
        visible = visible_text(text)
        if visible.count("<tool_call>") > 1:
            return None
        # Use the installed upstream syntax parser, including its EOS flush and
        # implicit reasoning boundary. Schema and type checks remain local.
        parsed = self._syntax_parser.extract_tool_calls_from_content(
            visible, self._syntax_request)
        if len(parsed.tool_calls) != 1:
            return None
        function = parsed.tool_calls[0].function
        name = function.name
        if name not in self.schemas:
            return None
        schema = self.schemas[name]
        try:
            raw = json.loads(function.arguments)
            if (not isinstance(raw, dict)
                    or set(raw) - set(schema.get("properties", {}))
                    or set(schema.get("required", [])) - set(raw)):
                return None
            arguments = {key: typed_value(value if isinstance(value, str) else json.dumps(value),
                                         schema["properties"][key]) for key, value in raw.items()}
        except (ValueError, TypeError):
            return None
        return {"name": name, "arguments": arguments}

    def parse_main_tool_call(self, text):
        return self._parse(text)

    def parse_probe_tool_call(self, text):
        return self._parse("<tool_call>\n<function" + (text or ""))

    def prediction_prompt_ids(self, prompt_ids, visible_ids, text):
        # '=' merges with common function-name prefixes (e.g. '=get').
        # Forcing the standalone '=' token excludes that natural continuation.
        suffix = "\n\n<tool_call>\n<function"
        if "</think>" not in text:
            suffix = "\n</think>" + suffix
        return list(prompt_ids) + list(visible_ids) + self.tokenizer.encode(
            suffix, add_special_tokens=False)

    def speculative_tool_message(self, call):
        body = "\n<function=" + call["name"] + ">\n"
        for key, value in call.get("arguments", {}).items():
            value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            body += f"<parameter={key}>\n{value}\n</parameter>\n"
        return "<tool_call>" + body + "</function>\n"

    def draft_body(self, call):
        return self.speculative_tool_message(call)[len("<tool_call>"):]

    def extract_probe_logprobs(self, logprobs_obj):
        result = {"tokens": [], "token_logprobs": [], "top_logprobs": []}
        source = logprobs_obj or {}
        # Score the generated call header after '<function', including '=name'
        # merged tokens and the closing '>', but never argument tokens.
        for i, token in enumerate(source.get("tokens") or []):
            for key in result:
                values = source.get(key) or []
                result[key].append(values[i] if i < len(values) else None)
            if ">" in token:
                result["name_span_complete"] = True
                break
        return result

    def span_confidence(self, logprobs, *, skip_first=True):
        values = logprobs.get("token_logprobs", [])
        if not logprobs.get("name_span_complete") or not values or any(v is None for v in values):
            return None
        return math.exp(min(values))

    def render_main_prompt(self, messages, *, enable_thinking):
        normalized = copy.deepcopy(messages)
        for message in normalized:
            if message.get("role") != "assistant":
                continue
            text = visible_text(message.get("content") or "")
            call = self._parse(text)
            if call:
                match = FUNCTION.search(text)
                start = text.find("<tool_call>")
                if start < 0:
                    start = text.index("<function=")
                # An EOS-flushed header has no closing function span. Its
                # remainder belongs to the call, not to visible assistant text.
                tail = text[match.end():].strip() if match else ""
                if tail.startswith("</tool_call>"):
                    tail = tail[len("</tool_call>"):].strip()
                message["content"] = "\n".join(
                    part for part in [text[:start].strip(), tail] if part)
                message["tool_calls"] = [{"type": "function", "function": call}]
            else:
                message["content"] = text
            message["reasoning_content"] = ""
        return self.tokenizer.apply_chat_template(normalized, tools=self.tools,
            tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)

    def validate_actor_completion(self, text):
        call = self._parse(text)
        if "</think>" not in text and not ("<tool_call>" in text and call is not None):
            raise RuntimeError("Qwen3.5 has neither an explicit reasoning end nor a valid implicit tool boundary")
        if ("<tool_call>" in visible_text(text) or "<function=" in visible_text(text)) and call is None:
            raise RuntimeError("Qwen3.5 produced an unknown, incomplete-argument or ill-typed tool call")

    def token_metrics(self, text):
        ids = self.tokenizer.encode(text or "", add_special_tokens=False)
        pos = ids.index(self.tool_open_token_id) if self.tool_open_token_id in ids else len(ids)
        return {"cot_tokens": pos, "tool_call_tokens": len(ids) - pos,
                "post_tool_call_tokens": 0, "total_decode_tokens": len(ids),
                "tool_call_closed": "</tool_call>" in (text or ""),
                "tool_call_parse_ok": self.parse_main_tool_call(text) is not None}
