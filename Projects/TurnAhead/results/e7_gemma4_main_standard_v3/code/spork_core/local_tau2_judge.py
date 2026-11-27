"""Explicit local NL judge for native tau2 ALL scoring, outside latency timing.

This preserves the installed evaluator's prompts and reward combination, but
substitutes a declared local model. It is not the published benchmark judge.
Empty, partial, malformed or truncated judgment is an error, never a pass.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from .evaluation_artifacts import digest, write_json
from tau2.data_model.message import AssistantMessage
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.utils.llm_utils import to_litellm_messages


def validate_judgment(text, expected):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    result = json.loads(text)
    rows = result.get("results") if isinstance(result, dict) else None
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(r, dict) or not isinstance(r.get("expectedOutcome"), str)
                   or type(r.get("metExpectation")) is not bool
                   or not isinstance(r.get("reasoning"), str) for r in rows)
            or Counter(r["expectedOutcome"] for r in rows) != Counter(expected)):
        raise ValueError("Local judge did not score every expected assertion exactly once")
    return json.dumps(result, ensure_ascii=False)


class LocalNLJudge:
    def __init__(self, runtime, tokenizer, cache_dir, *, seed=42, max_tokens=4096):
        self.runtime, self.tokenizer = runtime, tokenizer
        self.cache_dir = Path(cache_dir)
        self.seed, self.max_tokens = seed, max_tokens
        self.calls = []

    def evaluate(self, simulation, task, domain):
        expected = task.evaluation_criteria.nl_assertions if task.evaluation_criteria else []

        def local_generate(*, messages, call_name, **_native_defaults):
            if call_name != "nl_assertions_eval":
                raise ValueError("Unexpected evaluator request")
            prompt = self.tokenizer.apply_chat_template(to_litellm_messages(messages),
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) + self.max_tokens > 40960:
                raise ValueError("Judge context budget exceeded")
            http = self.runtime.judge_http
            body = http.body(ids, max_tokens=self.max_tokens, seed=self.seed,
                             stop=["<|im_end|>"])
            payload = {"request": body, "expected_assertions": expected}
            key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            path = self.cache_dir / (key + ".json")
            seal = path.with_suffix(".json.seal")
            cached = path.exists() or seal.exists()
            if cached:
                if not path.is_file() or not seal.is_file() or digest(path) != json.loads(seal.read_text())["sha256"]:
                    raise ValueError("Judge cache is incomplete or modified")
                saved = json.loads(path.read_text())
                if saved["payload"] != payload:
                    raise ValueError("Judge cache input changed")
                choice = saved["choice"]
            else:
                choice = self.runtime.run(http.complete(self.runtime.session, ids,
                    max_tokens=self.max_tokens, seed=self.seed, stop=["<|im_end|>"]))
            self.calls.append({"cached": cached, "key": key, "choice": choice,
                               "prompt_token_ids": ids, "generation_settings": body})
            if choice["finish_reason"] != "stop":
                raise ValueError("Local judge generation truncated")
            content = validate_judgment(choice["text"], expected)
            if not cached:
                write_json(path, {"payload": payload, "choice": choice})
                write_json(seal, {"sha256": digest(path)})
            return AssistantMessage(role="assistant", content=content)

        with patch("tau2.evaluator.evaluator_nl_assertions.generate", local_generate):
            return evaluate_simulation(simulation, task, EvaluationType.ALL,
                solo_mode=False, domain=domain, strict_replay=True)
