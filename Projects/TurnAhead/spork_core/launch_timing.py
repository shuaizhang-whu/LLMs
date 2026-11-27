"""One causal first-launch decision for SPORK's existing D2/D3 runner.

Actor token IDs, not SSE chunk counts, determine the decision and wait points.
The Actor continues streaming while the CPU controller runs in a worker thread.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from .prefix_replay import formal_action_started, wait_for_tokens

MODES = {"original", "now", "wait", "off", "bert"}
CONFIG_MODES = {
    "timing_spork": "original",
    "timing_now": "now",
    "timing_wait": "wait",
    "timing_off": "off",
    "timing_bert": "bert",
    "timing_masked": "bert",
}


class LaunchController:
    """Load once on CPU; inference accepts only the frozen visible input."""

    def __init__(self, checkpoint, cpu_threads=4):
        import torch
        from .timing_controller import TimingController
        if type(cpu_threads) is not int or cpu_threads <= 0:
            raise ValueError("cpu_threads must be a positive integer")
        torch.set_num_threads(cpu_threads)
        self.path = str(Path(checkpoint).resolve())
        self.model, self.tokenizer = TimingController.load(self.path, device="cpu")
        self.weight_sha256 = hashlib.sha256((Path(self.path) / "controller.safetensors").read_bytes()).hexdigest()

    def predict(self, inp):
        return self.model.predict_inputs([inp], self.tokenizer)[0]

    def info(self):
        return {"checkpoint": self.path, "weight_sha256": self.weight_sha256,
                "device": "cpu", "input_ablation": self.model.config.input_ablation}


def visible_input(messages, tools, prefix_text, prefix_tokens, wait_tokens):
    return {"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools),
            "prefix_text": prefix_text, "visible_generated_tokens": prefix_tokens,
            "wait_tokens": wait_tokens}


def validate_launch_settings(mode, controller, prefix_tokens, wait_tokens):
    """Validate before an Actor request is started, including direct API calls."""
    if mode not in MODES:
        raise ValueError(f"Unknown first-launch mode: {mode!r}")
    if any(type(n) is not int or n <= 0 for n in (prefix_tokens, wait_tokens)):
        raise ValueError("First-launch token budgets must be positive integers")
    if mode == "bert" and controller is None:
        raise ValueError("A loaded controller is required")


async def choose_first_launch(state, main_task, tokenizer, messages, tools, *,
                              mode, controller=None, prefix_tokens=32, wait_tokens=64,
                              run_controller=None):
    """Return (first prediction text, its visible token count, trace).

    None text means no prediction this turn. Waiting is measured from the
    original decision point (32 -> 96), not from when BERT finishes computing.
    A 'now' prediction sees the frozen 32 tokens even if the Actor advanced.
    """
    validate_launch_settings(mode, controller, prefix_tokens, wait_tokens)
    trace = {"mode": mode, "policy": None, "status": "pending", "controller_inference_ms": 0.,
             "decision_prefix_tokens": prefix_tokens, "wait_tokens": wait_tokens,
             "token_count_source": "vllm_return_token_ids", "controller_calls": 0}
    if mode == "off":
        trace.update(policy="off", status="disabled")
        return None, 0, trace
    target = 1 if mode == "original" else prefix_tokens
    if not await wait_for_tokens(state, main_task, target):
        trace["status"] = "main_finished_or_formal_action_before_decision"
        return None, 0, trace
    # Original SPORK uses everything visible in the first received SSE chunk.
    # The learned policy instead requires the exact training-time 32-token cut.
    if mode == "original":
        target = len(state.token_ids)
    ids = list(state.token_ids[:target])
    text = tokenizer.decode(ids, skip_special_tokens=False)
    if mode != "original" and ("</think>" in text or formal_action_started(text)):
        trace["status"] = "reasoning_ended_before_decision"
        return None, 0, trace
    trace.update(decision_prefix_tokens=target, decision_prefix_text=text, decision_prefix_token_ids=ids,
                 available_tokens_at_decision=len(state.token_ids))
    policy = "now" if mode == "original" else mode
    if mode == "bert":
        inp = visible_input(messages, tools, text, target, wait_tokens)
        trace["input_sha256"] = hashlib.sha256(json.dumps(inp, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        start = time.perf_counter()
        trace["controller_calls"] = 1
        try:
            # The live turn supplies a cancellation-safe worker owner. Standalone
            # callers may retain the simple CPU helper for offline checks.
            worker = run_controller or asyncio.to_thread
            prediction = await worker(controller.predict, inp)
            values = [float(prediction[k]) for k in ("q_now_s", "q_wait_s")]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Non-finite controller gains")
            expected = ("off", "now", "wait")[max(range(3), key=lambda i: [0., *values][i])]
            if prediction["policy"] != expected:
                raise ValueError("Controller policy disagrees with predicted gains")
            policy = expected
            # Copy only model outputs; an arbitrary return dict cannot overwrite
            # the causal prefix, token counts, call count or status in the trace.
            trace.update(q_now_s=values[0], q_wait_s=values[1])
        except Exception as exc:
            # Finish the Actor normally, retain a visible failed-control record.
            trace.update(policy="off", status="controller_error", error=f"{type(exc).__name__}: {exc}")
            return None, 0, trace
        finally:
            trace["controller_inference_ms"] = (time.perf_counter() - start) * 1000.
    trace.update(policy=policy, available_tokens_after_decision=len(state.token_ids))
    if policy == "off":
        trace["status"] = "disabled"
        return None, 0, trace
    if main_task.done() or state.finished_at is not None or formal_action_started(state.text):
        trace["status"] = "main_finished_or_formal_action_during_decision"
        return None, 0, trace
    if policy == "wait":
        trace["wait_target_tokens"] = prefix_tokens + wait_tokens
        if not await wait_for_tokens(state, main_task, prefix_tokens + wait_tokens):
            trace["status"] = "main_finished_or_formal_action_while_waiting"
            return None, 0, trace
        text, target = state.text, len(state.token_ids)
    trace.update(status="ready", first_prediction_visible_tokens=target,
                 available_tokens_at_first_prediction=len(state.token_ids))
    return text, target, trace
