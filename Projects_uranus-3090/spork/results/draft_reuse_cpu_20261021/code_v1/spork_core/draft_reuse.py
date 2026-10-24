"""Opt-in, request-local D3 reuse; all proposals still use vLLM verification.

The suffix lookup is inspired by ToolSpec (arXiv:2604.13519), not a new
verification rule. `rolling` and `bank_prefix` separate ordinary repeated
reuse and retention from suffix realignment. No action is committed here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .vllm029_d3 import D3Bridge, SporkWorkerExtension as OriginalWorkerExtension

MODES = ("single", "rolling", "bank_prefix", "bank_suffix")


@dataclass(frozen=True)
class Proposal:
    tokens: tuple[int, ...] = ()
    kind: str = "none"
    matched: int = 0


class DraftBank:
    """Only register drafts already available; `body` contains visible tokens."""

    def __init__(self, mode="bank_suffix", capacity=5, min_suffix=5, max_suffix=16):
        if mode not in MODES or capacity < 1 or not 1 <= min_suffix <= max_suffix:
            raise ValueError("Invalid draft reuse configuration")
        self.mode, self.capacity = mode, capacity
        self.min_suffix, self.max_suffix = min_suffix, max_suffix
        self.drafts = []
        self.fired = False

    def add(self, tokens):
        draft = tuple(tokens)
        if not draft or len(draft) > 2048 or any(type(t) is not int or t < 0 for t in draft):
            raise ValueError("Draft must contain 1..2048 nonnegative token IDs")
        self.drafts = [d for d in self.drafts if d != draft]
        self.drafts.append(draft)
        self.drafts = self.drafts[-self.capacity:]
        self.fired = False

    def propose(self, body, budget):
        body = tuple(body)
        if budget <= 0 or not self.drafts or 151658 in body:
            return Proposal()
        if self.mode == "single":
            if self.fired:
                return Proposal()
            self.fired = True
        candidates = self.drafts[-1:] if self.mode in ("single", "rolling") else self.drafts
        for draft in reversed(candidates):
            if len(body) < len(draft) and draft[:len(body)] == body:
                return Proposal(draft[len(body):len(body) + budget], "prefix", len(body))
        if self.mode != "bank_suffix":
            return Proposal()
        # Prefer longest alignment, then newest draft, then rightmost match.
        # Never inspect the target continuation when selecting a proposal.
        for width in range(min(len(body), self.max_suffix), self.min_suffix - 1, -1):
            suffix = body[-width:]
            for draft in reversed(candidates):
                for end in range(len(draft) - 1, width - 1, -1):
                    if draft[end - width:end] == suffix:
                        return Proposal(draft[end:end + budget], "suffix", width)
        return Proposal()


class ReuseBridge(D3Bridge):
    """Reuse the baseline binding, sampler and accepted-token accounting."""

    def __init__(self, runner, mode="bank_suffix"):
        if mode not in MODES:
            raise ValueError(f"Unsupported SPORK_DRAFT_REUSE_MODE: {mode}")
        self.mode = mode
        self.banks = {}
        super().__init__(runner)
        self.counts.update(prefix_reuse_injections=0, suffix_reuse_injections=0)

    def register(self, request_id, draft_tokens, prompt_len):
        # Validate before modifying the inherited state.
        checked = DraftBank(self.mode)
        checked.add(draft_tokens)
        result = super().register(request_id, draft_tokens, prompt_len)
        if result.get("status") == "ok":
            rid = result["engine_request_id"]
            self.banks.setdefault(rid, DraftBank(self.mode)).add(draft_tokens)
        return result

    def clear(self):
        super().clear()
        self.banks.clear()

    def snapshot(self):
        return {**super().snapshot(), "draft_reuse_mode": self.mode,
                "retained_drafts": sum(len(b.drafts) for b in self.banks.values())}

    def propose(self, num_speculative_tokens, sampled_token_ids, num_tokens_no_spec,
                token_ids_cpu, slot_mappings=None):
        if self.mode == "single":
            return super().propose(num_speculative_tokens, sampled_token_ids,
                                   num_tokens_no_spec, token_ids_cpu, slot_mappings)
        drafts = self.original_propose(num_speculative_tokens, sampled_token_ids,
                                       num_tokens_no_spec, token_ids_cpu,
                                       slot_mappings=slot_mappings)
        if num_speculative_tokens <= 0:
            return drafts
        for i, rid in enumerate(self.runner.input_batch.req_ids):
            state, bank = self.states.get(rid), self.banks.get(rid)
            if not state or bank is None or not sampled_token_ids[i] or rid in self.pending:
                continue
            end = int(num_tokens_no_spec[i])
            seq = token_ids_cpu[i, state["prompt_len"]:end].tolist()
            positions = [j for j, token in enumerate(seq) if token == 151657]
            if not positions:
                continue
            body = seq[positions[-1] + 1:]
            budget = min(num_speculative_tokens, self.drafter.k, self.drafter.max_model_len - end)
            proposal = bank.propose(body, budget)
            if not proposal.tokens:
                continue
            drafts[i] = list(proposal.tokens)
            self.pending[rid] = list(proposal.tokens)
            self.counts["inject_count"] += 1
            self.counts["draft_token_count"] += len(proposal.tokens)
            self.counts[proposal.kind + "_reuse_injections"] += 1
            self.last_inject_info = {"engine_request_id": rid, "batch_row": i,
                                     "draft_tokens": len(proposal.tokens),
                                     "body_tokens_already_generated": len(body),
                                     "alignment": proposal.kind, "matched": proposal.matched}
        return drafts


class SporkWorkerExtension(OriginalWorkerExtension):
    """Select with --worker-extension-cls spork_core.draft_reuse.SporkWorkerExtension."""

    def spork_set_draft_tokens(self, request_id, draft_tokens, prompt_len=0):
        runner = self.model_runner
        if getattr(runner, "drafter", None) is None:
            return {"status": "inactive", "has_drafter": False}
        if not hasattr(runner, "_spork_d3_bridge"):
            runner._spork_d3_bridge = ReuseBridge(
                runner, os.environ.get("SPORK_DRAFT_REUSE_MODE", "bank_suffix"))
        return runner._spork_d3_bridge.register(request_id, draft_tokens, prompt_len)
