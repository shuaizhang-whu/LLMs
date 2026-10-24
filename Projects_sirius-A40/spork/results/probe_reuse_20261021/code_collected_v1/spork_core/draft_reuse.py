"""Reuse an existing probe as token drafts, never as an accepted tool action.

Inspired by ToolSpec's retrieval drafting (arXiv:2604.13519). This is a small
SPORK-specific experiment, not a reproduction of ToolSpec's schema FSM/tree.
The target model's ordinary rejection sampler must verify every proposal.
Only the current request's probe and already emitted tokens are inspected.
"""
from __future__ import annotations

from dataclasses import dataclass


REUSE_MODES = ("single", "prefix", "suffix")


@dataclass(frozen=True)
class Proposal:
    tokens: list[int]
    source: str
    matched_tokens: int
    draft_offset: int


class ProbeDraft:
    """A causal cursor over one probe; ``single`` preserves the legacy policy."""

    def __init__(self, tokens, mode="single", min_match=4, max_match=32):
        if mode not in REUSE_MODES:
            raise ValueError(f"Unknown probe reuse mode: {mode}")
        if not 1 <= min_match <= max_match:
            raise ValueError("Require 1 <= min_match <= max_match")
        self.tokens = list(tokens)
        self.mode = mode
        self.min_match, self.max_match = min_match, max_match
        self.fired = False
        self.last_length = -1
        self.prefix_mismatch = False

    def propose(self, body, limit):
        """Return a draft or None; body contains *verified* generated tokens only."""
        if limit <= 0 or (self.mode == "single" and self.fired):
            return None
        body = list(body)
        if len(body) <= self.last_length:
            return None
        self.last_length = len(body)
        if body == self.tokens[:len(body)]:
            self.fired = True
            tail = self.tokens[len(body):len(body) + limit]
            return Proposal(tail, "prefix", len(body), len(body)) if tail else None
        self.prefix_mismatch = True
        if self.mode != "suffix":
            self.fired = True
            return None
        # Rejoin a later matching span after the target has corrected a value.
        # Ambiguous matches abstain rather than choosing a convenient occurrence.
        for n in range(min(len(body), self.max_match), self.min_match - 1, -1):
            suffix = body[-n:]
            offsets = [end for end in range(n, len(self.tokens))
                       if self.tokens[end-n:end] == suffix]
            if not offsets:
                continue
            tails = [self.tokens[end:end + limit] for end in offsets]
            if any(tail != tails[0] for tail in tails[1:]):
                return None
            return Proposal(tails[0], "suffix", n, offsets[0])
        return None
