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
        self.suffix_index = {}
        if mode == "suffix":
            for end in range(min_match, len(self.tokens)):
                key = tuple(self.tokens[end-min_match:end])
                self.suffix_index.setdefault(key, []).append(end)

    def propose(self, body, limit):
        """Return a draft or None; body contains *verified* generated tokens only."""
        if limit <= 0 or (self.mode == "single" and self.fired):
            return None
        if self.mode == "prefix" and self.prefix_mismatch:
            return None  # An appended continuation cannot repair a past mismatch.
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
        if len(body) < self.min_match:
            return None
        candidates = self.suffix_index.get(tuple(body[-self.min_match:]), ())
        longest, offsets = 0, []
        for end in candidates:
            matched = self.min_match
            while (matched < min(len(body), end, self.max_match)
                   and body[-matched-1] == self.tokens[end-matched-1]):
                matched += 1
            if matched > longest:
                longest, offsets = matched, [end]
            elif matched == longest:
                offsets.append(end)
        if not offsets:
            return None
        tails = [self.tokens[end:end + limit] for end in offsets]
        if any(tail != tails[0] for tail in tails[1:]):
            return None
        return Proposal(tails[0], "suffix", longest, offsets[0])
