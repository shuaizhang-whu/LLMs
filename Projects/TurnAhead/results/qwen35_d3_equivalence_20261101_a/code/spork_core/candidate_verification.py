"""Experimental candidate verification at a shortened reasoning boundary.

ETA motivates when to end reasoning; SpecAgent and Speculative Uncertainty
motivate checking a supplied proposal rather than decoding it again. This
module measures exact token likelihoods, not semantic correctness. Agreement
with a *shortened* context does NOT prove agreement with the untruncated Actor.
It never executes a tool or changes the committed transcript.
"""
from __future__ import annotations

import math
import asyncio
import time
import uuid

import aiohttp


def decision_prefix(tokenizer, prompt_ids, visible_ids, visible_text):
    """Close reasoning without forcing a tool call or any particular action."""
    if "</think>" in visible_text or "<tool_call>" in visible_text:
        raise ValueError("Candidate checks require an unfinished reasoning prefix")
    return (list(prompt_ids) + list(visible_ids)
            + tokenizer.encode("\n</think>\n\n", add_special_tokens=False))


def summarize_candidate(choice, prompt_ids, candidate_start):
    """Fail closed if any candidate token lacks a correctly aligned score."""
    if not 0 < candidate_start < len(prompt_ids):
        raise ValueError("Invalid candidate boundary")
    if choice.get("prompt_token_ids") != list(prompt_ids):
        raise ValueError("Server scored a different prompt")
    rows = choice.get("prompt_logprobs")
    if rows is None or len(rows) != len(prompt_ids):
        raise ValueError("Missing or misaligned prompt logprobs")
    scores = []
    mismatches = []
    for pos in range(candidate_start, len(prompt_ids)):
        row = rows[pos]
        token = prompt_ids[pos]
        info = row.get(str(token), row.get(token)) if row else None
        if info is None or not math.isfinite(info["logprob"]):
            raise ValueError(f"Candidate token {pos} has no finite score")
        lp = float(info["logprob"])
        best = max(float(item["logprob"]) for item in row.values())
        # The response must include the best token, not only the supplied token.
        if not any(item.get("rank") == 1 for item in row.values()):
            raise ValueError("Top-1 evidence is missing")
        greedy = lp >= best - 1e-6
        scores.append(lp)
        if not greedy:
            mismatches.append({"offset": pos - candidate_start, "token_id": token,
                               "logprob": lp, "best_logprob": best,
                               "rank": info.get("rank")})
    return {"all_greedy": not mismatches, "candidate_tokens": len(scores),
            "tool_open_probability": math.exp(scores[0]),
            "min_probability": math.exp(min(scores)),
            "mean_logprob": sum(scores) / len(scores),
            "greedy_fraction": 1 - len(mismatches) / len(scores),
            "mismatches": mismatches}


async def score_candidate(session, client, prefix_ids, candidate_ids, *, seed=42):
    """One real prefill request; all request and response costs are timed.

    vLLM may recompute the full prompt when prompt_logprobs is requested. The
    caller must measure this cost; a prefix-cache hit is not assumed here.
    """
    prompt = list(prefix_ids) + list(candidate_ids)
    body = client.body(prompt, max_tokens=1, seed=seed)
    body.update(prompt_logprobs=1)
    start = time.perf_counter()
    async with session.post(client.base_url + "/completions", json=body,
                            timeout=aiohttp.ClientTimeout(total=client.timeout_s)) as response:
        if response.status != 200:
            raise RuntimeError(f"candidate scoring HTTP {response.status}: "
                               f"{(await response.text())[:1200]}")
        result = await response.json()
    score = summarize_candidate(result["choices"][0], prompt, len(prefix_ids))
    score.update(wall_s=time.perf_counter() - start,
                 usage=result.get("usage", {}), prefix_tokens=len(prefix_ids))
    return score


def tool_probability(choice, token_text="<tool_call>"):
    """The unforced first-token mass, not conditional tool-name confidence."""
    logs = choice.get("logprobs") or {}
    top = logs.get("top_logprobs") or []
    if not top:
        raise ValueError("Free probe did not return top logprobs")
    value = top[0].get(token_text)
    return 0.0 if value is None else math.exp(value)


async def decision_probe(session, client, tokenizer, prefix_ids, *, candidate=None,
                         use_d3=False, threshold=0.9, max_tokens=160, seed=42):
    """Generate an unforced decision, optionally reusing the previous call draft.

    A one-token free probe cheaply rejects no-tool branches. If a tool is likely,
    a continuation generates the actual call. An old candidate only supplies
    token drafts to vLLM's original rejection sampler; it never forces a call.
    D3 state is request-bound. The enclosing turn, not this helper, clears it.
    """
    start = time.perf_counter()
    free = await client.complete(session, prefix_ids, max_tokens=1, seed=seed, logprobs=20)
    probability = tool_probability(free)
    open_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    result = {"tool_probability": probability, "free_wall_s": free["wall_s"],
              "free_usage": free.get("usage", {}), "call_text": None,
              "d3_registered": False, "registration_attempts": []}
    if free["token_ids"] != open_ids or probability < threshold:
        return {**result, "wall_s": time.perf_counter() - start}
    request_id = "candidate-check-" + uuid.uuid4().hex
    completion = asyncio.create_task(client.complete(
        session, list(prefix_ids) + free["token_ids"], max_tokens=max_tokens,
        seed=seed, stop=("</tool_call>",), logprobs=5, request_id=request_id))
    try:
        if use_d3 and candidate is not None:
            # Register after the serving engine knows this request. The first
            # output token can already have arrived; D3 checks its whole prefix.
            deadline = time.perf_counter() + 1.0
            while not completion.done() and time.perf_counter() < deadline:
                await asyncio.sleep(0.015)
                registration = await client.set_draft(
                    session, request_id, len(prefix_ids), candidate, tokenizer)
                result["registration_attempts"].append(registration)
                if registration.get("status") == "ok":
                    result["d3_registered"] = True
                    break
                if (registration.get("error") != "main_request_not_active_or_ambiguous"
                        or registration.get("matches", 0) > 1):
                    raise RuntimeError(f"D3 candidate registration failed: {registration}")
        choice = await completion
    finally:
        if not completion.done():
            completion.cancel()
        await asyncio.gather(completion, return_exceptions=True)
    result.update(call_text="<tool_call>" + choice["text"],
                  continuation_token_ids=choice["token_ids"],
                  continuation_logprobs=choice.get("logprobs"),
                  finish_reason=choice["finish_reason"],
                  continuation_usage=choice.get("usage", {}),
                  continuation_wall_s=choice["wall_s"],
                  wall_s=time.perf_counter() - start)
    return result
