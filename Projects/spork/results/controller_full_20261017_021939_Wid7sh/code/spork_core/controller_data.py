"""Past-only inputs and task-group splits for the launch-timing controller."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import torch


TARGET_NAMES = ("gain_now_s", "gain_wait_s")
QUALITY_NAMES = tuple(f"{p}_same_{kind}_as_off"
                      for p in ("now", "wait") for kind in ("action", "result", "state"))
INPUT_VERSION = "task_context_observation_reasoning_v1"


def load_controller_rows(paths, quality_policy="matched"):
    """Keep originals untouched and report every omitted example.

    `matched` is a conservative data selection setting, not proof of task
    success or an unbiased causal effect. `all-valid` retains flagged outcomes.
    """
    if quality_policy not in ("matched", "all-valid"):
        raise ValueError("quality_policy must be matched or all-valid")
    kept, excluded, sources, seen = [], [], [], set()
    total = 0
    conditions = set()
    for value in paths:
        path = Path(value).resolve()
        raw = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for line_number, line in enumerate(raw.decode().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            location = f"{path}:{line_number}"
            sid = row.get("id")
            if not isinstance(sid, str) or sid in seen:
                raise ValueError(f"Missing/duplicate sample ID at {location}: {sid}")
            seen.add(sid)
            meta = row["metadata"]
            expected_group = f'{meta["domain"]}:{meta["task_id"]}'
            if row.get("group_id") != expected_group:
                raise ValueError(f"Task group disagrees with metadata: {sid}")
            # Values collected under different Actor/tool cost settings cannot
            # silently share a predictor that does not observe those settings.
            conditions.add((meta["model"], meta["measurement"], meta["tool_floor_s"]))
            if row.get("label_valid") is not True:
                excluded.append({"id": sid, "reason": "invalid_label"})
                continue
            targets = row["targets"]
            if any(not isinstance(targets.get(k), (int, float)) or isinstance(targets[k], bool)
                   or not math.isfinite(targets[k]) for k in TARGET_NAMES):
                raise ValueError(f"Non-finite or missing target: {sid}")
            inp = row["input"]
            if not isinstance(inp.get("prefix_text"), str) or not inp["prefix_text"]:
                raise ValueError(f"Missing current reasoning prefix: {sid}")
            for key in ("visible_generated_tokens", "wait_tokens"):
                if type(inp.get(key)) is not int or inp[key] <= 0:
                    raise ValueError(f"Invalid visible feature {key}: {sid}")
            flags = row["quality_flags"]
            if any(type(flags.get(k)) is not bool for k in QUALITY_NAMES):
                raise ValueError(f"Missing outcome-consistency flags: {sid}")
            if quality_policy == "matched" and not all(flags[k] for k in QUALITY_NAMES):
                excluded.append({"id": sid, "reason": "outcomes_differ",
                                 "flags": {k: flags[k] for k in QUALITY_NAMES}})
                continue
            kept.append(row)
    if len(conditions) > 1:
        raise ValueError("Mixed Actor/model, measurement or tool-floor settings; use separate datasets")
    if not kept:
        raise ValueError("No usable rows after the declared quality selection")
    return kept, {"sources": sources, "rows_read": total, "rows_kept": len(kept),
                  "quality_policy": quality_policy, "excluded": excluded,
                  "groups_kept": len({r["group_id"] for r in kept}),
                  "all_rows_pilot": all(r.get("split") == "pilot" for r in kept),
                  "label_stability_evaluated": False}


def split_by_task(rows, validation_fraction=.2, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = sorted({row["group_id"] for row in rows})
    if len(groups) < 2:
        raise ValueError("At least two task groups are needed for a disjoint split")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    validation_groups = set(groups[:count])
    train = [r for r in rows if r["group_id"] not in validation_groups]
    validation = [r for r in rows if r["group_id"] in validation_groups]
    return train, validation, {
        "seed": seed, "validation_fraction": validation_fraction,
        "train_groups": sorted({r["group_id"] for r in train}),
        "validation_groups": sorted(validation_groups),
        "train_ids": [r["id"] for r in train], "validation_ids": [r["id"] for r in validation],
    }


def partition_controller_rows(rows, validation_fraction=.2, seed=42):
    """Honor a preassigned task split; keep reserved test rows out of training.

Legacy pilot rows retain their original grouped train/validation behavior.
"""
    assigned = {r.get("split") for r in rows}
    if assigned == {"pilot"}:
        train, validation, report = split_by_task(rows, validation_fraction, seed)
        report.update(mode="pilot_group_split", test_groups=[], test_ids=[])
        return train, validation, [], report
    if not assigned or not assigned <= {"train", "validation", "test"}:
        raise ValueError("Cannot mix pilot/unassigned rows with a predefined task split")
    ownership = {}
    for row in rows:
        group, split = row["group_id"], row["split"]
        if group in ownership and ownership[group] != split:
            raise ValueError(f"Task group crosses predefined splits: {group}")
        ownership[group] = split
    partitions = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    if not partitions["train"] or not partitions["validation"]:
        raise ValueError("Both training and validation must retain usable examples")
    report = {"mode": "predefined_task_split", "test_used_for_checkpoint_selection": False}
    for split, examples in partitions.items():
        report[split + "_groups"] = sorted({r["group_id"] for r in examples})
        report[split + "_ids"] = [r["id"] for r in examples]
    return partitions["train"], partitions["validation"], partitions["test"], report


def visible_text_fields(inp):
    """Only the `input` object is accepted, never targets or outcome metadata."""
    messages = inp["messages"]
    if not isinstance(messages, list) or not all(isinstance(m.get("content", ""), str) for m in messages):
        raise ValueError("Expected text-only saved message history")
    task = next((m["content"] for m in messages if m.get("role") == "user"), "")
    policy = "\n".join(m["content"] for m in messages if m.get("role") == "system")
    recent_tool = next((m["content"] for m in reversed(messages) if m.get("role") == "tool"), "(none)")
    recent_action = next((m["content"] for m in reversed(messages) if m.get("role") == "assistant"), "(none)")
    tool_names = []
    for tool in inp.get("tools", []):
        fn = tool.get("function", tool)
        if fn.get("name"):
            tool_names.append(fn["name"])
    return {
        "task": "TASK\n" + task,
        "context": "POLICY\n" + policy + "\nAVAILABLE TOOLS\n" + ", ".join(tool_names)
                   + "\nPREVIOUS ACTION\n" + recent_action,
        "observation": "RECENT TOOL RESULT\n" + recent_tool,
        "reasoning": "CURRENT REASONING\n" + inp["prefix_text"],
    }


def _clip(ids, budget, tail_only=False):
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return list(ids)
    if tail_only:
        return list(ids[-budget:])
    front = (budget + 1) // 2
    back = budget - front
    return list(ids[:front]) + (list(ids[-back:]) if back else [])


class ControllerInputEncoder:
    """Reserve space for the current reasoning instead of truncating it away.

    Text is re-tokenized with the small encoder's tokenizer. Saved Qwen token
    IDs belong to the Actor and are never used as encoder vocabulary indices.
    """

    def __init__(self, tokenizer, max_length=512, input_ablation="none"):
        if max_length < 32:
            raise ValueError("max_length must be at least 32")
        if any(getattr(tokenizer, k) is None for k in ("cls_token_id", "sep_token_id", "pad_token_id")):
            raise ValueError("BERT-style CLS, SEP and PAD tokens are required")
        if input_ablation not in ("none", "mask_reasoning"):
            raise ValueError("Unknown controller input ablation")
        if input_ablation == "mask_reasoning" and tokenizer.mask_token_id is None:
            raise ValueError("The reasoning ablation requires a MASK token")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ablation = input_ablation

    def encode(self, inp):
        fields = visible_text_fields(inp)
        # Tokenize whole fields first, then apply the explicit per-field budget.
        tokens = {k: self.tokenizer.encode(v, add_special_tokens=False, truncation=False, verbose=False)
                  for k, v in fields.items()}
        budget = self.max_length - 5  # CLS and one SEP after each of four fields.
        used = {"reasoning": min(len(tokens["reasoning"]), budget // 3),
                "task": min(len(tokens["task"]), budget // 3),
                "observation": min(len(tokens["observation"]), budget // 4)}
        used["context"] = min(len(tokens["context"]), budget - sum(used.values()))
        remaining = budget - sum(used.values())
        for key in ("reasoning", "task", "observation", "context"):
            extra = min(remaining, len(tokens[key]) - used[key])
            used[key] += extra
            remaining -= extra
        ids = [self.tokenizer.cls_token_id]
        for key in ("task", "context", "observation", "reasoning"):
            clipped = _clip(tokens[key], used[key], tail_only=key == "reasoning")
            if key == "reasoning" and self.input_ablation == "mask_reasoning":
                # Preserve the other fields' budgets and every token position.
                # This removes current-reasoning content, but retains its length.
                clipped = [self.tokenizer.mask_token_id] * len(clipped)
            ids.extend(clipped)
            ids.append(self.tokenizer.sep_token_id)
        features = [math.log1p(inp[k]) / 10.0 for k in ("visible_generated_tokens", "wait_tokens")]
        return {"input_ids": ids, "numeric_features": features,
                "token_counts": {k: {"original": len(tokens[k]), "used": used[k]} for k in tokens}}

    def encode_rows(self, rows):
        return [{**self.encode(row["input"]), "id": row["id"], "group_id": row["group_id"],
                 "targets": [float(row["targets"][k]) for k in TARGET_NAMES]} for row in rows]

    def collate(self, examples):
        width = max(len(e["input_ids"]) for e in examples)
        ids = torch.full((len(examples), width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, ex in enumerate(examples):
            n = len(ex["input_ids"])
            ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
            mask[i, :n] = 1
        result = {"input_ids": ids, "attention_mask": mask,
                  "numeric_features": torch.tensor([e["numeric_features"] for e in examples], dtype=torch.float32)}
        if all("targets" in e for e in examples):
            result["targets"] = torch.tensor([e["targets"] for e in examples], dtype=torch.float32)
        return result
