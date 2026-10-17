#!/usr/bin/env python3
"""Train a local launch-timing encoder, or run a one-batch CPU functional check.

cpu-check uses a tiny RANDOM encoder and is not a research training result.
train loads a local pretrained BERT/DistilBERT encoder; it never calls vLLM,
downloads weights, changes the Actor, or launches a model server.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spork_core.controller_data import (  # noqa: E402
    ControllerInputEncoder, INPUT_VERSION, load_controller_rows, split_by_task, visible_text_fields,
)
from spork_core.timing_controller import ControllerConfig, TimingController, POLICIES  # noqa: E402


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def tiny_random_controller(training_rows, config):
    """Build a <0.2M parameter model locally, without any pretrained downloads."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.normalizers import Lowercase
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import AutoModel, DistilBertConfig, PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel(unk_token="[UNK]"))
    backend.normalizer = Lowercase()
    backend.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(vocab_size=2048, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    # Even this test-only vocabulary uses training groups exclusively.
    backend.train_from_iterator((text for row in training_rows
                                 for text in visible_text_fields(row["input"]).values()), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]", model_max_length=config.max_length)
    enc_config = DistilBertConfig(vocab_size=len(tokenizer), dim=32, hidden_dim=64,
                                n_layers=1, n_heads=4, max_position_embeddings=config.max_length,
                                dropout=0., attention_dropout=0.)
    return TimingController(AutoModel.from_config(enc_config), config), tokenizer


def forward_batch(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    targets = batch.pop("targets")
    values = model(**batch)
    return values, targets


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for batch in loader:
        values, targets = forward_batch(model, batch, device)
        predictions.append(values.cpu())
        labels.append(targets.cpu())
    pred, truth = torch.cat(predictions), torch.cat(labels)
    if not torch.isfinite(pred).all():
        raise RuntimeError("Non-finite validation predictions")
    selected = TimingController.policies(pred)
    observed_gains = torch.cat((torch.zeros(len(truth), 1), truth), dim=1)
    realized = observed_gains.gather(1, selected[:, None]).mean().item()
    return {
        "rows": len(truth), "mse_seconds_squared": (pred - truth).square().mean().item(),
        "mae_now_s": (pred[:, 0] - truth[:, 0]).abs().mean().item(),
        "mae_wait_s": (pred[:, 1] - truth[:, 1]).abs().mean().item(),
        "selected_policy_counts": {p: int((selected == i).sum()) for i, p in enumerate(POLICIES)},
        "offline_selected_recorded_gain_s": realized,
        "fixed_policy_recorded_gains_s": {p: observed_gains[:, i].mean().item()
                                          for i, p in enumerate(POLICIES)},
        "scope": "Held-out recorded labels only; excludes controller serving overhead and is not online speedup.",
    }


def cpu_check(model, tokenizer, processor, train_encoded, validation_rows, optimizer, output):
    batch = processor.collate(train_encoded[:2])
    before_head = model.head[1].weight.detach().clone()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values, targets = forward_batch(model, batch, "cpu")
    loss = model.loss(values, targets)
    if not torch.isfinite(loss) or values.shape != targets.shape:
        raise RuntimeError("Invalid controller output or loss")
    loss.backward()
    encoder_grad = sum(float(p.grad.detach().abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
    head_grad = sum(float(p.grad.detach().abs().sum()) for p in model.head.parameters() if p.grad is not None)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if not encoder_grad > 0 or not head_grad > 0 or not torch.isfinite(torch.tensor(grad_norm)):
        raise RuntimeError("Expected finite nonzero gradients in both encoder and regression head")
    optimizer.step()
    if torch.equal(before_head, model.head[1].weight):
        raise RuntimeError("Optimizer did not update the regression head")
    model.eval()
    with torch.no_grad():
        expected, _ = forward_batch(model, batch, "cpu")
    checkpoint = output / "cpu_check_checkpoint"
    model.save(checkpoint, tokenizer)
    restored, restored_tokenizer = TimingController.load(checkpoint)
    with torch.no_grad():
        actual, _ = forward_batch(restored, batch, "cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    inp = validation_rows[0]["input"]
    if processor.encode(inp) != ControllerInputEncoder(restored_tokenizer, model.config.max_length).encode(inp):
        raise RuntimeError("Tokenizer/preprocessing changed after checkpoint restore")
    predictions = restored.predict_inputs([inp], restored_tokenizer)
    if predictions[0]["policy"] not in POLICIES:
        raise RuntimeError("Invalid policy output")
    report = {
        "passed": True, "mode": "cpu-check", "device": "cpu",
        "purpose": "One-batch functional check of a tiny RANDOM encoder; not a trained research controller.",
        "optimizer_steps": 1, "batch_size": len(targets), "output_shape": list(values.shape),
        "loss_finite": True, "encoder_has_gradient": encoder_grad > 0, "head_has_gradient": head_grad > 0,
        "optimizer_changed_weights": True, "checkpoint_reload_max_error": (actual - expected).abs().max().item(),
        "restored_tokenizer_matches": True, "prediction_api_checked": True,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "performance_evaluation_performed": False, "checkpoint": checkpoint.name,
    }
    write_json(output / "cpu_check_report.json", report)
    write_json(checkpoint / "purpose.json", {"purpose": report["purpose"], "deployable_trained_policy": False})
    return report


def run(args):
    if args.mode == "cpu-check" and args.device != "cpu":
        raise ValueError("cpu-check requires --device cpu")
    if args.mode == "train" and args.encoder is None:
        raise ValueError("train requires --encoder pointing to local pretrained weights")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows, data_report = load_controller_rows(args.data, args.quality_policy)
    train_rows, validation_rows, split = split_by_task(rows, args.validation_fraction, args.seed)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "data_report.json", data_report)
    write_json(args.output / "split.json", split)
    cfg = ControllerConfig(max_length=args.max_length, head_hidden_size=args.head_hidden_size,
                           dropout=0. if args.mode == "cpu-check" else args.dropout)
    if args.mode == "cpu-check":
        model, tokenizer = tiny_random_controller(train_rows, cfg)
    else:
        from transformers import AutoTokenizer
        model = TimingController.from_encoder_path(args.encoder, cfg)
        tokenizer = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True)
    processor = ControllerInputEncoder(tokenizer, cfg.max_length)
    train_encoded = processor.encode_rows(train_rows)
    validation_encoded = processor.encode_rows(validation_rows)
    model.fit_target_scaling([ex["targets"] for ex in train_encoded])
    model.to(args.device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    code_paths = [Path(__file__), ROOT / "spork_core/controller_data.py", ROOT / "spork_core/timing_controller.py"]
    manifest = {
        "args": {k: [str(x) for x in v] if k == "data" else str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
        "controller_config": asdict(cfg), "input_version": INPUT_VERSION,
        "encoder_origin": "tiny_random_functional_check" if args.mode == "cpu-check" else str(args.encoder),
        "target_mean_from_training_only": model.target_mean.tolist(),
        "target_scale_from_training_only": model.target_scale.tolist(),
        "numeric_features": ["log1p(visible_generated_tokens)/10", "log1p(wait_tokens)/10"],
        "packages": {n: version(n) for n in ("torch", "transformers", "tokenizers", "safetensors")},
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
        "actor_loaded_or_modified": False, "vllm_used": False,
        "data_caveat": "Quality matching is a selection rule, not proof of task success or label stability.",
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "tokenization_report.json", [
        {"id": ex["id"], "encoded_length": len(ex["input_ids"]), "fields": ex["token_counts"]}
        for ex in train_encoded + validation_encoded])
    for p in code_paths:
        target = args.output / "code" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    if args.mode == "cpu-check":
        result = cpu_check(model, tokenizer, processor, train_encoded, validation_rows, optimizer, args.output)
        print(json.dumps({**result, "rows_read": data_report["rows_read"], "rows_kept": len(rows),
                          "train_rows": len(train_rows), "validation_rows": len(validation_rows),
                          "output": str(args.output)}, ensure_ascii=False, indent=2))
        return result
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_encoded, batch_size=args.batch_size, shuffle=True,
                              generator=generator, collate_fn=processor.collate)
    validation_loader = DataLoader(validation_encoded, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=processor.collate)
    best = float("inf")
    best_checkpoint = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            values, targets = forward_batch(model, batch, args.device)
            loss = model.loss(values, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss in epoch {epoch}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient in epoch {epoch}")
            optimizer.step()
            losses.append((float(loss.detach()), len(targets)))
        metrics = evaluate(model, validation_loader, args.device)
        metrics.update(epoch=epoch, training_normalized_mse=sum(v * n for v, n in losses) / sum(n for _, n in losses))
        with (args.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        if metrics["mse_seconds_squared"] < best:
            best = metrics["mse_seconds_squared"]
            best_checkpoint = f"checkpoint_epoch_{epoch:03d}"
            model.save(args.output / best_checkpoint, tokenizer)
    result = {"mode": "train", "epochs_completed": args.epochs, "best_checkpoint": best_checkpoint,
              "best_validation_mse_seconds_squared": best, "all_input_rows_pilot": data_report["all_rows_pilot"],
              "online_performance_evaluated": False}
    write_json(args.output / "training_report.json", result)
    return result


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("cpu-check", "train"), default="cpu-check")
    ap.add_argument("--data", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    ap.add_argument("--encoder", type=Path, help="Local pretrained BERT/DistilBERT directory for train mode")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--quality-policy", choices=("matched", "all-valid"), default="matched")
    ap.add_argument("--validation-fraction", type=float, default=.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--head-hidden-size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=.01)
    ap.add_argument("--cpu-threads", type=int, default=2)
    args = ap.parse_args()
    if args.max_length < 32 or any(getattr(args, k) <= 0 for k in
                                 ("head_hidden_size", "epochs", "batch_size", "encoder_lr", "head_lr", "cpu_threads")):
        ap.error("Invalid length, batch size, epoch count or optimizer parameter")
    if args.weight_decay < 0 or not 0 <= args.dropout < 1 or not 0 < args.validation_fraction < 1:
        ap.error("Invalid weight decay, dropout or validation fraction")
    return args


if __name__ == "__main__":
    run(parse_args())
