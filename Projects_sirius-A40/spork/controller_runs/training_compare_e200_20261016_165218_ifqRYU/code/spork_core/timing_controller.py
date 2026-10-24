"""Small text encoder plus two regression outputs, expressed in seconds."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .controller_data import ControllerInputEncoder, INPUT_VERSION, TARGET_NAMES


POLICIES = ("off", "now", "wait")


@dataclass
class ControllerConfig:
    max_length: int = 512
    head_hidden_size: int = 128
    dropout: float = .1
    input_version: str = INPUT_VERSION
    input_ablation: str = "none"


class TimingController(nn.Module):
    def __init__(self, encoder, config=None):
        super().__init__()
        self.config = config or ControllerConfig()
        if encoder.config.model_type not in ("bert", "distilbert"):
            raise ValueError("First version supports BERT and DistilBERT encoders")
        if self.config.input_version != INPUT_VERSION:
            raise ValueError("Unsupported controller input preprocessing version")
        if self.config.input_ablation not in ("none", "mask_reasoning"):
            raise ValueError("Unsupported controller input ablation")
        if self.config.max_length > encoder.config.max_position_embeddings:
            raise ValueError("max_length exceeds encoder position capacity")
        self.encoder = encoder
        dim = encoder.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(dim + 2), nn.Linear(dim + 2, self.config.head_hidden_size),
                                  nn.GELU(), nn.Dropout(self.config.dropout),
                                  nn.Linear(self.config.head_hidden_size, 2))
        self.register_buffer("target_mean", torch.zeros(2))
        self.register_buffer("target_scale", torch.ones(2))

    @classmethod
    def from_encoder_path(cls, path, config=None):
        enc_config = AutoConfig.from_pretrained(path, local_files_only=True)
        if enc_config.model_type not in ("bert", "distilbert"):
            raise ValueError("Provide a local BERT/DistilBERT encoder, not the SPORK Actor")
        encoder = AutoModel.from_pretrained(path, local_files_only=True)
        return cls(encoder, config)

    def fit_target_scaling(self, training_targets):
        targets = torch.as_tensor(training_targets, dtype=torch.float32, device=self.target_mean.device)
        if targets.ndim != 2 or targets.shape[1] != 2 or not len(targets) or not torch.isfinite(targets).all():
            raise ValueError("Expected finite training targets with shape [N, 2]")
        # Fit on training groups only. A one-second floor avoids unstable division
        # for constant/near-constant targets in very small functional checks.
        self.target_mean.copy_(targets.mean(0))
        self.target_scale.copy_(targets.std(0, unbiased=False).clamp_min(1.0))

    def forward(self, input_ids, attention_mask, numeric_features):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        normalized = self.head(torch.cat((pooled, numeric_features.to(pooled.dtype)), dim=-1))
        return normalized * self.target_scale + self.target_mean

    def loss(self, predicted_seconds, target_seconds):
        if predicted_seconds.shape != target_seconds.shape or predicted_seconds.shape[-1] != 2:
            raise ValueError("Expected matching [batch, 2] values and targets")
        return (((predicted_seconds - target_seconds) / self.target_scale) ** 2).mean()

    @staticmethod
    def policies(values):
        if values.ndim != 2 or values.shape[1] != 2 or not torch.isfinite(values).all():
            raise ValueError("Expected finite [batch, 2] predicted gains")
        # An exact zero tie chooses off. Ties between positive heads choose now.
        off = torch.zeros((len(values), 1), dtype=values.dtype, device=values.device)
        return torch.cat((off, values), dim=1).argmax(dim=1)

    def save(self, directory, tokenizer):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.encoder.config.save_pretrained(path / "encoder_config")
        tokenizer.save_pretrained(path / "tokenizer")
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, str(path / "controller.safetensors"))
        (path / "controller_config.json").write_text(json.dumps(
            {"schema_version": 1, "config": asdict(self.config), "targets": list(TARGET_NAMES),
             "output_unit": "seconds", "policy_order": list(POLICIES)}, indent=2) + "\n")

    @classmethod
    def load(cls, directory, device="cpu"):
        path = Path(directory)
        data = json.loads((path / "controller_config.json").read_text())
        if data["schema_version"] != 1 or data["targets"] != list(TARGET_NAMES):
            raise ValueError("Incompatible controller checkpoint")
        enc_config = AutoConfig.from_pretrained(path / "encoder_config", local_files_only=True)
        model = cls(AutoModel.from_config(enc_config), ControllerConfig(**data["config"]))
        model.load_state_dict(load_file(str(path / "controller.safetensors")), strict=True)
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
        return model.to(device).eval(), tokenizer

    @torch.no_grad()
    def predict_inputs(self, inputs, tokenizer):
        """Inference-only interface, accepting past-only input dictionaries."""
        if not inputs:
            return []
        self.eval()
        processor = ControllerInputEncoder(tokenizer, self.config.max_length, self.config.input_ablation)
        batch = processor.collate([processor.encode(inp) for inp in inputs])
        device = self.target_mean.device
        values = self(**{k: v.to(device) for k, v in batch.items()})
        policy_ids = self.policies(values).cpu().tolist()
        return [{"q_now_s": float(v[0]), "q_wait_s": float(v[1]), "policy": POLICIES[p]}
                for v, p in zip(values.cpu().tolist(), policy_ids)]
