#!/usr/bin/env python3
"""
tier1_spikebert.py

Standalone SpikeBERT-style Knowledge Distillation script for Vietnamese
sentiment analysis on UIT-VSFC.

Dataset parquet columns:
    text:  segmented Vietnamese sentence
    label: integer label in {0, 1, 2}

Teacher:
    Frozen PhoBERT classifier fine-tuned on UIT-VSFC.

Student:
    PhoBERT backbone with FFN sub-layers in the last N transformer blocks
    replaced by SpikeBERTFFN modules trained with BPTT and Arctan surrogate.

Energy reporting:
    Metric 1: Theoretical energy on 45nm neuromorphic hardware.
    Metric 2: Actual GPU inference energy via NVML.
    Metric 3: Peak training memory via torch.cuda.max_memory_allocated().
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LABEL_NAMES = ["negative", "neutral", "positive"]
ALIGNMENT_LAYERS = [3, 6, 9, 12]
NUM_LABELS = 3

MAC_ENERGY_PJ = 4.6
AC_ENERGY_PJ = 0.9

LOGGER = logging.getLogger("tier1_spikebert")


@dataclass
class Config:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 16
    eval_batch_size: int = 32
    spiking_layers: int = 4

    output_dir: str = "outputs/tier1"
    teacher_ckpt: str = "uit/uit-models/phobert_vsfc/best_model.pth"
    model_name: str = "vinai/phobert-base-v2"

    train_path: str = "uit/cache/train_segmented.parquet"
    dev_path: str = "uit/cache/dev_segmented.parquet"
    test_path: str = "uit/cache/test_segmented.parquet"

    max_length: int = 256
    lr: float = 2e-5
    weight_decay: float = 0.01

    alpha: float = 0.3
    temperature_kd: float = 4.0
    lambda_f: float = 0.1

    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    num_workers: int = 2

    local_files_only: bool = False
    amp: bool = True

    spike_timesteps: int = 4
    spike_threshold: float = 1.0
    surrogate_alpha: float = 2.0


class VSFCDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        text = str(row["text"])
        label = int(row["label"])

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class ArctanSpike(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x > 0).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        denominator = 1.0 + ((math.pi / 2.0) * alpha * x).pow(2)
        grad = alpha / denominator
        return grad_output * grad, None


class SpikeBERTFFN(nn.Module):
    """
    Replaces the standard FFN inside a transformer block.

    The module uses:
    - T-step LIF membrane integration,
    - hard reset after spike,
    - Arctan surrogate gradient,
    - linear1 + BatchNorm once before the timestep loop.
    """

    def __init__(
        self,
        linear1: nn.Linear,
        linear2: nn.Linear,
        timesteps: int = 4,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()

        self.linear1 = linear1
        self.linear2 = linear2

        self.timesteps = int(timesteps)
        self.threshold = float(threshold)
        self.surrogate_alpha = float(surrogate_alpha)

        self.bn = nn.BatchNorm1d(linear1.out_features)

        self.firing_rate = 0.0
        self._spike_sum = 0.0
        self._spike_count = 0.0

    def _batch_norm_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        flat = x.reshape(batch_size * seq_len, hidden_dim)
        normed = self.bn(flat)
        return normed.reshape(batch_size, seq_len, hidden_dim)

    def reset_firing_statistics(self) -> None:
        self.firing_rate = 0.0
        self._spike_sum = 0.0
        self._spike_count = 0.0

    def _forward_fp32(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.float()

        # Apply linear1 + BN once before the timestep loop.
        # This avoids updating BatchNorm running statistics T times per forward.
        current = self.linear1(x)
        current = self._batch_norm_sequence(current)

        membrane = torch.zeros(
            current.size(0),
            current.size(1),
            self.linear1.out_features,
            dtype=torch.float32,
            device=current.device,
        )
        output_sum = torch.zeros(
            current.size(0),
            current.size(1),
            self.linear2.out_features,
            dtype=torch.float32,
            device=current.device,
        )

        spike_sum = 0.0
        spike_count = 0.0

        for _ in range(self.timesteps):
            membrane = membrane + current
            spikes = ArctanSpike.apply(membrane - self.threshold, self.surrogate_alpha)

            output_sum = output_sum + self.linear2(spikes)

            # Hard reset.
            membrane = membrane * (1.0 - spikes.detach())

            spike_sum += float(spikes.detach().sum().item())
            spike_count += float(spikes.numel())

        if spike_count > 0:
            self._spike_sum += spike_sum
            self._spike_count += spike_count
            self.firing_rate = float(self._spike_sum / max(self._spike_count, 1.0))

        return output_sum / float(self.timesteps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # LIF dynamics are numerically sensitive; force them to run in FP32
        # even when the outer training loop enables AMP.
        if hidden_states.is_cuda and hasattr(torch, "amp"):
            with torch.amp.autocast(device_type="cuda", enabled=False):
                output = self._forward_fp32(hidden_states)
        else:
            output = self._forward_fp32(hidden_states)

        return output.to(hidden_states.dtype)


class IdentityIntermediate(nn.Module):
    """Identity replacement for RobertaIntermediate in patched FFN layers."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class SpikeBERTOutputWrapper(nn.Module):
    """
    Replacement for RobertaOutput in selected layers.

    RobertaLayer normally computes:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

    We set:
        layer.intermediate = IdentityIntermediate()

    Therefore hidden_states passed into this wrapper is attention_output.
    The wrapper applies SpikeBERTFFN directly to input_tensor, then keeps
    the original dropout + LayerNorm.
    """

    def __init__(self, spiking_ffn: SpikeBERTFFN, original_output: nn.Module):
        super().__init__()
        self.spiking_ffn = spiking_ffn
        self.dropout = original_output.dropout
        self.LayerNorm = original_output.LayerNorm

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        del hidden_states
        ffn_output = self.spiking_ffn(input_tensor)
        ffn_output = self.dropout(ffn_output)
        return self.LayerNorm(ffn_output + input_tensor)


def load_phobert_backbone(config: Config) -> nn.Module:
    """
    Load PhoBERT with eager attention.

    Student patches FFN submodules inside RobertaLayer but leaves the
    HuggingFace RobertaModel forward path intact. Eager attention avoids
    SDPA mask incompatibilities on newer Transformers/PyTorch versions.
    """
    try:
        return AutoModel.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModel.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        return model


class PhoBERTTeacher(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.encoder = load_phobert_backbone(config)
        hidden_size = int(self.encoder.config.hidden_size)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)

        self.load_checkpoint(config.teacher_ckpt)
        self.freeze()

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")

        state = safe_torch_load(path, map_location="cpu")
        state_dict = extract_state_dict(state)
        state_dict = normalize_checkpoint_keys(state_dict)

        incompatible = self.load_state_dict(state_dict, strict=False)

        LOGGER.info("Loaded teacher checkpoint from %s", path)
        LOGGER.info("Teacher missing keys: %s", incompatible.missing_keys)
        LOGGER.info("Teacher unexpected keys: %s", incompatible.unexpected_keys)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls)

        return {
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "last_hidden_state": outputs.last_hidden_state,
        }


class SpikeBERTStudent(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.encoder = load_phobert_backbone(config)
        hidden_size = int(self.encoder.config.hidden_size)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)

        self.spiking_layer_indices = self._select_spiking_layers(config.spiking_layers)

        # Plain dict to avoid duplicate registration.
        # Actual SpikeBERTFFN modules are registered inside Roberta layers.
        self.spiking_ffns: Dict[str, SpikeBERTFFN] = {}

        self.feature_projectors = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                )
                for layer in ALIGNMENT_LAYERS
                if layer <= int(self.encoder.config.num_hidden_layers)
            }
        )

        self._replace_ffn_layers()

    def _select_spiking_layers(self, spiking_layers: int) -> List[int]:
        num_layers = int(self.encoder.config.num_hidden_layers)
        layers = max(0, min(int(spiking_layers), num_layers))
        return list(range(num_layers - layers, num_layers))

    def _replace_ffn_layers(self) -> None:
        for layer_index, layer in enumerate(self.encoder.encoder.layer):
            if layer_index not in self.spiking_layer_indices:
                continue

            spiking_ffn = SpikeBERTFFN(
                linear1=layer.intermediate.dense,
                linear2=layer.output.dense,
                timesteps=self.config.spike_timesteps,
                threshold=self.config.spike_threshold,
                surrogate_alpha=self.config.surrogate_alpha,
            )

            original_output = layer.output

            layer.intermediate = IdentityIntermediate()
            layer.output = SpikeBERTOutputWrapper(
                spiking_ffn=spiking_ffn,
                original_output=original_output,
            )

            self.spiking_ffns[str(layer_index)] = spiking_ffn

        LOGGER.info("Spiking FFN layer indices: %s", self.spiking_layer_indices)

    def reset_firing_statistics(self) -> None:
        for module in self.spiking_ffns.values():
            module.reset_firing_statistics()

    def mean_firing_rate(self) -> float:
        rates = [module.firing_rate for module in self.spiking_ffns.values()]
        if len(rates) == 0:
            return 0.0
        return float(np.mean(rates))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls)

        return {
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "last_hidden_state": outputs.last_hidden_state,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier 1 SpikeBERT-style PhoBERT to SNN KD for UIT-VSFC"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--spiking_layers", type=int, default=4)

    parser.add_argument("--output_dir", type=str, default="outputs/tier1")
    parser.add_argument("--teacher_ckpt", type=str, default="uit/uit-models/phobert_vsfc/best_model.pth")

    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--train_path", type=str, default="uit/cache/train_segmented.parquet")
    parser.add_argument("--dev_path", type=str, default="uit/cache/dev_segmented.parquet")
    parser.add_argument("--test_path", type=str, default="uit/cache/test_segmented.parquet")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--temperature_kd", type=float, default=4.0)
    parser.add_argument("--lambda_f", type=float, default=0.1)

    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument("--spike_timesteps", type=int, default=4)
    parser.add_argument("--spike_threshold", type=float, default=1.0)
    parser.add_argument("--surrogate_alpha", type=float, default=2.0)

    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_torch_load(path: Path, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """
    Extract a tensor-only state_dict.

    This avoids passing non-tensor metadata such as epoch, config, or f1_macro
    into load_state_dict().
    """
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model", "teacher_state_dict", "net"]:
            value = checkpoint.get(key)
            if isinstance(value, dict) and all(isinstance(k, str) for k in value.keys()):
                filtered = {k: v for k, v in value.items() if torch.is_tensor(v)}
                if filtered:
                    return filtered

        if all(isinstance(k, str) for k in checkpoint.keys()):
            filtered = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
            if filtered:
                return filtered

    raise ValueError("Could not find a usable tensor-only state_dict in the checkpoint")


def normalize_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}

    prefixes = ["module.", "model.", "teacher.", "student."]

    for key, value in state_dict.items():
        new_key = key

        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        if new_key.startswith("backbone."):
            new_key = "encoder." + new_key[len("backbone."):]
        if new_key.startswith("phobert."):
            new_key = "encoder." + new_key[len("phobert."):]
        if new_key.startswith("roberta."):
            new_key = "encoder." + new_key[len("roberta."):]

        normalized[new_key] = value

    return normalized


def read_vsfc_split(path: str) -> pd.DataFrame:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")

    dataframe = pd.read_parquet(split_path)

    required = {"text", "label"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {split_path}: {missing}. "
            f"Expected parquet columns: text, label"
        )

    dataframe = dataframe[["text", "label"]].copy()
    dataframe["text"] = dataframe["text"].fillna("").astype(str)
    dataframe["label"] = dataframe["label"].astype(int)
    dataframe = dataframe[dataframe["label"].isin([0, 1, 2])].reset_index(drop=True)

    if len(dataframe) == 0:
        raise ValueError(f"No usable rows found in {split_path}")

    return dataframe


def make_dataloaders(
    config: Config,
    tokenizer,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    train_df = read_vsfc_split(config.train_path)
    dev_df = read_vsfc_split(config.dev_path)
    test_df = read_vsfc_split(config.test_path)

    labels = train_df["label"].to_numpy(dtype=np.int64)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=labels,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    LOGGER.info("Train size=%d Dev size=%d Test size=%d", len(train_df), len(dev_df), len(test_df))
    LOGGER.info("Train label counts: %s", train_df["label"].value_counts().sort_index().to_dict())
    LOGGER.info("Class weights: %s", class_weights_tensor.tolist())

    train_dataset = VSFCDataset(train_df, tokenizer, config.max_length)
    dev_dataset = VSFCDataset(dev_df, tokenizer, config.max_length)
    test_dataset = VSFCDataset(test_df, tokenizer, config.max_length)

    cuda_available = torch.cuda.is_available()
    persistent_workers = config.num_workers > 0 and os.name != "nt"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=cuda_available,
        persistent_workers=persistent_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=cuda_available,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=cuda_available,
        persistent_workers=persistent_workers,
    )

    return train_loader, dev_loader, test_loader, class_weights_tensor


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def compute_feature_loss(
    student: SpikeBERTStudent,
    student_hidden_states,
    teacher_hidden_states,
) -> torch.Tensor:
    if student_hidden_states is None or teacher_hidden_states is None:
        device = next(student.parameters()).device
        return torch.zeros((), device=device)

    losses: List[torch.Tensor] = []

    for layer in ALIGNMENT_LAYERS:
        if layer >= len(student_hidden_states) or layer >= len(teacher_hidden_states):
            continue

        if str(layer) not in student.feature_projectors:
            continue

        projector = student.feature_projectors[str(layer)]

        student_hidden = projector(student_hidden_states[layer].float())
        teacher_hidden = teacher_hidden_states[layer].detach().float()

        losses.append(F.mse_loss(student_hidden, teacher_hidden))

    if len(losses) == 0:
        device = next(student.parameters()).device
        return torch.zeros((), device=device)

    return torch.stack(losses).mean()


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    return F.kl_div(log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def amp_autocast(device: torch.device, enabled: bool):
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def get_snn_modules(student) -> Dict[str, nn.Module]:
    if hasattr(student, "spiking_ffns"):
        return student.spiking_ffns
    if hasattr(student, "implicit_ffns"):
        return student.implicit_ffns
    return {}


def reset_snn_energy_statistics(student) -> None:
    """
    Reset firing/convergence counters before the dedicated inference-energy pass.
    This makes layer firing rates in the energy report correspond to that pass.
    """
    if hasattr(student, "reset_firing_statistics"):
        student.reset_firing_statistics()

    for module in get_snn_modules(student).values():
        if hasattr(module, "firing_rate"):
            module.firing_rate = 0.0
        if hasattr(module, "avg_anderson_steps"):
            module.avg_anderson_steps = 0.0
        if hasattr(module, "skipped_steps"):
            module.skipped_steps = 0
        if hasattr(module, "num_forward_calls"):
            module.num_forward_calls = 0
        if hasattr(module, "_total_anderson_steps"):
            module._total_anderson_steps = 0.0
        if hasattr(module, "_total_firing"):
            module._total_firing = 0.0
        if hasattr(module, "_spike_sum"):
            module._spike_sum = 0.0
        if hasattr(module, "_spike_count"):
            module._spike_count = 0.0


def get_energy_timesteps(config: Config) -> int:
    return int(getattr(config, "spike_timesteps", getattr(config, "energy_timesteps", 1)))


def get_snn_forward_activity_counts(student) -> Dict[str, int]:
    """
    Count whether each SNN layer actually saw an inference/training forward pass.

    Tier 1 SpikeBERTFFN does not have num_forward_calls, but it accumulates
    _spike_count even if the model is silent. Therefore _spike_count > 0 is
    the correct indicator that inference was run.
    """
    counts: Dict[str, int] = {}

    for layer_id, module in get_snn_modules(student).items():
        if hasattr(module, "num_forward_calls"):
            counts[str(layer_id)] = int(getattr(module, "num_forward_calls", 0))
        elif hasattr(module, "_spike_count"):
            counts[str(layer_id)] = int(float(getattr(module, "_spike_count", 0.0)) > 0.0)
        else:
            counts[str(layer_id)] = 0

    return counts


def compute_theoretical_energy_report(
    tier_name: str,
    config: Config,
    student,
    teacher,
    test_metrics: Dict[str, object],
) -> Dict[str, object]:
    """
    Theoretical energy estimate in pJ/sample.

    Method:
    - MAC cost = 4.6 pJ.
    - AC cost  = 0.9 pJ.
    - Self-attention remains ANN and always uses MAC.
    - Only explicitly replaced SNN FFN layers use AC.
    - Attention FLOPs use projection-only approximation:
        seq_len * hidden_dim * hidden_dim * 4
      covering Q/K/V/O projections.
    - FFN FLOPs:
        seq_len * (hidden_dim * intermediate_dim + intermediate_dim * hidden_dim)

    This is a neuromorphic-hardware theoretical estimate, not measured GPU energy.
    """
    del teacher

    hidden_dim = int(student.encoder.config.hidden_size)
    intermediate_dim = int(student.encoder.config.intermediate_size)
    num_layers = int(student.encoder.config.num_hidden_layers)
    seq_len = int(config.max_length)
    num_labels = NUM_LABELS
    timesteps = get_energy_timesteps(config)

    snn_modules = get_snn_modules(student)
    snn_layer_ids = {int(k) for k in snn_modules.keys()}
    snn_forward_activity_counts = get_snn_forward_activity_counts(student)

    if snn_modules and not any(count > 0 for count in snn_forward_activity_counts.values()):
        raise RuntimeError(
            "No SNN forward activity was recorded before theoretical energy computation. "
            "Run student inference before compute_theoretical_energy_report()."
        )

    attention_flops = seq_len * hidden_dim * hidden_dim * 4
    ffn_flops = seq_len * (
        hidden_dim * intermediate_dim
        + intermediate_dim * hidden_dim
    )
    classifier_flops = hidden_dim * num_labels

    embedding_lookup_proxy_ops = seq_len * hidden_dim

    teacher_layers: List[Dict[str, object]] = []
    student_layers: List[Dict[str, object]] = []

    teacher_total_pj = 0.0
    student_total_pj = 0.0

    teacher_layers.append(
        {
            "name": "embedding_lookup_proxy",
            "kind": "embedding_lookup_not_counted_as_mac",
            "ops_proxy": int(embedding_lookup_proxy_ops),
            "energy_pj": 0.0,
            "note": "Embedding lookup is not counted as dense MAC energy.",
        }
    )
    student_layers.append(
        {
            "name": "embedding_lookup_proxy",
            "kind": "embedding_lookup_not_counted_as_mac",
            "ops_proxy": int(embedding_lookup_proxy_ops),
            "energy_pj": 0.0,
            "note": "Embedding lookup is not counted as dense MAC energy.",
        }
    )

    for layer_idx in range(num_layers):
        teacher_attn_energy = attention_flops * MAC_ENERGY_PJ
        teacher_ffn_energy = ffn_flops * MAC_ENERGY_PJ
        teacher_total_pj += teacher_attn_energy + teacher_ffn_energy

        teacher_layers.append(
            {
                "name": f"layer_{layer_idx}_self_attention",
                "kind": "ANN_attention_MAC",
                "flops": int(attention_flops),
                "energy_per_op_pj": MAC_ENERGY_PJ,
                "energy_pj": float(teacher_attn_energy),
            }
        )
        teacher_layers.append(
            {
                "name": f"layer_{layer_idx}_ffn",
                "kind": "ANN_FFN_MAC",
                "flops": int(ffn_flops),
                "energy_per_op_pj": MAC_ENERGY_PJ,
                "energy_pj": float(teacher_ffn_energy),
            }
        )

        student_attn_energy = attention_flops * MAC_ENERGY_PJ
        student_total_pj += student_attn_energy

        student_layers.append(
            {
                "name": f"layer_{layer_idx}_self_attention",
                "kind": "ANN_attention_MAC",
                "flops": int(attention_flops),
                "energy_per_op_pj": MAC_ENERGY_PJ,
                "energy_pj": float(student_attn_energy),
                "note": "Self-attention remains ANN in the current hybrid architecture.",
            }
        )

        if layer_idx in snn_layer_ids:
            module = snn_modules[str(layer_idx)]
            firing_rate = float(getattr(module, "firing_rate", 0.0))
            snn_ffn_energy = timesteps * firing_rate * ffn_flops * AC_ENERGY_PJ
            student_total_pj += snn_ffn_energy

            student_layers.append(
                {
                    "name": f"layer_{layer_idx}_ffn",
                    "kind": "SNN_FFN_AC",
                    "flops_equiv_ann": int(ffn_flops),
                    "timesteps": int(timesteps),
                    "firing_rate": firing_rate,
                    "energy_per_op_pj": AC_ENERGY_PJ,
                    "energy_pj": float(snn_ffn_energy),
                }
            )
        else:
            ann_ffn_energy = ffn_flops * MAC_ENERGY_PJ
            student_total_pj += ann_ffn_energy

            student_layers.append(
                {
                    "name": f"layer_{layer_idx}_ffn",
                    "kind": "ANN_FFN_MAC",
                    "flops": int(ffn_flops),
                    "energy_per_op_pj": MAC_ENERGY_PJ,
                    "energy_pj": float(ann_ffn_energy),
                }
            )

    teacher_classifier_energy = classifier_flops * MAC_ENERGY_PJ
    student_classifier_energy = classifier_flops * MAC_ENERGY_PJ
    teacher_total_pj += teacher_classifier_energy
    student_total_pj += student_classifier_energy

    teacher_layers.append(
        {
            "name": "classifier_head",
            "kind": "ANN_classifier_MAC",
            "flops": int(classifier_flops),
            "energy_per_op_pj": MAC_ENERGY_PJ,
            "energy_pj": float(teacher_classifier_energy),
        }
    )
    student_layers.append(
        {
            "name": "classifier_head",
            "kind": "ANN_classifier_MAC",
            "flops": int(classifier_flops),
            "energy_per_op_pj": MAC_ENERGY_PJ,
            "energy_pj": float(student_classifier_energy),
        }
    )

    student_teacher_ratio = student_total_pj / max(teacher_total_pj, 1e-12)
    teacher_student_saving = teacher_total_pj / max(student_total_pj, 1e-12)
    student_mean_firing_rate = float(student.mean_firing_rate())

    firing_rate_warning = None
    if snn_modules and any(float(getattr(module, "firing_rate", 0.0)) == 0.0 for module in snn_modules.values()):
        firing_rate_warning = (
            "Some or all SNN firing rates are zero. This may be valid if the model is silent, "
            "but it should be checked before claiming energy savings."
        )

    return {
        "tier": tier_name,
        "seed": int(config.seed),
        "metric_name": "Theoretical energy (45nm neuromorphic, pJ per sample)",
        "mac_energy_pj": MAC_ENERGY_PJ,
        "ac_energy_pj": AC_ENERGY_PJ,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_layers": num_layers,
        "snn_report_timesteps": timesteps,
        "snn_forward_activity_counts": snn_forward_activity_counts,
        "student_mean_firing_rate": student_mean_firing_rate,
        "firing_rate_warning": firing_rate_warning,
        "attention_flops_formula": "seq_len * hidden_dim * hidden_dim * 4, projection-only approximation",
        "ffn_flops_formula": "seq_len * (hidden_dim * intermediate_dim + intermediate_dim * hidden_dim)",
        "teacher_total_energy_pj_per_sample": float(teacher_total_pj),
        "student_total_energy_pj_per_sample": float(student_total_pj),
        "student_teacher_theoretical_energy_ratio": float(student_teacher_ratio),
        "teacher_student_theoretical_saving_ratio": float(teacher_student_saving),
        "teacher_layers": teacher_layers,
        "student_layers": student_layers,
        "test_metrics": {
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "f1_macro": float(test_metrics["f1_macro"]),
            "f1_weighted": float(test_metrics["f1_weighted"]),
            "f1_per_class": [float(x) for x in test_metrics["f1_per_class"]],
        },
        "note": (
            "This is a theoretical estimate based on 45nm MAC/AC operation costs. "
            "Actual GPU energy is reported separately. Self-attention remains ANN "
            "and uses MAC cost. Only explicit SNN FFN layers use AC cost. "
            "Embedding lookup is not counted as dense MAC energy."
        ),
    }


def resolve_nvml_gpu_index() -> int:
    """
    Resolve physical NVML GPU index.

    In SLURM/MPS, CUDA_VISIBLE_DEVICES may be set to the selected physical GPU.
    torch.cuda.current_device() may return 0 after remapping, which is not
    always the same as the physical NVML index.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",")[0].strip()
        if first.isdigit():
            return int(first)
    return int(torch.cuda.current_device())


class NvmlPowerSampler:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = float(interval_seconds)
        self.samples: List[Tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._pynvml = None
        self.started = False

    def start(self) -> bool:
        if not torch.cuda.is_available():
            return False

        try:
            import pynvml

            self._pynvml = pynvml
            pynvml.nvmlInit()
            gpu_index = resolve_nvml_gpu_index()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.samples = []
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
            self.started = True
            return True
        except Exception as exc:
            LOGGER.info("NVML power sampler disabled: %s", exc)
            self.started = False
            return False

    def _sample_loop(self) -> None:
        assert self._pynvml is not None
        assert self._handle is not None

        while not self._stop_event.is_set():
            try:
                timestamp = time.perf_counter()
                power_watts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self.samples.append((timestamp, float(power_watts)))
            except Exception:
                pass

            time.sleep(self.interval_seconds)

    def stop(self, num_samples: int, wall_elapsed_seconds: float) -> Dict[str, object]:
        if not self.started:
            return {
                "available": False,
                "num_samples": int(num_samples),
                "avg_power_watts": None,
                "elapsed_seconds": float(wall_elapsed_seconds),
                "energy_joules": None,
                "energy_per_sample_joules": None,
                "num_power_samples": 0,
            }

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        try:
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()
        except Exception:
            pass

        if len(self.samples) == 0:
            return {
                "available": False,
                "num_samples": int(num_samples),
                "avg_power_watts": None,
                "elapsed_seconds": float(wall_elapsed_seconds),
                "energy_joules": None,
                "energy_per_sample_joules": None,
                "num_power_samples": 0,
            }

        avg_power = sum(power for _, power in self.samples) / len(self.samples)
        elapsed = max(float(wall_elapsed_seconds), 1e-9)
        energy_joules = avg_power * elapsed
        energy_per_sample = energy_joules / max(int(num_samples), 1)

        return {
            "available": True,
            "num_samples": int(num_samples),
            "avg_power_watts": float(avg_power),
            "elapsed_seconds": float(elapsed),
            "energy_joules": float(energy_joules),
            "energy_per_sample_joules": float(energy_per_sample),
            "num_power_samples": int(len(self.samples)),
            "note": (
                "This is real GPU board-power sampling via NVML during inference. "
                "On shared MPS GPUs, this may include power from other processes."
            ),
        }


@torch.no_grad()
def measure_inference_gpu_energy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    model_name: str,
) -> Dict[str, object]:
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()

    sampler = NvmlPowerSampler(interval_seconds=0.05)
    sampler.start()

    num_samples = 0
    start_time = time.perf_counter()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        batch_size = int(batch["labels"].size(0))
        num_samples += batch_size

        with amp_autocast(device, amp_enabled and device.type == "cuda"):
            _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                output_hidden_states=False,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    wall_elapsed = time.perf_counter() - start_time
    stats = sampler.stop(num_samples=num_samples, wall_elapsed_seconds=wall_elapsed)
    stats["model_name"] = model_name
    stats["wall_elapsed_seconds"] = float(wall_elapsed)

    return stats


def get_peak_training_memory_report(device: torch.device) -> Dict[str, object]:
    if device.type != "cuda":
        return {
            "available": False,
            "peak_memory_allocated_bytes": None,
            "peak_memory_allocated_mb": None,
        }

    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    return {
        "available": True,
        "peak_memory_allocated_bytes": peak_bytes,
        "peak_memory_allocated_mb": float(peak_bytes / (1024 ** 2)),
        "note": (
            "Peak GPU memory allocated during training, measured by "
            "torch.cuda.max_memory_allocated(). Capture this before extra energy-measurement inference."
        ),
    }


def write_energy_report(
    output_dir: Path,
    tier_name: str,
    config: Config,
    teacher,
    student,
    test_loader: DataLoader,
    device: torch.device,
    test_metrics: Dict[str, object],
    peak_training_memory: Dict[str, object],
) -> Dict[str, object]:
    amp_enabled = config.amp and device.type == "cuda"

    teacher_gpu_energy = measure_inference_gpu_energy(
        model=teacher,
        loader=test_loader,
        device=device,
        amp_enabled=amp_enabled,
        model_name="teacher_phobert",
    )

    reset_snn_energy_statistics(student)

    student_gpu_energy = measure_inference_gpu_energy(
        model=student,
        loader=test_loader,
        device=device,
        amp_enabled=amp_enabled,
        model_name=f"student_{tier_name}",
    )

    theoretical = compute_theoretical_energy_report(
        tier_name=tier_name,
        config=config,
        student=student,
        teacher=teacher,
        test_metrics=test_metrics,
    )

    teacher_eps = teacher_gpu_energy.get("energy_per_sample_joules")
    student_eps = student_gpu_energy.get("energy_per_sample_joules")

    if teacher_eps is not None and student_eps is not None:
        gpu_ratio = float(student_eps) / max(float(teacher_eps), 1e-12)
    else:
        gpu_ratio = None

    report = {
        "tier": tier_name,
        "seed": int(config.seed),
        "theoretical_energy": theoretical,
        "actual_gpu_energy_a100": {
            "teacher": teacher_gpu_energy,
            "student": student_gpu_energy,
            "student_teacher_energy_per_sample_ratio": gpu_ratio,
            "expected_interpretation": (
                "The student may consume more GPU energy than the teacher because "
                "current GPU hardware does not natively accelerate spike-based "
                "event-driven computation. The theoretical energy estimate is "
                "for dedicated neuromorphic hardware."
            ),
        },
        "training_memory": peak_training_memory,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"energy_report_seed{config.seed}.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    LOGGER.info("Saved detailed energy report to %s", report_path)

    return report


def train_one_epoch(
    config: Config,
    student: SpikeBERTStudent,
    teacher: PhoBERTTeacher,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ce_loss_fn: nn.Module,
    scaler,
    device: torch.device,
) -> Dict[str, float]:
    student.train()
    teacher.eval()
    student.reset_firing_statistics()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0
    steps = 0

    use_amp = config.amp and device.type == "cuda"

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_outputs = teacher(
                batch["input_ids"],
                batch["attention_mask"],
                output_hidden_states=True,
            )

        with amp_autocast(device, use_amp):
            student_outputs = student(
                batch["input_ids"],
                batch["attention_mask"],
                output_hidden_states=True,
            )

            ce_loss = ce_loss_fn(student_outputs["logits"].float(), batch["labels"])

            kd_loss = compute_kd_loss(
                student_outputs["logits"].float(),
                teacher_outputs["logits"].float(),
                config.temperature_kd,
            )

            feature_loss = compute_feature_loss(
                student,
                student_outputs["hidden_states"],
                teacher_outputs["hidden_states"],
            )

            loss = (
                config.alpha * ce_loss
                + (1.0 - config.alpha) * kd_loss
                + config.lambda_f * feature_loss
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        steps += 1
        total_loss += float(loss.detach().item())
        total_ce += float(ce_loss.detach().item())
        total_kd += float(kd_loss.detach().item())
        total_feat += float(feature_loss.detach().item())

    denom = max(steps, 1)

    return {
        "loss": total_loss / denom,
        "ce": total_ce / denom,
        "kd": total_kd / denom,
        "feature": total_feat / denom,
        "firing_rate": student.mean_firing_rate(),
    }


@torch.no_grad()
def evaluate(
    config: Config,
    student: SpikeBERTStudent,
    loader: DataLoader,
    ce_loss_fn: nn.Module,
    device: torch.device,
    reset_firing_stats: bool = True,
) -> Dict[str, object]:
    student.eval()

    if reset_firing_stats:
        student.reset_firing_statistics()

    use_amp = config.amp and device.type == "cuda"

    y_true: List[int] = []
    y_pred: List[int] = []

    total_loss = 0.0
    steps = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with amp_autocast(device, use_amp):
            outputs = student(
                batch["input_ids"],
                batch["attention_mask"],
                output_hidden_states=False,
            )
            loss = ce_loss_fn(outputs["logits"].float(), batch["labels"])

        predictions = outputs["logits"].argmax(dim=-1)

        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

        total_loss += float(loss.detach().item())
        steps += 1

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    f1_macro = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    f1_weighted = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0

    return {
        "loss": total_loss / max(steps, 1),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": [float(x) for x in per_class_f1],
        "firing_rate": student.mean_firing_rate(),
    }


def save_checkpoint(
    path: Path,
    student: SpikeBERTStudent,
    config: Config,
    epoch: int,
    f1_macro: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "f1_macro": f1_macro,
        },
        path,
    )


def load_student_checkpoint(
    path: Path,
    student: SpikeBERTStudent,
    device: torch.device,
) -> None:
    checkpoint = safe_torch_load(path, map_location=str(device))
    state_dict = extract_state_dict(checkpoint)
    incompatible = student.load_state_dict(state_dict, strict=False)

    LOGGER.info("Loaded best student checkpoint: %s", path)
    LOGGER.info("Student missing keys on reload: %s", incompatible.missing_keys)
    LOGGER.info("Student unexpected keys on reload: %s", incompatible.unexpected_keys)


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        spiking_layers=args.spiking_layers,
        output_dir=args.output_dir,
        teacher_ckpt=args.teacher_ckpt,
        model_name=args.model_name,
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        max_length=args.max_length,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        temperature_kd=args.temperature_kd,
        lambda_f=args.lambda_f,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        local_files_only=args.local_files_only,
        amp=not args.no_amp,
        spike_timesteps=args.spike_timesteps,
        spike_threshold=args.spike_threshold,
        surrogate_alpha=args.surrogate_alpha,
    )


def main(args: argparse.Namespace) -> None:
    configure_logging()

    config = build_config(args)
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint = output_dir / f"best_tier1_seed{config.seed}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Using device: %s", device)
    LOGGER.info("Config: %s", asdict(config))

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
        use_fast=False,
    )

    train_loader, dev_loader, test_loader, class_weights = make_dataloaders(
        config,
        tokenizer,
    )

    teacher = PhoBERTTeacher(config).to(device)
    student = SpikeBERTStudent(config).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    class_weights = class_weights.to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    total_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = int(config.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = make_grad_scaler(
        device,
        enabled=config.amp and device.type == "cuda",
    )

    best_dev_f1 = -1.0

    for epoch in range(1, config.epochs + 1):
        train_stats = train_one_epoch(
            config=config,
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            ce_loss_fn=ce_loss_fn,
            scaler=scaler,
            device=device,
        )

        dev_metrics = evaluate(
            config=config,
            student=student,
            loader=dev_loader,
            ce_loss_fn=ce_loss_fn,
            device=device,
            reset_firing_stats=True,
        )

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f ce=%.4f kd=%.4f feat=%.4f | "
            "dev_loss=%.4f dev_acc=%.4f dev_f1_macro=%.4f dev_f1_weighted=%.4f | firing_rate=%.4f",
            epoch,
            config.epochs,
            train_stats["loss"],
            train_stats["ce"],
            train_stats["kd"],
            train_stats["feature"],
            dev_metrics["loss"],
            dev_metrics["accuracy"],
            dev_metrics["f1_macro"],
            dev_metrics["f1_weighted"],
            dev_metrics["firing_rate"],
        )

        print(
            f"[MARKER] tier=tier1 seed={config.seed} epoch={epoch} "
            f"acc={dev_metrics['accuracy']:.4f} "
            f"f1_macro={dev_metrics['f1_macro']:.4f} "
            f"f1_weighted={dev_metrics['f1_weighted']:.4f}",
            flush=True,
        )

        if float(dev_metrics["f1_macro"]) > best_dev_f1:
            best_dev_f1 = float(dev_metrics["f1_macro"])
            save_checkpoint(best_checkpoint, student, config, epoch, best_dev_f1)
            LOGGER.info("Saved best checkpoint to %s", best_checkpoint)

    if best_checkpoint.exists():
        load_student_checkpoint(best_checkpoint, student, device)

    peak_training_memory = get_peak_training_memory_report(device)

    test_metrics = evaluate(
        config=config,
        student=student,
        loader=test_loader,
        ce_loss_fn=ce_loss_fn,
        device=device,
        reset_firing_stats=True,
    )

    f1_neg, f1_neu, f1_pos = test_metrics["f1_per_class"]

    print(
        f"[MARKER] tier=tier1 seed={config.seed} TEST "
        f"acc={test_metrics['accuracy']:.4f} "
        f"f1_macro={test_metrics['f1_macro']:.4f} "
        f"f1_neg={f1_neg:.4f} "
        f"f1_neu={f1_neu:.4f} "
        f"f1_pos={f1_pos:.4f}",
        flush=True,
    )

    energy_report = write_energy_report(
        output_dir=output_dir,
        tier_name="tier1",
        config=config,
        teacher=teacher,
        student=student,
        test_loader=test_loader,
        device=device,
        test_metrics=test_metrics,
        peak_training_memory=peak_training_memory,
    )

    theoretical = energy_report["theoretical_energy"]
    gpu_energy = energy_report["actual_gpu_energy_a100"]

    print(
        f"[MARKER] theoretical_energy tier=tier1 "
        f"student_pj_per_sample={theoretical['student_total_energy_pj_per_sample']:.2f} "
        f"teacher_pj_per_sample={theoretical['teacher_total_energy_pj_per_sample']:.2f} "
        f"teacher_student_saving={theoretical['teacher_student_theoretical_saving_ratio']:.2f}x "
        f"mean_firing_rate={theoretical['student_mean_firing_rate']:.4f}",
        flush=True,
    )

    print(
        f"[MARKER] gpu_energy tier=tier1 "
        f"student_teacher_ratio={gpu_energy['student_teacher_energy_per_sample_ratio']}",
        flush=True,
    )

    print(
        f"[MARKER] train_memory tier=tier1 "
        f"peak_mb={peak_training_memory['peak_memory_allocated_mb']}",
        flush=True,
    )


if __name__ == "__main__":
    main(parse_args())
