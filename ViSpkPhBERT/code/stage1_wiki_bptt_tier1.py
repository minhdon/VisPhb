#!/usr/bin/env python3
"""
stage1_wiki_bptt_tier1.py

Stage 1 Wiki KD for Tier-1 BPTT spiking PhoBERT student.

Purpose:
    PhoBERT pretrained teacher
    -> SpikeBERTStudent-style BPTT student
    -> representation distillation on unlabeled segmented Vietnamese Wikipedia.

Input:
    wiki_vi_20231101_segmented/
    ├── data-00000-of-00004.arrow
    ├── data-00001-of-00004.arrow
    ├── data-00002-of-00004.arrow
    ├── data-00003-of-00004.arrow
    ├── dataset_info.json
    └── state.json

Output:
    uit-models/stage1_wiki_bptt/seed_42/
    ├── best_stage1_wiki_bptt_seed42.pt
    ├── last_stage1_wiki_bptt_seed42.pt
    └── stage1_wiki_bptt_report_seed42.json

This file is intended for:
    - tier1_spikebert_2stage.py

Important:
    This checkpoint is for Tier-1 BPTT only.
    Do NOT load implicit Stage-1 checkpoints into Tier-1 BPTT unless doing
    a deliberately loose ablation with strict=False.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from torch.autograd import Function
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOGGER = logging.getLogger("stage1_wiki_bptt")

ALIGNMENT_LAYERS = [3, 6, 9, 12]
NUM_LABELS = 3


@dataclass
class Config:
    seed: int = 42

    data_dir: str = "wiki_vi_20231101_segmented"
    output_dir: str = "uit-models/stage1_wiki_bptt"
    model_name: str = "vinai/phobert-base-v2"

    epochs: int = 1
    max_steps: int = 0
    batch_size: int = 4
    eval_batch_size: int = 4
    max_samples: int = 0
    max_length: int = 128
    num_workers: int = 2

    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0

    spiking_layers: int = 2
    spike_timesteps: int = 16
    beta: float = 0.9
    threshold: float = 1.0
    surrogate_alpha: float = 2.0
    reset_mode: str = "subtract"  # subtract | zero

    lambda_embedding: float = 0.2
    lambda_hidden: float = 1.0
    lambda_last: float = 1.0

    save_every_steps: int = 500
    log_every_steps: int = 1000

    local_files_only: bool = False
    amp: bool = True
    resume: str = ""


# =========================================================
# General utilities
# =========================================================

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def infer_project_root() -> Path:
    try:
        file_path = Path(__file__).resolve()
        search_paths = [file_path.parent, *file_path.parents]
    except NameError:
        search_paths = [Path.cwd(), *Path.cwd().parents]

    for parent in search_paths:
        if (parent / "wiki_vi_20231101_segmented").exists():
            return parent
        if (parent / "wiki_vi_20231101").exists():
            return parent
        if (parent / "uit-vsfc").exists() or (parent / "cache").exists():
            return parent

    return Path.cwd()


def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)

    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    return (project_root / path).resolve()


def make_output_dir(base_output_dir: Path, seed: int) -> Path:
    seed_name = f"seed_{seed}"
    if base_output_dir.name == seed_name:
        return base_output_dir
    return base_output_dir / seed_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def get_peak_memory_report(device: torch.device) -> Dict[str, Any]:
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
    }


# =========================================================
# BPTT surrogate spike components
# =========================================================

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
    BPTT spiking FFN replacement for a Transformer FFN block.

    Original PhoBERT FFN:
        intermediate.dense: hidden -> intermediate
        activation
        output.dense: intermediate -> hidden

    Here:
        input current = linear1(hidden)
        LIF dynamics are unrolled for T steps
        average spike rate is fed to linear2
    """

    def __init__(
        self,
        linear1: nn.Linear,
        linear2: nn.Linear,
        layer_index: int,
        spike_timesteps: int = 16,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
        reset_mode: str = "subtract",
    ):
        super().__init__()

        if reset_mode not in {"subtract", "zero"}:
            raise ValueError(f"reset_mode must be 'subtract' or 'zero', got {reset_mode}")

        self.linear1 = linear1
        self.linear2 = linear2

        self.layer_index = int(layer_index)
        self.spike_timesteps = int(spike_timesteps)
        self.threshold = float(threshold)
        self.surrogate_alpha = float(surrogate_alpha)
        self.reset_mode = str(reset_mode)

        self.beta = nn.Parameter(torch.tensor(float(beta)))

        self.firing_rate = 0.0
        self.num_forward_calls = 0
        self._total_firing = 0.0
        self._last_firing_rate = 0.0
        self.global_step = 0

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    def post_step_clamp(self) -> None:
        with torch.no_grad():
            self.beta.data.clamp_(0.0, 1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        input_dtype = hidden_states.dtype

        x_flat = hidden_states.reshape(batch_size * seq_len, hidden_dim).float()

        # Compute static input current once. This is much cheaper than computing
        # linear1 at every time step and is common in rate-based BPTT SNN blocks.
        input_current = F.linear(
            x_flat,
            self.linear1.weight.float(),
            None if self.linear1.bias is None else self.linear1.bias.float(),
        )

        membrane = torch.zeros_like(input_current)
        spike_sum = torch.zeros_like(input_current)

        beta = self.beta.float().clamp(0.0, 1.0)
        threshold = self.threshold

        for _ in range(self.spike_timesteps):
            membrane = beta * membrane + input_current
            spike = ArctanSpike.apply(membrane - threshold, self.surrogate_alpha)
            spike_sum = spike_sum + spike

            if self.reset_mode == "subtract":
                membrane = membrane - spike.detach() * threshold
            else:
                membrane = membrane * (1.0 - spike.detach())

        spike_rate = spike_sum / float(max(1, self.spike_timesteps))

        current_firing = float(spike_rate.detach().mean().item())
        self.num_forward_calls += 1
        self._last_firing_rate = current_firing
        self._total_firing += current_firing
        self.firing_rate = self._total_firing / max(1, self.num_forward_calls)

        output_flat = F.linear(
            spike_rate,
            self.linear2.weight.float(),
            None if self.linear2.bias is None else self.linear2.bias.float(),
        )

        return output_flat.reshape(batch_size, seq_len, self.linear2.out_features).to(input_dtype)

    def stats(self) -> Dict[str, float]:
        return {
            "layer_index": float(self.layer_index),
            "spike_timesteps": float(self.spike_timesteps),
            "firing_rate": float(self.firing_rate),
            "last_firing_rate": float(self._last_firing_rate),
            "num_forward_calls": float(self.num_forward_calls),
            "beta": float(self.beta.detach().float().item()),
            "threshold": float(self.threshold),
        }


class IdentityIntermediate(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class SpikeBERTOutputWrapper(nn.Module):
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


# =========================================================
# Model definitions
# =========================================================

def load_phobert_backbone(config: Config) -> nn.Module:
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


class SpikeBERTStudent(nn.Module):
    """
    Tier-1 BPTT student.

    Names intentionally match Tier 1:
        encoder
        dropout
        classifier
        spiking_ffns
        feature_projectors

    This makes the Stage-1 BPTT checkpoint suitable for:
        - tier1_spikebert_2stage.py
    """

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.encoder = load_phobert_backbone(config)

        hidden_size = int(self.encoder.config.hidden_size)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)

        self.spiking_layer_indices = self._select_spiking_layers(config.spiking_layers)

        # Plain dict is intentional. Actual modules are registered through:
        #     layer.output = SpikeBERTOutputWrapper(...)
        # so they appear in state_dict through encoder.* paths.
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
                layer_index=layer_index,
                spike_timesteps=self.config.spike_timesteps,
                beta=self.config.beta,
                threshold=self.config.threshold,
                surrogate_alpha=self.config.surrogate_alpha,
                reset_mode=self.config.reset_mode,
            )

            original_output = layer.output

            layer.intermediate = IdentityIntermediate()
            layer.output = SpikeBERTOutputWrapper(
                spiking_ffn=spiking_ffn,
                original_output=original_output,
            )

            self.spiking_ffns[str(layer_index)] = spiking_ffn

        LOGGER.info("Stage-1 BPTT spiking FFN layer indices: %s", self.spiking_layer_indices)

    def set_global_step(self, global_step: int) -> None:
        for module in self.spiking_ffns.values():
            module.set_global_step(global_step)

    def post_step_clamp(self) -> None:
        for module in self.spiking_ffns.values():
            module.post_step_clamp()

    def mean_firing_rate(self) -> float:
        rates = [module.firing_rate for module in self.spiking_ffns.values()]
        if len(rates) == 0:
            return 0.0
        return float(np.mean(rates))

    def per_layer_stats(self) -> Dict[str, Dict[str, float]]:
        return {str(layer_id): module.stats() for layer_id, module in self.spiking_ffns.items()}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> Dict[str, Any]:
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


# =========================================================
# Dataset
# =========================================================

class WikiTextTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        tokenizer,
        text_column: str,
        max_length: int,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.hf_dataset[int(index)]
        text = str(row.get(self.text_column, "") or "").strip()

        if not text:
            text = " "

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
        }


def detect_text_column(dataset: Dataset) -> str:
    candidates = ["text", "content", "sentence", "document"]

    for col in candidates:
        if col in dataset.column_names:
            return col

    raise ValueError(
        f"Could not detect text column. Available columns: {dataset.column_names}. "
        f"Expected one of: {candidates}"
    )


def load_local_wiki_dataset(config: Config, project_root: Path) -> Tuple[Dataset, str, Path]:
    data_dir = resolve_path(config.data_dir, project_root)

    if not data_dir.exists():
        raise FileNotFoundError(f"Wiki dataset folder not found: {data_dir}")

    if not (data_dir / "state.json").exists():
        raise FileNotFoundError(f"Missing state.json in dataset folder: {data_dir}")

    if not (data_dir / "dataset_info.json").exists():
        raise FileNotFoundError(f"Missing dataset_info.json in dataset folder: {data_dir}")

    dataset_obj = load_from_disk(str(data_dir))

    if isinstance(dataset_obj, DatasetDict):
        datasets = [split_dataset for split_dataset in dataset_obj.values()]
        dataset = concatenate_datasets(datasets)
    elif isinstance(dataset_obj, Dataset):
        dataset = dataset_obj
    else:
        raise TypeError(f"Unsupported dataset object from load_from_disk: {type(dataset_obj)}")

    text_column = detect_text_column(dataset)

    if config.max_samples and config.max_samples > 0:
        max_samples = min(int(config.max_samples), len(dataset))
        dataset = dataset.shuffle(seed=config.seed)
        dataset = dataset.select(range(max_samples))

    LOGGER.info("Loaded segmented wiki dataset from %s", data_dir)
    LOGGER.info("Dataset size: %d", len(dataset))
    LOGGER.info("Columns: %s", dataset.column_names)
    LOGGER.info("Text column: %s", text_column)

    return dataset, text_column, data_dir


def estimate_underscore_ratio(dataset: Dataset, text_column: str, n: int = 1000) -> Dict[str, float]:
    n = min(int(n), len(dataset))
    token_count = 0
    underscore_count = 0

    for i in range(n):
        text = str(dataset[i][text_column] or "")
        tokens = text.split()
        token_count += len(tokens)
        underscore_count += sum(1 for tok in tokens if "_" in tok)

    ratio = underscore_count / max(token_count, 1)

    return {
        "samples": float(n),
        "tokens": float(token_count),
        "underscore_tokens": float(underscore_count),
        "underscore_ratio": float(ratio),
    }


def make_dataloader(
    config: Config,
    dataset: Dataset,
    tokenizer,
    text_column: str,
) -> DataLoader:
    torch_dataset = WikiTextTorchDataset(
        hf_dataset=dataset,
        tokenizer=tokenizer,
        text_column=text_column,
        max_length=config.max_length,
    )

    cuda_available = torch.cuda.is_available()
    persistent_workers = config.num_workers > 0 and os.name != "nt"

    return DataLoader(
        torch_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=cuda_available,
        persistent_workers=persistent_workers,
    )


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


# =========================================================
# Loss
# =========================================================

def masked_mse(
    student_tensor: torch.Tensor,
    teacher_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    student_tensor = student_tensor.float()
    teacher_tensor = teacher_tensor.detach().float()

    mask = attention_mask.float().unsqueeze(-1)
    squared = (student_tensor - teacher_tensor).pow(2) * mask

    denom = mask.sum() * student_tensor.size(-1)
    denom = denom.clamp_min(1.0)

    return squared.sum() / denom


def compute_stage1_losses(
    student: SpikeBERTStudent,
    student_outputs: Dict[str, Any],
    teacher_outputs,
    attention_mask: torch.Tensor,
    config: Config,
) -> Dict[str, torch.Tensor]:
    student_hidden = student_outputs["hidden_states"]
    teacher_hidden = teacher_outputs.hidden_states

    if student_hidden is None or teacher_hidden is None:
        raise RuntimeError("Stage-1 KD requires output_hidden_states=True for both teacher and student.")

    embedding_loss = masked_mse(
        student_hidden[0],
        teacher_hidden[0],
        attention_mask,
    )

    hidden_losses: List[torch.Tensor] = []

    for layer in ALIGNMENT_LAYERS:
        if layer >= len(student_hidden) or layer >= len(teacher_hidden):
            continue

        if str(layer) in student.feature_projectors:
            projected_student = student.feature_projectors[str(layer)](student_hidden[layer].float())
        else:
            projected_student = student_hidden[layer].float()

        hidden_losses.append(
            masked_mse(projected_student, teacher_hidden[layer], attention_mask)
        )

    if hidden_losses:
        hidden_loss = torch.stack(hidden_losses).mean()
    else:
        hidden_loss = torch.zeros((), device=attention_mask.device)

    last_loss = masked_mse(
        student_outputs["last_hidden_state"],
        teacher_outputs.last_hidden_state,
        attention_mask,
    )

    total_loss = (
        config.lambda_embedding * embedding_loss
        + config.lambda_hidden * hidden_loss
        + config.lambda_last * last_loss
    )

    return {
        "loss": total_loss,
        "embedding": embedding_loss,
        "hidden": hidden_loss,
        "last": last_loss,
    }


# =========================================================
# Checkpointing
# =========================================================

def save_checkpoint(
    path: Path,
    student: SpikeBERTStudent,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    config: Config,
    epoch: int,
    global_step: int,
    best_loss: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    state = student.state_dict()

    payload = {
        "stage": "stage1_wiki_bptt",
        "student_arch": "spikebert_bptt_phobert_ffn",
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": asdict(config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "extra": extra or {},
    }

    torch.save(payload, path)


def load_resume_checkpoint(
    path: Path,
    student: SpikeBERTStudent,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> Tuple[int, int, float]:
    checkpoint = safe_torch_load(path, map_location=device)

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("student_state_dict")
    if state_dict is None:
        raise ValueError(f"No model_state_dict/student_state_dict in checkpoint: {path}")

    incompatible = student.load_state_dict(state_dict, strict=False)

    LOGGER.info("Resumed model from %s", path)
    LOGGER.info("Missing keys: %s", incompatible.missing_keys)
    LOGGER.info("Unexpected keys: %s", incompatible.unexpected_keys)

    if len(incompatible.missing_keys) > 10 or len(incompatible.unexpected_keys) > 10:
        LOGGER.warning(
            "[RESUME_STAGE1] Many missing/unexpected keys (%d missing, %d unexpected). "
            "Architecture may have changed since this checkpoint was saved.",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_loss = float(checkpoint.get("best_loss", float("inf")))

    return epoch, global_step, best_loss


def write_report(
    output_dir: Path,
    config: Config,
    data_dir: Path,
    dataset_size: int,
    text_column: str,
    underscore_stats: Dict[str, float],
    best_loss: float,
    global_step: int,
    student: SpikeBERTStudent,
    device: torch.device,
    best_checkpoint: Path,
    last_checkpoint: Path,
) -> None:
    report = {
        "stage": "stage1_wiki_bptt",
        "student_arch": "spikebert_bptt_phobert_ffn",
        "seed": int(config.seed),
        "data_dir": str(data_dir),
        "dataset_size": int(dataset_size),
        "text_column": text_column,
        "segmentation_check": underscore_stats,
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "model_name": config.model_name,
        "spiking_layers": int(config.spiking_layers),
        "spike_timesteps": int(config.spike_timesteps),
        "student_mean_firing_rate": float(student.mean_firing_rate()),
        "per_layer_stats": student.per_layer_stats(),
        "training_memory": get_peak_memory_report(device),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "config": asdict(config),
        "note": (
            "This is Stage-1 Wiki representation KD for Tier-1 BPTT SpikeBERTStudent. "
            "Teacher is pretrained PhoBERT and is frozen. Student is PhoBERT with selected "
            "FFN layers replaced by BPTT spiking FFN modules. This checkpoint is intended "
            "for tier1_spikebert_2stage.py only."
        ),
    }

    report_path = output_dir / f"stage1_wiki_bptt_report_seed{config.seed}.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved report to %s", report_path)


# =========================================================
# Training
# =========================================================

def train_stage1(
    config: Config,
    teacher: nn.Module,
    student: SpikeBERTStudent,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    output_dir: Path,
    start_epoch: int = 0,
    start_global_step: int = 0,
    initial_best_loss: float = float("inf"),
) -> Tuple[float, int, Path, Path]:
    teacher.eval()

    use_amp = config.amp and device.type == "cuda"

    best_loss = float(initial_best_loss)
    global_step = int(start_global_step)

    best_checkpoint = output_dir / f"best_stage1_wiki_bptt_seed{config.seed}.pt"
    last_checkpoint = output_dir / f"last_stage1_wiki_bptt_seed{config.seed}.pt"

    for epoch in range(start_epoch + 1, config.epochs + 1):
        student.train()

        running_loss = 0.0
        running_embedding = 0.0
        running_hidden = 0.0
        running_last = 0.0
        running_count = 0

        epoch_start = time.perf_counter()

        for batch_idx, batch in enumerate(loader, start=1):
            if config.max_steps > 0 and global_step >= config.max_steps:
                break

            student.set_global_step(global_step)

            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            # Teacher target is FP32 for stable hidden-state KD targets.
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )

            # Student BPTT may use AMP, but internal spike simulation casts to FP32.
            with amp_autocast(device, use_amp):
                student_outputs = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )

                losses = compute_stage1_losses(
                    student=student,
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    attention_mask=batch["attention_mask"],
                    config=config,
                )

                loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            student.post_step_clamp()

            global_step += 1
            scheduler.step()

            loss_value = float(loss.detach().item())
            emb_value = float(losses["embedding"].detach().item())
            hidden_value = float(losses["hidden"].detach().item())
            last_value = float(losses["last"].detach().item())

            running_loss += loss_value
            running_embedding += emb_value
            running_hidden += hidden_value
            running_last += last_value
            running_count += 1

            if config.log_every_steps > 0 and global_step % config.log_every_steps == 0:
                print(
                    f"[MARKER] stage1_wiki_bptt seed={config.seed} "
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss_value:.6f} "
                    f"emb={emb_value:.6f} "
                    f"hidden={hidden_value:.6f} "
                    f"last={last_value:.6f} "
                    f"firing_rate={student.mean_firing_rate():.4f}",
                    flush=True,
                )

            if config.save_every_steps > 0 and global_step % config.save_every_steps == 0:
                save_checkpoint(
                    path=last_checkpoint,
                    student=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                    extra={
                        "reason": "periodic_last",
                        "current_batch_loss": loss_value,
                    },
                )
                LOGGER.info("Saved periodic checkpoint to %s", last_checkpoint)

        if running_count == 0:
            LOGGER.info("No training batches processed in epoch %d. Stopping.", epoch)
            break

        elapsed = time.perf_counter() - epoch_start

        epoch_loss = running_loss / running_count
        epoch_embedding = running_embedding / running_count
        epoch_hidden = running_hidden / running_count
        epoch_last = running_last / running_count

        is_best_epoch = epoch_loss < best_loss
        if is_best_epoch:
            best_loss = epoch_loss

        save_checkpoint(
            path=last_checkpoint,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            extra={
                "reason": "epoch_end_last",
                "epoch_avg_loss": epoch_loss,
                "epoch_avg_embedding_loss": epoch_embedding,
                "epoch_avg_hidden_loss": epoch_hidden,
                "epoch_avg_last_loss": epoch_last,
            },
        )

        if is_best_epoch:
            save_checkpoint(
                path=best_checkpoint,
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                extra={
                    "reason": "best_epoch_avg_loss",
                    "epoch_avg_loss": epoch_loss,
                    "epoch_avg_embedding_loss": epoch_embedding,
                    "epoch_avg_hidden_loss": epoch_hidden,
                    "epoch_avg_last_loss": epoch_last,
                },
            )

            LOGGER.info(
                "Saved best checkpoint by epoch average loss %.6f to %s",
                best_loss,
                best_checkpoint,
            )

        print(
            f"[MARKER] stage1_wiki_bptt seed={config.seed} epoch={epoch} "
            f"avg_loss={epoch_loss:.6f} "
            f"avg_emb={epoch_embedding:.6f} "
            f"avg_hidden={epoch_hidden:.6f} "
            f"avg_last={epoch_last:.6f} "
            f"best_loss={best_loss:.6f} "
            f"steps={global_step} "
            f"firing_rate={student.mean_firing_rate():.4f} "
            f"elapsed_sec={elapsed:.2f}",
            flush=True,
        )

        if config.max_steps > 0 and global_step >= config.max_steps:
            LOGGER.info("Reached max_steps=%d. Stopping.", config.max_steps)
            break

    return best_loss, global_step, best_checkpoint, last_checkpoint


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-1 Wiki representation KD for Tier-1 BPTT spiking PhoBERT."
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_dir", type=str, default="wiki_vi_20231101_segmented")
    parser.add_argument("--output_dir", type=str, default="uit-models/stage1_wiki_bptt")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--spiking_layers", type=int, default=2)
    parser.add_argument("--spike_timesteps", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--surrogate_alpha", type=float, default=2.0)
    parser.add_argument("--reset_mode", type=str, default="subtract", choices=["subtract", "zero"])

    parser.add_argument("--lambda_embedding", type=float, default=0.2)
    parser.add_argument("--lambda_hidden", type=float, default=1.0)
    parser.add_argument("--lambda_last", type=float, default=1.0)

    parser.add_argument("--save_every_steps", type=int, default=500)
    parser.add_argument("--log_every_steps", type=int, default=1000)

    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_samples=args.max_samples,
        max_length=args.max_length,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        spiking_layers=args.spiking_layers,
        spike_timesteps=args.spike_timesteps,
        beta=args.beta,
        threshold=args.threshold,
        surrogate_alpha=args.surrogate_alpha,
        reset_mode=args.reset_mode,
        lambda_embedding=args.lambda_embedding,
        lambda_hidden=args.lambda_hidden,
        lambda_last=args.lambda_last,
        save_every_steps=args.save_every_steps,
        log_every_steps=args.log_every_steps,
        local_files_only=args.local_files_only,
        amp=not args.no_amp,
        resume=args.resume,
    )


def main(args: argparse.Namespace) -> None:
    configure_logging()

    config = build_config(args)
    set_seed(config.seed)

    project_root = infer_project_root()

    data_dir = resolve_path(config.data_dir, project_root)
    base_output_dir = resolve_path(config.output_dir, project_root)
    output_dir = make_output_dir(base_output_dir, config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    config.data_dir = str(data_dir)
    config.output_dir = str(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Config: %s", asdict(config))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset, text_column, resolved_data_dir = load_local_wiki_dataset(config, project_root)

    underscore_stats = estimate_underscore_ratio(dataset, text_column, n=1000)
    LOGGER.info("Segmentation underscore stats: %s", underscore_stats)

    if underscore_stats["underscore_ratio"] < 0.01:
        LOGGER.warning(
            "underscore_ratio is very low. Dataset may not be word-segmented. "
            "PhoBERT usually expects segmented Vietnamese text."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
        use_fast=False,
    )

    loader = make_dataloader(
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
        text_column=text_column,
    )

    teacher = load_phobert_backbone(config).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = SpikeBERTStudent(config).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if config.max_steps > 0:
        total_steps = int(config.max_steps)
    else:
        total_steps = max(1, len(loader) * config.epochs)

    warmup_steps = int(config.warmup_ratio * total_steps)

    LOGGER.info("Total steps=%d | warmup steps=%d", total_steps, warmup_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = make_grad_scaler(
        device=device,
        enabled=config.amp and device.type == "cuda",
    )

    start_epoch = 0
    start_global_step = 0
    best_loss = float("inf")

    if config.resume:
        resume_path = resolve_path(config.resume, project_root)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        start_epoch, start_global_step, best_loss = load_resume_checkpoint(
            path=resume_path,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    best_loss, global_step, best_checkpoint, last_checkpoint = train_stage1(
        config=config,
        teacher=teacher,
        student=student,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        output_dir=output_dir,
        start_epoch=start_epoch,
        start_global_step=start_global_step,
        initial_best_loss=best_loss,
    )

    write_report(
        output_dir=output_dir,
        config=config,
        data_dir=resolved_data_dir,
        dataset_size=len(dataset),
        text_column=text_column,
        underscore_stats=underscore_stats,
        best_loss=best_loss,
        global_step=global_step,
        student=student,
        device=device,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
    )

    print(
        f"[MARKER] stage1_wiki_bptt DONE "
        f"seed={config.seed} "
        f"best_loss={best_loss:.6f} "
        f"global_step={global_step} "
        f"firing_rate={student.mean_firing_rate():.4f} "
        f"best_ckpt={best_checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main(parse_args())
