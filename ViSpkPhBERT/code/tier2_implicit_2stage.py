#!/usr/bin/env python3
"""
tier2_implicit_2stage.py

Tier 2 full two-stage implicit PhoBERT-SNN training.

Pipeline:
    Stage 1:
        Already trained by stage1_wiki_implicit.py
        PhoBERT pretrained teacher -> ImplicitPhoBERTStudent on segmented Wiki.

    Stage 2A:
        Task-based internal KD on UIT-VSFC:
            CE + KD logits + feature alignment

    Stage 2B:
        Prediction-layer distillation / final tuning:
            CE + KD logits, with feature loss reduced or disabled.

This file is intended to be a Vietnamese SpikingBERT-inspired full two-stage
implicit PhoBERT-SNN KD pipeline.

Important:
    This is NOT a full absolute reproduction of SpikingBERT because attention
    remains ANN and only selected FFN blocks are replaced by implicit SNN modules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOGGER = logging.getLogger("tier2_implicit_2stage")

ALIGNMENT_LAYERS = [3, 6, 9, 12]
NUM_LABELS = 3
LABEL_NAMES = ["negative", "neutral", "positive"]


# =========================================================
# Import Stage-1 implicit module
# =========================================================

def import_stage1_module():
    module_name = "stage1_wiki_implicit_runtime"

    existing = sys.modules.get(module_name)
    if existing is not None and hasattr(existing, "Config") and hasattr(existing, "ImplicitPhoBERTStudent"):
        return existing

    if existing is not None:
        sys.modules.pop(module_name, None)

    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / "stage1_wiki_implicit.py"

    if not module_path.exists():
        raise FileNotFoundError(
            f"Cannot find stage1_wiki_implicit.py at {module_path}. "
            f"Put tier2_implicit_2stage.py and stage1_wiki_implicit.py in the same folder."
        )

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import stage1_wiki_implicit.py from {module_path}")

    module = importlib.util.module_from_spec(spec)

    # Register before exec_module because dataclasses/runtime introspection may
    # expect sys.modules[__name__] to exist during module execution.
    # Roll back if execution fails to avoid a half-initialized module.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise

    return module


_STAGE1 = import_stage1_module()

Stage1Config = _STAGE1.Config
ImplicitPhoBERTStudent = _STAGE1.ImplicitPhoBERTStudent
safe_torch_load = _STAGE1.safe_torch_load
amp_autocast = _STAGE1.amp_autocast
make_grad_scaler = _STAGE1.make_grad_scaler
get_peak_memory_report = _STAGE1.get_peak_memory_report


# =========================================================
# Config
# =========================================================

@dataclass
class Config:
    seed: int = 42

    model_name: str = "vinai/phobert-base-v2"
    teacher_ckpt: str = "uit-models/phobert_vsfc/best_model.pth"
    stage1_ckpt: str = ""

    train_path: str = "cache/train_segmented.parquet"
    dev_path: str = "cache/dev_segmented.parquet"
    test_path: str = "cache/test_segmented.parquet"
    output_dir: str = "uit-models/tier2_implicit_2stage"

    text_column: str = "text"
    label_column: str = "label"

    task_kd_epochs: int = 15
    prediction_kd_epochs: int = 5
    max_steps: int = 0

    batch_size: int = 8
    eval_batch_size: int = 16
    max_length: int = 256
    num_workers: int = 2

    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0

    temperature_kd: float = 4.0
    alpha_ce: float = 0.5
    lambda_f: float = 0.1

    alpha_ce_final: float = 0.5
    lambda_f_final: float = 0.0

    spiking_layers: int = 4
    anderson_max_iter: int = 30
    anderson_tol: float = 1e-3
    anderson_m: int = 5
    neumann_k: int = 5
    threshold: float = 0.5
    surrogate_alpha: float = 2.0

    equilibrium_metric: str = "both"
    firing_rate_tol: float = 1e-4
    min_equilibrium_iter: int = 3

    save_every_steps: int = 0
    log_every_steps: int = 1000

    local_files_only: bool = False
    amp: bool = True
    resume: str = ""
    load_stage1_strict: bool = False

    measure_gpu_energy: bool = False
    max_energy_batches: int = 0


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
        if (parent / "cache").exists() and (parent / "uit-vsfc").exists():
            return parent
        if (parent / "uit-train").exists() and (parent / "uit-models").exists():
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


def make_default_stage1_ckpt(project_root: Path, config: Config) -> Path:
    return (
        project_root
        / "uit-models"
        / "stage1_wiki_implicit"
        / f"spk{int(config.spiking_layers)}"
        / f"seed_{int(config.seed)}"
        / f"best_stage1_wiki_implicit_seed{int(config.seed)}.pt"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_tensor_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "student_state_dict",
            "teacher_state_dict",
            "state_dict",
            "model",
            "net",
        ]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                filtered = {k: v for k, v in value.items() if torch.is_tensor(v)}
                if filtered:
                    return filtered

        filtered = {k: v for k, v in checkpoint.items() if isinstance(k, str) and torch.is_tensor(v)}
        if filtered:
            return filtered

    raise ValueError("Cannot extract tensor state_dict from checkpoint.")


def strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ["module.", "model.", "student.", "teacher."]

    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        out[new_key] = value

    return out


def get_checkpoint_config_value(checkpoint: Any, key: str) -> Optional[Any]:
    if not isinstance(checkpoint, dict):
        return None

    ckpt_config = checkpoint.get("config")

    if isinstance(ckpt_config, dict):
        return ckpt_config.get(key)

    if hasattr(ckpt_config, key):
        return getattr(ckpt_config, key)

    return None


def validate_stage1_checkpoint_config(checkpoint: Any, config: Config, ckpt_path: Path) -> None:
    if isinstance(checkpoint, dict):
        stage = checkpoint.get("stage")

        if stage is not None and stage != "stage1_wiki_implicit":
            raise ValueError(
                "Stage-1 checkpoint stage does not match implicit Tier-2/Tier-3 training. "
                f"checkpoint={ckpt_path} stage={stage!r}"
            )

    raw_spiking_layers = get_checkpoint_config_value(checkpoint, "spiking_layers")

    if raw_spiking_layers is None:
        LOGGER.warning(
            "[STAGE1 LOAD] checkpoint=%s has no config.spiking_layers metadata; "
            "cannot verify it before state_dict loading.",
            ckpt_path,
        )
        return

    checkpoint_layers = int(raw_spiking_layers)
    requested_layers = int(config.spiking_layers)

    if checkpoint_layers != requested_layers:
        raise ValueError(
            "Stage-1 checkpoint architecture does not match requested config. "
            f"checkpoint={ckpt_path} spiking_layers: checkpoint={checkpoint_layers}, "
            f"requested={requested_layers}"
        )


def build_stage1_config(config: Config) -> Stage1Config:
    stage_cfg = Stage1Config()

    stage_cfg.seed = config.seed
    stage_cfg.model_name = config.model_name
    stage_cfg.local_files_only = config.local_files_only

    stage_cfg.max_length = config.max_length

    stage_cfg.spiking_layers = config.spiking_layers
    stage_cfg.anderson_max_iter = config.anderson_max_iter
    stage_cfg.anderson_tol = config.anderson_tol
    stage_cfg.anderson_m = config.anderson_m
    stage_cfg.neumann_k = config.neumann_k
    stage_cfg.threshold = config.threshold
    stage_cfg.surrogate_alpha = config.surrogate_alpha

    if hasattr(stage_cfg, "equilibrium_metric"):
        stage_cfg.equilibrium_metric = config.equilibrium_metric
    if hasattr(stage_cfg, "firing_rate_tol"):
        stage_cfg.firing_rate_tol = config.firing_rate_tol
    if hasattr(stage_cfg, "min_equilibrium_iter"):
        stage_cfg.min_equilibrium_iter = config.min_equilibrium_iter

    return stage_cfg


# =========================================================
# Dataset
# =========================================================

def normalize_label(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        ivalue = int(value)
        if ivalue in {0, 1, 2}:
            return ivalue
        if ivalue == -1:
            return 0
        raise ValueError(f"Unsupported integer label: {ivalue}")

    text = str(value).strip().lower()

    mapping = {
        "negative": 0,
        "neg": 0,
        "tiêu cực": 0,
        "tieu cuc": 0,
        "0": 0,
        "-1": 0,

        "neutral": 1,
        "neu": 1,
        "trung tính": 1,
        "trung tinh": 1,
        "1": 1,

        "positive": 2,
        "pos": 2,
        "tích cực": 2,
        "tich cuc": 2,
        "2": 2,
    }

    if text not in mapping:
        raise ValueError(f"Unsupported label value: {value!r}")

    return mapping[text]


class ParquetSentimentDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        text_column: str,
        label_column: str,
        max_length: int,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        self.df = pd.read_parquet(path)

        if text_column not in self.df.columns:
            raise ValueError(f"Missing text column {text_column} in {path}. Columns={list(self.df.columns)}")
        if label_column not in self.df.columns:
            raise ValueError(f"Missing label column {label_column} in {path}. Columns={list(self.df.columns)}")

        self.texts = self.df[text_column].fillna("").astype(str).tolist()
        self.labels = [normalize_label(x) for x in self.df[label_column].tolist()]

        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = self.texts[int(index)].strip()
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
            "labels": torch.tensor(self.labels[int(index)], dtype=torch.long),
        }


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    cuda_available = torch.cuda.is_available()
    persistent_workers = num_workers > 0 and os.name != "nt"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=cuda_available,
        persistent_workers=persistent_workers,
    )


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


# =========================================================
# Teacher / Student loading
# =========================================================

def map_teacher_keys_for_auto_model(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Try to map common custom PhoBERT teacher keys to AutoModelForSequenceClassification keys.

    AutoModelForSequenceClassification for PhoBERT/RoBERTa usually expects:
        roberta.*
        classifier.dense.*
        classifier.out_proj.*

    Some custom checkpoints may contain:
        encoder.*
        phobert.*
        bert.*
        classifier.weight / classifier.bias
    """
    state_dict = strip_common_prefixes(state_dict)

    out: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("encoder."):
            new_key = "roberta." + new_key[len("encoder."):]
        elif new_key.startswith("phobert."):
            new_key = "roberta." + new_key[len("phobert."):]
        elif new_key.startswith("bert."):
            new_key = "roberta." + new_key[len("bert."):]
        elif new_key.startswith("base_model."):
            new_key = "roberta." + new_key[len("base_model."):]

        # Simple linear classifier -> RobertaClassificationHead out_proj
        if new_key == "classifier.weight":
            new_key = "classifier.out_proj.weight"
        elif new_key == "classifier.bias":
            new_key = "classifier.out_proj.bias"

        out[new_key] = value

    return out


def load_teacher(config: Config, device: torch.device, project_root: Path):
    teacher = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=NUM_LABELS,
        local_files_only=config.local_files_only,
    )

    ckpt_path = resolve_path(config.teacher_ckpt, project_root)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

    checkpoint = safe_torch_load(ckpt_path, map_location="cpu")
    raw_state = extract_tensor_state_dict(checkpoint)
    mapped_state = map_teacher_keys_for_auto_model(raw_state)

    incompatible = teacher.load_state_dict(mapped_state, strict=False)

    LOGGER.info("[TEACHER LOAD] checkpoint=%s", ckpt_path)
    LOGGER.info("[TEACHER LOAD] missing_keys=%s", incompatible.missing_keys)
    LOGGER.info("[TEACHER LOAD] unexpected_keys=%s", incompatible.unexpected_keys)

    critical_missing = [
        key for key in incompatible.missing_keys
        if key.startswith("classifier") or key.startswith("roberta.embeddings")
    ]
    if critical_missing:
        LOGGER.warning("[TEACHER LOAD] Critical missing keys detected: %s", critical_missing)

    teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad_(False)

    return teacher


def load_student(config: Config, device: torch.device, project_root: Path):
    stage_cfg = build_stage1_config(config)
    student = ImplicitPhoBERTStudent(stage_cfg)

    stage1_ckpt = resolve_path(config.stage1_ckpt, project_root)

    if stage1_ckpt.exists():
        checkpoint = safe_torch_load(stage1_ckpt, map_location="cpu")
        validate_stage1_checkpoint_config(checkpoint, config, stage1_ckpt)

        raw_state = extract_tensor_state_dict(checkpoint)
        raw_state = strip_common_prefixes(raw_state)

        incompatible = student.load_state_dict(raw_state, strict=config.load_stage1_strict)

        LOGGER.info("[STAGE1 LOAD] checkpoint=%s", stage1_ckpt)
        LOGGER.info("[STAGE1 LOAD] strict=%s", config.load_stage1_strict)
        LOGGER.info("[STAGE1 LOAD] missing_keys=%s", incompatible.missing_keys)
        LOGGER.info("[STAGE1 LOAD] unexpected_keys=%s", incompatible.unexpected_keys)

        if len(incompatible.missing_keys) > 30 or len(incompatible.unexpected_keys) > 30:
            LOGGER.warning(
                "[STAGE1 LOAD] Many missing/unexpected keys. "
                "Check spiking_layers and architecture compatibility."
            )
    else:
        raise FileNotFoundError(f"Stage-1 implicit checkpoint not found: {stage1_ckpt}")

    student.to(device)
    return student


# =========================================================
# Loss / metrics
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


def compute_feature_loss(
    student,
    student_hidden,
    teacher_hidden,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if student_hidden is None or teacher_hidden is None:
        raise RuntimeError("Feature loss requires output_hidden_states=True.")

    losses: List[torch.Tensor] = []

    for layer in ALIGNMENT_LAYERS:
        if layer >= len(student_hidden) or layer >= len(teacher_hidden):
            continue

        s = student_hidden[layer]
        t = teacher_hidden[layer]

        if hasattr(student, "feature_projectors") and str(layer) in student.feature_projectors:
            s = student.feature_projectors[str(layer)](s.float())

        losses.append(masked_mse(s, t, attention_mask))

    if not losses:
        return torch.zeros((), device=attention_mask.device)

    return torch.stack(losses).mean()


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temperature = float(temperature)

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float().detach() / temperature, dim=-1)

    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def compute_loss(
    config: Config,
    phase: str,
    student,
    student_outputs: Dict[str, Any],
    teacher_outputs,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    labels = batch["labels"]

    student_logits = student_outputs["logits"]
    teacher_logits = teacher_outputs.logits

    ce_loss = F.cross_entropy(student_logits.float(), labels)
    kd_loss = compute_kd_loss(student_logits, teacher_logits, config.temperature_kd)

    if phase == "task_ikd":
        lambda_f = config.lambda_f
        alpha_ce = config.alpha_ce
    elif phase == "prediction_kd":
        lambda_f = config.lambda_f_final
        alpha_ce = config.alpha_ce_final
    else:
        raise ValueError(f"Unknown phase: {phase}")

    feature_loss = compute_feature_loss(
        student=student,
        student_hidden=student_outputs["hidden_states"],
        teacher_hidden=teacher_outputs.hidden_states,
        attention_mask=batch["attention_mask"],
    )

    loss = (
        alpha_ce * ce_loss
        + (1.0 - alpha_ce) * kd_loss
        + lambda_f * feature_loss
    )

    return {
        "loss": loss,
        "ce": ce_loss,
        "kd": kd_loss,
        "feature": feature_loss,
    }


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, Any]:
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)

    report = classification_report(
        labels,
        preds,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
        "classification_report": report,
    }


# =========================================================
# Energy reporting
# =========================================================

def get_layer_firing_rates(student) -> Dict[str, float]:
    if hasattr(student, "per_layer_stats"):
        stats = student.per_layer_stats()
        return {
            str(layer): float(values.get("firing_rate", 0.0))
            for layer, values in stats.items()
        }

    return {}


def compute_theoretical_energy_report(student, config: Config) -> Dict[str, Any]:
    """
    Hybrid theoretical energy estimate.

    Assumptions:
        - Attention layers remain ANN and use MAC cost.
        - Non-spiking FFN layers use MAC cost.
        - Implicit SNN FFN layers use AC cost scaled by measured firing rate
          and average Anderson/equilibrium steps.
        - This is a theoretical estimate, not measured GPU energy.
    """
    mac_pj = 4.6
    ac_pj = 0.9

    hidden = int(student.encoder.config.hidden_size)
    intermediate = int(student.encoder.config.intermediate_size)
    num_layers = int(student.encoder.config.num_hidden_layers)
    seq_len = int(config.max_length)

    spiking_indices = set(int(x) for x in getattr(student, "implicit_layer_indices", []))
    per_layer_stats = student.per_layer_stats() if hasattr(student, "per_layer_stats") else {}

    attention_flops = seq_len * hidden * hidden * 4
    ffn_flops = seq_len * ((hidden * intermediate) + (intermediate * hidden))
    classifier_flops = hidden * NUM_LABELS

    layers = []
    total_energy = 0.0

    for layer_idx in range(num_layers):
        att_energy = attention_flops * mac_pj
        total_energy += att_energy

        layers.append(
            {
                "name": f"layer_{layer_idx}_self_attention_ANN",
                "type": "ANN_attention_MAC",
                "flops_equiv": float(attention_flops),
                "energy_pj_per_sample": float(att_energy),
            }
        )

        if layer_idx in spiking_indices:
            stat = per_layer_stats.get(str(layer_idx), {})
            firing_rate = float(stat.get("firing_rate", 0.0))
            avg_steps = float(stat.get("avg_anderson_steps", config.anderson_max_iter))
            ffn_energy = avg_steps * firing_rate * ffn_flops * ac_pj

            layers.append(
                {
                    "name": f"layer_{layer_idx}_ffn_implicit_SNN",
                    "type": "SNN_FFN_AC",
                    "flops_equiv": float(ffn_flops),
                    "avg_equilibrium_steps": float(avg_steps),
                    "firing_rate": float(firing_rate),
                    "energy_pj_per_sample": float(ffn_energy),
                }
            )
        else:
            ffn_energy = ffn_flops * mac_pj

            layers.append(
                {
                    "name": f"layer_{layer_idx}_ffn_ANN",
                    "type": "ANN_FFN_MAC",
                    "flops_equiv": float(ffn_flops),
                    "energy_pj_per_sample": float(ffn_energy),
                }
            )

        total_energy += ffn_energy

    classifier_energy = classifier_flops * mac_pj
    total_energy += classifier_energy

    layers.append(
        {
            "name": "classifier_head",
            "type": "ANN_classifier_MAC",
            "flops_equiv": float(classifier_flops),
            "energy_pj_per_sample": float(classifier_energy),
        }
    )

    # A pure ANN teacher-style estimate for the same attention+FFN blocks.
    teacher_total = num_layers * (attention_flops + ffn_flops) * mac_pj + classifier_energy
    saving_ratio = teacher_total / max(total_energy, 1e-12)

    return {
        "note": (
            "Theoretical energy estimate in pJ/sample. Attention remains ANN and uses MAC cost. "
            "Only implicit SNN FFN layers use AC cost scaled by measured firing rate and average "
            "equilibrium steps. This is not actual GPU energy."
        ),
        "mac_pj": mac_pj,
        "ac_pj": ac_pj,
        "seq_len": seq_len,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_layers": num_layers,
        "spiking_layers": sorted(list(spiking_indices)),
        "student_total_energy_pj_per_sample": float(total_energy),
        "teacher_equiv_total_energy_pj_per_sample": float(teacher_total),
        "teacher_student_theoretical_saving_ratio": float(saving_ratio),
        "layers": layers,
    }


def measure_gpu_energy_best_effort(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    kind: str,
) -> Dict[str, Any]:
    if device.type != "cuda":
        return {"available": False, "reason": "CUDA is not available", "kind": kind}

    try:
        import pynvml
    except Exception as exc:
        return {"available": False, "reason": f"pynvml import failed: {repr(exc)}", "kind": kind}

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as exc:
        return {"available": False, "reason": f"NVML init failed: {repr(exc)}", "kind": kind}

    model.eval()

    powers: List[float] = []
    num_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break

            batch = move_batch_to_device(batch, device)

            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                powers.append(power_mw / 1000.0)
            except Exception:
                pass

            _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=False,
                return_dict=True,
            )

            num_samples += int(batch["input_ids"].size(0))

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass

    if not powers or num_samples <= 0:
        return {"available": False, "reason": "No power samples collected", "kind": kind}

    avg_power_w = float(np.mean(powers))
    energy_j = avg_power_w * elapsed

    return {
        "available": True,
        "kind": kind,
        "avg_power_w": avg_power_w,
        "elapsed_sec": float(elapsed),
        "num_samples": int(num_samples),
        "energy_joules": float(energy_j),
        "energy_per_sample_joules": float(energy_j / max(num_samples, 1)),
        "num_power_samples": int(len(powers)),
        "note": "Best-effort NVML GPU energy; noisy under shared/MPS execution.",
    }


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate(
    config: Config,
    student,
    teacher,
    loader: DataLoader,
    device: torch.device,
    phase: str,
) -> Dict[str, Any]:
    student.eval()
    teacher.eval()

    if hasattr(student, "set_global_step"):
        student.set_global_step(0)

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feature = 0.0
    n_batches = 0

    all_labels: List[int] = []
    all_preds: List[int] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        teacher_outputs = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

        student_outputs = student(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )

        losses = compute_loss(
            config=config,
            phase=phase,
            student=student,
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            batch=batch,
        )

        logits = student_outputs["logits"].float()
        preds = torch.argmax(logits, dim=-1)

        all_labels.extend(batch["labels"].detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        total_loss += float(losses["loss"].detach().item())
        total_ce += float(losses["ce"].detach().item())
        total_kd += float(losses["kd"].detach().item())
        total_feature += float(losses["feature"].detach().item())
        n_batches += 1

    metrics = compute_metrics(all_labels, all_preds)

    denom = max(n_batches, 1)
    metrics.update(
        {
            "loss": total_loss / denom,
            "ce_loss": total_ce / denom,
            "kd_loss": total_kd / denom,
            "feature_loss": total_feature / denom,
            "num_batches": int(n_batches),
            "num_samples": int(len(all_labels)),
            "phase": phase,
        }
    )

    return metrics


# =========================================================
# Checkpointing
# =========================================================

def save_checkpoint(
    path: Path,
    student,
    optimizer,
    scheduler,
    scaler,
    config: Config,
    epoch: int,
    global_step: int,
    best_metric: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "stage": "tier2_implicit_2stage",
        "student_arch": "implicit_phobert_ffn",
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": asdict(config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "extra": extra or {},
    }

    torch.save(payload, path)


def load_training_resume(
    resume_path: Path,
    student,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> Tuple[int, int, float]:
    checkpoint = safe_torch_load(resume_path, map_location=device)
    state_dict = extract_tensor_state_dict(checkpoint)
    state_dict = strip_common_prefixes(state_dict)

    incompatible = student.load_state_dict(state_dict, strict=False)

    LOGGER.info("[RESUME] checkpoint=%s", resume_path)
    LOGGER.info("[RESUME] missing_keys=%s", incompatible.missing_keys)
    LOGGER.info("[RESUME] unexpected_keys=%s", incompatible.unexpected_keys)

    if len(incompatible.missing_keys) > 10 or len(incompatible.unexpected_keys) > 10:
        LOGGER.warning(
            "[RESUME] Many missing/unexpected keys (%d missing, %d unexpected). "
            "Architecture may have changed since this checkpoint was saved. "
            "Check spiking_layers, timesteps/solver settings, and student architecture.",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_metric = float(checkpoint.get("best_metric", -1.0))

    return epoch, global_step, best_metric


# =========================================================
# Training
# =========================================================

def train_one_epoch(
    config: Config,
    phase: str,
    epoch: int,
    student,
    teacher,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    global_step: int,
) -> Tuple[int, Dict[str, float]]:
    student.train()
    teacher.eval()

    use_amp = config.amp and device.type == "cuda"

    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    running_feature = 0.0
    n_batches = 0

    start = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        if config.max_steps > 0 and global_step >= config.max_steps:
            break

        if hasattr(student, "set_global_step"):
            student.set_global_step(global_step)

        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        with amp_autocast(device, use_amp):
            student_outputs = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )

            losses = compute_loss(
                config=config,
                phase=phase,
                student=student,
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                batch=batch,
            )

            loss = losses["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if hasattr(student, "post_step_clamp"):
            student.post_step_clamp()

        global_step += 1
        scheduler.step()

        loss_value = float(loss.detach().item())
        ce_value = float(losses["ce"].detach().item())
        kd_value = float(losses["kd"].detach().item())
        feature_value = float(losses["feature"].detach().item())

        running_loss += loss_value
        running_ce += ce_value
        running_kd += kd_value
        running_feature += feature_value
        n_batches += 1

        if config.log_every_steps > 0 and global_step % config.log_every_steps == 0:
            avg_steps, skipped_pct = student.convergence_stats()
            print(
                f"[MARKER] tier=tier2_2stage seed={config.seed} "
                f"phase={phase} epoch={epoch} step={global_step} "
                f"loss={loss_value:.6f} ce={ce_value:.6f} kd={kd_value:.6f} "
                f"feature={feature_value:.6f} firing_rate={student.mean_firing_rate():.4f} "
                f"anderson_steps={avg_steps:.2f} skipped_pct={skipped_pct:.2f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    denom = max(n_batches, 1)

    return global_step, {
        "train_loss": running_loss / denom,
        "train_ce": running_ce / denom,
        "train_kd": running_kd / denom,
        "train_feature": running_feature / denom,
        "train_batches": float(n_batches),
        "elapsed_sec": float(elapsed),
    }


def train_all(
    config: Config,
    student,
    teacher,
    train_loader,
    dev_loader,
    test_loader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    output_dir: Path,
    start_epoch: int = 0,
    start_global_step: int = 0,
    initial_best_metric: float = -1.0,
) -> Tuple[Dict[str, Any], Path, Path]:
    best_metric = float(initial_best_metric)
    global_step = int(start_global_step)

    best_checkpoint = output_dir / f"best_tier2_implicit_2stage_seed{config.seed}.pt"
    last_checkpoint = output_dir / f"last_tier2_implicit_2stage_seed{config.seed}.pt"

    history: List[Dict[str, Any]] = []

    total_epochs = int(config.task_kd_epochs + config.prediction_kd_epochs)

    for epoch in range(start_epoch + 1, total_epochs + 1):
        if epoch <= config.task_kd_epochs:
            phase = "task_ikd"
        else:
            phase = "prediction_kd"

        global_step, train_stats = train_one_epoch(
            config=config,
            phase=phase,
            epoch=epoch,
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            global_step=global_step,
        )

        dev_metrics = evaluate(
            config=config,
            student=student,
            teacher=teacher,
            loader=dev_loader,
            device=device,
            phase=phase,
        )

        current_metric = float(dev_metrics["f1_macro"])
        is_best = current_metric > best_metric

        if is_best:
            best_metric = current_metric

        epoch_record = {
            "epoch": int(epoch),
            "phase": phase,
            "global_step": int(global_step),
            "train": train_stats,
            "dev": dev_metrics,
            "best_metric": float(best_metric),
            "is_best": bool(is_best),
            "mean_firing_rate": float(student.mean_firing_rate()),
            "per_layer_stats": student.per_layer_stats(),
        }
        history.append(epoch_record)

        avg_steps, skipped_pct = student.convergence_stats()

        print(
            f"[MARKER] tier=tier2_2stage seed={config.seed} "
            f"epoch={epoch} phase={phase} "
            f"dev_f1_macro={dev_metrics['f1_macro']:.4f} "
            f"dev_f1_weighted={dev_metrics['f1_weighted']:.4f} "
            f"dev_acc={dev_metrics['accuracy']:.4f} "
            f"best_f1_macro={best_metric:.4f} "
            f"firing_rate={student.mean_firing_rate():.4f} "
            f"anderson_steps={avg_steps:.2f} skipped_pct={skipped_pct:.2f}",
            flush=True,
        )

        save_checkpoint(
            path=last_checkpoint,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            extra={
                "reason": "epoch_end_last",
                "epoch_record": epoch_record,
            },
        )

        if is_best:
            save_checkpoint(
                path=best_checkpoint,
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                extra={
                    "reason": "best_dev_macro_f1",
                    "epoch_record": epoch_record,
                },
            )
            LOGGER.info("Saved best checkpoint to %s", best_checkpoint)

        if config.save_every_steps > 0 and global_step % config.save_every_steps == 0:
            periodic_path = output_dir / f"periodic_tier2_implicit_2stage_seed{config.seed}_step{global_step}.pt"
            save_checkpoint(
                path=periodic_path,
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                extra={"reason": "periodic"},
            )

        if config.max_steps > 0 and global_step >= config.max_steps:
            LOGGER.info("Reached max_steps=%d. Stopping training.", config.max_steps)
            break

    if best_checkpoint.exists():
        checkpoint = safe_torch_load(best_checkpoint, map_location=device)
        state_dict = extract_tensor_state_dict(checkpoint)
        state_dict = strip_common_prefixes(state_dict)
        student.load_state_dict(state_dict, strict=False)
        LOGGER.info("Loaded best checkpoint for final test: %s", best_checkpoint)
    else:
        LOGGER.warning("Best checkpoint not found. Testing current student state.")

    final_phase = "prediction_kd" if config.prediction_kd_epochs > 0 else "task_ikd"

    test_metrics = evaluate(
        config=config,
        student=student,
        teacher=teacher,
        loader=test_loader,
        device=device,
        phase=final_phase,
    )

    report = {
        "stage": "tier2_implicit_2stage",
        "student_arch": "implicit_phobert_ffn",
        "seed": int(config.seed),
        "best_dev_f1_macro": float(best_metric),
        "test_metrics": test_metrics,
        "history": history,
        "config": asdict(config),
        "mean_firing_rate": float(student.mean_firing_rate()),
        "per_layer_stats": student.per_layer_stats(),
        "convergence": {
            "avg_anderson_steps": float(student.convergence_stats()[0]),
            "skipped_pct": float(student.convergence_stats()[1]),
        },
        "training_memory": get_peak_memory_report(device),
        "theoretical_energy": compute_theoretical_energy_report(student, config),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "note": (
            "Tier 2 full two-stage implicit PhoBERT-SNN KD. "
            "Stage 1 checkpoint is loaded from stage1_wiki_implicit.py. "
            "Stage 2 uses task-based internal KD followed by prediction-layer distillation."
        ),
    }

    if config.measure_gpu_energy:
        report["actual_gpu_energy"] = {
            "student": measure_gpu_energy_best_effort(
                model=student,
                loader=test_loader,
                device=device,
                max_batches=config.max_energy_batches,
                kind="student",
            ),
            "teacher": measure_gpu_energy_best_effort(
                model=teacher,
                loader=test_loader,
                device=device,
                max_batches=config.max_energy_batches,
                kind="teacher",
            ),
        }

        s = report["actual_gpu_energy"]["student"]
        t = report["actual_gpu_energy"]["teacher"]

        if s.get("available") and t.get("available"):
            report["actual_gpu_energy"]["student_teacher_ratio"] = (
                s["energy_per_sample_joules"] / max(t["energy_per_sample_joules"], 1e-12)
            )

    report_path = output_dir / f"tier2_implicit_2stage_report_seed{config.seed}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved report to %s", report_path)

    print(
        f"[MARKER] tier=tier2_2stage DONE seed={config.seed} "
        f"best_dev_f1_macro={best_metric:.4f} "
        f"test_f1_macro={test_metrics['f1_macro']:.4f} "
        f"test_f1_weighted={test_metrics['f1_weighted']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"firing_rate={student.mean_firing_rate():.4f} "
        f"best_ckpt={best_checkpoint}",
        flush=True,
    )

    return report, best_checkpoint, last_checkpoint


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier 2 full two-stage implicit PhoBERT-SNN KD."
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, default="")

    parser.add_argument("--train_path", type=str, default="cache/train_segmented.parquet")
    parser.add_argument("--dev_path", type=str, default="cache/dev_segmented.parquet")
    parser.add_argument("--test_path", type=str, default="cache/test_segmented.parquet")
    parser.add_argument("--output_dir", type=str, default="uit-models/tier2_implicit_2stage")

    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")

    parser.add_argument("--task_kd_epochs", type=int, default=15)
    parser.add_argument("--prediction_kd_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--temperature_kd", type=float, default=4.0)
    parser.add_argument("--alpha_ce", type=float, default=0.5)
    parser.add_argument("--lambda_f", type=float, default=0.1)
    parser.add_argument("--alpha_ce_final", type=float, default=0.5)
    parser.add_argument("--lambda_f_final", type=float, default=0.0)

    parser.add_argument("--spiking_layers", type=int, default=4)
    parser.add_argument("--anderson_max_iter", type=int, default=30)
    parser.add_argument("--anderson_tol", type=float, default=1e-3)
    parser.add_argument("--anderson_m", type=int, default=5)
    parser.add_argument("--neumann_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--surrogate_alpha", type=float, default=2.0)

    parser.add_argument(
        "--equilibrium_metric",
        type=str,
        default="both",
        choices=["residual", "firing_rate", "both"],
    )
    parser.add_argument("--firing_rate_tol", type=float, default=1e-4)
    parser.add_argument("--min_equilibrium_iter", type=int, default=3)

    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--log_every_steps", type=int, default=1000)

    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--load_stage1_strict", action="store_true")

    parser.add_argument("--measure_gpu_energy", action="store_true")
    parser.add_argument("--max_energy_batches", type=int, default=0)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        seed=args.seed,
        model_name=args.model_name,
        teacher_ckpt=args.teacher_ckpt,
        stage1_ckpt=args.stage1_ckpt,
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        task_kd_epochs=args.task_kd_epochs,
        prediction_kd_epochs=args.prediction_kd_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        temperature_kd=args.temperature_kd,
        alpha_ce=args.alpha_ce,
        lambda_f=args.lambda_f,
        alpha_ce_final=args.alpha_ce_final,
        lambda_f_final=args.lambda_f_final,
        spiking_layers=args.spiking_layers,
        anderson_max_iter=args.anderson_max_iter,
        anderson_tol=args.anderson_tol,
        anderson_m=args.anderson_m,
        neumann_k=args.neumann_k,
        threshold=args.threshold,
        surrogate_alpha=args.surrogate_alpha,
        equilibrium_metric=args.equilibrium_metric,
        firing_rate_tol=args.firing_rate_tol,
        min_equilibrium_iter=args.min_equilibrium_iter,
        save_every_steps=args.save_every_steps,
        log_every_steps=args.log_every_steps,
        local_files_only=args.local_files_only,
        amp=not args.no_amp,
        resume=args.resume,
        load_stage1_strict=args.load_stage1_strict,
        measure_gpu_energy=args.measure_gpu_energy,
        max_energy_batches=args.max_energy_batches,
    )


def main(args: argparse.Namespace) -> None:
    configure_logging()

    config = build_config(args)
    set_seed(config.seed)

    project_root = infer_project_root()

    output_base = resolve_path(config.output_dir, project_root)
    output_dir = make_output_dir(output_base, config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.stage1_ckpt:
        config.stage1_ckpt = str(make_default_stage1_ckpt(project_root, config))

    config.output_dir = str(output_dir)
    config.teacher_ckpt = str(resolve_path(config.teacher_ckpt, project_root))
    config.stage1_ckpt = str(resolve_path(config.stage1_ckpt, project_root))
    config.train_path = str(resolve_path(config.train_path, project_root))
    config.dev_path = str(resolve_path(config.dev_path, project_root))
    config.test_path = str(resolve_path(config.test_path, project_root))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Config: %s", asdict(config))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
        use_fast=False,
    )

    train_dataset = ParquetSentimentDataset(
        path=Path(config.train_path),
        tokenizer=tokenizer,
        text_column=config.text_column,
        label_column=config.label_column,
        max_length=config.max_length,
    )
    dev_dataset = ParquetSentimentDataset(
        path=Path(config.dev_path),
        tokenizer=tokenizer,
        text_column=config.text_column,
        label_column=config.label_column,
        max_length=config.max_length,
    )
    test_dataset = ParquetSentimentDataset(
        path=Path(config.test_path),
        tokenizer=tokenizer,
        text_column=config.text_column,
        label_column=config.label_column,
        max_length=config.max_length,
    )

    train_loader = make_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    dev_loader = make_dataloader(
        dataset=dev_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = make_dataloader(
        dataset=test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    teacher = load_teacher(config, device, project_root)
    student = load_student(config, device, project_root)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    total_epochs = int(config.task_kd_epochs + config.prediction_kd_epochs)
    if total_epochs <= 0:
        raise ValueError("task_kd_epochs + prediction_kd_epochs must be > 0")

    if config.max_steps > 0:
        total_steps = int(config.max_steps)
    else:
        total_steps = max(1, len(train_loader) * total_epochs)

    warmup_steps = int(config.warmup_ratio * total_steps)

    LOGGER.info("Total epochs=%d | total steps=%d | warmup steps=%d", total_epochs, total_steps, warmup_steps)

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
    best_metric = -1.0

    if config.resume:
        resume_path = resolve_path(config.resume, project_root)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        start_epoch, start_global_step, best_metric = load_training_resume(
            resume_path=resume_path,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    train_all(
        config=config,
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        output_dir=output_dir,
        start_epoch=start_epoch,
        start_global_step=start_global_step,
        initial_best_metric=best_metric,
    )


if __name__ == "__main__":
    main(parse_args())
