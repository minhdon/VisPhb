#!/usr/bin/env python3
"""
tier3_hybrid_2stage.py

Tier 3 full two-stage hybrid implicit PhoBERT-SNN training.

Pipeline:
    Stage 1:
        Already trained by stage1_wiki_implicit.py
        PhoBERT pretrained teacher -> ImplicitPhoBERTStudent on segmented Wiki.

    Stage 2A:
        Task-based internal KD on UIT-VSFC:
            CE + KD logits + feature alignment + embedding alignment
        with soft curriculum:
            CE/KD task weight is gradually increased, not switched abruptly.

    Stage 2B:
        Prediction-layer distillation / final tuning:
            CE + KD logits, with feature/embedding loss reduced or disabled.

This file is intended to be a Vietnamese SpikingBERT-inspired full two-stage
hybrid implicit PhoBERT-SNN KD pipeline.

Important:
    This is NOT an absolute reproduction of SpikingBERT because attention remains
    ANN and only selected FFN blocks are replaced by implicit SNN modules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOGGER = logging.getLogger("tier3_hybrid_2stage")

ALIGNMENT_LAYERS = [3, 6, 9, 12]
NUM_LABELS = 3
LABEL_NAMES = ["negative", "neutral", "positive"]


# =========================================================
# Import Tier-2 utilities and Stage-1 implicit architecture
# =========================================================

def import_python_module(module_name: str, file_name: str):
    existing = sys.modules.get(module_name)
    if existing is not None:
        if file_name == "tier2_implicit_2stage.py" and hasattr(existing, "ImplicitPhoBERTStudent"):
            return existing
        if file_name == "stage1_wiki_implicit.py" and hasattr(existing, "ImplicitPhoBERTStudent"):
            return existing
        sys.modules.pop(module_name, None)

    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / file_name

    if not module_path.exists():
        raise FileNotFoundError(
            f"Cannot find {file_name} at {module_path}. "
            f"Put tier3_hybrid_2stage.py, tier2_implicit_2stage.py, "
            f"and stage1_wiki_implicit.py in the same folder."
        )

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)

    # Register before exec_module for dataclass/runtime compatibility.
    # Roll back if execution fails.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise

    return module


_TIER2 = import_python_module("tier2_implicit_2stage_runtime", "tier2_implicit_2stage.py")

Stage1Config = _TIER2.Stage1Config
ImplicitPhoBERTStudent = _TIER2.ImplicitPhoBERTStudent

safe_torch_load = _TIER2.safe_torch_load
extract_tensor_state_dict = _TIER2.extract_tensor_state_dict
strip_common_prefixes = _TIER2.strip_common_prefixes
make_default_stage1_ckpt = _TIER2.make_default_stage1_ckpt
validate_stage1_checkpoint_config = _TIER2.validate_stage1_checkpoint_config

amp_autocast = _TIER2.amp_autocast
make_grad_scaler = _TIER2.make_grad_scaler
get_peak_memory_report = _TIER2.get_peak_memory_report

ParquetSentimentDataset = _TIER2.ParquetSentimentDataset
make_dataloader = _TIER2.make_dataloader
move_batch_to_device = _TIER2.move_batch_to_device
load_teacher = _TIER2.load_teacher
masked_mse = _TIER2.masked_mse
compute_kd_loss = _TIER2.compute_kd_loss
compute_theoretical_energy_report = _TIER2.compute_theoretical_energy_report


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
    output_dir: str = "uit-models/tier3_hybrid_2stage"

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
    lambda_embedding: float = 0.05

    alpha_ce_final: float = 0.5
    lambda_f_final: float = 0.0
    lambda_embedding_final: float = 0.0

    curriculum_warmup_ratio: float = 0.02
    curriculum_warmup_steps: int = 0
    curriculum_min_task_weight: float = 0.2

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# Hybrid student
# =========================================================

class HybridImplicitPhoBERTStudent(ImplicitPhoBERTStudent):
    """
    Tier-3 hybrid student.

    Base:
        ImplicitPhoBERTStudent from stage1_wiki_implicit.py

    Added:
        embedding_projector = Linear(hidden_size, hidden_size)

    Reason:
        Tier 3 uses embedding alignment in Stage 2. This projector is trainable.
        It is intentionally NOT Identity, because Identity would make the
        embedding alignment branch unable to learn its own alignment transform.
    """

    def __init__(self, config: Stage1Config):
        super().__init__(config)

        hidden_size = int(self.encoder.config.hidden_size)
        self.embedding_projector = nn.Linear(hidden_size, hidden_size)

    def project_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.embedding_projector(embeddings.float())


def load_student(config: Config, device: torch.device, project_root: Path):
    stage_cfg = build_stage1_config(config)
    student = HybridImplicitPhoBERTStudent(stage_cfg)

    stage1_ckpt = resolve_path(config.stage1_ckpt, project_root)

    if stage1_ckpt.exists():
        checkpoint = safe_torch_load(stage1_ckpt, map_location="cpu")
        validate_stage1_checkpoint_config(checkpoint, config, stage1_ckpt)

        raw_state = extract_tensor_state_dict(checkpoint)
        raw_state = strip_common_prefixes(raw_state)

        incompatible = student.load_state_dict(raw_state, strict=False)

        LOGGER.info("[STAGE1 LOAD] checkpoint=%s", stage1_ckpt)
        LOGGER.info("[STAGE1 LOAD] strict=%s", config.load_stage1_strict)
        LOGGER.info("[STAGE1 LOAD] missing_keys=%s", incompatible.missing_keys)
        LOGGER.info("[STAGE1 LOAD] unexpected_keys=%s", incompatible.unexpected_keys)

        expected_missing = [
            key for key in incompatible.missing_keys
            if key.startswith("embedding_projector.")
        ]

        if expected_missing:
            LOGGER.info(
                "[STAGE1 LOAD] Expected Tier-3-only missing embedding_projector keys: %s",
                expected_missing,
            )

        non_tier3_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("embedding_projector.")
        ]

        if config.load_stage1_strict and (non_tier3_missing or incompatible.unexpected_keys):
            raise RuntimeError(
                "Stage-1 checkpoint does not cleanly initialize Tier 3. "
                f"checkpoint={stage1_ckpt} "
                f"non_tier3_missing_keys={non_tier3_missing} "
                f"unexpected_keys={incompatible.unexpected_keys}"
            )

        if len(incompatible.missing_keys) > 40 or len(incompatible.unexpected_keys) > 40:
            LOGGER.warning(
                "[STAGE1 LOAD] Many missing/unexpected keys. "
                "Check spiking_layers and architecture compatibility."
            )
    else:
        raise FileNotFoundError(f"Stage-1 implicit checkpoint not found: {stage1_ckpt}")

    student.to(device)
    return student


# =========================================================
# Metrics / losses
# =========================================================

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


def compute_embedding_loss(
    student,
    student_hidden,
    teacher_hidden,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if student_hidden is None or teacher_hidden is None:
        raise RuntimeError("Embedding loss requires output_hidden_states=True.")

    student_embedding = student_hidden[0]
    teacher_embedding = teacher_hidden[0]

    if hasattr(student, "project_embeddings"):
        student_embedding = student.project_embeddings(student_embedding)
    elif hasattr(student, "embedding_projector"):
        student_embedding = student.embedding_projector(student_embedding.float())

    return masked_mse(student_embedding, teacher_embedding, attention_mask)


def curriculum_task_weight(
    config: Config,
    global_step: int,
    curriculum_steps: int,
    enabled: bool,
) -> float:
    if not enabled or curriculum_steps <= 0:
        return 1.0

    progress = min(1.0, float(global_step + 1) / float(max(1, curriculum_steps)))
    min_weight = float(config.curriculum_min_task_weight)
    min_weight = max(0.0, min(1.0, min_weight))

    return min_weight + (1.0 - min_weight) * progress


def compute_loss(
    config: Config,
    phase: str,
    student,
    student_outputs: Dict[str, Any],
    teacher_outputs,
    batch: Dict[str, torch.Tensor],
    global_step: int,
    curriculum_steps: int,
    use_curriculum: bool,
) -> Dict[str, torch.Tensor]:
    labels = batch["labels"]

    student_logits = student_outputs["logits"]
    teacher_logits = teacher_outputs.logits

    ce_loss = F.cross_entropy(student_logits.float(), labels)
    kd_loss = compute_kd_loss(student_logits, teacher_logits, config.temperature_kd)

    feature_loss = compute_feature_loss(
        student=student,
        student_hidden=student_outputs["hidden_states"],
        teacher_hidden=teacher_outputs.hidden_states,
        attention_mask=batch["attention_mask"],
    )

    embedding_loss = compute_embedding_loss(
        student=student,
        student_hidden=student_outputs["hidden_states"],
        teacher_hidden=teacher_outputs.hidden_states,
        attention_mask=batch["attention_mask"],
    )

    if phase == "task_ikd":
        alpha_ce = config.alpha_ce
        lambda_f = config.lambda_f
        lambda_e = config.lambda_embedding
        task_weight = curriculum_task_weight(
            config=config,
            global_step=global_step,
            curriculum_steps=curriculum_steps,
            enabled=use_curriculum,
        )
    elif phase == "prediction_kd":
        alpha_ce = config.alpha_ce_final
        lambda_f = config.lambda_f_final
        lambda_e = config.lambda_embedding_final
        task_weight = 1.0
    else:
        raise ValueError(f"Unknown phase: {phase}")

    task_loss = alpha_ce * ce_loss + (1.0 - alpha_ce) * kd_loss

    loss = (
        task_weight * task_loss
        + lambda_f * feature_loss
        + lambda_e * embedding_loss
    )

    return {
        "loss": loss,
        "task": task_loss,
        "ce": ce_loss,
        "kd": kd_loss,
        "feature": feature_loss,
        "embedding": embedding_loss,
        "task_weight": torch.tensor(float(task_weight), device=labels.device),
    }


# =========================================================
# GPU energy best-effort
# =========================================================

def forward_no_hidden(model, batch: Dict[str, torch.Tensor]):
    try:
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=False,
            return_dict=True,
        )
    except TypeError:
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=False,
        )


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

            _ = forward_no_hidden(model, batch)
            num_samples += int(batch["input_ids"].size(0))

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
    total_task = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feature = 0.0
    total_embedding = 0.0
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
            global_step=0,
            curriculum_steps=0,
            use_curriculum=False,
        )

        logits = student_outputs["logits"].float()
        preds = torch.argmax(logits, dim=-1)

        all_labels.extend(batch["labels"].detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        total_loss += float(losses["loss"].detach().item())
        total_task += float(losses["task"].detach().item())
        total_ce += float(losses["ce"].detach().item())
        total_kd += float(losses["kd"].detach().item())
        total_feature += float(losses["feature"].detach().item())
        total_embedding += float(losses["embedding"].detach().item())
        n_batches += 1

    metrics = compute_metrics(all_labels, all_preds)

    denom = max(n_batches, 1)
    metrics.update(
        {
            "loss": total_loss / denom,
            "task_loss": total_task / denom,
            "ce_loss": total_ce / denom,
            "kd_loss": total_kd / denom,
            "feature_loss": total_feature / denom,
            "embedding_loss": total_embedding / denom,
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
        "stage": "tier3_hybrid_2stage",
        "student_arch": "hybrid_implicit_phobert_ffn",
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
    curriculum_steps: int,
) -> Tuple[int, Dict[str, float]]:
    student.train()
    teacher.eval()

    use_amp = config.amp and device.type == "cuda"

    running_loss = 0.0
    running_task = 0.0
    running_ce = 0.0
    running_kd = 0.0
    running_feature = 0.0
    running_embedding = 0.0
    running_task_weight = 0.0
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
                global_step=global_step,
                curriculum_steps=curriculum_steps,
                use_curriculum=(phase == "task_ikd"),
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
        task_value = float(losses["task"].detach().item())
        ce_value = float(losses["ce"].detach().item())
        kd_value = float(losses["kd"].detach().item())
        feature_value = float(losses["feature"].detach().item())
        embedding_value = float(losses["embedding"].detach().item())
        task_weight_value = float(losses["task_weight"].detach().item())

        running_loss += loss_value
        running_task += task_value
        running_ce += ce_value
        running_kd += kd_value
        running_feature += feature_value
        running_embedding += embedding_value
        running_task_weight += task_weight_value
        n_batches += 1

        if config.log_every_steps > 0 and global_step % config.log_every_steps == 0:
            avg_steps, skipped_pct = student.convergence_stats()
            print(
                f"[MARKER] tier=tier3_2stage seed={config.seed} "
                f"phase={phase} epoch={epoch} step={global_step} "
                f"loss={loss_value:.6f} task={task_value:.6f} "
                f"ce={ce_value:.6f} kd={kd_value:.6f} "
                f"feature={feature_value:.6f} embedding={embedding_value:.6f} "
                f"task_weight={task_weight_value:.4f} "
                f"firing_rate={student.mean_firing_rate():.4f} "
                f"anderson_steps={avg_steps:.2f} skipped_pct={skipped_pct:.2f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    denom = max(n_batches, 1)

    return global_step, {
        "train_loss": running_loss / denom,
        "train_task": running_task / denom,
        "train_ce": running_ce / denom,
        "train_kd": running_kd / denom,
        "train_feature": running_feature / denom,
        "train_embedding": running_embedding / denom,
        "train_task_weight": running_task_weight / denom,
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
    curriculum_steps: int = 0,
) -> Tuple[Dict[str, Any], Path, Path]:
    best_metric = float(initial_best_metric)
    global_step = int(start_global_step)

    best_checkpoint = output_dir / f"best_tier3_hybrid_2stage_seed{config.seed}.pt"
    last_checkpoint = output_dir / f"last_tier3_hybrid_2stage_seed{config.seed}.pt"

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
            curriculum_steps=curriculum_steps,
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
            f"[MARKER] tier=tier3_2stage seed={config.seed} "
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
            periodic_path = output_dir / f"periodic_tier3_hybrid_2stage_seed{config.seed}_step{global_step}.pt"
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
        "stage": "tier3_hybrid_2stage",
        "student_arch": "hybrid_implicit_phobert_ffn",
        "seed": int(config.seed),
        "best_dev_f1_macro": float(best_metric),
        "test_metrics": test_metrics,
        "history": history,
        "config": asdict(config),
        "curriculum": {
            "curriculum_warmup_steps": int(curriculum_steps),
            "curriculum_warmup_ratio": float(config.curriculum_warmup_ratio),
            "curriculum_min_task_weight": float(config.curriculum_min_task_weight),
            "description": (
                "Soft curriculum: CE/KD task loss is not removed during warmup. "
                "It starts at curriculum_min_task_weight and increases to 1.0."
            ),
        },
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
            "Tier 3 full two-stage hybrid implicit PhoBERT-SNN KD. "
            "Stage 1 checkpoint is loaded from stage1_wiki_implicit.py. "
            "Stage 2 uses task-based internal KD with embedding alignment and "
            "soft curriculum, followed by prediction-layer distillation."
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

    report_path = output_dir / f"tier3_hybrid_2stage_report_seed{config.seed}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved report to %s", report_path)

    print(
        f"[MARKER] tier=tier3_2stage DONE seed={config.seed} "
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
        description="Tier 3 full two-stage hybrid implicit PhoBERT-SNN KD."
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, default="")

    parser.add_argument("--train_path", type=str, default="cache/train_segmented.parquet")
    parser.add_argument("--dev_path", type=str, default="cache/dev_segmented.parquet")
    parser.add_argument("--test_path", type=str, default="cache/test_segmented.parquet")
    parser.add_argument("--output_dir", type=str, default="uit-models/tier3_hybrid_2stage")

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
    parser.add_argument("--lambda_embedding", type=float, default=0.05)

    parser.add_argument("--alpha_ce_final", type=float, default=0.5)
    parser.add_argument("--lambda_f_final", type=float, default=0.0)
    parser.add_argument("--lambda_embedding_final", type=float, default=0.0)

    parser.add_argument("--curriculum_warmup_ratio", type=float, default=0.02)
    parser.add_argument("--curriculum_warmup_steps", type=int, default=0)
    parser.add_argument("--curriculum_min_task_weight", type=float, default=0.2)

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
        lambda_embedding=args.lambda_embedding,
        alpha_ce_final=args.alpha_ce_final,
        lambda_f_final=args.lambda_f_final,
        lambda_embedding_final=args.lambda_embedding_final,
        curriculum_warmup_ratio=args.curriculum_warmup_ratio,
        curriculum_warmup_steps=args.curriculum_warmup_steps,
        curriculum_min_task_weight=args.curriculum_min_task_weight,
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

    optimizer_warmup_steps = int(config.warmup_ratio * total_steps)

    task_kd_steps = max(1, len(train_loader) * max(1, config.task_kd_epochs))

    if config.curriculum_warmup_steps > 0:
        curriculum_steps = int(config.curriculum_warmup_steps)
    else:
        curriculum_steps = int(config.curriculum_warmup_ratio * task_kd_steps)

    LOGGER.info(
        "Total epochs=%d | total steps=%d | optimizer warmup steps=%d | curriculum steps=%d",
        total_epochs,
        total_steps,
        optimizer_warmup_steps,
        curriculum_steps,
    )

    if curriculum_steps <= 0:
        LOGGER.info(
            "Curriculum warmup disabled because curriculum_steps=0. "
            "Set --curriculum_warmup_ratio > 0 or --curriculum_warmup_steps > 0 to enable it."
        )
    else:
        LOGGER.info(
            "Curriculum warmup enabled: curriculum_steps=%d, min_task_weight=%.4f",
            curriculum_steps,
            config.curriculum_min_task_weight,
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=optimizer_warmup_steps,
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
        curriculum_steps=curriculum_steps,
    )


if __name__ == "__main__":
    main(parse_args())
