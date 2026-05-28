from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from sklearn.metrics import accuracy_score, classification_report, f1_score
except Exception:
    accuracy_score = None
    classification_report = None
    f1_score = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
except Exception:
    AutoModel = None
    AutoTokenizer = None
    get_cosine_schedule_with_warmup = None

try:
    from underthesea import word_tokenize
except Exception:
    word_tokenize = None

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _PYNVML_AVAILABLE = True
except Exception:
    _pynvml = None
    _PYNVML_AVAILABLE = False


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


EXPERIMENT_KIND = "tier2"
TIER2_VARIANT = "b"
RUN_NAME = "tier2_variant_b_feature_stage2"
DESCRIPTION = f"UIT standalone training: {RUN_NAME}"
MODEL_NAME = "vinai/phobert-base-v2"
NUM_LABELS = 3
LABEL_NAMES = ["negative", "neutral", "positive"]
MAC_PJ = 4.6
AC_PJ = 0.9
MARKER_PREFIX = "[MARKER]"


def format_marker_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else str(value)
    text = str(value).replace(os.linesep, " ")
    if any(char.isspace() for char in text):
        return json.dumps(text, ensure_ascii=False)
    return text


def log_marker(event: str, **fields) -> None:
    payload = {"event": event, "run": RUN_NAME, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    payload.update({key: value for key, value in fields.items() if value is not None})
    text = " ".join(f"{key}={format_marker_value(value)}" for key, value in payload.items())
    print(f"{MARKER_PREFIX} {text}", flush=True)


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = infer_project_root()
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--data-dir", type=Path, default=root / "uit-vsfc")
    parser.add_argument("--output-dir", type=Path, default=root / "uit-models" / RUN_NAME)
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "uit-models")
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.cache/huggingface").expanduser())
    parser.add_argument("--teacher-ckpt", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run-batches", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--progress-bars", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--save-optimizer-state", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="off")
    parser.add_argument("--no-segmentation", action="store_true")
    parser.add_argument("--t-steps", type=int, default=4)
    parser.add_argument("--spiking-layers", type=int, default=12, choices=[3, 6, 9, 12])
    parser.add_argument("--temp-kd", type=float, default=4.0)
    parser.add_argument("--alpha-kd", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--k-slope", type=float, default=25.0)
    parser.add_argument("--feature-weight", type=float, default=0.1)
    parser.add_argument("--stage1-epochs", type=int, default=3 if EXPERIMENT_KIND == "tier3" else 5)
    parser.add_argument("--wiki-max-samples", type=int, default=0)
    parser.add_argument("--wiki-batch-size", type=int, default=None)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--augment-stage2", action="store_true")
    parser.add_argument("--t-conv", type=int, default=80)
    parser.add_argument("--v-th", type=float, default=1.0)
    parser.add_argument("--ssa-v-th", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--hybrid-variant", choices=["3a", "3b", "both"], default="3b")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(args: argparse.Namespace):
    require_runtime_dependencies()
    tokenizer = load_tokenizer(args)
    print("[DATA] Loading UIT-VSFC splits", flush=True)
    train_df = load_vsfc_split(args, "train")
    dev_df = load_vsfc_split(args, "dev")
    test_df = load_vsfc_split(args, "test")
    print_split_summary("train", train_df)
    print_split_summary("dev", dev_df)
    print_split_summary("test", test_df)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(VSFCDataset(train_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(VSFCDataset(dev_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(VSFCDataset(test_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    return train_loader, dev_loader, test_loader, tokenizer


def build_model(args: argparse.Namespace, device: torch.device):
    print("[MODEL] Resolving teacher checkpoint", flush=True)
    teacher_ckpt = resolve_teacher_checkpoint(args)
    print(f"[MODEL] Teacher checkpoint: {teacher_ckpt}", flush=True)
    teacher = PhoBERTTeacher(args, teacher_ckpt)
    print(f"[MODEL] Moving teacher to {device}", flush=True)
    teacher = teacher.to(device)
    sync_cuda_if_needed(device, "teacher.to(device)")
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    if EXPERIMENT_KIND in {"tier1", "tier3"}:
        student = EquilibriumPhoBERTStudent(args, teacher_ckpt)
    else:
        student = SpikeBERTStudent(args, teacher_ckpt)
    print(f"[MODEL] Moving student to {device}", flush=True)
    student = student.to(device)
    sync_cuda_if_needed(device, "student.to(device)")

    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    print(f"[MODEL] Teacher params: {teacher_params:,}", flush=True)
    print(f"[MODEL] Student params: {student_params:,}", flush=True)
    return TrainingBundle(student=student, teacher=teacher, args=args, tokenizer=None)


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs):
    args = model.args
    student = model.student
    teacher = model.teacher
    student.train()
    teacher.eval()
    student.reset_trackers()
    start = time.time()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feature = 0.0
    steps = 0
    skipped_steps = 0
    scaler = model.scaler
    total_batches = effective_len(loader, args.dry_run_batches)
    log_marker(
        "epoch_start",
        phase="train",
        submode=getattr(args, "_active_submode", None),
        seed=getattr(args, "_active_seed", None),
        seed_run=getattr(args, "_active_seed_run", None),
        epoch=f"{epoch}/{total_epochs}",
        total_batches=total_batches,
    )
    print(f"=== Epoch {epoch}/{total_epochs} ===", flush=True)

    iterator = make_iterator(loader, f"train ep{epoch}")
    for step, batch in enumerate(iterator, start=1):
        if should_stop(step, args.dry_run_batches):
            break
        batch = to_device(batch, device)
        input_ids = batch["input_ids"]
        if getattr(args, "augment_stage2", False):
            input_ids = augment_input_ids(input_ids, batch["attention_mask"], model.tokenizer)

        optimizer.zero_grad(set_to_none=True)
        need_hidden = needs_feature_alignment()
        with torch.no_grad():
            teacher_out = teacher(input_ids, batch["attention_mask"], output_hidden_states=need_hidden)

        with autocast_context(device):
            student_out = student(input_ids, batch["attention_mask"], output_hidden_states=need_hidden)
            if EXPERIMENT_KIND in {"tier1", "tier3"} and has_unconverged(student_out.get("convergence", [])):
                skipped_steps += 1
                continue
            loss, parts = compute_stage2_loss(
                student_out,
                teacher_out,
                batch["labels"],
                args,
                student_model=student,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        steps += 1
        total_loss += float(loss.item())
        total_ce += float(parts.get("ce", 0.0))
        total_kd += float(parts.get("kd", 0.0))
        total_feature += float(parts.get("feature", 0.0))

    denom = max(1, steps)
    return {
        "loss": total_loss / denom,
        "ce": total_ce / denom,
        "kd": total_kd / denom,
        "feature": total_feature / denom,
        "steps": steps,
        "skipped_steps": skipped_steps,
        "elapsed_sec": time.time() - start,
    }


@torch.no_grad()
def evaluate(model, loader, device, split_name):
    student = model.student
    args = model.args
    student.eval()
    student.reset_trackers()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    start = time.time()
    total_batches = effective_len(loader, args.dry_run_batches)
    log_marker(
        "eval_start",
        phase=split_name,
        submode=getattr(args, "_active_submode", None),
        seed=getattr(args, "_active_seed", None),
        seed_run=getattr(args, "_active_seed_run", None),
        total_batches=total_batches,
    )
    print(f"[EVAL] {split_name}: start batches={total_batches}", flush=True)

    iterator = make_iterator(loader, f"eval {split_name}")
    for step, batch in enumerate(iterator, start=1):
        if should_stop(step, args.dry_run_batches):
            break
        batch = to_device(batch, device)
        with autocast_context(device):
            out = student(batch["input_ids"], batch["attention_mask"], output_hidden_states=False)
            loss = F.cross_entropy(out["logits"].float(), batch["labels"])
        total_loss += float(loss.item())
        preds = out["logits"].argmax(dim=-1)
        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(y_true))
    metrics["elapsed_sec"] = time.time() - start
    metrics["firing_rates"] = student.get_firing_rates()
    print(
        f"[EVAL] {split_name}: loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
        f"f1_macro={metrics['f1_macro']:.4f} f1_weighted={metrics['f1_weighted']:.4f} "
        f"elapsed={metrics['elapsed_sec']:.1f}s",
        flush=True,
    )
    log_marker(
        "eval_end",
        phase=split_name,
        submode=getattr(args, "_active_submode", None),
        seed=getattr(args, "_active_seed", None),
        seed_run=getattr(args, "_active_seed_run", None),
        loss=metrics["loss"],
        accuracy=metrics["accuracy"],
        f1_weighted=metrics["f1_weighted"],
        elapsed_sec=metrics["elapsed_sec"],
    )
    if split_name == "test":
        print_classification_report(split_name, y_true, y_pred)
    return metrics, y_true, y_pred


@dataclass
class ConvergenceRecord:
    layer: int
    steps: int
    converged: bool
    delta_norm: float
    asr: float


@dataclass
class TrainingBundle:
    student: nn.Module
    teacher: nn.Module
    args: argparse.Namespace
    tokenizer: object | None
    scaler: object | None = None


class VSFCDataset(Dataset):
    def __init__(self, df, tokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        enc = self.tokenizer(
            str(row["text"]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


class WikiDataset(Dataset):
    """
    Wikipedia dataset for Stage 1 unsupervised distillation.

    Word segmentation is applied before tokenization because PhoBERT was
    pre-trained on word-segmented Vietnamese text.
    """

    def __init__(self, dataset, tokenizer, max_len: int, preprocess: bool = True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        raw_texts = [str(item.get("text", "")) for item in dataset]

        if preprocess:
            print(
                f"[WIKI] Pre-segmenting {len(raw_texts):,} documents "
                f"(this may take several minutes)...",
                flush=True,
            )
            self.texts = []
            for i, text in enumerate(raw_texts):
                self.texts.append(segment_vi(text))
                if (i + 1) % 10_000 == 0:
                    print(f"[WIKI] Segmented {i + 1:,}/{len(raw_texts):,}", flush=True)
            print("[WIKI] Segmentation complete.", flush=True)
        else:
            self.texts = raw_texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class PhoBERTTeacher(nn.Module):
    def __init__(self, args: argparse.Namespace, checkpoint_path: Path | None):
        super().__init__()
        self.encoder = load_backbone(args)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, NUM_LABELS)
        if checkpoint_path is not None and checkpoint_path.exists():
            state = safe_torch_load(checkpoint_path, map_location="cpu")
            print_checkpoint_diagnostics("Teacher", checkpoint_path, state)
            incompatible = self.load_state_dict(normalize_state_dict(state), strict=False)
            print_load_report("Teacher", incompatible)

    def forward(self, input_ids, attention_mask, output_hidden_states: bool = False):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return {"logits": self.classifier(cls), "hidden_states": out.hidden_states, "last_hidden_state": out.last_hidden_state}


class ArctanSpike(Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        return grad_output * (alpha / (2.0 * (1.0 + (alpha * x).pow(2)))), None


class SpikeBERTFFN(nn.Module):
    def __init__(self, intermediate_dense, output_dense, args):
        super().__init__()
        self.intermediate_dense = intermediate_dense
        self.output_dense = output_dense
        self.t_steps = args.t_steps
        self.threshold = 1.0
        self.k_slope = args.k_slope
        self.beta = nn.Parameter(torch.full((intermediate_dense.out_features,), float(args.beta)))
        self.total_spikes = 0.0
        self.total_neurons = 0.0
        self.last_rate = 0.0

    def forward(self, x):
        current = self.intermediate_dense(x)
        mem = torch.zeros_like(current)
        spike_sum = torch.zeros_like(current)
        out_sum = torch.zeros(x.size(0), x.size(1), self.output_dense.out_features, device=x.device, dtype=x.dtype)
        beta = torch.clamp(self.beta, 0.0, 1.0).view(1, 1, -1)
        for _ in range(self.t_steps):
            mem = current + beta * mem
            spike = ArctanSpike.apply(mem - self.threshold, self.k_slope)
            mem = mem - spike * self.threshold
            spike_sum = spike_sum + spike
            out_sum = out_sum + self.output_dense(spike)
        rate = spike_sum / float(self.t_steps)
        self.last_rate = float(rate.detach().mean().item())
        self.total_spikes += float(spike_sum.detach().sum().item())
        self.total_neurons += float(spike_sum.numel() * self.t_steps)
        return out_sum / float(self.t_steps)

    def reset_tracker(self):
        self.total_spikes = 0.0
        self.total_neurons = 0.0
        self.last_rate = 0.0

    @property
    def firing_rate(self) -> float:
        return self.last_rate if self.total_neurons <= 0 else float(self.total_spikes / self.total_neurons)


class SpikeBERTStudent(nn.Module):
    def __init__(self, args: argparse.Namespace, checkpoint_path: Path | None):
        super().__init__()
        self.encoder = load_backbone(args)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, NUM_LABELS)
        self.spiking_layer_indices = top_layer_indices(self.encoder.config.num_hidden_layers, args.spiking_layers)
        self.spiking_ffns = nn.ModuleDict()
        hidden_size = self.encoder.config.hidden_size
        self.feature_projections = nn.ModuleDict({
            str(idx): nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.LayerNorm(hidden_size),
            )
            for idx in [3, 6, 9, 12]
            if idx <= self.encoder.config.num_hidden_layers
        })
        for idx, layer in enumerate(self.encoder.encoder.layer):
            if idx in self.spiking_layer_indices:
                self.spiking_ffns[str(idx)] = SpikeBERTFFN(layer.intermediate.dense, layer.output.dense, args)
        if checkpoint_path is not None and checkpoint_path.exists():
            state = safe_torch_load(checkpoint_path, "cpu")
            print_checkpoint_diagnostics("Student source", checkpoint_path, state)
            incompatible = self.load_state_dict(normalize_state_dict(state), strict=False)
            print_load_report("Student init", incompatible)

    def _extended_attention_mask(self, attention_mask, input_shape):
        try:
            return self.encoder.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)
        except TypeError:
            return self.encoder.get_extended_attention_mask(attention_mask, input_shape)

    def forward(self, input_ids, attention_mask, output_hidden_states: bool = False):
        input_shape = input_ids.size()
        extended_attention_mask = self._extended_attention_mask(attention_mask, input_shape)
        hidden_states = self.encoder.embeddings(input_ids=input_ids)
        all_hidden = [hidden_states] if output_hidden_states else None
        for idx, layer_module in enumerate(self.encoder.encoder.layer):
            if idx in self.spiking_layer_indices:
                attention_outputs = layer_module.attention(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    head_mask=None,
                    output_attentions=False,
                )
                attention_output = attention_outputs[0]
                ffn_output = self.spiking_ffns[str(idx)](attention_output)
                ffn_output = layer_module.output.dropout(ffn_output)
                hidden_states = layer_module.output.LayerNorm(ffn_output + attention_output)
            else:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    head_mask=None,
                    output_attentions=False,
                )[0]
            if output_hidden_states:
                all_hidden.append(hidden_states)
        logits = self.classifier(self.dropout(hidden_states[:, 0, :]))
        return {"logits": logits, "hidden_states": tuple(all_hidden) if output_hidden_states else None, "last_hidden_state": hidden_states}

    def reset_trackers(self):
        for block in self.spiking_ffns.values():
            block.reset_tracker()

    def get_firing_rates(self):
        return {int(idx): block.firing_rate for idx, block in self.spiking_ffns.items()}


# EquilibriumFFN: LIF layer trained via equilibrium-state gradients.
# Forward pass runs LIF simulation until ASR converges with no gradient.
# Backward pass uses single-step unrolling from equilibrium as a practical
# approximation of implicit differentiation (SpikingBERT, Bal & Sengupta 2023).
# This avoids storing the full T-step graph while giving a better gradient signal
# than a straight-through proxy. It is not full implicit differentiation via
# Anderson acceleration, but it is tractable for GPU training.
class EquilibriumFFN(nn.Module):
    def __init__(self, intermediate_dense, output_dense, layer_idx: int, args: argparse.Namespace):
        super().__init__()
        self.intermediate_dense = intermediate_dense
        self.output_dense = output_dense
        self.layer_idx = layer_idx
        self.t_conv = args.t_conv
        self.threshold = args.v_th
        self.gamma = args.gamma
        self.eps = args.eps
        self.k_slope = getattr(args, "k_slope", 25.0)
        self.last_record = ConvergenceRecord(layer_idx, 0, True, 0.0, 0.0)
        self.total_asr = 0.0
        self.total_batches = 0

    def forward(self, x):
        current = self.intermediate_dense(x)

        asr_star, record = run_lif_to_equilibrium(
            current.detach(), self.t_conv, self.threshold, self.gamma, self.eps
        )
        record.layer = self.layer_idx
        self.last_record = record
        self.total_asr += float(record.asr)
        self.total_batches += 1

        mem_at_eq = current + asr_star.detach() * self.threshold * (self.gamma - 1.0)
        k = getattr(self, "k_slope", 25.0)
        spike_grad = ArctanSpike.apply(mem_at_eq - self.threshold, k)
        asr_with_grad = asr_star.detach() + (spike_grad - spike_grad.detach())

        return self.output_dense(asr_with_grad)

    def reset_tracker(self):
        self.last_record = ConvergenceRecord(self.layer_idx, 0, True, 0.0, 0.0)
        self.total_asr = 0.0
        self.total_batches = 0

    @property
    def firing_rate(self):
        return float(self.last_record.asr) if self.total_batches <= 0 else float(self.total_asr / self.total_batches)


class EquilibriumPhoBERTStudent(nn.Module):
    def __init__(self, args: argparse.Namespace, checkpoint_path: Path | None):
        super().__init__()
        self.encoder = load_backbone(args)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, NUM_LABELS)
        self.spiking_layer_indices = top_layer_indices(self.encoder.config.num_hidden_layers, args.spiking_layers)
        self.equilibrium_ffns = nn.ModuleDict()
        hidden_size = self.encoder.config.hidden_size
        self.feature_projections = nn.ModuleDict({
            str(idx): nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.LayerNorm(hidden_size),
            )
            for idx in [3, 6, 9, 12]
            if idx <= self.encoder.config.num_hidden_layers
        })
        for idx, layer in enumerate(self.encoder.encoder.layer):
            if idx in self.spiking_layer_indices:
                self.equilibrium_ffns[str(idx)] = EquilibriumFFN(layer.intermediate.dense, layer.output.dense, idx, args)
        self._last_records = []
        if checkpoint_path is not None and checkpoint_path.exists():
            state = safe_torch_load(checkpoint_path, "cpu")
            print_checkpoint_diagnostics("Equilibrium student source", checkpoint_path, state)
            incompatible = self.load_state_dict(normalize_state_dict(state), strict=False)
            print_load_report("Equilibrium student init", incompatible)

    def _extended_attention_mask(self, attention_mask, input_shape):
        try:
            return self.encoder.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)
        except TypeError:
            return self.encoder.get_extended_attention_mask(attention_mask, input_shape)

    def forward(self, input_ids, attention_mask, output_hidden_states: bool = False):
        self._last_records = []
        input_shape = input_ids.size()
        extended_attention_mask = self._extended_attention_mask(attention_mask, input_shape)
        hidden_states = self.encoder.embeddings(input_ids=input_ids)
        all_hidden = [hidden_states] if output_hidden_states else None
        for idx, layer_module in enumerate(self.encoder.encoder.layer):
            if idx in self.spiking_layer_indices:
                attention_outputs = layer_module.attention(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    head_mask=None,
                    output_attentions=False,
                )
                attention_output = attention_outputs[0]
                ffn_output = self.equilibrium_ffns[str(idx)](attention_output)
                self._last_records.append(self.equilibrium_ffns[str(idx)].last_record)
                ffn_output = layer_module.output.dropout(ffn_output)
                hidden_states = layer_module.output.LayerNorm(ffn_output + attention_output)
            else:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    head_mask=None,
                    output_attentions=False,
                )[0]
            if output_hidden_states:
                all_hidden.append(hidden_states)
        logits = self.classifier(self.dropout(hidden_states[:, 0, :]))
        return {
            "logits": logits,
            "hidden_states": tuple(all_hidden) if output_hidden_states else None,
            "last_hidden_state": hidden_states,
            "convergence": list(self._last_records),
        }

    def reset_trackers(self):
        for block in self.equilibrium_ffns.values():
            block.reset_tracker()
        self._last_records = []

    def get_firing_rates(self):
        return {int(idx): block.firing_rate for idx, block in self.equilibrium_ffns.items()}


def main() -> None:
    args = parse_args()
    start_ts = time.time()
    print_path_info(args)
    configure_environment(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[START] {DESCRIPTION}", flush=True)
    print(f"[START] timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    log_marker("run_start", output_dir=args.output_dir, seeds=",".join(str(seed) for seed in args.seeds))
    print(f"[ARGS] {json.dumps(json_safe(vars(args)), ensure_ascii=False, indent=2)}", flush=True)
    print_startup_info()
    device = get_device()
    train_loader, dev_loader, test_loader, tokenizer = load_data(args)

    seed_reports = []
    if EXPERIMENT_KIND == "tier3":
        submodes = ["3a", "3b"] if args.hybrid_variant == "both" else [args.hybrid_variant]
    elif EXPERIMENT_KIND == "tier1":
        submodes = ["tier1"]
    else:
        submodes = [TIER2_VARIANT]
    total_seed_runs = len(submodes) * len(args.seeds)
    seed_run_index = 0
    auto_resume = not args.no_auto_resume and not args.force_rerun
    for submode in submodes:
        for seed in args.seeds:
            seed_run_index += 1
            args._active_submode = submode
            args._active_seed = seed
            args._active_seed_run = f"{seed_run_index}/{total_seed_runs}"
            log_marker(
                "seed_start",
                submode=submode,
                seed=seed,
                seed_run=args._active_seed_run,
                total_epochs=args.epochs,
                output_dir=args.output_dir,
            )
            save_status(args, seed, submode, status="seed_start", extra={"seed_run": args._active_seed_run})
            report_path = seed_report_path(args, submode, seed)
            if auto_resume and report_path.exists():
                try:
                    completed_report = json.loads(report_path.read_text(encoding="utf-8"))
                    seed_reports.append(completed_report)
                    log_marker("seed_skip_completed", submode=submode, seed=seed, seed_run=args._active_seed_run, report=report_path)
                    save_status(args, seed, submode, status="seed_skip_completed", extra={"report": str(report_path)})
                    continue
                except Exception as exc:
                    print(f"[RESUME] warning could_not_read_report={report_path} error={exc}", flush=True)
            set_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            bundle = build_model(args, device)
            bundle.tokenizer = tokenizer
            bundle.submode = submode
            bundle.scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and args.amp_dtype == "fp16")
            latest_ckpt = find_latest_epoch_checkpoint(args, submode, seed) if auto_resume else None
            stage1_history = []
            if latest_ckpt is None and should_run_stage1(args, submode):
                stage1_history = run_stage1(bundle, tokenizer, device)
            optimizer = torch.optim.AdamW(bundle.student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            total_steps = max(1, effective_len(train_loader, args.dry_run_batches) * args.epochs)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )
            best_dev_f1 = -1.0
            best_path = args.output_dir / f"{RUN_NAME}_{submode}_seed{seed}_best.pt"
            history = []
            start_epoch = 1
            if latest_ckpt is not None:
                checkpoint = load_training_checkpoint(latest_ckpt, bundle, optimizer, scheduler, device)
                start_epoch = min(args.epochs + 1, int(checkpoint.get("epoch", 0)) + 1)
                best_dev_f1 = float(checkpoint.get("best_dev_f1_weighted", -1.0))
                history = list(checkpoint.get("history", []))
                stage1_history = list(checkpoint.get("stage1_history", stage1_history))
                log_marker(
                    "resume_loaded",
                    submode=submode,
                    seed=seed,
                    seed_run=args._active_seed_run,
                    checkpoint=latest_ckpt,
                    start_epoch=f"{start_epoch}/{args.epochs}",
                    best_dev_f1_weighted=best_dev_f1,
                )
            for epoch in range(start_epoch, args.epochs + 1):
                epoch_start = time.time()
                save_status(
                    args,
                    seed,
                    submode,
                    epoch,
                    status="epoch_start",
                    extra={"seed_run": args._active_seed_run},
                )
                train_stats = train_one_epoch(bundle, train_loader, optimizer, scheduler, device, epoch, args.epochs)
                dev_metrics, _, _ = evaluate(bundle, dev_loader, device, "dev")
                history.append({"epoch": epoch, "train": train_stats, "dev": dev_metrics})
                epoch_elapsed = time.time() - epoch_start
                print(
                    f"[EPOCH END] epoch={epoch}/{args.epochs} train_loss={train_stats['loss']:.4f} "
                    f"dev_loss={dev_metrics['loss']:.4f} dev_f1={dev_metrics['f1_weighted']:.4f} "
                    f"dev_acc={dev_metrics['accuracy']:.4f} elapsed={epoch_elapsed:.1f}s",
                    flush=True,
                )
                log_marker(
                    "epoch_end",
                    submode=submode,
                    seed=seed,
                    seed_run=args._active_seed_run,
                    epoch=f"{epoch}/{args.epochs}",
                    train_loss=train_stats["loss"],
                    dev_f1_weighted=dev_metrics["f1_weighted"],
                    dev_accuracy=dev_metrics["accuracy"],
                    elapsed_sec=epoch_elapsed,
                )
                save_status(
                    args,
                    seed,
                    submode,
                    epoch,
                    train_stats,
                    dev_metrics,
                    status="epoch_end",
                    extra={"seed_run": args._active_seed_run, "epoch_elapsed_sec": epoch_elapsed},
                )
                if dev_metrics["f1_weighted"] > best_dev_f1:
                    best_dev_f1 = dev_metrics["f1_weighted"]
                    torch.save(bundle.student.state_dict(), best_path)
                    print(f"[CHECKPOINT] New best dev F1={best_dev_f1:.4f}, saved to {best_path}", flush=True)
                    log_marker(
                        "checkpoint_saved",
                        submode=submode,
                        seed=seed,
                        seed_run=args._active_seed_run,
                        epoch=f"{epoch}/{args.epochs}",
                        best_dev_f1_weighted=best_dev_f1,
                        path=best_path,
                    )
                save_training_checkpoint(
                    args,
                    bundle,
                    optimizer,
                    scheduler,
                    epoch,
                    submode,
                    seed,
                    best_dev_f1,
                    best_path,
                    history,
                    stage1_history,
                )
            if best_path.exists():
                bundle.student.load_state_dict(extract_model_state(safe_torch_load(best_path, map_location=device)))
            else:
                print(f"[CHECKPOINT] warning best checkpoint missing, using current weights path={best_path}", flush=True)
                torch.save(bundle.student.state_dict(), best_path)
            test_metrics, y_true, y_pred = evaluate(bundle, test_loader, device, "test")
            theoretical_energy = estimate_energy(bundle.student, args)
            gpu_energy = measure_gpu_energy(bundle, test_loader, device, args)
            seed_report = {
                "seed": seed,
                "submode": submode,
                "best_dev_f1_weighted": best_dev_f1,
                "test_metrics": test_metrics,
                "energy": {
                    "theoretical_neuromorphic": theoretical_energy,
                    "actual_gpu_a100": gpu_energy,
                },
                "history": history,
                "stage1_history": stage1_history,
                "checkpoint": str(best_path),
                "peak_gpu_memory_mb": peak_memory_mb(device),
            }
            save_json(seed_report, args.output_dir / f"{RUN_NAME}_{submode}_seed{seed}_report.json")
            seed_reports.append(seed_report)
            log_marker(
                "seed_end",
                submode=submode,
                seed=seed,
                seed_run=args._active_seed_run,
                best_dev_f1_weighted=best_dev_f1,
                test_f1_weighted=test_metrics["f1_weighted"],
                checkpoint=best_path,
            )
            save_status(
                args,
                seed,
                submode,
                args.epochs,
                status="seed_end",
                extra={
                    "seed_run": args._active_seed_run,
                    "best_dev_f1_weighted": best_dev_f1,
                    "test_f1_weighted": test_metrics["f1_weighted"],
                    "checkpoint": str(best_path),
                },
            )
    summary = summarize(seed_reports)
    final_report = {"config": json_safe(vars(args)), "seed_reports": seed_reports, "summary": summary}
    save_json(final_report, args.output_dir / "report.json")
    print(f"[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    total_elapsed = time.time() - start_ts
    log_marker("run_end", elapsed_sec=total_elapsed, report=args.output_dir / "report.json")
    print(f"[END] total_elapsed={total_elapsed:.1f}s report={args.output_dir / 'report.json'}", flush=True)


def require_runtime_dependencies() -> None:
    missing = []
    for name, obj in {
        "pandas": pd,
        "sklearn": accuracy_score,
        "transformers": AutoModel,
        "tqdm": tqdm,
    }.items():
        if obj is None:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing dependencies: {missing}. Install with: pip install -r uit/requirements-uit.txt")


def configure_environment(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir.expanduser().resolve()
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_ASSETS_CACHE"] = str(cache_dir / "assets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["UIT_AMP_DTYPE"] = args.amp_dtype
    os.environ["UIT_PROGRESS_BARS"] = "1" if args.progress_bars else "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "TRANSFORMERS_CACHE"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def print_path_info(args: argparse.Namespace) -> None:
    print(f"[PATH] __file__={Path(__file__).resolve()}", flush=True)
    print(f"[PATH] PROJECT_ROOT={infer_project_root()}", flush=True)
    print(f"[PATH] DATA_DIR={args.data_dir}", flush=True)
    print(f"[PATH] OUTPUT_DIR={args.output_dir}", flush=True)
    print(f"[PATH] CHECKPOINT_DIR={args.checkpoint_dir}", flush=True)
    print(f"[PATH] CACHE_DIR={args.cache_dir}", flush=True)


def print_startup_info() -> None:
    print(f"[PYTHON] {sys.version.replace(os.linesep, ' ')}", flush=True)
    print(f"[TORCH] {torch.__version__} cuda_build={torch.version.cuda}", flush=True)
    print(f"[CUDA] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[CUDA] CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')}", flush=True)
    print(f"[CUDA] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)
    print(f"[CUDA] UIT_AMP_DTYPE={os.environ.get('UIT_AMP_DTYPE')}", flush=True)
    print(f"[LOGGING] progress_bars={os.environ.get('UIT_PROGRESS_BARS')}", flush=True)
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)
        free = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"[GPU] Free VRAM: {free:.1f} GB", flush=True)
    else:
        print("[GPU] No CUDA found, running on CPU", flush=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_cuda_if_needed(device: torch.device, label: str) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
        print(f"[CUDA] sync ok after {label}", flush=True)


def load_tokenizer(args):
    print(f"[TOKENIZER] Loading {MODEL_NAME}", flush=True)
    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        cache_dir=str(args.cache_dir),
        local_files_only=args.local_files_only,
    )


def load_backbone(args):
    print(f"[BACKBONE] Loading {MODEL_NAME}", flush=True)
    return AutoModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        cache_dir=str(args.cache_dir),
        local_files_only=args.local_files_only,
    )


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def normalize_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def extract_model_state(state):
    if isinstance(state, dict) and "model_state" in state:
        return state["model_state"]
    return normalize_state_dict(state)


def epoch_checkpoint_path(args, submode: str, seed: int, epoch: int) -> Path:
    return args.output_dir / f"{RUN_NAME}_{submode}_seed{seed}_epoch{epoch:03d}.pt"


def seed_report_path(args, submode: str, seed: int) -> Path:
    return args.output_dir / f"{RUN_NAME}_{submode}_seed{seed}_report.json"


def find_latest_epoch_checkpoint(args, submode: str, seed: int) -> Path | None:
    patterns = [
        f"{RUN_NAME}_{submode}_seed{seed}_epoch*.pt",
        f"{RUN_NAME}_{submode}_seed{seed}_epoch*.pth",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in args.output_dir.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def cleanup_old_epoch_checkpoints(args, submode: str, seed: int, keep_path: Path) -> None:
    for path in args.output_dir.glob(f"{RUN_NAME}_{submode}_seed{seed}_epoch*.pt"):
        if path != keep_path:
            try:
                path.unlink()
            except OSError as exc:
                print(f"[CHECKPOINT] warning could_not_remove={path} error={exc}", flush=True)


def save_training_checkpoint(
    args,
    bundle,
    optimizer,
    scheduler,
    epoch: int,
    submode: str,
    seed: int,
    best_dev_f1: float,
    best_path: Path,
    history: list,
    stage1_history: list,
) -> Path:
    path = epoch_checkpoint_path(args, submode, seed, epoch)
    payload = {
        "format_version": 2,
        "run_name": RUN_NAME,
        "submode": submode,
        "seed": seed,
        "epoch": epoch,
        "total_epochs": args.epochs,
        "best_dev_f1_weighted": best_dev_f1,
        "best_path": str(best_path),
        "model_state": bundle.student.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": bundle.scaler.state_dict() if bundle.scaler is not None else None,
        "history": json_safe(history),
        "stage1_history": json_safe(stage1_history),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.save_optimizer_state:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    cleanup_old_epoch_checkpoints(args, submode, seed, path)
    log_marker("epoch_checkpoint_saved", submode=submode, seed=seed, epoch=f"{epoch}/{args.epochs}", path=path)
    return path


def load_training_checkpoint(path: Path, bundle, optimizer, scheduler, device):
    checkpoint = safe_torch_load(path, map_location=device)
    bundle.student.load_state_dict(extract_model_state(checkpoint))
    if isinstance(checkpoint, dict):
        optimizer_state = checkpoint.get("optimizer_state")
        scheduler_state = checkpoint.get("scheduler_state")
        scaler_state = checkpoint.get("scaler_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None and scheduler is not None:
            scheduler.load_state_dict(scheduler_state)
        if scaler_state is not None and bundle.scaler is not None:
            bundle.scaler.load_state_dict(scaler_state)
        return checkpoint
    return {"epoch": 0, "best_dev_f1_weighted": -1.0, "history": [], "stage1_history": []}


def checkpoint_hash_head(path: Path, num_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(num_bytes))
    return digest.hexdigest()


def print_checkpoint_diagnostics(prefix: str, checkpoint_path: Path, raw_state) -> None:
    path = Path(checkpoint_path)
    exists = path.exists()
    print(f"[MODEL] {prefix} checkpoint path: {path}", flush=True)
    print(f"[MODEL] {prefix} checkpoint exists={exists}", flush=True)
    if exists:
        stat = path.stat()
        print(f"[MODEL] {prefix} checkpoint size_bytes={stat.st_size}", flush=True)
        print(f"[MODEL] {prefix} checkpoint mtime={stat.st_mtime}", flush=True)
        print(f"[MODEL] {prefix} checkpoint sha256_first_1mb={checkpoint_hash_head(path)}", flush=True)

    state = normalize_state_dict(raw_state)
    if not isinstance(state, dict):
        print(f"[MODEL] {prefix} checkpoint object_type={type(state)}", flush=True)
        return

    keys = list(state.keys())
    print(f"[MODEL] {prefix} checkpoint num_keys={len(keys)}", flush=True)
    print(f"[MODEL] {prefix} checkpoint first20_keys={keys[:20]}", flush=True)
    print(
        f"[MODEL] {prefix} checkpoint has_encoder_embeddings="
        f"{any(key.startswith('encoder.embeddings') for key in keys)}",
        flush=True,
    )
    print(f"[MODEL] {prefix} checkpoint has_classifier.weight={'classifier.weight' in state}", flush=True)
    print(f"[MODEL] {prefix} checkpoint has_classifier.bias={'classifier.bias' in state}", flush=True)
    classifier_weight = state.get("classifier.weight")
    classifier_bias = state.get("classifier.bias")
    print(f"[MODEL] {prefix} classifier.weight shape={getattr(classifier_weight, 'shape', None)}", flush=True)
    print(f"[MODEL] {prefix} classifier.bias shape={getattr(classifier_bias, 'shape', None)}", flush=True)


def print_load_report(prefix: str, incompatible) -> None:
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    print(f"[MODEL] {prefix} load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[MODEL] {prefix} missing_keys_first20={missing[:20]}", flush=True)
    if unexpected:
        print(f"[MODEL] {prefix} unexpected_keys_first20={unexpected[:20]}", flush=True)


def resolve_teacher_checkpoint(args: argparse.Namespace) -> Path:
    if args.teacher_ckpt is not None:
        return args.teacher_ckpt
    checkpoint_dir = args.checkpoint_dir.expanduser()
    candidates = [
        checkpoint_dir / "phobert_vsfc" / "best_model.pth",
        checkpoint_dir / "models" / "phobert_vsfc" / "best_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Teacher checkpoint not found. Pass --teacher-ckpt.")


def segment_vi(text: str) -> str:
    if word_tokenize is None:
        return str(text)
    try:
        return word_tokenize(str(text), format="text")
    except Exception:
        return str(text)


def load_vsfc_split(args: argparse.Namespace, split: str):
    split_dir = args.data_dir / split
    sents = split_dir / "sents.txt"
    labels = split_dir / "sentiments.txt"
    if not sents.exists() or not labels.exists():
        raise FileNotFoundError(f"Missing files for split {split}: {split_dir}")
    cache_dir = args.data_dir.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_segmented.parquet"
    if not args.no_segmentation and cache_path.exists():
        print(f"[DATA] {split}: cache hit {cache_path}", flush=True)
        return pd.read_parquet(cache_path)
    texts = sents.read_text(encoding="utf-8").splitlines()
    ys = [int(x.strip()) for x in labels.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame({"text": texts, "label": ys})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    if not args.no_segmentation:
        print(f"[DATA] {split}: segmenting {len(df)} samples", flush=True)
        df["text"] = df["text"].apply(segment_vi)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as exc:
            print(f"[DATA] {split}: could not write parquet cache: {exc}", flush=True)
    return df


def print_split_summary(name: str, df) -> None:
    dist = Counter(int(x) for x in df["label"].tolist())
    print(f"[DATA] {name}: size={len(df)} labels={dict(sorted(dist.items()))}", flush=True)


def load_wiki_loader(args: argparse.Namespace, tokenizer):
    if load_dataset is None:
        raise RuntimeError("datasets is required for Stage 1. Install with: pip install -r uit/requirements-uit.txt")
    print("[WIKI] Loading wikimedia/wikipedia 20231101.vi", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", cache_dir=str(args.cache_dir))
    if args.wiki_max_samples > 0:
        ds = ds.select(range(min(args.wiki_max_samples, len(ds))))
    print(f"[WIKI] samples={len(ds)}", flush=True)
    return DataLoader(
        WikiDataset(ds, tokenizer, args.max_len, preprocess=True),
        batch_size=args.wiki_batch_size or args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def top_layer_indices(num_layers: int, spiking_layers: int) -> set[int]:
    spiking_layers = max(0, min(num_layers, spiking_layers))
    return set(range(num_layers - spiking_layers, num_layers))


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def autocast_context(device: torch.device):
    amp_dtype = os.environ.get("UIT_AMP_DTYPE", "bf16").lower()
    if device.type != "cuda" or amp_dtype == "off":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def to_device(batch: dict, device: torch.device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def should_stop(step: int, dry_run_batches: int) -> bool:
    return dry_run_batches > 0 and step > dry_run_batches


def effective_len(loader, dry_run_batches: int) -> int:
    return min(len(loader), dry_run_batches) if dry_run_batches > 0 else len(loader)


def make_iterator(loader, desc: str):
    if tqdm is None:
        return loader
    disable_progress = os.environ.get("UIT_PROGRESS_BARS") != "1" and not sys.stdout.isatty()
    return tqdm(
        loader,
        desc=desc,
        file=sys.stdout,
        ncols=80,
        mininterval=10.0,
        leave=False,
        disable=disable_progress,
    )


def kd_logits_loss(student_logits, teacher_logits, temperature: float):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def alignment_layers(hidden_count: int) -> list[int]:
    return [idx for idx in (3, 6, 9, 12) if idx <= hidden_count - 1]


def feature_alignment_loss(student_hidden, teacher_hidden, cls_only: bool, student_model=None):
    """
    Compute MSE alignment loss between student and teacher hidden states.

    If available, each selected student hidden state is first mapped through a
    learnable projection belonging to the student model.
    """
    layers = alignment_layers(min(len(student_hidden), len(teacher_hidden)))
    if not layers:
        return student_hidden[-1].new_tensor(0.0)
    losses = []
    projections = getattr(student_model, "feature_projections", {}) if student_model is not None else {}
    for idx in layers:
        s_val = student_hidden[idx].float()
        t_val = teacher_hidden[idx].detach().float()
        if str(idx) in projections:
            s_val = projections[str(idx)](s_val)
        if cls_only:
            s_val = s_val[:, 0, :]
            t_val = t_val[:, 0, :]
        losses.append(F.mse_loss(s_val, t_val))
    return torch.stack(losses).mean()


def embedding_alignment_loss(student_hidden, teacher_hidden):
    return F.mse_loss(student_hidden[0].float(), teacher_hidden[0].detach().float())


def needs_feature_alignment() -> bool:
    return TIER2_VARIANT in {"b", "c"} or EXPERIMENT_KIND in {"tier1", "tier3"}


def compute_stage2_loss(student_out, teacher_out, labels, args, student_model=None):
    ce = F.cross_entropy(student_out["logits"].float(), labels)
    kd = kd_logits_loss(student_out["logits"].float(), teacher_out["logits"].float(), args.temp_kd)
    feature = ce.new_tensor(0.0)
    embedding = ce.new_tensor(0.0)

    if EXPERIMENT_KIND == "tier2" and TIER2_VARIANT == "a":
        loss = args.alpha_kd * ce + (1.0 - args.alpha_kd) * kd
    elif EXPERIMENT_KIND == "tier2" and TIER2_VARIANT == "b":
        feature = feature_alignment_loss(
            student_out["hidden_states"],
            teacher_out["hidden_states"],
            cls_only=True,
            student_model=student_model,
        )
        loss = args.alpha_kd * ce + (1.0 - args.alpha_kd) * kd + args.feature_weight * feature
    elif EXPERIMENT_KIND == "tier2" and TIER2_VARIANT == "c":
        feature = feature_alignment_loss(
            student_out["hidden_states"],
            teacher_out["hidden_states"],
            cls_only=False,
            student_model=student_model,
        )
        embedding = embedding_alignment_loss(student_out["hidden_states"], teacher_out["hidden_states"])
        loss = 0.1 * feature + 0.1 * embedding + kd + 0.1 * ce
    elif EXPERIMENT_KIND == "tier3" and getattr(args, "_active_submode", "3b") == "3a":
        loss = ce + kd
    else:
        feature = feature_alignment_loss(
            student_out["hidden_states"],
            teacher_out["hidden_states"],
            cls_only=False,
            student_model=student_model,
        )
        embedding = embedding_alignment_loss(student_out["hidden_states"], teacher_out["hidden_states"])
        loss = 0.1 * feature + 0.1 * embedding + kd + 0.1 * ce
    return loss, {"ce": float(ce.item()), "kd": float(kd.item()), "feature": float(feature.item()), "embedding": float(embedding.item())}


def run_lif_to_equilibrium(current, max_steps: int, threshold: float, gamma: float, eps: float):
    mem = torch.zeros_like(current)
    prev_spike = torch.zeros_like(current)
    running_rate = torch.zeros_like(current)
    last_delta = float("inf")
    converged = False
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            mem = current + gamma * mem - prev_spike * threshold
            spike = (mem >= threshold).to(current.dtype)
            running_next = running_rate + (spike - running_rate) / float(step)
            last_delta = torch.norm((running_next - running_rate).flatten(1), dim=1).mean().item()
            running_rate = running_next
            prev_spike = spike
            if step > 1 and last_delta <= eps:
                converged = True
                break
    return running_rate.detach(), ConvergenceRecord(-1, step, converged, float(last_delta), float(running_rate.mean().item()))


def has_unconverged(records) -> bool:
    return any(not record.converged for record in records)


def train_stage1_epoch(bundle, wiki_loader, optimizer, scheduler, device, epoch: int, total_epochs: int):
    args = bundle.args
    student = bundle.student
    teacher = bundle.teacher
    student.train()
    teacher.eval()
    total_loss = 0.0
    steps = 0
    skipped_steps = 0
    total_batches = effective_len(wiki_loader, args.dry_run_batches)
    log_marker(
        "stage1_epoch_start",
        phase="stage1",
        submode=getattr(args, "_active_submode", None),
        seed=getattr(args, "_active_seed", None),
        seed_run=getattr(args, "_active_seed_run", None),
        epoch=f"{epoch}/{total_epochs}",
        total_batches=total_batches,
    )
    print(f"=== Stage 1 Epoch {epoch}/{total_epochs} ===", flush=True)
    for step, batch in enumerate(make_iterator(wiki_loader, f"stage1 ep{epoch}"), start=1):
        if should_stop(step, args.dry_run_batches):
            break
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_out = teacher(batch["input_ids"], batch["attention_mask"], output_hidden_states=True)
        with autocast_context(device):
            student_out = student(batch["input_ids"], batch["attention_mask"], output_hidden_states=True)
            if EXPERIMENT_KIND in {"tier1", "tier3"} and has_unconverged(student_out.get("convergence", [])):
                skipped_steps += 1
                continue
            loss = feature_alignment_loss(
                student_out["hidden_states"],
                teacher_out["hidden_states"],
                cls_only=False,
                student_model=student,
            )
            loss = loss + embedding_alignment_loss(student_out["hidden_states"], teacher_out["hidden_states"])
        bundle.scaler.scale(loss).backward()
        bundle.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        bundle.scaler.step(optimizer)
        bundle.scaler.update()
        scheduler.step()
        steps += 1
        total_loss += float(loss.item())
    return {"loss": total_loss / max(1, steps), "steps": steps, "skipped_steps": skipped_steps}


def run_stage1(bundle, tokenizer, device):
    args = bundle.args
    if args.skip_stage1:
        print("[STAGE1] skipped by --skip-stage1", flush=True)
        return []
    print("[STAGE1] building Wikipedia loader", flush=True)
    wiki_loader = load_wiki_loader(args, tokenizer)
    optimizer = torch.optim.AdamW(bundle.student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, effective_len(wiki_loader, args.dry_run_batches) * args.stage1_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    history = []
    for epoch in range(1, args.stage1_epochs + 1):
        history.append(train_stage1_epoch(bundle, wiki_loader, optimizer, scheduler, device, epoch, args.stage1_epochs))
    return history


def should_run_stage1(args, submode: str) -> bool:
    if EXPERIMENT_KIND == "tier2":
        return TIER2_VARIANT == "c" and not args.skip_stage1
    if EXPERIMENT_KIND == "tier1":
        return not args.skip_stage1
    if EXPERIMENT_KIND == "tier3":
        return submode == "3b" and not args.skip_stage1
    return False


def augment_input_ids(input_ids, attention_mask, tokenizer):
    out = input_ids.clone()
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.unk_token_id
    valid = attention_mask.bool()
    for sid in tokenizer.all_special_ids:
        valid &= out.ne(sid)
    random_mask = torch.rand_like(out.float()).lt(0.1) & valid
    out[random_mask] = int(mask_id)
    return out


def compute_metrics(y_true, y_pred):
    if not y_true:
        return {"accuracy": 0.0, "f1_weighted": 0.0, "f1_macro": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def print_classification_report(split_name: str, y_true, y_pred):
    if not y_true:
        print(f"[REPORT] {split_name}: no samples", flush=True)
        return
    print(f"[REPORT] {split_name}", flush=True)
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, digits=4, zero_division=0), flush=True)


def estimate_energy(student, args):
    """
    Estimate theoretical energy on 45nm neuromorphic hardware.

    ANN layers use MAC operations (4.6 pJ), SNN layers use AC operations
    (0.9 pJ), and only FFN layers are replaced by SNN dynamics.
    """
    cfg = student.encoder.config
    firing_rates = student.get_firing_rates()
    seq = args.max_len
    hidden = cfg.hidden_size
    intermediate = cfg.intermediate_size
    t_steps = args.t_steps if EXPERIMENT_KIND == "tier2" else args.t_conv

    layers_detail = []
    total_ann_energy = 0.0
    total_snn_energy = 0.0

    emb_flops = seq * hidden
    emb_energy = emb_flops * MAC_PJ
    total_ann_energy += emb_energy

    spiking_indices = getattr(student, "spiking_layer_indices", set())
    for idx in range(cfg.num_hidden_layers):
        attn_flops = (4 * seq * hidden * hidden) + (2 * seq * seq * hidden)
        attn_energy = attn_flops * MAC_PJ
        total_ann_energy += attn_energy

        ffn_flops = 2 * seq * hidden * intermediate
        is_spiking = idx in spiking_indices
        rate = float(firing_rates.get(idx, 0.0 if is_spiking else 1.0))

        if is_spiking:
            sops = t_steps * rate * ffn_flops
            ffn_energy = sops * AC_PJ
            total_snn_energy += ffn_energy
        else:
            ffn_energy = ffn_flops * MAC_PJ
            total_ann_energy += ffn_energy

        layers_detail.append({
            "layer_idx": idx,
            "is_spiking": is_spiking,
            "firing_rate": rate,
            "attn_flops": attn_flops,
            "attn_energy_pj": attn_energy,
            "ffn_flops": ffn_flops,
            "ffn_energy_pj": ffn_energy,
        })

    cls_flops = hidden * NUM_LABELS
    cls_energy = cls_flops * MAC_PJ
    total_ann_energy += cls_energy

    total_energy = total_ann_energy + total_snn_energy
    return {
        "note": (
            "Theoretical estimate on 45nm neuromorphic hardware "
            "(Horowitz 2014). MAC=4.6pJ, AC=0.9pJ. "
            "Actual GPU energy measured separately via pynvml."
        ),
        "total_ann_energy_pj": total_ann_energy,
        "total_snn_energy_pj": total_snn_energy,
        "total_energy_pj": total_energy,
        "energy_reduction_pct": (
            100.0 * (1.0 - total_energy / total_ann_energy)
            if total_ann_energy > 0 else 0.0
        ),
        "layers": layers_detail,
    }


def measure_gpu_energy(model, loader, device, args):
    """
    Measure actual GPU power consumption during inference on NVIDIA GPUs.
    """
    if not _PYNVML_AVAILABLE or device.type != "cuda":
        return {
            "note": "pynvml not available or not running on CUDA device.",
            "energy_per_sample_joules": None,
            "avg_power_watts": None,
            "throughput_samples_per_sec": None,
        }

    student = model.student
    student.eval()
    student.reset_trackers()

    try:
        handle = _pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    except Exception as exc:
        return {
            "note": f"pynvml handle error: {exc}",
            "energy_per_sample_joules": None,
            "avg_power_watts": None,
            "throughput_samples_per_sec": None,
        }

    power_readings_mw = []
    total_samples = 0
    t_start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if should_stop(step, args.dry_run_batches):
                break
            batch = to_device(batch, device)
            try:
                power_readings_mw.append(float(_pynvml.nvmlDeviceGetPowerUsage(handle)))
            except Exception:
                pass
            with autocast_context(device):
                _ = student(batch["input_ids"], batch["attention_mask"], output_hidden_states=False)
            total_samples += batch["input_ids"].size(0)

    elapsed = time.time() - t_start
    if not power_readings_mw or total_samples == 0:
        return {
            "note": "No power readings collected.",
            "energy_per_sample_joules": None,
            "avg_power_watts": None,
            "throughput_samples_per_sec": None,
        }

    avg_power_w = (sum(power_readings_mw) / len(power_readings_mw)) / 1000.0
    total_energy_j = avg_power_w * elapsed
    return {
        "note": (
            "Actual GPU energy on NVIDIA A100. Higher than teacher is expected "
            "for T-step SNN simulation on GPU; neuromorphic theory is reported separately."
        ),
        "avg_power_watts": avg_power_w,
        "total_energy_joules": total_energy_j,
        "energy_per_sample_joules": total_energy_j / total_samples,
        "elapsed_seconds": elapsed,
        "total_samples": total_samples,
        "throughput_samples_per_sec": total_samples / elapsed if elapsed > 0 else None,
    }


def peak_memory_mb(device):
    return None if device.type != "cuda" else float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def save_status(
    args,
    seed,
    submode,
    epoch=None,
    train_stats=None,
    dev_metrics=None,
    status="running",
    extra=None,
):
    marker = f"{RUN_NAME}:{submode}:seed{seed}"
    if epoch is not None:
        marker = f"{marker}:epoch{epoch}"
    payload = {
        "marker": marker,
        "run_name": RUN_NAME,
        "status": status,
        "seed": seed,
        "submode": submode,
        "seed_run": getattr(args, "_active_seed_run", None),
        "epoch": epoch,
        "total_epochs": args.epochs,
        "output_dir": str(args.output_dir),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if train_stats is not None:
        payload["train"] = train_stats
    if dev_metrics is not None:
        payload["dev"] = dev_metrics
    if extra:
        payload.update(json_safe(extra))
    save_json(payload, args.output_dir / "status_current.json")


def summarize(seed_reports):
    summary = {}
    for key in ("accuracy", "f1_weighted", "f1_macro"):
        values = [float(item["test_metrics"][key]) for item in seed_reports]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
