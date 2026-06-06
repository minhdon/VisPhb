"""Model loading and inference utilities for the ViSPhB Streamlit demo."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from registry import DEFAULT_MODEL_NAME, ModelVariant

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LABELS: Tuple[str, ...] = ("negative", "neutral", "positive")
LABELS_VI: Dict[str, str] = {
    "negative": "Tiêu cực",
    "neutral": "Trung tính",
    "positive": "Tích cực",
}
CHECKPOINT_KEYS: Tuple[str, ...] = (
    "model_state_dict",
    "student_state_dict",
    "state_dict",
    "model",
)


class InferenceError(RuntimeError):
    """Raised when one model cannot be loaded or executed."""


def _cache_resource(func):
    if st is None:
        return func
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is None:
            return func
    except Exception:
        pass
    return st.cache_resource(show_spinner=False)(func)


@dataclass
class RuntimeBundle:
    model: nn.Module
    tokenizer: Any
    device: torch.device
    load_info: Dict[str, Any]


@dataclass
class PredictionResult:
    label: str
    label_vi: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    device: str
    checkpoint_path: str
    runtime_firing_rate: Optional[float] = None


class PhoBERTSentimentClassifier(nn.Module):
    """Small inference wrapper matching the baseline checkpoint keys."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dropout_rate: float = 0.25,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        try:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                attn_implementation="eager",
            )
        except TypeError:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, len(LABELS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_rep))


def choose_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise InferenceError("CUDA được chọn nhưng torch.cuda.is_available() = False.")
    return torch.device(choice)


def safe_torch_load(path: Path, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in CHECKPOINT_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                tensors = {k: v for k, v in value.items() if isinstance(k, str) and torch.is_tensor(v)}
                if tensors:
                    return tensors
        tensors = {k: v for k, v in checkpoint.items() if isinstance(k, str) and torch.is_tensor(v)}
        if tensors:
            return tensors
    raise InferenceError(
        "Không tìm thấy state_dict tensor trong checkpoint. "
        f"App hỗ trợ các key: {', '.join(CHECKPOINT_KEYS)}."
    )


def normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "student.")
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value
    return normalized


def checkpoint_config(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("config"), Mapping):
        return dict(checkpoint["config"])
    return {}


def import_module_from_path(module_path: Path):
    if not module_path.exists():
        raise InferenceError(
            f"Không tìm thấy file kiến trúc: {module_path}. "
            "TODO: nối registry với class model thật hoặc đặt script vào thư mục code/."
        )
    module_name = f"visphb_demo_{module_path.stem}_{abs(hash(str(module_path.resolve())))}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise InferenceError(f"Không tạo được import spec cho {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def apply_config_values(
    config: Any,
    saved_config: Mapping[str, Any],
    seed: Optional[int],
    spiking_layers: Optional[int],
    max_length: int,
    local_files_only: bool,
    model_name: str,
) -> Any:
    for key, value in saved_config.items():
        if hasattr(config, key):
            setattr(config, key, value)

    if seed is not None and hasattr(config, "seed"):
        setattr(config, "seed", int(seed))
    if spiking_layers is not None and hasattr(config, "spiking_layers"):
        setattr(config, "spiking_layers", int(spiking_layers))
    if hasattr(config, "max_length"):
        setattr(config, "max_length", int(max_length))
    if hasattr(config, "max_len"):
        setattr(config, "max_len", int(max_length))
    if hasattr(config, "local_files_only"):
        setattr(config, "local_files_only", bool(local_files_only))
    if hasattr(config, "model_name"):
        setattr(config, "model_name", model_name)
    if hasattr(config, "amp"):
        setattr(config, "amp", False)
    return config


def _build_teacher(
    checkpoint: Any,
    local_files_only: bool,
    model_name: str,
) -> nn.Module:
    saved = checkpoint_config(checkpoint)
    dropout = 0.25
    if isinstance(saved.get("dropout"), (int, float)):
        dropout = float(saved["dropout"])
    model = PhoBERTSentimentClassifier(
        model_name=model_name,
        dropout_rate=dropout,
        local_files_only=local_files_only,
    )
    return model


def _build_student(
    variant: ModelVariant,
    checkpoint: Any,
    max_length: int,
    local_files_only: bool,
    model_name: str,
) -> nn.Module:
    if variant.module_path is None or variant.class_name is None or variant.config_class is None:
        raise InferenceError(
            "Registry chưa có architecture builder cho model này. "
            "TODO: thêm module_path, config_class và class_name trong registry.py."
        )

    try:
        module = import_module_from_path(variant.module_path)
    except Exception as exc:
        raise InferenceError(f"Không import được kiến trúc {variant.module_path}: {exc}") from exc

    config_cls = getattr(module, variant.config_class, None)
    model_cls = getattr(module, variant.class_name, None)
    if config_cls is None or model_cls is None:
        raise InferenceError(
            f"Không tìm thấy {variant.config_class}/{variant.class_name} trong {variant.module_path}. "
            "TODO: cập nhật registry.py để trỏ đúng class model thật."
        )

    config = apply_config_values(
        config_cls(),
        checkpoint_config(checkpoint),
        variant.seed,
        variant.spiking_layers,
        max_length,
        local_files_only,
        model_name,
    )

    if variant.uses_stage_config:
        build_stage1_config = getattr(module, "build_stage1_config", None)
        if build_stage1_config is None:
            raise InferenceError(
                f"{variant.module_path.name} không có build_stage1_config(). "
                "TODO: thêm adapter config cho architecture two-stage."
            )
        config = build_stage1_config(config)

    try:
        return model_cls(config)
    except Exception as exc:
        raise InferenceError(f"Không khởi tạo được model {variant.display_name}: {exc}") from exc


def _checkpoint_mtime(path: Optional[Path]) -> float:
    if path is None or not path.exists():
        return 0.0
    return path.stat().st_mtime


@_cache_resource
def load_runtime_cached(
    variant_id: str,
    family_key: str,
    checkpoint_path: str,
    checkpoint_mtime: float,
    module_path: str,
    class_name: str,
    config_class: str,
    uses_stage_config: bool,
    seed: Optional[int],
    spiking_layers: Optional[int],
    max_length: int,
    device_choice: str,
    local_files_only: bool,
    model_name: str,
) -> RuntimeBundle:
    del variant_id, checkpoint_mtime  # cache key only

    device = choose_device(device_choice)
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise InferenceError(f"Không tìm thấy checkpoint: {ckpt_path}")

    checkpoint = safe_torch_load(ckpt_path, map_location="cpu")

    variant = ModelVariant(
        family_key=family_key,
        family_label=family_key,
        short_label=family_key,
        description="",
        architecture=family_key,
        model_root=Path("."),
        run_dir=ckpt_path.parent,
        checkpoint_path=ckpt_path,
        report_paths=(),
        seed=seed,
        spiking_layers=spiking_layers,
        module_path=Path(module_path) if module_path else None,
        class_name=class_name or None,
        config_class=config_class or None,
        uses_stage_config=uses_stage_config,
        model_name=model_name,
    )

    if family_key == "teacher":
        model = _build_teacher(checkpoint, local_files_only, model_name)
    else:
        model = _build_student(variant, checkpoint, max_length, local_files_only, model_name)

    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        use_fast=False,
    )

    load_info = {
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
        "checkpoint_path": str(ckpt_path),
        "device": str(device),
    }
    return RuntimeBundle(model=model, tokenizer=tokenizer, device=device, load_info=load_info)


def load_runtime(
    variant: ModelVariant,
    max_length: int,
    device_choice: str,
    local_files_only: bool,
    model_name: str = DEFAULT_MODEL_NAME,
) -> RuntimeBundle:
    if variant.checkpoint_path is None:
        raise InferenceError(f"Không tìm thấy checkpoint cho {variant.display_name}.")
    return load_runtime_cached(
        variant.variant_id,
        variant.family_key,
        str(variant.checkpoint_path),
        _checkpoint_mtime(variant.checkpoint_path),
        "" if variant.module_path is None else str(variant.module_path),
        variant.class_name or "",
        variant.config_class or "",
        variant.uses_stage_config,
        variant.seed,
        variant.spiking_layers,
        int(max_length),
        device_choice,
        bool(local_files_only),
        model_name,
    )


def _extract_logits(outputs: Any) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, Mapping) and torch.is_tensor(outputs.get("logits")):
        return outputs["logits"]
    if hasattr(outputs, "logits") and torch.is_tensor(outputs.logits):
        return outputs.logits
    raise InferenceError("Model forward không trả logits ở dạng tensor.")


def predict_text(
    variant: ModelVariant,
    text: str,
    max_length: int,
    device_choice: str,
    local_files_only: bool,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[PredictionResult, Dict[str, Any]]:
    if not text.strip():
        raise InferenceError("Câu nhập đang trống.")

    runtime = load_runtime(variant, max_length, device_choice, local_files_only, model_name)
    tokenizer = runtime.tokenizer
    model = runtime.model
    device = runtime.device

    encoded = tokenizer(
        text,
        max_length=int(max_length),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    if hasattr(model, "reset_firing_statistics"):
        try:
            model.reset_firing_statistics()
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        logits = _extract_logits(outputs)
        probs = F.softmax(logits, dim=-1).detach().cpu()[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0

    probabilities = {label: float(probs[index]) for index, label in enumerate(LABELS)}
    pred_index = int(torch.argmax(probs).item())
    label = LABELS[pred_index]

    runtime_firing_rate: Optional[float] = None
    if hasattr(model, "mean_firing_rate"):
        try:
            runtime_firing_rate = float(model.mean_firing_rate())
        except Exception:
            runtime_firing_rate = None

    result = PredictionResult(
        label=label,
        label_vi=LABELS_VI[label],
        confidence=float(probs[pred_index]),
        probabilities=probabilities,
        latency_ms=latency_ms,
        device=str(device),
        checkpoint_path=str(variant.checkpoint_path),
        runtime_firing_rate=runtime_firing_rate,
    )
    return result, runtime.load_info
