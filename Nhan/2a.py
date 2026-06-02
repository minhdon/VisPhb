# %% [markdown]
# # tier2_variant_a_logit_kd Kaggle Notebook
# 
# Converted from `uit/uit-train/entrypoints/tier2_variant_a_logit_kd.py`.
# 
# Default cells run full training: 3 seeds, 5 epochs, max_len=256, and no dry-run batch cap. Optional smoke test is disabled by default.
# 

# %%
# Cell 1 - Environment check
import os
import sys
import subprocess
from pathlib import Path

print("Python:", sys.version.replace("\n", " "))
print("cwd:", Path.cwd())
print("IS_KAGGLE:", Path("/kaggle/input").exists())

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Torch import/check failed:", repr(exc))

if Path("/kaggle/input").exists():
    print("\n/kaggle/input:")
    for item in sorted(Path("/kaggle/input").iterdir()):
        print(" -", item)

print("\nnvidia-smi:")
try:
    result = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
    print(result.stdout or result.stderr)
except Exception as exc:
    print("nvidia-smi failed:", repr(exc))


# %%
# Cell 2 - Install/import dependencies
# Kaggle usually has torch, pandas, sklearn, tqdm, transformers.
# Optional packages are installed only if INSTALL_MISSING_OPTIONAL=True.
import importlib.util
import subprocess
import sys

INSTALL_MISSING_OPTIONAL = False
OPTIONAL_PIP_PACKAGES = {
    "underthesea": "underthesea",
    "datasets": "datasets",
    "pynvml": "nvidia-ml-py",
}

missing_optional = [
    pip_name for module_name, pip_name in OPTIONAL_PIP_PACKAGES.items()
    if importlib.util.find_spec(module_name) is None
]
if missing_optional:
    print("Missing optional packages:", missing_optional)
    print("To install in Kaggle, set INSTALL_MISSING_OPTIONAL=True and rerun this cell.")
    if INSTALL_MISSING_OPTIONAL:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing_optional])

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

try:
    from datasets import load_from_disk
except Exception:
    load_from_disk = None

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

print("word_tokenize available:", word_tokenize is not None)
print("datasets/load_from_disk available:", load_from_disk is not None)
print("pynvml available:", _PYNVML_AVAILABLE)


# %%
# Cell 3 - Config
from pathlib import Path
from types import SimpleNamespace

IS_KAGGLE = Path("/kaggle/input").exists()
KAGGLE_INPUT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working") if IS_KAGGLE else Path.cwd()

# Fill these two names when you upload Kaggle datasets.
DATASET_NAME = "<dataset-name>"
MODEL_DATASET_NAME = "<model-dataset-name>"

EXPERIMENT_KIND = "tier2"
TIER2_VARIANT = "a"
RUN_NAME = "tier2_variant_a_logit_kd"
DESCRIPTION = f"UIT Kaggle notebook: {RUN_NAME}"

MODEL_NAME = "vinai/phobert-base-v2"
NUM_LABELS = 3
LABEL_NAMES = ["negative", "neutral", "positive"]
MAC_PJ = 4.6
AC_PJ = 0.9
MARKER_PREFIX = "[MARKER]"

# Full training defaults. Use Cell 12 for an optional limited-batch smoke test.
SEEDS = [42, 52, 62]
EPOCHS = 5
BATCH_SIZE = 8
MAX_LEN = 256
DRY_RUN_BATCHES = 0
NUM_WORKERS = 2

LR = 2e-5
WEIGHT_DECAY = 0.01
AMP_DTYPE = "bf16" if torch.cuda.is_available() else "off"  # "bf16", "fp16", or "off"
PROGRESS_BARS = False
LOCAL_FILES_ONLY = False
NO_SEGMENTATION = False

SPIKING_LAYERS = 12
T_STEPS = 4
TEMP_KD = 4.0
ALPHA_KD = 0.5
BETA = 0.9
K_SLOPE = 25.0
FEATURE_WEIGHT = 0.1

# Stage 1 runs only for Tier1 / Tier2-C / Tier3. Set SKIP_STAGE1=True if Wiki is not attached.
SKIP_STAGE1 = not (EXPERIMENT_KIND in {"tier1", "tier3"} or (EXPERIMENT_KIND == "tier2" and TIER2_VARIANT == "c"))
STAGE1_EPOCHS = 3 if EXPERIMENT_KIND == "tier3" else 5
WIKI_MAX_SAMPLES = 0
WIKI_BATCH_SIZE = None
AUGMENT_STAGE2 = False

# Tier1/Tier3 equilibrium knobs.
T_CONV = 80
V_TH = 1.0
SSA_V_TH = 0.25
GAMMA = 0.99
EPS = 1e-3
HYBRID_VARIANT = "3b"

OUTPUT_DIR = WORKING_DIR / "outputs" / RUN_NAME
CACHE_DIR = WORKING_DIR / "cache" / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _valid_dataset_name(name: str) -> bool:
    return bool(name) and not name.startswith("<")

def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return Path(paths[0]) if paths else None

def _find_uit_vsfc(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = []
    for p in root.rglob("uit-vsfc"):
        if (p / "train" / "sents.txt").exists() and (p / "train" / "sentiments.txt").exists():
            candidates.append(p)
    for p in root.iterdir():
        if (p / "train" / "sents.txt").exists() and (p / "train" / "sentiments.txt").exists():
            candidates.append(p)
    return sorted(candidates, key=lambda x: len(str(x)))[0] if candidates else None

def _find_teacher_ckpt(root: Path) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob("best_model.pth"))
    preferred = [p for p in matches if "phobert_vsfc" in p.parts]
    return (preferred or matches)[0] if matches else None

def _is_hf_disk_dataset(path: Path) -> bool:
    return path.is_dir() and (path / "state.json").exists() and (path / "dataset_info.json").exists()

def _find_wiki_dataset(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = []
    for p in root.rglob("wiki_vi_20231101"):
        if _is_hf_disk_dataset(p):
            candidates.append(p)
    for p in root.rglob("state.json"):
        parent = p.parent
        if _is_hf_disk_dataset(parent) and "wiki" in parent.name.lower():
            candidates.append(parent)
    return sorted(set(candidates), key=lambda x: len(str(x)))[0] if candidates else None

data_candidates = []
teacher_candidates = []
wiki_candidates = []

if IS_KAGGLE:
    if _valid_dataset_name(DATASET_NAME):
        data_candidates += [KAGGLE_INPUT / DATASET_NAME / "uit-vsfc", KAGGLE_INPUT / DATASET_NAME]
    found_data = _find_uit_vsfc(KAGGLE_INPUT)
    if found_data:
        data_candidates.append(found_data)

    if _valid_dataset_name(MODEL_DATASET_NAME):
        teacher_candidates += [
            KAGGLE_INPUT / MODEL_DATASET_NAME / "phobert_vsfc" / "best_model.pth",
            KAGGLE_INPUT / MODEL_DATASET_NAME / "best_model.pth",
        ]
    found_ckpt = _find_teacher_ckpt(KAGGLE_INPUT)
    if found_ckpt:
        teacher_candidates.append(found_ckpt)

    if _valid_dataset_name(DATASET_NAME):
        wiki_candidates += [KAGGLE_INPUT / DATASET_NAME / "wiki_vi_20231101"]
    found_wiki = _find_wiki_dataset(KAGGLE_INPUT)
    if found_wiki:
        wiki_candidates.append(found_wiki)
else:
    project_root = Path.cwd()
    data_candidates += [project_root / "uit" / "uit-vsfc", project_root / "uit-vsfc"]
    teacher_candidates += [
        project_root / "uit" / "uit-models" / "phobert_vsfc" / "best_model.pth",
        project_root / "uit-models" / "phobert_vsfc" / "best_model.pth",
    ]
    wiki_candidates += [
        project_root / "wiki_vi_20231101",
        project_root / "uit" / "wiki_vi_20231101",
    ]

DATA_DIR = _first_existing(data_candidates) or (KAGGLE_INPUT / DATASET_NAME / "uit-vsfc")
TEACHER_CKPT = _first_existing(teacher_candidates) or (KAGGLE_INPUT / MODEL_DATASET_NAME / "phobert_vsfc" / "best_model.pth")
WIKI_LOCAL_DIR = _first_existing(wiki_candidates) or (KAGGLE_INPUT / DATASET_NAME / "wiki_vi_20231101")
CHECKPOINT_DIR = TEACHER_CKPT.parent.parent if TEACHER_CKPT.exists() and TEACHER_CKPT.parent.name == "phobert_vsfc" else TEACHER_CKPT.parent

def refresh_args(**overrides):
    cfg = dict(
        data_dir=Path(DATA_DIR),
        output_dir=Path(OUTPUT_DIR),
        checkpoint_dir=Path(CHECKPOINT_DIR),
        cache_dir=Path(CACHE_DIR),
        teacher_ckpt=Path(TEACHER_CKPT) if TEACHER_CKPT else None,
        local_files_only=LOCAL_FILES_ONLY,
        seeds=list(SEEDS),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        max_len=MAX_LEN,
        num_workers=NUM_WORKERS,
        dry_run_batches=DRY_RUN_BATCHES,
        log_every=100,
        progress_bars=PROGRESS_BARS,
        no_auto_resume=True,
        force_rerun=False,
        save_optimizer_state=False,
        min_free_disk_gb=2.0,
        amp_dtype=AMP_DTYPE,
        no_segmentation=NO_SEGMENTATION,
        t_steps=T_STEPS,
        spiking_layers=SPIKING_LAYERS,
        temp_kd=TEMP_KD,
        alpha_kd=ALPHA_KD,
        beta=BETA,
        k_slope=K_SLOPE,
        feature_weight=FEATURE_WEIGHT,
        stage1_epochs=STAGE1_EPOCHS,
        wiki_max_samples=WIKI_MAX_SAMPLES,
        wiki_batch_size=WIKI_BATCH_SIZE,
        wiki_local_dir=Path(WIKI_LOCAL_DIR) if WIKI_LOCAL_DIR else None,
        skip_stage1=SKIP_STAGE1,
        augment_stage2=AUGMENT_STAGE2,
        t_conv=T_CONV,
        v_th=V_TH,
        ssa_v_th=SSA_V_TH,
        gamma=GAMMA,
        eps=EPS,
        hybrid_variant=HYBRID_VARIANT,
    )
    cfg.update(overrides)
    args = SimpleNamespace(**cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["UIT_AMP_DTYPE"] = args.amp_dtype
    os.environ["UIT_PROGRESS_BARS"] = "1" if args.progress_bars else "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(args.cache_dir / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(os.environ["HF_HOME"]) / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    return args

args = refresh_args()
print("RUN_NAME:", RUN_NAME)
print("DATA_DIR:", DATA_DIR)
print("TEACHER_CKPT:", TEACHER_CKPT)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("WIKI_LOCAL_DIR:", WIKI_LOCAL_DIR)
print("EPOCHS:", EPOCHS, "SEEDS:", SEEDS, "DRY_RUN_BATCHES:", DRY_RUN_BATCHES)


# %%
# Cell 4 - Path diagnostics
def _safe_load_for_diagnostics(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)

def _suggest_paths():
    root = Path("/kaggle/input")
    if not root.exists():
        return
    print("\nPossible UIT-VSFC directories:")
    for p in root.rglob("uit-vsfc"):
        print(" -", p)
    for p in root.iterdir():
        if (p / "train" / "sents.txt").exists():
            print(" -", p)
    print("\nPossible teacher checkpoints:")
    for p in root.rglob("best_model.pth"):
        print(" -", p)

print("DATA_DIR:", DATA_DIR, "exists=", DATA_DIR.exists())
print("TEACHER_CKPT:", TEACHER_CKPT, "exists=", TEACHER_CKPT.exists())
print("OUTPUT_DIR:", OUTPUT_DIR, "exists=", OUTPUT_DIR.exists())

if not DATA_DIR.exists() or not TEACHER_CKPT.exists():
    _suggest_paths()

if not DATA_DIR.exists():
    raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}. Update DATASET_NAME or DATA_DIR in Cell 3.")
if not TEACHER_CKPT.exists():
    raise FileNotFoundError(f"TEACHER_CKPT not found: {TEACHER_CKPT}. Upload the fine-tuned PhoBERT checkpoint and update MODEL_DATASET_NAME or TEACHER_CKPT.")

print("Teacher checkpoint size MB:", TEACHER_CKPT.stat().st_size / 1024**2)
state = _safe_load_for_diagnostics(TEACHER_CKPT, map_location="cpu")
state_for_keys = state.get("model_state", state.get("state_dict", state)) if isinstance(state, dict) else state
if isinstance(state_for_keys, dict):
    keys = list(state_for_keys.keys())
    print("Checkpoint first 20 keys:", keys[:20])
    for key in ("classifier.weight", "module.classifier.weight"):
        if key in state_for_keys:
            print(f"{key} shape:", tuple(state_for_keys[key].shape))
            break
else:
    print("Checkpoint object type:", type(state_for_keys))
del state, state_for_keys
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# %%
# Cell 5 - Utility functions
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

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"

def estimate_tensor_bytes(value, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        storage_id = id(value.untyped_storage()) if hasattr(value, "untyped_storage") else id(value.storage())
        if storage_id in seen:
            return 0
        seen.add(storage_id)
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(estimate_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(estimate_tensor_bytes(item, seen) for item in value)
    return 0

def safe_torch_save(payload, path: Path, min_free_disk_gb: float = 2.0) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    estimated = estimate_tensor_bytes(payload)
    free = shutil.disk_usage(path.parent).free
    min_free = int(min_free_disk_gb * 1024**3)
    print(f"[CHECKPOINT] save_prepare path={path} estimated={format_bytes(estimated)} free={format_bytes(free)}")
    if free < min_free or (estimated and free < estimated + min_free):
        print(f"[CHECKPOINT][CRITICAL] skip_save path={path} free={format_bytes(free)} estimated={format_bytes(estimated)} min_free_after={format_bytes(min_free)}", flush=True)
        return False
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        return True
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"[CHECKPOINT][CRITICAL] save_failed path={path} error={exc}", flush=True)
        return False

def checkpoint_hash_head(path: Path, num_bytes: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        hasher.update(handle.read(num_bytes))
    return hasher.hexdigest()

def print_checkpoint_diagnostics(prefix: str, checkpoint_path: Path, raw_state) -> None:
    print(f"[MODEL] {prefix} checkpoint path: {checkpoint_path}", flush=True)
    print(f"[MODEL] {prefix} checkpoint exists={checkpoint_path.exists()}", flush=True)
    if checkpoint_path.exists():
        print(f"[MODEL] {prefix} checkpoint size_bytes={checkpoint_path.stat().st_size}", flush=True)
        print(f"[MODEL] {prefix} checkpoint sha256_first_1mb={checkpoint_hash_head(checkpoint_path)}", flush=True)
    state = extract_model_state(raw_state)
    if isinstance(state, dict):
        keys = list(state.keys())
        print(f"[MODEL] {prefix} checkpoint num_keys={len(keys)}", flush=True)
        print(f"[MODEL] {prefix} checkpoint first20_keys={keys[:20]}", flush=True)
        for key in ("classifier.weight", "module.classifier.weight"):
            if key in state:
                print(f"[MODEL] {prefix} {key} shape={tuple(state[key].shape)}", flush=True)

def print_load_report(prefix: str, incompatible) -> None:
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    print(f"[MODEL] {prefix} load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[MODEL] {prefix} missing_keys_first20={missing[:20]}", flush=True)
    if unexpected:
        print(f"[MODEL] {prefix} unexpected_keys_first20={unexpected[:20]}", flush=True)

def resolve_teacher_checkpoint(args) -> Path:
    path = Path(args.teacher_ckpt) if args.teacher_ckpt is not None else Path(TEACHER_CKPT)
    if not path.exists():
        raise FileNotFoundError(f"Fine-tuned teacher checkpoint not found: {path}")
    return path

def segment_vi(text: str) -> str:
    if word_tokenize is None:
        return str(text)
    try:
        return word_tokenize(str(text), format="text")
    except Exception:
        return str(text)

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
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, digits=4, zero_division=0, output_dict=True)
    weighted = report.get("weighted avg", {})
    print(
        f"[REPORT] {split_name}: weighted_precision={weighted.get('precision', 0.0):.4f} "
        f"weighted_recall={weighted.get('recall', 0.0):.4f} weighted_f1={weighted.get('f1-score', 0.0):.4f}",
        flush=True,
    )

def print_split_summary(name: str, df) -> None:
    dist = Counter(int(x) for x in df["label"].tolist())
    print(f"[DATA] {name}: size={len(df)} labels={dict(sorted(dist.items()))}", flush=True)

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

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(obj), handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[DISK][WARN] could_not_write_json path={path} error={exc}", flush=True)

def summarize(seed_reports):
    summary = {}
    for key in ("accuracy", "f1_weighted", "f1_macro"):
        values = [float(item["test_metrics"][key]) for item in seed_reports if "test_metrics" in item]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary

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
    progress_enabled = os.environ.get("UIT_PROGRESS_BARS") == "1"
    return tqdm(loader, desc=desc, file=sys.stdout, ncols=80, mininterval=10.0, leave=False, disable=not progress_enabled)

def load_backbone(args):
    print(f"[BACKBONE] Loading {MODEL_NAME}", flush=True)
    return AutoModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        cache_dir=str(args.cache_dir),
        local_files_only=args.local_files_only,
    )

def load_tokenizer(args):
    print(f"[TOKENIZER] Loading {MODEL_NAME}", flush=True)
    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        cache_dir=str(args.cache_dir),
        local_files_only=args.local_files_only,
    )


# %%
# Cell 6 - Dataset and DataLoader
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

def load_vsfc_split(args, split: str):
    split_dir = Path(args.data_dir) / split
    sents = split_dir / "sents.txt"
    labels = split_dir / "sentiments.txt"
    if not sents.exists() or not labels.exists():
        raise FileNotFoundError(f"Missing files for split {split}: {split_dir}")

    cache_dir = Path(args.cache_dir) / "vsfc_segmented"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_segmented.pkl"
    if not args.no_segmentation and cache_path.exists():
        print(f"[DATA] {split}: cache hit {cache_path}", flush=True)
        return pd.read_pickle(cache_path)

    texts = sents.read_text(encoding="utf-8").splitlines()
    ys = [int(x.strip()) for x in labels.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame({"text": texts, "label": ys})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    if not args.no_segmentation:
        if word_tokenize is None:
            print("[DATA] underthesea not available; using raw text (no segmentation).", flush=True)
        else:
            print(f"[DATA] {split}: segmenting {len(df)} samples", flush=True)
            df["text"] = df["text"].apply(segment_vi)
            df.to_pickle(cache_path)
    return df

def build_loaders(args):
    tokenizer = load_tokenizer(args)
    print("[DATA] Loading UIT-VSFC splits", flush=True)
    train_df = load_vsfc_split(args, "train")
    dev_df = load_vsfc_split(args, "dev")
    test_df = load_vsfc_split(args, "test")
    print_split_summary("train", train_df)
    print_split_summary("dev", dev_df)
    print_split_summary("test", test_df)

    low_mps = False
    mps_raw = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    if mps_raw is not None:
        try:
            low_mps = int(mps_raw) < 20
        except ValueError:
            print(f"[WARN] Invalid CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={mps_raw}", flush=True)
    if low_mps and args.num_workers > 1:
        print(f"[WARN] Low MPS share; reducing num_workers from {args.num_workers} to 1", flush=True)
        args.num_workers = 1
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available() and not low_mps,
        "persistent_workers": args.num_workers > 0 and not low_mps,
    }
    train_loader = DataLoader(VSFCDataset(train_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(VSFCDataset(dev_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(VSFCDataset(test_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    return train_loader, dev_loader, test_loader, tokenizer, (train_df, dev_df, test_df)

train_loader, dev_loader, test_loader, tokenizer, dataframes = build_loaders(args)


# %%
# Cell 7 - Model classes
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


# %%
# Cell 8 - Loss functions and optional Stage 1 helpers
# CE = supervised cross entropy.
# KD = KL distillation between student and fine-tuned teacher logits.
# Feature/embedding losses are used by Variant B/C and Tier3 as in the original script.
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

def load_local_wiki_dataset(args):
    if load_from_disk is None:
        raise RuntimeError("Missing dependency: datasets. Install with: pip install datasets")
    wiki_dir = Path(args.wiki_local_dir).expanduser().resolve() if args.wiki_local_dir else Path(WIKI_LOCAL_DIR).expanduser().resolve()
    if not wiki_dir.exists():
        raise FileNotFoundError(f"Local Wiki dataset not found: {wiki_dir}")
    print(f"[WIKI] Loading local dataset from {wiki_dir}", flush=True)
    ds = load_from_disk(str(wiki_dir))
    if hasattr(ds, "keys") and "text" not in getattr(ds, "column_names", []):
        split_name = "train" if "train" in ds else next(iter(ds.keys()))
        print(f"[WIKI] using split={split_name}", flush=True)
        ds = ds[split_name]
    n_total = len(ds)
    print(f"[WIKI] total_samples={n_total}", flush=True)
    if args.wiki_max_samples and args.wiki_max_samples > 0:
        n = min(args.wiki_max_samples, n_total)
        ds = ds.select(range(n))
        print(f"[WIKI] using_samples={n}", flush=True)
    else:
        print(f"[WIKI] using_samples={n_total}", flush=True)
    return ds

def load_wiki_loader(args, tokenizer):
    ds = load_local_wiki_dataset(args)
    low_mps = False
    mps_raw = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    if mps_raw is not None:
        try:
            low_mps = int(mps_raw) < 20
        except ValueError:
            pass
    return DataLoader(
        WikiDataset(ds, tokenizer, args.max_len, preprocess=True),
        batch_size=args.wiki_batch_size or args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not low_mps,
        persistent_workers=args.num_workers > 0 and not low_mps,
    )


# %%
# Cell 9 - Build teacher/student
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_grad_scaler(args, device):
    enabled = device.type == "cuda" and args.amp_dtype == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)

def build_model(args, device: torch.device):
    teacher_ckpt = resolve_teacher_checkpoint(args)
    print(f"[MODEL] Teacher checkpoint: {teacher_ckpt}", flush=True)
    print(f"[MODEL] exists={teacher_ckpt.exists()}", flush=True)

    teacher = PhoBERTTeacher(args, teacher_ckpt)
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    if EXPERIMENT_KIND in {"tier1", "tier3"}:
        student = EquilibriumPhoBERTStudent(args, teacher_ckpt)
    else:
        student = SpikeBERTStudent(args, teacher_ckpt)
    student = student.to(device)

    print(f"[MODEL] Teacher params: {count_parameters(teacher):,}", flush=True)
    print(f"[MODEL] Student params: {count_parameters(student):,}", flush=True)
    print(f"[MODEL] Student spiking_layer_indices={sorted(getattr(student, 'spiking_layer_indices', []))}", flush=True)
    return TrainingBundle(student=student, teacher=teacher, args=args, tokenizer=tokenizer, scaler=make_grad_scaler(args, device))

device = get_device()
bundle = build_model(args, device)


# %%
# Cell 10 - Training loop
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

def run_one_seed(args, train_loader, dev_loader, test_loader, tokenizer, device, seed: int, submode: str, evaluate_test: bool = True):
    set_seed(seed)
    args._active_seed = seed
    args._active_submode = submode
    args._active_seed_run = f"seed{seed}-{submode}"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = build_model(args, device)
    model.tokenizer = tokenizer
    model.submode = submode

    # Stage 1 is optional and disabled in debug config.
    stage1_history = []
    if should_run_stage1(args, submode):
        stage1_history = run_stage1(model, tokenizer, device)

    optimizer = torch.optim.AdamW(model.student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, effective_len(train_loader, args.dry_run_batches) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_dev_f1 = -1.0
    best_path = args.output_dir / f"{RUN_NAME}_{submode}_seed{seed}_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, args.epochs)
        dev_metrics, _, _ = evaluate(model, dev_loader, device, "dev")
        history.append({"epoch": epoch, "train": train_stats, "dev": dev_metrics})
        if dev_metrics["f1_weighted"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1_weighted"]
            safe_torch_save(model.student.state_dict(), best_path, args.min_free_disk_gb)
            print(f"[CHECKPOINT] New best dev F1={best_dev_f1:.4f}, saved to {best_path}", flush=True)

    if best_path.exists():
        model.student.load_state_dict(extract_model_state(safe_torch_load(best_path, map_location=device)))

    result = {
        "seed": seed,
        "submode": submode,
        "best_dev_f1_weighted": best_dev_f1,
        "history": history,
        "stage1_history": stage1_history,
        "checkpoint": str(best_path),
        "model": model,
        "best_path": best_path,
    }

    if evaluate_test:
        test_metrics, y_true, y_pred = evaluate(model, test_loader, device, "test")
        result.update({"test_metrics": test_metrics, "y_true": y_true, "y_pred": y_pred})
    return result


# %%
# Cell 11 - Run full training
# Full training defaults: 3 seeds, 5 epochs, full sequence length, no dry-run batch cap.
# For a quick smoke test, run Cell 12 instead of this cell.
args = refresh_args(
    seeds=list(SEEDS),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    max_len=MAX_LEN,
    dry_run_batches=DRY_RUN_BATCHES,
    spiking_layers=SPIKING_LAYERS,
    skip_stage1=SKIP_STAGE1,
    wiki_max_samples=WIKI_MAX_SAMPLES,
)
train_loader, dev_loader, test_loader, tokenizer, dataframes = build_loaders(args)

FULL_RESULTS = []
if EXPERIMENT_KIND == "tier3":
    submodes = ["3a", "3b"] if args.hybrid_variant == "both" else [args.hybrid_variant]
elif EXPERIMENT_KIND == "tier1":
    submodes = ["tier1"]
else:
    submodes = [TIER2_VARIANT]

for submode in submodes:
    for seed in args.seeds:
        FULL_RESULTS.append(run_one_seed(args, train_loader, dev_loader, test_loader, tokenizer, device, seed, submode, evaluate_test=True))

print("Full run done:", len(FULL_RESULTS))
for item in FULL_RESULTS:
    print({
        "seed": item["seed"],
        "submode": item["submode"],
        "best_dev_f1_weighted": item["best_dev_f1_weighted"],
        "test_metrics": item.get("test_metrics"),
        "checkpoint": str(item["best_path"]),
    })


# %%
# Cell 12 - Optional smoke test
# This cell is intentionally disabled by default. Flip RUN_SMOKE_TEST=True only for pipeline checks.
# It keeps the full epoch count but caps each epoch to two batches.
RUN_SMOKE_TEST = False

if RUN_SMOKE_TEST:
    smoke_args = refresh_args(
        seeds=[42],
        epochs=EPOCHS,
        batch_size=4,
        max_len=128,
        dry_run_batches=2,
        spiking_layers=3,
        skip_stage1=True,
        wiki_max_samples=1000,
    )
    smoke_train_loader, smoke_dev_loader, smoke_test_loader, smoke_tokenizer, smoke_dataframes = build_loaders(smoke_args)
    smoke_submode = "tier1" if EXPERIMENT_KIND == "tier1" else (smoke_args.hybrid_variant if EXPERIMENT_KIND == "tier3" else TIER2_VARIANT)
    SMOKE_TEST_RESULTS = [run_one_seed(smoke_args, smoke_train_loader, smoke_dev_loader, smoke_test_loader, smoke_tokenizer, device, 42, smoke_submode, evaluate_test=True)]
    print("Smoke test done:", len(SMOKE_TEST_RESULTS))
else:
    print("RUN_SMOKE_TEST=False, skipped smoke test.")


# %%
# Cell 13 - Evaluation and report
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
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
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
            f"Actual GPU energy on {device_name}. Higher than teacher is expected "
            "for T-step SNN simulation on GPU; neuromorphic theory is reported separately."
        ),
        "device_name": device_name,
        "avg_power_watts": avg_power_w,
        "total_energy_joules": total_energy_j,
        "energy_per_sample_joules": total_energy_j / total_samples,
        "elapsed_seconds": elapsed,
        "total_samples": total_samples,
        "throughput_samples_per_sec": total_samples / elapsed if elapsed > 0 else None,
    }

def save_run_outputs(results, output_dir: Path, run_label: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_reports = []

    for idx, item in enumerate(results):
        model = item["model"]
        seed = item["seed"]
        submode = item["submode"]

        test_metrics = item.get("test_metrics")
        y_true = item.get("y_true")
        y_pred = item.get("y_pred")
        if test_metrics is None or y_true is None or y_pred is None:
            test_metrics, y_true, y_pred = evaluate(model, test_loader, device, "test")

        theoretical_energy = estimate_energy(model.student, model.args)
        gpu_energy = measure_gpu_energy(model, test_loader, device, model.args)

        pred_df = pd.DataFrame({"label": y_true, "prediction": y_pred})
        pred_path = output_dir / f"predictions_{run_label}_{submode}_seed{seed}.csv"
        pred_df.to_csv(pred_path, index=False)

        best_model_copy = output_dir / f"best_model_{run_label}_{submode}_seed{seed}.pt"
        safe_torch_save(model.student.state_dict(), best_model_copy, model.args.min_free_disk_gb)

        seed_report = {
            "seed": seed,
            "submode": submode,
            "best_dev_f1_weighted": item["best_dev_f1_weighted"],
            "test_metrics": test_metrics,
            "energy": {
                "theoretical_neuromorphic": theoretical_energy,
                "actual_gpu": gpu_energy,
                "actual_gpu_a100": gpu_energy,  # legacy key for old analysis scripts
            },
            "history": item["history"],
            "stage1_history": item["stage1_history"],
            "checkpoint": str(item["best_path"]),
            "predictions_csv": str(pred_path),
            "best_model_pt": str(best_model_copy),
        }
        seed_reports.append(seed_report)
        save_json(seed_report, output_dir / f"{RUN_NAME}_{submode}_seed{seed}_report.json")

    final_report = {
        "config": json_safe(vars(args)),
        "seed_reports": seed_reports,
        "summary": summarize(seed_reports),
    }
    save_json(final_report, output_dir / "report.json")
    print(json.dumps(final_report["summary"], indent=2, ensure_ascii=False))
    print("Saved report:", output_dir / "report.json")
    return final_report

RESULTS_FOR_REPORT = globals().get("FULL_RESULTS", None) or globals().get("SMOKE_TEST_RESULTS", [])
REPORT = save_run_outputs(RESULTS_FOR_REPORT, OUTPUT_DIR, "full" if "FULL_RESULTS" in globals() else "smoke")


# %%
# Cell 14 - Download outputs
import zipfile

zip_path = WORKING_DIR / "outputs.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for file in OUTPUT_DIR.rglob("*"):
        if file.is_file():
            zf.write(file, arcname=file.relative_to(OUTPUT_DIR.parent))

print("Created:", zip_path)
print("Size MB:", zip_path.stat().st_size / 1024**2)
print("\nOutput files:")
for file in sorted(OUTPUT_DIR.rglob("*")):
    if file.is_file():
        print(" -", file, f"({file.stat().st_size / 1024**2:.2f} MB)")


# %% [markdown]
# # Checklist
# 
# - Kaggle GPU is enabled.
# - UIT-VSFC dataset is uploaded.
# - Fine-tuned teacher checkpoint is uploaded.
# - `DATA_DIR` points to UIT-VSFC.
# - `TEACHER_CKPT` points to the fine-tuned `best_model.pth`, not raw PhoBERT.
# - Full training cell completed, or optional smoke test passed.
# - `report.json`, `predictions.csv`, and `best_model*.pt` are saved in `/kaggle/working/outputs`.
# 


