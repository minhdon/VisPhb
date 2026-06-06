from pathlib import Path
import json
import os
import random
import time
import threading
from contextlib import nullcontext
from itertools import product as iterproduct

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from underthesea import word_tokenize

from hf_runtime import build_hf_load_kwargs, configure_hf_runtime


# =========================================================
# 1) PATHS - portable across local/server
# =========================================================
ROOT = Path(__file__).resolve().parents[2]
HF_LOAD_KWARGS = build_hf_load_kwargs(ROOT)
configure_hf_runtime(ROOT)

TRAIN_DIR = ROOT / "uit" / "uit-vsfc" / "train"
DEV_DIR   = ROOT / "uit" / "uit-vsfc" / "dev"
TEST_DIR  = ROOT / "uit" / "uit-vsfc" / "test"

# FIX #6: Cache thư mục cho segmented data
CACHE_DIR = ROOT / "uit" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ROOT / "uit" / "uit-models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = Path(os.getenv("PHOBERT_OUTPUT_DIR", str(MODEL_DIR / "phobert_vsfc")))
RUN_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH      = RUN_DIR / "best_model.pth"
BEST_CONFIG_PATH     = RUN_DIR / "best_config.json"
TEST_RESULTS_PATH    = RUN_DIR / "test_results.json"
EFFICIENCY_REPORT_PATH = RUN_DIR / "efficiency_report.json"


# =========================================================
# 2) CONFIG
# =========================================================
MODEL_NAME = "vinai/phobert-base-v2"
SEEDS = [42, 52, 62]
LABEL_NAMES = ["negative", "neutral", "positive"]

PARAM_GRID = {
    "lr":           [2e-5, 5e-5],
    "batch_size":   [8, 16, 32],
    "max_len":      [256],
    "dropout":      [0.25],
    "epochs":       [5],
    "weight_decay": [0.01],
    "warmup_ratio": [0.1],
}

EARLY_STOPPING_PATIENCE = 2

NUM_WORKERS = 0 if os.name == "nt" else 4
PIN_MEMORY  = True


# =========================================================
# 3) UTILITIES
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_autocast_context(device: torch.device):
    """
    Backward pass được xử lý bởi GradScaler riêng trong train_one_epoch.
    bfloat16 ổn định hơn float16 (không cần GradScaler), nhưng vẫn giữ
    GradScaler để tương thích nếu sau này chuyển sang float16.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

def save_json(obj, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def print_metrics(label: str, metrics: dict) -> None:
    """FIX #8: In kết quả metrics ra console rõ ràng."""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.4f}")
        else:
            print(f"  {k:<30}: {v}")
    print(f"{'='*50}\n")


# =========================================================
# 4) UNDERTHESEA SEGMENTATION & DATA LOADING
# =========================================================
def segment_vi(text: str) -> str:
    """Segment Vietnamese text using underthesea with underscore joining."""
    try:
        return word_tokenize(text, format="text")
    except Exception:
        return text

def load_split(split_dir: Path, apply_segmentation: bool = True) -> pd.DataFrame:
    """
    FIX #6: Cache kết quả sau khi segment để không phải chạy lại mỗi lần.
    Cache file được lưu tại CACHE_DIR/<split_name>_segmented.parquet
    """
    split_name = split_dir.name
    cache_path = CACHE_DIR / f"{split_name}_segmented.parquet"

    # Kiểm tra cache
    if apply_segmentation and cache_path.exists():
        print(f"[Cache HIT] Đọc dữ liệu đã segment từ: {cache_path}")
        return pd.read_parquet(cache_path)

    sents_path     = split_dir / "sents.txt"
    sentiments_path = split_dir / "sentiments.txt"

    with sents_path.open("r", encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]
    with sentiments_path.open("r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f]

    df = pd.DataFrame({"text": texts, "label": labels})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    if apply_segmentation:
        print(f"[Cache MISS] Tiến hành Tách từ cho {split_name} ({len(df)} mẫu)...")
        tqdm.pandas(desc="Segmenting")
        df["text"] = df["text"].progress_apply(segment_vi)
        df.to_parquet(cache_path, index=False)
        print(f"Đã lưu cache tại: {cache_path}")

    return df


class VSFCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text  = str(self.df.loc[idx, "text"])
        label = int(self.df.loc[idx, "label"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# =========================================================
# 5) EFFICIENCY TRACKING (POWER MONITOR)
# =========================================================
class DummyMonitor:
    """FIX #4: Fallback khi không có GPU, tránh crash khi chạy trên CPU."""
    def start(self): pass
    def stop(self): pass
    def summary(self):
        return {"avg_power_w": None, "max_power_w": None,
                "min_power_w": None, "num_power_samples": 0}
    def device_name(self): return "CPU"
    def close(self): pass


class PowerMonitor:
    def __init__(self, gpu_index: int = 0, interval_sec: float = 0.02):
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
        nvmlInit()
        self.gpu_index     = gpu_index
        self.interval_sec  = interval_sec
        self._running      = False
        self._thread       = None
        self.samples_watts = []
        self.handle        = nvmlDeviceGetHandleByIndex(gpu_index)

    def _worker(self):
        from pynvml import nvmlDeviceGetPowerUsage
        while self._running:
            try:
                mw = nvmlDeviceGetPowerUsage(self.handle)
                self.samples_watts.append(mw / 1000.0)
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self):
        self.samples_watts = []
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        if self._running:
            self._running = False
            if self._thread is not None:
                self._thread.join()

    def summary(self):
        if not self.samples_watts:
            return {"avg_power_w": None, "max_power_w": None,
                    "min_power_w": None, "num_power_samples": 0}
        return {
            "avg_power_w":       sum(self.samples_watts) / len(self.samples_watts),
            "max_power_w":       max(self.samples_watts),
            "min_power_w":       min(self.samples_watts),
            "num_power_samples": len(self.samples_watts),
        }

    def device_name(self):
        from pynvml import nvmlDeviceGetName
        name = nvmlDeviceGetName(self.handle)
        return name.decode("utf-8") if isinstance(name, bytes) else name

    def close(self):
        from pynvml import nvmlShutdown
        try:
            nvmlShutdown()
        except Exception:
            pass


def _make_monitor(device: torch.device):
    """FIX #4 + #5: Tạo monitor phù hợp, dùng current_device() cho GPU index."""
    if device.type != "cuda":
        return DummyMonitor()
    try:
        # FIX #5: dùng torch.cuda.current_device() thay vì device.index
        gpu_index = torch.cuda.current_device()
        return PowerMonitor(gpu_index=gpu_index, interval_sec=0.02)
    except Exception as e:
        print(f"[WARN] Không khởi tạo được PowerMonitor: {e}. Dùng DummyMonitor.")
        return DummyMonitor()


@torch.no_grad()
def measure_efficiency(model: nn.Module, dataloader: DataLoader,
                       device: torch.device, warmup_steps: int = 10):
    model.eval()

    # Warmup
    warmup_count = 0
    for batch in dataloader:
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with get_autocast_context(device):
            _ = model(ids, mask)
        warmup_count += 1
        if warmup_count >= warmup_steps:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    monitor      = _make_monitor(device)
    total_samples = 0
    start         = time.perf_counter()
    monitor.start()

    for batch in tqdm(dataloader, desc="Measuring efficiency"):
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with get_autocast_context(device):
            _ = model(ids, mask)
        total_samples += ids.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()
    monitor.stop()

    elapsed_sec      = end - start
    power_stats      = monitor.summary()
    peak_mem_bytes   = None

    if device.type == "cuda":
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)

    avg_power_w      = power_stats["avg_power_w"]
    total_energy_j   = None if avg_power_w is None else avg_power_w * elapsed_sec
    energy_per_sample_j = None if total_energy_j is None else total_energy_j / total_samples

    results = {
        "gpu_name":                  monitor.device_name(),
        "total_samples":             total_samples,
        "elapsed_sec":               elapsed_sec,
        "latency_per_sample_ms":     (elapsed_sec / total_samples) * 1000.0,
        "throughput_samples_per_sec": total_samples / elapsed_sec,
        "avg_power_w":               avg_power_w,
        "max_power_w":               power_stats["max_power_w"],
        "min_power_w":               power_stats["min_power_w"],
        "num_power_samples":         power_stats["num_power_samples"],
        "total_energy_j":            total_energy_j,
        "energy_per_sample_j":       energy_per_sample_j,
        "peak_gpu_memory_mb":        None if peak_mem_bytes is None else peak_mem_bytes / (1024 ** 2),
    }

    monitor.close()
    return results


# =========================================================
# 6) MODEL - PhoBERT Baseline
# =========================================================
class PhoBERTSentimentClassifier(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, dropout_rate: float = 0.25):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name, **HF_LOAD_KWARGS)
        hidden_size     = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)
        return self.classifier(cls_rep)


# =========================================================
# 7) TRAINING PIPELINE
# =========================================================
def compute_metrics(y_true, y_pred):
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0),
        "f1_macro":    f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "f1_micro":    f1_score(y_true, y_pred, labels=[0, 1, 2], average="micro", zero_division=0),
        "f1_negative": float(per_class_f1[0]),
        "f1_neutral":  float(per_class_f1[1]),
        "f1_positive": float(per_class_f1[2]),
        "f1_per_class": {
            label_name: float(score)
            for label_name, score in zip(LABEL_NAMES, per_class_f1)
        },
    }


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler: GradScaler):
    """FIX #1: Dùng GradScaler để xử lý backward pass an toàn với mixed precision."""
    model.train()
    total_loss = 0.0
    progress   = tqdm(dataloader, desc="Train", leave=False)

    for batch in progress:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with get_autocast_context(device):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = criterion(logits, labels)

        # GradScaler xử lý backward + unscale + clip + step
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        with get_autocast_context(device):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics         = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    return metrics


def build_dataloaders(tokenizer, df_train, df_dev, df_test, batch_size, max_len):
    pin = PIN_MEMORY and torch.cuda.is_available()
    train_loader = DataLoader(
        VSFCDataset(df_train, tokenizer, max_len),
        batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    dev_loader = DataLoader(
        VSFCDataset(df_dev, tokenizer, max_len),
        batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    test_loader = DataLoader(
        VSFCDataset(df_test, tokenizer, max_len),
        batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    return train_loader, dev_loader, test_loader


def run_single_config(config, seed, tokenizer, df_train, df_dev, df_test, device):
    set_seed(seed)
    train_loader, dev_loader, test_loader = build_dataloaders(
        tokenizer, df_train, df_dev, df_test,
        config["batch_size"], config["max_len"],
    )

    model     = PhoBERTSentimentClassifier(model_name=MODEL_NAME, dropout_rate=config["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    total_train_steps = len(train_loader) * config["epochs"]
    warmup_steps      = int(total_train_steps * config["warmup_ratio"])
    scheduler         = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_dev_f1 = -1.0
    best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        dev_metrics = evaluate(model, dev_loader, criterion, device)

        print(f"  Epoch {epoch}/{config['epochs']} | "
              f"train_loss={train_loss:.4f} | "
              f"dev_f1_w={dev_metrics['f1_weighted']:.4f} | "
              f"dev_f1_neutral={dev_metrics['f1_neutral']:.4f} | "
              f"dev_acc={dev_metrics['accuracy']:.4f}")

        if dev_metrics["f1_weighted"] > best_dev_f1:
            best_dev_f1  = dev_metrics["f1_weighted"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping tại epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device)

    return {
        "seed":                 seed,
        "best_dev_f1_weighted": best_dev_f1,
        "test_metrics":         test_metrics,
        "best_state":           best_state,  
    }


# =========================================================
# 8) MAIN EXECUTOR
# =========================================================
def build_config_grid(param_grid: dict) -> list[dict]:
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in iterproduct(*values)]


def main():
    device = get_device()
    print(f"Using device: {device}")

    print("\nLoading and Segmenting Datasets (cached nếu đã có)...")
    df_train = load_split(TRAIN_DIR, apply_segmentation=True)
    df_dev   = load_split(DEV_DIR,   apply_segmentation=True)
    df_test  = load_split(TEST_DIR,  apply_segmentation=True)
    print(f"Train: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        **HF_LOAD_KWARGS,
    )

    configs = build_config_grid(PARAM_GRID)
    print(f"\nTổng số config cần chạy: {len(configs)}")

    all_results          = []
    best_global_dev_mean = -1.0
    best_global_result   = None   # sẽ được gán trước khi dùng (FIX #3 guard ở cuối)

    for i, config in enumerate(configs, start=1):
        print(f"\n[Config {i}/{len(configs)}] {config}")
        seed_runs = []

        for seed in SEEDS:
            print(f"  → Seed {seed}")
            res = run_single_config(config, seed, tokenizer, df_train, df_dev, df_test, device)
            seed_runs.append(res)

        dev_scores = [x["best_dev_f1_weighted"] for x in seed_runs]

        summary = {
            "config":                  config,
            "dev_f1_weighted_mean":    float(np.mean(dev_scores)),
            "dev_f1_weighted_std":     float(np.std(dev_scores)),
            "seed_runs": [
                {
                    "seed":                 x["seed"],
                    "best_dev_f1_weighted": x["best_dev_f1_weighted"],
                    "test_metrics":         x["test_metrics"],
                }
                for x in seed_runs
            ],
        }
        all_results.append(summary)

        print_metrics(
            f"Config {i} Summary | dev_f1_w_mean={summary['dev_f1_weighted_mean']:.4f}",
            {
                f"seed_{x['seed']}_test_f1_w": x["test_metrics"]["f1_weighted"]
                for x in seed_runs
            }
            | {
                f"seed_{x['seed']}_test_f1_neutral": x["test_metrics"]["f1_neutral"]
                for x in seed_runs
            },
        )

        if summary["dev_f1_weighted_mean"] > best_global_dev_mean:
            best_global_dev_mean = summary["dev_f1_weighted_mean"]
            best_seed_result     = max(seed_runs, key=lambda x: x["best_dev_f1_weighted"])
            best_global_result   = {
                "config":     config,
                "summary":    summary,
                "best_seed":  best_seed_result["seed"],
            }
            torch.save(best_seed_result["best_state"], BEST_MODEL_PATH)
            save_json(best_global_result, BEST_CONFIG_PATH)
            print(f"  ✓ Best model updated (seed={best_seed_result['seed']}, dev_f1_w={best_global_dev_mean:.4f})")

    save_json({"all_results": all_results}, TEST_RESULTS_PATH)

    if best_global_result is None:
        raise RuntimeError("Không có config nào được chạy thành công! Kiểm tra lại PARAM_GRID và dữ liệu.")

    print("\n" + "="*60)
    print("BEST CONFIG:")
    print(json.dumps(best_global_result["config"], indent=2))
    print(f"Best dev F1 (weighted) mean: {best_global_dev_mean:.4f}")
    print("="*60)

    print("\n--- MEASURING EFFICIENCY FOR BEST BASELINE MODEL ---")
    best_model = PhoBERTSentimentClassifier(
        model_name=MODEL_NAME,
        dropout_rate=best_global_result["config"]["dropout"],
    ).to(device)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    _, _, test_loader = build_dataloaders(
        tokenizer, df_train, df_dev, df_test,
        best_global_result["config"]["batch_size"],
        best_global_result["config"]["max_len"],
    )
    efficiency_metrics = measure_efficiency(best_model, test_loader, device)
    save_json(efficiency_metrics, EFFICIENCY_REPORT_PATH)

    print_metrics("Efficiency Report", efficiency_metrics)
    print(f"Model lưu tại:            {BEST_MODEL_PATH}")
    print(f"Efficiency report lưu tại: {EFFICIENCY_REPORT_PATH}")


if __name__ == "__main__":
    main()
