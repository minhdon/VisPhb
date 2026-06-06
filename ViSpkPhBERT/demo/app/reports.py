"""Report loading and metric normalization for ViSPhB demo."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

try:  # Streamlit is present at runtime; this fallback keeps static checks light.
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


METRIC_KEYS: Tuple[str, ...] = (
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "f1_negative",
    "f1_neutral",
    "f1_positive",
    "mean_firing_rate",
    "theoretical_energy_saving",
    "peak_memory_mb",
    "gpu_energy_j",
    "gpu_energy_per_sample_j",
    "latency_ms",
    "throughput_samples_per_sec",
)

METRIC_LABELS: Dict[str, str] = {
    "accuracy": "Accuracy",
    "f1_macro": "F1-macro",
    "f1_weighted": "F1-weighted",
    "f1_negative": "F1-negative",
    "f1_neutral": "F1-neutral",
    "f1_positive": "F1-positive",
    "mean_firing_rate": "Mean firing rate",
    "theoretical_energy_saving": "Theoretical energy saving",
    "peak_memory_mb": "Peak memory (MB)",
    "gpu_energy_j": "GPU energy (J)",
    "gpu_energy_per_sample_j": "GPU energy / sample (J)",
    "latency_ms": "Latency / sample (ms)",
    "throughput_samples_per_sec": "Throughput (samples/s)",
}


def _cache_data(func):
    if st is None:
        return func
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is None:
            return func
    except Exception:
        pass
    return st.cache_data(show_spinner=False)(func)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_float(value: Any) -> Optional[float]:
    if _is_number(value):
        return float(value)
    return None


def _get_path(data: Mapping[str, Any], path: Iterable[Any]) -> Any:
    current: Any = data
    for key in path:
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return None
    return current


def _first_number(data: Mapping[str, Any], paths: Iterable[Tuple[Any, ...]]) -> Optional[float]:
    for path in paths:
        value = _as_float(_get_path(data, path))
        if value is not None:
            return value
    return None


def _merge_missing(target: Dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key not in target or target[key] is None:
            target[key] = value


def _classification_f1(metrics: Mapping[str, Any], label: str) -> Optional[float]:
    direct = _as_float(metrics.get(f"f1_{label}"))
    if direct is not None:
        return direct

    report = metrics.get("classification_report")
    if isinstance(report, Mapping):
        label_report = report.get(label)
        if isinstance(label_report, Mapping):
            return _as_float(label_report.get("f1-score"))
    per_class = metrics.get("f1_per_class")
    if isinstance(per_class, Mapping):
        return _as_float(per_class.get(label))
    if isinstance(per_class, list):
        label_to_index = {"negative": 0, "neutral": 1, "positive": 2}
        index = label_to_index.get(label)
        if index is not None and index < len(per_class):
            return _as_float(per_class[index])
    return None


def _metrics_from_test_metrics(test_metrics: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "accuracy": _as_float(test_metrics.get("accuracy")),
        "f1_macro": _as_float(test_metrics.get("f1_macro")),
        "f1_weighted": _as_float(test_metrics.get("f1_weighted")),
        "f1_negative": _classification_f1(test_metrics, "negative"),
        "f1_neutral": _classification_f1(test_metrics, "neutral"),
        "f1_positive": _classification_f1(test_metrics, "positive"),
    }


def _seed_metrics_from_best_config(data: Mapping[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    summary = data.get("summary")
    if not isinstance(summary, Mapping):
        return {}
    runs = summary.get("seed_runs")
    if not isinstance(runs, list):
        return {}
    chosen = None
    for run in runs:
        if isinstance(run, Mapping) and seed is not None and run.get("seed") == seed:
            chosen = run
            break
    if chosen is None and runs:
        chosen = runs[0]
    if isinstance(chosen, Mapping) and isinstance(chosen.get("test_metrics"), Mapping):
        return _metrics_from_test_metrics(chosen["test_metrics"])
    return {}


def _seed_metrics_from_all_results(data: Mapping[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    all_results = data.get("all_results")
    if not isinstance(all_results, list):
        return {}
    for result in all_results:
        if not isinstance(result, Mapping):
            continue
        runs = result.get("seed_runs")
        if not isinstance(runs, list):
            continue
        for run in runs:
            if isinstance(run, Mapping) and seed is not None and run.get("seed") == seed:
                metrics = run.get("test_metrics")
                if isinstance(metrics, Mapping):
                    return _metrics_from_test_metrics(metrics)
    return {}


def _extract_metric_block(data: Mapping[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    if isinstance(data.get("test_metrics"), Mapping):
        _merge_missing(output, _metrics_from_test_metrics(data["test_metrics"]))

    theoretical = data.get("theoretical_energy")
    if isinstance(theoretical, Mapping) and isinstance(theoretical.get("test_metrics"), Mapping):
        _merge_missing(output, _metrics_from_test_metrics(theoretical["test_metrics"]))

    _merge_missing(output, _seed_metrics_from_best_config(data, seed))
    _merge_missing(output, _seed_metrics_from_all_results(data, seed))
    return output


def _extract_efficiency_block(data: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    theoretical = data.get("theoretical_energy") if isinstance(data.get("theoretical_energy"), Mapping) else {}
    gpu_energy = data.get("actual_gpu_energy_a100") if isinstance(data.get("actual_gpu_energy_a100"), Mapping) else {}
    training_memory = data.get("training_memory") if isinstance(data.get("training_memory"), Mapping) else {}

    return {
        "mean_firing_rate": _first_number(
            data,
            [
                ("mean_firing_rate",),
                ("theoretical_energy", "student_mean_firing_rate"),
                ("test_metrics", "mean_firing_rate"),
            ],
        ),
        "theoretical_energy_saving": _first_number(
            data,
            [
                ("theoretical_energy", "teacher_student_theoretical_saving_ratio"),
                ("teacher_student_theoretical_saving_ratio",),
                ("theoretical_saving",),
            ],
        ),
        "peak_memory_mb": _first_number(
            data,
            [
                ("training_memory", "peak_memory_allocated_mb"),
                ("peak_gpu_memory_mb",),
                ("peak_memory_mb",),
            ],
        ),
        "gpu_energy_j": _first_number(
            data,
            [
                ("total_energy_j",),
                ("actual_gpu_energy_a100", "student", "energy_joules"),
                ("gpu_energy", "total_energy_j"),
            ],
        ),
        "gpu_energy_per_sample_j": _first_number(
            data,
            [
                ("energy_per_sample_j",),
                ("actual_gpu_energy_a100", "student", "energy_per_sample_joules"),
                ("gpu_energy", "energy_per_sample_j"),
            ],
        ),
        "latency_ms": _first_number(
            data,
            [
                ("latency_per_sample_ms",),
                ("gpu_energy", "latency_per_sample_ms"),
                ("actual_gpu_energy_a100", "student", "latency_per_sample_ms"),
            ],
        ),
        "throughput_samples_per_sec": _first_number(
            data,
            [
                ("throughput_samples_per_sec",),
                ("gpu_energy", "throughput_samples_per_sec"),
            ],
        ),
    }


@_cache_data
def load_json_reports(path_strings: Tuple[str, ...]) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for path_string in path_strings:
        path = Path(path_string)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                data["_source_path"] = str(path)
                reports.append(data)
        except Exception as exc:
            reports.append({"_source_path": str(path), "_load_error": str(exc)})
    return reports


@_cache_data
def normalize_reports(path_strings: Tuple[str, ...], seed: Optional[int]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {key: None for key in METRIC_KEYS}
    normalized["source_paths"] = list(path_strings)
    normalized["load_errors"] = []
    normalized["raw_count"] = 0

    for data in load_json_reports(path_strings):
        normalized["raw_count"] += 1
        if data.get("_load_error"):
            normalized["load_errors"].append(
                {"path": data.get("_source_path"), "error": data.get("_load_error")}
            )
            continue
        _merge_missing(normalized, _extract_metric_block(data, seed))
        _merge_missing(normalized, _extract_efficiency_block(data))

    return normalized


def format_metric(value: Any, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if _is_number(value):
        return f"{float(value):.{digits}f}{suffix}"
    return "N/A"


def metrics_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for key in METRIC_KEYS:
        rows.append({"Chỉ số": METRIC_LABELS[key], "Giá trị": format_metric(metrics.get(key))})
    return pd.DataFrame(rows)


def comparison_metric_row(metrics: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "F1-macro": format_metric(metrics.get("f1_macro")),
        "F1-weighted": format_metric(metrics.get("f1_weighted")),
        "F1-neutral": format_metric(metrics.get("f1_neutral")),
        "Firing rate": format_metric(metrics.get("mean_firing_rate")),
        "Energy saving": format_metric(metrics.get("theoretical_energy_saving")),
    }


def raw_reports_for_display(path_strings: Tuple[str, ...]) -> List[Dict[str, Any]]:
    return load_json_reports(path_strings)
