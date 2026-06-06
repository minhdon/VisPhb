"""Reusable Streamlit UI components for the ViSPhB demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
import streamlit as st

from inference import LABELS, LABELS_VI, PredictionResult
from reports import METRIC_LABELS, format_metric, metrics_table


LABEL_COLORS = {
    "negative": "#c94c4c",
    "neutral": "#b08900",
    "positive": "#2f8f5b",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        [data-testid="stSidebar"] {
            background: #f6f7f9;
            border-right: 1px solid #e5e7eb;
        }
        .visphb-title {
            font-size: 2.05rem;
            font-weight: 760;
            color: #172033;
            letter-spacing: 0;
            margin-bottom: 0.25rem;
        }
        .visphb-subtitle {
            color: #596275;
            font-size: 0.98rem;
            margin-bottom: 1.2rem;
        }
        .soft-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .prediction-card {
            border: 1px solid #d9dee8;
            border-left: 6px solid var(--label-color);
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        .prediction-label {
            font-size: 1.85rem;
            font-weight: 780;
            color: #172033;
            margin-bottom: 0.2rem;
        }
        .prediction-meta {
            color: #596275;
            font-size: 0.95rem;
        }
        .small-muted {
            color: #6b7280;
            font-size: 0.88rem;
        }
        .status-ok {
            color: #17663a;
            font-weight: 650;
        }
        .status-warn {
            color: #9a5b00;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header() -> None:
    st.markdown('<div class="visphb-title">ViSPhB Sentiment Demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="visphb-subtitle">PhoBERT teacher, SpikeBERT-style và SpikingBERT-style students cho phân loại cảm xúc tiếng Việt.</div>',
        unsafe_allow_html=True,
    )


def render_prediction_card(result: PredictionResult) -> None:
    color = LABEL_COLORS.get(result.label, "#64748b")
    st.markdown(
        f"""
        <div class="prediction-card" style="--label-color:{color}">
            <div class="prediction-label">{result.label} · {result.label_vi}</div>
            <div class="prediction-meta">
                Confidence: <b>{result.confidence:.4f}</b> ·
                Latency: <b>{result.latency_ms:.2f} ms</b> ·
                Device: <b>{result.device}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_dataframe(probabilities: Mapping[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label": label,
                "label_vi": LABELS_VI[label],
                "probability": float(probabilities.get(label, 0.0)),
            }
            for label in LABELS
        ]
    )


def render_probabilities(probabilities: Mapping[str, float]) -> None:
    df = probability_dataframe(probabilities)
    st.dataframe(
        df.assign(probability=lambda x: x["probability"].map(lambda v: f"{v:.4f}")),
        hide_index=True,
        use_container_width=True,
    )
    chart_df = df.set_index("label_vi")[["probability"]]
    st.bar_chart(chart_df, use_container_width=True)


def render_metric_overview(metrics: Mapping[str, Any]) -> None:
    cols = st.columns(4)
    highlights = ["accuracy", "f1_macro", "f1_weighted", "f1_neutral"]
    for col, key in zip(cols, highlights):
        col.metric(METRIC_LABELS[key], format_metric(metrics.get(key)))


def render_metrics_table(metrics: Mapping[str, Any]) -> None:
    st.dataframe(metrics_table(metrics), hide_index=True, use_container_width=True)


def render_checkpoint_info(checkpoint_path: str, run_dir: Path, load_info: Mapping[str, Any] | None = None) -> None:
    with st.expander("Thông tin checkpoint", expanded=False):
        st.write({"checkpoint": checkpoint_path, "run_dir": str(run_dir)})
        if load_info:
            missing = load_info.get("missing_keys") or []
            unexpected = load_info.get("unexpected_keys") or []
            st.caption(f"Missing keys: {len(missing)} · Unexpected keys: {len(unexpected)}")
            if missing:
                st.write("Missing key sample:", missing[:20])
            if unexpected:
                st.write("Unexpected key sample:", unexpected[:20])


def render_variant_status(has_checkpoint: bool, has_reports: bool) -> None:
    checkpoint_status = "có checkpoint" if has_checkpoint else "thiếu checkpoint"
    report_status = "có report" if has_reports else "thiếu report"
    cls = "status-ok" if has_checkpoint else "status-warn"
    st.markdown(
        f'<span class="{cls}">{checkpoint_status}</span>'
        f'<span class="small-muted"> · {report_status}</span>',
        unsafe_allow_html=True,
    )

