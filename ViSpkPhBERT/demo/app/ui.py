"""Reusable Streamlit UI components for the ViSPhB demo.

Dark-only UI.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from inference import LABELS, LABELS_VI, PredictionResult
from reports import METRIC_LABELS, format_metric, metrics_table


LABEL_COLORS = {
    "negative": "#fb7185",
    "neutral": "#facc15",
    "positive": "#4ade80",
}

LABEL_EMOJIS = {
    "negative": "⚠️",
    "neutral": "➖",
    "positive": "✅",
}

DARK_THEME = {
    "bg": "#070b16",
    "bg_soft": "#0b1020",
    "surface": "#101827",
    "surface_2": "#162033",
    "surface_3": "#202c43",
    "surface_4": "#2a3854",
    "text": "#f8fafc",
    "text_2": "#e5e7eb",
    "muted": "#9ca3af",
    "border": "rgba(255, 255, 255, 0.10)",
    "border_strong": "rgba(255, 255, 255, 0.18)",
    "primary": "#60a5fa",
    "primary_hover": "#93c5fd",
    "primary_soft": "rgba(96, 165, 250, 0.16)",
    "good": "#4ade80",
    "good_soft": "rgba(74, 222, 128, 0.14)",
    "warn": "#facc15",
    "warn_soft": "rgba(250, 204, 21, 0.14)",
    "danger": "#fb7185",
    "danger_soft": "rgba(251, 113, 133, 0.14)",
    "shadow": "0 14px 38px rgba(0, 0, 0, 0.42), 0 1px 2px rgba(0, 0, 0, 0.36)",
    "shadow_md": "0 24px 70px rgba(0, 0, 0, 0.55), 0 2px 8px rgba(0, 0, 0, 0.38)",
}


def _safe(value: Any) -> str:
    return html.escape(str(value))


def inject_css() -> None:
    """Inject dark-only modern CSS."""

    css_vars = "\n".join(
        f"            --vis-{key.replace('_', '-')}: {value};"
        for key, value in DARK_THEME.items()
    )

    css = f"""<style>
:root {{
{css_vars}
    --vis-radius-xs: 10px;
    --vis-radius-sm: 14px;
    --vis-radius: 20px;
    --vis-radius-lg: 28px;
}}

html,
body,
.stApp,
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 8% 0%, rgba(96, 165, 250, 0.13), transparent 30%),
        radial-gradient(circle at 95% 10%, rgba(74, 222, 128, 0.08), transparent 28%),
        linear-gradient(180deg, var(--vis-bg), var(--vis-bg-soft)) !important;
    color: var(--vis-text) !important;
}}

html,
body,
button,
input,
textarea,
select {{
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}}

.block-container {{
    padding-top: 1.55rem !important;
    padding-bottom: 3.25rem !important;
    max-width: 1220px !important;
}}

[data-testid="stSidebar"] {{
    background:
        linear-gradient(180deg, rgba(16, 24, 39, 0.98), rgba(11, 16, 32, 0.98)) !important;
    border-right: 1px solid var(--vis-border) !important;
    box-shadow: 12px 0 40px rgba(0, 0, 0, 0.35) !important;
}}

[data-testid="stSidebar"] h3 {{
    color: var(--vis-muted) !important;
    font-size: 0.73rem !important;
    font-weight: 850 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 1.15rem 0 0.45rem !important;
}}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: var(--vis-muted) !important;
}}

.sidebar-brand {{
    padding: 1.05rem 0.2rem 0.9rem;
    border-bottom: 1px solid var(--vis-border);
    margin-bottom: 0.95rem;
}}

.sidebar-brand-name {{
    font-size: 1.2rem;
    font-weight: 900;
    letter-spacing: -0.045em;
    color: var(--vis-text);
    display: flex;
    align-items: center;
    gap: 0.45rem;
}}

.sidebar-brand-sub {{
    color: var(--vis-muted);
    font-size: 0.82rem;
    line-height: 1.5;
    margin-top: 0.28rem;
}}

div[data-testid="stTabs"] [role="tablist"] {{
    width: fit-content !important;
    max-width: 100% !important;
    background: rgba(32, 44, 67, 0.84) !important;
    border: 1px solid var(--vis-border) !important;
    border-radius: 999px !important;
    padding: 0.25rem !important;
    gap: 0.15rem !important;
    box-shadow: var(--vis-shadow) !important;
}}

div[data-testid="stTabs"] button[role="tab"] {{
    border-radius: 999px !important;
    color: var(--vis-muted) !important;
    font-weight: 780 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.05rem !important;
    transition: all 0.16s ease !important;
}}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background: var(--vis-surface) !important;
    color: var(--vis-text) !important;
    box-shadow: var(--vis-shadow) !important;
}}

div[data-testid="stTabs"] [data-testid="stTabContent"] {{
    padding-top: 1.45rem !important;
}}

textarea,
input[type="text"] {{
    color: var(--vis-text) !important;
    background: rgba(7, 11, 22, 0.62) !important;
    border: 1px solid var(--vis-border-strong) !important;
    border-radius: var(--vis-radius-sm) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03) !important;
}}

textarea:focus,
input[type="text"]:focus {{
    border-color: var(--vis-primary) !important;
    box-shadow: 0 0 0 4px var(--vis-primary-soft) !important;
}}

.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {{
    color: var(--vis-text) !important;
    background: rgba(7, 11, 22, 0.62) !important;
    border: 1px solid var(--vis-border-strong) !important;
    border-radius: var(--vis-radius-sm) !important;
}}

div[data-testid="stButton"] > button {{
    border-radius: var(--vis-radius-sm) !important;
    border: 1px solid var(--vis-border-strong) !important;
    background: rgba(22, 32, 51, 0.92) !important;
    color: var(--vis-text) !important;
    min-height: 2.55rem !important;
    font-weight: 850 !important;
    box-shadow: var(--vis-shadow) !important;
    transition: all 0.16s ease !important;
}}

div[data-testid="stButton"] > button:hover {{
    border-color: var(--vis-primary) !important;
    transform: translateY(-1px) !important;
}}

div[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, #2563eb, var(--vis-primary)) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 14px 28px rgba(37, 99, 235, 0.34) !important;
}}

.vis-metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.82rem;
    margin: 1rem 0 1.1rem;
}}

.vis-metric-card {{
    background: linear-gradient(180deg, rgba(16, 24, 39, 0.96), rgba(22, 32, 51, 0.96));
    border: 1px solid var(--vis-border);
    border-radius: var(--vis-radius);
    box-shadow: var(--vis-shadow);
    padding: 1rem 1.12rem;
}}

.vis-metric-label {{
    color: var(--vis-muted);
    font-weight: 850;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.36rem;
}}

.vis-metric-value {{
    color: var(--vis-text);
    font-weight: 920;
    font-size: 1.48rem;
    letter-spacing: -0.045em;
    line-height: 1.1;
}}

div[data-testid="stMetric"] {{
    background: var(--vis-surface) !important;
    border: 1px solid var(--vis-border) !important;
    border-radius: var(--vis-radius) !important;
    box-shadow: var(--vis-shadow) !important;
    padding: 1rem 1.12rem !important;
}}

div[data-testid="stMetricLabel"] p {{
    color: var(--vis-muted) !important;
    font-weight: 850 !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}}

div[data-testid="stMetricValue"] {{
    color: var(--vis-text) !important;
    font-weight: 900 !important;
    letter-spacing: -0.04em !important;
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid var(--vis-border) !important;
    border-radius: var(--vis-radius) !important;
    overflow: hidden !important;
    box-shadow: var(--vis-shadow) !important;
}}

hr {{
    border-color: var(--vis-border) !important;
}}

div[data-testid="stAlert"] {{
    border-radius: var(--vis-radius-sm) !important;
}}

[data-testid="stExpander"] > details {{
    border: 1px solid var(--vis-border) !important;
    border-radius: var(--vis-radius-sm) !important;
    background: rgba(16, 24, 39, 0.92) !important;
    box-shadow: var(--vis-shadow) !important;
    overflow: hidden !important;
}}

[data-testid="stExpander"] > details > summary {{
    color: var(--vis-text) !important;
    background: rgba(16, 24, 39, 0.92) !important;
    padding: 0.72rem 1rem !important;
    list-style: none !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.55rem !important;
}}

[data-testid="stExpander"] > details > summary::-webkit-details-marker {{
    display: none !important;
}}

[data-testid="stExpander"] > details > summary::before {{
    content: "›";
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.2rem;
    height: 1.2rem;
    color: var(--vis-muted);
    font-size: 1.25rem;
    font-weight: 900;
    transition: transform 0.16s ease;
}}

[data-testid="stExpander"] > details[open] > summary::before {{
    transform: rotate(90deg);
}}

[data-testid="stExpander"] > details > summary p {{
    color: var(--vis-text) !important;
    margin: 0 !important;
    font-weight: 780 !important;
}}

[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span {{
    font-size: 0 !important;
    line-height: 0 !important;
}}

[data-testid="stSidebarCollapseButton"] span::before,
[data-testid="stSidebarCollapsedControl"] span::before {{
    content: "‹";
    font-size: 1.35rem;
    line-height: 1;
    color: var(--vis-text);
    font-weight: 900;
}}

.stColumn:empty {{
    display: none !important;
}}

.vis-hero {{
    position: relative;
    overflow: hidden;
    border: 1px solid var(--vis-border);
    border-radius: var(--vis-radius-lg);
    background:
        radial-gradient(circle at 86% 34%, var(--vis-primary-soft), transparent 34%),
        linear-gradient(135deg, rgba(16, 24, 39, 0.98), rgba(22, 32, 51, 0.94));
    box-shadow: var(--vis-shadow-md);
    padding: 2.15rem 2.35rem 1.95rem;
    margin-bottom: 1.7rem;
}}

.vis-kicker {{
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    border-radius: 999px;
    background: var(--vis-primary-soft);
    color: var(--vis-primary);
    font-size: 0.77rem;
    font-weight: 880;
    letter-spacing: 0.02em;
    padding: 0.32rem 0.72rem;
    margin-bottom: 0.85rem;
}}

.vis-title {{
    color: var(--vis-text);
    font-size: clamp(1.85rem, 3.6vw, 2.85rem);
    font-weight: 950;
    line-height: 1.02;
    letter-spacing: -0.065em;
    margin: 0 0 0.62rem;
}}

.vis-subtitle {{
    color: var(--vis-muted);
    max-width: 820px;
    font-size: 0.98rem;
    line-height: 1.68;
    margin: 0;
}}

.vis-hero-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 1.45rem;
}}

.vis-hero-stat {{
    border: 1px solid var(--vis-border);
    border-radius: var(--vis-radius-sm);
    background: rgba(7, 11, 22, 0.46);
    padding: 0.82rem 0.95rem;
    box-shadow: var(--vis-shadow);
}}

.vis-hero-stat b {{
    display: block;
    color: var(--vis-text);
    font-size: 0.96rem;
    font-weight: 880;
    line-height: 1.25;
}}

.vis-hero-stat span {{
    color: var(--vis-muted);
    font-size: 0.76rem;
}}

.vis-section {{
    padding-bottom: 0.9rem;
    margin: 0.1rem 0 1rem;
    border-bottom: 1px solid var(--vis-border);
}}

.vis-section-title {{
    color: var(--vis-text);
    font-size: 1.18rem;
    font-weight: 920;
    letter-spacing: -0.035em;
    line-height: 1.2;
    margin: 0;
}}

.vis-section-desc {{
    color: var(--vis-muted);
    font-size: 0.87rem;
    line-height: 1.5;
    margin-top: 0.22rem;
}}

.vis-card {{
    border: 1px solid var(--vis-border);
    border-radius: var(--vis-radius);
    background: var(--vis-surface);
    box-shadow: var(--vis-shadow);
    padding: 1.15rem 1.25rem;
}}

.vis-card-strong {{
    position: relative;
    overflow: hidden;
    border-radius: var(--vis-radius-lg);
    border: 1px solid var(--vis-border);
    background:
        radial-gradient(circle at 100% 0%, var(--vis-primary-soft), transparent 34%),
        linear-gradient(180deg, rgba(16, 24, 39, 0.98), rgba(22, 32, 51, 0.96));
    box-shadow: var(--vis-shadow-md);
}}

div[data-testid="stVerticalBlock"]:has(.vis-card-anchor[data-card="single-input"]),
div[data-testid="stVerticalBlock"]:has(.vis-card-anchor[data-card="comparison-input"]) {{
    position: relative !important;
    overflow: hidden !important;
    border-radius: var(--vis-radius-lg) !important;
    border: 1px solid var(--vis-border) !important;
    background:
        radial-gradient(circle at 100% 0%, var(--vis-primary-soft), transparent 34%),
        linear-gradient(180deg, rgba(16, 24, 39, 0.98), rgba(22, 32, 51, 0.96)) !important;
    box-shadow: var(--vis-shadow-md) !important;
    padding: 1.25rem 1.35rem 1.15rem !important;
    margin-bottom: 1rem !important;
}}

.vis-card-anchor {{
    display: none !important;
}}

.vis-input-card-head {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.85rem;
}}

.vis-input-card-title {{
    color: var(--vis-text);
    font-size: 0.98rem;
    font-weight: 900;
    letter-spacing: -0.02em;
}}

.vis-input-card-subtitle {{
    color: var(--vis-muted);
    font-size: 0.82rem;
    line-height: 1.45;
    margin-top: 0.15rem;
}}

.vis-input-card-badge {{
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.28rem 0.65rem;
    background: var(--vis-primary-soft);
    color: var(--vis-primary);
    font-size: 0.72rem;
    font-weight: 900;
    white-space: nowrap;
}}

.vis-model-card {{
    border: 1px solid var(--vis-border);
    border-radius: var(--vis-radius);
    background: linear-gradient(180deg, rgba(16, 24, 39, 0.98), rgba(22, 32, 51, 0.96));
    box-shadow: var(--vis-shadow);
    padding: 1.08rem 1.22rem;
    margin-bottom: 1rem;
}}

.vis-model-name {{
    color: var(--vis-text);
    font-size: 0.98rem;
    font-weight: 900;
    letter-spacing: -0.025em;
    margin-bottom: 0.28rem;
}}

.vis-model-meta {{
    color: var(--vis-muted);
    font-size: 0.8rem;
    line-height: 1.5;
    word-break: break-word;
}}

.vis-chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.42rem;
    margin-top: 0.75rem;
}}

.vis-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    border-radius: 999px;
    border: 1px solid var(--vis-border);
    background: rgba(7, 11, 22, 0.44);
    color: var(--vis-muted);
    font-size: 0.76rem;
    font-weight: 820;
    padding: 0.3rem 0.62rem;
}}

.vis-status-ok {{
    color: var(--vis-good) !important;
    border-color: rgba(74, 222, 128, 0.35) !important;
    background: var(--vis-good-soft) !important;
}}

.vis-status-warn {{
    color: var(--vis-warn) !important;
    border-color: rgba(250, 204, 21, 0.35) !important;
    background: var(--vis-warn-soft) !important;
}}

.prediction-card {{
    position: relative;
    overflow: hidden;
    border: 1px solid var(--vis-border);
    border-radius: var(--vis-radius-lg);
    background:
        radial-gradient(circle at 100% 0%, var(--label-pill), transparent 40%),
        linear-gradient(180deg, rgba(16, 24, 39, 0.98), rgba(22, 32, 51, 0.96));
    box-shadow: var(--vis-shadow-md);
    padding: 1.45rem 1.55rem 1.35rem;
    margin-bottom: 1rem;
}}

.prediction-card::before {{
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 5px;
    background: var(--label-color);
}}

.prediction-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 999px;
    border: 1px solid var(--label-border);
    background: var(--label-pill);
    color: var(--label-color);
    font-size: 0.77rem;
    font-weight: 900;
    padding: 0.3rem 0.7rem;
    margin-bottom: 0.85rem;
}}

.prediction-label {{
    color: var(--vis-text);
    font-size: 2.15rem;
    font-weight: 950;
    line-height: 1.05;
    letter-spacing: -0.055em;
    margin-bottom: 0.45rem;
}}

.prediction-meta-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.6rem;
    margin-top: 0.85rem;
}}

.prediction-meta-item {{
    border: 1px solid var(--vis-border);
    border-radius: var(--vis-radius-sm);
    background: rgba(7, 11, 22, 0.44);
    padding: 0.72rem 0.82rem;
}}

.prediction-meta-item span {{
    display: block;
    color: var(--vis-muted);
    font-size: 0.7rem;
    font-weight: 900;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.18rem;
}}

.prediction-meta-item b {{
    color: var(--vis-text);
    font-size: 0.94rem;
    font-weight: 850;
}}

.vis-note {{
    border: 1px solid rgba(96, 165, 250, 0.24);
    border-radius: var(--vis-radius-sm);
    background: var(--vis-primary-soft);
    color: var(--vis-text-2);
    font-size: 0.88rem;
    line-height: 1.65;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}}

.vis-note b {{
    color: var(--vis-text);
    font-weight: 900;
}}

.small-muted {{
    color: var(--vis-muted);
    font-size: 0.84rem;
}}

.status-ok {{
    color: var(--vis-good);
    font-weight: 850;
}}

.status-warn {{
    color: var(--vis-warn);
    font-weight: 850;
}}

@media (max-width: 900px) {{
    .vis-hero-grid,
    .vis-metric-grid,
    .prediction-meta-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
}}

@media (max-width: 640px) {{
    .vis-hero {{
        padding: 1.35rem 1.15rem;
    }}

    .vis-hero-grid,
    .vis-metric-grid,
    .prediction-meta-grid {{
        grid-template-columns: 1fr;
    }}

    .vis-title {{
        font-size: 1.85rem;
    }}
}}
</style>"""

    st.markdown(css, unsafe_allow_html=True)


def _html(block: str) -> None:
    """Render one-line HTML safely through Streamlit Markdown.

    Không dùng triple-quote có indent để tránh Streamlit hiểu nhầm thành code block.
    """
    st.markdown(block, unsafe_allow_html=True)


def sidebar_brand() -> None:
    _html(
        '<div class="sidebar-brand">'
        '<div class="sidebar-brand-name">⚡ ViSPhB Control</div>'
        '<div class="sidebar-brand-sub">'
        "Dark-only dashboard. Chọn checkpoint, seed, spiking layer và runtime cho demo."
        "</div>"
        "</div>"
    )


def page_header() -> None:
    _html(
        '<div class="vis-hero">'
        '<div class="vis-kicker">⚡ Vietnamese Sentiment · SNN/KD Demo</div>'
        '<h1 class="vis-title">ViSPhB Sentiment Lab</h1>'
        '<p class="vis-subtitle">'
        "Dark dashboard cho PhoBERT teacher, Tier 1 Direct KD và các student two-stage spiking "
        "trên UIT-VSFC — suy luận, so sánh metric và báo cáo efficiency trong một giao diện gọn gàng."
        "</p>"
        '<div class="vis-hero-grid">'
        '<div class="vis-hero-stat"><b>PhoBERT</b><span>Teacher baseline</span></div>'
        '<div class="vis-hero-stat"><b>Tier 1 – 3</b><span>Student variants</span></div>'
        '<div class="vis-hero-stat"><b>spk2 / spk4</b><span>Spiking layers</span></div>'
        '<div class="vis-hero-stat"><b>Energy</b><span>Metric &amp; report</span></div>'
        "</div>"
        "</div>"
    )


def render_section_header(
    title: str,
    description: str | None = None,
    eyebrow: str | None = None,
) -> None:
    eyebrow_html = f'<div class="vis-kicker">{_safe(eyebrow)}</div>' if eyebrow else ""
    desc_html = f'<div class="vis-section-desc">{_safe(description)}</div>' if description else ""

    _html(
        '<div class="vis-section">'
        f"{eyebrow_html}"
        f'<h2 class="vis-section-title">{_safe(title)}</h2>'
        f"{desc_html}"
        "</div>"
    )


def render_info_note(title: str, body: str) -> None:
    _html(
        '<div class="vis-note">'
        f"<b>{_safe(title)}</b><br>"
        f"{_safe(body)}"
        "</div>"
    )


def render_input_card_header(title: str, subtitle: str, badge: str = "UIT-VSFC") -> None:
    _html(
        '<div class="vis-input-card-head">'
        "<div>"
        f'<div class="vis-input-card-title">{_safe(title)}</div>'
        f'<div class="vis-input-card-subtitle">{_safe(subtitle)}</div>'
        "</div>"
        f'<span class="vis-input-card-badge">{_safe(badge)}</span>'
        "</div>"
    )


def render_model_card(
    display_name: str,
    description: str,
    checkpoint_path: str | None,
    run_dir: Path | str,
    has_checkpoint: bool,
    has_reports: bool,
) -> None:
    ckpt_cls = "vis-status-ok" if has_checkpoint else "vis-status-warn"
    rpt_cls = "vis-status-ok" if has_reports else "vis-status-warn"
    ckpt_text = "checkpoint ready" if has_checkpoint else "missing checkpoint"
    rpt_text = "report ready" if has_reports else "missing report"
    path = checkpoint_path or str(run_dir)

    _html(
        '<div class="vis-model-card">'
        f'<div class="vis-model-name">{_safe(display_name)}</div>'
        f'<div class="vis-model-meta">{_safe(description)}</div>'
        '<div class="vis-chip-row">'
        f'<span class="vis-chip {ckpt_cls}">● {_safe(ckpt_text)}</span>'
        f'<span class="vis-chip {rpt_cls}">● {_safe(rpt_text)}</span>'
        "</div>"
        f'<div class="vis-model-meta" style="margin-top:0.72rem">{_safe(path)}</div>'
        "</div>"
    )


def render_prediction_card(result: PredictionResult) -> None:
    color = LABEL_COLORS.get(result.label, "#60a5fa")
    emoji = LABEL_EMOJIS.get(result.label, "🔎")
    pill = color + "22"
    border = color + "66"

    _html(
        f'<div class="prediction-card" style="--label-color:{color};--label-pill:{pill};--label-border:{border};">'
        f'<div class="prediction-eyebrow">{emoji} Prediction result</div>'
        f'<div class="prediction-label">{_safe(result.label_vi)}</div>'
        f'<div class="small-muted">Raw label: <b>{_safe(result.label)}</b></div>'
        '<div class="prediction-meta-grid">'
        f'<div class="prediction-meta-item"><span>Confidence</span><b>{result.confidence:.4f}</b></div>'
        f'<div class="prediction-meta-item"><span>Latency</span><b>{result.latency_ms:.2f} ms</b></div>'
        f'<div class="prediction-meta-item"><span>Device</span><b>{_safe(result.device)}</b></div>'
        "</div>"
        "</div>"
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
        df.assign(probability=lambda x: x["probability"].map(lambda value: f"{value:.4f}")),
        hide_index=True,
        use_container_width=True,
    )

    chart_df = df.set_index("label_vi")[["probability"]]
    st.bar_chart(chart_df, use_container_width=True)


def render_metric_overview(metrics: Mapping[str, Any]) -> None:
    highlights = ["accuracy", "f1_macro", "f1_weighted", "f1_neutral"]

    cards = []
    for key in highlights:
        cards.append(
            '<div class="vis-metric-card">'
            f'<div class="vis-metric-label">{_safe(METRIC_LABELS[key])}</div>'
            f'<div class="vis-metric-value">{_safe(format_metric(metrics.get(key)))}</div>'
            "</div>"
        )

    _html(f'<div class="vis-metric-grid">{"".join(cards)}</div>')


def render_metrics_table(metrics: Mapping[str, Any]) -> None:
    st.dataframe(metrics_table(metrics), hide_index=True, use_container_width=True)


def render_checkpoint_info(
    checkpoint_path: str,
    run_dir: Path,
    load_info: Mapping[str, Any] | None = None,
) -> None:
    with st.expander("🧩 Thông tin checkpoint", expanded=False):
        st.write(
            {
                "checkpoint": checkpoint_path,
                "run_dir": str(run_dir),
            }
        )

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

    _html(
        f'<span class="{cls}">{checkpoint_status}</span>'
        f'<span class="small-muted"> · {report_status}</span>'
    )