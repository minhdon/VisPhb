"""Streamlit application for Vietnamese sentiment inference with ViSPhB models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from inference import DEFAULT_MODEL_NAME, predict_text
from registry import (
    FAMILY_SPECS,
    SEEDS,
    ModelVariant,
    default_model_root,
    family_options,
    get_variant,
    iter_variants,
    resolve_model_root,
    variant_from_id,
)
from reports import comparison_metric_row, format_metric, normalize_reports, raw_reports_for_display
from ui import (
    inject_css,
    page_header,
    render_checkpoint_info,
    render_info_note,
    render_input_card_header,
    render_metric_overview,
    render_metrics_table,
    render_model_card,
    render_prediction_card,
    render_probabilities,
    render_section_header,
    sidebar_brand,
)


DEFAULT_TEXT = "Môn học này rất thú vị, giảng viên giải thích dễ hiểu và bài tập vừa sức."


def _path_tuple(paths: Tuple[Path, ...]) -> Tuple[str, ...]:
    return tuple(str(path) for path in paths)


def _variant_metrics(variant: ModelVariant) -> Dict[str, object]:
    return normalize_reports(_path_tuple(variant.report_paths), variant.seed)


def _available_variants(model_root: str) -> List[ModelVariant]:
    return iter_variants(model_root, only_existing=True)


def _default_comparison_ids(variants: List[ModelVariant], seed: int) -> List[str]:
    wanted = [
        ("teacher", seed, None),
        ("tier1_direct", seed, 4),
        ("tier2_twostage", seed, 2),
        ("tier2_twostage", seed, 4),
    ]

    defaults: List[str] = []
    for family, wanted_seed, wanted_spk in wanted:
        for variant in variants:
            if (
                variant.family_key == family
                and variant.seed == wanted_seed
                and variant.spiking_layers == wanted_spk
                and variant.has_checkpoint
            ):
                defaults.append(variant.variant_id)
                break

    return defaults


def _style_status_column(value: str) -> str:
    if value == "OK":
        return "background-color: #052e16; color: #86efac; font-weight: 800"
    return "background-color: #450a0a; color: #fecaca; font-weight: 800"


def sidebar_controls() -> Tuple[str, str, int, bool, str, int, int | None, ModelVariant]:
    sidebar_brand()

    st.sidebar.markdown("### 📁 Project")
    model_root_value = st.sidebar.text_input(
        "Model root",
        value=default_model_root(),
        help="Có thể dùng models, uit-models hoặc đường dẫn tuyệt đối.",
    )
    resolved_root = resolve_model_root(model_root_value)
    st.sidebar.caption(f"Resolved: {resolved_root}")

    st.sidebar.markdown("### 🧠 Runtime")
    model_name = st.sidebar.text_input("HuggingFace model/local path", value=DEFAULT_MODEL_NAME)
    local_files_only = st.sidebar.toggle("local_files_only", value=True)
    device_choice = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    max_length = st.sidebar.selectbox("Max length", [64, 128, 256, 384, 512], index=2)

    st.sidebar.markdown("### 🧪 Model")
    options = family_options()
    family_key = st.sidebar.selectbox(
        "Model family",
        list(options.keys()),
        format_func=lambda key: options[key],
    )
    spec = FAMILY_SPECS[family_key]

    seed = st.sidebar.selectbox("Seed", list(SEEDS), index=0) if spec.supports_seed else None

    spiking_layers = None
    if spec.spiking_options:
        spiking_layers = st.sidebar.radio(
            "Spiking layers",
            list(spec.spiking_options),
            index=0,
            horizontal=True,
        )
        if family_key == "tier1_direct":
            st.sidebar.caption("Direct KD hiện có report/checkpoint ở cấu hình spk4.")

    variant = get_variant(family_key, model_root_value, seed, spiking_layers)

    st.sidebar.divider()
    render_model_card(
        display_name=variant.display_name,
        description=variant.description,
        checkpoint_path=str(variant.checkpoint_path) if variant.checkpoint_path else None,
        run_dir=variant.run_dir,
        has_checkpoint=variant.has_checkpoint,
        has_reports=variant.has_reports,
    )

    return (
        model_root_value,
        device_choice,
        int(max_length),
        bool(local_files_only),
        model_name,
        int(seed) if seed is not None else 42,
        spiking_layers,
        variant,
    )


def single_inference_tab(
    variant: ModelVariant,
    max_length: int,
    device_choice: str,
    local_files_only: bool,
    model_name: str,
) -> None:
    render_section_header(
        "Single inference",
        "Nhập một câu tiếng Việt, chạy model đang chọn và xem xác suất từng nhãn.",
        "🔎 Live prediction",
    )

    metrics = _variant_metrics(variant)

    input_card = st.container()
    with input_card:
        st.markdown(
            '<span class="vis-card-anchor" data-card="single-input"></span>',
            unsafe_allow_html=True,
        )
        render_input_card_header(
            title="Vietnamese sentence",
            subtitle="Nhập câu cần phân loại cảm xúc. Card này bao trọn textarea và nút chạy suy luận.",
            badge="UIT-VSFC",
        )
        text = st.text_area(
            "Câu tiếng Việt",
            value=DEFAULT_TEXT,
            height=146,
            placeholder="Ví dụ: Môn học này rất hay, giảng viên dạy dễ hiểu...",
            label_visibility="collapsed",
        )
        run_clicked = st.button("🚀 Chạy suy luận", type="primary", use_container_width=True)
        st.caption(variant.description)

    render_model_card(
        display_name=variant.display_name,
        description="Model hiện đang được chọn trong sidebar.",
        checkpoint_path=str(variant.checkpoint_path) if variant.checkpoint_path else None,
        run_dir=variant.run_dir,
        has_checkpoint=variant.has_checkpoint,
        has_reports=variant.has_reports,
    )

    render_metric_overview(metrics)

    if run_clicked:
        with st.spinner("Đang load checkpoint và chạy suy luận..."):
            try:
                result, load_info = predict_text(
                    variant,
                    text,
                    max_length=max_length,
                    device_choice=device_choice,
                    local_files_only=local_files_only,
                    model_name=model_name,
                )
                st.session_state["single_result"] = result
                st.session_state["single_load_info"] = load_info
            except Exception as exc:
                st.session_state.pop("single_result", None)
                st.session_state.pop("single_load_info", None)
                st.error(f"Không chạy được model: {exc}")

    result = st.session_state.get("single_result")
    load_info = st.session_state.get("single_load_info", {})

    if result is not None:
        st.divider()
        render_prediction_card(result)

        if result.runtime_firing_rate is not None:
            st.info(f"Runtime firing rate: {result.runtime_firing_rate:.4f}")

        render_section_header(
            "Label probabilities",
            "Xác suất softmax cho ba nhãn cảm xúc.",
        )
        render_probabilities(result.probabilities)
        render_checkpoint_info(result.checkpoint_path, variant.run_dir, load_info)

    st.divider()
    render_section_header(
        "Metric report của model",
        "Các metric được chuẩn hóa từ report JSON tìm thấy trong run directory.",
    )
    render_metrics_table(metrics)


def comparison_tab(
    model_root_value: str,
    seed_for_defaults: int,
    max_length: int,
    device_choice: str,
    local_files_only: bool,
    model_name: str,
) -> None:
    render_section_header(
        "Model comparison",
        "Chạy nhiều model trên cùng một câu để so sánh prediction, latency và metric đã lưu.",
        "⚖️ Compare",
    )

    compare_card = st.container()
    with compare_card:
        st.markdown(
            '<span class="vis-card-anchor" data-card="comparison-input"></span>',
            unsafe_allow_html=True,
        )
        render_input_card_header(
            title="Comparison sentence",
            subtitle="Một câu đầu vào dùng chung cho tất cả model được chọn.",
            badge="Multi-model",
        )
        text = st.text_area(
            "Câu dùng để so sánh",
            value=DEFAULT_TEXT,
            height=122,
            key="comparison_text",
            label_visibility="collapsed",
        )

    variants = _available_variants(model_root_value)
    checkpoint_variants = [variant for variant in variants if variant.has_checkpoint]

    if not checkpoint_variants:
        st.warning("Chưa tìm thấy checkpoint nào trong model root hiện tại.")
        return

    option_ids = [variant.variant_id for variant in checkpoint_variants]
    defaults = [
        item
        for item in _default_comparison_ids(checkpoint_variants, seed_for_defaults)
        if item in option_ids
    ]

    selected_ids = st.multiselect(
        "Model chạy đồng thời",
        option_ids,
        default=defaults,
        format_func=lambda vid: variant_from_id(vid, checkpoint_variants).display_name,
    )

    if st.button("🚀 Chạy so sánh", type="primary", use_container_width=True):
        rows: List[Dict[str, object]] = []
        progress = st.progress(0.0)

        for index, variant_id in enumerate(selected_ids, start=1):
            variant = variant_from_id(variant_id, checkpoint_variants)
            metrics = _variant_metrics(variant)
            base_row: Dict[str, object] = {
                "model": variant.family_label,
                "seed": variant.seed if variant.seed is not None else "N/A",
                "spiking_layers": variant.spiking_layers if variant.spiking_layers is not None else "N/A",
            }

            try:
                result, _ = predict_text(
                    variant,
                    text,
                    max_length=max_length,
                    device_choice=device_choice,
                    local_files_only=local_files_only,
                    model_name=model_name,
                )
                base_row.update(
                    {
                        "predicted label": result.label_vi,
                        "confidence": f"{result.confidence:.4f}",
                        "latency (ms)": f"{result.latency_ms:.2f}",
                        "status": "OK",
                    }
                )
            except Exception as exc:
                base_row.update(
                    {
                        "predicted label": "N/A",
                        "confidence": "N/A",
                        "latency (ms)": "N/A",
                        "status": str(exc),
                    }
                )

            base_row.update(comparison_metric_row(metrics))
            rows.append(base_row)
            progress.progress(index / max(len(selected_ids), 1))

        if rows:
            df = pd.DataFrame(rows)
            styled = df.style.applymap(_style_status_column, subset=["status"])
            st.dataframe(styled, hide_index=True, use_container_width=True)
        else:
            st.info("Chưa chọn model nào để so sánh.")


def metrics_tab(variant: ModelVariant) -> None:
    render_section_header(
        "Metrics",
        "Xem metric đã chuẩn hóa và mở raw JSON khi cần kiểm tra nguồn số liệu.",
        "📊 Reports",
    )

    metrics = _variant_metrics(variant)
    render_metric_overview(metrics)
    render_metrics_table(metrics)

    st.caption("Report JSON được hợp nhất từ các file tìm thấy trong thư mục model.")

    for report in raw_reports_for_display(_path_tuple(variant.report_paths)):
        source = report.get("_source_path", "unknown")
        with st.expander(f"📄 Report: {source}", expanded=False):
            if st.checkbox(f"Hiển thị raw JSON: {source}", value=False, key=f"raw_{source}"):
                st.json(report)
            else:
                st.write(
                    {
                        "source": source,
                        "top_level_keys": sorted(report.keys())[:40],
                    }
                )


def efficiency_tab(model_root_value: str) -> None:
    render_section_header(
        "Efficiency report",
        "Tổng hợp firing rate, theoretical saving, GPU energy và latency từ các report hiện có.",
        "🌱 Efficiency",
    )

    render_info_note(
        "Cách đọc energy",
        "Theoretical energy saving là ước tính theo MAC/AC và firing rate. "
        "GPU energy trên A100 là chi phí mô phỏng PyTorch, nên có thể không giảm dù theoretical saving lớn.",
    )

    rows: List[Dict[str, object]] = []

    for variant in _available_variants(model_root_value):
        if not variant.has_reports:
            continue

        metrics = _variant_metrics(variant)
        rows.append(
            {
                "model": variant.family_label,
                "seed": variant.seed if variant.seed is not None else "N/A",
                "spiking_layers": variant.spiking_layers if variant.spiking_layers is not None else "N/A",
                "F1-macro": format_metric(metrics.get("f1_macro")),
                "Firing rate": format_metric(metrics.get("mean_firing_rate")),
                "Theoretical saving": format_metric(metrics.get("theoretical_energy_saving")),
                "Peak memory MB": format_metric(metrics.get("peak_memory_mb")),
                "GPU energy/sample J": format_metric(metrics.get("gpu_energy_per_sample_j")),
                "Latency ms": format_metric(metrics.get("latency_ms")),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Chưa tìm thấy report efficiency trong model root hiện tại.")


def about_tab(model_root_value: str) -> None:
    render_section_header(
        "About",
        "Thông tin vận hành nhanh cho demo bảo vệ đồ án.",
        "ℹ️ Project",
    )

    render_info_note(
        "ViSPhB Streamlit Demo",
        "Ứng dụng hỗ trợ PhoBERT teacher, Tier 1 Direct KD và các cấu hình Two-stage KD "
        "cho Tier 1, Tier 2, Tier 3 ở hai mức spiking hóa 2 và 4.",
    )

    st.code(
        f"""VISPHB_MODEL_ROOT={model_root_value} streamlit run demo/app/app.py""",
        language="bash",
    )

    st.json(
        {
            "theme": "dark-only",
            "model_root": str(resolve_model_root(model_root_value)),
            "families": {key: spec.label for key, spec in FAMILY_SPECS.items()},
        }
    )


def main() -> None:
    st.set_page_config(
        page_title="ViSPhB Sentiment Lab",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()
    page_header()

    (
        model_root_value,
        device_choice,
        max_length,
        local_files_only,
        model_name,
        seed_for_defaults,
        _spiking_layers,
        variant,
    ) = sidebar_controls()

    tabs = st.tabs(
        [
            "🔎 Single inference",
            "⚖️ Model comparison",
            "📊 Metrics",
            "🌱 Efficiency report",
            "ℹ️ About",
        ]
    )

    with tabs[0]:
        single_inference_tab(
            variant,
            max_length,
            device_choice,
            local_files_only,
            model_name,
        )

    with tabs[1]:
        comparison_tab(
            model_root_value,
            seed_for_defaults,
            max_length,
            device_choice,
            local_files_only,
            model_name,
        )

    with tabs[2]:
        metrics_tab(variant)

    with tabs[3]:
        efficiency_tab(model_root_value)

    with tabs[4]:
        about_tab(model_root_value)


if __name__ == "__main__":
    main()