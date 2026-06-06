"""Streamlit application for Vietnamese sentiment inference with ViSPhB models."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from inference import DEFAULT_MODEL_NAME, InferenceError, predict_text
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
    render_metric_overview,
    render_metrics_table,
    render_prediction_card,
    render_probabilities,
    render_variant_status,
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


def sidebar_controls() -> Tuple[str, str, int, bool, str, int, int | None, ModelVariant]:
    st.sidebar.header("Cấu hình demo")

    model_root_value = st.sidebar.text_input(
        "Model root",
        value=default_model_root(),
        help="Có thể dùng models, uit-models hoặc đường dẫn tuyệt đối.",
    )
    resolved_root = resolve_model_root(model_root_value)
    st.sidebar.caption(f"Resolved: {resolved_root}")

    model_name = st.sidebar.text_input("HuggingFace model/local path", value=DEFAULT_MODEL_NAME)
    local_files_only = st.sidebar.checkbox("local_files_only", value=True)
    device_choice = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    max_length = st.sidebar.slider("Max length", min_value=32, max_value=512, value=256, step=32)

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
        spiking_layers = st.sidebar.selectbox(
            "Spiking layers",
            list(spec.spiking_options),
            index=0,
        )
        if family_key == "tier1_direct":
            st.sidebar.caption("Direct KD hiện có report/checkpoint ở cấu hình spk4.")

    variant = get_variant(family_key, model_root_value, seed, spiking_layers)
    st.sidebar.divider()
    st.sidebar.subheader("Model đang chọn")
    st.sidebar.write(variant.display_name)
    render_variant_status(variant.has_checkpoint, variant.has_reports)
    if variant.checkpoint_path:
        st.sidebar.caption(str(variant.checkpoint_path))
    else:
        st.sidebar.caption(str(variant.run_dir))

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
    st.subheader("Single inference")
    text = st.text_area("Câu tiếng Việt", value=DEFAULT_TEXT, height=110)

    col_run, col_status = st.columns([1, 3])
    run_clicked = col_run.button("Chạy suy luận", type="primary", use_container_width=True)
    col_status.caption(variant.description)

    metrics = _variant_metrics(variant)
    render_metric_overview(metrics)

    if run_clicked:
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
        left, right = st.columns([1.05, 1])
        with left:
            render_prediction_card(result)
            if result.runtime_firing_rate is not None:
                st.caption(f"Runtime firing rate: {result.runtime_firing_rate:.4f}")
        with right:
            render_probabilities(result.probabilities)
        render_checkpoint_info(result.checkpoint_path, variant.run_dir, load_info)

    st.divider()
    st.subheader("Metric report của model")
    render_metrics_table(metrics)


def comparison_tab(
    model_root_value: str,
    seed_for_defaults: int,
    max_length: int,
    device_choice: str,
    local_files_only: bool,
    model_name: str,
) -> None:
    st.subheader("So sánh mô hình")
    text = st.text_area("Câu dùng để so sánh", value=DEFAULT_TEXT, height=100, key="comparison_text")

    variants = _available_variants(model_root_value)
    checkpoint_variants = [variant for variant in variants if variant.has_checkpoint]
    if not checkpoint_variants:
        st.warning("Chưa tìm thấy checkpoint nào trong model root hiện tại.")
        return

    option_ids = [variant.variant_id for variant in checkpoint_variants]
    defaults = [item for item in _default_comparison_ids(checkpoint_variants, seed_for_defaults) if item in option_ids]
    selected_ids = st.multiselect(
        "Model chạy đồng thời",
        option_ids,
        default=defaults,
        format_func=lambda vid: variant_from_id(vid, checkpoint_variants).display_name,
    )

    if st.button("Chạy so sánh", type="primary"):
        rows: List[Dict[str, object]] = []
        for variant_id in selected_ids:
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
                        "predicted label": result.label,
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

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def metrics_tab(variant: ModelVariant) -> None:
    st.subheader("Metrics")
    metrics = _variant_metrics(variant)
    render_metric_overview(metrics)
    render_metrics_table(metrics)

    st.caption("Report JSON được hợp nhất từ các file tìm thấy trong thư mục model.")
    for report in raw_reports_for_display(_path_tuple(variant.report_paths)):
        source = report.get("_source_path", "unknown")
        with st.expander(f"Report: {source}", expanded=False):
            if st.checkbox(f"Hiển thị raw JSON: {source}", value=False, key=f"raw_{source}"):
                st.json(report)
            else:
                st.write({"source": source, "top_level_keys": sorted(report.keys())[:40]})


def efficiency_tab(model_root_value: str) -> None:
    st.subheader("Efficiency report")
    st.markdown(
        """
        Theoretical energy saving là ước tính dựa trên chênh lệch chi phí MAC/AC và firing rate của các tầng spiking. Chỉ số này phản ánh lợi ích tiềm năng trên phần cứng event-driven hoặc neuromorphic, không phải mức tiết kiệm đo trực tiếp trên GPU.

        GPU energy trên A100 đo chi phí mô phỏng bằng PyTorch. GPU dense không khai thác cơ chế spike thưa, nên năng lượng GPU có thể cao hơn teacher dù theoretical saving lớn hơn 1.

        Firing rate cao làm giảm lợi ích sparse computation. Khi firing rate tiến gần 1.0, module spiking hoạt động gần như một tầng rate-based dense và lợi thế AC/event-driven bị suy yếu.
        """
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
    st.subheader("About")
    st.markdown(
        """
        Demo này phục vụ trình bày đồ án ViSPhB trên UIT-VSFC. Ứng dụng hỗ trợ PhoBERT teacher, Tier 1 Direct KD và các cấu hình Two-stage KD cho Tier 1, Tier 2, Tier 3 ở hai mức spiking hóa 2 và 4.

        Registry tự tìm checkpoint `.pt` hoặc `.pth`, đọc report `.json`, chuẩn hóa metric và giữ app ổn định khi một model bị lỗi load. Các class SNN được import từ thư mục `code/`; nếu đổi tên file hoặc class, cập nhật `registry.py`.
        """
    )
    st.code(
        f"""VISPHB_MODEL_ROOT={model_root_value} streamlit run demo/app/app.py""",
        language="bash",
    )
    st.write(
        {
            "model_root": str(resolve_model_root(model_root_value)),
            "families": {key: spec.label for key, spec in FAMILY_SPECS.items()},
        }
    )


def main() -> None:
    st.set_page_config(
        page_title="ViSPhB Sentiment Demo",
        page_icon=None,
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
            "Single inference",
            "Model comparison",
            "Metrics",
            "Efficiency report",
            "About",
        ]
    )

    with tabs[0]:
        single_inference_tab(variant, max_length, device_choice, local_files_only, model_name)
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

