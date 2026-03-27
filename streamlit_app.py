#!/usr/bin/env python3
"""Minimal Streamlit interface for the EchoPilot ReAct agent."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import base64

import streamlit as st

from agents import get_intelligent_agent
from config import Config
from utils.video_utils import convert_video_to_h264

PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def load_agent():
    IntelligentAgent, _ = get_intelligent_agent()
    return IntelligentAgent(device=Config.DEVICE)


def _persist_upload(upload) -> Tuple[Path, Path]:
    """Write uploaded file to a temporary directory and return its path."""
    suffix = Path(upload.name or "input.mp4").suffix or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="echopilot_"))
    video_path = temp_dir / f"input{suffix}"
    with open(video_path, "wb") as handle:
        handle.write(upload.getbuffer())
    return video_path, temp_dir


def _extract_key_metrics(response) -> List[Tuple[str, str]]:
    metrics: List[Tuple[str, str]] = []
    results = response.execution_result.results or {}
    tool_results: Dict[str, Dict] = results.get("tool_results") or {}
    measurement = tool_results.get("echo_measurement_prediction")
    if isinstance(measurement, dict) and measurement.get("status") == "success":
        entries = measurement.get("measurements") or []
        if entries:
            data = entries[0].get("measurements", {})

            def _format_metric(key: str, label: str, precision: int = 1):
                info = data.get(key)
                if not isinstance(info, dict):
                    return
                value = info.get("value")
                unit = info.get("unit", "")
                if value is None:
                    return
                try:
                    value_str = f"{float(value):.{precision}f}"
                except (TypeError, ValueError):
                    value_str = str(value)
                unit_str = f" {unit}".strip()
                metrics.append((label, f"{value_str}{unit_str}"))

            for key, label in [
                ("ejection_fraction", "Ejection Fraction"),
                ("EF", "Ejection Fraction"),
            ]:
                if key in data:
                    _format_metric(key, label, precision=1)
                    break

            if "pulmonary_artery_pressure_continuous" in data:
                _format_metric("pulmonary_artery_pressure_continuous", "Pulmonary Artery Pressure", precision=1)
            if "dilated_ivc" in data:
                _format_metric("dilated_ivc", "IVC Diameter", precision=2)
    return metrics


def _render_looping_preview(uploaded_video) -> None:
    """Render the uploaded video as a looping, muted preview."""
    try:
        video_bytes = uploaded_video.getvalue()
    except Exception:
        return
    if not video_bytes:
        return

    b64 = base64.b64encode(video_bytes).decode("utf-8")
    st.markdown(
        f"""
        <video autoplay loop muted playsinline controls style="width: 100%; border-radius: 8px;">
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """,
        unsafe_allow_html=True,
    )


def _extract_segmentation_outputs(response) -> Dict[str, str]:
    """Best-effort extraction of segmentation video paths from tool results."""
    outputs: Dict[str, str] = {}
    try:
        results = response.execution_result.results or {}
        tool_results: Dict[str, Dict] = results.get("tool_results") or {}
        seg = tool_results.get("echo_segmentation")
    except Exception:
        return outputs

    def _harvest_from_obj(obj: Dict) -> None:
        if not isinstance(obj, dict):
            return
        # Direct paths
        for key in ("overlay_video_path", "mask_video_path"):
            val = obj.get(key)
            if isinstance(val, str):
                # Normalize to absolute path rooted at the project directory so
                # Streamlit can always locate the file.
                p = Path(val)
                if not p.is_absolute():
                    p = (PROJECT_ROOT / val).resolve()
                outputs[key] = str(p)
        # Nested in common wrappers
        for nested_key in ("output", "outputs", "data", "result"):
            nested = obj.get(nested_key)
            if isinstance(nested, dict):
                _harvest_from_obj(nested)

    if isinstance(seg, dict):
        # Common pattern: {"status": "...", ...}
        _harvest_from_obj(seg)
    elif isinstance(seg, (list, tuple)):
        for item in seg:
            if isinstance(item, dict):
                _harvest_from_obj(item)

    return outputs


def main() -> None:
    st.set_page_config(page_title="EchoPilot Agent", page_icon="🫀", layout="wide")
    st.title("EchoPilot · Echocardiography Co-Pilot")
    st.caption("Upload a study, ask a question, and run the analysis.")

    upload_col, preview_col = st.columns([2, 1])
    with upload_col:
        st.markdown("### Study & question")
        uploaded_video = st.file_uploader(
            "Echo video file",
            type=["mp4", "mov", "m4v", "avi", "wmv"],
            help="Standard ultrasound formats are supported.",
        )
        default_question = "Estimate the ejection fraction and note any major abnormalities."
        query = st.text_area("Clinical question", value=default_question, height=140)

    with preview_col:
        st.markdown("### Preview")
        if uploaded_video:
            # Show the uploaded clip as a looping, GIF-like preview.
            _render_looping_preview(uploaded_video)
        else:
            st.info("Upload a video to see the preview here.")

    response = None
    display_video: Path | None = None

    run_clicked = st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_video or not query.strip(),
    )
    if run_clicked:
        agent = load_agent()
        video_path, temp_dir = _persist_upload(uploaded_video)
        temp_display_dir = PROJECT_ROOT / "temp"
        temp_display_dir.mkdir(parents=True, exist_ok=True)
        display_target = temp_display_dir / f"display_{uuid.uuid4().hex}.mp4"
        display_video = Path(convert_video_to_h264(str(video_path), str(display_target)))

        with st.spinner("EchoPilot is analyzing the study..."):
            response = agent.process_query(query.strip(), str(video_path))

        # Clean up the original upload to save disk space
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                item.unlink(missing_ok=True)
            temp_dir.rmdir()

    if response:
        st.success("Analysis complete")
        metrics = _extract_key_metrics(response)
        seg_outputs = _extract_segmentation_outputs(response)

        container = st.container()
        video_col, metrics_col = container.columns([2, 1])

        overlay_path = seg_outputs.get("overlay_video_path")
        mask_path = seg_outputs.get("mask_video_path")

        with video_col:
            if display_video and display_video.exists():
                st.video(str(display_video))

            # If available, also show segmentation overlay/mask videos just below.
            if overlay_path or mask_path:
                st.markdown("#### Segmentation")
                if overlay_path:
                    st.video(overlay_path)
                elif mask_path:
                    st.video(mask_path)
        if metrics:
            with metrics_col:
                st.markdown("#### Key Measurements")
                for label, value in metrics:
                    st.metric(label, value)

        st.divider()
        st.markdown("#### EchoPilot Response")
        st.chat_message("user").write(query.strip())
        st.chat_message("assistant").write(response.response_text)


if __name__ == "__main__":
    main()
