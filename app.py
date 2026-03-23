from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from src.pipeline import process_frame


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = ROOT_DIR / "demo" / "test_video.mp4"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "traffic_detector.pt"


def _render_metrics(
    lane_counts: dict[str, int],
    densities: dict[str, float],
    green_times: dict[str, int],
    active_lane: str | None,
) -> None:
    lane_names = list(lane_counts.keys())

    count_columns = st.columns(len(lane_names))
    for column, lane_name in zip(count_columns, lane_names):
        column.metric(f"{lane_name} Count", lane_counts.get(lane_name, 0))

    density_columns = st.columns(len(lane_names))
    for column, lane_name in zip(density_columns, lane_names):
        column.metric(f"{lane_name} Density", f"{densities.get(lane_name, 0.0):.2f}")

    green_columns = st.columns(len(lane_names))
    for column, lane_name in zip(green_columns, lane_names):
        column.metric(f"{lane_name} Green", f"{green_times.get(lane_name, 0)} sec")

    active_column = st.columns(1)[0]
    active_column.metric("Active Lane", active_lane or "None")


def _render_corridor(route: list[str], corridor_states: dict[str, str], active_node: str | None) -> None:
    st.subheader("Emergency Corridor")
    if not route:
        st.write("No active emergency route")
        return

    st.write("Emergency route:", " → ".join(route))
    corridor_columns = st.columns(len(route))
    for column, node in zip(corridor_columns, route):
        label = f"{node}: {corridor_states.get(node, 'WAIT')}"
        if node == active_node:
            column.metric("Active Node", node)
        column.write(label)


def _resolve_video_source(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | None) -> tuple[Path | None, str | None]:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            return Path(temp_file.name), str(Path(temp_file.name))

    if DEFAULT_VIDEO_PATH.exists():
        return DEFAULT_VIDEO_PATH, None

    return None, None


def _release_capture(capture: cv2.VideoCapture | None) -> None:
    if capture is not None:
        capture.release()


def main() -> None:
    st.set_page_config(page_title="FlowX Dashboard", layout="wide")
    st.title("FlowX Traffic Intelligence Dashboard")

    with st.sidebar:
        st.header("Controls")
        cycle_time = st.slider("Cycle time (sec)", min_value=30, max_value=180, value=120, step=5)
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.05,
        )
        frame_skip = st.slider("Frame skip", min_value=1, max_value=5, value=1, step=1)
        emergency_mode = st.toggle("Emergency Mode", value=False)
        uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        start_processing = st.button("Start Processing", type="primary")

    video_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    corridor_placeholder = st.empty()
    corridor_status_placeholder = st.empty()

    video_path, temporary_path = _resolve_video_source(uploaded_video)
    if video_path is None:
        st.info("Upload a video or add `demo/test_video.mp4` to start processing.")
        return

    if not DEFAULT_MODEL_PATH.exists():
        st.error(f"Model not found: {DEFAULT_MODEL_PATH}")
        return

    st.caption(f"Video source: {video_path.name}")

    if not start_processing:
        status_placeholder.info("Ready to process video.")
        return

    capture: cv2.VideoCapture | None = None
    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            st.error(f"Unable to open video: {video_path}")
            return

        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                status_placeholder.success("Video processing completed.")
                break

            frame_index += 1
            if frame_index % frame_skip != 0:
                continue

            try:
                result = process_frame(
                    frame,
                    {
                        "model_path": str(DEFAULT_MODEL_PATH),
                        "confidence_threshold": confidence_threshold,
                        "cycle_time": cycle_time,
                        "emergency_mode": emergency_mode,
                        "minimum_green": 10,
                    },
                )
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                break

            frame_rgb = cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            if emergency_mode and result["emergency"]:
                status_placeholder.error("🚑 Emergency Mode Activated")
                corridor_status_placeholder.warning("🚑 Emergency Corridor Active")
            else:
                emergency_mode_status = "ON" if emergency_mode else "OFF"
                active_lane_text = result["active_lane"] or "None"
                status_placeholder.info(
                    f"Active lane: {active_lane_text} | Emergency mode: {emergency_mode_status}"
                )
                corridor_status_placeholder.empty()

            with metrics_placeholder.container():
                _render_metrics(
                    lane_counts=result["lane_counts"],
                    densities=result["densities"],
                    green_times=result["green_times"],
                    active_lane=result["active_lane"],
                )

            with corridor_placeholder.container():
                _render_corridor(
                    route=result.get("emergency_route", []),
                    corridor_states=result.get("corridor_states", {}),
                    active_node=result.get("active_node"),
                )
    finally:
        _release_capture(capture)
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
