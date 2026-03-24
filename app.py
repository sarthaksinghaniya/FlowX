from __future__ import annotations

from collections import deque
import tempfile
from time import perf_counter
import time
from pathlib import Path

import cv2
import pydeck as pdk
import streamlit as st

from src.comparison import simulate_ai, simulate_static
from src.density import DensityCalculator
from src.emergency_corridor import compute_path
from src.pipeline import process_frame
from src.signal_control import SignalController


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = ROOT_DIR / "demo" / "test_video.mp4"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "traffic_detector.pt"
DEMO_EMERGENCY_TRIGGER_FRAME = 45
_DENSITY_CALCULATOR = DensityCalculator()
INTERSECTIONS = [
    {"id": 1, "lat": 26.8467, "lon": 80.9462},
    {"id": 2, "lat": 26.8475, "lon": 80.9470},
    {"id": 3, "lat": 26.8482, "lon": 80.9480},
]


def _resolve_density_map(result: dict) -> dict[str, float]:
    densities = result.get("densities", {})
    if isinstance(densities, dict):
        return {str(k): float(v) for k, v in densities.items()}

    if isinstance(densities, list):
        lane_order = result.get("lane_order", [])
        if isinstance(lane_order, list) and lane_order and len(lane_order) == len(densities):
            return {str(lane): float(densities[idx]) for idx, lane in enumerate(lane_order)}
        return {f"lane_{idx + 1}": float(value) for idx, value in enumerate(densities)}

    return {}


def _render_metrics(
    lane_counts: dict[str, int],
    densities: dict[str, float],
    green_times: dict[str, int],
    active_lane: str | None,
    fps: float,
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

    summary_columns = st.columns(2)
    summary_columns[0].metric("Active Lane", active_lane or "None")
    summary_columns[1].metric("FPS", f"{fps:.1f}")


def get_signal_color(lane: int, selected_lane: int | None, timer: int) -> str:
    if lane == selected_lane:
        if timer <= 3:
            return "yellow"
        return "green"
    return "red"


def _lane_to_index(lane_value: str | int | None) -> int | None:
    if isinstance(lane_value, int):
        return lane_value if lane_value >= 0 else None
    if isinstance(lane_value, str) and lane_value.startswith("lane_"):
        lane_number = lane_value.split("_")[-1]
        if lane_number.isdigit():
            return max(int(lane_number) - 1, 0)
    return None


def _render_signal_panel(selected_lane: int | None, timer: int) -> None:
    st.subheader("Traffic Signal Animation")
    lane_label = f"Lane {selected_lane + 1}" if selected_lane is not None else "None"
    st.write(f"Current Lane: {lane_label}")
    st.write(f"Time Left: {max(timer, 0)}s")

    cols = st.columns(4)
    for i, col in enumerate(cols):
        color = get_signal_color(i, selected_lane, timer)
        lane_text = f"Lane {i + 1}"
        if color == "green":
            col.success(f"{lane_text}\n🟢 GREEN")
        elif color == "yellow":
            col.warning(f"{lane_text}\n🟡 YELLOW")
        else:
            col.error(f"{lane_text}\n🔴 RED")


def _tick_signal_state(speed: float) -> None:
    now = perf_counter()
    last_tick = float(st.session_state.signal_last_tick)
    elapsed = max(0.0, now - last_tick)
    tick_count = int(elapsed * speed)
    if tick_count <= 0:
        return

    for _ in range(tick_count):
        st.session_state.timer = max(int(st.session_state.timer) - 1, 0)
        if st.session_state.timer == 0:
            pending_lane = st.session_state.pending_signal_lane
            pending_green = max(int(st.session_state.pending_green_time), 4)
            if pending_lane is not None:
                st.session_state.signal_lane = pending_lane
            st.session_state.timer = pending_green

    st.session_state.signal_last_tick = last_tick + (tick_count / speed)


def _update_signal_animation(result: dict, speed: float) -> tuple[int | None, int]:
    active_lane = result.get("active_lane")
    selected_lane = _lane_to_index(active_lane)
    green_times = result.get("green_times", {})
    green_time = 10
    if isinstance(active_lane, str) and isinstance(green_times, dict):
        green_time = max(int(green_times.get(active_lane, 10)), 4)

    if st.session_state.signal_lane is None and selected_lane is not None:
        st.session_state.signal_lane = selected_lane
        st.session_state.pending_signal_lane = selected_lane
        st.session_state.pending_green_time = green_time
        st.session_state.timer = green_time
        st.session_state.signal_last_tick = perf_counter()
    else:
        if selected_lane is not None:
            st.session_state.pending_signal_lane = selected_lane
            st.session_state.pending_green_time = green_time
        _tick_signal_state(speed)

    return st.session_state.signal_lane, int(st.session_state.timer)


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


def _state_label(raw_state: str | None) -> str:
    if raw_state == "EMERGENCY_MODE":
        return "EMERGENCY"
    if raw_state == "CRASH_MODE":
        return "CRASH"
    return "NORMAL"


def _render_state_and_decision(result: dict) -> None:
    state = _state_label(result.get("state"))
    reason = result.get("reason") or "Normal flow allocation"
    selected_lane = result.get("selected_lane")
    if selected_lane is None or int(selected_lane) < 0:
        selected_lane_text = "N/A"
    else:
        selected_lane_text = f"Lane {int(selected_lane) + 1}"

    state_color = {
        "NORMAL": "#16a34a",
        "CRASH": "#f59e0b",
        "EMERGENCY": "#dc2626",
    }.get(state, "#475569")

    st.markdown(
        f"""
        <div style="border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;background:#f8fafc;">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
            <div style="font-size:0.85rem;color:#475569;">SYSTEM STATE</div>
            <div style="padding:4px 10px;border-radius:999px;background:{state_color};color:white;font-weight:700;font-size:0.78rem;">
              {state}
            </div>
          </div>
          <div style="margin-top:10px;font-size:0.9rem;color:#0f172a;font-weight:600;">AI Decision:</div>
          <div style="font-size:1rem;color:#111827;">
            {selected_lane_text} prioritized: {reason}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_crash_panel(result: dict) -> None:
    crash_class = result.get("crash_class")
    crash_conf = float(result.get("crash_confidence", 0.0))
    if not crash_class:
        crash_class = "minor"
        crash_conf = 0.0
        status_text = "No crash signal"
    else:
        status_text = crash_class.upper()

    color = {
        "major": "#dc2626",
        "moderate": "#f59e0b",
        "minor": "#16a34a",
    }.get(str(crash_class).lower(), "#16a34a")

    st.markdown(
        f"""
        <div style="border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;background:white;">
          <div style="font-size:0.85rem;color:#475569;">CRASH PANEL</div>
          <div style="margin-top:8px;display:flex;align-items:center;gap:10px;">
            <div style="width:12px;height:12px;border-radius:999px;background:{color};"></div>
            <div style="font-size:1rem;font-weight:700;color:#0f172a;">{status_text}</div>
          </div>
          <div style="margin-top:8px;font-size:0.9rem;color:#334155;">
            Confidence: {crash_conf:.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_flow_graphs(result: dict) -> None:
    st.subheader("Flow Intelligence Graphs")

    densities = _resolve_density_map(result)
    if densities:
        st.caption("Density per lane")
        st.bar_chart(densities)
    else:
        st.caption("Density per lane")
        st.write("No density data")

    history = st.session_state.signal_timeline
    if history:
        st.caption("Signal timeline")
        chart_data: dict[str, list[float]] = {}
        for entry in history:
            for lane_name in entry:
                chart_data.setdefault(lane_name, []).append(float(entry[lane_name]))
            for lane_name in list(chart_data.keys()):
                if lane_name not in entry:
                    chart_data[lane_name].append(0.0)
        st.line_chart(chart_data)
    else:
        st.caption("Signal timeline")
        st.write("No timeline data yet")


def _state_color(raw_state: str | None) -> list[int]:
    if raw_state == "EMERGENCY_MODE":
        return [255, 0, 0]
    if raw_state == "CRASH_MODE":
        return [255, 165, 0]
    return [0, 255, 0]


def _simulate_intersections(result: dict, processed_frames: int, emergency_mode: bool) -> list[dict]:
    selected_lane = _lane_to_index(result.get("active_lane"))
    fallback_state = str(result.get("state") or "NORMAL")
    intersections: list[dict] = []

    for index, node in enumerate(INTERSECTIONS):
        active_lane = ((processed_frames // max(1, 6 - index)) + index) % 4
        state = "NORMAL"

        if index == 0 and selected_lane is not None:
            active_lane = selected_lane
            state = fallback_state
        elif emergency_mode and processed_frames >= DEMO_EMERGENCY_TRIGGER_FRAME and index == 1:
            state = "EMERGENCY_MODE"
        elif processed_frames % 18 in range(5, 9) and index == 2:
            state = "CRASH_MODE"

        intersections.append(
            {
                "id": node["id"],
                "lat": node["lat"],
                "lon": node["lon"],
                "active_lane": active_lane + 1,
                "state": state,
                "color": _state_color(state),
            }
        )

    return intersections


def _road_segments(intersections: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for start, end in zip(intersections, intersections[1:]):
        segments.append(
            {
                "source": [start["lon"], start["lat"]],
                "target": [end["lon"], end["lat"]],
                "color": [80, 80, 80],
            }
        )
    return segments


def _render_map_view(intersections: list[dict]) -> None:
    st.subheader("🗺️ Traffic Map View")
    st.markdown('<div id="flowx_mapbox"></div>', unsafe_allow_html=True)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=intersections,
        get_position="[lon, lat]",
        get_color="color",
        get_radius=80,
        pickable=True,
    )
    roads_layer = pdk.Layer(
        "LineLayer",
        data=_road_segments(intersections),
        get_source_position="source",
        get_target_position="target",
        get_color="color",
        get_width=6,
        pickable=False,
    )
    view_state = pdk.ViewState(
        latitude=26.8475,
        longitude=80.9470,
        zoom=14,
        pitch=45,
    )
    tooltip = {
        "html": "<b>Intersection {id}</b><br/>State: {state}<br/>Lane: {active_lane}",
    }
    deck = pdk.Deck(
        layers=[roads_layer, layer],
        initial_view_state=view_state,
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)


def _render_comparison(
    ai_metrics: dict,
    static_metrics: dict,
) -> None:
    st.subheader("📊 Performance Comparison")

    ai_summary = ai_metrics.get("summary", {})
    static_summary = static_metrics.get("summary", {})

    comparison_columns = st.columns(3)
    comparison_columns[0].metric("AI Wait Time", f"{ai_summary.get('wait_time', 0.0):.2f}s")
    comparison_columns[0].metric("Static Wait Time", f"{static_summary.get('wait_time', 0.0):.2f}s")

    comparison_columns[1].metric("AI Queue Length", f"{ai_summary.get('queue_length', 0.0):.2f}")
    comparison_columns[1].metric("Static Queue Length", f"{static_summary.get('queue_length', 0.0):.2f}")

    comparison_columns[2].metric("AI Throughput", f"{ai_summary.get('throughput', 0.0):.2f}")
    comparison_columns[2].metric("Static Throughput", f"{static_summary.get('throughput', 0.0):.2f}")

    static_wait = float(static_summary.get("wait_time", 0.0))
    ai_wait = float(ai_summary.get("wait_time", 0.0))
    if static_wait > 0 and ai_wait <= static_wait:
        improvement = ((static_wait - ai_wait) / static_wait) * 100
        st.success(f"AI reduces wait time by {improvement:.1f}%")

    chart_data = {
        "AI": {
            "Wait Time": ai_summary.get("wait_time", 0.0),
            "Queue Length": ai_summary.get("queue_length", 0.0),
        },
        "Static": {
            "Wait Time": static_summary.get("wait_time", 0.0),
            "Queue Length": static_summary.get("queue_length", 0.0),
        },
    }
    st.bar_chart(chart_data)


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


def _init_session_state() -> None:
    if "event_log" not in st.session_state:
        st.session_state.event_log = deque(maxlen=6)
    if "last_active_lane" not in st.session_state:
        st.session_state.last_active_lane = None
    if "last_emergency" not in st.session_state:
        st.session_state.last_emergency = False
    if "last_corridor" not in st.session_state:
        st.session_state.last_corridor = False
    if "playback_reset_id" not in st.session_state:
        st.session_state.playback_reset_id = 0
    if "signal_timeline" not in st.session_state:
        st.session_state.signal_timeline = deque(maxlen=60)
    if "signal_lane" not in st.session_state:
        st.session_state.signal_lane = None
    if "pending_signal_lane" not in st.session_state:
        st.session_state.pending_signal_lane = None
    if "pending_green_time" not in st.session_state:
        st.session_state.pending_green_time = 10
    if "timer" not in st.session_state:
        st.session_state.timer = 0
    if "signal_last_tick" not in st.session_state:
        st.session_state.signal_last_tick = perf_counter()


def _log_event(message: str) -> None:
    st.session_state.event_log.appendleft(message)


def _reset_demo() -> None:
    st.session_state.event_log = deque(maxlen=6)
    st.session_state.last_active_lane = None
    st.session_state.last_emergency = False
    st.session_state.last_corridor = False
    st.session_state.playback_reset_id += 1
    st.session_state.signal_timeline = deque(maxlen=60)
    st.session_state.signal_lane = None
    st.session_state.pending_signal_lane = None
    st.session_state.pending_green_time = 10
    st.session_state.timer = 0
    st.session_state.signal_last_tick = perf_counter()


def _update_signal_timeline(result: dict) -> None:
    green_times = result.get("green_times", {})
    if not isinstance(green_times, dict) or not green_times:
        return
    st.session_state.signal_timeline.append({lane: float(sec) for lane, sec in green_times.items()})


def _generate_demo_corridor(route: list[str], current_index: int) -> dict[str, str]:
    states: dict[str, str] = {}
    for index, node in enumerate(route):
        if index == current_index:
            states[node] = "GREEN"
        elif index == current_index + 1:
            states[node] = "PREPARE"
        else:
            states[node] = "WAIT"
    return states


def _apply_demo_scenario(
    result: dict,
    processed_frames: int,
    cycle_time: int,
    emergency_mode: bool,
) -> dict:
    lane_counts = {
        "lane_1": 4,
        "lane_2": 18 + (processed_frames % 5),
        "lane_3": 3,
        "lane_4": 2,
    }
    densities = _DENSITY_CALCULATOR.compute(lane_counts)
    green_times = SignalController(cycle_time=cycle_time, minimum_green=10).allocate_green_times(densities)
    active_lane = max(green_times, key=green_times.get)
    emergency_detected = processed_frames >= DEMO_EMERGENCY_TRIGGER_FRAME
    emergency_route: list[str] = []
    corridor_states: dict[str, str] = {}
    active_node: str | None = None

    if emergency_detected and emergency_mode:
        green_times = {
            "lane_1": 10,
            "lane_2": max(10, cycle_time - 30),
            "lane_3": 10,
            "lane_4": 10,
        }
        active_lane = "lane_2"
        emergency_route = compute_path("A", "D")
        if emergency_route:
            corridor_step = min((processed_frames - DEMO_EMERGENCY_TRIGGER_FRAME) // 20, len(emergency_route) - 1)
            corridor_states = _generate_demo_corridor(emergency_route, corridor_step)
            active_node = emergency_route[corridor_step]

    result["lane_counts"] = lane_counts
    result["densities"] = densities
    result["green_times"] = green_times
    result["active_lane"] = active_lane
    result["emergency"] = emergency_detected
    result["emergency_route"] = emergency_route
    result["corridor_states"] = corridor_states
    result["active_node"] = active_node
    result["static_metrics"] = simulate_static(lane_counts)
    result["ai_metrics"] = simulate_ai(green_times, lane_counts)
    return result


def _draw_status_overlay(
    frame: cv2.typing.MatLike,
    active_lane: str | None,
    emergency_mode: bool,
    corridor_active: bool,
    fps: float,
) -> cv2.typing.MatLike:
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (340, 122), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    lines = [
        f"Active Lane: {active_lane or 'None'}",
        f"Emergency Mode: {'ON' if emergency_mode else 'OFF'}",
        f"Corridor Active: {'YES' if corridor_active else 'NO'}",
        f"FPS: {fps:.1f}",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, 36 + (index * 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def _render_log_panel() -> None:
    st.subheader("Recent Events")
    if not st.session_state.event_log:
        st.write("No events yet")
        return

    for message in st.session_state.event_log:
        st.write(f"- {message}")


def _update_event_log(result: dict, corridor_active: bool) -> None:
    active_lane = result.get("active_lane")
    emergency_detected = bool(result.get("emergency"))

    if active_lane and active_lane != st.session_state.last_active_lane:
        _log_event(f"{active_lane.replace('_', ' ').title()} turned green")
        st.session_state.last_active_lane = active_lane

    if emergency_detected and not st.session_state.last_emergency:
        _log_event("Emergency detected")
    st.session_state.last_emergency = emergency_detected

    if corridor_active and not st.session_state.last_corridor:
        _log_event("Corridor activated")
    st.session_state.last_corridor = corridor_active


def main() -> None:
    st.set_page_config(page_title="FlowX Dashboard", layout="wide")
    st.title("FlowX Traffic Intelligence Dashboard")
    _init_session_state()

    with st.sidebar:
        st.header("Controls")
        speed = st.slider(
            "Simulation Speed",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
        )
        st.write(f"Current Speed: {speed}x")
        auto_run = st.toggle("Auto Run Simulation", value=True)
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
        demo_mode = st.toggle("Demo Scenario Mode", value=False)
        uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        start_processing = st.button("Start Processing", type="primary")
        reset_requested = st.button("Reset Playback")

    if reset_requested:
        _reset_demo()
        st.rerun()

    left_column, right_column = st.columns([1.6, 1.0])

    with left_column:
        video_placeholder = st.empty()
        loading_placeholder = st.empty()

    with right_column:
        state_decision_placeholder = st.empty()
        crash_panel_placeholder = st.empty()
        map_placeholder = st.empty()
        status_placeholder = st.empty()
        signal_placeholder = st.empty()
        metrics_placeholder = st.empty()
        graphs_placeholder = st.empty()
        corridor_status_placeholder = st.empty()
        corridor_placeholder = st.empty()
        comparison_placeholder = st.empty()
        log_placeholder = st.empty()

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
        with signal_placeholder.container():
            _render_signal_panel(st.session_state.signal_lane, int(st.session_state.timer))
        return

    if not auto_run:
        status_placeholder.warning("Simulation paused. Enable Auto Run Simulation to run updates.")
        with signal_placeholder.container():
            _render_signal_panel(st.session_state.signal_lane, int(st.session_state.timer))
        return

    capture: cv2.VideoCapture | None = None
    previous_frame_time = perf_counter()
    processed_frames = 0
    try:
        loading_placeholder.info("Loading model...")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            st.error(f"Unable to open video: {video_path}")
            return

        frame_index = 0
        while True:
            if not auto_run:
                status_placeholder.warning("Simulation paused. Enable Auto Run Simulation to resume updates.")
                break

            loop_start = perf_counter()
            success, frame = capture.read()
            if not success or frame is None:
                status_placeholder.success("Video processing completed.")
                break

            frame_index += 1
            if frame_index % frame_skip != 0:
                continue

            try:
                loading_placeholder.info("Processing video...")
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

            processed_frames += 1
            if demo_mode:
                result = _apply_demo_scenario(
                    result=result,
                    processed_frames=processed_frames,
                    cycle_time=cycle_time,
                    emergency_mode=emergency_mode,
                )

            current_time = perf_counter()
            fps = 1.0 / max(current_time - previous_frame_time, 1e-6)
            previous_frame_time = current_time
            corridor_active = emergency_mode and bool(result.get("emergency_route"))
            _update_event_log(result, corridor_active)
            _update_signal_timeline(result)
            signal_lane, signal_timer = _update_signal_animation(result, speed)
            intersections = _simulate_intersections(
                result=result,
                processed_frames=processed_frames,
                emergency_mode=emergency_mode,
            )

            display_frame = _draw_status_overlay(
                frame=result["frame"].copy(),
                active_lane=result.get("active_lane"),
                emergency_mode=emergency_mode,
                corridor_active=corridor_active,
                fps=fps,
            )
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            if emergency_mode and result["emergency"]:
                status_placeholder.error("🚑 Emergency Mode Activated")
                if corridor_active:
                    corridor_status_placeholder.warning("🚑 Emergency Corridor Active")
                else:
                    corridor_status_placeholder.info("Emergency detected. Corridor pending.")
            else:
                emergency_mode_status = "ON" if emergency_mode else "OFF"
                active_lane_text = result["active_lane"] or "None"
                status_placeholder.info(
                    f"Active lane: {active_lane_text} | Emergency mode: {emergency_mode_status} | Demo mode: {'ON' if demo_mode else 'OFF'}"
                )
                corridor_status_placeholder.empty()

            with state_decision_placeholder.container():
                _render_state_and_decision(result)

            with crash_panel_placeholder.container():
                _render_crash_panel(result)

            with map_placeholder.container():
                _render_map_view(intersections)

            with signal_placeholder.container():
                _render_signal_panel(signal_lane, signal_timer)

            with metrics_placeholder.container():
                density_map = _resolve_density_map(result)
                _render_metrics(
                    lane_counts=result["lane_counts"],
                    densities=density_map,
                    green_times=result["green_times"],
                    active_lane=result["active_lane"],
                    fps=fps,
                )

            with graphs_placeholder.container():
                _render_flow_graphs(result)

            with corridor_placeholder.container():
                _render_corridor(
                    route=result.get("emergency_route", []),
                    corridor_states=result.get("corridor_states", {}),
                    active_node=result.get("active_node"),
                )

            with comparison_placeholder.container():
                _render_comparison(
                    ai_metrics=result.get("ai_metrics", {}),
                    static_metrics=result.get("static_metrics", {}),
                )

            with log_placeholder.container():
                _render_log_panel()

            target_interval = 1.0 / 12.0
            elapsed = perf_counter() - loop_start
            if elapsed < target_interval:
                time.sleep((target_interval - elapsed) / speed)

        loading_placeholder.empty()
    finally:
        loading_placeholder.empty()
        _release_capture(capture)
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
