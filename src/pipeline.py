from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import cv2

from config.lanes import FRAME_HEIGHT, FRAME_WIDTH, LANE_POLYGONS
from src.crash_inference import predict_crash
from src.comparison import simulate_ai, simulate_static
from src.density import DensityCalculator
from src.emergency_corridor import compute_path, generate_corridor
from src.inference import YOLOInferenceEngine
from src.lane_mapper import LaneMapper
from src.signal_control import SignalController
from src.utils import annotate_frame


_DETECTOR_CACHE: dict[str, YOLOInferenceEngine] = {}
_CRASH_PREDICTION_CACHE: dict[str, dict[str, Any] | None] = {}
_LANE_MAPPER = LaneMapper(LANE_POLYGONS)
_DENSITY_CALCULATOR = DensityCalculator()


def _get_required(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config else default


def _get_detector(model_path: str, confidence_threshold: float) -> YOLOInferenceEngine:
    resolved_model_path = str(Path(model_path).resolve())
    detector = _DETECTOR_CACHE.get(resolved_model_path)
    if detector is None:
        detector = YOLOInferenceEngine(resolved_model_path, confidence_threshold=confidence_threshold)
        _DETECTOR_CACHE[resolved_model_path] = detector
    detector.confidence_threshold = confidence_threshold
    return detector


def _find_emergency_lane(
    lane_detections: Mapping[str, list[dict]],
    confidence_threshold: float,
) -> str | None:
    best_lane: str | None = None
    best_confidence = -1.0

    for lane_name, detections in lane_detections.items():
        for detection in detections:
            if detection["class"] == 5 and detection["confidence"] > confidence_threshold:
                if detection["confidence"] > best_confidence:
                    best_confidence = detection["confidence"]
                    best_lane = lane_name

    return best_lane


def _prioritize_emergency_lane(
    green_times: dict[str, int],
    emergency_lane: str,
    cycle_time: int,
    minimum_green: int,
) -> dict[str, int]:
    prioritized = {lane_name: minimum_green for lane_name in green_times}
    reserved = minimum_green * max(len(prioritized) - 1, 0)
    prioritized[emergency_lane] = max(minimum_green, cycle_time - reserved)
    return prioritized


def _infer_crash_once(video_path: str | None) -> dict[str, Any] | None:
    if not video_path:
        return None
    if video_path not in _CRASH_PREDICTION_CACHE:
        _CRASH_PREDICTION_CACHE[video_path] = predict_crash(video_path)
    return _CRASH_PREDICTION_CACHE[video_path]


def _lane_name_to_index(lane_name: str | None) -> int:
    if not lane_name:
        return -1
    parts = lane_name.split("_")
    if len(parts) != 2:
        return -1
    try:
        return int(parts[1])
    except ValueError:
        return -1


def _resolve_crash_weight(crash_class: str | None) -> float:
    if crash_class == "major":
        return 1.0
    if crash_class == "moderate":
        return 0.6
    if crash_class == "minor":
        return 0.3
    return 0.0


def _resolve_crash_lane(lane_counts: Mapping[str, int], configured_lane: str | None) -> str | None:
    if configured_lane and configured_lane in lane_counts:
        return configured_lane
    if not lane_counts:
        return None
    return max(lane_counts, key=lane_counts.get)


def _increase_lane_green(
    green_times: dict[str, int],
    lane_name: str,
    increase_ratio: float,
    cycle_time: int,
    minimum_green: int,
) -> dict[str, int]:
    if lane_name not in green_times:
        return green_times

    updated = green_times.copy()
    extra_seconds = max(1, int(round(updated[lane_name] * increase_ratio)))
    updated[lane_name] += extra_seconds

    total_allocated = sum(updated.values())
    if total_allocated <= cycle_time:
        return updated

    excess = total_allocated - cycle_time
    other_lanes = [name for name in updated if name != lane_name]
    while excess > 0 and other_lanes:
        progress = False
        for other_lane in other_lanes:
            if excess <= 0:
                break
            if updated[other_lane] > minimum_green:
                updated[other_lane] -= 1
                excess -= 1
                progress = True
        if not progress:
            break
    return updated


def _highlight_active_lane(
    frame: cv2.typing.MatLike,
    lane_name: str | None,
) -> cv2.typing.MatLike:
    if not lane_name:
        return frame

    polygon = _LANE_MAPPER.lane_polygons.get(lane_name)
    if polygon is None:
        return frame

    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color=(0, 180, 0))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=4)
    cv2.putText(
        frame,
        f"Active Lane: {lane_name}",
        (14, FRAME_HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def _draw_emergency_banner(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 46), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    cv2.putText(
        frame,
        "EMERGENCY MODE ACTIVATED",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def process_frame(frame: cv2.typing.MatLike, config: Mapping[str, Any]) -> Dict[str, Any]:
    if frame is None or frame.size == 0:
        return {
            "frame": frame,
            "lane_counts": {lane_name: 0 for lane_name in LANE_POLYGONS},
            "densities": {lane_name: 0.0 for lane_name in LANE_POLYGONS},
            "green_times": {lane_name: 10 for lane_name in LANE_POLYGONS},
            "active_lane": None,
            "emergency": False,
            "emergency_route": [],
            "corridor_states": {},
            "active_node": None,
            "lane_priorities": [],
            "selected_lane": -1,
            "reason": "No frame data",
            "crash_class": None,
            "crash_confidence": 0.0,
            "crash_probs": [],
            "static_metrics": {
                "per_lane": {"wait_time": {}, "queue_length": {}, "throughput": {}},
                "summary": {"wait_time": 0.0, "queue_length": 0.0, "throughput": 0.0},
            },
            "ai_metrics": {
                "per_lane": {"wait_time": {}, "queue_length": {}, "throughput": {}},
                "summary": {"wait_time": 0.0, "queue_length": 0.0, "throughput": 0.0},
            },
        }

    model_path = str(_get_required(config, "model_path", Path("models/traffic_detector.pt")))
    confidence_threshold = float(_get_required(config, "confidence_threshold", 0.5))
    cycle_time = int(_get_required(config, "cycle_time", 120))
    emergency_mode_enabled = bool(_get_required(config, "emergency_mode", False))
    minimum_green = int(_get_required(config, "minimum_green", 10))
    emergency_start = str(_get_required(config, "emergency_start", "A"))
    emergency_destination = str(_get_required(config, "emergency_destination", "D"))
    crash_video_path = _get_required(config, "crash_video_path", None)
    configured_crash_lane = _get_required(config, "crash_lane", None)
    crash_lambda = float(_get_required(config, "crash_lambda", 0.5))

    resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    detector = _get_detector(model_path, confidence_threshold)
    detections = detector.infer(resized_frame)

    lane_counts, lane_detections = _LANE_MAPPER.map_detections(detections)
    queue_lengths = {lane_name: vehicle_count * 0.5 for lane_name, vehicle_count in lane_counts.items()}
    densities = _DENSITY_CALCULATOR.compute(lane_counts)
    effective_densities = densities.copy()

    crash_prediction = _infer_crash_once(crash_video_path)
    crash_class = crash_prediction["class"] if crash_prediction else None
    crash_confidence = float(crash_prediction["confidence"]) if crash_prediction else 0.0
    crash_probs = crash_prediction["probs"] if crash_prediction else []
    crash_weight = _resolve_crash_weight(crash_class)
    crash_lane = _resolve_crash_lane(lane_counts, configured_crash_lane)
    if crash_weight > 0.0 and crash_lane:
        effective_densities[crash_lane] = effective_densities.get(crash_lane, 0.0) + (crash_lambda * crash_weight)

    signal_controller = SignalController(cycle_time=cycle_time, minimum_green=minimum_green)
    green_times = signal_controller.allocate_green_times(effective_densities)
    decision_reason = "Normal density logic"

    if crash_class == "major" and crash_lane:
        green_times = _prioritize_emergency_lane(
            green_times=green_times,
            emergency_lane=crash_lane,
            cycle_time=cycle_time,
            minimum_green=minimum_green,
        )
        decision_reason = "Crash detected: major"
    elif crash_class == "moderate" and crash_lane:
        green_times = _increase_lane_green(
            green_times=green_times,
            lane_name=crash_lane,
            increase_ratio=0.40,
            cycle_time=cycle_time,
            minimum_green=minimum_green,
        )
        decision_reason = "Crash detected: moderate"
    elif crash_class == "minor" and crash_lane:
        green_times = _increase_lane_green(
            green_times=green_times,
            lane_name=crash_lane,
            increase_ratio=0.15,
            cycle_time=cycle_time,
            minimum_green=minimum_green,
        )
        decision_reason = "Crash detected: minor"

    emergency_lane = _find_emergency_lane(lane_detections, confidence_threshold)
    emergency_detected = emergency_lane is not None
    emergency_route: list[str] = []
    corridor_states: dict[str, str] = {}
    active_node: str | None = None

    if emergency_mode_enabled and emergency_detected:
        green_times = _prioritize_emergency_lane(
            green_times=green_times,
            emergency_lane=emergency_lane,
            cycle_time=cycle_time,
            minimum_green=minimum_green,
        )
        decision_reason = f"Emergency vehicle override: {emergency_lane}"
        emergency_route = compute_path(emergency_start, emergency_destination)
        corridor_states = generate_corridor(emergency_route)
        if emergency_route and corridor_states:
            active_node = emergency_route[0]

    active_lane = max(green_times, key=green_times.get) if green_times else None
    lane_priorities = sorted(green_times, key=green_times.get, reverse=True)
    selected_lane = _lane_name_to_index(active_lane)

    print("AI Decision:")
    print(f"  density values: {effective_densities}")
    print(f"  crash class: {crash_class}")
    print(f"  selected lane: {selected_lane}")
    if crash_class == "major" and crash_lane:
        print(f"Crash detected: major")
        print(f"Lane {_lane_name_to_index(crash_lane)} prioritized due to accident")

    processed_frame = annotate_frame(
        frame=resized_frame.copy(),
        detections=detections,
        lane_polygons=_LANE_MAPPER.lane_polygons,
        lane_counts=lane_counts,
        lane_densities=effective_densities,
        green_times=green_times,
    )
    processed_frame = _highlight_active_lane(processed_frame, active_lane)

    if emergency_mode_enabled and emergency_detected:
        processed_frame = _draw_emergency_banner(processed_frame)

    static_metrics = simulate_static(lane_counts)
    ai_metrics = simulate_ai(green_times, lane_counts)

    return {
        "frame": processed_frame,
        "lane_counts": lane_counts,
        "densities": effective_densities,
        "queue_lengths": queue_lengths,
        "green_times": green_times,
        "active_lane": active_lane,
        "lane_priorities": lane_priorities,
        "selected_lane": selected_lane,
        "reason": decision_reason,
        "crash_class": crash_class,
        "crash_confidence": crash_confidence,
        "crash_probs": crash_probs,
        "emergency": emergency_detected,
        "emergency_route": emergency_route,
        "corridor_states": corridor_states,
        "active_node": active_node,
        "static_metrics": static_metrics,
        "ai_metrics": ai_metrics,
    }
