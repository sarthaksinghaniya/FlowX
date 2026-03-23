from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import cv2

from config.lanes import FRAME_HEIGHT, FRAME_WIDTH, LANE_POLYGONS
from src.density import DensityCalculator
from src.emergency_corridor import compute_path, generate_corridor
from src.inference import YOLOInferenceEngine
from src.lane_mapper import LaneMapper
from src.signal_control import SignalController
from src.utils import annotate_frame


_DETECTOR_CACHE: dict[str, YOLOInferenceEngine] = {}
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
        }

    model_path = str(_get_required(config, "model_path", Path("models/traffic_detector.pt")))
    confidence_threshold = float(_get_required(config, "confidence_threshold", 0.5))
    cycle_time = int(_get_required(config, "cycle_time", 120))
    emergency_mode_enabled = bool(_get_required(config, "emergency_mode", False))
    minimum_green = int(_get_required(config, "minimum_green", 10))
    emergency_start = str(_get_required(config, "emergency_start", "A"))
    emergency_destination = str(_get_required(config, "emergency_destination", "D"))

    resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    detector = _get_detector(model_path, confidence_threshold)
    detections = detector.infer(resized_frame)

    lane_counts, lane_detections = _LANE_MAPPER.map_detections(detections)
    densities = _DENSITY_CALCULATOR.compute(lane_counts)
    signal_controller = SignalController(cycle_time=cycle_time, minimum_green=minimum_green)
    green_times = signal_controller.allocate_green_times(densities)

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
        emergency_route = compute_path(emergency_start, emergency_destination)
        corridor_states = generate_corridor(emergency_route)
        if emergency_route and corridor_states:
            active_node = emergency_route[0]

    active_lane = max(green_times, key=green_times.get) if green_times else None

    processed_frame = annotate_frame(
        frame=resized_frame.copy(),
        detections=detections,
        lane_polygons=_LANE_MAPPER.lane_polygons,
        lane_counts=lane_counts,
        lane_densities=densities,
        green_times=green_times,
    )
    processed_frame = _highlight_active_lane(processed_frame, active_lane)

    if emergency_mode_enabled and emergency_detected:
        processed_frame = _draw_emergency_banner(processed_frame)

    return {
        "frame": processed_frame,
        "lane_counts": lane_counts,
        "densities": densities,
        "green_times": green_times,
        "active_lane": active_lane,
        "emergency": emergency_detected,
        "emergency_route": emergency_route,
        "corridor_states": corridor_states,
        "active_node": active_node,
    }
