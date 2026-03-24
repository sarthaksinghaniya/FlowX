from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import cv2

from config.lanes import FRAME_HEIGHT, FRAME_WIDTH, LANE_POLYGONS
from src.comparison import simulate_ai, simulate_static
from src.crash_inference import predict_crash
from src.emergency_corridor import compute_path, generate_corridor
from src.inference import YOLOInferenceEngine
from src.lane_mapper import LaneMapper
from src.utils import annotate_frame


_DETECTOR_CACHE: dict[str, YOLOInferenceEngine] = {}
_CRASH_PREDICTION_CACHE: dict[str, dict[str, Any] | None] = {}
_LANE_MAPPER = LaneMapper(LANE_POLYGONS)
_LANE_WAIT_TIMES: dict[str, float] = {lane_name: 0.0 for lane_name in LANE_POLYGONS}
_STARVATION_AGE: dict[str, int] = {lane_name: 0 for lane_name in LANE_POLYGONS}

_STATE_NORMAL = "NORMAL"
_STATE_CRASH = "CRASH_MODE"
_STATE_EMERGENCY = "EMERGENCY_MODE"


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


def _infer_crash_once(video_path: str | None) -> dict[str, Any] | None:
    if not video_path:
        return None
    if video_path not in _CRASH_PREDICTION_CACHE:
        _CRASH_PREDICTION_CACHE[video_path] = predict_crash(video_path)
    return _CRASH_PREDICTION_CACHE[video_path]


def _lane_name_to_index(lane_name: str | None) -> int:
    if not lane_name:
        return -1
    try:
        return int(lane_name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _lane_index_to_name(lane_index: int) -> str | None:
    lane_name = f"lane_{lane_index}"
    return lane_name if lane_name in LANE_POLYGONS else None


def _crash_weight(crash_class: str | None) -> float:
    if crash_class == "major":
        return 1.0
    if crash_class == "moderate":
        return 0.6
    if crash_class == "minor":
        return 0.3
    return 0.0


def _find_emergency_lane_from_detections(
    lane_detections: Mapping[str, list[dict]],
    confidence_threshold: float,
) -> str | None:
    best_lane: str | None = None
    best_conf = -1.0
    for lane_name, detections in lane_detections.items():
        for det in detections:
            # Class 5 corresponds to emergency vehicle in current detector mapping.
            if det.get("class") == 5 and float(det.get("confidence", 0.0)) > confidence_threshold:
                if float(det["confidence"]) > best_conf:
                    best_conf = float(det["confidence"])
                    best_lane = lane_name
    return best_lane


def compute_density(
    lane_counts: Mapping[str, int],
    queue_lengths: Mapping[str, float],
    wait_times: Mapping[str, float],
    alpha: float,
    beta: float,
    gamma: float,
    capacity: float,
) -> dict[str, float]:
    densities: dict[str, float] = {}
    lane_names = set(lane_counts.keys()) | set(queue_lengths.keys()) | set(wait_times.keys())
    cap = max(capacity, 1e-6)
    for lane_name in lane_names:
        vi = float(lane_counts.get(lane_name, 0))
        qi = float(queue_lengths.get(lane_name, 0.0))
        ti = float(wait_times.get(lane_name, 0.0))
        densities[lane_name] = ((alpha * vi) + (beta * qi) + (gamma * ti)) / cap
    return densities


def apply_crash_adjustment(
    base_densities: Mapping[str, float],
    crash_class: str | None,
    crash_lane: str | None,
    crash_lambda: float = 0.5,
) -> tuple[dict[str, float], float]:
    adjusted = dict(base_densities)
    weight = _crash_weight(crash_class)
    if weight > 0.0 and crash_lane and crash_lane in adjusted:
        adjusted[crash_lane] = adjusted[crash_lane] + (crash_lambda * weight)
    return adjusted, weight


def apply_emergency_override(
    lane_names: list[str],
    emergency_lane: str,
    cycle_time: float,
) -> dict[str, float]:
    priorities = {lane_name: 0.0 for lane_name in lane_names}
    priorities[emergency_lane] = max(cycle_time, 1.0)
    return priorities


def _normalize_allocations_to_cycle(
    allocations: dict[str, float],
    cycle_time: float,
    minimum_green: float,
) -> dict[str, float]:
    if not allocations:
        return allocations

    lane_names = list(allocations.keys())
    min_green = max(minimum_green, 0.0)
    if cycle_time <= 0:
        return {lane_name: 0.0 for lane_name in lane_names}

    if cycle_time < min_green * len(lane_names):
        fallback = cycle_time / len(lane_names)
        return {lane_name: fallback for lane_name in lane_names}

    total = sum(allocations.values())
    if total <= 0:
        return {lane_name: min_green for lane_name in lane_names}

    normalized = {lane_name: (allocations[lane_name] / total) * cycle_time for lane_name in lane_names}
    for lane_name in lane_names:
        normalized[lane_name] = max(min_green, normalized[lane_name])

    adjusted_total = sum(normalized.values())
    if adjusted_total == cycle_time:
        return normalized

    if adjusted_total < cycle_time:
        remaining = cycle_time - adjusted_total
        sorted_lanes = sorted(normalized, key=normalized.get, reverse=True)
        idx = 0
        while remaining > 1e-9:
            lane_name = sorted_lanes[idx % len(sorted_lanes)]
            step = min(1.0, remaining)
            normalized[lane_name] += step
            remaining -= step
            idx += 1
        return normalized

    excess = adjusted_total - cycle_time
    sorted_lanes = sorted(normalized, key=normalized.get)
    while excess > 1e-9:
        progress = False
        for lane_name in sorted_lanes:
            if excess <= 1e-9:
                break
            removable = normalized[lane_name] - min_green
            if removable > 0:
                step = min(1.0, removable, excess)
                normalized[lane_name] -= step
                excess -= step
                progress = True
        if not progress:
            break
    return normalized


def decide_signal(
    densities: Mapping[str, float],
    crash_class: str | None,
    crash_lane: str | None,
    emergency_detected: bool,
    emergency_lane: str | None,
    cycle_time: float,
    minimum_green: float,
    starvation_threshold: int,
    starvation_boost: float,
) -> dict[str, Any]:
    lane_names = sorted(densities.keys())
    base_priorities = {lane_name: float(densities[lane_name]) for lane_name in lane_names}
    crash_factor = {lane_name: 0.0 for lane_name in lane_names}
    emergency_factor = {lane_name: 0.0 for lane_name in lane_names}
    state = _STATE_NORMAL
    reason = "Normal density logic"

    if emergency_detected and emergency_lane and emergency_lane in lane_names:
        state = _STATE_EMERGENCY
        reason = "Emergency override active"
        priorities = apply_emergency_override(lane_names, emergency_lane, cycle_time)
        green_times = {lane_name: 0.0 for lane_name in lane_names}
        green_times[emergency_lane] = max(float(cycle_time), float(minimum_green))
        selected_lane = emergency_lane
        return {
            "state": state,
            "reason": reason,
            "priorities": priorities,
            "green_times": green_times,
            "selected_lane": selected_lane,
        }

    if crash_class and crash_lane and crash_lane in lane_names:
        state = _STATE_CRASH
        reason = f"Crash detected: {crash_class}"
        crash_factor[crash_lane] = 0.5 * _crash_weight(crash_class)

    # Anti-starvation boost.
    for lane_name in lane_names:
        age = _STARVATION_AGE.get(lane_name, 0)
        base_priorities[lane_name] += min(float(age) * starvation_boost, 1.0)

    priorities = {
        lane_name: base_priorities[lane_name] + crash_factor[lane_name] + emergency_factor[lane_name]
        for lane_name in lane_names
    }

    forced_lane: str | None = None
    for lane_name in lane_names:
        if _STARVATION_AGE.get(lane_name, 0) >= starvation_threshold:
            forced_lane = lane_name
            break

    if forced_lane:
        priorities[forced_lane] = max(priorities[forced_lane], max(priorities.values(), default=0.0) + 1.0)
        reason = f"Anti-starvation applied: {forced_lane}"

    selected_lane = max(priorities, key=priorities.get) if priorities else None
    green_times = _normalize_allocations_to_cycle(dict(priorities), cycle_time=float(cycle_time), minimum_green=float(minimum_green))

    if selected_lane and selected_lane in green_times:
        _STARVATION_AGE[selected_lane] = 0
        for lane_name in lane_names:
            if lane_name != selected_lane:
                _STARVATION_AGE[lane_name] = _STARVATION_AGE.get(lane_name, 0) + 1

    return {
        "state": state,
        "reason": reason,
        "priorities": priorities,
        "green_times": green_times,
        "selected_lane": selected_lane,
    }


def _highlight_active_lane(frame: cv2.typing.MatLike, lane_name: str | None) -> cv2.typing.MatLike:
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


def run_pipeline(frame: cv2.typing.MatLike, config: Mapping[str, Any]) -> Dict[str, Any]:
    if frame is None or frame.size == 0:
        return {
            "selected_lane": -1,
            "green_time": 0.0,
            "state": _STATE_NORMAL,
            "reason": "No frame data",
            "densities": [],
            "priorities": [],
            "crash_class": None,
            "emergency": False,
        }

    # Config.
    model_path = str(_get_required(config, "model_path", Path("models/traffic_detector.pt")))
    confidence_threshold = float(_get_required(config, "confidence_threshold", 0.5))
    cycle_time = float(_get_required(config, "cycle_time", 120))
    minimum_green = float(_get_required(config, "minimum_green", 10))

    alpha = float(_get_required(config, "alpha", 0.7))
    beta = float(_get_required(config, "beta", 0.3))
    gamma = float(_get_required(config, "gamma", 0.1))
    capacity = float(_get_required(config, "capacity", 30))
    crash_lambda = float(_get_required(config, "lambda", 0.5))

    emergency_mode_enabled = bool(_get_required(config, "emergency_mode", False))
    emergency_flag = bool(_get_required(config, "emergency_flag", False))
    emergency_lane_idx = int(_get_required(config, "emergency_lane", -1))
    configured_crash_lane_idx = int(_get_required(config, "crash_lane", -1))

    starvation_threshold = int(_get_required(config, "starvation_threshold", 30))
    starvation_boost = float(_get_required(config, "starvation_boost", 0.03))
    frame_interval_sec = float(_get_required(config, "frame_interval_sec", 1.0))

    crash_video_path = _get_required(config, "crash_video_path", None)
    emergency_start = str(_get_required(config, "emergency_start", "A"))
    emergency_destination = str(_get_required(config, "emergency_destination", "D"))

    resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    detector = _get_detector(model_path, confidence_threshold)
    detections = detector.infer(resized_frame)

    # Input extraction.
    lane_counts, lane_detections = _LANE_MAPPER.map_detections(detections)  # Vi
    queue_lengths = {lane_name: float(count) * 0.5 for lane_name, count in lane_counts.items()}  # Qi

    # Update Ti wait-times (sec).
    for lane_name in lane_counts:
        _LANE_WAIT_TIMES.setdefault(lane_name, 0.0)
        _STARVATION_AGE.setdefault(lane_name, 0)

    crash_prediction = _infer_crash_once(crash_video_path)
    crash_class = crash_prediction["class"] if crash_prediction else None
    crash_confidence = float(crash_prediction["confidence"]) if crash_prediction else 0.0
    crash_probs = crash_prediction["probs"] if crash_prediction else []

    configured_crash_lane = _lane_index_to_name(configured_crash_lane_idx)
    crash_lane = configured_crash_lane if configured_crash_lane in lane_counts else None
    if crash_class and not crash_lane and lane_counts:
        crash_lane = max(lane_counts, key=lane_counts.get)

    densities = compute_density(
        lane_counts=lane_counts,
        queue_lengths=queue_lengths,
        wait_times=_LANE_WAIT_TIMES,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        capacity=capacity,
    )
    effective_densities, _ = apply_crash_adjustment(
        base_densities=densities,
        crash_class=crash_class,
        crash_lane=crash_lane,
        crash_lambda=crash_lambda,
    )

    emergency_lane_from_detection = _find_emergency_lane_from_detections(lane_detections, confidence_threshold)
    emergency_lane_from_config = _lane_index_to_name(emergency_lane_idx)
    emergency_lane = emergency_lane_from_config or emergency_lane_from_detection
    emergency_detected = (emergency_mode_enabled and emergency_lane is not None) or emergency_flag
    if emergency_flag and emergency_lane is None:
        emergency_lane = max(lane_counts, key=lane_counts.get) if lane_counts else None

    decision = decide_signal(
        densities=effective_densities,
        crash_class=crash_class,
        crash_lane=crash_lane,
        emergency_detected=emergency_detected,
        emergency_lane=emergency_lane,
        cycle_time=cycle_time,
        minimum_green=minimum_green,
        starvation_threshold=starvation_threshold,
        starvation_boost=starvation_boost,
    )

    selected_lane_name = decision["selected_lane"]
    selected_lane = _lane_name_to_index(selected_lane_name)
    green_times = decision["green_times"]
    selected_green = float(green_times.get(selected_lane_name, 0.0)) if selected_lane_name else 0.0

    # Update wait times after signal decision.
    for lane_name in _LANE_WAIT_TIMES:
        if lane_name == selected_lane_name:
            _LANE_WAIT_TIMES[lane_name] = 0.0
        else:
            _LANE_WAIT_TIMES[lane_name] = _LANE_WAIT_TIMES.get(lane_name, 0.0) + frame_interval_sec

    emergency_route: list[str] = []
    corridor_states: dict[str, str] = {}
    active_node: str | None = None
    if emergency_detected and emergency_lane:
        emergency_route = compute_path(emergency_start, emergency_destination)
        corridor_states = generate_corridor(emergency_route)
        if emergency_route and corridor_states:
            active_node = emergency_route[0]

    # Logging.
    print("AI Decision Summary:")
    print(f"  densities: {effective_densities}")
    print(f"  crash: {crash_class}")
    print(f"  emergency: {emergency_detected}")
    print(f"  selected lane: {selected_lane}")
    print(f"  reason: {decision['reason']}")

    processed_frame = annotate_frame(
        frame=resized_frame.copy(),
        detections=detections,
        lane_polygons=_LANE_MAPPER.lane_polygons,
        lane_counts=lane_counts,
        lane_densities=effective_densities,
        green_times={lane: int(round(val)) for lane, val in green_times.items()},
    )
    processed_frame = _highlight_active_lane(processed_frame, selected_lane_name)
    if emergency_detected:
        processed_frame = _draw_emergency_banner(processed_frame)

    static_metrics = simulate_static(lane_counts)
    ai_metrics = simulate_ai({lane: int(round(val)) for lane, val in green_times.items()}, lane_counts)

    lane_order = sorted(effective_densities.keys())
    densities_list = [float(effective_densities[lane]) for lane in lane_order]
    priorities_list = [float(decision["priorities"].get(lane, 0.0)) for lane in lane_order]

    result = {
        # Requested final output contract.
        "selected_lane": selected_lane,
        "green_time": selected_green,
        "state": decision["state"],
        "reason": decision["reason"],
        "densities": densities_list,
        "priorities": priorities_list,
        "crash_class": crash_class,
        "emergency": emergency_detected,
        # Useful details.
        "crash_confidence": crash_confidence,
        "crash_probs": crash_probs,
        "lane_order": lane_order,
        "lane_priorities": sorted(decision["priorities"], key=decision["priorities"].get, reverse=True),
        # Backward-compatible fields used by app.
        "frame": processed_frame,
        "lane_counts": lane_counts,
        "queue_lengths": queue_lengths,
        "green_times": {lane: int(round(val)) for lane, val in green_times.items()},
        "active_lane": selected_lane_name,
        "emergency_route": emergency_route,
        "corridor_states": corridor_states,
        "active_node": active_node,
        "static_metrics": static_metrics,
        "ai_metrics": ai_metrics,
    }
    return result


def process_frame(frame: cv2.typing.MatLike, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Backward-compatible wrapper for existing app integrations."""
    return run_pipeline(frame, config)

