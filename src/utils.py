from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


CLASS_COLORS = {
    "default": (0, 255, 255),
    "lane": (255, 180, 0),
    "text_bg": (30, 30, 30),
    "text_fg": (255, 255, 255),
}


def draw_lane_polygons(
    frame: cv2.typing.MatLike, lane_polygons: Dict[str, np.ndarray]
) -> cv2.typing.MatLike:
    for lane_name, polygon in lane_polygons.items():
        cv2.polylines(frame, [polygon], isClosed=True, color=CLASS_COLORS["lane"], thickness=2)
        anchor_x, anchor_y = polygon[0]
        cv2.putText(
            frame,
            lane_name,
            (int(anchor_x), int(anchor_y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            CLASS_COLORS["lane"],
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_bounding_boxes(
    frame: cv2.typing.MatLike, detections: Iterable[dict]
) -> cv2.typing.MatLike:
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        centroid = detection["centroid"]
        label = f"id:{detection['class']} {detection['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), CLASS_COLORS["default"], 2)
        cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            CLASS_COLORS["default"],
            2,
            cv2.LINE_AA,
        )
    return frame


def overlay_lane_stats(
    frame: cv2.typing.MatLike,
    lane_counts: Dict[str, int],
    lane_densities: Dict[str, float],
    green_times: Dict[str, int],
) -> cv2.typing.MatLike:
    lines: List[str] = []
    for lane_name in lane_counts:
        lines.append(
            f"{lane_name}: count={lane_counts[lane_name]} "
            f"density={lane_densities.get(lane_name, 0.0):.2f} "
            f"green={green_times.get(lane_name, 0)}s"
        )

    if not lines:
        lines.append("No lane data available")

    start_x = 10
    start_y = 25
    line_height = 24
    box_width = 420
    box_height = 12 + (len(lines) * line_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_width, 5 + box_height), CLASS_COLORS["text_bg"], -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (start_x, start_y + (index * line_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            CLASS_COLORS["text_fg"],
            2,
            cv2.LINE_AA,
        )

    return frame


def annotate_frame(
    frame: cv2.typing.MatLike,
    detections: Iterable[dict],
    lane_polygons: Dict[str, np.ndarray],
    lane_counts: Dict[str, int],
    lane_densities: Dict[str, float],
    green_times: Dict[str, int],
) -> cv2.typing.MatLike:
    draw_lane_polygons(frame, lane_polygons)
    draw_bounding_boxes(frame, detections)
    overlay_lane_stats(frame, lane_counts, lane_densities, green_times)
    return frame
