from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


class LaneMapper:
    def __init__(self, lane_polygons: Dict[str, List[Tuple[int, int]]]) -> None:
        self._lane_polygons = {
            lane_name: np.array(points, dtype=np.int32)
            for lane_name, points in lane_polygons.items()
        }

    @property
    def lane_polygons(self) -> Dict[str, np.ndarray]:
        return self._lane_polygons

    def map_detections(
        self, detections: Iterable[dict]
    ) -> tuple[Dict[str, int], Dict[str, List[dict]]]:
        lane_counts = {lane_name: 0 for lane_name in self._lane_polygons}
        lane_detections = {lane_name: [] for lane_name in self._lane_polygons}

        for detection in detections:
            centroid = detection.get("centroid")
            if centroid is None:
                continue

            lane_name = self._find_lane_for_point(centroid)
            if lane_name is None:
                continue

            lane_counts[lane_name] += 1
            lane_detections[lane_name].append(detection)

        return lane_counts, lane_detections

    def _find_lane_for_point(self, point: Tuple[int, int]) -> str | None:
        for lane_name, polygon in self._lane_polygons.items():
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                return lane_name
        return None
