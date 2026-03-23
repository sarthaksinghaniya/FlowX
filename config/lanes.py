from __future__ import annotations

from typing import Dict, List, Tuple

Point = Tuple[int, int]
Polygon = List[Point]


LANE_POLYGONS: Dict[str, Polygon] = {
    "lane_1": [(0, 300), (160, 220), (220, 479), (0, 479)],
    "lane_2": [(160, 220), (300, 180), (360, 479), (220, 479)],
    "lane_3": [(300, 180), (460, 220), (420, 479), (360, 479)],
    "lane_4": [(460, 220), (639, 300), (639, 479), (420, 479)],
}


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
