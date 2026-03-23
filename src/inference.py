from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
from ultralytics import YOLO


Detection = Dict[str, Any]


class YOLOInferenceEngine:
    def __init__(self, model_path: str, confidence_threshold: float = 0.25) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(str(self.model_path))
        self.logger.info("Loaded YOLO model from %s", self.model_path)

    def infer(self, frame: cv2.typing.MatLike) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
        )
        return self._parse_results(results)

    def _parse_results(self, results: List[Any]) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(value) for value in xyxy]
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)

                detections.append(
                    {
                        "class": class_id,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "centroid": (centroid_x, centroid_y),
                    }
                )

        return detections
