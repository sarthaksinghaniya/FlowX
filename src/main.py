from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.lanes import FRAME_HEIGHT, FRAME_WIDTH, LANE_POLYGONS
from src.density import DensityCalculator
from src.inference import YOLOInferenceEngine
from src.lane_mapper import LaneMapper
from src.signal_control import SignalController
from src.utils import annotate_frame


LOGGER = logging.getLogger("flowx")


class FlowXPipeline:
    def __init__(self, model_path: Path, video_path: Path) -> None:
        self.model_path = model_path
        self.video_path = video_path
        self.inference_engine = YOLOInferenceEngine(str(model_path))
        self.lane_mapper = LaneMapper(LANE_POLYGONS)
        self.density_calculator = DensityCalculator()
        self.signal_controller = SignalController()

    def run(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        window_name = "FlowX Traffic Intelligence"
        try:
            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    LOGGER.info("Reached end of video stream or received empty frame.")
                    break

                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                detections = self.inference_engine.infer(frame)
                lane_counts, _lane_detections = self.lane_mapper.map_detections(detections)
                lane_densities = self.density_calculator.compute(lane_counts)
                green_times = self.signal_controller.allocate_green_times(lane_densities)

                annotated_frame = annotate_frame(
                    frame=frame.copy(),
                    detections=detections,
                    lane_polygons=self.lane_mapper.lane_polygons,
                    lane_counts=lane_counts,
                    lane_densities=lane_densities,
                    green_times=green_times,
                )

                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    LOGGER.info("ESC pressed. Exiting FlowX pipeline.")
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    pipeline = FlowXPipeline(
        model_path=ROOT_DIR / "models" / "traffic_detector.pt",
        video_path=ROOT_DIR / "demo" / "test_video.mp4",
    )
    pipeline.run()


if __name__ == "__main__":
    main()
