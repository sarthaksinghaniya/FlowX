from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _safe_import(module_name: str, fallback_module: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception:
        if fallback_module:
            try:
                return importlib.import_module(fallback_module)
            except Exception:
                print(f"Module failed: {module_name}")
                raise
        print(f"Module failed: {module_name}")
        raise


def run_pipeline(video_path: str | Path, config: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    pipeline_module = _safe_import("src.pipeline")
    crash_module = _safe_import("src.crash_inference")
    density_module = _safe_import("src.density")
    signal_module = _safe_import("src.signal", fallback_module="src.signal_control")
    emergency_module = _safe_import("src.emergency", fallback_module="src.emergency_corridor")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Module failed: video_io (unable to open {video_path})")

    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("Module failed: video_io (unable to read frame)")

    result = pipeline_module.run_pipeline(frame, config)

    if verbose:
        print("Verbose:")
        try:
            detector = pipeline_module._get_detector(  # type: ignore[attr-defined]
                str(config.get("model_path", "models/traffic_detector.pt")),
                float(config.get("confidence_threshold", 0.5)),
            )
            detections = detector.infer(cv2.resize(frame, (pipeline_module.FRAME_WIDTH, pipeline_module.FRAME_HEIGHT)))
            print("detections:", len(detections))
        except Exception:
            print("detections: <not available>")

        print("densities:", result.get("densities"))
        print(
            "crash output:",
            {
                "class": result.get("crash_class"),
                "confidence": result.get("crash_confidence"),
                "probs": result.get("crash_probs"),
            },
        )
        print("modules:", pipeline_module.__name__, crash_module.__name__, density_module.__name__, signal_module.__name__, emergency_module.__name__)

    return result


def validate_result(result: dict[str, Any]) -> None:
    assert "selected_lane" in result
    assert "green_time" in result
    assert "state" in result

    assert isinstance(result["selected_lane"], int)
    assert float(result["green_time"]) > 0
    assert result["state"] in ["NORMAL", "CRASH_MODE", "EMERGENCY_MODE"]


def main(verbose: bool = True) -> None:
    video_path = "demo/test_video.mp4"
    config = {
        "cycle_time": 120,
        "alpha": 0.5,
        "beta": 0.4,
        "gamma": 0.1,
        "lambda": 0.5,
        "minimum_green": 10,
        "crash_video_path": video_path,
        "model_path": "models/traffic_detector.pt",
        "confidence_threshold": 0.5,
    }

    try:
        result = run_pipeline(video_path, config, verbose=verbose)
        validate_result(result)
    except Exception as exc:
        # Keep failure output explicit and module-oriented for fast debugging.
        message = str(exc)
        if "Module failed:" in message:
            print(message)
        else:
            print(f"Module failed: pipeline ({exc})")
        return

    print("===== FLOWX SYSTEM CHECK =====")
    print()
    print(f"State: {result.get('state')}")
    print(f"Reason: {result.get('reason')}")
    print()
    print(f"Selected Lane: {result.get('selected_lane')}")
    print(f"Green Time: {result.get('green_time')}")
    print()

    crash_class = result.get("crash_class")
    crash_conf = result.get("crash_confidence")
    if crash_class:
        print(f"Crash: {crash_class} ({crash_conf})")
    else:
        print("Crash: None")

    print(f"Emergency: {result.get('emergency')}")
    print("Lane densities:", result.get("densities"))


if __name__ == "__main__":
    main(verbose=True)
