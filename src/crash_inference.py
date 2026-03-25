from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from training.crash_model import CrashSeverityModel
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from training.crash_model import CrashSeverityModel


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = Path(BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "models", "crash_classifier.pth")
NUM_FRAMES = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAPPING = {
    0: "major",
    1: "moderate",
    2: "minor",
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

_MODEL: CrashSeverityModel | None = None


def _load_model() -> CrashSeverityModel | None:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    print("Loading model from:", MODEL_PATH)
    print("File exists:", os.path.exists(MODEL_PATH))

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        return None

    try:
        model = CrashSeverityModel().to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as exc:
        print(f"Failed to load crash model: {exc}")
        return None

    _MODEL = model
    return _MODEL


def preprocess_frames(frames: list[np.ndarray]) -> torch.Tensor | None:
    if not frames:
        return None

    prepared_frames = frames[:NUM_FRAMES]
    while len(prepared_frames) < NUM_FRAMES:
        prepared_frames.append(prepared_frames[-1].copy())

    processed_tensors: list[torch.Tensor] = []
    try:
        for frame in prepared_frames:
            if frame is None or frame.size == 0:
                return None
            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0  # HWC, [0, 1]
            frame_tensor = frame_tensor.permute(2, 0, 1).contiguous()  # CHW
            frame_tensor = (frame_tensor - IMAGENET_MEAN) / IMAGENET_STD
            processed_tensors.append(frame_tensor)
    except Exception:
        return None

    stacked = torch.stack(processed_tensors, dim=0)  # (16, 3, 224, 224)
    return stacked.unsqueeze(0)  # (1, 16, 3, 224, 224)


def sample_frames_from_video(video_path: str | Path, num_frames: int = NUM_FRAMES) -> list[np.ndarray]:
    video_path = Path(video_path)
    if not video_path.exists():
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return []

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    sampled_frames: list[np.ndarray] = []
    last_valid: np.ndarray | None = None

    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()
        if success and frame is not None:
            sampled_frames.append(frame)
            last_valid = frame
        elif last_valid is not None:
            sampled_frames.append(last_valid.copy())

    capture.release()

    if not sampled_frames:
        return []

    while len(sampled_frames) < num_frames:
        sampled_frames.append(sampled_frames[-1].copy())
    return sampled_frames[:num_frames]


def _predict_from_frames(frames: list[np.ndarray]) -> dict[str, Any] | None:
    model = _load_model()
    if model is None:
        return None

    input_tensor = preprocess_frames(frames)
    if input_tensor is None:
        return None
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    probabilities = [float(p.item()) for p in probs]
    return {
        "class": CLASS_MAPPING[pred_idx],
        "confidence": confidence,
        "probs": probabilities,
    }


def predict_crash(video_path: str | Path) -> dict[str, Any] | None:
    frames = sample_frames_from_video(video_path, num_frames=NUM_FRAMES)
    if not frames:
        return None

    result = _predict_from_frames(frames)
    if result is None:
        return None

    print("Probabilities:", result["probs"])
    return result


def predict_from_stream(frame_buffer: list[np.ndarray]) -> dict[str, Any] | None:
    if len(frame_buffer) < NUM_FRAMES:
        return None
    frames = frame_buffer[-NUM_FRAMES:]
    if not frames:
        return None
    return _predict_from_frames(frames)


def test_inference() -> None:
    video_path = str(ROOT_DIR / "demo" / "test_video.mp4")

    if not os.path.exists(video_path):
        print("Test video not found")
        return

    result = predict_crash(video_path)
    if result:
        print("Prediction Result:")
        print("Class:", result["class"])
        print("Confidence:", result["confidence"])
        print("Probs:", result["probs"])


if __name__ == "__main__":
    test_inference()
