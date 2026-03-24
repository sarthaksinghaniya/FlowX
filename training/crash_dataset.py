from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CrashVideoDataset(Dataset):
    """PyTorch Dataset for crash severity video classification."""

    CLASS_TO_LABEL = {
        "major": 0,
        "moderate": 1,
        "minor": 2,
    }

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __init__(self, root_dir: str | Path, split: str, num_frames: int = 16) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.split_dir = self.root_dir / split

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}'. Use one of: train, val, test.")
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be > 0.")

        self.samples: list[tuple[Path, int]] = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No valid videos found in {self.split_dir}")

    def _build_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for class_name, label in self.CLASS_TO_LABEL.items():
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue

            for video_path in sorted(class_dir.rglob("*")):
                if not video_path.is_file() or video_path.suffix.lower() not in self.VIDEO_EXTENSIONS:
                    continue
                if self._is_valid_video(video_path):
                    samples.append((video_path, label))
        return samples

    @staticmethod
    def _is_valid_video(video_path: Path) -> bool:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return False

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            capture.release()
            return False

        ok, frame = capture.read()
        capture.release()
        return bool(ok and frame is not None)

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, total_frames: int) -> np.ndarray:
        # linspace naturally repeats indices when total_frames < num_frames.
        return np.linspace(0, total_frames - 1, self.num_frames, dtype=np.int32)

    def _read_sampled_frames(self, video_path: Path) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            capture.release()
            raise RuntimeError(f"Video has no frames: {video_path}")

        indices = self._sample_indices(total_frames)
        frames: list[np.ndarray] = []
        last_valid: np.ndarray | None = None

        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                last_valid = frame
                frames.append(frame)
            elif last_valid is not None:
                frames.append(last_valid.copy())

        capture.release()

        if not frames:
            raise RuntimeError(f"Failed to decode sampled frames from {video_path}")

        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())
        return frames[: self.num_frames]

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0  # HWC, [0, 1]
        frame_tensor = frame_tensor.permute(2, 0, 1).contiguous()  # CHW
        frame_tensor = (frame_tensor - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return frame_tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # Safely skip corrupted videos by trying following samples.
        attempts = 0
        max_attempts = len(self.samples)
        current_index = index % len(self.samples)

        while attempts < max_attempts:
            video_path, label = self.samples[current_index]
            try:
                frames = self._read_sampled_frames(video_path)
                video_tensor = torch.stack([self._preprocess_frame(frame) for frame in frames], dim=0)
                return video_tensor, label  # (T, C, H, W), int
            except Exception:
                attempts += 1
                current_index = (current_index + 1) % len(self.samples)

        raise RuntimeError("No decodable video found in dataset after retrying all samples.")


def _resolve_default_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    primary = repo_root / "video_dataset" / "traffic_videos2"
    if primary.exists():
        return primary

    fallback = (
        repo_root
        / "video_dataset"
        / "traffic_videos"
        / "CrashBest"
        / "traffic_videos2"
        / "Balanced Accident Video Dataset"
    )
    return fallback


def run_basic_test(root_dir: str | Path, split: str = "train", num_frames: int = 16, num_samples: int = 3) -> None:
    dataset = CrashVideoDataset(root_dir=root_dir, split=split, num_frames=num_frames)
    print(f"Loaded split='{split}' from: {Path(root_dir)}")
    print(f"Total valid videos: {len(dataset)}")

    sample_video, sample_label = dataset[0]
    print(f"One sample shape: {tuple(sample_video.shape)}")
    print(f"One sample label: {sample_label}")

    max_iter = min(num_samples, len(dataset))
    for idx in range(max_iter):
        video_tensor, label = dataset[idx]
        print(f"Sample {idx}: shape={tuple(video_tensor.shape)}, label={label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick test for CrashVideoDataset")
    parser.add_argument("--root-dir", default=str(_resolve_default_root()), help="Dataset root containing train/val/test.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames sampled per video.")
    parser.add_argument("--num-samples", type=int, default=3, help="How many samples to print.")
    args = parser.parse_args()

    run_basic_test(
        root_dir=args.root_dir,
        split=args.split,
        num_frames=args.num_frames,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
