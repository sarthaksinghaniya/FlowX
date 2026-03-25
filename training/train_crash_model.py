from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from training.crash_dataset import CrashVideoDataset
    from training.crash_model import CrashSeverityModel
except ModuleNotFoundError:
    # Allow running as: python training/train_crash_model.py
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from training.crash_dataset import CrashVideoDataset
    from training.crash_model import CrashSeverityModel


def resolve_default_root() -> Path:
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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_class_weights_from_labels(labels: list[int], num_classes: int = 3) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        counts[label] += 1.0

    if (counts == 0).any():
        missing = (counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"Some classes have 0 samples in train split: {missing}")

    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    if len(loader) == 0:
        return 0.0

    for videos, labels in tqdm(loader, desc="Train", leave=False):
        if videos is None or labels is None or videos.numel() == 0 or labels.numel() == 0:
            continue

        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(videos)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = videos.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / max(total, 1)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    if len(loader) == 0:
        return 0.0, 0.0, 0.0

    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Val", leave=False):
            if videos is None or labels is None or videos.numel() == 0 or labels.numel() == 0:
                continue

            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(videos)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            batch_size = videos.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

    val_loss = running_loss / max(total, 1)
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro") if all_labels else 0.0
    return val_loss, accuracy, macro_f1


def save_loss_curve(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CrashSeverityModel Training Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CrashSeverityModel on crash videos.")
    parser.add_argument("--root-dir", default=str(resolve_default_root()), help="Dataset root containing train/val/test")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames sampled per video")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument(
        "--model-save-path",
        default="models/crash_classifier.pth",
        help="Best model checkpoint path",
    )
    parser.add_argument(
        "--log-save-path",
        default="outputs/crash_training_logs.json",
        help="Training log JSON path",
    )
    parser.add_argument(
        "--curve-save-path",
        default="outputs/loss_curve.png",
        help="Loss curve image path",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    train_dataset = CrashVideoDataset(root_dir=args.root_dir, split="train", num_frames=args.num_frames)
    val_dataset = CrashVideoDataset(root_dir=args.root_dir, split="val", num_frames=args.num_frames)

    train_labels = getattr(train_dataset, "labels", None)
    if train_labels is None:
        train_labels = [label for _, label in train_dataset.samples]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    class_weights = compute_class_weights_from_labels(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = CrashSeverityModel(
        num_classes=3,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model_save_path = Path("models/crash_classifier.pth")
    log_save_path = Path(args.log_save_path)
    curve_save_path = Path(args.curve_save_path)
    os.makedirs(model_save_path.parent, exist_ok=True)
    os.makedirs(log_save_path.parent, exist_ok=True)
    os.makedirs(curve_save_path.parent, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve_epochs = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": val_acc,
                "f1_macro": val_f1,
            }
        )

        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_acc:.4f}")
        print(f"F1 Score: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/crash_classifier.pth")
            print("Model saved at:", os.path.abspath("models/crash_classifier.pth"))
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without val loss improvement.")
            break

    save_loss_curve(train_losses, val_losses, curve_save_path)

    summary = {
        "device": str(device),
        "dataset_root": str(Path(args.root_dir).resolve()),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "class_weights": class_weights.detach().cpu().tolist(),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "model_path": str(model_save_path.resolve()),
        "loss_curve_path": str(curve_save_path.resolve()),
        "epochs_ran": len(history),
        "history": history,
    }
    log_save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Best model saved to: {model_save_path}")
    print(f"Training logs saved to: {log_save_path}")
    print(f"Loss curve saved to: {curve_save_path}")


if __name__ == "__main__":
    main()
