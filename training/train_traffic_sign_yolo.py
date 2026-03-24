from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_SIGNS_DIR = ROOT_DIR / "traffic_sign_dataset" / "signs"
DEFAULT_IMAGES_DIR = DEFAULT_SIGNS_DIR / "ts" / "ts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a separate YOLOv8 traffic sign model.")
    parser.add_argument("--train-list", default="train.txt", help="Path to the training image list file.")
    parser.add_argument("--val-list", default="test.txt", help="Path to the validation image list file.")
    parser.add_argument("--classes-file", default="classes.names", help="Path to the class names file.")
    parser.add_argument(
        "--dataset-yaml",
        default=str(CONFIG_DIR / "traffic_sign_data.yaml"),
        help="Path where the generated dataset YAML should be written.",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model to fine-tune.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--project",
        default=str(ROOT_DIR / "runs" / "traffic_sign_training"),
        help="Output project directory for this new traffic sign training.",
    )
    parser.add_argument(
        "--name",
        default="ts_yolo_model",
        help="Run name for traffic sign training. Older vehicle runs are not reused.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device, for example 'cpu', '0', or '0,1'.",
    )
    parser.add_argument(
        "--sample-image",
        default="sample_image.jpg",
        help="Optional sample image path for post-training inference.",
    )
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_IMAGES_DIR),
        help="Directory containing the images referenced by the train/val text files.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run evaluation using --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path to evaluate/predict with. If empty, training uses best.pt; eval-only uses models/traffic_sign_detector.pt.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip validation step after training.",
    )
    parser.add_argument(
        "--eval-project",
        default=str(ROOT_DIR / "runs" / "traffic_sign_eval"),
        help="Project directory for evaluation outputs.",
    )
    parser.add_argument(
        "--eval-name",
        default="ts_yolo_eval",
        help="Run name for evaluation outputs.",
    )
    parser.add_argument(
        "--best-output",
        default=str(MODELS_DIR / "traffic_sign_detector.pt"),
        help="Where to save/copy the best checkpoint.",
    )
    return parser.parse_args()


def load_class_names(classes_file: Path) -> list[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_file}")
    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {classes_file}")
    return names


def resolve_input_path(path_str: str, fallback_dir: Path) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    fallback_candidate = fallback_dir / path_str
    if fallback_candidate.exists():
        return fallback_candidate

    return candidate


def normalize_image_list(list_path: Path, images_dir: Path, output_name: str) -> Path:
    if not list_path.exists():
        raise FileNotFoundError(f"Image list not found: {list_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    normalized_lines: list[str] = []
    missing_files: list[str] = []

    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        candidate = Path(line)
        if candidate.exists():
            resolved = candidate.resolve()
        else:
            resolved = (images_dir / candidate.name).resolve()

        if resolved.exists():
            normalized_lines.append(str(resolved))
        else:
            missing_files.append(line)

    if not normalized_lines:
        raise FileNotFoundError(
            f"No valid image paths could be resolved from {list_path}. "
            f"Missing examples: {missing_files[:5]}"
        )

    normalized_path = CONFIG_DIR / output_name
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")
    return normalized_path


def build_dataset_yaml(train_list: Path, val_list: Path, classes_file: Path, output_yaml: Path) -> Path:
    if not train_list.exists():
        raise FileNotFoundError(f"Train file not found: {train_list}")
    if not val_list.exists():
        raise FileNotFoundError(f"Validation file not found: {val_list}")

    class_names = load_class_names(classes_file)
    dataset_yaml = {
        "train": str(train_list.resolve()),
        "val": str(val_list.resolve()),
        "nc": len(class_names),
        "names": class_names,
    }

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    return output_yaml


def copy_checkpoint(source_weights: Path, destination: Path) -> Path | None:
    if not source_weights.exists():
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_weights.resolve() == destination.resolve():
        return destination
    shutil.copy2(source_weights, destination)
    return destination


def resolve_checkpoint_path(args: argparse.Namespace, train_save_dir: Path | None = None) -> Path:
    if args.checkpoint:
        return resolve_input_path(args.checkpoint, ROOT_DIR)

    if train_save_dir is not None:
        return train_save_dir / "weights" / "best.pt"

    return Path(args.best_output)


def to_serializable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_evaluation_outputs(
    metrics: object,
    output_dir: Path,
    checkpoint_path: Path,
    dataset_yaml: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dict = getattr(metrics, "results_dict", {}) or {}
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "metrics": to_serializable(results_dict),
    }

    json_path = output_dir / "evaluation_metrics.json"
    txt_path = output_dir / "evaluation_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_lines = [
        f"checkpoint: {summary['checkpoint']}",
        f"dataset_yaml: {summary['dataset_yaml']}",
        f"timestamp_utc: {summary['timestamp_utc']}",
        "",
        "metrics:",
    ]
    for key, value in summary["metrics"].items():
        txt_lines.append(f"- {key}: {value}")
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    return json_path, txt_path


def run_evaluation(model: YOLO, dataset_yaml: Path, args: argparse.Namespace) -> tuple[object, Path]:
    metrics = model.val(
        data=str(dataset_yaml),
        device=args.device,
        project=args.eval_project,
        name=args.eval_name,
        exist_ok=True,
        plots=True,
    )
    eval_save_dir = Path(metrics.save_dir) if hasattr(metrics, "save_dir") else Path(args.eval_project) / args.eval_name
    return metrics, eval_save_dir


def main() -> None:
    args = parse_args()

    train_list = resolve_input_path(args.train_list, DEFAULT_SIGNS_DIR)
    val_list = resolve_input_path(args.val_list, DEFAULT_SIGNS_DIR)
    classes_file = resolve_input_path(args.classes_file, DEFAULT_SIGNS_DIR)
    images_dir = resolve_input_path(args.images_dir, ROOT_DIR)
    normalized_train_list = normalize_image_list(train_list, images_dir, "traffic_sign_train_resolved.txt")
    normalized_val_list = normalize_image_list(val_list, images_dir, "traffic_sign_val_resolved.txt")
    dataset_yaml = build_dataset_yaml(
        train_list=normalized_train_list,
        val_list=normalized_val_list,
        classes_file=classes_file,
        output_yaml=Path(args.dataset_yaml),
    )

    train_save_dir: Path | None = None
    if not args.eval_only:
        train_results = YOLO(args.model).train(
            data=str(dataset_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            device=args.device,
            exist_ok=False,
            workers=0,
        )
        train_save_dir = Path(train_results.save_dir)
        training_best = train_save_dir / "weights" / "best.pt"
        copied_weights = copy_checkpoint(training_best, Path(args.best_output))
        if copied_weights is not None:
            print(f"Copied best checkpoint to {copied_weights}")
        else:
            print(f"Best checkpoint not found at {training_best}")

    checkpoint_for_eval = resolve_checkpoint_path(args, train_save_dir=train_save_dir)
    if not checkpoint_for_eval.exists():
        raise FileNotFoundError(f"Checkpoint for evaluation not found: {checkpoint_for_eval}")

    if args.eval_only:
        copied_checkpoint = copy_checkpoint(checkpoint_for_eval, Path(args.best_output))
        if copied_checkpoint is not None:
            print(f"Saved checkpoint copy to {copied_checkpoint}")

    eval_model = YOLO(str(checkpoint_for_eval))
    if not args.skip_eval:
        metrics, eval_save_dir = run_evaluation(eval_model, dataset_yaml, args)
        metrics_json, metrics_txt = save_evaluation_outputs(
            metrics=metrics,
            output_dir=eval_save_dir,
            checkpoint_path=checkpoint_for_eval,
            dataset_yaml=dataset_yaml,
        )
        print(metrics)
        print(f"Saved evaluation outputs to {eval_save_dir}")
        print(f"Saved evaluation metrics: {metrics_json}")
        print(f"Saved evaluation summary: {metrics_txt}")
    else:
        print("Skipping evaluation (--skip-eval enabled).")

    sample_image = resolve_input_path(args.sample_image, DEFAULT_SIGNS_DIR)
    if sample_image.exists():
        inference_project = args.project if not args.eval_only else args.eval_project
        inference_name = f"{args.name}_inference" if not args.eval_only else f"{args.eval_name}_inference"
        eval_model.predict(
            source=str(sample_image),
            conf=0.5,
            save=True,
            project=inference_project,
            name=inference_name,
            exist_ok=True,
        )
        print(f"Saved sample inference outputs for {sample_image}")
    else:
        print(f"Sample image not found, skipping inference: {sample_image}")


if __name__ == "__main__":
    main()
