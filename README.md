# FlowX

FlowX is currently a computer-vision traffic analysis project in active development. The repo already contains a working dataset pipeline, YOLOv8 training runs, and experiment outputs, but it is not yet a finished end-user application with a stable prediction or deployment flow.

## Current Status

The strongest completed milestone in this repo is vehicle detection training with YOLOv8 on a prepared traffic dataset.

- Dataset prepared and split for YOLO training
- Training artifacts saved under `runs/detect/`
- Validation metrics available for the latest strong run
- Supporting dataset utilities added under `tools/`
- Early forecasting/research work also exists in `DCRNN-master/`, but it is not yet integrated into the main workflow

## Progress Snapshot

Based on `reports/final_dataset_stats.md`:

- Total images: 20,714
- Train split: 14,499
- Validation split: 4,143
- Test split: 2,072
- Class counts:
  - `car`: 77,341
  - `truck`: 9,208
  - `bus`: 3,108
  - `motorcycle`: 50

Based on `runs/detect/traffic_detector7/results.csv` after 30 epochs on CPU:

- Precision: 0.8103
- Recall: 0.5590
- mAP@50: 0.6173
- mAP@50-95: 0.5619

This makes `traffic_detector7` the clearest current benchmark in the repository.

## What Is Working

- Raw traffic images and videos can be gathered into a training dataset
- Auto-annotation and dataset rebuilding scripts exist
- YOLO dataset config has been generated
- YOLOv8 training runs successfully and stores plots, confusion matrices, and metrics
- Repo includes pretrained YOLO weights (`yolov8n.pt`, `yolov8s.pt`) for experimentation

## What Is Not Finished Yet

This repo still needs consolidation before it should be treated as a polished app:

- There is no stable top-level inference app or UI entry point yet
- `models/` is currently empty, so the trained detector has not been promoted into a clean reusable artifact here
- Dataset scripts use inconsistent paths and class lists in different places
- Some pipeline files are placeholders or partially automated rather than production-ready
- The DCRNN forecasting work is present, but not connected to the detection pipeline

## Key Project Files

- `training/train_yolo.py`: basic YOLO training entry point
- `tools/populate_dataset.py`: initial dataset creation flow
- `tools/rebuild_dataset.py`: dataset rebuild flow and final stats generation
- `tools/validate_dataset.py`: dataset validation utility
- `configs/traffic_dataset.yaml`: YOLO dataset config
- `reports/final_dataset_stats.md`: latest dataset summary
- `runs/detect/traffic_detector7/`: latest strong training run outputs

## Recommended Next Steps

To move FlowX from project state to app state, the next priorities should be:

1. Standardize dataset paths and class definitions across all scripts.
2. Export and save the best trained model into `models/`.
3. Add a single inference script for image/video prediction.
4. Define whether DCRNN forecasting is part of the main product or a separate experiment.
5. Add a clean demo flow and usage instructions for new contributors.

## Development Notes

- Python dependencies are listed in `requirements.txt`
- The current training script uses `ultralytics`
- Latest visible YOLO training run used:
  - model: `yolov8n.pt`
  - epochs: `30`
  - image size: `640`
  - batch size: `16`
  - device: `cpu`

## Bottom Line

FlowX has made real progress on dataset preparation and traffic object detection. The machine learning experimentation layer is active and producing measurable results, but the repo still needs integration, cleanup, and a proper inference surface before it can be described as a complete app.
