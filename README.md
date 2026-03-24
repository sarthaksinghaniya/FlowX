# FlowX

FlowX is a traffic intelligence project with a Streamlit dashboard and YOLO-based model workflows for:

- vehicle detection and adaptive signal simulation
- emergency green-corridor logic
- traffic sign detection training + evaluation

## What Is Working Now

- Real-time dashboard entrypoint: `app.py`
- Core runtime modules: `src/`
- Vehicle model usage in app: `models/traffic_detector.pt`
- Traffic sign training/evaluation script: `training/train_traffic_sign_yolo.py`
- Traffic sign checkpoint target: `models/traffic_sign_detector.pt`
- Evaluation artifacts are saved under `runs/traffic_sign_eval/`

## Repository Layout

- `app.py` Streamlit dashboard
- `src/` pipeline, signal, emergency corridor, comparison logic
- `training/` YOLO training scripts
- `config/` dataset and lane configs
- `models/` model checkpoints
- `demo/` demo media used by dashboard
- `runs/` training/validation/evaluation outputs
- `traffic_sign_dataset/` traffic sign dataset files
- `video_dataset/` traffic video assets
- `reports/`, `outputs/`, `notebooks/` research and generated artifacts

## Setup

Install base dependencies:

```bash
pip install -r requirements.txt
pip install streamlit
```

## Run Dashboard

```bash
streamlit run app.py
```

Default assets used by dashboard:

- Video: `demo/test_video.mp4`
- Vehicle model: `models/traffic_detector.pt`

## Traffic Sign Training + Evaluation

Train (and evaluate after training):

```bash
python training/train_traffic_sign_yolo.py \
  --train-list train.txt \
  --val-list test.txt \
  --classes-file classes.names \
  --device cpu
```

Evaluate only from existing checkpoint:

```bash
python training/train_traffic_sign_yolo.py \
  --eval-only \
  --checkpoint models/traffic_sign_detector.pt \
  --train-list train.txt \
  --val-list test.txt \
  --classes-file classes.names \
  --device cpu \
  --eval-name ts_yolo_eval
```

### Evaluation Outputs

Each eval run saves:

- standard YOLO validation plots/images
- `evaluation_metrics.json`
- `evaluation_summary.txt`

at:

- `runs/traffic_sign_eval/<eval-name>/`

## Latest Traffic Sign Eval Snapshot

From run saved at `runs/traffic_sign_eval/ts_yolo_model6_eval` (March 24, 2026):

- Precision: `0.9719`
- Recall: `0.8619`
- mAP50: `0.9416`
- mAP50-95: `0.7944`

## Notes

- `training/train_yolo.py` is a minimal baseline script; active traffic-sign workflow is in `training/train_traffic_sign_yolo.py`.
- If `sample_image.jpg` is not present, sample inference step is skipped automatically.
