# FlowX

FlowX is a real-time traffic intelligence demo that combines YOLOv8 vehicle detection, adaptive lane-based signal control, emergency green corridor simulation, and AI-vs-static performance comparison in a Streamlit dashboard.

## Features

- Real-time traffic detection with YOLOv8
- Adaptive lane-wise signal allocation
- Emergency green corridor simulation across intersections
- Performance comparison between static timing and FlowX AI control

## Current Progress

- Streamlit dashboard is integrated and demo-ready via `app.py`
- Real-time pipeline modules are organized under `src/`
- Emergency corridor and AI-vs-static comparison modules are connected into the dashboard
- Research notebooks have been generated and executed successfully
- Sample outputs are being written to `outputs/`

## Progress Snapshot

- Dataset prepared: 20,714 images
- Train/val/test split: 14,499 / 4,143 / 2,072
- Latest YOLO training benchmark:
  - Precision: 0.8103
  - Recall: 0.5590
  - mAP@50: 0.6173
  - mAP@50-95: 0.5619
- Notebook execution status: 8 of 8 completed successfully

## Project Structure

- `src/` core logic and integration modules
- `config/` lane and dataset configuration
- `models/` trained model weights
- `demo/` sample demo video
- `notebooks/` research and experimentation workflow
- `outputs/` generated notebook artifacts and demo exports
- `reports/` project metrics and summaries
- `app.py` Streamlit dashboard entry point

## How To Run

```bash
streamlit run app.py
```

## Tech Stack

- YOLOv8
- OpenCV
- Streamlit

## Notes

- Default demo media is loaded from `demo/test_video.mp4`
- Sample notebook image is available at `demo/sample_cctv_frame.jpg`
- Default model path is `models/traffic_detector.pt`
- Dataset and experiment summaries are available in `reports/`
- Notebook run summary is available at `outputs/notebook_run_summary.json`
