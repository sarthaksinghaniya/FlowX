# FlowX

FlowX is a real-time traffic intelligence demo that combines YOLOv8 vehicle detection, adaptive lane-based signal control, emergency green corridor simulation, and AI-vs-static performance comparison in a Streamlit dashboard.

## Features

- Real-time traffic detection with YOLOv8
- Adaptive lane-wise signal allocation
- Emergency green corridor simulation across intersections
- Performance comparison between static timing and FlowX AI control

## Project Structure

- `src/` core logic and integration modules
- `config/` lane and dataset configuration
- `models/` trained model weights
- `demo/` sample demo video
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
- Default model path is `models/traffic_detector.pt`
- Dataset and experiment summaries are available in `reports/`
