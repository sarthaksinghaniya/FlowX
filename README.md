# FlowX — AI Traffic Intelligence System

FlowX is an AI-powered traffic intelligence platform built for adaptive traffic control, crash-aware response, and emergency route prioritization. It combines computer vision, machine learning, and decision intelligence into a single real-time dashboard for smart mobility scenarios.

## Overview

FlowX is designed as a real-time AI traffic control system that:

- Analyzes live or recorded traffic video
- Detects vehicles and estimates lane-level density
- Identifies crash conditions and severity signals
- Prioritizes emergency movement through dynamic signal control
- Visualizes decisions, signals, and intersections in an interactive dashboard

At its core, FlowX connects perception, inference, and traffic decision logic into one operational pipeline.

## Problem Statement

Urban traffic systems still rely heavily on static traffic lights and fixed timing plans. This leads to:

- Inefficient signal allocation during variable traffic demand
- Congestion buildup at overloaded lanes
- Delayed emergency movement through busy intersections
- No direct crash-awareness in traffic signal decisions

Traditional systems react too slowly to live road conditions and provide little intelligence during abnormal events.

## Solution

FlowX addresses these gaps through adaptive, AI-driven traffic management:

- Adaptive signal timing based on live density estimation
- Crash-aware decision support for safer control behavior
- Emergency green-corridor logic for high-priority vehicle passage
- Transparent dashboard visualizations for operators, demos, and evaluations

## Key Features

- Real-time vehicle detection using YOLO-based vision pipelines
- Density-based signal optimization across lanes
- Crash severity detection with an AI model
- Emergency vehicle priority and green-corridor routing
- AI decision engine with priority logic and finite-state control
- Multi-intersection Mapbox visualization with pydeck
- Animated traffic signal simulation
- Simulation speed control for demos and analysis
- Performance comparison dashboard for AI vs static control behavior

## System Architecture

```text
Video Input
  -> YOLO Detection
  -> Lane Mapping
  -> Density Engine
  -> Crash Detection
  -> Emergency Override
  -> Priority Engine
  -> Signal Control
  -> Dashboard
```

## Core Algorithms

### Density Estimation

```text
D_i = (alpha V_i + beta Q_i + gamma T_i) / Capacity
```

Where:

- `V_i` = vehicle count for lane `i`
- `Q_i` = queue pressure or backlog estimate
- `T_i` = temporal weighting term
- `Capacity` = normalized lane capacity

### Green Time Allocation

```text
G_i = (D_i / sum(D)) x Cycle Time
```

This allocates green duration proportionally to lane demand.

### Crash-Aware Adjustment

```text
D_eff = D_i + lambda x Crash Weight
```

This increases effective priority when crash severity requires intervention.

### Control Logic

FlowX uses:

- A finite-state machine with `NORMAL`, `CRASH`, and `EMERGENCY` modes
- Priority-based decision logic for lane selection and override handling
- Signal timing policies that support normal flow, crash response, and emergency routing

## Project Structure

```text
FlowX/
├── app.py
├── src/
├── training/
├── models/
├── config/
├── demo/
├── outputs/
├── runs/
├── reports/
├── tests/
└── README.md
```

Key directories:

- `app.py` — Streamlit dashboard entrypoint
- `src/` — core pipeline, density logic, emergency corridor, signal control, and comparison utilities
- `training/` — model training scripts including crash-model training
- `models/` — trained checkpoints and model assets
- `config/` — configuration files for lanes, datasets, and runtime setup
- `demo/` — demo video assets used by the dashboard
- `outputs/` — generated artifacts, processed outputs, and exported results
- `runs/` — experiment logs, evaluation snapshots, and training run artifacts
- `reports/` — analysis summaries and supporting documentation

## How to Run

### Install

```bash
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run app.py
```

## Streamlit Deployment

FlowX is set up for Streamlit Community Cloud deployment with:

- `app.py` as the app entrypoint
- `requirements.txt` for Python dependencies
- `packages.txt` for Linux system packages needed by OpenCV/video decoding
- `.streamlit/config.toml` for Streamlit app configuration

Deploy steps:

1. Push the repository to GitHub
2. Open Streamlit Community Cloud
3. Create a new app from this repo
4. Set the main file path to `app.py`
5. Deploy

Notes:

- Keep `demo/test_video.mp4` and `models/traffic_detector.pt` in the repo because the dashboard expects them by default
- `models/crash_classifier.pth` is optional at runtime; if it is absent, crash inference gracefully falls back instead of breaking the app
- First deploy can take a while because `torch` and `ultralytics` are large dependencies

### Train Crash Model

```bash
python training/train_crash_model.py
```

Default dashboard flow uses:

- Demo video assets from `demo/`
- Model checkpoints from `models/`
- Generated outputs and logs under `outputs/` and `runs/`

## Results

FlowX is designed to deliver measurable operational gains:

- Reduced vehicle waiting time through adaptive signal allocation
- Improved throughput compared with static timing approaches
- Faster emergency response through priority-aware signal switching
- Better traffic visibility through real-time AI state monitoring

## Demo Scenarios

The system supports clear demonstration flows for:

- Normal traffic balancing
- Crash-detected adaptive response
- Emergency override and green-corridor activation

## Future Scope

- Multi-intersection scaling across larger urban grids
- Reinforcement learning for signal optimization
- Smart-city platform integration
- IoT and vehicle-to-infrastructure communication
- Live edge deployment with connected traffic hardware

## Why FlowX Is Unique

FlowX stands out because it combines:

- Crash-aware traffic control instead of density-only optimization
- Emergency intelligence with route-priority behavior
- Transparent AI decision visualization for operators and judges
- Real-time adaptability across signals, state transitions, and map views

## Screenshots

Add project screenshots to the repository and update the paths below.

### Dashboard

![FlowX Dashboard](outputs/images/dashboard-placeholder.png)

### Map View

![FlowX Map View](outputs/images/map-view-placeholder.png)

### Signal Animation

![FlowX Signal Animation](outputs/images/signal-animation-placeholder.png)

## Outputs and Logs

FlowX stores generated artifacts and experiment traces in the repository structure:

- `outputs/` for processed media, exports, visual outputs, and dashboard assets
- `runs/` for model runs, evaluation logs, and experiment artifacts
- `reports/` for summaries, findings, and supporting analysis

Recommended examples to include over time:

- Dashboard screenshots
- Detection output frames
- Crash inference logs
- Signal timing comparison charts
- Evaluation summaries and run metadata

## License

MIT License
