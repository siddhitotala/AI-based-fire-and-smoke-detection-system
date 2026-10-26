# FireGuard AI - Real-Time Fire & Smoke Detection System

Simple fire and smoke monitoring dashboard.

## Quick Start

### Streamlit UI

Place your YOLOv8 fire/smoke model at:

```text
backend/models/fire_smoke_yolov8.pt
```

Then run:

```bash
cd backend
pip install -r requirements.txt
streamlit run app.py
```

## Recommended Workflow

- Use the Streamlit UI at `http://localhost:8501`
- Upload a video, select the demo media path, or connect a live source
- View annotated detections, logs, and snapshots in the Streamlit tabs

## Included Features

- Live monitoring controls (Webcam, RTSP, Video Upload)
- YOLOv8 fire/smoke detection
- Temporal verification (false-positive reduction)
- SAFE / WARNING / CRITICAL status
- Detection logs + CSV export
- Snapshot gallery
- Confidence threshold settings
- Multi-camera simulation
