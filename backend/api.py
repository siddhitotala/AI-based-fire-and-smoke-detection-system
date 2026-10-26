from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detector import FireSmokeDetector, TemporalVerifier, ensure_model_exists
from utils import append_log, ensure_csv, save_snapshot


BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "data" / "detection_logs.csv"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
PROCESSED_DIR = BASE_DIR / "processed"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "fire_smoke_yolov8.pt"
BASE_MEDIA_URL = "http://localhost:8000"

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FireGuard AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media/snapshots", StaticFiles(directory=str(SNAPSHOT_DIR)), name="snapshots")
app.mount("/media/processed", StaticFiles(directory=str(PROCESSED_DIR)), name="processed")


class UploadSummary(BaseModel):
    source_name: str
    frames_processed: int
    alerts_found: int
    fire_alerts: int
    smoke_alerts: int
    latest_status: str
    snapshot_paths: List[str]
    snapshot_urls: List[str]
    annotated_video_url: str | None = None


def status_from_conf(confidence: float) -> str:
    if confidence >= 0.8:
        return "CRITICAL"
    if confidence >= 0.55:
        return "WARNING"
    return "SAFE"


def build_detector() -> FireSmokeDetector:
    model_path = next((path for path in [DEFAULT_MODEL_PATH, BASE_DIR / "models" / "best.pt"] if path.exists()), DEFAULT_MODEL_PATH)
    ensure_model_exists(str(model_path))
    return FireSmokeDetector(model_path=str(model_path))


def analyze_video(video_path: Path, source_name: str) -> UploadSummary:
    detector = build_detector()
    ensure_csv(LOG_PATH)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    verifier = TemporalVerifier(required_frames=2)
    frames_processed = 0
    alerts_found = 0
    fire_alerts = 0
    smoke_alerts = 0
    latest_status = "SAFE"
    snapshot_paths: List[str] = []
    annotated_video_path = PROCESSED_DIR / f"{source_name}_annotated.mp4"

    writer = None
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    while capture.isOpened() and frames_processed < 120:
        ok, frame = capture.read()
        if not ok:
            break

        if writer is None:
            if not frame_width or not frame_height:
                frame_height, frame_width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(annotated_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_width, frame_height),
            )

        detections = detector.infer_frame(frame)
        has_positive = len(detections) > 0
        verified = verifier.check(has_positive)

        if verified and detections:
            top_detection = max(detections, key=lambda item: item["confidence"])
            status = status_from_conf(top_detection["confidence"])
            latest_status = status
            snapshot = save_snapshot(frame, SNAPSHOT_DIR, source_name, top_detection["type"])
            snapshot_paths.append(str(snapshot))
            append_log(
                LOG_PATH,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source": source_name,
                    "type": top_detection["type"],
                    "confidence": f"{top_detection['confidence']:.3f}",
                    "status": status,
                    "verified_frames": 5,
                    "snapshot_path": str(snapshot),
                },
            )
            alerts_found += 1
            if top_detection["type"] == "Fire":
                fire_alerts += 1
            else:
                smoke_alerts += 1

        annotated_frame = detector.annotate_frame(frame, detections)
        writer.write(annotated_frame)
        frames_processed += 1

    capture.release()
    if writer is not None:
        writer.release()

    return UploadSummary(
        source_name=source_name,
        frames_processed=frames_processed,
        alerts_found=alerts_found,
        fire_alerts=fire_alerts,
        smoke_alerts=smoke_alerts,
        latest_status=latest_status,
        snapshot_paths=snapshot_paths[:5],
        snapshot_urls=[f"{BASE_MEDIA_URL}/media/snapshots/{Path(path).name}" for path in snapshot_paths[:5]],
        annotated_video_url=(f"{BASE_MEDIA_URL}/media/processed/{annotated_video_path.name}" if annotated_video_path.exists() else None),
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/process-upload", response_model=UploadSummary)
async def process_upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "uploaded_video.mp4").suffix or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(await file.read())
    temp_file.flush()
    temp_path = Path(temp_file.name)

    summary = analyze_video(temp_path, Path(file.filename or temp_path.name).stem)
    return summary
