from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2


def ensure_csv(log_path: Path) -> None:
    if log_path.exists():
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "source",
                "type",
                "confidence",
                "status",
                "verified_frames",
                "snapshot_path",
            ],
        )
        writer.writeheader()


def append_log(log_path: Path, row: Dict) -> None:
    ensure_csv(log_path)
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "source",
                "type",
                "confidence",
                "status",
                "verified_frames",
                "snapshot_path",
            ],
        )
        writer.writerow(row)


def save_snapshot(frame, output_dir: Path, source_name: str, event_type: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{source_name.replace(' ', '_')}_{event_type}_{timestamp}.jpg"
    out_path = output_dir / filename
    cv2.imwrite(str(out_path), frame)
    return out_path
