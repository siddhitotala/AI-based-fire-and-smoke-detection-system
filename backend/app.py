from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
import base64

import cv2
import pandas as pd
import streamlit as st

from detector import FireSmokeDetector, TemporalVerifier, ensure_model_exists
from utils import save_snapshot


BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = BASE_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Relative path instead of hardcoded Windows path - works locally AND on deployment
DEMO_MEDIA_PATH = BASE_DIR / "demo" / "video1.mp4"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "fire_smoke_yolov8.pt",
    BASE_DIR / "models" / "best.pt",
]
WEBCAM_BACKEND_OPTIONS = ["Auto", "DirectShow (Windows)", "Media Foundation (Windows)"]


def is_cloud_deployment() -> bool:
    """Detect if running on Streamlit Community Cloud (or any cloud deploy where we set the secret)."""
    try:
        return bool(st.secrets.get("IS_CLOUD_DEPLOYMENT", False))
    except Exception:
        return False


def open_webcam_capture(cam_index: int, backend_mode: str) -> cv2.VideoCapture:
    backend_map = {
        "Auto": None,
        "DirectShow (Windows)": cv2.CAP_DSHOW,
        "Media Foundation (Windows)": cv2.CAP_MSMF,
    }
    preferred = backend_map.get(backend_mode)

    if preferred is None:
        return cv2.VideoCapture(cam_index)

    capture = cv2.VideoCapture(cam_index, preferred)
    if capture.isOpened():
        return capture

    # Fallback to default backend if preferred one fails.
    return cv2.VideoCapture(cam_index)


def probe_webcam(cam_index: int, backend_mode: str) -> bool:
    capture = open_webcam_capture(cam_index, backend_mode)
    if not capture.isOpened():
        capture.release()
        return False

    ok, frame = capture.read()
    capture.release()
    return bool(ok and frame is not None and frame.size > 0)


def auto_detect_webcam(max_index: int = 4) -> tuple[int, str] | None:
    for cam_index in range(max_index + 1):
        for backend_mode in WEBCAM_BACKEND_OPTIONS:
            if probe_webcam(cam_index, backend_mode):
                return cam_index, backend_mode
    return None


def status_from_conf(confidence: float) -> str:
    if confidence >= 0.8:
        return "CRITICAL"
    if confidence >= 0.55:
        return "WARNING"
    return "SAFE"


def load_logs() -> pd.DataFrame:
    init_session_state()
    session_alerts = st.session_state.get("session_alerts", [])
    if not session_alerts:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "source",
                "type",
                "confidence",
                "status",
                "verified_frames",
                "snapshot_path",
            ]
        )
    return pd.DataFrame(session_alerts)


def init_session_state() -> None:
    if "session_alerts" not in st.session_state:
        st.session_state.session_alerts = []


def append_session_alert(row: dict) -> None:
    init_session_state()
    st.session_state.session_alerts.append(row)


def init_page() -> None:
    st.set_page_config(
        page_title="FireGuard AI",
        page_icon="🔥",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .critical {background: rgba(225, 29, 72, 0.18); padding: 12px 14px; border-radius: 16px; border: 1px solid rgba(225, 29, 72, 0.55);}
        .warning {background: rgba(245, 158, 11, 0.18); padding: 12px 14px; border-radius: 16px; border: 1px solid rgba(245, 158, 11, 0.55);}
        .safe {background: rgba(16, 185, 129, 0.16); padding: 12px 14px; border-radius: 16px; border: 1px solid rgba(16, 185, 129, 0.5);}
        .hero {padding: 1.2rem 1.35rem; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.08); background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(38, 8, 15, 0.92));}
        .hero h1 {margin: 0;}
        .hero p {margin: 0.35rem 0 0 0; color: rgba(226, 232, 240, 0.82);}
        .guide-card {padding: 0.95rem 1rem; border-radius: 18px; border: 1px solid rgba(255, 255, 255, 0.08); background: rgba(15, 23, 42, 0.72);}
        .guide-card strong {display: block; margin-bottom: 0.25rem; color: white;}
        .guide-card span {color: rgba(226, 232, 240, 0.78); font-size: 0.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading YOLOv8 detector...")
def cached_detector(model_path: str, fire_threshold: float, smoke_threshold: float) -> FireSmokeDetector:
    ensure_model_exists(model_path)
    return FireSmokeDetector(
        model_path=model_path,
        fire_threshold=fire_threshold,
        smoke_threshold=smoke_threshold,
    )


def build_detector() -> FireSmokeDetector:
    default_model_path = next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[0])
    fire_threshold = 0.60
    smoke_threshold = 0.55

    return cached_detector(str(default_model_path), fire_threshold, smoke_threshold)


def get_source() -> dict:
    st.sidebar.subheader("Input Source")
    source_mode = st.sidebar.selectbox(
        "Select source",
        ["Webcam", "RTSP", "Video Upload", "Demo Media Path"],
        index=3,
    )

    cloud = is_cloud_deployment()

    if source_mode == "Webcam":
        if cloud:
            st.sidebar.warning(
                "⚠️ Webcam access isn't available in this cloud demo (no camera attached to the server). "
                "Please use 'Video Upload' or 'Demo Media Path' instead."
            )
            return {"kind": "none"}

        auto_scan = st.sidebar.button("Auto-detect webcam")
        if auto_scan:
            detected = auto_detect_webcam(max_index=4)
            if detected is None:
                st.sidebar.error("No working webcam found in indexes 0-4.")
            else:
                st.session_state["webcam_index"] = int(detected[0])
                st.session_state["webcam_backend"] = detected[1]
                st.sidebar.success(f"Detected webcam index {detected[0]} using {detected[1]}.")

        cam_index = st.sidebar.number_input(
            "Webcam index",
            min_value=0,
            value=0,
            step=1,
            key="webcam_index",
        )
        backend_mode = st.sidebar.selectbox(
            "Webcam backend",
            WEBCAM_BACKEND_OPTIONS,
            index=1,
            key="webcam_backend",
        )
        st.sidebar.caption("Tip: If webcam fails, try index 1 and switch backend mode.")
        capture = open_webcam_capture(int(cam_index), backend_mode)

        if not capture.isOpened():
            st.sidebar.error("Webcam open failed. Close apps like Zoom/Teams and try another index/backend.")

        return {
            "kind": "video",
            "capture": capture,
            "source_name": f"webcam_{cam_index}",
        }

    if source_mode == "RTSP":
        if cloud:
            st.sidebar.warning(
                "⚠️ RTSP streams from private/local networks aren't reachable from this cloud demo. "
                "Please use 'Video Upload' or 'Demo Media Path' instead."
            )
            return {"kind": "none"}

        rtsp_url = st.sidebar.text_input("RTSP URL", value="rtsp://127.0.0.1:8554/live")
        return {
            "kind": "video",
            "capture": cv2.VideoCapture(rtsp_url),
            "source_name": "rtsp_stream",
        }

    if source_mode == "Video Upload":
        file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if not file:
            return {"kind": "none"}

        source_name = Path(file.name).stem
        if st.session_state.get("last_uploaded_video") != file.name:
            st.session_state.last_uploaded_video = file.name
            append_session_alert(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source": source_name,
                    "type": "Video Upload",
                    "confidence": "1.00",
                    "status": "READY",
                    "verified_frames": 0,
                    "snapshot_path": "",
                }
            )

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix)
        tmp.write(file.read())
        tmp.flush()
        tmp.close()

        capture = cv2.VideoCapture(tmp.name)
        if not capture.isOpened():
            st.sidebar.error(
                "Failed to open uploaded video. Try converting to a standard MP4 (H.264) or use a different file."
            )
            return {"kind": "none"}

        return {
            "kind": "video",
            "capture": capture,
            "source_name": source_name,
        }

    demo_path = st.sidebar.text_input("Demo file path", value=str(DEMO_MEDIA_PATH))
    path = Path(demo_path.strip())
    if not path.exists():
        st.sidebar.info("Use a local image or video path for a quick demo. Place your demo file at backend/demo/video1.mp4 to use the default.")
        return {"kind": "none"}

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return {"kind": "image", "image_path": path, "source_name": path.stem}

    # For video files
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        st.sidebar.error(f"Failed to open video file: {path}")
        return {"kind": "none"}

    return {
        "kind": "video",
        "capture": capture,
        "source_name": path.stem,
    }


def detection_loop(detector: FireSmokeDetector, capture: cv2.VideoCapture, source_name: str) -> None:
    st.sidebar.subheader("Realtime Tuning")
    temporal_frames = st.sidebar.slider("Temporal verification frames (1 = immediate)", 1, 12, 1)
    process_every_n = st.sidebar.slider("Process every Nth frame", 1, 4, 1)
    max_display_width = st.sidebar.slider("Preview width", 640, 1600, 960, 80)
    verifier = TemporalVerifier(required_frames=temporal_frames)

    run_button = st.sidebar.button("Start Detection")
    max_frames = st.sidebar.number_input("Frames to process", min_value=100, max_value=5000, value=1200, step=100)

    if not run_button:
        return

    if not capture.isOpened():
        st.error(f"Failed to open video source: {source_name}")
        return

    frame_box = st.empty()
    status_box = st.empty()
    snapshot_interval = 45

    processed = 0
    while capture.isOpened() and processed < int(max_frames):
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                st.warning("End of video stream or failed to read frame")
                break
        except Exception as e:
            st.error(f"Error reading video frame: {str(e)}")
            break

        if process_every_n > 1 and (processed % process_every_n) != 0:
            annotated = frame
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_box.image(frame_rgb, channels="RGB", width='stretch')
            processed += 1
            continue

        detections = detector.infer_frame(frame)
        has_positive = len(detections) > 0
        verified = verifier.check(has_positive)

        if verified and detections:
            top_detection = max(detections, key=lambda d: d["confidence"])
            status = status_from_conf(top_detection["confidence"])
            timestamp = datetime.now().isoformat(timespec="seconds")
            snap_path = save_snapshot(frame, SNAPSHOT_DIR, source_name, top_detection["type"])
            row = {
                "timestamp": timestamp,
                "source": source_name,
                "type": top_detection["type"],
                "confidence": f"{top_detection['confidence']:.3f}",
                "status": status,
                "verified_frames": temporal_frames,
                "snapshot_path": str(snap_path),
            }

            append_session_alert(row)

            if status == "CRITICAL":
                status_box.markdown("<div class='critical'>CRITICAL fire/smoke confidence detected.</div>", unsafe_allow_html=True)
            else:
                status_box.markdown("<div class='warning'>Alert verified by temporal filter.</div>", unsafe_allow_html=True)
        else:
            status_box.markdown("<div class='safe'>Monitoring active: SAFE</div>", unsafe_allow_html=True)

        if (processed + 1) % snapshot_interval == 0:
            snapshot_path = save_snapshot(frame, SNAPSHOT_DIR, source_name, "interval")
            append_session_alert(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source": source_name,
                    "type": "Frame Snapshot",
                    "confidence": "0.00",
                    "status": "SNAPSHOT",
                    "verified_frames": 0,
                    "snapshot_path": str(snapshot_path),
                }
            )

        annotated = detector.annotate_frame(frame, detections)
        if annotated.shape[1] > max_display_width:
            new_h = int(annotated.shape[0] * (max_display_width / annotated.shape[1]))
            annotated = cv2.resize(annotated, (max_display_width, new_h), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_box.image(frame_rgb, channels="RGB", width='stretch')
        processed += 1

    capture.release()
    st.success("Detection session completed.")


def detection_from_image(detector: FireSmokeDetector, image_path: Path, source_name: str) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        st.error("Unable to read the selected image file.")
        return

    detections = detector.infer_frame(image)
    annotated = detector.annotate_frame(image, detections)
    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption=f"Demo image: {source_name}", width='stretch')

    if detections:
        top_detection = max(detections, key=lambda d: d["confidence"])
        status = status_from_conf(top_detection["confidence"])
        timestamp = datetime.now().isoformat(timespec="seconds")
        snap_path = save_snapshot(image, SNAPSHOT_DIR, source_name, top_detection["type"])

        row = {
            "timestamp": timestamp,
            "source": source_name,
            "type": top_detection["type"],
            "confidence": f"{top_detection['confidence']:.3f}",
            "status": status,
            "verified_frames": 1,
            "snapshot_path": str(snap_path),
        }
        append_session_alert(row)
        st.success(f"Detection saved as {status} and snapshot captured.")
    else:
        st.info("No fire or smoke detected in this image.")


def show_logs_and_exports(log_df: pd.DataFrame) -> None:
    st.subheader("Detection Logs")
    st.dataframe(log_df, use_container_width=True)

    csv_data = log_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Logs as CSV",
        data=csv_data,
        file_name="fireguard_detection_logs.csv",
        mime="text/csv",
    )


def show_snapshot_gallery(log_df: pd.DataFrame) -> None:
    st.subheader("Snapshot Gallery")
    snapshot_paths: List[str] = []
    if "snapshot_path" in log_df.columns:
        snapshot_df = log_df.dropna(subset=["snapshot_path"])
        snapshot_df = snapshot_df[snapshot_df["snapshot_path"].astype(str).str.strip() != ""]
        if "timestamp" in snapshot_df.columns:
            snapshot_df = snapshot_df.sort_values("timestamp", ascending=False)
        snapshot_paths = [p for p in snapshot_df["snapshot_path"].tolist() if p and Path(p).is_file()]

    if not snapshot_paths:
        st.info("No snapshots available yet.")
        return

    cols = st.columns(3)
    for idx, snap in enumerate(snapshot_paths[:30]):
        snap_path = Path(snap)
        if not snap_path.is_file():
            continue

        image = cv2.imread(str(snap_path))
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cols[idx % 3].image(image, caption=snap_path.name, channels="RGB", use_container_width=True)


def main() -> None:
    init_page()
    st.title("FireGuard AI - Real-Time Fire & Smoke Detection System")

    init_session_state()
    log_df = load_logs()
    st.markdown(
        """
        <div class="hero">
          <p>Simple live fire and smoke monitoring with snapshots, logs, and annotated detections.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Quick Start")
    step1, step2, step3 = st.columns(3)
    step1.markdown(
        '<div class="guide-card"><strong>1. Pick a source</strong><span>Use the demo video, upload a file, or connect a webcam / RTSP stream.</span></div>',
        unsafe_allow_html=True,
    )
    step2.markdown(
        '<div class="guide-card"><strong>2. Start monitoring</strong><span>Use the sidebar controls to begin detection and tune sensitivity.</span></div>',
        unsafe_allow_html=True,
    )
    step3.markdown(
        '<div class="guide-card"><strong>3. Review results</strong><span>See annotated frames, logs, and snapshots in the tabs below.</span></div>',
        unsafe_allow_html=True,
    )

    tab_live, tab_logs, tab_gallery = st.tabs(["Live Monitoring", "Activity Log", "Snapshots"])

    try:
        detector = build_detector()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with tab_live:
        st.subheader("Live Monitoring")
        source = get_source()
        if source.get("kind") == "none":
            st.warning("Select a webcam, RTSP stream, uploaded video, or demo path to start monitoring.")
        elif source.get("kind") == "image":
            detection_from_image(detector, source["image_path"], source["source_name"])
        elif not source["capture"].isOpened():
            st.error("Unable to open the selected stream/source.")
        else:
            detection_loop(detector, source["capture"], source["source_name"])

    with tab_logs:
        st.subheader("Activity Log")
        show_logs_and_exports(load_logs())

    with tab_gallery:
        st.subheader("Snapshots")
        show_snapshot_gallery(load_logs())


if __name__ == "__main__":
    main()