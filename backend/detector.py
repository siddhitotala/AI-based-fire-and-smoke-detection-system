from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path
from typing import List

import cv2

try:
	from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
	YOLO = None


class FireSmokeDetector:
	def __init__(self, model_path: str, fire_threshold: float = 0.6, smoke_threshold: float = 0.55):
		self.model_path = Path(model_path)
		self.fire_threshold = float(fire_threshold)
		self.smoke_threshold = float(smoke_threshold)

		if YOLO is None:
			warnings.warn("ultralytics not available; detector will use heuristic fallback")
			self.model = None
			self.names = {}
			self.heuristic_fallback = True
		else:
			try:
				self.model = YOLO(str(self.model_path))
				# model.names maps class ids to names
				self.names = getattr(self.model, "names", {}) or {}
				self.heuristic_fallback = False
			except Exception:
				warnings.warn("Failed to load YOLO model; using heuristic fallback detector")
				self.model = None
				self.names = {}
				self.heuristic_fallback = True

	def _normalize_bbox(self, bbox) -> List[float]:
		if bbox is None:
			return [0.0, 0.0, 0.0, 0.0]

		if hasattr(bbox, "tolist"):
			bbox = bbox.tolist()

		if isinstance(bbox, (list, tuple)):
			if len(bbox) == 0:
				return [0.0, 0.0, 0.0, 0.0]
			if len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
				return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
			if len(bbox) == 1 and isinstance(bbox[0], (list, tuple)):
				return self._normalize_bbox(bbox[0])
			if len(bbox) > 0 and isinstance(bbox[0], (list, tuple)):
				first = bbox[0]
				if len(first) >= 4:
					return [float(first[0]), float(first[1]), float(first[2]), float(first[3])]

		try:
			flat = list(bbox)
			if len(flat) >= 4:
				return [float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])]
		except Exception:
			pass

		return [0.0, 0.0, 0.0, 0.0]

	def infer_frame(self, frame) -> List[dict]:
		# If model unavailable or failed to load, use heuristic fallback
		if getattr(self, "heuristic_fallback", False) or self.model is None:
			return self._heuristic_detect(frame)

		# ultralytics accepts RGB images
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		try:
			results = self.model.predict(img, imgsz=640, conf=0.25, verbose=False)
		except TypeError:
			# Older ultralytics versions use call syntax
			results = self.model(img)

		detections: List[dict] = []
		if not results:
			return detections

		res = results[0]
		boxes = getattr(res, "boxes", None)
		if boxes is None:
			return detections

		for box in boxes:
			try:
				conf = float(box.conf) if hasattr(box, "conf") else float(box.conf.cpu().numpy())
			except Exception:
				conf = 0.0

			try:
				cls = int(box.cls) if hasattr(box, "cls") else int(box.cls.cpu().numpy())
			except Exception:
				cls = 0

			label = self.names.get(cls, str(cls))

			try:
				xyxy = box.xyxy if hasattr(box, "xyxy") else box.xyxy.cpu().numpy()
				bbox = self._normalize_bbox(xyxy)
			except Exception:
				bbox = [0.0, 0.0, 0.0, 0.0]

			detections.append({"type": label, "confidence": conf, "bbox": bbox})

		return detections

	def annotate_frame(self, frame, detections: List[dict]):
		out = frame.copy()
		for det in detections:
			bbox = self._normalize_bbox(det.get("bbox", [0, 0, 0, 0]))
			x1, y1, x2, y2 = [int(v) for v in bbox]
			conf = det.get("confidence", 0.0)
			label = f"{det.get('type','')}: {conf:.2f}"
			cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
			cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
		return out

	def _heuristic_detect(self, frame) -> List[dict]:
		"""Simple color/area-based heuristic to detect fire (orange/red blobs)
		and smoke (low-saturation grayish regions). Returns same detection format.
		"""
		detections: List[dict] = []
		if frame is None:
			return detections

		h, w = frame.shape[:2]
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# Fire: detect warm colors (orange/yellow/red)
		lower_fire1 = (0, 100, 120)
		upper_fire1 = (20, 255, 255)
		lower_fire2 = (160, 100, 120)
		upper_fire2 = (179, 255, 255)
		mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
		mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
		fire_mask = cv2.bitwise_or(mask1, mask2)
		fire_mask = cv2.medianBlur(fire_mask, 5)
		contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area < max(1000, 0.001 * h * w):
				continue
			x, y, ww, hh = cv2.boundingRect(cnt)
			conf = min(0.99, float(area) / (h * w))
			detections.append({"type": "Fire", "confidence": conf, "bbox": [x, y, x + ww, y + hh]})

		# Smoke: detect low-saturation, medium-brightness regions
		smoke_mask = cv2.inRange(hsv, (0, 0, 80), (179, 60, 230))
		smoke_mask = cv2.GaussianBlur(smoke_mask, (9, 9), 0)
		contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area < max(3000, 0.002 * h * w):
				continue
			x, y, ww, hh = cv2.boundingRect(cnt)
			conf = min(0.9, float(area) / (h * w) * 2)
			detections.append({"type": "Smoke", "confidence": conf, "bbox": [x, y, x + ww, y + hh]})

		return detections


class TemporalVerifier:
	def __init__(self, required_frames: int = 3):
		self.required_frames = int(required_frames)
		self.buffer = deque(maxlen=self.required_frames)

	def check(self, has_positive: bool) -> bool:
		self.buffer.append(bool(has_positive))
		return len(self.buffer) == self.required_frames and all(self.buffer)


def ensure_model_exists(path: str | Path) -> None:
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"Model not found: {p}")