from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
import requests

from .core import CooldownFilter, Recognition, normalize_plate
from .storage import EventStore

LOG = logging.getLogger("edge-anpr")


class PlateDetector:
    """ONNX detector with a lightweight contour fallback for setup/demo."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.mode = cfg.get("mode", "contour")
        self.confidence = float(cfg.get("confidence", 0.45))
        model = Path(cfg.get("model_path", ""))
        self.net = cv2.dnn.readNetFromONNX(str(model)) if self.mode == "onnx" else None

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        if self.net is not None:
            return self._detect_onnx(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / max(h, 1)
            if 2.0 <= ratio <= 6.5 and w >= 100 and h >= 22:
                boxes.append((x, y, w, h, 0.5))
        return boxes[:5]

    def _detect_onnx(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True)
        self.net.setInput(blob)
        output = np.squeeze(self.net.forward())
        if output.ndim == 2 and output.shape[0] < output.shape[1]:
            output = output.T
        boxes = []
        for row in output:
            score = float(np.max(row[4:]))
            if score < self.confidence:
                continue
            cx, cy, w, h = row[:4]
            x = int((cx - w / 2) * width / 640)
            y = int((cy - h / 2) * height / 640)
            boxes.append((max(x, 0), max(y, 0), int(w * width / 640), int(h * height / 640), score))
        return boxes


def run(config: dict[str, Any]) -> None:
    source = config.get("source", 0)
    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    camera = config.get("camera", {})
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera.get("width", 1280)))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera.get("height", 720)))
    capture.set(cv2.CAP_PROP_FPS, int(camera.get("fps", 30)))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    detector = PlateDetector(config.get("detector", {}))
    ocr_cfg = config.get("ocr", {})
    store_cfg = config.get("storage", {})
    store = EventStore(store_cfg.get("database", "data/anpr.db"))
    output_dir = Path(store_cfg.get("captures", "data/captures"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cooldown = CooldownFilter(float(config.get("events", {}).get("cooldown_seconds", 10)))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for x, y, w, h, detection_confidence in detector.detect(frame):
                crop = frame[y : y + h, x : x + w]
                if crop.size == 0:
                    continue
                data = pytesseract.image_to_data(crop, lang=ocr_cfg.get("language", "eng"), output_type=pytesseract.Output.DICT)
                best = max(zip(data["conf"], data["text"]), default=(-1, ""), key=lambda item: float(item[0]))
                ocr_confidence, raw_text = float(best[0]), best[1]
                plate = normalize_plate(raw_text)
                if len(plate) < 4 or ocr_confidence < float(ocr_cfg.get("min_confidence", 45)):
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 80), 2)
                cv2.putText(frame, plate, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2)
                if cooldown.allow(plate):
                    stamp = time.time()
                    image_path = None
                    if store_cfg.get("save_images", True):
                        image_path = str(output_dir / f"{plate}_{int(stamp)}.jpg")
                        cv2.imwrite(image_path, crop)
                    event = Recognition(plate, min(ocr_confidence / 100.0, detection_confidence), stamp, image_path)
                    store.add(event)
                    webhook = config.get("events", {}).get("webhook_url")
                    if webhook:
                        try:
                            requests.post(webhook, json=event.__dict__, timeout=2).raise_for_status()
                        except requests.RequestException as exc:
                            LOG.warning("Webhook failed: %s", exc)
                    LOG.info("plate=%s confidence=%.2f", plate, event.confidence)
            if config.get("display", True):
                cv2.imshow("Jetson Edge ANPR", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        store.close()
        cv2.destroyAllWindows()

