"""Fast CPU face tracking, with optional Haar and E4B backends."""

from collections import deque
import math
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


class FaceTracker:
    """Track presence independently of the slower scene-description model."""

    def __init__(
        self,
        enabled: bool = True,
        interval_s: float = 0.03,
        sleep_after_s: float = 5.0,
        backend: str = "yunet",
    ) -> None:
        self.enabled = enabled
        self.backend = backend
        self.interval_s = interval_s
        self.sleep_after_s = sleep_after_s
        self._lock = threading.Lock()
        self._detect_lock = threading.Lock()
        self._generation = 0
        self._absent_since: float | None = None
        self._last_seen: float | None = None
        self._state: dict[str, Any] = {}
        self._observations: deque[float] = deque(maxlen=60)
        self._error: str | None = None
        self._cascades = []
        self._detector = None
        if enabled and backend == "yunet":
            try:
                model = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
                self._detector = cv2.FaceDetectorYN.create(
                    str(model),
                    "",
                    (320, 240),
                    0.7,
                    0.3,
                    5000,
                    cv2.dnn.DNN_BACKEND_OPENCV,
                    cv2.dnn.DNN_TARGET_CPU,
                )
            except (cv2.error, AttributeError, ValueError):
                self._error = "YuNet face detector unavailable"
        if enabled and backend == "haar":
            try:
                for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
                    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
                    if cascade.empty():
                        raise ValueError("Bundled Haar cascade unavailable")
                    self._cascades.append(cascade)
            except (cv2.error, AttributeError, ValueError):
                self._error = "Face detector unavailable"

    def reset(self) -> None:
        """Forget presence when capture stops; camera downtime is not an empty room."""
        with self._lock:
            self._generation += 1
            self._state = {}
            self._observations.clear()
            self._absent_since = self._last_seen = None

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        with self._lock:
            times = self._observations
            rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None
            return {
                "enabled": self.enabled,
                "source": self.backend,
                "available": False,
                "present": None,
                "error": self._error,
                **self._state,
                "last_seen_at": self._last_seen,
                "tracking_hz": round(rate, 1) if rate is not None else None,
                "absent_seconds": (max(0.0, now - self._absent_since) if self._absent_since is not None else 0.0),
                "sleep_after_s": self.sleep_after_s,
            }

    def locate(self, frame: NDArray[np.uint8], captured_at: float) -> dict[str, Any]:
        """Locate a face in this exact frame, without changing live presence."""
        if not self.enabled or self._error or self.backend not in ("haar", "yunet"):
            return {"available": False, "present": None, "observed_at": captured_at, "error": self._error}
        started = time.perf_counter()
        try:
            height, width = frame.shape[:2]
            scale = min(1.0, (320 if self.backend == "yunet" else 400) / max(height, width))
            small = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            h, w = small.shape[:2]
            with self._detect_lock:
                if self.backend == "yunet":
                    if small.ndim == 2:
                        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
                    self._detector.setInputSize((w, h))
                    _, detections = self._detector.detect(small)
                    boxes = [] if detections is None else list(detections[:, :4])
                else:
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
                    gray = cv2.equalizeHist(gray)
                    min_side = max(24, round(min(w, h) * 0.13))
                    front, profile = self._cascades
                    boxes = list(
                        front.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
                    )
                    boxes.extend(
                        profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(min_side, min_side))
                    )
                    flipped = profile.detectMultiScale(
                        cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=3, minSize=(min_side, min_side)
                    )
                    boxes.extend((w - x - bw, y, bw, bh) for x, y, bw, bh in flipped)
            faces = []
            for x, y, bw, bh in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
                # Neural boxes may extend outside the frame. Keep every valid face.
                left, top = max(0.0, float(x)), max(0.0, float(y))
                right, bottom = min(w, float(x + bw)), min(h, float(y + bh))
                bw, bh = right - left, bottom - top
                if bw <= 0 or bh <= 0:
                    continue
                face = {
                    "x": float(2 * (left + bw / 2) / w - 1),
                    "y": float(2 * (top + bh / 2) / h - 1),
                    "bbox": [float(left / w), float(top / h), float(bw / w), float(bh / h)],
                    "closeness": float(min(1.0, bh / h)),
                }
                # Haar front/profile passes can report the same person more than once.
                if any(
                    abs(face["x"] - seen["x"]) < min(face["bbox"][2], seen["bbox"][2]) * 0.6
                    and abs(face["y"] - seen["y"]) < min(face["bbox"][3], seen["bbox"][3]) * 0.6
                    for seen in faces
                ):
                    continue
                faces.append(face)
            result: dict[str, Any] = {
                "available": True, "present": bool(faces), "observed_at": captured_at, "faces": faces,
            }
            if faces:
                # Preserve the largest-face fields for existing preview consumers.
                result.update(faces[0])
            result["tracking_ms"] = round((time.perf_counter() - started) * 1000, 3)
        except (cv2.error, ValueError):
            result = {"available": False, "present": None, "observed_at": captured_at, "error": "Face detection failed"}
        return result

    def update(self, frame: NDArray[np.uint8], captured_at: float) -> dict[str, Any] | None:
        if not self.enabled or self._error:
            return {"available": False, "present": None, "observed_at": captured_at, "error": self._error}
        if self.backend not in ("haar", "yunet"):
            return None
        with self._lock:
            if captured_at - self._state.get("observed_at", 0) < self.interval_s:
                return None
            generation = self._generation
        result = self.locate(frame, captured_at)
        with self._lock:
            if generation != self._generation:
                return None
            self._accept(result)
            return result

    def publish(self, result: dict[str, Any]) -> None:
        """Publish a validated E4B observation when its owning mind accepts the frame."""
        if not self.enabled:
            return
        with self._lock:
            self._accept(result)

    def _accept(self, result: dict[str, Any]) -> None:
        captured_at = result["observed_at"]
        previous_at = self._state.get("observed_at", captured_at)
        self._state = result
        self._observations.append(captured_at)
        if result.get("available") and result.get("present") is False:
            freshness_s = 6 if self.backend == "e4b" else 3
            if self._absent_since is None or captured_at - previous_at > freshness_s:
                self._absent_since = captured_at
        else:
            # Uncertain/malformed observations break the run of confirmed empty frames.
            self._absent_since = None
            if result.get("available") and result.get("present") is True:
                self._last_seen = captured_at


def face_from_model(value: object, captured_at: float) -> dict[str, Any]:
    """Validate the 0..1000 corner box; unknown data must not imply an empty room."""
    result: dict[str, Any] = {
        "source": "e4b",
        "available": False,
        "present": None,
        "observed_at": captured_at,
    }
    if not isinstance(value, dict) or value.get("presence") not in ("present", "absent", "uncertain"):
        result["error"] = "Face observation unavailable"
        return result
    native_box = "box_2d" in value
    presence, box = value["presence"], value.get("box_2d" if native_box else "box")
    result.update(available=True, present={"present": True, "absent": False, "uncertain": None}[presence])
    if box is None:
        return result
    if presence != "present":
        result.update(present=None, error="Contradictory face observation")
        return result
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in box)
    ):
        result["error"] = "Face location unavailable"
        return result
    corners = [v / 1000 for v in box]
    if native_box:
        top, left, bottom, right = corners
    else:
        # Accept the previous custom X/Y key for older response templates.
        left, top, right, bottom = corners
    if right <= left or bottom <= top:
        result["error"] = "Face location unavailable"
        return result
    result.update(
        bbox=[left, top, right - left, bottom - top],
        x=left + right - 1,
        y=top + bottom - 1,
        closeness=bottom - top,
    )
    return result


def face_overlay(frame: NDArray[np.uint8], face: dict[str, Any]) -> NDArray[np.uint8]:
    """Draw a box, gaze point and coordinates on a display copy, never the model input."""
    if not face.get("available") or not face.get("present") or not face.get("bbox"):
        return frame
    preview = frame.copy()
    height, width = frame.shape[:2]
    x, y, bw, bh = face["bbox"]
    left, top = round(x * width), round(y * height)
    right, bottom = round((x + bw) * width), round((y + bh) * height)
    center = (round((x + bw / 2) * width), round((y + bh / 2) * height))
    color = (190, 220, 70)
    cv2.rectangle(preview, (left, top), (right, bottom), (15, 20, 20), 4)
    cv2.rectangle(preview, (left, top), (right, bottom), color, 2)
    cv2.drawMarker(preview, center, (15, 20, 20), cv2.MARKER_CROSS, 18, 4)
    cv2.drawMarker(preview, center, color, cv2.MARKER_CROSS, 18, 2)
    label = f"Face  X {face['x']:+.2f}  Y {face['y']:+.2f}"
    (label_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    label_at = (max(0, min(left, width - label_width - 4)), max(16, top - 8))
    cv2.putText(preview, label, label_at, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(preview, label, label_at, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return preview
