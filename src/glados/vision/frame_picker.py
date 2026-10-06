"""Bounded recent camera frames, scored on the CPU for focus and freshness."""

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import cv2
from numba import njit
import numpy as np
from numpy.typing import NDArray

from .face_tracker import FaceTracker, face_overlay


@njit(cache=True)
def laplacian_variance(gray: NDArray[np.uint8]) -> float:
    """Focus score without intermediate gradient arrays or Python pixel loops."""
    height, width = gray.shape
    count = (height - 2) * (width - 2)
    if height < 3 or width < 3:
        return 0.0
    total = 0.0
    squared = 0.0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            value = 4.0 * gray[y, x] - gray[y - 1, x] - gray[y + 1, x] - gray[y, x - 1] - gray[y, x + 1]
            total += value
            squared += value * value
    return max(0.0, squared / count - (total / count) ** 2)


def focus_score(frame: NDArray[np.uint8]) -> float:
    """Score a consistent small luminance image; higher usually means sharper."""
    height, width = frame.shape[:2]
    scale = min(1.0, 320 / max(height, width))
    if scale < 1:
        frame = cv2.resize(
            frame, (max(3, round(width * scale)), max(3, round(height * scale))), interpolation=cv2.INTER_AREA
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(laplacian_variance(gray))


@dataclass(frozen=True)
class CameraFrame:
    image: NDArray[np.uint8]
    captured_at: float
    sharpness: float
    scoring_ms: float
    sequence: int


class MotionTracker:
    """Small CPU frame differences with exposure compensation and fast attack/slow decay."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._previous: NDArray[np.int16] | None = None
        self._observed_at = 0.0
        self._activity = 0.0

    def reset(self) -> None:
        with self._lock:
            self._previous = None
            self._observed_at = 0.0
            self._activity = 0.0

    def update(self, image: NDArray[np.uint8], captured_at: float) -> None:
        small = cv2.resize(image, (96, 72), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.int16)
        with self._lock:
            elapsed = captured_at - self._observed_at
            if self._previous is None or not 0 < elapsed <= 1:
                self._activity = 0.0
            else:
                delta = gray - self._previous
                # A uniform exposure shift is not scene motion.
                delta = delta - np.median(delta)
                changed_fraction = float(np.mean(np.abs(delta) > 12))
                activity = min(1.0, changed_fraction / 0.05)
                smoothing_s = 0.15 if activity > self._activity else 0.8
                weight = 1 - math.exp(-elapsed / smoothing_s)
                self._activity += weight * (activity - self._activity)
            self._previous = gray
            self._observed_at = captured_at

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            # Disconnected/stale frames cannot keep the schedule in its active state.
            activity = self._activity if 0 <= time.time() - self._observed_at <= 1 else 0.0
            return {"activity": round(activity, 4), "observed_at": self._observed_at}


class RecentFrames:
    """Bound memory and pick only from the most recent scene, never an old sharp frame."""

    def __init__(self, capacity: int = 8, window_s: float = 0.25, max_age_s: float = 1.0) -> None:
        self._frames: deque[CameraFrame] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.window_s = window_s
        self.max_age_s = max_age_s

    def add(self, frame: CameraFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def pick(self, now: float | None = None, after: int = -1) -> tuple[CameraFrame | None, int]:
        now = time.time() if now is None else now
        with self._lock:
            if not self._frames:
                return None, 0
            cutoff = max(now - self.max_age_s, self._frames[-1].captured_at - self.window_s)
            candidates = [frame for frame in self._frames if frame.captured_at >= cutoff and frame.sequence > after]
            # Prefer the newer frame if equally sharp. No inference requests are queued here.
            return (
                (max(candidates, key=lambda f: (f.sharpness, f.captured_at)), len(candidates))
                if candidates
                else (None, 0)
            )


class CameraSampler:
    """Drain the webcam continuously so waiting for model capacity cannot stale its buffer."""

    def __init__(
        self,
        camera: int | str = 0,
        window_s: float = 0.25,
        fps: int = 30,
        face_tracking: bool = True,
        face_interval_s: float = 0.03,
        sleep_after_s: float = 5.0,
        face_backend: str = "yunet",
    ) -> None:
        self.camera = camera
        self._camera_revision = 0
        self.fps = fps
        self.frames = RecentFrames(window_s=window_s)
        self.faces = FaceTracker(face_tracking, face_interval_s, sleep_after_s, face_backend)
        self.motion = MotionTracker()
        self._stop = threading.Event()
        self._enabled = threading.Event()
        self._enabled.set()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._live_frame: tuple[CameraFrame, dict[str, Any]] | None = None
        self._preview_cache: dict[bool, tuple[bytes, int]] = {}
        self._capture_times: deque[float] = deque(maxlen=60)
        self._status: dict[str, Any] = {"connected": False, "error": None, "frames": 0}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Compile once before capture; the real frame loop then stays inexpensive.
        laplacian_variance(np.zeros((4, 4), dtype=np.uint8))
        self._thread = threading.Thread(target=self._run, name="VisionCamera", daemon=True)
        self._thread.start()

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled.set()
        else:
            self._enabled.clear()
            self.frames.clear()
            self._ready.clear()
            self.faces.reset()
            with self._lock:
                self.motion.reset()
                self._live_frame = None
                self._capture_times.clear()

    def select_camera(self, camera: int | str) -> None:
        """The capture thread owns reopening; invalidate frames from the old input."""
        with self._lock:
            self.camera = camera
            self._camera_revision += 1
            self._status.update(connected=False, error=None)
            self._live_frame = None
            self._capture_times.clear()
            self.frames.clear()
            self._ready.clear()
            self.faces.reset()
            self.motion.reset()

    def preview(self, overlay: bool = True) -> tuple[bytes, int] | None:
        """Encode the newest captured frame with only its own face detection.

        JPEGs are made on demand for optic-feed and inspector viewers, with
        separate shared caches for raw frames and annotated frames. No preview encoding runs when nobody is watching.
        """
        with self._preview_lock:
            with self._lock:
                latest = self._live_frame if self._enabled.is_set() and self._status["connected"] else None
            if latest is None:
                return None
            frame, face = latest
            if time.time() - frame.captured_at > 1:
                return None
            cached = self._preview_cache.get(overlay)
            if cached and cached[1] == frame.sequence:
                return cached
            image = face_overlay(frame.image, face) if overlay else frame.image
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return None
            with self._lock:
                if not self._enabled.is_set() or not self._status["connected"]:
                    return None
            self._preview_cache[overlay] = (encoded.tobytes(), frame.sequence)
            return self._preview_cache[overlay]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            status = {**self._status, "enabled": self._enabled.is_set(), "device": self.camera}
            times = self._capture_times
            rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None
            status["capture_hz"] = round(rate, 1) if rate is not None else None
        face = self.faces.snapshot()
        if not status["enabled"] or not status["connected"]:
            face.update(available=False, present=None)
        return {**status, "face": face, "motion": self.motion.snapshot()}

    def wait_ready(self, timeout: float = 1.5) -> bool:
        return self._ready.wait(timeout)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2)

    def _run(self) -> None:
        capture = None
        capture_revision = -1
        sequence = 0
        try:
            while not self._stop.is_set():
                with self._lock:
                    camera, revision = self.camera, self._camera_revision
                if capture is not None and capture_revision != revision:
                    capture.release()
                    capture = None
                if not self._enabled.is_set():
                    if capture is not None:
                        capture.release()
                        capture = None
                    with self._lock:
                        self._status["connected"] = False
                        self._live_frame = None
                        self._capture_times.clear()
                    self._stop.wait(0.05)
                    continue
                if capture is None:
                    capture = cv2.VideoCapture(camera)
                    capture_revision = revision
                    if isinstance(camera, int) or str(camera).startswith("/dev/video"):
                        # USB webcams often deliver raw YUYV below the requested
                        # frame rate; prefer their compressed 30 fps mode.
                        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    capture.set(cv2.CAP_PROP_FPS, self.fps)
                    # Keep one buffer queued while the CPU processes the other;
                    # a single V4L2 buffer misses alternate sensor frames.
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                cycle_started = time.perf_counter()
                ok, image = capture.read()
                with self._lock:
                    if capture_revision != self._camera_revision:
                        continue
                if not ok or image is None:
                    with self._lock:
                        self._status.update(connected=False, error="Webcam did not return a frame")
                        self._live_frame = None
                        self._capture_times.clear()
                    capture.release()
                    capture = None
                    self.frames.clear()
                    self._ready.clear()
                    self.faces.reset()
                    self.motion.reset()
                    self._stop.wait(1)
                    continue
                captured_at = time.time()
                started = time.perf_counter()
                sharpness = focus_score(image)
                scoring_ms = (time.perf_counter() - started) * 1000
                sequence += 1
                with self._lock:
                    if capture_revision != self._camera_revision:
                        continue
                    if self._enabled.is_set():
                        self.motion.update(image, captured_at)
                        frame = CameraFrame(image, captured_at, sharpness, scoring_ms, sequence)
                        self.frames.add(frame)
                        self._ready.set()
                        face = self.faces.update(image, captured_at)
                        # Hold the last detected frame between CPU tracking ticks to avoid flicker.
                        if face is not None or not self.faces.enabled or self.faces.backend == "e4b":
                            self._live_frame = (frame, face or {})
                    self._capture_times.append(captured_at)
                    self._status.update(
                        connected=True,
                        error=None,
                        frames=sequence,
                        captured_at=captured_at,
                        scoring_ms=round(scoring_ms, 3),
                    )
                self._stop.wait(max(0, 1 / self.fps - (time.perf_counter() - cycle_started)))
        except (cv2.error, OSError, ValueError) as exc:
            with self._lock:
                self._status.update(connected=False, error=f"Camera failed: {type(exc).__name__}")
        finally:
            if capture is not None:
                capture.release()
            self.frames.clear()
            self._ready.clear()
            self.faces.reset()
            self.motion.reset()
            with self._lock:
                self._status["connected"] = False
                self._live_frame = None
                self._capture_times.clear()
