"""ASCII camera feed for real-time terminal display."""

from __future__ import annotations

import threading
import time

import cv2
from loguru import logger
import numpy as np
from numpy.typing import NDArray

from .ascii_renderer import frame_to_ascii


class AsciiCameraFeed:
    """Captures camera frames and converts them to ASCII for TUI display.

    Runs in a background thread, capturing at approximately 15 FPS.
    """

    def __init__(
        self,
        camera_index: int = 0,
        fps: float = 15.0,
        width: int = 60,
        height: int = 20,
    ) -> None:
        """Initialize the ASCII camera feed.

        Args:
            camera_index: Camera device index
            fps: Target frames per second
            width: ASCII output width in characters
            height: ASCII output height in characters
        """
        self._camera_index = camera_index
        self._interval = 1.0 / fps
        self._width = width
        self._height = height

        self._capture: cv2.VideoCapture | None = None
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

        # Thread-safe frame buffer
        self._lock = threading.Lock()
        self._current_frame: str = ""
        self._enabled = False

    def start(self) -> None:
        """Start the camera feed thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._shutdown.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AsciiCameraFeed")
        self._thread.start()
        logger.info("AsciiCameraFeed started (camera={}, {}x{}).", self._camera_index, self._width, self._height)

    def stop(self) -> None:
        """Stop the camera feed thread."""
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        logger.info("AsciiCameraFeed stopped.")

    def enable(self) -> None:
        """Enable frame capture."""
        self._enabled = True

    def disable(self) -> None:
        """Disable frame capture (saves CPU when panel hidden)."""
        self._enabled = False
        with self._lock:
            self._current_frame = ""

    @property
    def enabled(self) -> bool:
        """Check if feed is enabled."""
        return self._enabled

    def get_frame(self) -> str:
        """Get the current ASCII frame.

        Returns:
            ASCII art string, or empty string if no frame available
        """
        with self._lock:
            return self._current_frame

    def _run(self) -> None:
        """Main capture loop."""
        while not self._shutdown.is_set():
            loop_start = time.perf_counter()

            if self._enabled:
                self._capture_and_convert()

            # Sleep for remaining interval
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, self._interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _ensure_capture(self) -> bool:
        """Ensure camera capture is open."""
        if self._capture is not None and self._capture.isOpened():
            return True

        if self._capture is not None:
            self._capture.release()

        self._capture = cv2.VideoCapture(self._camera_index)
        if not self._capture.isOpened():
            logger.warning("AsciiCameraFeed: Unable to open camera {}.", self._camera_index)
            return False

        return True

    def _capture_and_convert(self) -> None:
        """Capture a frame and convert to ASCII."""
        if not self._ensure_capture():
            return

        assert self._capture is not None
        ret, frame = self._capture.read()
        if not ret or frame is None:
            return

        # Convert to ASCII
        ascii_frame = frame_to_ascii(frame, self._width, self._height)

        with self._lock:
            self._current_frame = ascii_frame
