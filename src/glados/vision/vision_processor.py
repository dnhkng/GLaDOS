"""Vision processor that periodically captures camera frames and generates scene descriptions using FastVLM."""

from __future__ import annotations

import queue
import threading
import time

import cv2
from loguru import logger
import numpy as np
from numpy.typing import NDArray

from .constants import VISION_DEFAULT_PROMPT
from .fastvlm import FastVLM
from .vision_config import VisionConfig
from .vision_request import VisionRequest
from .vision_state import VisionState


class VisionProcessor:
    """Periodically captures camera frames and updates a vision snapshot using local ONNX FastVLM."""

    def __init__(
        self,
        vision_state: VisionState,
        processing_active_event: threading.Event,
        shutdown_event: threading.Event,
        config: VisionConfig,
        request_queue: queue.Queue[VisionRequest] | None = None,
    ) -> None:
        """Initialize VisionProcessor.

        Args:
            vision_state: Store for the latest vision snapshot
            processing_active_event: Event indicating if processing is active
            shutdown_event: Event to signal shutdown
            config: Vision module configuration
            request_queue: Queue for on-demand tool requests
        """
        self.vision_state = vision_state
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.config = config
        self._request_queue = request_queue

        # Load FastVLM model
        self._model = FastVLM(config.model_dir)

        # Camera capture
        self._capture: cv2.VideoCapture | None = None
        self._ensure_capture_ready()

        # Frame differencing for scene change detection
        self._last_frame: NDArray[np.uint8] | None = None
        self._last_features: NDArray[np.float32] | None = None
        self._prompt_cache: dict[tuple[str, int], str] = {}

    def run(self) -> None:
        """Main processing loop for the vision processor thread."""
        logger.info("VisionProcessor thread started.")
        try:
            while not self.shutdown_event.is_set():
                loop_started = time.perf_counter()

                if self._process_tool_request():
                    self._sleep(loop_started)
                    continue

                if not self._ensure_capture_ready():
                    self._sleep(loop_started)
                    continue

                frame = self._grab_frame()
                if frame is None:
                    self._sleep(loop_started)
                    continue

                # Resize frame for scene-change detection
                processed = self._preprocess_frame(frame)

                # Skip if scene hasn't changed significantly
                if not self._scene_changed(processed):
                    logger.debug("VisionProcessor: Scene unchanged, skipping VLM inference.")
                    self._sleep(loop_started)
                    continue

                # Store frame for next comparison
                self._last_frame = processed.copy()
                self._prompt_cache.clear()

                # Get scene description using local ONNX model
                description = self._get_description(frame, prompt=VISION_DEFAULT_PROMPT, max_tokens=self.config.max_tokens)

                if description:
                    self.vision_state.update(description)
                    logger.success("Vision snapshot updated: {}", description)

                self._sleep(loop_started)

        except Exception as ex:
            logger.exception("VisionProcessor uncaught exception: {}", ex)
        finally:
            if self._capture is not None:
                self._capture.release()
            logger.info("VisionProcessor thread finished.")

    def _ensure_capture_ready(self) -> bool:
        """Ensure camera capture is ready.

        Returns:
            True if camera is ready, False otherwise
        """
        if self._capture is not None and self._capture.isOpened():
            return True

        if self._capture is not None:
            self._capture.release()

        self._capture = cv2.VideoCapture(self.config.camera_index)
        if not self._capture.isOpened():
            logger.error(
                "VisionProcessor: Unable to open camera index {}. Retrying in {:.1f}s.",
                self.config.camera_index,
                self.config.capture_interval_seconds,
            )
            return False

        logger.success("VisionProcessor: Camera {} opened successfully.", self.config.camera_index)
        return True

    def _grab_frame(self) -> NDArray[np.uint8] | None:
        """Grab a frame from the camera.

        Returns:
            Frame as uint8 array (BGR, HWC) or None if capture failed
        """
        assert self._capture is not None
        ret, frame = self._capture.read()
        if not ret or frame is None:
            logger.warning("VisionProcessor: Failed to capture frame from camera {}.", self.config.camera_index)
            return None
        return frame

    def _preprocess_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Resize frame to target resolution while maintaining aspect ratio.

        Args:
            frame: Input frame (BGR, HWC, uint8)

        Returns:
            Resized frame (BGR, HWC, uint8)
        """
        target_resolution = self.config.resolution
        height, width = frame.shape[:2]
        max_dim = max(height, width)

        if max_dim <= target_resolution:
            return frame

        scale = target_resolution / float(max_dim)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        return resized

    def _scene_changed(self, current_frame: NDArray[np.uint8]) -> bool:
        """Check if scene has changed significantly from last processed frame.

        Args:
            current_frame: Current frame to compare

        Returns:
            True if scene changed above threshold, False otherwise
        """
        if self._last_frame is None:
            return True

        # Ensure frames are same size for comparison
        if current_frame.shape != self._last_frame.shape:
            return True

        # Compute normalized absolute difference
        diff = cv2.absdiff(current_frame, self._last_frame)
        change_ratio = float(np.mean(diff)) / 255.0

        return change_ratio > self.config.scene_change_threshold

    def _get_description(
        self,
        frame: NDArray[np.uint8],
        prompt: str,
        max_tokens: int,
    ) -> str | None:
        """Get scene description using local ONNX FastVLM model.

        Args:
            frame: Input frame (BGR, HWC, uint8)

        Returns:
            Scene description or None if inference failed
        """
        prompt = prompt.strip() if prompt else ""
        try:
            vision_features = self._model.encode_image(frame)
            description = self._model.describe_from_features(
                vision_features,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            self._last_features = vision_features
            if description:
                self._prompt_cache[(prompt, int(max_tokens))] = description
            return description
        except Exception as e:
            logger.error("FastVLM inference failed: {}", e)
            return None

    def _process_tool_request(self) -> bool:
        """Handle one pending tool request, if available."""
        if self._request_queue is None:
            return False

        try:
            request = self._request_queue.get_nowait()
        except queue.Empty:
            return False

        if not self._ensure_capture_ready():
            request.response_queue.put("error: camera not available")
            return True

        frame = self._grab_frame()
        if frame is None:
            request.response_queue.put("error: failed to capture frame")
            return True

        processed = self._preprocess_frame(frame)
        reuse_cached = self._last_features is not None and not self._scene_changed(processed)

        if reuse_cached:
            logger.debug("VisionProcessor: Reusing cached vision features for tool request.")
            prompt = request.prompt.strip() if request.prompt else ""
            cache_key = (prompt, int(request.max_tokens))
            description = self._prompt_cache.get(cache_key)
            if description is None:
                description = self._model.describe_from_features(
                    self._last_features,
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                )
                if description:
                    self._prompt_cache[cache_key] = description
        else:
            self._last_frame = processed.copy()
            self._prompt_cache.clear()
            description = self._get_description(
                frame,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
            )
        if not description:
            request.response_queue.put("error: vision inference failed")
            return True

        request.response_queue.put(description)
        return True

    def _sleep(self, loop_started: float) -> None:
        """Sleep until next capture interval.

        Args:
            loop_started: Time when loop iteration started
        """
        elapsed = time.perf_counter() - loop_started
        sleep_time = max(0.0, self.config.capture_interval_seconds - elapsed)
        if sleep_time:
            self.shutdown_event.wait(timeout=sleep_time)
