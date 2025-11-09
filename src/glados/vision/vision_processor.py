from __future__ import annotations

import base64
import queue
import threading
import time
from typing import Final

import cv2
from loguru import logger
import numpy as np
from numpy.typing import NDArray
import requests

from .constants import (
    VLM_SYSTEM_PROMPT,
)
from .vision_config import VisionConfig

class VisionProcessor:
    """Periodically captures camera frames and sends them to a VLM for description, then enqueues result to the llm."""

    def __init__(
        self,
        llm_queue: queue.Queue[str],
        processing_active_event: threading.Event,
        shutdown_event: threading.Event,
        config: VisionConfig,
    ) -> None:
        self.llm_queue = llm_queue
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.config = config

        self._session = requests.Session()
        self._headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            self._headers["Authorization"] = f"Bearer {self.config.api_key}"

        self._capture = None
        self._ensure_capture_ready()

    def run(self) -> None:
        logger.info("VisionProcessor thread started.")
        try:
            while not self.shutdown_event.is_set():
                loop_started = time.perf_counter()

                if not self.processing_active_event.is_set():
                    self._sleep(loop_started)
                    continue

                if not self._ensure_capture_ready():
                    self._sleep(loop_started)
                    continue

                frame = self._grab_frame()
                if frame is None:
                    self._sleep(loop_started)
                    continue

                processed_frame = self._preprocess_frame(frame)
                payload_image = self._encode_frame(processed_frame)
                if payload_image is None:
                    self._sleep(loop_started)
                    continue

                description = self._post_vision_query(payload_image)

                if self.llm_queue.qsize() >= 1: # LLM is busy, avoid flooding the queue with vision updates
                    logger.info("VisionProcessor: Skipped a vision update.")
                    self._sleep(loop_started)
                    continue

                if description:
                    formatted_message = f"[vision] {description}"
                    logger.success(f"Vision update enqueued: {formatted_message}")
                    self.llm_queue.put(formatted_message)
                    self.processing_active_event.set()

                self._sleep(loop_started)
        except Exception as ex:
            logger.error("VisionProcessor uncaught exception: {}", ex)
        finally:
            if self._capture is not None:
                self._capture.release()
            self._session.close()
            logger.info("VisionProcessor thread finished.")

    def _ensure_capture_ready(self) -> bool:
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
        assert self._capture is not None
        ret, frame = self._capture.read()
        if not ret or frame is None:
            logger.warning("VisionProcessor: Failed to capture frame from camera {}.", self.config.camera_index)
            return None
        return frame

    def _preprocess_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Resize frame to fit within target resolution while maintaining aspect ratio."""
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

    def _encode_frame(self, frame: NDArray[np.uint8]) -> bytes | None:
        # use jpg with compression
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success or buffer is None:
            logger.warning("VisionProcessor: Failed to encode frame to JPEG.")
            return None
        return buffer.tobytes()

    def _post_vision_query(self, image_bytes: bytes) -> str | None:
        # encode image to base64
        image_payload = base64.b64encode(image_bytes).decode("ascii")
        body = {
            "model": self.config.vlm_model,
            "stream": False,
            # for the VLM we discard previous messages to improve accuracy
            "messages": [
                {
                    "role": "system",
                    "content": VLM_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "images": [image_payload],
                }
            ],
        }

        content: str | None = None
        try:
            response = self._session.post(
                str(self.config.completion_url),
                headers=self._headers,
                json=body,
                timeout=self.config.capture_interval_seconds, # timeout before next capture
            )
            response.raise_for_status()
            message = response.json().get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
        except Exception as ex:
            logger.error("VisionProcessor: Vision request failed: {}", ex)
            return None
        if content is None:
            logger.warning("VisionProcessor: Received empty content from VLM response")

        return content

    def _sleep(self, loop_started: float) -> None:
        elapsed = time.perf_counter() - loop_started
        sleep_time = max(0.0, self.config.capture_interval_seconds - elapsed)
        if sleep_time:
            self.shutdown_event.wait(timeout=sleep_time)
