"""List host capture/playback devices and switch the running backends."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import threading
from typing import Any

_SWITCH_LOCK = threading.Lock()


def camera_devices() -> list[dict[str, Any]]:
    """Query Linux V4L2 capabilities without opening a capture stream."""
    import sys
    if not sys.platform.startswith("linux"):
        return []
    import fcntl
    devices = []
    for path in sorted(Path("/dev").glob("video*")):
        descriptor = None
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            capability = bytearray(104)  # struct v4l2_capability
            fcntl.ioctl(descriptor, 0x80685600, capability, True)  # VIDIOC_QUERYCAP
            capabilities, device_caps = struct.unpack_from("II", capability, 84)
            effective = device_caps if capabilities & 0x80000000 else capabilities
            if effective & (0x1 | 0x1000):  # VIDEO_CAPTURE or VIDEO_CAPTURE_MPLANE; exclude metadata nodes
                name = bytes(capability[16:48]).split(b"\0", 1)[0].decode(errors="replace")
                devices.append({"id": str(path), "name": name or path.name})
        except OSError:
            continue
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return devices


def device_snapshot(engine: Any) -> dict[str, Any]:
    audio_io = getattr(engine, "audio_io", None)
    audio = {"available": False, "input": [], "output": [], "reason": "Local audio device selection is unavailable for this backend."}
    if hasattr(audio_io, "device_snapshot"):
        try:
            audio = audio_io.device_snapshot()
        except Exception as exc:
            audio["reason"] = f"Audio devices could not be listed: {exc}"
    vision = getattr(engine, "vision_agent", None)
    camera = vision.camera.snapshot().get("device", vision.settings.camera_spec) if vision else None
    selected = f"/dev/video{camera}" if type(camera) is int else camera
    cameras = camera_devices() if vision else []
    if selected is not None and not any(row["id"] == selected for row in cameras):
        # Preserve a configured stream URL/path or a disconnected camera as an honest selected entry.
        cameras.append({"id": str(selected), "name": "Current camera", "current_only": True})
    return {"audio": audio, "camera": {"available": vision is not None, "devices": cameras, "selected": selected}}


def select_device(engine: Any, kind: str, device: Any) -> None:
    with _SWITCH_LOCK:
        if kind == "camera":
            vision = getattr(engine, "vision_agent", None)
            if vision is None:
                raise ValueError("Vision Core is unavailable")
            if not isinstance(device, str):
                raise ValueError("Choose a listed webcam")
            inventory = device_snapshot(engine)["camera"]
            if device == inventory["selected"]:
                return
            if not any(row["id"] == device and not row.get("current_only") for row in inventory["devices"]):
                raise ValueError("The selected webcam is no longer available; refresh the device list")
            vision.select_camera(device)
        elif kind in {"microphone", "speaker"}:
            audio = getattr(engine, "audio_io", None)
            if not hasattr(audio, "select_device"):
                raise ValueError("Local audio device selection is unavailable for this backend")
            if device is not None and (type(device) is not int or device < 0):
                raise ValueError("Choose a listed audio device or the system default")
            choices = device_snapshot(engine)["audio"]
            direction = "input" if kind == "microphone" else "output"
            if device is not None and not any(row["id"] == device for row in choices[direction]):
                raise ValueError("The selected audio device is no longer available; refresh the device list")
            if kind == "microphone":
                muted = engine.asr_muted_event.is_set()
                engine.set_asr_muted(True)
                try:
                    audio.select_device(kind, device)
                finally:
                    engine.set_asr_muted(muted)
            else:
                audio.select_device(kind, device)
        else:
            raise ValueError("Choose camera, microphone or speaker")
        bus = getattr(engine, "observability_bus", None)
        if bus:
            bus.emit("devices", "selected", f"{kind}: {device if device is not None else 'system default'}")
