"""ASCII renderer for camera frames using block characters."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

# Block characters for 5-level grayscale (dark to light)
BLOCKS = " ░▒▓█"


def frame_to_ascii(
    frame: NDArray[np.uint8],
    width: int = 60,
    height: int = 20,
) -> str:
    """Convert a BGR camera frame to ASCII art using block characters.

    Args:
        frame: BGR image from OpenCV (HWC format, uint8)
        width: Target width in characters
        height: Target height in characters

    Returns:
        Multi-line string of block characters representing the frame
    """
    if frame is None or frame.size == 0:
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to target dimensions
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    # Map pixel values (0-255) to block indices (0-4)
    # Using integer division: 0-50->0, 51-101->1, 102-152->2, 153-203->3, 204-255->4
    indices = np.clip(resized // 51, 0, 4)

    # Convert to characters
    lines = []
    for row in indices:
        line = "".join(BLOCKS[i] for i in row)
        lines.append(line)

    return "\n".join(lines)
