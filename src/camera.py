"""Camera Stream Management Module.

Handles camera lifecycle including device acquisition, reading frames safely,
and releasing hardware resources and GUI windows cleanly.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from config import CAMERA_INDEX

logger = logging.getLogger(__name__)


# ── Camera Stream Lifecycle ──────────────────────────────────────────────────
def open_camera(camera_index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """Initialize and open the video capture hardware device.

    Args:
        camera_index: Numeric index of the video device (default from config).

    Returns:
        An open cv2.VideoCapture instance.

    Raises:
        RuntimeError: If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video camera at index {camera_index}. "
            "Please verify device connection and camera access permissions."
        )
    return cap


def read_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """Read a single frame from the video capture stream.

    Args:
        cap: Active cv2.VideoCapture instance.

    Returns:
        Numpy array representing the BGR frame, or None if reading failed.
    """
    if not cap or not cap.isOpened():
        return None

    ret, frame = cap.read()
    if not ret or frame is None:
        logger.warning("Failed to grab valid frame from video capture stream.")
        return None

    return frame


def release_camera(cap: Optional[cv2.VideoCapture]) -> None:
    """Release the video capture resource and destroy all OpenCV windows.

    Args:
        cap: The cv2.VideoCapture instance to release.
    """
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
