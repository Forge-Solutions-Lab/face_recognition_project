"""Region of Interest (ROI) and Image Preprocessing Module.

Provides pure, modular image transformation functions including bounding box drawing,
cropping, colorspace conversion, dimensional resizing, and feature flattening.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from config import BOX_COLOR_BGR, BOX_THICKNESS, IMG_SIZE, ROI_SIZE

BoxCoords = Tuple[int, int, int, int]


# ── ROI Geometric & Visual Operations ────────────────────────────────────────
def draw_roi_box(
    frame: np.ndarray,
    roi_size: int = ROI_SIZE,
    color: Tuple[int, int, int] = BOX_COLOR_BGR,
    thickness: int = BOX_THICKNESS,
) -> Tuple[np.ndarray, BoxCoords]:
    """Calculate central coordinates and draw the orange ROI bounding box on frame.

    Args:
        frame: Input BGR video frame.
        roi_size: Side length of the square ROI box in pixels.
        color: BGR color tuple for the rectangle border.
        thickness: Border line thickness in pixels.

    Returns:
        Tuple containing (annotated_frame, (x, y, w, h)).
    """
    h, w = frame.shape[:2]
    # Calculate top-left point to center the box
    x = max(0, (w - roi_size) // 2)
    y = max(0, (h - roi_size) // 2)
    
    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + roi_size, y + roi_size), color, thickness)
    
    return annotated, (x, y, roi_size, roi_size)


def crop_roi(frame: np.ndarray, box: BoxCoords) -> np.ndarray:
    """Extract and return the image sub-array bounded by the specified box coordinates.

    Args:
        frame: Source image or video frame.
        box: (x, y, w, h) bounding rectangle tuple.

    Returns:
        Cropped numpy image slice.
    """
    x, y, w, h = box
    return frame[y : y + h, x : x + w]


# ── Image Transformation & Preprocessing ─────────────────────────────────────
def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an image from BGR / RGB colorspace to single-channel 8-bit grayscale.

    Args:
        img: Input image array.

    Returns:
        Grayscale image array. If already single-channel, returns input as-is.
    """
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_frame(img: np.ndarray, size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    """Resize an image to fixed target dimensions using area/linear interpolation.

    Args:
        img: Input image array.
        size: Target (width, height) tuple.

    Returns:
        Resized image array.
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def flatten(img: np.ndarray) -> np.ndarray:
    """Flatten a 2D or 3D image matrix into a 1D feature vector.

    Args:
        img: Input image array.

    Returns:
        1D float/uint8 numpy array representing the image vector.
    """
    return img.flatten().astype(np.float32)
