"""Central Configuration Module for Enterprise Face Recognition System.

This module defines all hyperparameters, file paths, visual constants, and camera
settings used across the collection, training, and real-time inference pipelines.
"""

from __future__ import annotations

import os
from typing import Tuple

# ── Image & ROI Dimensions ───────────────────────────────────────────────────
ROI_SIZE: int = 200
IMG_SIZE: int = 64
IMG_SHAPE: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)

# ── KNN Hyperparameters & Thresholds ─────────────────────────────────────────
K_NEIGHBORS: int = 3
CONFIDENCE_THRESH: float = 0.6

# ── Storage & Directory Paths ────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data", "faces")
MODEL_DIR: str = os.path.join(BASE_DIR, "data")
MODEL_PATH: str = os.path.join(MODEL_DIR, "model.npz")

# ── Camera & Visual Overlay Constants ────────────────────────────────────────
CAMERA_INDEX: int = 0
BOX_COLOR_BGR: Tuple[int, int, int] = (0, 165, 255)       # Orange in BGR (0, 165, 255)
BOX_COLOR_SUCCESS_BGR: Tuple[int, int, int] = (0, 220, 0) # Green for match
BOX_COLOR_UNKNOWN_BGR: Tuple[int, int, int] = (0, 0, 255) # Red for unknown
BOX_THICKNESS: int = 2
TEXT_SCALE: float = 0.7
TEXT_THICKNESS: int = 2

# ── Window Names ─────────────────────────────────────────────────────────────
WINDOW_MAIN: str = "Camera"
WINDOW_ROI: str = "ROI"
