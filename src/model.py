"""Model Dataset Loading, Serialization, and Deserialization Module.

Provides utilities to read raw face image directories, build flattened dataset
matrices, and persist / restore trained models using compressed NumPy (.npz) archives.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import cv2
import numpy as np

from config import DATA_DIR, IMG_SHAPE, MODEL_PATH
from src.roi import flatten, resize_frame, to_grayscale

logger = logging.getLogger(__name__)


# ── Dataset Loading & Preparation ────────────────────────────────────────────
def train_model(
    data_dir: str = DATA_DIR,
    target_size: Tuple[int, int] = IMG_SHAPE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Scan the dataset directory, load all face images, preprocess, and construct (X, Y).

    Args:
        data_dir: Root path containing member subdirectories (e.g. data/faces/6601001_สมชาย/).
        target_size: (width, height) tuple for standardizing image dimensions.

    Returns:
        Tuple (X, Y, unique_labels) where:
            X: (N, D) float32 matrix of flattened face vectors.
            Y: (N,) string array of corresponding member labels (folder names).
            unique_labels: List of unique label strings found in the dataset.

    Raises:
        FileNotFoundError: If data_dir does not exist or contains no valid images.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    X_list: List[np.ndarray] = []
    Y_list: List[str] = []

    subdirs = sorted(os.listdir(data_dir))
    for folder_name in subdirs:
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path) or folder_name.startswith("."):
            continue

        image_files = sorted(os.listdir(folder_path))
        for filename in image_files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder_path, filename)
            # Read image using OpenCV
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Could not read image file: %s", img_path)
                continue

            # Standardize: Grayscale -> Resize -> 1D Flatten
            gray = to_grayscale(img)
            resized = resize_frame(gray, size=target_size)
            vec = flatten(resized)

            X_list.append(vec)
            Y_list.append(folder_name)

    if not X_list:
        raise ValueError(
            f"No valid training images found in '{data_dir}'. "
            "Please collect samples using collect.py first."
        )

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=str)
    unique_labels = sorted(list(set(Y_list)))

    return X, Y, unique_labels


# ── Persistence (.npz) ───────────────────────────────────────────────────────
def save_model(
    X: np.ndarray,
    Y: np.ndarray,
    labels: List[str],
    path: str = MODEL_PATH,
) -> None:
    """Save training matrices and label dictionary into a compressed .npz archive.

    Args:
        X: (N, D) training feature matrix.
        Y: (N,) ground-truth label array.
        labels: List of unique registered label strings.
        path: Output file path (.npz).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        Y=Y,
        labels=np.array(labels, dtype=str),
    )
    logger.info("Successfully saved model archive to %s", path)


def load_model(path: str = MODEL_PATH) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and unpack trained model matrices and label index from a .npz file.

    Args:
        path: Path to the .npz archive.

    Returns:
        Tuple of (X, Y, labels).

    Raises:
        FileNotFoundError: If the model file is not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Please run train.py to generate and save the model first."
        )

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    labels = data["labels"].tolist()

    return X, Y, labels
