"""Face Recognition Core Library Package."""

from src.camera import open_camera, read_frame, release_camera
from src.roi import crop_roi, draw_roi_box, flatten, resize_frame, to_grayscale
from src.knn import compute_distances, knn_predict
from src.model import load_model, save_model, train_model

__all__ = [
    "open_camera",
    "read_frame",
    "release_camera",
    "draw_roi_box",
    "crop_roi",
    "to_grayscale",
    "resize_frame",
    "flatten",
    "compute_distances",
    "knn_predict",
    "train_model",
    "save_model",
    "load_model",
]
