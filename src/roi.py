import cv2
import numpy as np
from config import ROI_SIZE, IMG_SIZE
def mirror(frame):
    return cv2.flip(frame, 1)
def roi_box(frame, size=ROI_SIZE):
    h, w = frame.shape[:2]
    return max(0, (w - size) // 2), max(0, (h - size) // 2), size
def crop(frame, box):
    x, y, s = box
    return frame[y:y + s, x:x + s]
def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def resize(img, size=IMG_SIZE):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
def preprocess(frame, box):
    return resize(to_gray(crop(frame, box)))
def flatten(img):
    return img.flatten().astype(np.float32) / 255.0
