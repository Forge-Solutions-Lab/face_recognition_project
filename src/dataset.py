import os
import re
import cv2
import numpy as np
from config import DATA_DIR
from src.roi import resize, flatten
IMG_EXT = (".jpg", ".jpeg", ".png")

def save_image(path, img):
    cv2.imencode(os.path.splitext(path)[1], img)[1].tofile(path)

def load_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))

def list_member_dirs(data_dir=DATA_DIR):
    if not os.path.isdir(data_dir):
        return []
    return sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("."))

def next_image_path(folder):
    nums = [int(m.group(1)) for f in list_images(folder) if (m := re.match(r"img_(\d+)", f))]
    return os.path.join(folder, f"img_{max(nums, default=0) + 1:03d}.jpg")

def label_of(dir_name):
    return dir_name.split("_")[0]

def load_member(folder):
    return [flatten(resize(load_image(os.path.join(folder, f)))) for f in list_images(folder)]

def load_dataset(data_dir=DATA_DIR):
    X, y = [], []
    for d in list_member_dirs(data_dir):
        vecs = load_member(os.path.join(data_dir, d))
        X += vecs
        y += [label_of(d)] * len(vecs)
    return np.array(X, dtype=np.float32), np.array(y)

def move_images(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in list_images(src):
        os.replace(os.path.join(src, f), next_image_path(dst))

def clear_images(folder):
    for f in list_images(folder):
        os.remove(os.path.join(folder, f))
