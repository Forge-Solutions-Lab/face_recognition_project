import os
ROI_SIZE = 200
IMG_SIZE = 64
K_NEIGHBORS = 3
CONFIDENCE_THRESH = 0.6
DIST_THRESH = 10.0
TEST_RATIO = 0.2
CAMERA_INDEX = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "faces")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model.npz")
WINDOW_MAIN = "Camera"
WINDOW_ROI = "ROI"
COLOR_BOX = (0, 165, 255)
COLOR_OK = (0, 220, 0)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
