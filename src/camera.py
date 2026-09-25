import cv2
from config import CAMERA_INDEX

def open_camera(index=CAMERA_INDEX):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    return cap

def read_frame(cap):
    ok, frame = cap.read()
    return frame if ok else None

def read_key():
    return chr(cv2.waitKey(1) & 0xFF).lower()

def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
