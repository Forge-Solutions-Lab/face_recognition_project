import cv2
from config import ROI_SIZE, WINDOW_MAIN, WINDOW_ROI
def draw_box(frame, box, color, thickness=2):
    x, y, s = box
    cv2.rectangle(frame, (x, y), (x + s, y + s), color, thickness)
def draw_text(frame, text, pos, color, scale=0.6):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
def enlarge(img, size=ROI_SIZE):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
def show_windows(frame, roi):
    cv2.imshow(WINDOW_MAIN, frame)
    cv2.imshow(WINDOW_ROI, enlarge(roi))
