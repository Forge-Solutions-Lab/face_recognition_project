import os
import sys
from config import K_NEIGHBORS, CONFIDENCE_THRESH, DIST_THRESH, MODEL_PATH, COLOR_OK, COLOR_UNKNOWN, COLOR_TEXT
from members import find_member
from src.camera import open_camera, read_frame, read_key, release_camera
from src.roi import mirror, roi_box, preprocess, flatten
from src.display import draw_box, draw_text, show_windows
from src.knn import knn_predict, vote_confidence, mean_distance
from src.model import load_model
def load_or_exit():
    if not os.path.exists(MODEL_PATH):
        sys.exit("ไม่พบโมเดล กรุณารัน train.py ก่อน")
    return load_model()
def recognize(X, y, roi):
    label, k_indices, distances = knn_predict(X, y, flatten(roi), min(K_NEIGHBORS, len(X)))
    return label, vote_confidence(y, k_indices, label), mean_distance(distances, k_indices)
def is_match(conf, dist):
    return conf >= CONFIDENCE_THRESH and dist <= DIST_THRESH
def member_text(label, matched, field):
    m = find_member(label)
    if not matched:
        return "Unknown"
    return f"{label} {m[field]}" if m else str(label)
def draw_result(frame, box, text, conf, dist, color):
    x, y, _ = box
    draw_box(frame, box, color)
    draw_text(frame, text, (x, y - 35), color, 0.8)
    draw_text(frame, f"conf {conf:.2f} | dist {dist:.2f}", (x, y - 10), COLOR_TEXT, 0.5)
def announce(text, last):
    if text != last:
        print(f"พบ: {text}")
    return text
def predict_loop(cap, X, y):
    last = None
    while (frame := read_frame(cap)) is not None:
        frame = mirror(frame)
        box = roi_box(frame)
        roi = preprocess(frame, box)
        label, conf, dist = recognize(X, y, roi)
        matched = is_match(conf, dist)
        draw_result(frame, box, member_text(label, matched, "name_en"), conf, dist, COLOR_OK if matched else COLOR_UNKNOWN)
        last = announce(member_text(label, matched, "name"), last)
        show_windows(frame, roi)
        if read_key() in ("q", "\x1b"):
            break
def main():
    X, y = load_or_exit()
    print(f"โหลดโมเดล: {len(X)} รูป | กด Q เพื่อออก")
    cap = open_camera()
    try:
        predict_loop(cap, X, y)
    finally:
        release_camera(cap)
if __name__ == "__main__":
    main()
