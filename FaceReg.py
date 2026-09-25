import glob
import os
import cv2
import numpy as np
from FaceTrain import DATA_DIR, CAMERA, get_face, show
K, CONF_THRESH, DIST_THRESH = 3, 0.6, 12.0

def knn(X, y, z, k=K):
    d = np.sqrt(np.sum((X - z) ** 2, axis=1))
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)], idx, d

def load_data():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.jpg")))
    X = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE).flatten() / 255.0 for p in paths]
    return np.array(X, dtype=np.float32), np.array([os.path.basename(os.path.dirname(p)) for p in paths])

if __name__ == "__main__":
    X, y = load_data()
    if len(X) == 0:
        raise SystemExit("ไม่พบรูป กรุณารัน FaceTrain.py ก่อน")
    cap, last = cv2.VideoCapture(0, CAMERA), None
    ret, frame = cap.read()
    while ret:
        frame = cv2.flip(frame, 1)
        face, box = get_face(frame)
        label, idx, d = knn(X, y, face.flatten() / 255.0)
        conf, dist = np.mean(y[idx] == label), d[idx].mean()
        sid, en, th = label.split("_", 2)
        ok = conf >= CONF_THRESH and dist <= DIST_THRESH
        color, text, msg = ((0, 220, 0), f"{sid} {en}", f"{sid} {th}") if ok else ((0, 0, 255), "Unknown", "Unknown")
        if msg != last:
            print("พบ:", msg)
        last = msg
        if show(frame, face, box, color, text, f"conf {conf:.2f} | dist {dist:.2f}") in ("q", "\x1b"):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
