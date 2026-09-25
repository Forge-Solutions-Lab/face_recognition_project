import os
import cv2
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faces")
ROI_RATIO, IMG_SIZE = 0.45, 64
CAMERA = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

def get_face(frame):
    h, w = frame.shape[:2]
    s = int(min(h, w) * ROI_RATIO)
    x, y = (w - s) // 2, (h - s) // 2
    face = cv2.cvtColor(frame[y:y + s, x:x + s], cv2.COLOR_BGR2GRAY)
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), (x, y, s)

def show(frame, face, box, color, *texts):
    (x, y, s), scale, t = box, frame.shape[0] / 600, max(2, frame.shape[0] // 300)
    cv2.rectangle(frame, (x, y), (x + s, y + s), color, t)
    for i, text in enumerate(texts):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
        org = (x, y - int((len(texts) - i - 0.6) * 40 * scale))
        cv2.rectangle(frame, (org[0] - 4, org[1] - th - 8), (org[0] + tw + 4, org[1] + 8), (0, 0, 0), -1)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, t)
    cv2.imshow("Camera", frame)
    cv2.imshow("ROI", cv2.resize(face, (200, 200), interpolation=cv2.INTER_NEAREST))
    return chr(cv2.waitKey(1) & 0xFF).lower()

if __name__ == "__main__":
    cap, faces = cv2.VideoCapture(0, CAMERA), []
    ret, frame = cap.read()
    while ret:
        frame = cv2.flip(frame, 1)
        face, box = get_face(frame)
        key = show(frame, face, box, (0, 165, 255), f"Saved: {len(faces)}", "S = Save | Q = Finish")
        if key == "s":
            faces.append(face)
        if key in ("q", "\x1b"):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if faces:
        sid, en, th = [" ".join(input(p).split()) for p in ("รหัสนักศึกษา: ", "ชื่อ (อังกฤษ): ", "ชื่อ (ไทย): ")]
        folder = os.path.join(DATA_DIR, f"{sid}_{en}_{th}")
        os.makedirs(folder, exist_ok=True)
        for i, face in enumerate(faces, len(os.listdir(folder)) + 1):
            cv2.imencode(".jpg", face)[1].tofile(os.path.join(folder, f"img_{i:03d}.jpg"))
        print(f"บันทึก {len(faces)} รูปที่ {folder}")
