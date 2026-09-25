import cv2
import numpy as np
import os

# ==============================================================================
# กำหนดค่าพารามิเตอร์และขนาดมาตรฐาน
# ==============================================================================
FRAME_W, FRAME_H = 640, 480
FACE_W, FACE_H = 140, 180  # ขนาดของใบหน้า (กว้าง 140 x สูง 180 พิกเซล)
X1 = (FRAME_W - FACE_W) // 2
Y1 = (FRAME_H - FACE_H) // 2
X2 = X1 + FACE_W
Y2 = Y1 + FACE_H
EXPECTED_SIZE = FACE_W * FACE_H

# โหลด Haar Cascade แบบปลอดภัย (ถ้ามี)
face_cascade = None
try:
    if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
except Exception:
    face_cascade = None


def draw_tech_bracket(img, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, length=20):
    """วาดกรอบเล็งใบหน้าแบบ Sci-Fi สวยงาม ชัดเจน"""
    # วาดมุมทั้ง 4 ด้าน
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1 - length, y2), color, thickness)

    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    # วาดกรอบเส้นบางรอบนอก
    cv2.rectangle(img, (x1, y1), (x2, y2), (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3)), 1)

    # จุดกากบาทตรงกลาง (Center Crosshair)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.line(img, (cx - 10, cy), (cx + 10, cy), color, 1)
    cv2.line(img, (cx, cy - 10), (cx, cy + 10), color, 1)


def create_dual_dashboard(color_frame, face_crop_gray, title="FACE RECOGNITION SYSTEM", info_lines=None, flash=False):
    """สร้าง Dashboard คู่ (ฝั่งซ้าย: กล้องสีสด | ฝั่งขวา: พรีวิวใบหน้าขาวดำ + สถิติข้อมูล)"""
    canvas = np.zeros((540, 980, 3), dtype=np.uint8)
    canvas[:] = (25, 28, 36)  # Dark Slate background

    if flash:
        color_frame = cv2.addWeighted(color_frame, 0.4, np.full_like(color_frame, 255), 0.6, 0)

    # วางกล้องสี (ฝั่งซ้าย)
    canvas[45:45 + FRAME_H, 20:20 + FRAME_W] = color_frame

    # จัดการพรีวิวใบหน้าขาวดำ (ฝั่งขวา)
    right_panel_x = 680
    face_preview = cv2.resize(face_crop_gray, (220, 260))
    face_preview_bgr = cv2.cvtColor(face_preview, cv2.COLOR_GRAY2BGR)

    canvas[70:70 + 260, right_panel_x:right_panel_x + 220] = face_preview_bgr
    cv2.rectangle(canvas, (right_panel_x - 2, 68), (right_panel_x + 222, 70 + 262), (0, 200, 255), 2)
    cv2.putText(canvas, "CROPPED GRAYSCALE ROI", (right_panel_x + 10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # แถบหัวข้อด้านบน (Header Bar)
    cv2.rectangle(canvas, (0, 0), (980, 35), (15, 18, 24), -1)
    cv2.putText(canvas, title, (25, 24), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 200), 2)

    # แผงแสดงข้อมูลสถานะ (Info Panel)
    cv2.rectangle(canvas, (right_panel_x - 10, 345), (right_panel_x + 270, 510), (35, 40, 50), -1)
    cv2.rectangle(canvas, (right_panel_x - 10, 345), (right_panel_x + 270, 510), (60, 70, 85), 1)

    if info_lines:
        y_text = 375
        for label, val, text_color in info_lines:
            cv2.putText(canvas, f"{label}:", (right_panel_x, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(canvas, str(val), (right_panel_x + 90, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)
            y_text += 28

    cv2.putText(canvas, "[S] Save Face Image   |   [Q] Quit", (25, 532),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

    return canvas


# ==============================================================================
# ส่วนที่ 1: เก็บข้อมูลใบหน้า (Data Collection)
# ==============================================================================
def collect_face_data():
    name = input("\nกรุณาใส่ชื่อเจ้าของใบหน้า (เช่น parinya): ").strip()
    if not name:
        print("[ข้อผิดพลาด] ชื่อต้องไม่เป็นค่าว่าง")
        return

    os.makedirs(name, exist_ok=True)
    existing_files = [f for f in os.listdir(name) if f.lower().endswith(('.jpg', '.png'))]
    count = len(existing_files) + 1

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ข้อผิดพลาด] ไม่สามารถเปิดกล้องได้")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    flash_frames = 0
    print(f"\n--- เริ่มเก็บข้อมูลของ '{name}' ---")
    print("จัดตำแหน่งใบหน้าให้อยู่ในกรอบ -> กด 's' เพื่อบันทึกรูป | กด 'q' เพื่อเสร็จสิ้น")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # กระจกเงา (Mirror View)
        if frame.shape[0] != FRAME_H or frame.shape[1] != FRAME_W:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_crop_gray = gray[Y1:Y2, X1:X2]

        bracket_color = (0, 255, 120)
        sensor_status = "Target Ready"
        if face_cascade is not None:
            try:
                detected_faces = face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(80, 80))
                if len(detected_faces) > 0:
                    sensor_status = "Face Detected"
                else:
                    bracket_color = (0, 160, 255)
                    sensor_status = "Align Face..."
            except Exception:
                pass

        draw_tech_bracket(frame, X1, Y1, X2, Y2, color=bracket_color, thickness=2)

        info_lines = [
            ("User", name, (0, 255, 200)),
            ("Saved", f"{count - 1} imgs", (0, 255, 100)),
            ("Sensor", sensor_status, bracket_color),
            ("Size", f"{FACE_W}x{FACE_H} px", (200, 200, 200))
        ]

        dashboard = create_dual_dashboard(
            color_frame=frame,
            face_crop_gray=face_crop_gray,
            title=f"COLLECTING DATA: {name.upper()}",
            info_lines=info_lines,
            flash=(flash_frames > 0)
        )

        if flash_frames > 0:
            flash_frames -= 1

        cv2.imshow('Face Recognition Studio', dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            file_path = os.path.join(name, f"{count}.jpg")
            cv2.imwrite(file_path, face_crop_gray)
            print(f" [บันทึกสำเร็จ] ภาพที่ {count} -> {file_path}")
            count += 1
            flash_frames = 3
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[เสร็จสิ้น] บันทึกภาพของ '{name}' ทั้งหมด {count - 1} ภาพเรียบร้อยแล้ว\n")


# ==============================================================================
# ส่วนที่ 2: โหลดและเตรียมข้อมูล (Data Preparation)
# ==============================================================================
def load_and_prepare_data():
    X_list, y_list = [], []

    for folder in os.listdir():
        if os.path.isdir(folder) and not folder.startswith('.'):
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    if img.shape != (FACE_H, FACE_W):
                        img = cv2.resize(img, (FACE_W, FACE_H))

                    flat_vec = img.flatten().astype(np.float32) / 255.0
                    X_list.append(flat_vec)
                    y_list.append(folder)

    return np.array(X_list), np.array(y_list)


# ==============================================================================
# ส่วนที่ 3: KNN + ทำนายผลแบบเรียลไทม์ (Real-time Recognition)
# ==============================================================================
# ปรับ Threshold ให้เหมาะสมกับขนาด 25,200 พิกเซล (1200.0)
def knn(X, y, z, k=3, threshold=1200.0):
    """
    คำนวณ Squared Euclidean Distance: d = sum((X - z)^2)
    ถ้าความต่าง min_dist > threshold จะระบุว่าเป็น Unknown
    """
    d = np.sum((X - z) ** 2, axis=1)
    min_dist = np.min(d)

    # ดึง k ลำดับแรก
    k_actual = min(k, len(X))
    idx = np.argsort(d)[:k_actual]
    classes, votes = np.unique(y[idx], return_counts=True)
    best_match = classes[np.argmax(votes)]

    # เช็คเงื่อนไขคนแปลกหน้า (ถ้าระยะห่างเกิน 1200.0 ถือว่าไม่ตรง)
    if min_dist > threshold:
        return "Unknown", min_dist

    return best_match, min_dist


def real_time_recognition(k=3):
    print("\n--- กำลังโหลดข้อมูลภาพจากโฟลเดอร์... ---")
    X, y = load_and_prepare_data()

    if len(X) == 0:
        print("[ข้อผิดพลาด] ไม่พบข้อมูลใบหน้าในระบบ กรุณาเลือกเมนู 1 เพื่อเก็บข้อมูลก่อน")
        return

    print(f"[สำเร็จ] โหลดข้อมูลทั้งหมด {len(X)} ภาพ | รายชื่อในระบบ: {list(set(y))}")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ข้อผิดพลาด] ไม่สามารถเปิดกล้องได้")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    print("\n--- เริ่มต้นสแกนและตรวจจับใบหน้าแบบ Real-time (กด 'q' เพื่อออก) ---\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # กระจกเงา
        if frame.shape[0] != FRAME_H or frame.shape[1] != FRAME_W:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_crop_gray = gray[Y1:Y2, X1:X2]
        z = face_crop_gray.flatten().astype(np.float32) / 255.0

        # ทำนายผลด้วย KNN (Threshold = 1200.0)
        predicted_name, distance = knn(X, y, z, k=k, threshold=1200.0)

        is_known = (predicted_name != "Unknown")
        theme_color = (0, 255, 100) if is_known else (0, 70, 255)

        draw_tech_bracket(frame, X1, Y1, X2, Y2, color=theme_color, thickness=2)

        cv2.putText(frame, f"ID: {predicted_name}", (X1, Y1 - 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, theme_color, 2)

        # คำนวณความมั่นใจ (Confidence %)
        confidence = max(0, min(100, int((1 - (distance / 1200.0)) * 100)))

        info_lines = [
            ("Match", predicted_name, theme_color),
            ("Distance", f"{distance:.1f}", (220, 220, 220)),
            ("Confidence", f"{confidence}%" if is_known else "--", theme_color),
            ("Status", "VERIFIED" if is_known else "UNKNOWN", theme_color),
            ("Database", f"{len(X)} samples", (180, 180, 180))
        ]

        dashboard = create_dual_dashboard(
            color_frame=frame,
            face_crop_gray=face_crop_gray,
            title="LIVE FACE SCANNER & RECOGNITION (KNN)",
            info_lines=info_lines
        )

        cv2.imshow('Face Recognition Studio', dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[เสร็จสิ้น] ปิดโปรแกรมเรียบร้อยแล้ว\n")


# ==============================================================================
# เมนูหลัก
# ==============================================================================
if __name__ == "__main__":
    while True:
        print("=" * 55)
        print("  SMART FACE RECOGNITION STUDIO (OpenCV + KNN)")
        print("=" * 55)
        print(" 1. เก็บข้อมูลใบหน้า (Data Collection + Dual View)")
        print(" 2. เริ่มสแกนและจดจำใบหน้า (Real-time Recognition)")
        print(" 3. ออกจากโปรแกรม (Exit)")
        choice = input("เลือกเมนู (1/2/3): ").strip()

        if choice == '1':
            collect_face_data()
        elif choice == '2':
            real_time_recognition(k=3)
        elif choice == '3':
            print("ปิดโปรแกรม สวัสดีครับ")
            break
        else:
            print("[แจ้งเตือน] กรุณาเลือก 1, 2 หรือ 3 เท่านั้น\n")
