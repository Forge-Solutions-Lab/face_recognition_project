import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import time

# ==========================================
# 1. ฟังก์ชัน KNN + เช็คว่าเป็นคนแปลกหน้าหรือไม่
# ==========================================
def knn(X, y, z, k=5, threshold=3000):
    # คำนวณ Euclidean distance เฉลี่ยต่อพิกเซล
    d = np.mean((X - z) ** 2, axis=1)
    min_dist = np.min(d)

    # ถ้าระยะห่างมากเกินกว่าค่า threshold ถือว่าเป็นคนแปลกหน้า (Unknown)
    if min_dist > threshold:
        return "Unknown", min_dist

    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)], min_dist

# ==========================================
# 2. ฟังก์ชันบันทึกเวลาเข้างานลง CSV
# ==========================================
attendance_file = 'attendance.csv'
logged_users = {} # เก็บเวลาที่เพิ่งบันทึกเพื่อป้องกันการบันทึกซ้ำซ้อน

def log_attendance(name):
    if name == "Unknown":
        return None

    current_time = time.time()
    # ป้องกันการบันทึกซ้ำภายใน 30 วินาที
    if name in logged_users and (current_time - logged_users[name]) < 30:
        return None

    logged_users[name] = current_time
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    file_exists = os.path.isfile(attendance_file)
    with open(attendance_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ชื่อ-นามสกุล", "วันที่", "เวลาเข้างาน"])
        writer.writerow([name, date_str, time_str])

    print(f" [บันทึกสำเร็จ] คุณ {name} เข้างานเมื่อเวลา {time_str}")
    return time_str

# ==========================================
# 3. โหลดภาพทั้งหมดเข้า Dataset
# ==========================================
X, y = [], []
for folder in os.listdir():
    if os.path.isdir(folder) and not folder.startswith('.'):
        for img_name in os.listdir(folder):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (100, 100))
                    X.append(img_resized.flatten())
                    y.append(folder)

X = np.array(X, dtype=np.float32)
y = np.array(y)

if len(X) == 0:
    print("[ERROR] ยังไม่มีข้อมูลใบหน้า! กรุณารันไฟล์ 1_collect_faces.py เพื่อเก็บใบหน้าก่อน")
    exit()

print(f"[READY] โหลดข้อมูลเรียบร้อย: ทั้งหมด {len(X)} รูป จากรายชื่อ: {list(set(y))}")

# ==========================================
# 4. เปิดกล้องตรวจจับและบันทึกเวลา Real-time
# ==========================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
last_status_msg = "Ready"

print("\n--- เริ่มต้นระบบลงเวลาเข้างาน (กด 'q' เพื่อปิดโปรแกรม) ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    for (x, y_pos, w, h) in faces:
        # ครอปและปรับขนาดเป็น 100x100
        face_crop = cv2.resize(gray[y_pos:y_pos+h, x:x+w], (100, 100))
        z = face_crop.flatten().astype(np.float32)

        # ทำนายผลด้วย KNN
        pred_name, dist = knn(X, y, z, k=5, threshold=3000)

        # ถ้าเป็นคนที่รู้จัก -> กรอบสีเขียว, ถ้าแปลกหน้า -> กรอบสีแดง
        color = (0, 255, 0) if pred_name != "Unknown" else (0, 0, 255)
        
        # บันทึกลงระบบ Attendance
        log_time = log_attendance(pred_name)
        if log_time:
            last_status_msg = f"Check-in: {pred_name} at {log_time}"

        # แสดงกรอบและชื่อ
        cv2.rectangle(frame, (x, y_pos), (x + w, y_pos + h), color, 2)
        cv2.putText(frame, f"{pred_name}", (x, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # แถบแสดงสถานะด้านบน
    cv2.rectangle(frame, (0, 0), (640, 40), (40, 40, 40), -1)
    cv2.putText(frame, f"Status: {last_status_msg}", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Smart Attendance System (Face Recognition)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
