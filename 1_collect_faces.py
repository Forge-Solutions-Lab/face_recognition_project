import cv2
import os

# 1. โหลดตัวตรวจจับใบหน้าอัตโนมัติ (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

name = input("ใส่ชื่อของบุคคล (เช่น parinya): ").strip()
if not name:
    name = "parinya"

os.makedirs(name, exist_ok=True)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = len(os.listdir(name))
max_samples = count + 30 # เก็บเพิ่มอีก 30 รูป

print(f"\n[INFO] เริ่มเก็บข้อมูลใบหน้าของ: '{name}'")
print("[INFO] ขยับใบหน้าไปมาเล็กน้อย โปรแกรมจะถ่ายภาพให้อัตโนมัติเมื่อเจอดวงหน้า")
print("[INFO] กด 'q' เพื่อออกก่อนกำหนด\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ตรวจจับใบหน้าอัตโนมัติในภาพ
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # วาดกรอบสีเขียวรอบใบหน้า
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # ครอปเฉพาะหน้า และ Resize เป็นขนาดมาตรฐาน 100x100 พิกเซล
        face_crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))

        # บันทึกภาพอัตโนมัติ
        count += 1
        cv2.imwrite(f"{name}/{count}.jpg", face_crop)
        
        cv2.putText(frame, f"Saved: {count}/{max_samples}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # หน่วงเวลานิดหน่อยเพื่อให้ภาพมีความหลากหลายของมุมหน้า
        cv2.waitKey(100)
        break # จับทีละ 1 ใบหน้าในแต่ละเฟรม

    cv2.imshow('Face Collector', frame)

    if count >= max_samples or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

print(f"[SUCCESS] บันทึกข้อมูลใบหน้าของ '{name}' เรียบร้อยแล้ว (ทั้งหมด {count} รูป)")
cap.release()
cv2.destroyAllWindows()
