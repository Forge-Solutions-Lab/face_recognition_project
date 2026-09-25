import os
from config import DATA_DIR, PENDING_DIR, COLOR_BOX, COLOR_TEXT
from members import register_member, member_dir_name
from src.camera import open_camera, read_frame, read_key, release_camera
from src.roi import mirror, roi_box, preprocess
from src.display import draw_box, draw_text, show_windows
from src.dataset import save_image, next_image_path, list_images, move_images, clear_images
from src.form import ask_member_process

def draw_overlay(frame, box, count):
    draw_box(frame, box, COLOR_BOX)
    draw_text(frame, f"Saved: {count}", (20, 30), COLOR_TEXT)
    draw_text(frame, "S = Save | Q = Finish", (20, 60), COLOR_TEXT)

def save_sample(roi):
    path = next_image_path(PENDING_DIR)
    save_image(path, roi)
    print(f"บันทึก {os.path.basename(path)}")

def collect_loop(cap):
    count = len(list_images(PENDING_DIR))
    while (frame := read_frame(cap)) is not None:
        frame = mirror(frame)
        box = roi_box(frame)
        roi = preprocess(frame, box)
        draw_overlay(frame, box, count)
        show_windows(frame, roi)
        key = read_key()
        if key == "s":
            save_sample(roi)
            count += 1
        if key in ("q", "\x1b"):
            return count
    return count

def capture():
    cap = open_camera()
    try:
        return collect_loop(cap)
    finally:
        release_camera(cap)

def register(member):
    move_images(PENDING_DIR, os.path.join(DATA_DIR, member_dir_name(member)))
    register_member(member)
    print(f"ลงทะเบียน {member['id']} {member['name']} เรียบร้อย")

def main():
    os.makedirs(PENDING_DIR, exist_ok=True)
    count = capture()
    if count == 0:
        return print("ไม่ได้ถ่ายรูป ยกเลิก")
    member = ask_member_process(count)
    if member is None:
        clear_images(PENDING_DIR)
        return print("ยกเลิก ลบรูปที่ถ่ายไว้แล้ว")
    register(member)

if __name__ == "__main__":
    main()
