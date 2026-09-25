import os
import sys
from config import DATA_DIR, COLOR_BOX, COLOR_TEXT
from members import MEMBERS, member_dir_name
from src.camera import open_camera, read_frame, read_key, release_camera
from src.roi import mirror, roi_box, preprocess
from src.display import draw_box, draw_text, show_windows
from src.dataset import save_image, next_image_path, list_images
def print_members():
    for i, m in enumerate(MEMBERS, 1):
        print(f"[{i}] {m['id']} {m['name']}")
def choose_member():
    choice = input("เลือกหมายเลขสมาชิก: ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(MEMBERS):
        sys.exit("เลือกไม่ถูกต้อง")
    return MEMBERS[int(choice) - 1]
def member_folder(member):
    folder = os.path.join(DATA_DIR, member_dir_name(member))
    os.makedirs(folder, exist_ok=True)
    return folder
def draw_overlay(frame, box, member, count):
    draw_box(frame, box, COLOR_BOX)
    draw_text(frame, f"{member['id']} {member['name_en']} | Saved: {count}", (20, 30), COLOR_TEXT)
    draw_text(frame, "S = Save | Q = Quit", (20, 60), COLOR_TEXT)
def save_sample(folder, roi):
    path = next_image_path(folder)
    save_image(path, roi)
    print(f"บันทึก {os.path.basename(path)}")
def collect_loop(cap, member, folder):
    count = len(list_images(folder))
    while (frame := read_frame(cap)) is not None:
        frame = mirror(frame)
        box = roi_box(frame)
        roi = preprocess(frame, box)
        draw_overlay(frame, box, member, count)
        show_windows(frame, roi)
        key = read_key()
        if key == "s":
            save_sample(folder, roi)
            count += 1
        if key in ("q", "\x1b"):
            break
def main():
    print_members()
    member = choose_member()
    folder = member_folder(member)
    cap = open_camera()
    try:
        collect_loop(cap, member, folder)
    finally:
        release_camera(cap)
if __name__ == "__main__":
    main()
