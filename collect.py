"""Phase 1: Face Dataset Collection CLI.

Opens a dual-window camera interface allowing users to capture face training samples
into structured directories with real-time ROI visualization.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2

from config import (
    BOX_COLOR_BGR,
    DATA_DIR,
    IMG_SHAPE,
    ROI_SIZE,
    WINDOW_MAIN,
    WINDOW_ROI,
)
from members import format_member_dir_name, get_all_members, get_member_by_id
from src.camera import open_camera, read_frame, release_camera
from src.roi import crop_roi, draw_roi_box, resize_frame, to_grayscale


# ── Interactive Member Selection ─────────────────────────────────────────────
def select_or_create_member() -> str:
    """Prompt the operator to select an existing member or register a new one.

    Returns:
        Standardized folder name (e.g. '6601001_สมชาย').
    """
    members = get_all_members()
    print("=" * 60)
    print("📸 ระบบเก็บตัวอย่างใบหน้า (Face Dataset Collection)")
    print("=" * 60)
    print("รายชื่อสมาชิกในระบบ:")
    for idx, m in enumerate(members, start=1):
        print(f"  [{idx}] รหัส: {m['id']} | ชื่อ: {m['name']}")
    print(f"  [N] กรอกรหัสและชื่อใหม่")
    print("-" * 60)

    choice = input("เลือกหมายเลขสมาชิก หรือพิมพ์ N เพื่อเพิ่มใหม่: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(members):
        selected = members[int(choice) - 1]
        folder_name = format_member_dir_name(selected["id"], selected["name"])
    elif choice.upper() == "N":
        custom_id = input("กรอกรหัสสมาชิก (ID): ").strip()
        custom_name = input("กรอกชื่อสมาชิก (Name): ").strip()
        if not custom_id or not custom_name:
            print("❌ ข้อมูลไม่ถูกต้อง ยกเลิกการทำงาน")
            sys.exit(1)
        folder_name = format_member_dir_name(custom_id, custom_name)
    else:
        # Check if user directly entered an ID
        member = get_member_by_id(choice)
        if member:
            folder_name = format_member_dir_name(member["id"], member["name"])
        else:
            print("❌ เลือกรายการไม่ถูกต้อง ยกเลิกการทำงาน")
            sys.exit(1)

    return folder_name


# ── Collection Loop ──────────────────────────────────────────────────────────
def run_collection(folder_name: str) -> None:
    """Run real-time video capture loop to take face sample snapshots.

    Args:
        folder_name: Destination folder under DATA_DIR.
    """
    target_dir = os.path.join(DATA_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    # Count existing images in target folder
    existing_files = [f for f in os.listdir(target_dir) if f.lower().endswith((".jpg", ".png"))]
    img_counter = len(existing_files) + 1

    print(f"\n📁 บันทึกภาพลงที่: {target_dir}")
    print("📷 กำลังเปิดกล้อง...")
    print("💡 วิธีใช้งาน:")
    print("   - จัดใบหน้าให้อยู่ในกรอบสี่เหลี่ยมสีส้ม")
    print("   - กดปุ่ม [S] เพื่อถ่ายและบันทึก 1 รูป")
    print("   - กดปุ่ม [Q] เพื่อออกจากระบบ\n")

    cap = open_camera()

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                continue

            # Draw ROI bounding box on main frame
            annotated_frame, box = draw_roi_box(
                frame, roi_size=ROI_SIZE, color=BOX_COLOR_BGR, thickness=2
            )

            # Crop and preprocess ROI
            cropped = crop_roi(frame, box)
            gray_roi = to_grayscale(cropped)
            roi_resized = resize_frame(gray_roi, size=IMG_SHAPE)

            # Add status text to main window
            status_text = f"Target: {folder_name} | Saved: {img_counter - 1} imgs"
            help_text = "Press 'S' to Save | 'Q' to Quit"
            cv2.putText(
                annotated_frame,
                status_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                help_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # Show Dual Windows
            cv2.imshow(WINDOW_MAIN, annotated_frame)
            # Resize ROI display window for clear viewing
            display_roi = cv2.resize(roi_resized, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(WINDOW_ROI, display_roi)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") or key == ord("S"):
                filename = f"img_{img_counter:03d}.jpg"
                save_path = os.path.join(target_dir, filename)
                # Save the 64x64 grayscale ROI image
                cv2.imwrite(save_path, roi_resized)
                print(f"✅ บันทึกรูปที่ {img_counter}: {filename}")
                img_counter += 1
            elif key == ord("q") or key == ord("Q") or key == 27:
                print("\n🚪 สิ้นสุดการเก็บข้อมูลสำหรับสมาชิกนี้")
                break
    finally:
        release_camera(cap)


# ── Main Entrypoint ──────────────────────────────────────────────────────────
def main() -> None:
    folder_name = select_or_create_member()
    run_collection(folder_name)


if __name__ == "__main__":
    main()
