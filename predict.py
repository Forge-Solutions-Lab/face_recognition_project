"""Phase 3: Real-Time Face Recognition Inference CLI.

Loads the trained KNN model, captures live camera frames, extracts and processes the
ROI in real time, computes KNN distance rankings, and displays classification results
along with confidence metrics on a dual-window interface.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    BOX_COLOR_BGR,
    BOX_COLOR_SUCCESS_BGR,
    BOX_COLOR_UNKNOWN_BGR,
    BOX_THICKNESS,
    CONFIDENCE_THRESH,
    IMG_SHAPE,
    K_NEIGHBORS,
    MODEL_PATH,
    ROI_SIZE,
    WINDOW_MAIN,
    WINDOW_ROI,
)
from src.camera import open_camera, read_frame, release_camera
from src.knn import knn_predict
from src.model import load_model
from src.roi import crop_roi, draw_roi_box, flatten, resize_frame, to_grayscale


# ── Text Drawing Helper (Supporting Thai & English Fonts) ────────────────────
def draw_text_with_thai_support(
    img_bgr: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 24,
    color_bgr: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw text onto a BGR OpenCV image with support for Thai Unicode characters.

    Attempts to use system Thai fonts (e.g. Tahoma, Sukhumvit, Ayuthaya) via PIL.
    Falls back gracefully to OpenCV putText if font rendering fails.

    Args:
        img_bgr: Source BGR image array.
        text: Text string (supports Thai characters).
        position: (x, y) coordinate tuple.
        font_size: Font size in pixels.
        color_bgr: (B, G, R) color tuple.

    Returns:
        Annotated BGR image array.
    """
    try:
        # Possible Mac / Linux / Windows Thai fonts
        candidate_fonts = [
            "/System/Library/Fonts/Supplemental/Tahoma.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/SukhumvitSet.ttc",
            "/System/Library/Fonts/Ayuthaya.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "tahoma.ttf",
            "arial.ttf",
        ]
        font: Optional[ImageFont.FreeTypeFont] = None
        for fpath in candidate_fonts:
            if os.path.exists(fpath):
                try:
                    font = ImageFont.truetype(fpath, font_size)
                    break
                except Exception:
                    continue

        if font is None:
            font = ImageFont.load_default()

        # Convert OpenCV BGR to PIL RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Convert BGR color to RGB
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text(position, text, font=font, fill=color_rgb)

        # Convert back to OpenCV BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback to OpenCV built-in Hershey font
        annotated = img_bgr.copy()
        cv2.putText(
            annotated,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size / 30.0,
            color_bgr,
            2,
        )
        return annotated


# ── Real-Time Inference Loop ─────────────────────────────────────────────────
def run_prediction() -> None:
    """Execute the live real-time face classification loop."""
    print("=" * 60)
    print("🎥 กำลังเริ่มต้นระบบตรวจจับใบหน้าแบบเรียลไทม์ (Real-time Prediction)")
    print("=" * 60)

    # 1. Load Trained Model
    try:
        X_train, y_train, labels = load_model(MODEL_PATH)
        print(f"✅ โหลดโมเดลสำเร็จ: มีตัวอย่างทั้งหมด {len(X_train)} รูป จาก {len(labels)} คลาส")
    except Exception as err:
        print(f"❌ โหลดโมเดลไม่สำเร็จ: {err}")
        print("💡 กรุณารัน train.py เพื่อสร้างโมเดลก่อน")
        sys.exit(1)

    print("📷 กำลังเปิดกล้อง...")
    print("💡 กดปุ่ม [Q] บนหน้าต่างกล้องเพื่อออกจากโปรแกรม\n")

    cap = open_camera()

    # FPS Calculation trackers
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                continue

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # 2. Extract and Preprocess ROI
            annotated_frame, box = draw_roi_box(
                frame, roi_size=ROI_SIZE, color=BOX_COLOR_BGR, thickness=BOX_THICKNESS
            )
            x, y, w, h = box

            cropped = crop_roi(frame, box)
            gray_roi = to_grayscale(cropped)
            roi_resized = resize_frame(gray_roi, size=IMG_SHAPE)
            x_query = flatten(roi_resized)

            # 3. KNN Prediction
            predicted_label, confidence = knn_predict(
                X_train, y_train, x_query, k=K_NEIGHBORS
            )

            # 4. Determine Acceptance vs Unknown Threshold
            is_recognized = confidence >= CONFIDENCE_THRESH
            box_color = BOX_COLOR_SUCCESS_BGR if is_recognized else BOX_COLOR_UNKNOWN_BGR

            # Draw dynamic colored box to reflect recognition status
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, BOX_THICKNESS)

            # Format status and label text
            if is_recognized:
                display_label = f"Match: {predicted_label}"
                conf_text = f"Confidence: {confidence * 100:.1f}% (K={K_NEIGHBORS})"
            else:
                display_label = "Unknown / Low Confidence"
                conf_text = f"Confidence: {confidence * 100:.1f}% (< {CONFIDENCE_THRESH * 100:.0f}%)"

            # Render text on frame with Thai unicode support
            annotated_frame = draw_text_with_thai_support(
                annotated_frame,
                display_label,
                (x, max(20, y - 40)),
                font_size=22,
                color_bgr=box_color,
            )
            annotated_frame = draw_text_with_thai_support(
                annotated_frame,
                conf_text,
                (x, max(45, y - 15)),
                font_size=18,
                color_bgr=(200, 200, 200),
            )

            # FPS & System HUD overlay
            fps_text = f"FPS: {fps:.1f} | Press 'Q' to Exit"
            cv2.putText(
                annotated_frame,
                fps_text,
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # 5. Show Dual Windows
            cv2.imshow(WINDOW_MAIN, annotated_frame)
            display_roi = cv2.resize(
                roi_resized, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_NEAREST
            )
            cv2.imshow(WINDOW_ROI, display_roi)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:
                print("\n🚪 สิ้นสุดการทำงาน Real-Time Prediction")
                break
    finally:
        release_camera(cap)


# ── Main Entrypoint ──────────────────────────────────────────────────────────
def main() -> None:
    run_prediction()


if __name__ == "__main__":
    main()
