"""Phase 2: Model Training and Serialization CLI.

Scans the face dataset directory, constructs feature matrices and ground-truth labels,
and persists the trained KNN model to a compressed NumPy archive.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from config import DATA_DIR, IMG_SHAPE, MODEL_PATH
from src.model import save_model, train_model


# ── Training Execution ───────────────────────────────────────────────────────
def run_training() -> None:
    """Execute the full dataset loading, matrix compilation, and model export pipeline."""
    print("=" * 60)
    print("🧠 กำลังเริ่มกระบวนการเทรนโมเดล KNN (Face Recognition Training)")
    print("=" * 60)
    print(f"📂 ไดเรกทอรีข้อมูล: {DATA_DIR}")
    print(f"📐 ขนาดภาพมาตรฐาน: {IMG_SHAPE[0]}x{IMG_SHAPE[1]} pixels (Grayscale)")

    if not os.path.exists(DATA_DIR):
        print(f"❌ ไม่พบโฟลเดอร์ {DATA_DIR} กรุณารัน collect.py ก่อน")
        sys.exit(1)

    start_time = time.time()

    try:
        X, Y, labels = train_model(data_dir=DATA_DIR, target_size=IMG_SHAPE)
    except Exception as err:
        print(f"❌ เกิดข้อผิดพลาดในการโหลดข้อมูล: {err}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # Display dataset statistics
    print("-" * 60)
    print(f"📊 สรุปสถิติข้อมูลที่โหลดได้:")
    print(f"   - จำนวนรูปทั้งหมด (Total Samples): {X.shape[0]} รูป")
    print(f"   - มิติของ Feature Vector (Dimensions): {X.shape[1]} features")
    print(f"   - จำนวนคลาส (Total Classes): {len(labels)} คน")
    print("-" * 60)
    print("👥 รายละเอียดแต่ละคลาส:")
    for label in labels:
        count = int(np.sum(Y == label))
        print(f"   • {label}: {count} รูป")
    print("-" * 60)

    # Save model
    save_model(X, Y, labels, path=MODEL_PATH)
    print(f"💾 บันทึกโมเดลเรียบร้อยที่: {MODEL_PATH}")
    print(f"⏱️ ใช้เวลาเทรนทั้งหมด: {elapsed:.3f} วินาที")
    print("=" * 60)
    print("🎉 พร้อมสำหรับการรัน Real-time Prediction (python predict.py) แล้ว!")


# ── Main Entrypoint ──────────────────────────────────────────────────────────
def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
