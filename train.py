import sys
import numpy as np
from config import K_NEIGHBORS, TEST_RATIO, MODEL_PATH
from members import find_member
from src.dataset import load_dataset
from src.model import save_model, split_data
from src.evaluate import predict_batch, accuracy, k_distances
def print_summary(X, y):
    print(f"รูปทั้งหมด: {X.shape[0]} | Features: {X.shape[1]} | คน: {len(set(y))}")
    for label, count in zip(*np.unique(y, return_counts=True)):
        m = find_member(label)
        print(f"  {label} {m['name'] if m else '?'}: {count} รูป")
def evaluate_model(X, y):
    X_tr, y_tr, X_te, y_te = split_data(X, y, TEST_RATIO)
    return predict_batch(X_tr, y_tr, X_te, min(K_NEIGHBORS, len(X_tr))), y_te
def print_evaluation(results, y_te):
    if not results:
        return print("รูปน้อยเกินไป วัด accuracy ไม่ได้")
    print(f"Accuracy: {accuracy(results, y_te) * 100:.1f}% (ทดสอบ {len(y_te)} รูป)")
    print(f"แนะนำ DIST_THRESH ≈ {k_distances(results).max():.2f}")
def main():
    X, y = load_dataset()
    if len(X) == 0:
        sys.exit("ไม่พบรูป กรุณารัน collect.py ก่อน")
    print_summary(X, y)
    print_evaluation(*evaluate_model(X, y))
    save_model(X, y)
    print(f"บันทึกโมเดลที่ {MODEL_PATH}")
if __name__ == "__main__":
    main()
