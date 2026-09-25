import os
import numpy as np
from config import MODEL_PATH

def save_model(X, y, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X, y=y)

def load_model(path=MODEL_PATH):
    data = np.load(path)
    return data["X"], data["y"]

def split_data(X, y, test_ratio, seed=42):
    idx = np.random.default_rng(seed).permutation(len(X))
    n_test = int(len(X) * test_ratio)
    return X[idx[n_test:]], y[idx[n_test:]], X[idx[:n_test]], y[idx[:n_test]]
