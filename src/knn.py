"""K-Nearest Neighbors (KNN) Classifier Module From Scratch.

Implements pure numpy-based distance calculation and majority-voting classification
with confidence score estimation based on the teacher's baseline algorithm.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from config import K_NEIGHBORS


# ── Distance Computation ─────────────────────────────────────────────────────
def compute_distances(X_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance from query point x_new to all training samples.

    Args:
        X_train: (N, D) matrix of N training vectors of dimension D.
        x_new: (D,) query vector.

    Returns:
        (N,) array containing Euclidean distances.
    """
    # Vectorized Euclidean distance calculation
    diff = X_train - x_new
    return np.sqrt(np.sum(diff ** 2, axis=1))


# ── KNN Classification & Confidence Estimation ───────────────────────────────
def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_new: np.ndarray,
    k: int = K_NEIGHBORS,
) -> Tuple[str, float]:
    """Predict the class label and confidence score for a new query vector.

    Args:
        X_train: (N, D) training features matrix.
        y_train: (N,) ground-truth label array.
        x_new: (D,) feature vector to classify.
        k: Number of nearest neighbors to consider (clamped to sample size).

    Returns:
        Tuple of (predicted_label, confidence_score) where confidence is in [0.0, 1.0].
    """
    if len(X_train) == 0:
        return "Unknown", 0.0

    # Ensure k does not exceed the number of available training samples
    effective_k = max(1, min(k, len(X_train)))

    # Compute distances to all training points
    distances = compute_distances(X_train, x_new)

    # Find the indices of the top-k closest samples
    sorted_indices = np.argsort(distances)
    k_nearest_indices = sorted_indices[:effective_k]
    k_nearest_labels = y_train[k_nearest_indices]

    # Majority vote
    unique_classes, vote_counts = np.unique(k_nearest_labels, return_counts=True)
    best_idx = int(np.argmax(vote_counts))
    predicted_label = str(unique_classes[best_idx])
    
    # Confidence score defined as proportion of top-k votes belonging to the winning class
    confidence = float(vote_counts[best_idx] / effective_k)

    return predicted_label, confidence
