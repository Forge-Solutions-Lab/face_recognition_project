import numpy as np
from src.knn import knn_predict, mean_distance
def predict_batch(X_train, y_train, X_test, k):
    return [knn_predict(X_train, y_train, x, k) for x in X_test]
def accuracy(results, y_test):
    return float(np.mean(np.array([r[0] for r in results]) == y_test))
def k_distances(results):
    return np.array([mean_distance(r[2], r[1]) for r in results])
