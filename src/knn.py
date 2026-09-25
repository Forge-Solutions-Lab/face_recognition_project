import numpy as np
def knn_predict(X_train, y_train, x_new, k=3):
    distances = np.sqrt(np.sum((X_train - x_new) ** 2, axis=1))
    sorted_indices = np.argsort(distances)
    k_indices = sorted_indices[:k]
    k_nearest_labels = y_train[k_indices]
    unique_classes, counts = np.unique(k_nearest_labels, return_counts=True)
    predicted_class = unique_classes[np.argmax(counts)]
    return predicted_class, k_indices, distances
def vote_confidence(y_train, k_indices, predicted_class):
    return float(np.mean(y_train[k_indices] == predicted_class))
def mean_distance(distances, k_indices):
    return float(np.mean(distances[k_indices]))
