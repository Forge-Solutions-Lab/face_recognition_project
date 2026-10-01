import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from FaceTrain import get_face

def knn(X, y, z, k=1):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]

y = []
X = []
for f in os.listdir('data'):
    if not os.path.isfile("data/"+f) and not f.startswith("."):
        for i in os.listdir("data/"+f):
            if i.endswith(".jpg"):
                x = plt.imread("data/"+f+"/"+i)
                X.append(x.flatten())
                y.append(f)
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y)
while True:
    frame, face = get_face()
    z = face.flatten().astype(float)
    label = knn(X, y, z)
    cv2.putText(frame, label, (720, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
