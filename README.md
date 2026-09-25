# 👤 Enterprise Face Recognition System (KNN From Scratch)

A modular, enterprise-grade real-time face recognition system built in Python utilizing custom K-Nearest Neighbors (KNN) classification, dual-window OpenCV camera streams, and dynamic Region of Interest (ROI) preprocessing.

---

## 🏛️ Architecture & Project Structure

```
face_recognition_project/
├── config.py             # Central system constants & hyperparameters
├── members.py            # Extensible member roster catalog
├── requirements.txt      # Project dependencies
├── README.md             # Documentation & execution guide
├── .gitignore            # Git exclusion rules
├── src/
│   ├── __init__.py       # Package exports
│   ├── camera.py         # Camera capture lifecycle management
│   ├── roi.py            # Geometric crop, grayscale & flatten transformations
│   ├── knn.py            # Custom KNN classifier & distance matrix calculation
│   └── model.py          # Dataset ingestion & .npz persistence
├── data/
│   └── faces/            # Member sample repositories
├── collect.py            # Phase 1: Real-time face sample collector
├── train.py              # Phase 2: Feature extraction & KNN model training
└── predict.py            # Phase 3: Real-time dual-window inference HUD
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone or navigate to the project directory
cd face_recognition_project

# 2. Activate Conda / Virtual Environment
conda activate knn

# 3. Install required packages
pip install -r requirements.txt
```

---

## 🚀 Execution Workflow

### Step 1: Member Registration (`members.py`)
Add new members to the `MEMBERS` list in `members.py`:
```python
MEMBERS = [
    {"id": "6601001", "name": "สมชาย"},
    {"id": "6601002", "name": "สมหญิง"},
]
```

### Step 2: Phase 1 - Collect Face Samples (`collect.py`)
```bash
python collect.py
```
* Select a member from the interactive menu or register a new one.
* Position face inside the central orange ROI box.
* Press **`S`** to capture a snapshot (recommended: 20-30 samples per person).
* Press **`Q`** to finish.

### Step 3: Phase 2 - Train KNN Model (`train.py`)
```bash
python train.py
```
* Loads all face images from `data/faces/`.
* Converts images into 1D feature vectors ($64 \times 64 = 4,096$ dimensions).
* Saves the trained dataset to `data/model.npz`.

### Step 4: Phase 3 - Real-Time Recognition (`predict.py`)
```bash
python predict.py
```
* Opens dual-window stream:
  * **Main Camera:** Live stream with color-coded bounding box (Green for match, Red for unknown) + Thai Unicode name & ID overlay.
  * **ROI Window:** $64 \times 64$ grayscale face preview.
* Press **`Q`** to exit.
