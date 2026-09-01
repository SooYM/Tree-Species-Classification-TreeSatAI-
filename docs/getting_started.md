# Getting Started Guide

This guide walks you through setting up your environment, running the PureForest web application, utilizing the verification sandbox, and training custom models.

---

## 1. Prerequisites & Environment Setup

Ensure you have Python 3.8 or higher installed on your system.

### Step 1: Create and Activate a Virtual Environment
```bash
# Navigate to project root
cd "Forestation 2"

# Create a clean virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Launching the Web Server

Start the multithreaded application server:

```bash
python3 app.py
```

- By default, the server starts on `http://localhost:8080`.
- To run on a custom port, set the `PORT` environment variable:
  ```bash
  PORT=9000 python3 app.py
  ```

Open your web browser and navigate to `http://localhost:8080`.

---

## 3. Web Dashboard User Guide

### 3.1 Overview & Statistics Tab
- Displays the global territorial partition statistics across France (339 km², 40 departments, 135,569 patches).
- Displays the **13 Target Semantic Classes** and their constituent species breakdown.
- Highlights the **Model Performance Dashboard** showing live Overall Accuracy, Macro F1-Score, Macro Precision, and Macro Recall.

### 3.2 Accuracy Verifier & Sandbox Tab
- **Drag-and-Drop Ingestion**:
  1. Open your local file explorer and navigate to any species directory test folder (e.g. `imagery-Fagus_sylvatica/test/` or `imagery-Abies_alba/test/`).
  2. Select one or more `.tiff` image files and drag them directly onto the dropzone.
  3. The server classifies each patch in real-time, displays prediction badges, renders True Color and False Color CIR views, and updates the confusion matrix.
- **Bulk Split Evaluation**:
  1. In the sidebar, select the target split (`Test Split` or `Train Split`).
  2. Choose a sample size (e.g., `1,000 random patches` or `All patches`).
  3. Click **Run Bulk Evaluation**.
  4. The background daemon thread evaluates the dataset without freezing the UI, displaying live progress.

### 3.3 Analytics & Visualization Tools
- **True Color vs. False Color (CIR) Toggle**:
  - Click **True Color** to view standard RGB (Red, Green, Blue bands).
  - Click **False Color (CIR)** to view Color-Infrared (NIR mapped to Red channel), highlighting vegetative vitality.
- **Interactive Confusion Matrix**:
  - Rows represent Ground Truth (GT) classes; columns represent Predictions (Pred).
  - Hover over any cell to see exact patch counts and class pairings.
- **Per-Class Metrics**:
  - Detailed breakdown of Support, TP, TN, FP, FN, Precision, Recall, F1-Score, and Intersection over Union (IoU) per semantic class.
- **Misclassified Patches & Lightbox**:
  - Click the **Misclassified List** subtab to review instances where model predictions deviated from ground truth.
  - Click **View Image** or any row to open the side-by-side high-resolution Lightbox viewer.

---

## 4. Training Models from Scratch

### 4.1 Training the Deep Learning Model (EfficientNetV2-S)
To train the 4-channel adapted EfficientNetV2-S network on your dataset:

```bash
# Train for 20 epochs from scratch
python3 train_efficientnet.py --epochs 20

# Resume training from an existing checkpoint
python3 train_efficientnet.py --epochs 20 --resume --checkpoint checkpoint.pth

# Train with reset validation tracking
python3 train_efficientnet.py --epochs 30 --resume --reset-best-acc
```

- Best model weights will be saved automatically to `efficientnet_v2_forest.pth`.
- Progress and metrics will be logged in real-time to `training_log.txt`.

### 4.2 Training the Baseline Random Forest Model
To train the handcrafted 42-feature Random Forest model:

```bash
python3 train_classifier.py
```

- Subsamples up to 800 training files and 200 validation files per class.
- Trains 200 estimators across all CPU cores.
- Saves the trained model to `rf_model.joblib`.
