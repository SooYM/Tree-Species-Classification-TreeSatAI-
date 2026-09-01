"""
PureForest Baseline Random Forest Classifier Training Pipeline.

This script trains a baseline Random Forest classifier on handcrafted multi-spectral
features extracted from 4-channel VHR aerial imagery patches (NIR, Red, Green, Blue).
It subsamples the dataset to prevent class imbalance, extracts a 42-dimensional
feature vector per patch, trains the ensemble model, evaluates validation accuracy,
and serializes the trained model to `rf_model.joblib`.

Features Extracted per Patch (42 dimensions total):
    1. Spectral Band Statistics (8 features):
       - Mean and standard deviation of NIR, Red, Green, and Blue bands.
    2. Vegetation Index Statistics (2 features):
       - Mean and standard deviation of Normalized Difference Vegetation Index (NDVI).
       - NDVI = (NIR - Red) / (NIR + Red + 1e-8)
    3. Spectral Distribution Histograms (32 features):
       - 8-bin intensity histogram per channel normalized by total pixel count (250x250).

Usage:
    Run standalone training:
        $ python3 train_classifier.py
"""

import os
import random
from typing import Dict, List, Optional, Tuple

from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Dynamically resolve project directory
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH: str = os.path.join(BASE_DIR, "rf_model.joblib")

# Target semantic class names indexed by class ID (0-12)
CLASS_MAPPING: Dict[int, str] = {
    0: "Deciduous oak",
    1: "Evergreen oak",
    2: "Beech",
    3: "Chestnut",
    4: "Black locust",
    5: "Maritime pine",
    6: "Scotch pine",
    7: "Black pine",
    8: "Aleppo pine",
    9: "Fir",
    10: "Spruce",
    11: "Larch",
    12: "Douglas"
}


def extract_features(img_path: str) -> Optional[List[float]]:
    """Extract a 42-dimensional spectral and textural feature vector from a 4-channel TIFF image.

    Opens the specified TIFF file, verifies that it possesses 4 channels (NIR, Red, Green, Blue),
    computes basic summary statistics, vegetation index (NDVI), and 8-bin normalized color histograms.

    Args:
        img_path: Absolute or relative file path to the 4-channel TIFF image.

    Returns:
        Optional[List[float]]: 42-dimensional feature vector, or None if file cannot be read or
            channel dimensions are invalid.

    Example:
        >>> feats = extract_features("imagery-Fagus_sylvatica/train/TRAIN-Fagus_sylvatica-C2-1_1_1.tiff")
        >>> if feats is not None:
        ...     print(f"Extracted {len(feats)} features")
    """
    try:
        with Image.open(img_path) as img:
            arr = np.array(img)  # Expected shape: (250, 250, 4)
            if arr.ndim != 3 or arr.shape[2] < 4:
                return None

            # Channel order: Band 0=NIR, Band 1=Red, Band 2=Green, Band 3=Blue
            nir = arr[:, :, 0].astype(np.float32)
            red = arr[:, :, 1].astype(np.float32)
            green = arr[:, :, 2].astype(np.float32)
            blue = arr[:, :, 3].astype(np.float32)

            # 1. Summary statistics per channel (8 features)
            features: List[float] = [
                float(np.mean(nir)), float(np.std(nir)),
                float(np.mean(red)), float(np.std(red)),
                float(np.mean(green)), float(np.std(green)),
                float(np.mean(blue)), float(np.std(blue))
            ]

            # 2. Normalized Difference Vegetation Index (NDVI) calculation (2 features)
            # Epsilon is added to avoid division by zero in zero-reflectance shadow regions.
            denom = nir + red
            denom[denom == 0] = 1e-8
            ndvi = (nir - red) / denom
            features.extend([float(np.mean(ndvi)), float(np.std(ndvi))])

            # 3. 8-bin intensity histogram per channel (32 features)
            # Normalized by total pixel count (250 * 250 = 62,500)
            total_pixels = float(arr.shape[0] * arr.shape[1])
            for ch in [nir, red, green, blue]:
                hist, _ = np.histogram(ch, bins=8, range=(0, 255))
                hist_normalized = hist / total_pixels
                features.extend([float(val) for val in hist_normalized.tolist()])

            return features
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None


def main() -> None:
    """Execute the Random Forest training, evaluation, and model serialization workflow."""
    species_dirs = [d for d in os.listdir(BASE_DIR) if d.startswith("imagery-")]

    train_files: List[Tuple[str, int]] = []
    val_files: List[Tuple[str, int]] = []

    # Traverse all available species directories and collect file paths
    for s_dir in species_dirs:
        species_path = os.path.join(BASE_DIR, s_dir)

        # Ingest Training Split
        train_path = os.path.join(species_path, "train")
        if os.path.exists(train_path):
            for f in os.listdir(train_path):
                if f.endswith(".tiff") or f.endswith(".tif"):
                    parts = f.split("-")
                    if len(parts) >= 3 and parts[2].startswith("C"):
                        class_id = int(parts[2][1:])
                        train_files.append((os.path.join(train_path, f), class_id))

        # Ingest Validation Split
        val_path = os.path.join(species_path, "val")
        if os.path.exists(val_path):
            for f in os.listdir(val_path):
                if f.endswith(".tiff") or f.endswith(".tif"):
                    parts = f.split("-")
                    if len(parts) >= 3 and parts[2].startswith("C"):
                        class_id = int(parts[2][1:])
                        val_files.append((os.path.join(val_path, f), class_id))

    print(f"Total available training files: {len(train_files)}")
    print(f"Total available validation files: {len(val_files)}")

    if not train_files:
        print("No training images found. Ensure dataset folders ('imagery-*') are present.")
        return

    # Subsample to cap class imbalance: maximum 800 train and 200 val per semantic class
    train_by_class: Dict[int, List[str]] = {}
    for path, cid in train_files:
        train_by_class.setdefault(cid, []).append(path)

    val_by_class: Dict[int, List[str]] = {}
    for path, cid in val_files:
        val_by_class.setdefault(cid, []).append(path)

    subsampled_train: List[Tuple[str, int]] = []
    subsampled_val: List[Tuple[str, int]] = []

    random.seed(42)
    for cid in range(13):
        paths_train = train_by_class.get(cid, [])
        if paths_train:
            sampled_train = random.sample(paths_train, min(len(paths_train), 800))
            for p in sampled_train:
                subsampled_train.append((p, cid))

        paths_val = val_by_class.get(cid, [])
        if paths_val:
            sampled_val = random.sample(paths_val, min(len(paths_val), 200))
            for p in sampled_val:
                subsampled_val.append((p, cid))

    print(f"Sampled training files: {len(subsampled_train)}")
    print(f"Sampled validation files: {len(subsampled_val)}")

    # Extract training features
    print("Extracting training features...")
    x_train: List[List[float]] = []
    y_train: List[int] = []
    for i, (path, cid) in enumerate(subsampled_train):
        feats = extract_features(path)
        if feats is not None:
            x_train.append(feats)
            y_train.append(cid)
        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1}/{len(subsampled_train)} train files")

    # Extract validation features
    print("Extracting validation features...")
    x_val: List[List[float]] = []
    y_val: List[int] = []
    for i, (path, cid) in enumerate(subsampled_val):
        feats = extract_features(path)
        if feats is not None:
            x_val.append(feats)
            y_val.append(cid)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(subsampled_val)} val files")

    x_train_arr = np.array(x_train)
    y_train_arr = np.array(y_train)
    x_val_arr = np.array(x_val)
    y_val_arr = np.array(y_val)

    # Train Random Forest Classifier with 200 estimators and parallel threading
    print("Training Random Forest Classifier (n_estimators=200, max_depth=20)...")
    classifier = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    classifier.fit(x_train_arr, y_train_arr)

    # Evaluate validation metrics
    predictions = classifier.predict(x_val_arr)
    accuracy = accuracy_score(y_val_arr, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    target_names = [CLASS_MAPPING[i] for i in sorted(CLASS_MAPPING.keys()) if i in y_val_arr]
    print(classification_report(y_val_arr, predictions, target_names=target_names))

    # Persist model
    joblib.dump(classifier, MODEL_SAVE_PATH)
    print(f"Model successfully saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
