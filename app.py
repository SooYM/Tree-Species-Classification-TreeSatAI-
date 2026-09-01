"""
PureForest Model Verification & Classification Backend Application.

This module implements a multithreaded HTTP web server that powers the
PureForest tree species verification sandbox. It provides RESTful API
endpoints for single-image inference, bulk asynchronous dataset evaluation,
cached evaluation retrieval, dataset statistics, and multi-band image rendering
(True Color RGB and False Color CIR views).

The server supports two machine learning backends:
1. PyTorch 4-Channel EfficientNetV2-S (Deep Learning, preferred)
2. Scikit-learn Random Forest Classifier (Handcrafted spectral features, fallback)

Typical Usage:
    Run the server using the default port (8080):
        $ python3 app.py

    Or specify a custom port via the PORT environment variable:
        $ PORT=9000 python3 app.py

API Endpoints Summary:
    GET  /                     -> Serves public/index.html
    GET  /api/stats            -> Returns dataset split counts and active model info
    GET  /api/last_evaluation   -> Returns cached evaluation predictions and metrics
    GET  /api/job_status       -> Polling endpoint for asynchronous bulk evaluation jobs
    GET  /api/image            -> Returns base64 encoded RGB and CIR PNGs for a given TIFF
    POST /api/predict          -> Classifies an uploaded 4-channel TIFF image
    POST /api/evaluate_test    -> Spawns background worker thread for bulk split evaluation
"""

import os
import sys
import http.server
import socketserver
import json
import base64
import io
import urllib.parse
import random
import threading
import uuid
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image
import numpy as np
import joblib

# ----------------- Configuration & Directory Paths -----------------
# Dynamically resolve repository base directory to maintain portability across environments.
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
PORT: int = int(os.environ.get("PORT", 8080))
RF_MODEL_PATH: str = os.path.join(BASE_DIR, "rf_model.joblib")
PYTORCH_MODEL_PATH: str = os.path.join(BASE_DIR, "efficientnet_v2_forest.pth")
DATA_DIR: str = BASE_DIR
CACHE_EVALUATION_PATH: str = os.path.join(DATA_DIR, "last_evaluation.json")

# Semantic class mapping (0 to 12) for the 13 PureForest target classes.
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

# Mapping of Latin species names (as formatted in file names) to semantic class IDs.
SPECIES_TO_CLASS: Dict[str, int] = {
    "Quercus_robur": 0,
    "Quercus_pubescens": 0,
    "Quercus_petraea": 0,
    "Quercus_rubra": 0,
    "Quercus_ilex": 1,
    "Fagus_sylvatica": 2,
    "Castanea_sativa": 3,
    "Robinia_pseudoacacia": 4,
    "Pinus_pinaster": 5,
    "Pinus_sylvestris": 6,
    "Pinus_nigra_laricio": 7,
    "Pinus_nigra": 7,
    "Pinus_halepensis": 8,
    "Abies_nordmanniana": 9,
    "Abies_alba": 9,
    "Picea_abies": 10,
    "Larix_decidua": 11,
    "Pseudotsuga_menziesii": 12
}

# Ground truth statistics for PureForest territorial partition splits and semantic classes.
DATASET_STATS: Dict[str, Any] = {
    "splits": {
        "Train": {"area_km2": 172.78, "patches": 69111, "polygons": 330},
        "Val": {"area_km2": 33.81, "patches": 13523, "polygons": 58},
        "Test": {"area_km2": 132.34, "patches": 52935, "polygons": 61},
    },
    "classes": [
        {"id": 0, "name": "Deciduous oak", "train": 15840, "val": 4374, "test": 27841, "species": ["Quercus robur", "Quercus pubescens", "Quercus petraea", "Quercus rubra"]},
        {"id": 1, "name": "Evergreen oak", "train": 11609, "val": 372, "test": 10380, "species": ["Quercus ilex"]},
        {"id": 2, "name": "Beech", "train": 7008, "val": 1626, "test": 4036, "species": ["Fagus sylvatica"]},
        {"id": 3, "name": "Chestnut", "train": 3337, "val": 147, "test": 200, "species": ["Castanea sativa"]},
        {"id": 4, "name": "Black locust", "train": 1663, "val": 323, "test": 317, "species": ["Robinia pseudoacacia"]},
        {"id": 5, "name": "Maritime pine", "train": 4568, "val": 960, "test": 2040, "species": ["Pinus pinaster"]},
        {"id": 6, "name": "Scotch pine", "train": 11330, "val": 2429, "test": 4506, "species": ["Pinus sylvestris"]},
        {"id": 7, "name": "Black pine", "train": 4356, "val": 942, "test": 1928, "species": ["Pinus nigra laricio", "Pinus nigra"]},
        {"id": 8, "name": "Aleppo pine", "train": 4028, "val": 233, "test": 438, "species": ["Pinus halepensis"]},
        {"id": 9, "name": "Fir", "train": 96, "val": 722, "test": 22, "species": ["Abies nordmanniana", "Abies alba"]},
        {"id": 10, "name": "Spruce", "train": 2579, "val": 627, "test": 868, "species": ["Picea abies"]},
        {"id": 11, "name": "Larch", "train": 2536, "val": 503, "test": 255, "species": ["Larix decidua"]},
        {"id": 12, "name": "Douglas", "train": 161, "val": 265, "test": 104, "species": ["Pseudotsuga menziesii"]},
    ]
}

# In-memory dictionary to track ongoing and completed asynchronous evaluation jobs.
EVAL_JOBS: Dict[str, Dict[str, Any]] = {}

# ----------------- Model Loading Setup -----------------
pytorch_model: Any = None
rf_model: Any = None
loaded_model_type: Optional[str] = None
torch_available: bool = False

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    torch_available = True

    def get_pytorch_architecture() -> torch.nn.Module:
        """Construct the modified 4-channel EfficientNetV2-S model architecture.

        Replaces the standard 3-channel (RGB) input convolution layer with a 4-channel
        convolution (Red, Green, Blue, NIR) and updates the final classifier head to
        output logits for the 13 semantic tree classes.

        Returns:
            torch.nn.Module: The configured EfficientNetV2-S model architecture.

        Example:
            >>> model = get_pytorch_architecture()
            >>> print(model.features[0][0].in_channels)
            4
        """
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        original_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        model.features[0][0] = new_conv
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 13)
        return model

    if os.path.exists(PYTORCH_MODEL_PATH):
        # Select best available accelerator: CUDA -> Apple MPS -> CPU.
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        pytorch_model = get_pytorch_architecture()
        pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=DEVICE))
        pytorch_model.to(DEVICE)
        pytorch_model.eval()
        loaded_model_type = f"PyTorch (EfficientNetV2 on {DEVICE.type.upper()})"
        print(f"Successfully loaded PyTorch (EfficientNetV2) model on {DEVICE.type.upper()}.")
except Exception as e:
    print(f"PyTorch loading bypassed or failed: {e}")

# Fallback: load Random Forest model if PyTorch model is unavailable.
if pytorch_model is None and os.path.exists(RF_MODEL_PATH):
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        loaded_model_type = "Random Forest (Baseline)"
        print("Successfully loaded Random Forest model.")
    except Exception as e:
        print(f"Error loading Random Forest model: {e}")

if loaded_model_type is None:
    print("Warning: No pre-trained model weights found. Run training script first.")


# ----------------- Helper Functions -----------------

def extract_rf_features(arr: np.ndarray) -> Optional[List[float]]:
    """Extract spectral statistics, NDVI indices, and color histograms for Random Forest.

    The PureForest TIFF image bands are ordered as:
        Band 0: Near-Infrared (NIR)
        Band 1: Red
        Band 2: Green
        Band 3: Blue

    Args:
        arr: Numpy array of shape (height, width, 4) containing pixel intensities.

    Returns:
        List[float]: A 42-dimensional feature vector containing:
            - Mean and standard deviation for each of the 4 bands (8 features)
            - Mean and standard deviation of NDVI (2 features)
            - Normalized 8-bin histograms for each of the 4 bands (32 features)
        Returns None if extraction fails or input dimensions are invalid.

    Example:
        >>> dummy_img = np.random.randint(0, 255, (250, 250, 4), dtype=np.uint8)
        >>> feats = extract_rf_features(dummy_img)
        >>> len(feats)
        42
    """
    try:
        nir = arr[:, :, 0].astype(np.float32)
        red = arr[:, :, 1].astype(np.float32)
        green = arr[:, :, 2].astype(np.float32)
        blue = arr[:, :, 3].astype(np.float32)

        features: List[float] = [
            float(np.mean(nir)), float(np.std(nir)),
            float(np.mean(red)), float(np.std(red)),
            float(np.mean(green)), float(np.std(green)),
            float(np.mean(blue)), float(np.std(blue))
        ]

        # Calculate Normalized Difference Vegetation Index (NDVI)
        # Epsilon is added to denominator to avoid division by zero in shadow/water pixels.
        denom = nir + red
        denom[denom == 0] = 1e-8
        ndvi = (nir - red) / denom
        features.extend([float(np.mean(ndvi)), float(np.std(ndvi))])

        # Compute 8-bin histograms per band normalized by patch pixel count (250x250 = 62,500)
        for ch in [nir, red, green, blue]:
            hist, _ = np.histogram(ch, bins=8, range=(0, 255))
            hist_norm = hist / (250 * 250)
            features.extend([float(val) for val in hist_norm.tolist()])

        return features
    except Exception as e:
        print(f"Error extracting RF features: {e}")
        return None


def parse_filename(filename: str) -> Tuple[Optional[int], str, str]:
    """Parse PureForest patch filename convention to extract ground-truth labels.

    PureForest filename convention:
        `{SPLIT}-{species}-C{class_id}-{patch_id}.tiff`
        e.g., 'TEST-Abies_alba-C9-407_1_108.tiff'

    Args:
        filename: Name of the TIFF image file.

    Returns:
        Tuple containing:
            - true_class_id (Optional[int]): Integer class ID (0-12), or None if unparseable.
            - true_class_name (str): Semantic class name (e.g. 'Fir', 'Beech').
            - true_species (str): Latin species name with spaces (e.g. 'Abies alba').

    Example:
        >>> cid, cname, spec = parse_filename("TEST-Fagus_sylvatica-C2-100_1_2.tiff")
        >>> cid, cname, spec
        (2, 'Beech', 'Fagus sylvatica')
    """
    name, _ = os.path.splitext(os.path.basename(filename))
    parts = name.split("-")
    true_class_id: Optional[int] = None
    true_class_name: str = "Unknown"
    true_species: str = "Unknown"

    if len(parts) >= 3:
        class_part = parts[2]
        if class_part.startswith("C") and class_part[1:].isdigit():
            true_class_id = int(class_part[1:])
            true_class_name = CLASS_MAPPING.get(true_class_id, "Unknown")
        true_species = parts[1].replace("_", " ")
    else:
        lower_name = name.lower()
        for species_key, cid in SPECIES_TO_CLASS.items():
            if species_key.lower() in lower_name:
                true_class_id = cid
                true_class_name = CLASS_MAPPING[cid]
                true_species = species_key.replace("_", " ")
                break

    return true_class_id, true_class_name, true_species


def get_all_split_files(split_name: str) -> List[str]:
    """Gather all TIFF image paths corresponding to a dataset split across species directories.

    Args:
        split_name: Name of the dataset split ('train', 'val', or 'test').

    Returns:
        List[str]: List of absolute file paths matching the split criteria.
    """
    files: List[str] = []
    if not os.path.exists(DATA_DIR):
        return files

    species_dirs = [d for d in os.listdir(DATA_DIR) if d.startswith("imagery-")]
    for s_dir in species_dirs:
        split_path = os.path.join(DATA_DIR, s_dir, split_name)
        if os.path.exists(split_path):
            for f in os.listdir(split_path):
                if f.endswith(".tiff") or f.endswith(".tif"):
                    files.append(os.path.join(split_path, f))
    return files


def find_test_file_by_name(filename: str) -> Optional[str]:
    """Search for a specific TIFF image file across dataset subdirectories.

    Args:
        filename: Base filename of the target TIFF image.

    Returns:
        Optional[str]: Absolute path if found, otherwise None.
    """
    parts = filename.split("-")
    if len(parts) >= 2:
        species = parts[1]
        for split in ["test", "train", "val"]:
            potential_path = os.path.join(DATA_DIR, f"imagery-{species}", split, filename)
            if os.path.exists(potential_path):
                return potential_path

    if os.path.exists(DATA_DIR):
        species_dirs = [d for d in os.listdir(DATA_DIR) if d.startswith("imagery-")]
        for s_dir in species_dirs:
            for split in ["test", "train", "val"]:
                path = os.path.join(DATA_DIR, s_dir, split, filename)
                if os.path.exists(path):
                    return path
    return None


# ----------------- Asynchronous Evaluation Thread Worker -----------------

def run_evaluation_job(job_id: str, file_list: List[str], split: str = "test") -> None:
    """Execute bulk dataset evaluation in a background worker thread.

    Batches image tensors for high-throughput GPU/CPU PyTorch inference or performs
    sequential feature extraction for Random Forest. Updates `EVAL_JOBS[job_id]`
    with live progress and caches results upon completion.

    Args:
        job_id: Unique UUID identifying this evaluation job.
        file_list: List of absolute file paths to evaluate.
        split: The name of the split being evaluated ('test', 'train', 'val').
    """
    EVAL_JOBS[job_id] = {
        "status": "running",
        "processed": 0,
        "total": len(file_list),
        "result": None
    }

    try:
        predictions_raw: List[Tuple[str, int]] = []
        total_files = len(file_list)

        if pytorch_model is not None:
            # Batch PyTorch inference (batch size 64)
            batch_size = 64
            DEVICE = next(pytorch_model.parameters()).device

            for idx in range(0, total_files, batch_size):
                if EVAL_JOBS[job_id]["status"] == "cancelled":
                    return

                batch_paths = file_list[idx: idx + batch_size]
                batch_tensors = []
                valid_paths = []

                for path in batch_paths:
                    try:
                        with Image.open(path) as img:
                            arr = np.array(img).astype(np.float32)
                            # Permute bands: PureForest is [NIR, Red, Green, Blue] -> Model expects [Red, Green, Blue, NIR]
                            arr = arr[:, :, [1, 2, 3, 0]]
                            # Scale pixel values from [0, 255] to [0.0, 1.0]
                            arr = arr / 255.0
                            tensor = torch.from_numpy(arr).permute(2, 0, 1)
                            batch_tensors.append(tensor)
                            valid_paths.append(path)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")

                if batch_tensors:
                    tensors = torch.stack(batch_tensors).to(DEVICE)
                    with torch.no_grad():
                        outputs = pytorch_model(tensors)
                        pred_ids = torch.argmax(outputs, dim=1).cpu().numpy()

                    for path, pred_id in zip(valid_paths, pred_ids):
                        predictions_raw.append((os.path.basename(path), int(pred_id)))

                EVAL_JOBS[job_id]["processed"] += len(batch_paths)

        elif rf_model is not None:
            # Sequential Random Forest feature extraction & inference
            for idx, path in enumerate(file_list):
                if EVAL_JOBS[job_id]["status"] == "cancelled":
                    return

                try:
                    with Image.open(path) as img:
                        arr = np.array(img)
                        feats = extract_rf_features(arr)
                        if feats is not None:
                            pred_id = int(rf_model.predict([feats])[0])
                            predictions_raw.append((os.path.basename(path), pred_id))
                except Exception as e:
                    print(f"Error predicting RF for {path}: {e}")

                EVAL_JOBS[job_id]["processed"] += 1

        else:
            raise ValueError("No prediction model loaded on the server.")

        # Compile evaluation results and compute overall metrics
        predictions_list: List[Dict[str, Any]] = []
        correct_count = 0

        for filename, pred_id in predictions_raw:
            true_class_id, true_class_name, true_species = parse_filename(filename)
            correct = (pred_id == true_class_id) if true_class_id is not None else None
            if correct:
                correct_count += 1

            predictions_list.append({
                "filename": filename,
                "predicted_class_id": pred_id,
                "predicted_class_name": CLASS_MAPPING.get(pred_id, "Unknown"),
                "true_class_id": true_class_id,
                "true_class_name": true_class_name,
                "true_species": true_species,
                "correct": correct
            })

        acc = (correct_count / len(file_list)) if file_list else 0.0

        EVAL_JOBS[job_id]["result"] = {
            "total_evaluated": len(file_list),
            "correct_count": correct_count,
            "accuracy": acc,
            "predictions": predictions_list
        }
        EVAL_JOBS[job_id]["status"] = "completed"
        print(f"Evaluation Job {job_id} completed successfully. Accuracy: {acc*100:.2f}%")

        # Persist evaluation result to disk cache for fast dashboard loading
        cache_name = "last_evaluation.json" if split == "test" else f"last_evaluation_{split}.json"
        cache_path = os.path.join(DATA_DIR, cache_name)
        try:
            with open(cache_path, "w") as f:
                json.dump(EVAL_JOBS[job_id]["result"], f)
            print(f"Successfully saved evaluation cache to {cache_path}")
        except Exception as ce:
            print(f"Failed to cache evaluation results: {ce}")

    except Exception as e:
        print(f"Error in evaluation job {job_id}: {e}")
        EVAL_JOBS[job_id]["status"] = "failed"
        EVAL_JOBS[job_id]["error"] = str(e)


# ----------------- HTTP Request Handler -----------------

class PureForestHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP Request Handler serving static frontend assets and REST API endpoints."""

    def log_message(self, format: str, *args: Any) -> None:
        """Filter log messages to only output warning (4xx) and error (5xx) status codes."""
        if len(args) >= 2 and ("404" in str(args[1]) or "500" in str(args[1])):
            print(f"[{self.log_date_time_string()}] {format % args}")

    def do_OPTIONS(self) -> None:
        """Handle HTTP OPTIONS preflight requests for CORS compliance."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle HTTP GET requests for API data and static web assets."""
        url = urllib.parse.urlparse(self.path)
        path = url.path
        query = urllib.parse.parse_qs(url.query)

        # GET /api/stats
        if path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            response_data = dict(DATASET_STATS)
            response_data["loaded_model"] = loaded_model_type or "None (Please train)"
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        # GET /api/last_evaluation
        if path == "/api/last_evaluation":
            if os.path.exists(CACHE_EVALUATION_PATH):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    with open(CACHE_EVALUATION_PATH, "rb") as f:
                        self.wfile.write(f.read())
                except Exception as e:
                    self.wfile.write(json.dumps({"error": f"Failed to read cache file: {e}"}).encode("utf-8"))
                return
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"message": "No cached evaluation available"}).encode("utf-8"))
                return

        # GET /api/job_status?job_id=<UUID>
        if path == "/api/job_status":
            job_id = query.get("job_id", [""])[0]
            if not job_id or job_id not in EVAL_JOBS:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Job ID {job_id} not found"}).encode("utf-8"))
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(EVAL_JOBS[job_id]).encode("utf-8"))
            return

        # GET /api/image?filename=<name.tiff>
        if path == "/api/image":
            filename = query.get("filename", [""])[0]
            if not filename:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing filename parameter"}).encode("utf-8"))
                return

            file_path = find_test_file_by_name(filename)
            if not file_path:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Image file {filename} not found in dataset"}).encode("utf-8"))
                return

            try:
                with Image.open(file_path) as img:
                    arr = np.array(img)

                # True Color (RGB): Extract bands 1, 2, 3 (Red, Green, Blue)
                tc_arr = arr[:, :, 1:4]
                tc_img = Image.fromarray(tc_arr.astype(np.uint8))
                tc_io = io.BytesIO()
                tc_img.save(tc_io, format="PNG")
                tc_base64 = base64.b64encode(tc_io.getvalue()).decode("utf-8")

                # False Color (Color Infrared / CIR): Display NIR -> Red channel, Red -> Green channel, Green -> Blue channel
                fc_arr = np.zeros((250, 250, 3), dtype=np.uint8)
                fc_arr[:, :, 0] = arr[:, :, 0]  # Channel 0: NIR -> Displayed as Red
                fc_arr[:, :, 1] = arr[:, :, 1]  # Channel 1: Red -> Displayed as Green
                fc_arr[:, :, 2] = arr[:, :, 2]  # Channel 2: Green -> Displayed as Blue
                fc_img = Image.fromarray(fc_arr)
                fc_io = io.BytesIO()
                fc_img.save(fc_io, format="PNG")
                fc_base64 = base64.b64encode(fc_io.getvalue()).decode("utf-8")

                res = {
                    "filename": filename,
                    "true_color_image": tc_base64,
                    "false_color_image": fc_base64
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Failed to load image: {e}"}).encode("utf-8"))
                return

        # Static asset serving from public/ directory
        if path == "/":
            path = "/index.html"

        clean_path = os.path.normpath(path).lstrip("/")
        public_dir = os.path.join(BASE_DIR, "public")
        file_path = os.path.join(public_dir, clean_path)

        # Prevent directory traversal attacks
        if not os.path.abspath(file_path).startswith(os.path.abspath(public_dir)) or not os.path.exists(file_path) or os.path.isdir(file_path):
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"File Not Found")
            return

        mime_types = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "text/javascript",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".json": "application/json"
        }
        _, ext = os.path.splitext(file_path)
        mime = mime_types.get(ext.lower(), "application/octet-stream")

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Internal Server Error: {e}".encode("utf-8"))

    def do_POST(self) -> None:
        """Handle HTTP POST requests for single prediction and batch evaluation initiation."""
        url = urllib.parse.urlparse(self.path)
        path = url.path

        # POST /api/predict: Single image inference
        if path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Empty body"}).encode("utf-8"))
                return

            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                filename = data.get("filename", "")
                base64_content = data.get("content", "")

                if not filename or not base64_content:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing filename or content"}).encode("utf-8"))
                    return

                img_data = base64.b64decode(base64_content)
                img = Image.open(io.BytesIO(img_data))
                arr = np.array(img)

                # Validate input image shape: must be a 3D array with at least 4 bands
                if arr.ndim != 3 or arr.shape[2] < 4:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Uploaded file is not a 4-channel TIFF image"}).encode("utf-8"))
                    return

                pred_class_id: Optional[int] = None

                # 1. Primary Inference: PyTorch EfficientNetV2
                if pytorch_model is not None:
                    try:
                        norm_arr = arr.astype(np.float32)
                        norm_arr = norm_arr[:, :, [1, 2, 3, 0]]  # Permute from [NIR, R, G, B] to [R, G, B, NIR]
                        norm_arr = norm_arr / 255.0  # Normalize to [0.0, 1.0]
                        img_tensor = torch.from_numpy(norm_arr).permute(2, 0, 1).unsqueeze(0)
                        img_tensor = img_tensor.to(next(pytorch_model.parameters()).device)
                        with torch.no_grad():
                            outputs = pytorch_model(img_tensor)
                            pred_class_id = int(torch.argmax(outputs, dim=1).item())
                    except Exception as pe:
                        print(f"Error predicting with PyTorch model: {pe}. Falling back to RF.")

                # 2. Fallback Inference: Random Forest
                if pred_class_id is None:
                    if rf_model is not None:
                        feats = extract_rf_features(arr)
                        if feats is not None:
                            pred_class_id = int(rf_model.predict([feats])[0])
                    else:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "No trained model weights loaded on the server."}).encode("utf-8"))
                        return

                pred_class_name = CLASS_MAPPING.get(pred_class_id, "Unknown")
                true_class_id, true_class_name, true_species = parse_filename(filename)

                # Generate rendered base64 PNGs for interactive UI display
                tc_arr = arr[:, :, 1:4]
                tc_img = Image.fromarray(tc_arr.astype(np.uint8))
                tc_io = io.BytesIO()
                tc_img.save(tc_io, format="PNG")
                tc_base64 = base64.b64encode(tc_io.getvalue()).decode("utf-8")

                fc_arr = np.zeros((250, 250, 3), dtype=np.uint8)
                fc_arr[:, :, 0] = arr[:, :, 0]
                fc_arr[:, :, 1] = arr[:, :, 1]
                fc_arr[:, :, 2] = arr[:, :, 2]
                fc_img = Image.fromarray(fc_arr)
                fc_io = io.BytesIO()
                fc_img.save(fc_io, format="PNG")
                fc_base64 = base64.b64encode(fc_io.getvalue()).decode("utf-8")

                correct = (pred_class_id == true_class_id) if true_class_id is not None else None

                res = {
                    "filename": filename,
                    "predicted_class_id": pred_class_id,
                    "predicted_class_name": pred_class_name,
                    "true_class_id": true_class_id,
                    "true_class_name": true_class_name,
                    "true_species": true_species,
                    "correct": correct,
                    "true_color_image": tc_base64,
                    "false_color_image": fc_base64
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))

            except Exception as e:
                print(f"Error handling POST /api/predict: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Server processing error: {e}"}).encode("utf-8"))

        # POST /api/evaluate_test: Bulk asynchronous evaluation job starter
        elif path == "/api/evaluate_test":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"

            try:
                data = json.loads(body.decode("utf-8")) if body else {}
                limit = data.get("limit", 0)
                split = data.get("split", "test")
                if split not in ["test", "train", "val"]:
                    split = "test"

                print(f"Scanning {split} split directory for bulk evaluation. Limit: {limit}")
                all_files = get_all_split_files(split)
                total_available = len(all_files)

                if total_available == 0:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"No {split} TIFF files found in workspace directories"}).encode("utf-8"))
                    return

                random.seed(42)
                if limit and 0 < limit < total_available:
                    sampled_files = random.sample(all_files, limit)
                else:
                    sampled_files = all_files

                # Generate unique job identifier and launch background daemon worker
                job_id = str(uuid.uuid4())
                print(f"Spawning evaluation background thread for job {job_id} (Evaluating {len(sampled_files)} files)")

                worker_thread = threading.Thread(target=run_evaluation_job, args=(job_id, sampled_files, split))
                worker_thread.daemon = True
                worker_thread.start()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "processing",
                    "job_id": job_id,
                    "total": len(sampled_files)
                }).encode("utf-8"))
                return

            except Exception as e:
                print(f"Error starting bulk evaluation job: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Server failed to start job: {e}"}).encode("utf-8"))


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Multithreaded TCP Server enabling concurrent request handling without blocking."""
    allow_reuse_address = True


def main() -> None:
    """Entry point for starting the PureForest verification web server."""
    os.makedirs(os.path.join(BASE_DIR, "public"), exist_ok=True)
    server_address = ("", PORT)
    with ThreadedTCPServer(server_address, PureForestHandler) as httpd:
        print(f"PureForest validation webapp running at http://localhost:{PORT}")
        print(f"Currently active model backend: {loaded_model_type or 'None'}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()
