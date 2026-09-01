# Testing & Debugging Guide

This manual explains how to execute automated test suites, verify TIFF multi-spectral data integrity, debug backend inference pipelines, and troubleshoot common runtime issues.

---

## 1. Running Automated Tests

The test suite is located in the `tests/` directory and covers patch filename decoding, 42-dimensional spectral feature extraction, dataset integrity, and model architecture definitions.

### 1.1 Running with Python `unittest`
```bash
# Run all tests from the repository root
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 1.2 Running with `pytest`
```bash
# Run tests with detailed output
pytest tests/ -v
```

---

## 2. Verifying Multi-Channel TIFF Data Integrity

PureForest images require 4 distinct spectral bands in `[NIR, Red, Green, Blue]` order with dimensions `(250, 250)`.

You can inspect any TIFF patch using Python:
```python
from PIL import Image
import numpy as np

img_path = "imagery-Abies_alba/test/TEST-Abies_alba-C9-407_1_108.tiff"
with Image.open(img_path) as img:
    arr = np.array(img)
    print(f"Shape: {arr.shape}")       # Must be (250, 250, 4)
    print(f"Data type: {arr.dtype}")   # uint8 or float32
    print(f"Band 0 (NIR) Mean: {np.mean(arr[:, :, 0]):.2f}")
    print(f"Band 1 (Red) Mean: {np.mean(arr[:, :, 1]):.2f}")
    print(f"Band 2 (Green) Mean: {np.mean(arr[:, :, 2]):.2f}")
    print(f"Band 3 (Blue) Mean: {np.mean(arr[:, :, 3]):.2f}")
```

---

## 3. Common Troubleshooting Scenarios

### Scenario A: Server Port `8080` Already in Use
**Symptom**: `OSError: [Errno 48] Address already in use`  
**Solution**:
1. Check what process is occupying the port:
   ```bash
   lsof -i :8080
   ```
2. Or run the server on an alternative port:
   ```bash
   PORT=8090 python3 app.py
   ```

---

### Scenario B: "Uploaded file is not a 4-channel TIFF image" (HTTP 400)
**Symptom**: When dragging an image, the sandbox alerts that the uploaded file is not a 4-channel TIFF.  
**Cause**: The uploaded file is either a standard 3-channel RGB image (JPEG/PNG) or an unsupported format.  
**Solution**: Ensure that patches are sourced from the PureForest dataset directories containing `.tiff` files.

---

### Scenario C: "No pre-trained model loaded. Run training script first"
**Symptom**: Server console outputs a warning that neither `efficientnet_v2_forest.pth` nor `rf_model.joblib` was found.  
**Solution**:
1. Train the baseline Random Forest model quickly:
   ```bash
   python3 train_classifier.py
   ```
2. Or train/download the PyTorch EfficientNetV2 weights:
   ```bash
   python3 train_efficientnet.py --epochs 10
   ```

---

### Scenario D: PyTorch MPS / CUDA Device Failures
**Symptom**: Out-of-memory or fallback to CPU on Apple Silicon or CUDA.  
**Explanation**: In `app.py` and `train_efficientnet.py`, device detection automatically falls back gracefully:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
```
If GPU memory is constrained during training, decrease `--batch-size` to `16` or `8`.
