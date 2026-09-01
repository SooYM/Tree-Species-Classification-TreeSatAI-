# REST API Reference Documentation

The PureForest Web Backend exposes RESTful endpoints for model predictions, asynchronous dataset evaluation, cached analytics, and dynamic image rendering.

---

## 1. Overview & Protocol Conventions

- **Base URL**: `http://localhost:8080` (or configured host/port)
- **Data Exchange Format**: Standard JSON (`Content-Type: application/json`)
- **CORS Support**: All endpoints return `Access-Control-Allow-Origin: *` to enable cross-origin integrations.
- **Image Serialization**: Multi-channel rendered image tiles are encoded as standard base64 PNG data strings.

---

## 2. API Endpoints

### 2.1 Dataset Statistics & Active Model
Retrieves the dataset partition stats across splits and lists the currently active ML model backend.

- **Method**: `GET`
- **Endpoint**: `/api/stats`
- **Headers**: None required

#### Success Response (`200 OK`):
```json
{
  "splits": {
    "Train": { "area_km2": 172.78, "patches": 69111, "polygons": 330 },
    "Val": { "area_km2": 33.81, "patches": 13523, "polygons": 58 },
    "Test": { "area_km2": 132.34, "patches": 52935, "polygons": 61 }
  },
  "classes": [
    {
      "id": 0,
      "name": "Deciduous oak",
      "train": 15840,
      "val": 4374,
      "test": 27841,
      "species": ["Quercus robur", "Quercus pubescens", "Quercus petraea", "Quercus rubra"]
    },
    ...
  ],
  "loaded_model": "PyTorch (EfficientNetV2 on MPS)"
}
```

---

### 2.2 Single Image Prediction
Classifies an uploaded 4-channel TIFF image and returns the prediction along with True Color and False Color CIR views.

- **Method**: `POST`
- **Endpoint**: `/api/predict`
- **Content-Type**: `application/json`

#### Request Body:
```json
{
  "filename": "TEST-Abies_alba-C9-407_1_108.tiff",
  "content": "<base64-encoded binary string of TIFF file>"
}
```

#### Success Response (`200 OK`):
```json
{
  "filename": "TEST-Abies_alba-C9-407_1_108.tiff",
  "predicted_class_id": 9,
  "predicted_class_name": "Fir",
  "true_class_id": 9,
  "true_class_name": "Fir",
  "true_species": "Abies alba",
  "correct": true,
  "true_color_image": "iVBORw0KGgoAAAANSUhEUgAAA...",
  "false_color_image": "iVBORw0KGgoAAAANSUhEUgAAA..."
}
```

#### Error Responses:
- `400 Bad Request`: Empty body, missing `filename`/`content`, or invalid TIFF format (less than 4 channels).
- `500 Internal Server Error`: No model weights loaded or unexpected decoding failure.

---

### 2.3 Bulk Dataset Evaluation Starter
Initiates an asynchronous background worker thread to evaluate a sample or entire dataset split from disk.

- **Method**: `POST`
- **Endpoint**: `/api/evaluate_test`
- **Content-Type**: `application/json`

#### Request Body:
```json
{
  "limit": 1000,
  "split": "test"
}
```
*Note: Set `limit: 0` to evaluate all available patches in the split.*

#### Success Response (`200 OK`):
```json
{
  "status": "processing",
  "job_id": "3f8b89e2-8924-4f40-b463-3dc822d64024",
  "total": 1000
}
```

---

### 2.4 Bulk Evaluation Job Status Polling
Queries the live execution state and progress of an ongoing evaluation job.

- **Method**: `GET`
- **Endpoint**: `/api/job_status?job_id=<UUID>`

#### Query Parameters:
| Parameter | Type | Required | Description |
|---|---|---|---|
| `job_id` | `string` (UUID) | Yes | The job identifier returned by `/api/evaluate_test` |

#### Running State Response (`200 OK`):
```json
{
  "status": "running",
  "processed": 450,
  "total": 1000,
  "result": null
}
```

#### Completed State Response (`200 OK`):
```json
{
  "status": "completed",
  "processed": 1000,
  "total": 1000,
  "result": {
    "total_evaluated": 1000,
    "correct_count": 852,
    "accuracy": 0.852,
    "predictions": [
      {
        "filename": "TEST-Fagus_sylvatica-C2-100_1_2.tiff",
        "predicted_class_id": 2,
        "predicted_class_name": "Beech",
        "true_class_id": 2,
        "true_class_name": "Beech",
        "true_species": "Fagus sylvatica",
        "correct": true
      }
    ]
  }
}
```

---

### 2.5 Retrieve Cached Evaluation
Fetches the results of the last completed evaluation from the local disk cache (`last_evaluation.json`).

- **Method**: `GET`
- **Endpoint**: `/api/last_evaluation`

#### Success Response (`200 OK`):
```json
{
  "total_evaluated": 1000,
  "correct_count": 852,
  "accuracy": 0.852,
  "predictions": [ ... ]
}
```

#### Cache Miss (`404 Not Found`):
```json
{
  "message": "No cached evaluation available"
}
```

---

### 2.6 On-Demand Image Channel Fetching
Dynamically extracts and returns True Color (RGB) and False Color (CIR) PNGs for a given TIFF file located on disk.

- **Method**: `GET`
- **Endpoint**: `/api/image?filename=<name.tiff>`

#### Query Parameters:
| Parameter | Type | Required | Description |
|---|---|---|---|
| `filename` | `string` | Yes | Name of the TIFF patch (e.g. `TEST-Abies_alba-C9-407_1_108.tiff`) |

#### Success Response (`200 OK`):
```json
{
  "filename": "TEST-Abies_alba-C9-407_1_108.tiff",
  "true_color_image": "iVBORw0KGgoAAAANSUhEUgAAA...",
  "false_color_image": "iVBORw0KGgoAAAANSUhEUgAAA..."
}
```
