# Data Schema & Metadata Specification

This document details the dataset schemas, taxonomic metadata dictionaries, file naming conventions, and in-memory application state structures used throughout the PureForest verification platform.

---

## 1. Taxonomic Dictionary Schema (`PureForestID-dictionnary.csv`)

The taxonomic dictionary defines the mapping between the 18 botanical tree species, their bilingual common names, taxonomic classifications, and their unified semantic class index (0 to 12).

| Column Name | Data Type | Nullable | Description & Example Values |
|---|---|---|---|
| `class_index` | `INTEGER` (0..12) | No | Target semantic class index (e.g. `0`, `1`, `2`) |
| `class_name_en` | `VARCHAR(64)` | No | English semantic class name (e.g. `Deciduous oak`, `Beech`) |
| `class_name_fr` | `VARCHAR(64)` | No | French semantic class name (e.g. `Chêne décidu`, `Hêtre`) |
| `species_name_en` | `VARCHAR(64)` | No | English constituent species name (e.g. `Sessile oak`, `European beech`) |
| `species_name_fr` | `VARCHAR(64)` | No | French constituent species name (e.g. `Chêne sessile`, `Hêtre`) |
| `species_name_latin`| `VARCHAR(64)` | No | Latin binomial nomenclature (e.g. `Quercus petraea`, `Fagus sylvatica`) |
| `hierarchy_2` | `VARCHAR(32)` | No | Genus classification (e.g. `Quercus`, `Fagus`, `Pinus`, `Abies`) |
| `hierarchy_1` | `VARCHAR(32)` | No | Broad leaf type classification: `Broadleaf` or `Needleleaf` |

### Complete Taxonomic Mapping Table

| Class ID | Semantic Class (EN) | Constituent Latin Species | Genus | Leaf Type |
|:---:|---|---|---|---|
| **0** | Deciduous oak | *Quercus robur*, *Quercus pubescens*, *Quercus petraea*, *Quercus rubra* | Quercus | Broadleaf |
| **1** | Evergreen oak | *Quercus ilex* | Quercus | Broadleaf |
| **2** | Beech | *Fagus sylvatica* | Fagus | Broadleaf |
| **3** | Chestnut | *Castanea sativa* | Castanea | Broadleaf |
| **4** | Black locust | *Robinia pseudoacacia* | Robinia | Broadleaf |
| **5** | Maritime pine | *Pinus pinaster* | Pinus | Needleleaf |
| **6** | Scotch pine | *Pinus sylvestris* | Pinus | Needleleaf |
| **7** | Black pine | *Pinus nigra*, *Pinus nigra laricio* | Pinus | Needleleaf |
| **8** | Aleppo pine | *Pinus halepensis* | Pinus | Needleleaf |
| **9** | Fir | *Abies alba*, *Abies nordmanniana* | Abies | Needleleaf |
| **10** | Spruce | *Picea abies* | Picea | Needleleaf |
| **11** | Larch | *Larix decidua* | Larix | Needleleaf |
| **12** | Douglas | *Pseudotsuga menziesii* | Pseudotsuga | Needleleaf |

---

## 2. Patch Metadata Schema (`PureForest-patches.csv`)

For full-scale GIS analyses, individual patch locations and forest polygon associations are recorded in `PureForest-patches.csv`.

| Field Name | Type | Description |
|---|---|---|
| `patch_id` | `VARCHAR(128)` | Unique patch string containing species, class, and grid coordinates |
| `french_department_id` | `VARCHAR(8)` | 3-digit French department administrative code (e.g. `013`, `083`) |
| `annotation_id` | `INTEGER` | Forest polygon annotation identifier to preserve spatial integrity |
| `split` | `ENUM('train','val','test')` | Partition split assignment (clustered by forest polygon) |
| `bdforetv3_index` | `INTEGER` | Reference ID from IGN BD Forêt Version 3 database |
| `name_latin` | `VARCHAR(64)` | Latin species name |
| `class_name` | `VARCHAR(64)` | Semantic class English name |
| `class_index` | `INTEGER` | Semantic class index (0 to 12) |

---

## 3. File Organization & Naming Schema

### 3.1 Directory Layout Pattern
```
imagery-{Latin_Species_Name}/
  ├── train/
  │   └── TRAIN-{Latin_Species_Name}-C{Class_ID}-{Patch_ID}.tiff
  ├── val/
  │   └── VAL-{Latin_Species_Name}-C{Class_ID}-{Patch_ID}.tiff
  └── test/
      └── TEST-{Latin_Species_Name}-C{Class_ID}-{Patch_ID}.tiff
```

### 3.2 Filename Regular Expression
Patch files follow this exact regular expression:
```regex
^(TRAIN|VAL|TEST)-([A-Za-z]+_[A-Za-z_]+)-C([0-9]|1[0-2])-([0-9_]+)\.(tiff|tif)$
```

- Group 1 (`SPLIT`): `TRAIN`, `VAL`, or `TEST`
- Group 2 (`Species`): Latin name with underscores (e.g., `Pinus_sylvestris`)
- Group 3 (`Class ID`): Integer from `0` to `12`
- Group 4 (`Patch ID`): Grid and spatial tile coordinates (e.g., `407_1_108`)

---

## 4. In-Memory Application State & Cache Schema

### 4.1 Asynchronous Evaluation Job State (`EVAL_JOBS[job_id]`)
Managed in server RAM during background evaluation:

```json
{
  "status": "running | completed | failed | cancelled",
  "processed": 1250,
  "total": 5000,
  "result": {
    "total_evaluated": 5000,
    "correct_count": 4260,
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
  },
  "error": null
}
```

### 4.2 Local Evaluation Cache (`last_evaluation.json`)
Persisted to disk to ensure instant dashboard rendering upon page reload:

```json
{
  "total_evaluated": 1000,
  "correct_count": 852,
  "accuracy": 0.852,
  "predictions": [ ... ]
}
```

### 4.3 Single Prediction Payload (`POST /api/predict`)

#### Request Body Schema:
```json
{
  "filename": "TEST-Abies_alba-C9-407_1_108.tiff",
  "content": "<base64-encoded binary string of 4-channel TIFF>"
}
```

#### Response Body Schema:
```json
{
  "filename": "TEST-Abies_alba-C9-407_1_108.tiff",
  "predicted_class_id": 9,
  "predicted_class_name": "Fir",
  "true_class_id": 9,
  "true_class_name": "Fir",
  "true_species": "Abies alba",
  "correct": true,
  "true_color_image": "<base64-encoded RGB PNG>",
  "false_color_image": "<base64-encoded CIR PNG>"
}
```
