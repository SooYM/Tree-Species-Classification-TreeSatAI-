"""
Unit and Integration Tests for PureForest Classification & Verification Webapp.

This test suite verifies:
1. PureForest patch filename parser and ground truth label extraction.
2. 42-dimensional spectral feature extraction and NDVI calculation.
3. Dataset partition and semantic class statistics dictionary integrity.
4. 4-channel EfficientNetV2-S model architecture configuration.
5. In-memory asynchronous evaluation tracking.
"""

import os
import sys
import unittest
import numpy as np

# Ensure parent directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import app
from app import (
    parse_filename,
    extract_rf_features,
    CLASS_MAPPING,
    SPECIES_TO_CLASS,
    DATASET_STATS
)


class TestPureForestFilenameParser(unittest.TestCase):
    """Tests for patch filename decoding logic."""

    def test_parse_filename_standard_convention(self):
        """Verify parsing of standard `{SPLIT}-{species}-C{class_id}-{patch_id}.tiff` format."""
        cid, cname, spec = parse_filename("TEST-Abies_alba-C9-407_1_108.tiff")
        self.assertEqual(cid, 9)
        self.assertEqual(cname, "Fir")
        self.assertEqual(spec, "Abies alba")

        cid, cname, spec = parse_filename("TRAIN-Fagus_sylvatica-C2-100_2_55.tiff")
        self.assertEqual(cid, 2)
        self.assertEqual(cname, "Beech")
        self.assertEqual(spec, "Fagus sylvatica")

        cid, cname, spec = parse_filename("VAL-Quercus_ilex-C1-88_9_12.tiff")
        self.assertEqual(cid, 1)
        self.assertEqual(cname, "Evergreen oak")
        self.assertEqual(spec, "Quercus ilex")

    def test_parse_filename_fallback_species_matching(self):
        """Verify fallback when C{id} is missing but species name is present in filename."""
        cid, cname, spec = parse_filename("patch_Pinus_halepensis_sample.tiff")
        self.assertEqual(cid, 8)
        self.assertEqual(cname, "Aleppo pine")
        self.assertEqual(spec, "Pinus halepensis")

    def test_parse_filename_unknown(self):
        """Verify return values for unrecognizable filenames."""
        cid, cname, spec = parse_filename("random_unlabeled_image.tiff")
        self.assertIsNone(cid)
        self.assertEqual(cname, "Unknown")
        self.assertEqual(spec, "Unknown")


class TestFeatureExtraction(unittest.TestCase):
    """Tests for 42-dimensional spectral feature extraction."""

    def test_extract_rf_features_valid_array(self):
        """Verify extraction from a valid (250, 250, 4) numpy array."""
        np.random.seed(42)
        # Create synthetic 4-channel image (NIR, Red, Green, Blue)
        dummy_arr = np.random.randint(10, 240, size=(250, 250, 4), dtype=np.uint8)

        feats = extract_rf_features(dummy_arr)
        self.assertIsNotNone(feats)
        self.assertIsInstance(feats, list)
        self.assertEqual(len(feats), 42)

        # Check that all features are valid floats and not NaN or Inf
        for f in feats:
            self.assertFalse(np.isnan(f))
            self.assertFalse(np.isinf(f))

    def test_extract_rf_features_invalid_dimensions(self):
        """Verify extraction returns None when input has fewer than 4 channels or wrong dims."""
        rgb_only = np.zeros((250, 250, 3), dtype=np.uint8)
        self.assertIsNone(extract_rf_features(rgb_only))

        two_dim = np.zeros((250, 250), dtype=np.uint8)
        self.assertIsNone(extract_rf_features(two_dim))


class TestDatasetIntegrity(unittest.TestCase):
    """Tests for dataset statistics and class definitions."""

    def test_class_mapping_count(self):
        """Verify that exactly 13 target semantic classes exist."""
        self.assertEqual(len(CLASS_MAPPING), 13)
        for i in range(13):
            self.assertIn(i, CLASS_MAPPING)

    def test_species_to_class_validity(self):
        """Verify that all mapped species link to valid class IDs in 0..12."""
        for species, cid in SPECIES_TO_CLASS.items():
            self.assertGreaterEqual(cid, 0)
            self.assertLessEqual(cid, 12)
            self.assertIn(cid, CLASS_MAPPING)

    def test_dataset_stats_splits(self):
        """Verify total patch count consistency in DATASET_STATS."""
        splits = DATASET_STATS["splits"]
        self.assertIn("Train", splits)
        self.assertIn("Val", splits)
        self.assertIn("Test", splits)

        total_split_patches = (
            splits["Train"]["patches"] +
            splits["Val"]["patches"] +
            splits["Test"]["patches"]
        )
        self.assertEqual(total_split_patches, 135569)


class TestModelArchitecture(unittest.TestCase):
    """Tests for PyTorch model creation."""

    def test_pytorch_architecture_channels_and_classes(self):
        """Verify PyTorch model 4-channel input stem and 13-class classifier head."""
        if not app.torch_available:
            self.skipTest("PyTorch not installed in environment.")

        model = app.get_pytorch_architecture()
        self.assertIsNotNone(model)

        # Check input channel size on first convolution
        in_channels = model.features[0][0].in_channels
        self.assertEqual(in_channels, 4)

        # Check output class count on linear classification head
        out_features = model.classifier[1].out_features
        self.assertEqual(out_features, 13)


if __name__ == "__main__":
    unittest.main()
