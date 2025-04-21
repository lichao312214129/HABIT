"""
Unit tests for HabitatFeatureExtractor class
"""

import pytest
import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import tempfile
import shutil

from habit.core.habitat_analysis.feature_extraction import HabitatFeatureExtractor


@pytest.fixture
def temp_dir():
    """Create temporary directories for testing."""
    test_dir = tempfile.mkdtemp()
    raw_img_dir = os.path.join(test_dir, "raw_img")
    habitats_dir = os.path.join(test_dir, "habitats")
    out_dir = os.path.join(test_dir, "output")
    
    # Create directory structure
    os.makedirs(raw_img_dir, exist_ok=True)
    os.makedirs(habitats_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    # Create nested structure for raw images
    os.makedirs(os.path.join(raw_img_dir, "images", "test_subject", "T1"), exist_ok=True)
    os.makedirs(os.path.join(raw_img_dir, "masks", "test_subject", "T1"), exist_ok=True)
    
    # Create dummy images
    img_array = np.zeros((10, 10, 10), dtype=np.float32)
    img_array[3:7, 3:7, 3:7] = 1.0  # Add a simple cube
    
    # Create test image and mask
    img = sitk.GetImageFromArray(img_array)
    sitk.WriteImage(img, os.path.join(raw_img_dir, "images", "test_subject", "T1", "img.nrrd"))
    
    mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_array[3:7, 3:7, 3:7] = 1  # Same cube as ROI
    mask = sitk.GetImageFromArray(mask_array)
    sitk.WriteImage(mask, os.path.join(raw_img_dir, "masks", "test_subject", "T1", "mask.nrrd"))
    
    # Create test habitat map
    habitat_array = np.zeros((10, 10, 10), dtype=np.uint8)
    habitat_array[3:5, 3:7, 3:7] = 1  # First habitat
    habitat_array[5:7, 3:7, 3:7] = 2  # Second habitat
    habitat = sitk.GetImageFromArray(habitat_array)
    sitk.WriteImage(habitat, os.path.join(habitats_dir, "test_subject_habitats.nrrd"))
    
    # Create simple parameter files
    with open(os.path.join(test_dir, "params_non_habitat.yaml"), "w") as f:
        f.write("""
# Test parameter file
featureClass:
  firstorder: []
setting:
  normalize: false
  resampledPixelSpacing: [1, 1, 1]
        """)
    
    with open(os.path.join(test_dir, "params_habitat.yaml"), "w") as f:
        f.write("""
# Test parameter file
featureClass:
  firstorder: []
setting:
  normalize: false
  resampledPixelSpacing: [1, 1, 1]
        """)
    
    yield {
        "test_dir": test_dir,
        "raw_img_dir": raw_img_dir,
        "habitats_dir": habitats_dir,
        "out_dir": out_dir,
        "params_non_habitat": os.path.join(test_dir, "params_non_habitat.yaml"),
        "params_habitat": os.path.join(test_dir, "params_habitat.yaml")
    }
    
    # Cleanup after tests
    shutil.rmtree(test_dir)


def test_feature_extractor_initialization(temp_dir):
    """Test HabitatFeatureExtractor initialization."""
    extractor = HabitatFeatureExtractor(
        params_file_of_non_habitat=temp_dir["params_non_habitat"],
        params_file_of_habitat=temp_dir["params_habitat"],
        raw_img_folder=temp_dir["raw_img_dir"],
        habitats_map_folder=temp_dir["habitats_dir"],
        out_dir=temp_dir["out_dir"],
        n_processes=1
    )
    
    assert extractor.params_file_of_non_habitat == temp_dir["params_non_habitat"]
    assert extractor.params_file_of_habitat == temp_dir["params_habitat"]
    assert extractor.raw_img_folder == temp_dir["raw_img_dir"]
    assert extractor.habitats_map_folder == temp_dir["habitats_dir"]
    assert extractor.out_dir == temp_dir["out_dir"]


def test_get_mask_and_raw_files(temp_dir):
    """Test get_mask_and_raw_files method."""
    extractor = HabitatFeatureExtractor(
        params_file_of_non_habitat=temp_dir["params_non_habitat"],
        params_file_of_habitat=temp_dir["params_habitat"],
        raw_img_folder=temp_dir["raw_img_dir"],
        habitats_map_folder=temp_dir["habitats_dir"],
        out_dir=temp_dir["out_dir"],
        n_processes=1
    )
    
    images_paths, habitat_paths = extractor.get_mask_and_raw_files()
    
    assert "test_subject" in images_paths
    assert "T1" in images_paths["test_subject"]
    assert "test_subject" in habitat_paths
    assert os.path.basename(habitat_paths["test_subject"]) == "test_subject_habitats.nrrd"


def test_extract_radiomics_features_for_whole_habitat(temp_dir):
    """Test extract_radiomics_features_for_whole_habitat method."""
    habitat_path = os.path.join(temp_dir["habitats_dir"], "test_subject_habitats.nrrd")
    
    features = HabitatFeatureExtractor.extract_radiomics_features_for_whole_habitat(
        habitat_path, 
        temp_dir["params_habitat"]
    )
    
    # Should return a dictionary with features
    assert isinstance(features, dict)
    # Should include some firstorder features
    assert any("firstorder" in key for key in features.keys())


@pytest.mark.parametrize("feature_types", [
    ["traditional"],
    ["non_radiomics"],
    ["whole_habitat"],
    ["each_habitat"],
    ["msi"],
    ["traditional", "non_radiomics", "whole_habitat", "each_habitat", "msi"]
])
def test_extract_features_with_different_types(temp_dir, feature_types):
    """Test extracting different types of features."""
    extractor = HabitatFeatureExtractor(
        params_file_of_non_habitat=temp_dir["params_non_habitat"],
        params_file_of_habitat=temp_dir["params_habitat"],
        raw_img_folder=temp_dir["raw_img_dir"],
        habitats_map_folder=temp_dir["habitats_dir"],
        out_dir=temp_dir["out_dir"],
        n_processes=1
    )
    
    # Run feature extraction
    extractor.run(
        feature_types=feature_types,
        n_habitats=2,
        mode="extract"
    )
    
    # Check if feature file was created
    feature_files = [f for f in os.listdir(temp_dir["out_dir"]) if f.endswith('.npy')]
    assert len(feature_files) > 0, "No feature files were created"


def test_traditional_radiomics_extraction(temp_dir):
    """Test traditional radiomics feature extraction."""
    extractor = HabitatFeatureExtractor(
        params_file_of_non_habitat=temp_dir["params_non_habitat"],
        params_file_of_habitat=temp_dir["params_habitat"],
        raw_img_folder=temp_dir["raw_img_dir"],
        habitats_map_folder=temp_dir["habitats_dir"],
        out_dir=temp_dir["out_dir"],
        n_processes=1
    )
    
    # Get file paths
    images_paths, habitat_paths = extractor.get_mask_and_raw_files()
    
    # Extract features for one subject
    subject_id = "test_subject"
    image_path = images_paths[subject_id]["T1"]
    habitat_path = habitat_paths[subject_id]
    
    # Extract traditional radiomics features
    features = HabitatFeatureExtractor.extract_tranditional_radiomics(
        image_path,
        habitat_path,
        subject_id,
        temp_dir["params_non_habitat"]
    )
    
    # Should have features for both habitats
    assert 1 in features, "Features for habitat 1 not found"
    assert 2 in features, "Features for habitat 2 not found"
    
    # Each habitat should have firstorder features
    assert any("firstorder" in key for key in features[1].keys()), "No firstorder features for habitat 1"
    assert any("firstorder" in key for key in features[2].keys()), "No firstorder features for habitat 2" 