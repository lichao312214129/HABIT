# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Shared fixtures for API-level integration and golden tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = PROJECT_ROOT / "tests" / "fixtures" / "api_golden"


@pytest.fixture(scope="session")
def golden_msi_ith_data() -> Dict[str, Any]:
    """Load fixed MSI / ITH golden expectations for the synthetic habitat cube."""
    path = GOLDEN_DIR / "msi_ith_golden.json"
    if not path.is_file():
        pytest.skip(f"Golden fixture not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def synthetic_habitat_array() -> np.ndarray:
    """
    Build the deterministic 5x5x5 label map used in golden MSI / ITH tests.

    Returns:
        Integer array with habitat labels 1, 2, 3 in separate sub-regions.
    """
    arr = np.zeros((5, 5, 5), dtype=np.int32)
    arr[1:3, 1:3, 1:3] = 1
    arr[3:5, 1:3, 1:3] = 2
    arr[1:3, 3:5, 1:3] = 3
    return arr


@pytest.fixture
def synthetic_preprocess_dataset(tmp_path: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Create a minimal resample-only preprocessing dataset under ``tmp_path``.

    Returns:
        Tuple of (data_dir, config_dict) suitable for ``PreprocessingConfig``.
    """
    data_dir = tmp_path / "preprocess_input"
    image_dir = data_dir / "images" / "sub001" / "delay2"
    mask_dir = data_dir / "masks" / "sub001" / "delay2"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    rng = np.random.RandomState(42)
    array: np.ndarray = rng.randint(0, 100, size=(8, 8, 8)).astype(np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(image_dir / "delay2.nii.gz"))

    mask_array: np.ndarray = (array > 50).astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_array)
    mask.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(mask, str(mask_dir / "delay2.nii.gz"))

    config_dict: Dict[str, Any] = {
        "data_dir": str(data_dir),
        "out_dir": str(tmp_path / "preprocess_out"),
        "processes": 1,
        "auto_select_first_file": True,
        "random_state": 42,
        "preprocessing": {
            "resample": {
                "images": ["delay2"],
                "target_spacing": [2.0, 2.0, 2.0],
                "img_mode": "bilinear",
            }
        },
        "save_options": {
            "save_intermediate": False,
            "intermediate_steps": [],
        },
    }
    return data_dir, config_dict


@pytest.fixture
def synthetic_feature_extraction_dataset(
    tmp_path: Path,
    synthetic_habitat_array: np.ndarray,
) -> Dict[str, Any]:
    """
    Create a minimal feature-extraction dataset (MSI / ITH / non_radiomics only).

    Returns:
        Dict of paths and values for ``FeatureExtractionConfig.model_validate``.
    """
    raw_root = tmp_path / "raw"
    habitats_dir = tmp_path / "habitats"
    image_dir = raw_root / "images" / "sub001" / "delay2"
    mask_dir = raw_root / "masks" / "sub001" / "delay2"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    habitats_dir.mkdir(parents=True)

    intensity: np.ndarray = np.linspace(10, 100, num=125, dtype=np.float32).reshape(
        (5, 5, 5)
    )
    image = sitk.GetImageFromArray(intensity)
    sitk.WriteImage(image, str(image_dir / "delay2.nii.gz"))

    mask = sitk.GetImageFromArray((intensity > 50).astype(np.uint8))
    sitk.WriteImage(mask, str(mask_dir / "delay2.nii.gz"))

    habitat_image = sitk.GetImageFromArray(synthetic_habitat_array.astype(np.uint32))
    sitk.WriteImage(habitat_image, str(habitats_dir / "sub001_habitats.nrrd"))

    return {
        "raw_img_folder": str(raw_root),
        "habitats_map_folder": str(habitats_dir),
        "out_dir": str(tmp_path / "features_out"),
        "n_processes": 1,
        "habitat_pattern": "*_habitats.nrrd",
        "feature_types": ["non_radiomics", "msi", "ith_score"],
        "n_habitats": 3,
        "debug": False,
    }
