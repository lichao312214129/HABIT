# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Pipeline smoke tests using synthetic in-memory datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import SimpleITK as sitk

import habit


@pytest.mark.integration
def test_preprocess_smoke_via_public_api(
    synthetic_preprocess_dataset: tuple[Path, Dict[str, Any]],
) -> None:
    """Resample-only preprocessing runs end-to-end through ``habit.run_preprocess``."""
    _, config_dict = synthetic_preprocess_dataset
    config = habit.PreprocessingConfig.model_validate(config_dict)
    habit.run_preprocess(config)

    output_root = Path(config_dict["out_dir"])
    image_outputs = list(output_root.rglob("delay2.nii.gz"))
    assert image_outputs, "Expected resampled image output"

    image = sitk.ReadImage(str(image_outputs[0]))
    spacing = image.GetSpacing()
    assert spacing == pytest.approx((2.0, 2.0, 2.0))


@pytest.mark.integration
def test_feature_extraction_smoke_via_public_api(
    synthetic_feature_extraction_dataset: Dict[str, Any],
) -> None:
    """MSI / ITH / non_radiomics extraction runs through ``habit.run_feature_extraction``."""
    config = habit.FeatureExtractionConfig.model_validate(
        synthetic_feature_extraction_dataset
    )
    habit.run_feature_extraction(config)

    out_dir = Path(synthetic_feature_extraction_dataset["out_dir"])
    expected_csvs = {
        "habitat_basic_features.csv",
        "msi_features.csv",
        "ith_scores.csv",
    }
    produced = {path.name for path in out_dir.glob("*.csv")}
    assert expected_csvs.issubset(produced)
