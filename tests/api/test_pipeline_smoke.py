# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Pipeline smoke tests using synthetic in-memory datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import SimpleITK as sitk

from habit.api.habitat import FeatureExtractionConfig, run_feature_extraction
from habit.api.preprocessing import PreprocessingConfig, run_preprocess


@pytest.mark.integration
def test_preprocess_smoke_via_public_api(
    synthetic_preprocess_dataset: tuple[Path, Dict[str, Any]],
) -> None:
    """Resample-only preprocessing runs end-to-end through ``run_preprocess``."""
    _, config_dict = synthetic_preprocess_dataset
    config = PreprocessingConfig.model_validate(config_dict)
    run_preprocess(config)

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
    """MSI / ITH / non_radiomics extraction runs through ``run_feature_extraction``."""
    config = FeatureExtractionConfig.model_validate(
        synthetic_feature_extraction_dataset
    )
    run_feature_extraction(config)

    out_dir = Path(synthetic_feature_extraction_dataset["out_dir"])
    expected_csvs = {
        "habitat_basic_features.csv",
        "msi_features.csv",
        "ith_scores.csv",
    }
    produced = {path.name for path in out_dir.glob("*.csv")}
    assert expected_csvs.issubset(produced)
