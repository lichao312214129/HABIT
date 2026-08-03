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
"""Golden-value tests for MSI feature extraction."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from habit.core.habitat_analysis.habitat_features.msi_features import (
    MSIFeatureExtractor,
)


@pytest.mark.unit
def test_msi_matrix_golden(
    synthetic_habitat_array: np.ndarray,
    golden_msi_ith_data: Dict[str, Any],
) -> None:
    """MSI co-occurrence matrix must match the frozen golden snapshot."""
    extractor = MSIFeatureExtractor(voxel_cutoff=10)
    matrix: np.ndarray = extractor.calculate_MSI_matrix(
        synthetic_habitat_array,
        unique_class=golden_msi_ith_data["n_habitats"] + 1,
    )
    expected = np.asarray(golden_msi_ith_data["msi_matrix"], dtype=np.int64)
    np.testing.assert_array_equal(matrix, expected)


@pytest.mark.unit
def test_msi_features_golden(
    synthetic_habitat_array: np.ndarray,
    golden_msi_ith_data: Dict[str, Any],
) -> None:
    """MSI first/second-order features must match the frozen golden snapshot."""
    extractor = MSIFeatureExtractor(voxel_cutoff=10)
    matrix = extractor.calculate_MSI_matrix(
        synthetic_habitat_array,
        unique_class=golden_msi_ith_data["n_habitats"] + 1,
    )
    features: Dict[str, float] = extractor.calculate_MSI_features(matrix, "golden")
    expected: Dict[str, float] = golden_msi_ith_data["msi_features"]

    assert set(features.keys()) == set(expected.keys())
    for key, expected_value in expected.items():
        assert features[key] == pytest.approx(expected_value, rel=0, abs=1e-9)
