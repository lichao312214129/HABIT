# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Golden-value tests for ITH feature extraction."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
import SimpleITK as sitk

from habit.core.habitat_analysis.habitat_features.ith_features import (
    ITHFeatureExtractor,
)


@pytest.mark.unit
def test_ith_features_golden(
    synthetic_habitat_array: np.ndarray,
    golden_msi_ith_data: Dict[str, Any],
) -> None:
    """ITH score and summary fields must match the frozen golden snapshot."""
    habitat_image = sitk.GetImageFromArray(synthetic_habitat_array.astype(np.uint32))
    extractor = ITHFeatureExtractor()
    result: Dict[str, Any] = extractor.extract_ith_features(habitat_image)

    assert "error" not in result
    expected: Dict[str, Any] = golden_msi_ith_data["ith_features"]
    for key, expected_value in expected.items():
        assert result[key] == pytest.approx(expected_value, rel=0, abs=1e-9)
