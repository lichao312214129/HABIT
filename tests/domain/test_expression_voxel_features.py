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
"""Tests for the safe arithmetic ``expression`` voxel feature extractor."""

from __future__ import annotations

import numpy as np
import pytest

from habit._protocols import VoxelFeatureExtractor
from habit.voxel_features import (
    ExpressionVoxelFeatures,
    VoxelFeatureExtractorRegistry,
)
from habit.exceptions import HABITAPIError

from .conftest import make_subject


@pytest.mark.unit
def test_expression_is_registered_and_satisfies_protocol() -> None:
    """The built-in expression extractor is a first-class registry citizen."""
    assert "expression" in VoxelFeatureExtractorRegistry.available()
    extractor = VoxelFeatureExtractorRegistry.create(
        "expression",
        features={"ratio": "T1 / (T2 + eps)"},
    )
    assert isinstance(extractor, VoxelFeatureExtractor)


@pytest.mark.unit
def test_expression_square_ratio_matches_numpy() -> None:
    """``square(T1 / (T2 ** 3 + eps))`` reproduces the NumPy formula."""
    subject = make_subject("P1", modalities=("T1", "T2"), seed=3)
    eps = 1e-6
    extractor = ExpressionVoxelFeatures(
        features={"t1_over_t2_sq": "square(T1 / (T2 ** 3 + eps))"},
        eps=eps,
    )
    field = extractor(subject)
    assert field.feature_names == ("t1_over_t2_sq",)

    mask = np.asarray(subject.mask("tumor").data) > 0
    t1 = np.asarray(subject.image("T1").data)[mask]
    t2 = np.asarray(subject.image("T2").data)[mask]
    expected = np.square(t1 / (t2 ** 3 + eps))
    np.testing.assert_allclose(field.values[:, 0], expected)


@pytest.mark.unit
def test_expression_accepts_caret_as_power() -> None:
    """Users may write ``^`` for power; it is rewritten to ``**``."""
    subject = make_subject("P1", modalities=("T1", "T2"), seed=1)
    field = ExpressionVoxelFeatures(
        features={"powered": "T1 / (T2 ^ 3 + eps)"},
    )(subject)
    mask = np.asarray(subject.mask("tumor").data) > 0
    t1 = np.asarray(subject.image("T1").data)[mask]
    t2 = np.asarray(subject.image("T2").data)[mask]
    np.testing.assert_allclose(field.values[:, 0], t1 / (t2 ** 3 + 1e-8))


@pytest.mark.unit
def test_expression_rejects_attribute_access() -> None:
    """Attribute access must not escape the sandbox."""
    with pytest.raises(HABITAPIError, match="unsupported syntax"):
        ExpressionVoxelFeatures(features={"bad": "T1.__class__"})


@pytest.mark.unit
def test_expression_rejects_unknown_function() -> None:
    """Only the documented function whitelist is callable."""
    with pytest.raises(HABITAPIError, match="unknown function"):
        ExpressionVoxelFeatures(features={"bad": "sin(T1)"})


@pytest.mark.unit
def test_expression_rejects_import_like_calls() -> None:
    """``__import__`` and similar names are not in the whitelist."""
    with pytest.raises(HABITAPIError, match="unknown function"):
        ExpressionVoxelFeatures(features={"bad": "__import__('os')"})
