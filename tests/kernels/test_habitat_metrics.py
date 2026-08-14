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
"""Tests for the L0 habitat metric kernels.

The reference values guard two properties: the kernels reproduce the exact
semantics of the established v0.1 implementations (verified against a
brute-force re-implementation of the v0.1 loops), and the derived feature
formulas are correct on hand-computable examples.
"""

from __future__ import annotations

import numpy as np
import pytest

from habit.kernels.habitat_metrics import (
    habitat_ith_dispersion,
    habitat_region_stats,
    habitat_volume_fractions,
    ith_score,
    msi_features_from_matrix,
    spatial_interaction_matrix,
)


def _reference_msi_matrix(habitat_array: np.ndarray, unique_class: int) -> np.ndarray:
    """Brute-force re-implementation of the v0.1 ``calculate_MSI_matrix``."""
    roi_z, roi_y, roi_x = np.where(habitat_array != 0)
    if len(roi_z) == 0:
        return np.zeros((unique_class, unique_class), dtype=np.int64)
    box = habitat_array[
        roi_z.min() : roi_z.max() + 1,
        roi_y.min() : roi_y.max() + 1,
        roi_x.min() : roi_x.max() + 1,
    ]
    box = np.pad(box, ((1, 1), (1, 1), (1, 1)), "constant", constant_values=0)
    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    matrix = np.zeros((unique_class, unique_class), dtype=np.int64)
    for z in range(box.shape[0]):
        for y in range(box.shape[1]):
            for x in range(box.shape[2]):
                current = box[z, y, x]
                for dz, dy, dx in offsets:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (
                        0 <= nz < box.shape[0]
                        and 0 <= ny < box.shape[1]
                        and 0 <= nx < box.shape[2]
                    ):
                        matrix[current, box[nz, ny, nx]] += 1
    return matrix


@pytest.mark.unit
def test_spatial_interaction_matrix_matches_v01_reference() -> None:
    """The vectorised kernel reproduces the v0.1 triple loop exactly."""
    rng = np.random.RandomState(7)
    labels = rng.randint(0, 4, size=(6, 7, 5))
    expected = _reference_msi_matrix(labels, 4)
    actual = spatial_interaction_matrix(labels, 4)
    np.testing.assert_array_equal(actual, expected)
    # Symmetry follows from visiting all six directed offsets.
    np.testing.assert_array_equal(actual, actual.T)


@pytest.mark.unit
def test_spatial_interaction_matrix_records_background_border() -> None:
    """A single voxel interacts with the padded background on all six faces."""
    labels = np.zeros((3, 3, 3), dtype=np.int64)
    labels[1, 1, 1] = 2
    matrix = spatial_interaction_matrix(labels, 3)
    assert matrix[2, 0] == 6
    assert matrix[0, 2] == 6
    # v0.1 semantics: background voxels inside the padded bounding box also
    # interact with each other (3x3x3 box -> 96 background-background pairs).
    assert matrix[0, 0] == 96
    assert matrix.sum() == 108


@pytest.mark.unit
def test_spatial_interaction_matrix_empty_array() -> None:
    """An all-background array yields an all-zero matrix."""
    matrix = spatial_interaction_matrix(np.zeros((4, 4, 4), dtype=np.int64), 3)
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 0


@pytest.mark.unit
def test_spatial_interaction_matrix_rejects_non_3d() -> None:
    """Only volumetric label arrays are accepted."""
    with pytest.raises(ValueError):
        spatial_interaction_matrix(np.zeros((4, 4), dtype=np.int64), 2)


@pytest.mark.unit
def test_msi_features_from_matrix_hand_computed() -> None:
    """Hand-computed first/second-order values on a 2-class matrix."""
    matrix = np.array([[2.0, 4.0], [4.0, 6.0]])
    features = msi_features_from_matrix(matrix)
    assert features["firstorder_0_and_1"] == pytest.approx(4.0)
    assert features["firstorder_1_and_1"] == pytest.approx(6.0)
    # denominator = tril sum with row 0 removed = 4 + 6 = 10.
    assert features["firstorder_normalized_0_and_1"] == pytest.approx(0.4)
    assert features["firstorder_normalized_1_and_1"] == pytest.approx(0.6)
    assert features["contrast"] == pytest.approx(0.8)
    assert features["homogeneity"] == pytest.approx(1.2)
    assert features["correlation"] == pytest.approx(-2.0 / 3.0)
    assert features["energy"] == pytest.approx(0.72)


@pytest.mark.unit
def test_msi_features_from_matrix_key_scheme_matches_v01() -> None:
    """Every v0.1 key appears for a 3-class matrix (background + 2 habitats)."""
    features = msi_features_from_matrix(np.ones((3, 3)))
    expected_keys = {
        "firstorder_0_and_1",
        "firstorder_0_and_2",
        "firstorder_1_and_2",
        "firstorder_1_and_1",
        "firstorder_2_and_2",
        "firstorder_normalized_0_and_1",
        "firstorder_normalized_0_and_2",
        "firstorder_normalized_1_and_2",
        "firstorder_normalized_1_and_1",
        "firstorder_normalized_2_and_2",
        "contrast",
        "homogeneity",
        "correlation",
        "energy",
    }
    assert set(features) == expected_keys


@pytest.mark.unit
def test_msi_features_from_matrix_zero_denominator() -> None:
    """A matrix with no non-background interaction yields zero normalisation."""
    matrix = np.zeros((2, 2))
    features = msi_features_from_matrix(matrix)
    assert features["firstorder_normalized_0_and_1"] == 0.0
    assert features["energy"] == 0.0
    # Degenerate marginals fall back to correlation 1.0, as in v0.1.
    assert features["correlation"] == 1.0


@pytest.mark.unit
def test_msi_features_from_matrix_validates_input() -> None:
    """Non-square or negative matrices are rejected."""
    with pytest.raises(ValueError):
        msi_features_from_matrix(np.ones((2, 3)))
    with pytest.raises(ValueError):
        msi_features_from_matrix(np.array([[1.0, -1.0], [0.0, 1.0]]))


@pytest.mark.unit
def test_habitat_volume_fractions() -> None:
    """Fractions are computed over the non-background volume."""
    labels = np.array([[[0, 1], [1, 2]], [[2, 2], [0, 0]]])
    fractions = habitat_volume_fractions(labels, habitat_ids=(1, 2, 3))
    assert fractions[1] == pytest.approx(2 / 5)
    assert fractions[2] == pytest.approx(3 / 5)
    assert fractions[3] == 0.0


@pytest.mark.unit
def test_habitat_volume_fractions_empty() -> None:
    """An empty map yields zero fractions rather than a division error."""
    fractions = habitat_volume_fractions(np.zeros((2, 2, 2), dtype=int), (1, 2))
    assert fractions == {1: 0.0, 2: 0.0}


@pytest.mark.unit
def test_habitat_region_stats_and_ith_score() -> None:
    """Region stats drive the ITH score exactly as the v0.1 formula states."""
    labels = np.zeros((4, 4, 4), dtype=np.int64)
    labels[0, 0, 0] = 1  # isolated region of habitat 1
    labels[3, 3, 3] = 1  # second isolated region of habitat 1
    labels[1, 1, 0] = 2
    labels[1, 1, 1] = 2  # one connected region of habitat 2 (size 2)
    stats = habitat_region_stats(labels)
    assert stats[1] == (2, 1)
    assert stats[2] == (1, 2)
    # total = 4; summation = (1/2) + (2/1) = 2.5; ith = 1 - 2.5/4 = 0.375
    assert ith_score(labels) == pytest.approx(0.375)
    # d_1 = 1 - (1/2)/2 = 0.75; d_2 = 1 - (2/1)/2 = 0.0
    # volume-weighted mean = (2/4)*0.75 + (2/4)*0.0 = 0.375
    dispersion = habitat_ith_dispersion(labels)
    assert dispersion[1] == pytest.approx(0.75)
    assert dispersion[2] == pytest.approx(0.0)
    sizes = {1: 2, 2: 2}
    weighted = sum(dispersion[hid] * sizes[hid] for hid in dispersion) / 4
    assert weighted == pytest.approx(ith_score(labels))


@pytest.mark.unit
def test_ith_score_degenerate_maps() -> None:
    """Empty and single-region maps score zero."""
    assert ith_score(np.zeros((3, 3, 3), dtype=int)) == 0.0
    assert habitat_ith_dispersion(np.zeros((3, 3, 3), dtype=int)) == {}
    labels = np.zeros((3, 3, 3), dtype=int)
    labels[0:2, 0:2, 0:2] = 1
    assert ith_score(labels) == pytest.approx(0.0)
    assert habitat_ith_dispersion(labels)[1] == pytest.approx(0.0)
