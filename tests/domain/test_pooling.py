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
"""Contract tests for the fan-in / fan-out pooling atoms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.habitat_model._base import pool_supervoxel_features
from habit.pipeline.pooling import PooledUnits, fan_in
from habit.exceptions import CompatibilityError, HABITAPIError

from .conftest import make_supervoxelization, two_cluster_units


@pytest.mark.unit
def test_fan_in_pools_in_cohort_order_with_subject_index() -> None:
    """Rows follow cohort order; the index maps every block to its subject."""
    units = two_cluster_units(("P1", "P2", "P3"), supervoxels_per_subject=4)
    pooled = fan_in(units)

    assert isinstance(pooled, PooledUnits)
    assert pooled.subject_ids == ("P1", "P2", "P3")
    assert pooled.boundaries == ((0, 4), (4, 8), (8, 12))
    assert pooled.frame.shape == (12, 2)
    assert pooled.feature_names == ("f1", "f2")
    # Cohort order is preserved row by row (never sorted or shuffled).
    expected = pd.concat(
        [unit.feature_frame() for unit in units], ignore_index=True
    )
    pd.testing.assert_frame_equal(pooled.frame, expected)


@pytest.mark.unit
def test_fan_in_matrix_matches_the_fitters_internal_pooling() -> None:
    """The atom must not grow a second pooling numerics next to the fitter."""
    units = two_cluster_units(("P1", "P2"), supervoxels_per_subject=6)
    matrix, feature_names = pool_supervoxel_features(units)

    pooled = fan_in(units)
    assert pooled.feature_names == feature_names
    np.testing.assert_array_equal(pooled.matrix, matrix)
    assert pooled.matrix.dtype == np.float64


@pytest.mark.unit
def test_fan_out_routes_a_cohort_vector_back_to_subjects() -> None:
    """A pooled row-wise quantity splits along the recorded index."""
    units = two_cluster_units(("P1", "P2", "P3"), supervoxels_per_subject=4)
    pooled = fan_in(units)
    labels = np.arange(pooled.frame.shape[0])

    pieces = pooled.fan_out(labels)
    assert list(pieces) == ["P1", "P2", "P3"]
    np.testing.assert_array_equal(pieces["P1"], labels[0:4])
    np.testing.assert_array_equal(pieces["P2"], labels[4:8])
    np.testing.assert_array_equal(pieces["P3"], labels[8:12])
    # Round-trip: concatenating the pieces rebuilds the cohort vector.
    np.testing.assert_array_equal(
        np.concatenate([pieces[subject_id] for subject_id in pooled.subject_ids]),
        labels,
    )


@pytest.mark.unit
def test_fan_out_rejects_a_misaligned_vector() -> None:
    """A vector that does not span the pooled rows is refused, not truncated."""
    pooled = fan_in(two_cluster_units(("P1", "P2")))
    with pytest.raises(HABITAPIError, match="fan_out expects one value per pooled row"):
        pooled.fan_out(np.zeros(pooled.frame.shape[0] - 1))


@pytest.mark.unit
def test_fan_in_requires_at_least_one_unit() -> None:
    """An empty cohort has nothing to pool."""
    with pytest.raises(HABITAPIError, match="at least one clustering unit"):
        fan_in([])


@pytest.mark.unit
def test_fan_in_rejects_mismatched_feature_columns() -> None:
    """Subjects with different feature spaces cannot share a cohort matrix."""
    first = two_cluster_units(("P1",))[0]
    other_features = pd.DataFrame(
        np.zeros((4, 3)), columns=["f1", "f2", "f3"],
        index=pd.Index(np.arange(1, 5), name="supervoxel"),
    )
    second = make_supervoxelization("P2", other_features)
    with pytest.raises(CompatibilityError, match="provides features"):
        fan_in([first, second])


@pytest.mark.unit
def test_fan_in_rejects_duplicate_subject_ids() -> None:
    """Duplicate ids would make fan-out routing ambiguous."""
    units = two_cluster_units(("P1", "P1"))
    with pytest.raises(HABITAPIError, match="twice"):
        fan_in(units)
