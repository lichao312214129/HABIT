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
"""SlicSupervoxelizer participates in Seedable like kmeans/gmm."""

from __future__ import annotations

import numpy as np
import pytest

from habit import make_synthetic_cohort
from habit.domain import RawVoxelFeatures, SlicSupervoxelizer
from habit.domain.protocols import Seedable

pytest.importorskip("skimage")


def test_slic_is_seedable_with_default_seed_zero() -> None:
    """SLIC exposes set_random_state; default seed matches other supervoxelizers."""
    svx = SlicSupervoxelizer(n_supervoxels=12, compactness=0.1)
    assert isinstance(svx, Seedable)
    assert svx._seed == 0
    svx.set_random_state(7)
    assert svx._seed == 7


def test_slic_seed_does_not_change_current_skimage_partitions() -> None:
    """Current skimage SLIC ignores RNG; seeds must not silently diverge labels."""
    cohort = make_synthetic_cohort(n_subjects=1, shape=(16, 16, 16), rng=3)
    field = RawVoxelFeatures(modalities=["T1", "T2"])(cohort[0])
    a = SlicSupervoxelizer(n_supervoxels=10, compactness=0.1)
    b = SlicSupervoxelizer(n_supervoxels=10, compactness=0.1)
    a.set_random_state(0)
    b.set_random_state(99)
    la = a(field).label_array
    lb = b(field).label_array
    np.testing.assert_array_equal(la, lb)
