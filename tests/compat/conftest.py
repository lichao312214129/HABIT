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
"""Shared fixtures for the ecosystem-adapter tests.

The synthetic builders live in ``tests.domain.conftest`` (the cloud
constraint forbids real imaging data; small in-memory volumes exercise the
adapters deterministically). This module only re-exports them as fixtures
and adds the small habitat specification the sklearn adapter tests share.
"""

from __future__ import annotations

import pytest

from habit.spec import HabitatSpec, Spec
from tests.domain.conftest import make_feature_table, make_subject

__all__ = ["make_feature_table", "make_subject"]


@pytest.fixture
def direct_spec() -> HabitatSpec:
    """One-step (direct clustering) habitat specification, fully seeded."""
    return HabitatSpec(
        name="compat_direct",
        voxel_feature_extractor=Spec(
            name="raw", params={"modalities": ["T1"], "roi": "tumor"}
        ),
        supervoxelizer=None,
        habitat_model_fitter=Spec(name="kmeans", params={"n_habitats": 2}),
        habitat_assigner=Spec(name="nearest_centroid"),
        habitat_features=(Spec(name="volume"),),
        random_seed=0,
    )


@pytest.fixture
def two_step_spec() -> HabitatSpec:
    """Two-step habitat specification with a SLIC supervoxel stage."""
    return HabitatSpec(
        name="compat_two_step",
        voxel_feature_extractor=Spec(
            name="raw", params={"modalities": ["T1"], "roi": "tumor"}
        ),
        supervoxelizer=Spec(name="slic", params={"n_supervoxels": 8}),
        habitat_model_fitter=Spec(name="kmeans", params={"n_habitats": 2}),
        habitat_assigner=Spec(name="nearest_centroid"),
        habitat_features=(Spec(name="volume"),),
        random_seed=0,
    )
