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
"""Guard against docs inventing Registry.create kwargs / Seedable methods.

Mirrors the critical snippets in ``docs/source/api/domain_habitat.rst`` and
related API pages. Sphinx does not execute ``code-block`` examples.
"""

from __future__ import annotations

import pytest

from habit import make_synthetic_cohort
from habit.domain import (
    HabitatAssignerRegistry,
    HabitatFeatureExtractorRegistry,
    HabitatModelFitterRegistry,
    RawVoxelFeatures,
    SupervoxelFeatureExtractorRegistry,
    SupervoxelizerRegistry,
    VoxelFeatureExtractorRegistry,
)
from habit.domain.protocols import Seedable
from habit.exceptions import ConfigurationError


def test_supervoxelizer_docs_use_n_supervoxels_not_sklearn_names() -> None:
    """kmeans/gmm reject sklearn-shaped kwargs; correct name is n_supervoxels."""
    with pytest.raises(ConfigurationError):
        SupervoxelizerRegistry.create("kmeans", n_clusters=40)
    with pytest.raises(ConfigurationError):
        SupervoxelizerRegistry.create("gmm", n_components=40)

    slic = SupervoxelizerRegistry.create("slic", n_supervoxels=50)
    km = SupervoxelizerRegistry.create("kmeans", n_supervoxels=40)
    gmm = SupervoxelizerRegistry.create("gmm", n_supervoxels=40)
    assert isinstance(slic, Seedable)
    assert isinstance(km, Seedable)
    assert isinstance(gmm, Seedable)
    slic.set_random_state(42)
    km.set_random_state(42)
    gmm.set_random_state(42)


def test_nearest_centroid_assigner_requires_model() -> None:
    """Docs must not show a bare create('nearest_centroid')."""
    with pytest.raises(ConfigurationError):
        HabitatAssignerRegistry.create("nearest_centroid")

    cohort = make_synthetic_cohort(n_subjects=2, shape=(10, 10, 10), rng=2)
    voxel = RawVoxelFeatures(modalities=["T1", "T2"])
    km = SupervoxelizerRegistry.create("kmeans", n_supervoxels=6, n_init=3)
    km.set_random_state(2)
    units = [km(voxel(subject)) for subject in cohort]
    fitter = HabitatModelFitterRegistry.create(
        "kmeans", n_habitats=2, n_init=3
    )
    fitter.set_random_state(2)
    model = fitter.fit(units, cohort=cohort)
    assigner = HabitatAssignerRegistry.create("nearest_centroid", model=model)
    assert assigner.model.model_id == model.model_id


def test_habitat_feature_and_voxel_creates_from_api_docs() -> None:
    """Zero-arg / documented creates used on the habitat API page."""
    VoxelFeatureExtractorRegistry.create("raw", modalities=["T1", "T2"])
    SupervoxelFeatureExtractorRegistry.create("mean_voxel_features")
    for name in (
        "msi",
        "ith_score",
        "volume",
        "non_radiomics",
        "traditional",
        "whole_habitat",
        "each_habitat",
    ):
        HabitatFeatureExtractorRegistry.create(name)
