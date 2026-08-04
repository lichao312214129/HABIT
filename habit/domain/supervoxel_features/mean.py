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
"""Mean voxel features: the default supervoxel description."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from habit.contracts.habitat import Supervoxelization, VoxelFeatureField
from habit.contracts.subject import Subject
from habit.domain.supervoxel_features._base import (
    aggregate_voxel_means,
    with_features,
)
from habit.domain.supervoxel_features.registry import (
    SupervoxelFeatureExtractorRegistry,
)
from habit.spec.specs import Spec

__all__ = ["MeanVoxelFeatures", "MeanVoxelFeaturesParams"]


class MeanVoxelFeaturesParams(BaseModel):
    """Constructor parameters for :class:`MeanVoxelFeatures` (none)."""


@SupervoxelFeatureExtractorRegistry.register("mean_voxel_features")
class MeanVoxelFeatures:
    """
    Describe each supervoxel by the mean of the voxel features within it.

    The v0.1 default (``supervoxel_level: {method: mean_voxel_features()}``)
    and the summary every built-in supervoxelizer already attaches. It exists
    as a registered component for two reasons: a study can state its choice
    explicitly instead of relying on a default, and a partition obtained from
    elsewhere (a saved label map, a third-party segmenter) can be described
    without re-running the supervoxelization.

    Voxel features are not recoverable from a
    :class:`~habit.contracts.habitat.Supervoxelization` alone, so the field
    that produced the partition must be supplied at construction time when
    calling this extractor standalone. Constructed without one, it returns
    the partition's existing features unchanged, which is the idempotent
    behaviour a pipeline needs when the user names the default explicitly.

    Args:
        field: Voxel features to aggregate. Optional; see above.
    """

    def __init__(self, field: Optional[VoxelFeatureField] = None) -> None:
        self._field = field

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name="mean_voxel_features", params={})

    def __call__(
        self,
        subject: Subject,
        partition: Supervoxelization,
    ) -> Supervoxelization:
        """
        Recompute per-supervoxel means over the partition's regions.

        Args:
            subject: Unused; means need no intensity access beyond the voxel
                features already computed. The parameter is part of the
                protocol so every extractor is interchangeable.
            partition: The subject's supervoxel partition.

        Returns:
            The partition with mean features attached.
        """
        if self._field is None:
            return with_features(partition, partition.features, self.spec)
        features = aggregate_voxel_means(self._field, partition.label_array)
        return with_features(partition, features, self.spec)


SupervoxelFeatureExtractorRegistry.register_params_model(
    "mean_voxel_features", MeanVoxelFeaturesParams
)
