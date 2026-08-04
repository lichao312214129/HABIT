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
"""Shared machinery for the built-in supervoxelizers."""

from __future__ import annotations

import numpy as np

from habit.contracts.habitat import Supervoxelization, VoxelFeatureField
from habit.domain.supervoxel_features import aggregate_voxel_means
from habit.spec.specs import Spec

__all__ = ["partition_from_voxel_labels"]


def partition_from_voxel_labels(
    field: VoxelFeatureField,
    voxel_labels: np.ndarray,
    spec: Spec,
) -> Supervoxelization:
    """
    Render per-voxel cluster assignments as a supervoxel partition.

    Args:
        field: Per-voxel features the assignments were computed from.
        voxel_labels: One 1-based label per ROI voxel, row-aligned with
            ``field.values``.
        spec: The supervoxelizer's specification.

    Returns:
        The partition (``0`` outside the ROI) summarised by feature means.
    """
    shape = tuple(int(v) for v in field.geometry.shape)
    labels = np.zeros(shape, dtype=np.int32)
    labels[tuple(field.voxel_index.T)] = np.asarray(voxel_labels, dtype=np.int32)
    features = aggregate_voxel_means(field, labels)
    provenance = field.provenance.derive(
        produced_by=f"supervoxelizer.{spec.name}",
        spec_fingerprint=spec.fingerprint(),
    )
    return Supervoxelization(
        subject_id=field.subject_id,
        label_array=labels,
        features=features,
        geometry=field.geometry,
        provenance=provenance,
    )
