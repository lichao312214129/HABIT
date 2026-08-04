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
"""Shared machinery for describing supervoxels.

The mean aggregation lives here rather than inside any one supervoxelizer:
every partition needs a default summary, and having a single implementation
is what guarantees ``mean_voxel_features`` and the built-in supervoxelizers
produce byte-identical numbers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.habitat import Supervoxelization, VoxelFeatureField
from habit.spec.specs import Spec

__all__ = [
    "SUPERVOXEL_INDEX_NAME",
    "aggregate_voxel_means",
    "partition_labels",
    "voxel_counts",
    "with_features",
]

#: Index name of every per-supervoxel feature frame. Downstream cohort steps
#: join partitions on it, so it is part of the contract rather than cosmetic.
SUPERVOXEL_INDEX_NAME = "supervoxel"


def aggregate_voxel_means(
    field: VoxelFeatureField,
    label_array: np.ndarray,
) -> pd.DataFrame:
    """
    Average each voxel feature within each supervoxel.

    This is the v0.1 ``calculate_supervoxel_means`` semantics on the v1
    contracts: empty labels are skipped rather than filled with ``NaN``, and
    the row order follows ascending supervoxel id.

    Args:
        field: Per-voxel features for one subject.
        label_array: Supervoxel id per voxel over the full grid; ``0``
            denotes voxels outside the ROI.

    Returns:
        One row per non-empty supervoxel, indexed by supervoxel id, with one
        column per voxel feature.
    """
    voxel_labels = np.asarray(label_array)[tuple(field.voxel_index.T)]
    frame = pd.DataFrame(field.values, columns=list(field.feature_names))
    frame[SUPERVOXEL_INDEX_NAME] = voxel_labels
    # Background rows can appear when a partition leaves ROI voxels
    # unassigned; they are not a supervoxel and must not become a row.
    frame = frame[frame[SUPERVOXEL_INDEX_NAME] > 0]
    features = frame.groupby(SUPERVOXEL_INDEX_NAME, sort=True).mean()
    # Pin the index dtype: it is inherited from the label array, which is
    # int32 here and int64 elsewhere, and a frame produced by one path must
    # compare equal to the same frame produced by another.
    features.index = features.index.astype(np.int64, copy=False)
    features.index.name = SUPERVOXEL_INDEX_NAME
    return features


def partition_labels(partition: Supervoxelization) -> np.ndarray:
    """
    Return the non-background supervoxel ids of a partition, ascending.

    Args:
        partition: The subject's supervoxel partition.

    Returns:
        Sorted array of positive integer labels.

    Raises:
        HABITAPIError: If the partition contains no supervoxel at all.
    """
    labels = np.unique(np.asarray(partition.label_array))
    labels = labels[labels > 0].astype(np.int64, copy=False)
    if labels.size == 0:
        raise HABITAPIError(
            f"Supervoxelization of subject {partition.subject_id!r} contains "
            "no non-zero label; there is nothing to describe."
        )
    return labels


def voxel_counts(partition: Supervoxelization) -> pd.Series:
    """
    Return the voxel count of every supervoxel.

    Args:
        partition: The subject's supervoxel partition.

    Returns:
        Counts indexed by supervoxel id.
    """
    labels = np.asarray(partition.label_array).ravel()
    labels = labels[labels > 0]
    unique, counts = np.unique(labels, return_counts=True)
    series = pd.Series(counts, index=unique.astype(np.int64, copy=False))
    series.index.name = SUPERVOXEL_INDEX_NAME
    return series


def with_features(
    partition: Supervoxelization,
    features: pd.DataFrame,
    spec: Spec,
) -> Supervoxelization:
    """
    Return the same partition carrying newly computed features.

    The label array and geometry are reused unchanged -- an extractor
    describes regions, it never redraws them -- and provenance is chained so
    the record shows which extractor produced the numbers.

    Args:
        partition: The partition being described.
        features: One row per supervoxel, indexed by supervoxel id.
        spec: The extractor's specification.

    Returns:
        A new :class:`Supervoxelization` with ``features`` replaced.
    """
    described = features.copy()
    described.index = described.index.astype(np.int64, copy=False)
    described.index.name = SUPERVOXEL_INDEX_NAME
    described = described.sort_index()
    provenance = partition.provenance.derive(
        produced_by=f"supervoxel_feature_extractor.{spec.name}",
        spec_fingerprint=spec.fingerprint(),
    )
    return Supervoxelization(
        subject_id=partition.subject_id,
        label_array=partition.label_array,
        features=described,
        geometry=partition.geometry,
        provenance=provenance,
    )


def resolve_modality_names(
    subject_modalities: Tuple[str, ...],
    requested: Tuple[str, ...],
    *,
    owner: str,
    subject_id: str,
) -> Tuple[str, ...]:
    """
    Validate requested modality names against what a subject carries.

    Args:
        subject_modalities: Modality keys present on the subject.
        requested: Requested modality names; empty selects all of them.
        owner: Extractor name used in the error message.
        subject_id: Subject identifier used in the error message.

    Returns:
        The modality names to extract from, in the requested order.

    Raises:
        HABITAPIError: If a requested modality is absent.
    """
    if not requested:
        return tuple(subject_modalities)
    missing = [name for name in requested if name not in subject_modalities]
    if missing:
        raise HABITAPIError(
            f"{owner}: subject {subject_id!r} does not provide modalities "
            f"{missing}; available: {sorted(subject_modalities)}."
        )
    return tuple(requested)
