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
"""Shared machinery for the built-in voxel feature extractors.

Every family answers the same question -- "one value per ROI voxel per
feature" -- and therefore shares the same preconditions: the ROI defines the
voxel population, each modality must sit on the ROI's grid, and rows are
emitted in the C order of ``np.argwhere(mask > 0)``. Keeping that contract in
one place is what lets different families be concatenated row-for-row.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.image import ImageVolume
from habit.contracts.provenance import Provenance
from habit.contracts.subject import Subject
from habit.exceptions import GeometryError, HABITAPIError
from habit.spec.specs import Spec

__all__ = [
    "resolve_voxel_modalities",
    "roi_voxels",
    "aligned_image",
    "build_voxel_field",
]


def resolve_voxel_modalities(
    subject: Subject,
    modalities: Optional[Sequence[str]],
    *,
    owner: str,
) -> Tuple[str, ...]:
    """
    Resolve which modalities to read, validating them against the subject.

    Args:
        subject: Subject whose images are considered.
        modalities: Requested modality names in feature order, or ``None`` /
            empty for every image the subject carries in insertion order.
        owner: Extractor name used in error messages.

    Returns:
        The validated modality names, in feature order.

    Raises:
        HABITAPIError: If the subject carries no image, or a requested
            modality is absent.
    """
    if not modalities:
        resolved = tuple(subject.images.keys())
        if not resolved:
            raise HABITAPIError(
                f"{owner}: subject {subject.subject_id!r} carries no image."
            )
        return resolved
    resolved = tuple(str(name) for name in modalities)
    missing = [name for name in resolved if name not in subject.images]
    if missing:
        raise HABITAPIError(
            f"{owner}: subject {subject.subject_id!r} does not provide "
            f"modalities {missing}; available: {sorted(subject.images)}."
        )
    return resolved


def roi_voxels(subject: Subject, roi: Optional[str]) -> Tuple[ImageVolume, np.ndarray, np.ndarray]:
    """
    Return the ROI mask together with its boolean selector and coordinates.

    Args:
        subject: Subject providing the mask.
        roi: Mask key, or ``None`` for the subject's single mask.

    Returns:
        ``(mask, inside, voxel_index)`` where ``inside`` selects ROI voxels on
        the mask grid and ``voxel_index`` lists their coordinates in the same
        C order the selector produces.
    """
    mask = subject.mask(roi)
    inside = np.asarray(mask.data) > 0
    return mask, inside, np.argwhere(inside)


def aligned_image(
    subject: Subject,
    modality: str,
    mask: ImageVolume,
    *,
    owner: str,
) -> np.ndarray:
    """
    Read one modality and assert it shares the ROI's voxel grid.

    Args:
        subject: Subject providing the image.
        modality: Modality key to read.
        mask: The ROI mask whose grid the image must match.
        owner: Extractor name used in the error message.

    Returns:
        The modality's voxel array.

    Raises:
        GeometryError: If the modality and the mask are on different grids.
    """
    image = subject.image(modality)
    if not image.geometry.is_compatible_with(mask.geometry):
        raise GeometryError(
            f"{owner}: subject {subject.subject_id!r} modality {modality!r} "
            "and the ROI mask do not share a compatible voxel grid."
        )
    return np.asarray(image.data)


def build_voxel_field(
    subject: Subject,
    mask: ImageVolume,
    voxel_index: np.ndarray,
    feature_names: Sequence[str],
    values: np.ndarray,
    spec: Spec,
) -> VoxelFeatureField:
    """
    Assemble a :class:`VoxelFeatureField` with provenance for one subject.

    Args:
        subject: Subject the features describe.
        mask: ROI mask defining the field's geometry.
        voxel_index: ROI voxel coordinates, matching ``values`` row order.
        feature_names: Column names, in column order.
        values: Voxel-by-feature matrix.
        spec: The extractor's specification, recorded in provenance.

    Returns:
        The per-voxel feature field.

    Raises:
        HABITAPIError: If ``values`` does not describe every ROI voxel with
            one column per feature name.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise HABITAPIError(
            f"{spec.name}: expected a 2-D voxel-by-feature matrix, "
            f"got shape {matrix.shape}."
        )
    if matrix.shape[0] != voxel_index.shape[0]:
        raise HABITAPIError(
            f"{spec.name}: produced {matrix.shape[0]} rows for "
            f"{voxel_index.shape[0]} ROI voxels of subject "
            f"{subject.subject_id!r}."
        )
    names: List[str] = [str(name) for name in feature_names]
    if matrix.shape[1] != len(names):
        raise HABITAPIError(
            f"{spec.name}: produced {matrix.shape[1]} columns for "
            f"{len(names)} feature names."
        )
    provenance = Provenance.source("subject_images").derive(
        produced_by=f"voxel_feature_extractor.{spec.name}",
        spec_fingerprint=spec.fingerprint(),
    )
    return VoxelFeatureField(
        subject_id=subject.subject_id,
        feature_names=tuple(names),
        values=matrix,
        voxel_index=voxel_index,
        geometry=mask.geometry,
        provenance=provenance,
    )
