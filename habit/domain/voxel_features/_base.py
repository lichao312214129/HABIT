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

import dataclasses
from typing import List, Optional, Sequence, Tuple

import numpy as np

from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.image import ImageVolume, MaskVolume
from habit.contracts.provenance import Provenance
from habit.contracts.subject import Subject
from habit.domain.geometry_align import (
    GEOMETRY_ALIGN_METADATA_KEY,
    ON_GEOMETRY_MISMATCH_DEFAULT,
    align_mask_to_reference,
    coerce_on_geometry_mismatch,
)
from habit.exceptions import GeometryError, HABITAPIError
from habit.spec.specs import Spec

__all__ = [
    "resolve_voxel_modalities",
    "resolve_source_modalities",
    "roi_voxels",
    "aligned_image",
    "build_voxel_field",
]


def resolve_source_modalities(
    modality: Optional[str],
    modalities: Sequence[str],
    as_: Optional[str],
    *,
    owner: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Resolve ``(modalities, source_labels)`` from the singular/plural forms.

    Extractors accept either ``modality="T1"`` (the explicit single-modality
    form used inside feature trees) or ``modalities=["T1", "T2"]`` (the
    historical convenience that stacks several modalities in one node). The
    source label is what output columns are named after: the ``as_`` alias
    when given, else the modality name itself.

    Args:
        modality: Single modality key, or ``None``.
        modalities: Modality keys in feature order; must be empty when
            ``modality`` is set.
        as_: Optional output-column alias. Only valid for exactly one
            resolved modality, since it renames ONE source.
        owner: Extractor name used in error messages.

    Returns:
        ``(modalities, source_labels)`` of equal length, in feature order.

    Raises:
        HABITAPIError: If both forms are given, or ``as_`` meets more than
            one modality.
    """
    if modality is not None:
        if modalities:
            raise HABITAPIError(
                f"{owner}: pass either 'modality' (one key) or 'modalities' "
                "(a list), not both."
            )
        resolved: Tuple[str, ...] = (str(modality),)
    else:
        resolved = tuple(str(name) for name in modalities)
    if as_ is not None:
        if len(resolved) != 1:
            raise HABITAPIError(
                f"{owner}: 'as_' renames ONE modality's output and "
                f"therefore requires exactly one modality; got {resolved}."
            )
        labels: Tuple[str, ...] = (str(as_),)
    else:
        labels = resolved
    return resolved, labels


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


def roi_voxels(
    subject: Subject,
    roi: Optional[str],
    *,
    reference_modality: Optional[str] = None,
    on_geometry_mismatch: str = ON_GEOMETRY_MISMATCH_DEFAULT,
) -> Tuple[MaskVolume, np.ndarray, np.ndarray]:
    """
    Return the ROI mask together with its boolean selector and coordinates.

    When the mask and a reference image modality disagree on geometry, the
    default policy nearest-neighbour resamples the mask onto the image grid
    (see :mod:`habit.domain.geometry_align`). Pass
    ``on_geometry_mismatch="strict"`` to raise :class:`GeometryError` instead.

    Args:
        subject: Subject providing the mask (and images for alignment).
        roi: Mask key, or ``None`` for the subject's single mask.
        reference_modality: Image key used as the alignment target. When
            ``None``, the first modality in insertion order is used.
        on_geometry_mismatch: ``"resample_mask"`` (default) or ``"strict"``.

    Returns:
        ``(mask, inside, voxel_index)`` where ``inside`` selects ROI voxels on
        the (possibly resampled) mask grid and ``voxel_index`` lists their
        coordinates in the same C order the selector produces.
    """
    mask = subject.mask(roi)
    policy = coerce_on_geometry_mismatch(on_geometry_mismatch)
    if subject.images:
        ref_key = (
            str(reference_modality)
            if reference_modality is not None
            else next(iter(subject.images))
        )
        reference = subject.image(ref_key)
        mask = align_mask_to_reference(
            mask,
            reference,
            on_geometry_mismatch=policy,
            subject_id=subject.subject_id,
            roi_name=roi,
            reference_label=ref_key,
        )
    inside = np.asarray(mask.data) > 0
    return mask, inside, np.argwhere(inside)


def aligned_image(
    subject: Subject,
    modality: str,
    mask: ImageVolume,
    *,
    owner: str,
    on_geometry_mismatch: str = ON_GEOMETRY_MISMATCH_DEFAULT,
) -> np.ndarray:
    """
    Read one modality after ensuring it shares the ROI's voxel grid.

    Callers should normally obtain ``mask`` from :func:`roi_voxels` (or
    :func:`~habit.domain.geometry_align.align_subject_masks`) so the mask
    already sits on a reference image grid. When a residual mismatch remains
    (for example another modality on a different grid than the reference),
    the default policy resamples the mask onto *this* modality for the
    compatibility check path used by single-modality reads; multi-modality
    extractors that share one ``voxel_index`` still require all modalities
    to agree with the mask returned by :func:`roi_voxels`.

    Args:
        subject: Subject providing the image.
        modality: Modality key to read.
        mask: The ROI mask whose grid the image must match.
        owner: Extractor name used in the error message.
        on_geometry_mismatch: ``"resample_mask"`` (default) or ``"strict"``.
            Under ``resample_mask``, a mismatch raises only when the mask
            cannot be treated as already aligned (see note above); the
            helper still raises :class:`GeometryError` so shared voxel
            indices stay consistent across modalities.

    Returns:
        The modality's voxel array.

    Raises:
        GeometryError: If the modality and the mask are on different grids.
    """
    image = subject.image(modality)
    if image.geometry.is_compatible_with(mask.geometry):
        return np.asarray(image.data)

    policy = coerce_on_geometry_mismatch(on_geometry_mismatch)
    if policy == "resample_mask" and isinstance(mask, MaskVolume):
        # Resample would change which voxels ``inside`` / ``voxel_index``
        # refer to. Refuse here so extractors keep a single consistent grid;
        # alignment belongs in roi_voxels / align_subject_masks.
        pass
    raise GeometryError(
        f"{owner}: subject {subject.subject_id!r} modality {modality!r} "
        "and the ROI mask do not share a compatible voxel grid. "
        "Align the mask onto a reference modality first "
        "(default on_geometry_mismatch=resample_mask via roi_voxels / "
        "SubjectPipeline), or ensure all modalities share one grid."
    )


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
    align_meta = None
    metadata = getattr(mask, "metadata", None)
    if isinstance(metadata, dict):
        align_meta = metadata.get(GEOMETRY_ALIGN_METADATA_KEY)
    if align_meta is None and isinstance(subject.metadata, dict):
        align_meta = subject.metadata.get(GEOMETRY_ALIGN_METADATA_KEY)
    if align_meta is not None:
        provenance = dataclasses.replace(
            provenance,
            notes={**dict(provenance.notes), GEOMETRY_ALIGN_METADATA_KEY: align_meta},
        )
    return VoxelFeatureField(
        subject_id=subject.subject_id,
        feature_names=tuple(names),
        values=matrix,
        voxel_index=voxel_index,
        geometry=mask.geometry,
        provenance=provenance,
    )
