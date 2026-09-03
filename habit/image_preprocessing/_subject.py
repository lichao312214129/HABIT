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
"""Helpers that rebuild a Subject after an image-preprocessing step."""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence

import numpy as np

from habit.contracts.image import ArrayImageRef, ImageVolume, MaskVolume
from habit.contracts.subject import Subject
from habit.exceptions import HABITAPIError

__all__ = [
    "mask_array",
    "rebuild_subject",
    "replace_from_sitk",
    "select_modalities",
]


def select_modalities(
    subject: Subject,
    images: Optional[Sequence[str]],
) -> list[str]:
    """
    Resolve which modality keys a step should process.

    Args:
        subject: Source subject.
        images: Explicit keys, or ``None`` for every image on the subject.

    Returns:
        Modality keys in caller order.

    Raises:
        HABITAPIError: If a requested key is absent.
    """
    keys = list(images) if images else list(subject.images.keys())
    missing = [key for key in keys if key not in subject.images]
    if missing:
        raise HABITAPIError(
            f"Image preprocessor references modalities absent from subject "
            f"{subject.subject_id!r}: {missing}. "
            f"Available: {sorted(subject.images)}."
        )
    return keys


def mask_array(subject: Subject, mask_roi: Optional[str]) -> Optional[np.ndarray]:
    """
    Load an ROI mask array, or ``None`` when no ROI is selected.

    Args:
        subject: Source subject.
        mask_roi: ROI key, or ``None``.

    Returns:
        Label array, or ``None``.

    Raises:
        HABITAPIError: If ``mask_roi`` is set but missing.
    """
    if mask_roi is None:
        return None
    if mask_roi not in subject.masks:
        raise HABITAPIError(
            f"Subject {subject.subject_id!r} has no ROI {mask_roi!r}. "
            f"Available: {sorted(subject.masks)}."
        )
    return np.asarray(subject.mask(mask_roi).data)


def rebuild_subject(
    subject: Subject,
    images: Dict[str, ArrayImageRef],
    masks: Dict[str, ArrayImageRef],
) -> Subject:
    """
    Return a copy of ``subject`` with replaced image/mask refs.

    Args:
        subject: Source subject (metadata is copied).
        images: Full replacement image mapping.
        masks: Full replacement mask mapping.

    Returns:
        New subject; input is not mutated.
    """
    return dataclasses.replace(subject, images=images, masks=masks)


def replace_from_sitk(
    subject: Subject,
    *,
    modality: str,
    sitk_image: object,
    is_mask: bool = False,
    roi_name: Optional[str] = None,
) -> ArrayImageRef:
    """
    Wrap a SimpleITK result as an in-memory image/mask reference.

    Args:
        subject: Source subject (provides subject_id).
        modality: Image modality or ROI name used in ``from_sitk``.
        sitk_image: SimpleITK volume.
        is_mask: Wrap as a mask volume when ``True``.
        roi_name: Unused; kept for call-site clarity.

    Returns:
        In-memory reference with geometry taken from the SimpleITK header.
    """
    del roi_name
    if is_mask:
        volume = MaskVolume.from_sitk(
            sitk_image, modality=modality, subject_id=subject.subject_id
        )
    else:
        volume = ImageVolume.from_sitk(
            sitk_image, modality=modality, subject_id=subject.subject_id
        )
    return ArrayImageRef(array=volume.data, geometry=volume.geometry)
