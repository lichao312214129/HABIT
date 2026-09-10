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
"""Display-only bounding-box cropping for zoomed viz figures.

Cropping changes only WHICH voxels are drawn, never their values: spacing
and direction cosines are untouched, so the oriented, physical-extent
rendering in the calling plotter stays valid on the cropped arrays. This is
a presentation zoom (the tumour fills the panel), not a resampling step.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from habit.exceptions import HABITAPIError

__all__ = ["validate_crop_to", "bbox_slices"]


def validate_crop_to(crop_to: str, *, allowed: Tuple[str, ...], caller: str) -> str:
    """
    Normalize and validate a ``crop_to`` mode string.

    Args:
        crop_to: User-supplied mode (case-insensitive).
        allowed: Modes the caller supports, e.g. ``("none", "labels")``.
        caller: Public function name used in error messages.

    Returns:
        The normalized mode (lower-case).

    Raises:
        HABITAPIError: When the mode is not in ``allowed``.
    """
    mode = str(crop_to).strip().lower()
    if mode not in allowed:
        raise HABITAPIError(
            f"{caller}: crop_to must be one of {allowed}; got {crop_to!r}."
        )
    return mode


def bbox_slices(
    mask: np.ndarray,
    pad: int,
    *,
    caller: str,
    mask_name: str,
) -> Tuple[slice, ...]:
    """
    Per-axis slices covering ``mask > 0`` with a ``pad``-voxel margin.

    Args:
        mask: 2D or 3D array; voxels ``> 0`` define the bounding box.
        pad: Extra voxels of anatomical context around the box (clamped at
            the volume edges).
        caller: Public function name used in error messages.
        mask_name: Which mask produced the box (for error messages).

    Returns:
        Tuple of ``slice`` objects, one per array axis.

    Raises:
        HABITAPIError: When the mask is not 2D/3D or has no foreground.
    """
    binary = np.asarray(mask) > 0
    if binary.ndim not in (2, 3):
        raise HABITAPIError(
            f"{caller}: crop_to needs a 2D or 3D {mask_name}; "
            f"got {binary.ndim}D."
        )
    if not np.any(binary):
        raise HABITAPIError(
            f"{caller}: crop_to requested but {mask_name} has no foreground "
            "voxels; nothing to zoom to."
        )
    margin = max(0, int(pad))
    slices = []
    for axis in range(binary.ndim):
        other_axes = tuple(i for i in range(binary.ndim) if i != axis)
        projection = np.any(binary, axis=other_axes)
        hits = np.flatnonzero(projection)
        low = max(0, int(hits[0]) - margin)
        high = min(binary.shape[axis], int(hits[-1]) + margin + 1)
        slices.append(slice(low, high))
    return tuple(slices)
