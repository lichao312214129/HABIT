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
"""Nyúl histogram standardization (pure numpy)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["DEFAULT_NYUL_PERCENTILES", "nyul_standardize_volume"]

DEFAULT_NYUL_PERCENTILES: tuple[float, ...] = (
    1.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    99.0,
)


def _target_landmarks(
    percentiles: Sequence[float],
    target_min: float,
    target_max: float,
) -> np.ndarray:
    """Map percentile landmarks onto ``[target_min, target_max]``."""
    return np.asarray(
        [
            target_min + (float(p) / 100.0) * (target_max - target_min)
            for p in percentiles
        ],
        dtype=np.float32,
    )


def _source_landmarks(
    image: np.ndarray,
    percentiles: Sequence[float],
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Intensity values at ``percentiles`` (mask, else positive voxels)."""
    if mask is not None:
        voxels = image[np.asarray(mask) > 0]
    else:
        voxels = image[image > 0]
    if voxels.size == 0:
        voxels = image.ravel()
    return np.percentile(voxels, list(percentiles)).astype(np.float32)


def _piecewise_linear_map(
    image: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Map ``image`` from source landmarks onto target landmarks."""
    output = np.zeros_like(image, dtype=np.float32)
    mask_below = image <= source[0]
    if source[0] != 0:
        output[mask_below] = image[mask_below] * (target[0] / source[0])
    else:
        output[mask_below] = image[mask_below]

    for i in range(len(source) - 1):
        src_low, src_high = source[i], source[i + 1]
        tgt_low, tgt_high = target[i], target[i + 1]
        segment = (image > src_low) & (image <= src_high)
        width = src_high - src_low
        if width > 0:
            slope = (tgt_high - tgt_low) / width
            output[segment] = image[segment] * slope + (tgt_low - slope * src_low)
        else:
            output[segment] = (tgt_low + tgt_high) / 2.0

    mask_above = image > source[-1]
    if np.any(mask_above):
        src_range = source[-1] - source[-2]
        tgt_range = target[-1] - target[-2]
        if src_range > 0:
            slope = tgt_range / src_range
            output[mask_above] = image[mask_above] * slope + (
                target[-1] - slope * source[-1]
            )
        else:
            output[mask_above] = target[-1]
    return output


def nyul_standardize_volume(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    percentiles: Optional[Sequence[float]] = None,
    target_min: float = 0.0,
    target_max: float = 100.0,
) -> np.ndarray:
    """
    Apply Nyúl piecewise-linear histogram standardization.

    Args:
        image: Intensity volume of any numeric dtype.
        mask: Optional ROI; landmarks use ``mask > 0`` when given,
            otherwise all positive voxels (v0.1 behaviour).
        percentiles: Landmark percentiles. Default is the Nyúl decile set.
        target_min: Intensity at percentile 0 on the standard scale.
        target_max: Intensity at percentile 100 on the standard scale.

    Returns:
        ``float32`` volume on the standard scale, same shape as ``image``.
    """
    arr = np.asarray(image, dtype=np.float32)
    marks = (
        DEFAULT_NYUL_PERCENTILES
        if percentiles is None
        else tuple(float(p) for p in percentiles)
    )
    source = _source_landmarks(arr, marks, mask)
    target = _target_landmarks(marks, float(target_min), float(target_max))
    return _piecewise_linear_map(arr, source, target)
