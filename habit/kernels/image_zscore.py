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
"""Pure-numpy z-score for image volumes.

Medical volumes are often stored as integer types (``int16`` / ``uint16``).
Arithmetic in that dtype truncates fractional and negative z-scores to zero,
so this kernel always promotes to ``float32`` before subtracting the mean
and dividing by the standard deviation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["zscore_normalize_volume"]

#: Floor for standard deviation; below this, use 1.0 to avoid divide-by-zero.
_STD_EPS: float = 1e-10


def zscore_normalize_volume(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    clip_values: Optional[Tuple[float, float]] = None,
    eps: float = _STD_EPS,
) -> np.ndarray:
    """
    Z-score an intensity volume in floating point.

    Args:
        image: Intensity array of any numeric dtype / shape.
        mask: Optional ROI mask (same shape as ``image``). When provided,
            mean and std are computed only where ``mask != 0``; the
            transform is still applied to the full volume.
        clip_values: Optional ``(low, high)`` clip after normalisation.
        eps: Minimum allowed standard deviation before substituting ``1.0``.

    Returns:
        ``float32`` array with the same shape as ``image``.

    Raises:
        ValueError: If ``image`` is empty, or ``mask`` is given but has no
            non-zero voxels.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("zscore_normalize_volume: image must not be empty.")

    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.shape != arr.shape:
            raise ValueError(
                "zscore_normalize_volume: mask shape "
                f"{mask_arr.shape} != image shape {arr.shape}."
            )
        sample = arr[mask_arr != 0]
        if sample.size == 0:
            raise ValueError(
                "zscore_normalize_volume: mask has no non-zero voxels."
            )
    else:
        sample = arr.ravel()

    mean_val = np.float32(sample.mean())
    std_val = float(sample.std())  # population std (ddof=0), matches SitK Sigma
    if std_val < eps:
        std_val = 1.0

    out = (arr - mean_val) / np.float32(std_val)
    if clip_values is not None:
        low, high = float(clip_values[0]), float(clip_values[1])
        out = np.clip(out, low, high)
    return np.asarray(out, dtype=np.float32)
