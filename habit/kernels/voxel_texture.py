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
"""L0 pure-math kernels for voxel-level texture maps.

These are whole-image transforms: an intensity array in, a same-shaped map of
per-voxel texture out. Restricting the result to an ROI is the caller's job,
which is deliberate -- computing on the full image first means voxels at the
ROI border still see their true neighbourhood.

The formulas replicate the v0.1 ``LocalEntropyExtractor`` semantics so that
migrated features stay numerically comparable with published results.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["local_entropy_map"]


def local_entropy_map(
    image: np.ndarray,
    *,
    kernel_size: int = 3,
    bins: int = 32,
) -> np.ndarray:
    """
    Shannon entropy of the intensity histogram in each voxel's neighbourhood.

    Intensities are min-max normalised over the whole array and discretised
    into ``bins`` levels; for each level, a box convolution counts its
    occurrences in every neighbourhood, and the per-level probabilities are
    accumulated into ``-sum(p * log2 p)``.

    Neighbourhood counts are normalised by the full box volume rather than by
    the number of in-image voxels, matching v0.1: near the array border the
    box extends into implicit zeros, so border entropies are damped. This is
    kept deliberately, because the ROI normally sits well inside the image.

    Args:
        image: Intensity array; 3-D in habitat analysis, but any
            dimensionality with a matching cubic box is accepted.
        kernel_size: Neighbourhood edge length in voxels. Even values are
            incremented so the neighbourhood stays centred.
        bins: Number of intensity bins.

    Returns:
        A float64 entropy map with the same shape as ``image``, in bits.

    Raises:
        ValueError: If ``kernel_size`` is not positive or ``bins`` is below 2.
    """
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive; got {kernel_size}.")
    if bins < 2:
        raise ValueError(f"bins must be at least 2; got {bins}.")

    array = np.asarray(image, dtype=np.float64)
    low = float(array.min()) if array.size else 0.0
    high = float(array.max()) if array.size else 0.0
    if high > low:
        normalized = (array - low) / (high - low)
    else:
        # A constant image carries no information; every neighbourhood is
        # identical, so the discretised map is uniformly zero.
        normalized = np.zeros_like(array)

    edge = int(kernel_size) + 1 if int(kernel_size) % 2 == 0 else int(kernel_size)
    footprint = np.ones((edge,) * array.ndim, dtype=np.float64)
    box_volume = float(edge**array.ndim)

    binned = np.round(normalized * (bins - 1)).astype(np.int64)
    entropy = np.zeros_like(normalized, dtype=np.float64)
    for level in range(int(bins)):
        counts = ndimage.convolve(
            (binned == level).astype(np.float64),
            footprint,
            mode="constant",
            cval=0.0,
        )
        probability = counts / box_volume
        with np.errstate(divide="ignore", invalid="ignore"):
            contribution = -probability * np.log2(probability)
        contribution[~np.isfinite(contribution)] = 0.0
        entropy += contribution
    return entropy
