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
"""L0 pure-math kernels for habitat-level metrics.

These functions are the stable, independently reviewable definitions of
HABIT's habitat metrics (``habit.kernels.habitat_metrics`` is the path cited
when a reviewer asks for the exact formula). They are pure: no IO, no state,
no logging, no configuration -- arrays in, numbers out.

The formulas replicate the semantics of the established v0.1 implementations
(``MSIFeatureExtractor.calculate_MSI_matrix``,
``ITHFeatureExtractor.extract_ith_features``) so that migrated features remain
numerically comparable with previously published results.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import ndimage

__all__ = [
    "spatial_interaction_matrix",
    "msi_features_from_matrix",
    "habitat_volume_fractions",
    "habitat_region_stats",
    "habitat_ith_dispersion",
    "ith_score",
]

#: Face-connected neighbourhood offsets in 3D, matching the v0.1 MSI matrix.
_FACE_OFFSETS_3D: Tuple[Tuple[int, int, int], ...] = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


def spatial_interaction_matrix(
    label_array: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Count face-connected neighbour pairs between habitat classes (MSI matrix).

    Entry ``[i, j]`` is the number of directed face-neighbour pairs with the
    centre voxel labelled ``i`` and the neighbour labelled ``j``. The array is
    cropped to the bounding box of non-zero labels and padded with one zero
    layer first, so boundary voxels record an interaction with background --
    the exact semantics of the v0.1 ``calculate_MSI_matrix``.

    Args:
        label_array: Integer habitat labels, ``0`` denoting background.
        n_classes: Number of classes including background; sets the matrix
            size to ``(n_classes, n_classes)``.

    Returns:
        Symmetric int64 matrix of shape ``(n_classes, n_classes)``; all zeros
        when the array contains no non-zero label.
    """
    labels = np.asarray(label_array)
    if labels.ndim != 3:
        raise ValueError(
            f"spatial_interaction_matrix expects a 3D array; got {labels.ndim}D."
        )
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    nonzero = np.nonzero(labels)
    if nonzero[0].size == 0:
        return matrix
    bbox = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in nonzero
    )
    box = np.pad(labels[bbox], 1, mode="constant", constant_values=0)
    # Count every directed centre->neighbour pair with vectorised slicing;
    # visiting all six offsets is what makes the resulting matrix symmetric.
    for dz, dy, dx in _FACE_OFFSETS_3D:
        center_src = [slice(None)] * 3
        neighbor_src = [slice(None)] * 3
        for axis, offset in enumerate((dz, dy, dx)):
            if offset < 0:
                center_src[axis] = slice(-offset, None)
                neighbor_src[axis] = slice(None, offset)
            elif offset > 0:
                center_src[axis] = slice(None, -offset)
                neighbor_src[axis] = slice(offset, None)
        centers = box[tuple(center_src)].ravel()
        neighbors = box[tuple(neighbor_src)].ravel()
        np.add.at(matrix, (centers, neighbors), 1)
    return matrix


def msi_features_from_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """
    Derive the MSI feature set from a spatial interaction matrix.

    Replicates the v0.1 ``MSIFeatureExtractor.calculate_MSI_features``
    definition exactly, so migrated features stay numerically comparable
    with previously published results:

    * first-order counts ``firstorder_{i}_and_{j}`` for the strict upper
      triangle (including the background row) plus the non-background
      diagonal;
    * the same entries normalised by the sum of the lower triangle with the
      background row removed (zero denominator yields all-zero normalised
      features, as in v0.1);
    * second-order ``contrast`` / ``homogeneity`` / ``correlation`` /
      ``energy`` computed on the normalised matrix (correlation falls back
      to ``1.0`` when a marginal standard deviation vanishes).

    Args:
        matrix: Square non-negative interaction matrix, typically from
            :func:`spatial_interaction_matrix`.

    Returns:
        Feature name to value mapping with the exact v0.1 key scheme.

    Raises:
        ValueError: If the matrix is not square or contains negatives.
    """
    msi_matrix = np.asarray(matrix, dtype=np.float64)
    if msi_matrix.ndim != 2 or msi_matrix.shape[0] != msi_matrix.shape[1]:
        raise ValueError(
            "msi_features_from_matrix expects a square matrix; got shape "
            f"{msi_matrix.shape}."
        )
    if np.any(msi_matrix < 0):
        raise ValueError("msi_features_from_matrix received negative entries.")
    n_classes = msi_matrix.shape[0]

    features: Dict[str, float] = {}
    # First order: off-diagonal upper triangle (including background row 0),
    # then the non-background diagonal.
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            features[f"firstorder_{i}_and_{j}"] = float(msi_matrix[i, j])
    for i in range(1, n_classes):
        features[f"firstorder_{i}_and_{i}"] = float(msi_matrix[i, i])

    # Normalisation denominator: lower triangle including the diagonal,
    # with the background row excluded.
    denominator_mat = np.tril(msi_matrix, k=0)
    denominator_mat[0] = 0
    denominator = float(denominator_mat.sum())
    if denominator == 0.0:
        normalised = np.zeros_like(msi_matrix)
    else:
        normalised = msi_matrix / denominator

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            features[f"firstorder_normalized_{i}_and_{j}"] = float(normalised[i, j])
    for i in range(1, n_classes):
        features[f"firstorder_normalized_{i}_and_{i}"] = float(normalised[i, i])

    i_indices, j_indices = np.indices(normalised.shape)
    squared_delta = (i_indices - j_indices) ** 2
    features["contrast"] = float(np.sum(squared_delta * normalised))
    features["homogeneity"] = float(np.sum(normalised / (1.0 + squared_delta)))

    px = normalised.sum(axis=1)
    py = normalised.sum(axis=0)
    ux = float(np.sum(px * np.arange(px.size)))
    uy = float(np.sum(py * np.arange(py.size)))
    sigmax = float(np.sqrt(np.sum(px * (np.arange(px.size) - ux) ** 2)))
    sigmay = float(np.sqrt(np.sum(py * (np.arange(py.size) - uy) ** 2)))
    if sigmax > 0 and sigmay > 0:
        sum_p_ij = float(np.sum(normalised * i_indices * j_indices))
        features["correlation"] = (sum_p_ij - ux * uy) / (sigmax * sigmay)
    else:
        features["correlation"] = 1.0
    features["energy"] = float(np.sum(normalised**2))
    return features


def habitat_volume_fractions(
    label_array: np.ndarray,
    habitat_ids: Iterable[int],
) -> Dict[int, float]:
    """
    Compute each habitat's voxel fraction of the non-background volume.

    Args:
        label_array: Integer habitat labels, ``0`` denoting background.
        habitat_ids: Habitat ids to report, in output order. Ids absent from
            the array receive fraction ``0.0``.

    Returns:
        Mapping of habitat id to fraction in ``[0, 1]``; all zeros when the
        array contains no non-background voxel.
    """
    labels = np.asarray(label_array)
    total = int(np.count_nonzero(labels))
    fractions: Dict[int, float] = {}
    for habitat_id in habitat_ids:
        count = int(np.count_nonzero(labels == habitat_id))
        fractions[int(habitat_id)] = (count / total) if total > 0 else 0.0
    return fractions


def habitat_region_stats(label_array: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """
    Measure connected-component fragmentation per habitat.

    Connected components use face connectivity, matching the SimpleITK
    ``ConnectedComponent`` default used by the v0.1 ITH implementation.

    Args:
        label_array: Integer habitat labels, ``0`` denoting background.

    Returns:
        Mapping of habitat id to ``(num_regions, largest_region_size)`` in
        voxels. Habitats absent from the array do not appear.
    """
    labels = np.asarray(label_array)
    stats: Dict[int, Tuple[int, int]] = {}
    for habitat_id in (int(v) for v in np.unique(labels) if v != 0):
        components, num_regions = ndimage.label(labels == habitat_id)
        if num_regions == 0:
            stats[habitat_id] = (0, 0)
            continue
        sizes = ndimage.sum_labels(
            np.ones_like(components), components, index=np.arange(1, num_regions + 1)
        )
        stats[habitat_id] = (int(num_regions), int(sizes.max()))
    return stats


def habitat_ith_dispersion(label_array: np.ndarray) -> Dict[int, float]:
    """
    Per-habitat ITH (dispersion) on the same formula as :func:`ith_score`.

    For habitat ``i`` with voxel count ``S_i``, largest component
    ``S_i,max``, and ``n_i`` connected regions::

        d_i = 1 - (S_i,max / n_i) / S_i

    The global ITH score is the volume-weighted mean of these values.
    A single connected blob scores ``0``; many small fragments approach
    ``1``.

    Args:
        label_array: Integer habitat labels, ``0`` denoting background.

    Returns:
        Mapping of habitat id to dispersion in ``[0, 1)``. Habitats
        absent from the array do not appear. An empty map returns ``{}``.
    """
    labels = np.asarray(label_array)
    stats = habitat_region_stats(labels)
    dispersion: Dict[int, float] = {}
    for habitat_id, (num_regions, largest) in stats.items():
        size = int(np.count_nonzero(labels == habitat_id))
        if num_regions <= 0 or size <= 0:
            dispersion[int(habitat_id)] = 0.0
            continue
        dispersion[int(habitat_id)] = float(1.0 - (largest / num_regions) / size)
    return dispersion


def ith_score(label_array: np.ndarray) -> float:
    """
    Compute the ITH score (topological fragmentation) of a habitat map.

    Definition (unchanged from v0.1)::

        ith = 1 - (1 / S_total) * sum_i( S_i,max / n_i )

    where ``S_i,max`` is the largest connected-component size of habitat
    ``i``, ``n_i`` its number of connected regions, and ``S_total`` the total
    non-background voxel count.

    Args:
        label_array: Integer habitat labels, ``0`` denoting background.

    Returns:
        Score in ``[0, 1)``; ``0.0`` for an empty or single-region map.
    """
    labels = np.asarray(label_array)
    total = int(np.count_nonzero(labels))
    if total == 0:
        return 0.0
    summation = 0.0
    for num_regions, largest in habitat_region_stats(labels).values():
        if num_regions > 0:
            summation += largest / num_regions
    return float(1.0 - summation / total)
