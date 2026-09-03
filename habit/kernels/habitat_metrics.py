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

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy import ndimage

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional accelerator
    njit = None
    _HAS_NUMBA = False

__all__ = [
    "spatial_interaction_matrix",
    "msi_features_from_matrix",
    "habitat_volume_fractions",
    "habitat_region_stats",
    "habitat_ith_dispersion",
    "ith_score",
]

#: Positive face-connected offsets. Counting only +z/+y/+x and then adding
#: the transpose recovers the six directed pairs of the v0.1 MSI matrix:
#: each unordered face ``(a, b)`` contributes once to ``M[L(a), L(b)]`` and
#: the reverse direction is filled by ``M + M.T`` (diagonal doubled).
_POS_FACE_OFFSETS_3D: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (0, 1, 0),
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

    Implementation (definition unchanged): count each unordered face once
    along ``+z/+y/+x``, then ``M + M.T`` restores the six directed pairs.
    A numba kernel is used when numba is installed; otherwise a numpy
    ``bincount`` histogram of the same pairs.

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
    n_classes_i = int(n_classes)
    matrix = np.zeros((n_classes_i, n_classes_i), dtype=np.int64)
    nonzero = np.nonzero(labels)
    if nonzero[0].size == 0:
        return matrix
    bbox = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in nonzero
    )
    # Contiguous int64 so the numba kernel can index ``M[L(x), L(x')]``
    # directly and the numpy fallback can pack pairs into ``bincount``.
    box = np.ascontiguousarray(
        np.pad(labels[bbox], 1, mode="constant", constant_values=0),
        dtype=np.int64,
    )
    if _HAS_NUMBA and _count_directed_face_pairs_numba is not None:
        return _count_directed_face_pairs_numba(box, n_classes_i)
    return _count_directed_face_pairs_numpy(box, n_classes_i)


def _count_directed_face_pairs_numpy(
    box: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Numpy fallback: histogram of +z/+y/+x pairs, then symmetrise.

    ``np.bincount`` on packed ``centre * n_classes + neighbour`` indices is
    the same integer histogram as six directed ``np.add.at`` passes, but
    without scatter-add collisions into the tiny ``K x K`` matrix.

    Args:
        box: Padded integer label volume, C-contiguous, values in
            ``[0, n_classes)``.
        n_classes: Square matrix size, including background.

    Returns:
        np.ndarray: Symmetric int64 matrix of shape ``(n_classes, n_classes)``.
    """
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for dz, dy, dx in _POS_FACE_OFFSETS_3D:
        center_src = [slice(None)] * 3
        neighbor_src = [slice(None)] * 3
        for axis, offset in enumerate((dz, dy, dx)):
            # Each positive-face offset is 1 along one axis and 0 on the
            # other two. The zero axes must keep ``slice(None)`` so both
            # views stay the full length of those dimensions.
            #
            # ``slice(None, -offset)`` cannot be used when ``offset == 0``:
            # in Python ``-0 == 0``, so that becomes ``slice(None, 0)``,
            # which is an empty slice. ``centers`` then has shape ``(0,)``
            # while ``neighbors`` is the full ravel, and ``bincount`` /
            # arithmetic raises
            # ``ValueError: operands could not be broadcast together``.
            # Skipping the zero component is definition-preserving: those
            # axes are not shifted.
            if offset == 0:
                continue
            center_src[axis] = slice(None, -offset)
            neighbor_src[axis] = slice(offset, None)
        centers = box[tuple(center_src)].ravel()
        neighbors = box[tuple(neighbor_src)].ravel()
        packed = centers * n_classes + neighbors
        matrix += np.bincount(
            packed, minlength=n_classes * n_classes
        ).reshape(n_classes, n_classes)
    return matrix + matrix.T


if _HAS_NUMBA:

    @njit(cache=True)
    def _count_directed_face_pairs_numba(
        box: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        """Compiled +z/+y/+x face counts, then ``M + M.T``.

        Each in-bounds unordered face is visited once. Adding the transpose
        restores the six directed pairs of the v0.1 triple loop (diagonal
        entries are doubled, matching one increment from each side of the
        face). Integer increments only: the matrix is identical, not
        approximately equal.

        Args:
            box: Padded int64 label volume.
            n_classes: Square matrix size, including background.

        Returns:
            np.ndarray: Symmetric int64 matrix of shape
            ``(n_classes, n_classes)``.
        """
        nz, ny, nx = box.shape
        matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    current = box[z, y, x]
                    if z + 1 < nz:
                        matrix[current, box[z + 1, y, x]] += 1
                    if y + 1 < ny:
                        matrix[current, box[z, y + 1, x]] += 1
                    if x + 1 < nx:
                        matrix[current, box[z, y, x + 1]] += 1
        out = np.empty((n_classes, n_classes), dtype=np.int64)
        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    out[i, j] = matrix[i, j] * 2
                else:
                    out[i, j] = matrix[i, j] + matrix[j, i]
        return out

else:  # pragma: no cover - no numba
    _count_directed_face_pairs_numba = None


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


def _crop_nonzero(label_array: np.ndarray) -> np.ndarray:
    """Crop to the bounding box of non-background voxels.

    Connected-component counts and sizes are translation-invariant, so
    dropping the empty field does not change :func:`habitat_region_stats`
    or :func:`ith_score`. One-step maps stored on the full CT lattice
    are otherwise dominated by background voxels.

    Args:
        label_array: Integer habitat labels, ``0`` denoting background.

    Returns:
        np.ndarray: Contiguous crop, or the original array when empty.
    """
    labels = np.asarray(label_array)
    nonzero = np.nonzero(labels)
    if nonzero[0].size == 0:
        return labels
    bbox = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in nonzero)
    return np.ascontiguousarray(labels[bbox])


def habitat_region_stats(label_array: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """
    Measure connected-component fragmentation per habitat.

    Connected components use face connectivity, matching the SimpleITK
    ``ConnectedComponent`` default used by the v0.1 ITH implementation.
    The volume is cropped to the tumour bounding box first; that does
    not change region counts or sizes.

    Args:
        label_array: Integer habitat labels, ``0`` denoting background.

    Returns:
        Mapping of habitat id to ``(num_regions, largest_region_size)`` in
        voxels. Habitats absent from the array do not appear.
    """
    labels = _crop_nonzero(label_array)
    stats: Dict[int, Tuple[int, int]] = {}
    for habitat_id in (int(v) for v in np.unique(labels) if v != 0):
        components, num_regions = ndimage.label(labels == habitat_id)
        if num_regions == 0:
            stats[habitat_id] = (0, 0)
            continue
        # index 0 is the component-map background; skip it
        sizes = np.bincount(components.ravel())[1:]
        stats[habitat_id] = (
            int(num_regions),
            int(sizes.max()) if sizes.size else 0,
        )
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
    labels = _crop_nonzero(label_array)
    stats = habitat_region_stats(labels)
    dispersion: Dict[int, float] = {}
    for habitat_id, (num_regions, largest) in stats.items():
        size = int(np.count_nonzero(labels == habitat_id))
        if num_regions <= 0 or size <= 0:
            dispersion[int(habitat_id)] = 0.0
            continue
        dispersion[int(habitat_id)] = float(1.0 - (largest / num_regions) / size)
    return dispersion


def ith_score(
    label_array: np.ndarray,
    region_stats: Optional[Dict[int, Tuple[int, int]]] = None,
) -> float:
    """
    Compute the ITH score (topological fragmentation) of a habitat map.

    Definition (unchanged from v0.1)::

        ith = 1 - (1 / S_total) * sum_i( S_i,max / n_i )

    where ``S_i,max`` is the largest connected-component size of habitat
    ``i``, ``n_i`` its number of connected regions, and ``S_total`` the total
    non-background voxel count.

    Args:
        label_array: Integer habitat labels, ``0`` denoting background.
        region_stats: Optional precomputed :func:`habitat_region_stats`
            result. Pass this when the caller already labelled components
            so the volume is not walked twice.

    Returns:
        Score in ``[0, 1)``; ``0.0`` for an empty or single-region map.
    """
    labels = np.asarray(label_array)
    total = int(np.count_nonzero(labels))
    if total == 0:
        return 0.0
    stats = habitat_region_stats(labels) if region_stats is None else region_stats
    summation = 0.0
    for num_regions, largest in stats.values():
        if num_regions > 0:
            summation += largest / num_regions
    return float(1.0 - summation / total)
