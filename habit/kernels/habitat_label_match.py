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
"""Match and remap habitat label ids across independently clustered maps.

Two habitat analyses of the same anatomy typically emit permuted integer
ids: cluster 1 of the second fit may be cluster 3 of the first. This kernel
recovers a {moving_id: reference_id} assignment and applies it.

Matching is either:

* **centroid** -- Hungarian assignment on Euclidean distance between
  per-habitat feature centroids (the in-memory analogue of the test-retest
  mapper, which pairs habitats by median feature vectors);
* **overlap** -- Hungarian assignment on maximal voxel overlap (the
  Prior 2024 ``munkres`` step used by habitat Dice).

Arrays in, arrays / dicts out. No HABIT types, no IO.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "align_label_array",
    "habitat_intensity_centroids",
    "habitat_spatial_centroids",
    "match_label_ids",
    "match_labels_by_centroid",
    "match_labels_by_overlap",
    "present_habitat_ids",
    "remap_label_array",
]


def present_habitat_ids(label_array: np.ndarray) -> np.ndarray:
    """
    Return the sorted non-background habitat ids of a label image.

    Args:
        label_array: Integer habitat labels; ``0`` is background.

    Returns:
        Sorted 1-D int64 array of ids strictly greater than zero.
    """
    labels = np.unique(np.asarray(label_array))
    return labels[labels != 0].astype(np.int64, copy=False)


def habitat_intensity_centroids(
    image: np.ndarray,
    label_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean image intensity of every non-background habitat.

    A scalar image yields a centroid of shape ``(n_habitats, 1)``. An image
    with a trailing feature axis ``(..., n_features)`` yields
    ``(n_habitats, n_features)``. The reduction is the **mean** of voxels
    in that habitat (same quantity k-means uses for a cluster centre),
    not the median.

    Args:
        image: Intensity volume aligned with ``label_array``, or the same
            shape plus a trailing feature axis.
        label_array: Integer habitat labels; ``0`` is background.

    Returns:
        ``(ids, centroids)`` where ``ids`` is sorted and ``centroids[i]``
        is the mean feature vector of habitat ``ids[i]``.

    Raises:
        ValueError: If the spatial shapes differ.
    """
    labels = np.asarray(label_array)
    values = np.asarray(image, dtype=np.float64)
    if values.shape[: labels.ndim] != labels.shape:
        raise ValueError(
            "habitat_intensity_centroids: image spatial shape "
            f"{values.shape[: labels.ndim]} does not match labels {labels.shape}."
        )
    if values.ndim == labels.ndim:
        values = values[..., np.newaxis]
    elif values.ndim != labels.ndim + 1:
        raise ValueError(
            "habitat_intensity_centroids: image must match labels or have "
            f"one trailing feature axis; got {values.shape} vs {labels.shape}."
        )
    ids = present_habitat_ids(labels)
    n_features = int(values.shape[-1])
    centroids = np.zeros((ids.size, n_features), dtype=np.float64)
    flat_labels = labels.reshape(-1)
    flat_values = values.reshape(-1, n_features)
    for row, habitat_id in enumerate(ids):
        selector = flat_labels == habitat_id
        if not np.any(selector):
            continue
        centroids[row] = flat_values[selector].mean(axis=0)
    return ids, centroids


def habitat_spatial_centroids(
    label_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean voxel-index coordinate of every non-background habitat.

    Used when no image / feature centroids are available. Coordinates are
    ``(z, y, x)`` for 3-D maps and ``(y, x)`` for 2-D maps.

    Args:
        label_array: Integer habitat labels; ``0`` is background.

    Returns:
        ``(ids, centroids)`` with ``centroids`` shaped
        ``(n_habitats, label_array.ndim)``.
    """
    labels = np.asarray(label_array)
    ids = present_habitat_ids(labels)
    coords = np.indices(labels.shape, dtype=np.float64)
    centroids = np.zeros((ids.size, labels.ndim), dtype=np.float64)
    for row, habitat_id in enumerate(ids):
        selector = labels == habitat_id
        if not np.any(selector):
            continue
        centroids[row] = np.array(
            [coords[axis][selector].mean() for axis in range(labels.ndim)],
            dtype=np.float64,
        )
    return ids, centroids


def match_labels_by_centroid(
    reference_ids: np.ndarray,
    reference_centroids: np.ndarray,
    moving_ids: np.ndarray,
    moving_centroids: np.ndarray,
) -> Dict[int, int]:
    """
    Pair moving habitats to reference habitats by centroid distance.

    Hungarian assignment minimises the total Euclidean distance between
    centroid rows. Each moving id is mapped to at most one reference id.

    Args:
        reference_ids: Habitat ids aligned with ``reference_centroids`` rows.
        reference_centroids: Feature centroids, shape
            ``(n_reference, n_features)``.
        moving_ids: Habitat ids aligned with ``moving_centroids`` rows.
        moving_centroids: Feature centroids, shape
            ``(n_moving, n_features)``.

    Returns:
        Mapping ``{moving_id: reference_id}`` for every assigned pair.
        Empty when either side has no habitats.

    Raises:
        ValueError: If centroid feature widths differ.
    """
    ref_ids = np.asarray(reference_ids, dtype=np.int64).reshape(-1)
    mov_ids = np.asarray(moving_ids, dtype=np.int64).reshape(-1)
    ref_cent = np.asarray(reference_centroids, dtype=np.float64)
    mov_cent = np.asarray(moving_centroids, dtype=np.float64)
    if ref_ids.size == 0 or mov_ids.size == 0:
        return {}
    if ref_cent.ndim != 2 or mov_cent.ndim != 2:
        raise ValueError(
            "match_labels_by_centroid: centroids must be 2-D; "
            f"got {ref_cent.ndim}D and {mov_cent.ndim}D."
        )
    if ref_cent.shape[0] != ref_ids.size or mov_cent.shape[0] != mov_ids.size:
        raise ValueError(
            "match_labels_by_centroid: centroid rows must match ids; "
            f"got {ref_cent.shape[0]} vs {ref_ids.size} and "
            f"{mov_cent.shape[0]} vs {mov_ids.size}."
        )
    if ref_cent.shape[1] != mov_cent.shape[1]:
        raise ValueError(
            "match_labels_by_centroid: centroid feature width mismatch; "
            f"got {ref_cent.shape[1]} vs {mov_cent.shape[1]}."
        )
    # Pairwise Euclidean distances: rows = moving, columns = reference.
    delta = mov_cent[:, None, :] - ref_cent[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    rows, columns = linear_sum_assignment(distance)
    return {
        int(mov_ids[row]): int(ref_ids[column])
        for row, column in zip(rows.tolist(), columns.tolist())
    }


def match_labels_by_overlap(
    reference: np.ndarray,
    moving: np.ndarray,
) -> Dict[int, int]:
    """
    Pair moving habitats to reference habitats by maximal voxel overlap.

    This is the Prior 2024 Hungarian / ``munkres`` step. The assignment is
    the same pairing ``habitat_stability(..., method="overlap")`` uses.

    Args:
        reference: Reference integer label image.
        moving: Moving integer label image, same shape as ``reference``.

    Returns:
        Mapping ``{moving_id: reference_id}`` for every assigned pair.

    Raises:
        ValueError: If the arrays have different shapes.
    """
    ref_labels = np.asarray(reference)
    mov_labels = np.asarray(moving)
    if ref_labels.shape != mov_labels.shape:
        raise ValueError(
            "match_labels_by_overlap: label shapes must match; "
            f"got {ref_labels.shape} vs {mov_labels.shape}."
        )
    ref_ids = present_habitat_ids(ref_labels)
    mov_ids = present_habitat_ids(mov_labels)
    if ref_ids.size == 0 or mov_ids.size == 0:
        return {}
    overlap = np.zeros((mov_ids.size, ref_ids.size), dtype=np.int64)
    for column, ref_id in enumerate(ref_ids):
        selector = ref_labels == ref_id
        overlap[:, column] = [
            int(np.count_nonzero(selector & (mov_labels == mov_id)))
            for mov_id in mov_ids
        ]
    rows, columns = linear_sum_assignment(-overlap)
    return {
        int(mov_ids[row]): int(ref_ids[column])
        for row, column in zip(rows.tolist(), columns.tolist())
    }


def remap_label_array(
    label_array: np.ndarray,
    mapping: Mapping[int, int],
    reserved_ids: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """
    Rewrite non-zero labels according to ``{old_id: new_id}``.

    Background (``0``) is never remapped. Matched ids follow ``mapping``.
    Unmatched positive ids are **not** left as-is: that would merge them
    with a habitat that was remapped onto the same integer (for example
    moving ``{1, 2, 3}`` with ``{3: 1, 2: 2}`` would turn leftover ``1``
    and remapped ``3`` into the same color). Leftovers are rewritten to
    unused ids starting at ``max(reserved_ids ∪ mapping values) + 1``,
    in sorted leftover order. ``reserved_ids`` should be the reference
    habitat ids; when omitted, only the mapping targets are reserved.
    An empty ``mapping`` is an identity (no leftover rewrite).

    A two-pass shift avoids collisions when the mapping swaps ids. The
    shift is larger than every old or new id so a leftover assigned to
    ``max(old) + 1`` is not mistaken for a still-shifted voxel.

    Args:
        label_array: Integer habitat labels.
        mapping: ``{moving_id: reference_id}`` assignment.
        reserved_ids: Ids that leftovers must not reuse (typically the
            reference habitat ids). Mapping targets are always reserved.

    Returns:
        A new int32 array with remapped ids.
    """
    labels = np.asarray(label_array, dtype=np.int32)
    remapped = labels.copy()
    if not mapping:
        return remapped

    complete: Dict[int, int] = {int(old_id): int(new_id) for old_id, new_id in mapping.items()}
    reserved = {int(new_id) for new_id in complete.values() if int(new_id) > 0}
    if reserved_ids is not None:
        reserved.update(int(habitat_id) for habitat_id in reserved_ids if int(habitat_id) > 0)
    next_id = (max(reserved) if reserved else 0) + 1
    # Sorted so leftover 1, 4 become max_ref+1, max_ref+2 rather than
    # depending on unique() encounter order.
    for old_id in present_habitat_ids(labels).tolist():
        habitat_id = int(old_id)
        if habitat_id in complete:
            continue
        complete[habitat_id] = next_id
        next_id += 1

    # Offset must exceed every source and destination id. Otherwise a
    # leftover rewritten to max(old)+1 lands in the "still shifted"
    # band and is subtracted back into a colliding original id.
    offset_candidates = [int(labels.max()) if labels.size else 0]
    offset_candidates.extend(complete.keys())
    offset_candidates.extend(complete.values())
    offset = max(offset_candidates)
    if offset <= 0:
        return remapped

    remapped[remapped != 0] += offset
    for old_id, new_id in complete.items():
        remapped[remapped == int(old_id) + offset] = int(new_id)
    still_shifted = remapped > offset
    remapped[still_shifted] -= offset
    return remapped


def match_label_ids(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    image: Optional[np.ndarray] = None,
    moving_image: Optional[np.ndarray] = None,
    method: str = "centroid",
    reference_centroids: Optional[np.ndarray] = None,
    moving_centroids: Optional[np.ndarray] = None,
    reference_ids: Optional[np.ndarray] = None,
    moving_ids: Optional[np.ndarray] = None,
) -> Dict[int, int]:
    """
    Return ``{moving_id: reference_id}`` for one matching method.

    ``method="centroid"`` (default) pairs by Hungarian assignment on
    Euclidean distance between per-habitat **mean** feature vectors:
    explicit centroid matrices if given, else mean intensity of
    ``image`` / ``moving_image``, else spatial (voxel-index) means.
    ``method="overlap"`` pairs by maximal voxel overlap.

    Args:
        reference: Reference integer label image.
        moving: Moving integer label image.
        image: Optional intensity / feature volume for the reference map.
            Also used for the moving map when ``moving_image`` is omitted.
        moving_image: Optional intensity / feature volume for the moving map.
        method: ``"centroid"`` (default) or ``"overlap"``.
        reference_centroids: Optional explicit reference centroids.
        moving_centroids: Optional explicit moving centroids.
        reference_ids: Ids aligned with ``reference_centroids`` rows.
            Defaults to ``1 .. n_rows`` when centroids are given, else
            the present labels.
        moving_ids: Ids aligned with ``moving_centroids`` rows.

    Returns:
        Mapping ``{moving_id: reference_id}`` for every assigned pair.

    Raises:
        ValueError: If ``method`` is unknown, shapes differ, or centroid
            inputs are incomplete.
    """
    ref_labels = np.asarray(reference)
    mov_labels = np.asarray(moving)
    if ref_labels.shape != mov_labels.shape:
        raise ValueError(
            "match_label_ids: label shapes must match; "
            f"got {ref_labels.shape} vs {mov_labels.shape}."
        )
    resolved = str(method).strip().lower()
    if resolved == "overlap":
        return match_labels_by_overlap(ref_labels, mov_labels)
    if resolved != "centroid":
        raise ValueError(
            f"align_label_array: method must be 'centroid' or 'overlap'; "
            f"got {method!r}."
        )
    have_explicit = reference_centroids is not None or moving_centroids is not None
    if have_explicit:
        if reference_centroids is None or moving_centroids is None:
            raise ValueError(
                "align_label_array: reference_centroids and moving_centroids "
                "must be provided together."
            )
        ref_ids = (
            np.asarray(reference_ids, dtype=np.int64)
            if reference_ids is not None
            else np.arange(1, np.asarray(reference_centroids).shape[0] + 1, dtype=np.int64)
        )
        mov_ids = (
            np.asarray(moving_ids, dtype=np.int64)
            if moving_ids is not None
            else np.arange(1, np.asarray(moving_centroids).shape[0] + 1, dtype=np.int64)
        )
        return match_labels_by_centroid(
            ref_ids, reference_centroids, mov_ids, moving_centroids
        )
    if image is not None:
        ref_image = np.asarray(image)
        mov_image = np.asarray(moving_image) if moving_image is not None else ref_image
        ref_ids, ref_cent = habitat_intensity_centroids(ref_image, ref_labels)
        mov_ids, mov_cent = habitat_intensity_centroids(mov_image, mov_labels)
        return match_labels_by_centroid(ref_ids, ref_cent, mov_ids, mov_cent)
    ref_ids, ref_cent = habitat_spatial_centroids(ref_labels)
    mov_ids, mov_cent = habitat_spatial_centroids(mov_labels)
    return match_labels_by_centroid(ref_ids, ref_cent, mov_ids, mov_cent)


def align_label_array(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    image: Optional[np.ndarray] = None,
    moving_image: Optional[np.ndarray] = None,
    method: str = "centroid",
    reference_centroids: Optional[np.ndarray] = None,
    moving_centroids: Optional[np.ndarray] = None,
    reference_ids: Optional[np.ndarray] = None,
    moving_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Remap ``moving`` ids onto the ``reference`` id space.

    Centroid matching prefers explicit centroid matrices (cluster centres
    from two independent fits). Otherwise it uses per-habitat **mean**
    intensity of ``image`` / ``moving_image``, then spatial means.

    Args:
        reference: Reference integer label image.
        moving: Moving integer label image.
        image: Optional intensity volume for the reference map. Also used
            for the moving map when ``moving_image`` is omitted.
        moving_image: Optional intensity volume for the moving map.
        method: ``"centroid"`` (default) or ``"overlap"``.
        reference_centroids: Optional explicit reference centroids.
        moving_centroids: Optional explicit moving centroids.
        reference_ids: Ids aligned with ``reference_centroids`` rows.
            Defaults to ``1 .. n_rows`` when centroids are given, else
            the present labels.
        moving_ids: Ids aligned with ``moving_centroids`` rows.

    Returns:
        Remapped copy of ``moving``. Matched ids use the reference id
        space. Extra moving habitats (more clusters than the reference)
        receive unused ids starting at ``max(reference ids) + 1``.

    Raises:
        ValueError: If ``method`` is unknown or centroid inputs are incomplete.
    """
    mov_labels = np.asarray(moving)
    mapping = match_label_ids(
        reference,
        moving,
        image=image,
        moving_image=moving_image,
        method=method,
        reference_centroids=reference_centroids,
        moving_centroids=moving_centroids,
        reference_ids=reference_ids,
        moving_ids=moving_ids,
    )
    return remap_label_array(
        mov_labels,
        mapping,
        reserved_ids=present_habitat_ids(np.asarray(reference)),
    )
