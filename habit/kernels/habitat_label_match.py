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

Two habitat analyses typically emit permuted integer ids: cluster 1 of
the second fit may be cluster 3 of the first. This kernel recovers a
``{moving_id: reference_id}`` assignment and applies it.

Matching is one of:

* **features** -- Hungarian assignment on habitat-level texture
  summaries. Pass **unscaled** means or medians (original voxel
  textures, not per-tumour MinMax / z-score cluster centres). The
  operator then column-z-scores the stacked rows (or applies a locked
  cohort ``location`` / ``scale``) so Energy and Coarseness share one
  ruler, then minimises Euclidean or ``1 - correlation`` cost.
  Volume fraction is an optional tie-break only.
* **centroid** -- same Hungarian path with ``standardize="none"`` and
  Euclidean cost. Kept for same-image intensity / spatial means and
  for callers that already live in one commensurate space.
* **overlap** -- Hungarian assignment on maximal voxel overlap (the
  Prior 2024 ``munkres`` step used by habitat Dice). Requires a shared
  grid; it cannot name habitats across patients.

Arrays in, arrays / dicts out. No HABIT types, no IO.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

__all__ = [
    "FEATURE_MATCH_METRICS",
    "FEATURE_MATCH_SCALES",
    "align_label_array",
    "feature_match_cost_matrix",
    "fit_feature_match_scale",
    "adjusted_rand_index",
    "habitat_dice_from_mapping",
    "habitat_intensity_centroids",
    "habitat_spatial_centroids",
    "habitat_volume_fraction_vector",
    "match_label_ids",
    "match_labels_by_centroid",
    "match_labels_by_features",
    "match_labels_by_overlap",
    "overlap_count_table",
    "present_habitat_ids",
    "remap_label_array",
    "standardize_feature_rows",
]

#: Cost metrics for :func:`match_labels_by_features`. Lower is better.
FEATURE_MATCH_METRICS: Tuple[str, ...] = (
    "euclidean",
    "pearson",
    "spearman",
    "cosine",
    "manhattan",
    "chebyshev",
)

#: Column-wise cohort scalers applied to stacked habitat feature rows.
FEATURE_MATCH_SCALES: Tuple[str, ...] = ("none", "zscore")

FeatureMatchMetric = Literal[
    "euclidean", "pearson", "spearman", "cosine", "manhattan", "chebyshev"
]
FeatureMatchScale = Literal["none", "zscore"]

#: Degenerate column / row scale replaced by 1 so a constant feature
#: becomes 0 after z-score instead of NaN.
_SCALE_FLOOR: float = 1e-12

#: When volume fractions are supplied and ``volume_weight`` is left at
#: the default, add this much of |Δvolume| so volume only breaks ties
#: (feature costs after z-score are O(1); 1e-3 cannot flip a real gap).
_DEFAULT_VOLUME_TIEBREAK: float = 1e-3


def present_habitat_ids(label_array: np.ndarray) -> np.ndarray:
    """
    Return the sorted non-background habitat ids of a label image.

    Background (``0``) is dropped before ``unique`` so a full-CT lattice
    with a small ROI does not pay for sorting tens of millions of zeros.

    Args:
        label_array: Integer habitat labels; ``0`` is background.

    Returns:
        Sorted 1-D int64 array of ids strictly greater than zero.
    """
    labels = np.asarray(label_array).reshape(-1)
    if labels.size == 0:
        return np.empty(0, dtype=np.int64)
    positive = labels[labels != 0]
    if positive.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(positive).astype(np.int64, copy=False)


def habitat_intensity_centroids(
    image: np.ndarray,
    label_array: np.ndarray,
    *,
    reduction: Literal["mean", "median"] = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-habitat image / texture summary of every non-background habitat.

    A scalar image yields a centroid of shape ``(n_habitats, 1)``. An image
    with a trailing feature axis ``(..., n_features)`` yields
    ``(n_habitats, n_features)``. Default reduction is the **mean** of
    voxels in that habitat (same quantity k-means stores as a centre).
    ``reduction="median"`` is the test-retest table convention.

    For cross-patient naming pass the **unscaled** texture volume here,
    then :func:`match_labels_by_features` (cohort z-score). Do not pass
    a per-tumour MinMax / z-score copy: those axes are not comparable
    across subjects.

    Args:
        image: Intensity volume aligned with ``label_array``, or the same
            shape plus a trailing feature axis.
        label_array: Integer habitat labels; ``0`` is background.
        reduction: ``"mean"`` (default) or ``"median"``.

    Returns:
        ``(ids, centroids)`` where ``ids`` is sorted and ``centroids[i]``
        is the summary vector of habitat ``ids[i]``.

    Raises:
        ValueError: If the spatial shapes differ or ``reduction`` is unknown.
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
    resolved = str(reduction).strip().lower()
    if resolved not in ("mean", "median"):
        raise ValueError(
            "habitat_intensity_centroids: reduction must be 'mean' or "
            f"'median'; got {reduction!r}."
        )
    ids = present_habitat_ids(labels)
    n_features = int(values.shape[-1])
    centroids = np.zeros((ids.size, n_features), dtype=np.float64)
    flat_labels = labels.reshape(-1)
    flat_values = values.reshape(-1, n_features)
    reducer = np.mean if resolved == "mean" else np.median
    for row, habitat_id in enumerate(ids):
        selector = flat_labels == habitat_id
        if not np.any(selector):
            continue
        centroids[row] = reducer(flat_values[selector], axis=0)
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


def habitat_volume_fraction_vector(
    label_array: np.ndarray,
    habitat_ids: np.ndarray,
) -> np.ndarray:
    """
    Volume fractions aligned with ``habitat_ids`` rows.

    Each value is (voxels of that id) / (non-background voxels). Absent
    ids receive ``0.0``. Used only as a Hungarian tie-break, not as a
    primary matching feature.

    Args:
        label_array: Integer habitat labels; ``0`` is background.
        habitat_ids: Ids aligned with the feature-row matrix.

    Returns:
        1-D float64 vector of length ``habitat_ids.size``.
    """
    labels = np.asarray(label_array).reshape(-1)
    ids = np.asarray(habitat_ids, dtype=np.int64).reshape(-1)
    total = int(np.count_nonzero(labels))
    fractions = np.zeros(ids.size, dtype=np.float64)
    if total == 0 or ids.size == 0:
        return fractions
    positive = labels[labels != 0]
    if positive.size == 0:
        return fractions
    max_id = int(max(int(positive.max()), int(ids.max())))
    counts = np.bincount(positive.astype(np.int64, copy=False), minlength=max_id + 1)
    for row, habitat_id in enumerate(ids.tolist()):
        hid = int(habitat_id)
        if 0 < hid < counts.size:
            fractions[row] = float(counts[hid]) / float(total)
    return fractions


def fit_feature_match_scale(
    feature_blocks: Sequence[np.ndarray],
    method: FeatureMatchScale = "zscore",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a locked cohort scaler on stacked unscaled habitat rows.

    Call this once on every patient's unscaled habitat summaries, then
    pass the returned ``(location, scale)`` into
    :func:`match_labels_by_features` so every subject uses the same
    ruler. Fitting on two maps alone is the pairwise fallback.

    Args:
        feature_blocks: One ``(n_habitats, n_features)`` block per
            subject (or any other grouping). Empty blocks are skipped.
        method: ``"zscore"`` (mean / population std). ``"none"`` returns
            zeros and ones of the feature width.

    Returns:
        ``(location, scale)`` each shaped ``(n_features,)``. Degenerate
        columns get ``scale = 1``.

    Raises:
        ValueError: If no finite row remains or feature widths differ.
    """
    resolved = str(method).strip().lower()
    if resolved not in FEATURE_MATCH_SCALES:
        raise ValueError(
            "fit_feature_match_scale: method must be one of "
            f"{FEATURE_MATCH_SCALES}; got {method!r}."
        )
    stacked = _stack_feature_blocks(feature_blocks, caller="fit_feature_match_scale")
    if resolved == "none":
        n_features = int(stacked.shape[1])
        return (
            np.zeros(n_features, dtype=np.float64),
            np.ones(n_features, dtype=np.float64),
        )
    location = np.mean(stacked, axis=0)
    scale = np.std(stacked, axis=0, ddof=0)
    scale = np.where(np.abs(scale) < _SCALE_FLOOR, 1.0, scale)
    return location, scale


def standardize_feature_rows(
    rows: np.ndarray,
    *,
    method: FeatureMatchScale = "zscore",
    location: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Column-wise standardise habitat feature rows.

    ``location`` / ``scale`` lock a previously fitted cohort scaler.
    When omitted and ``method="zscore"``, statistics are fit on
    ``rows`` itself.

    Args:
        rows: Feature matrix, shape ``(n_habitats, n_features)``.
        method: ``"none"`` or ``"zscore"``.
        location: Optional locked per-feature mean.
        scale: Optional locked per-feature std (zeros already replaced).

    Returns:
        ``(scaled_rows, location, scale)``.

    Raises:
        ValueError: If ``method`` is unknown or the locked stats are
            incomplete / wrong width.
    """
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"standardize_feature_rows: rows must be 2-D; got {matrix.ndim}D."
        )
    have_locked = location is not None or scale is not None
    if have_locked:
        if location is None or scale is None:
            raise ValueError(
                "standardize_feature_rows: location and scale must be "
                "provided together."
            )
        loc = np.asarray(location, dtype=np.float64).reshape(-1)
        scl = np.asarray(scale, dtype=np.float64).reshape(-1)
        if loc.size != matrix.shape[1] or scl.size != matrix.shape[1]:
            raise ValueError(
                "standardize_feature_rows: locked stats width "
                f"{int(loc.size)}/{int(scl.size)} does not match "
                f"{matrix.shape[1]} features."
            )
        scl = np.where(np.abs(scl) < _SCALE_FLOOR, 1.0, scl)
        return (matrix - loc) / scl, loc, scl
    resolved = str(method).strip().lower()
    if resolved == "none":
        n_features = int(matrix.shape[1])
        loc = np.zeros(n_features, dtype=np.float64)
        scl = np.ones(n_features, dtype=np.float64)
        return matrix.copy(), loc, scl
    if resolved != "zscore":
        raise ValueError(
            "standardize_feature_rows: method must be one of "
            f"{FEATURE_MATCH_SCALES}; got {method!r}."
        )
    loc, scl = fit_feature_match_scale((matrix,), method="zscore")
    return (matrix - loc) / scl, loc, scl


def feature_match_cost_matrix(
    moving_features: np.ndarray,
    reference_features: np.ndarray,
    metric: FeatureMatchMetric = "euclidean",
) -> np.ndarray:
    """
    Pairwise cost matrix, shape ``(n_moving, n_reference)``.

    Pearson / Spearman costs are ``1 - r`` so Hungarian still minimises.
    A constant row (undefined correlation) is treated as uncorrelated
    (cost ``1``). Cosine cost is ``1 - cosine``. Correlation metrics
    need at least two features.

    Args:
        moving_features: Already-standardised moving rows.
        reference_features: Already-standardised reference rows.
        metric: One of :data:`FEATURE_MATCH_METRICS`.

    Returns:
        Float64 cost matrix. Lower is a better match.

    Raises:
        ValueError: If widths differ or the metric is unknown / needs
            more features than available.
    """
    moving = np.asarray(moving_features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    if moving.ndim != 2 or reference.ndim != 2:
        raise ValueError(
            "feature_match_cost_matrix: features must be 2-D; "
            f"got {moving.ndim}D and {reference.ndim}D."
        )
    if moving.shape[1] != reference.shape[1]:
        raise ValueError(
            "feature_match_cost_matrix: feature width mismatch; "
            f"got {moving.shape[1]} vs {reference.shape[1]}."
        )
    resolved = str(metric).strip().lower()
    if resolved not in FEATURE_MATCH_METRICS:
        raise ValueError(
            "feature_match_cost_matrix: metric must be one of "
            f"{FEATURE_MATCH_METRICS}; got {metric!r}."
        )
    n_features = int(moving.shape[1])
    if resolved in ("pearson", "spearman") and n_features < 2:
        raise ValueError(
            f"feature_match_cost_matrix: {resolved} needs at least 2 "
            f"features; got {n_features}."
        )
    if resolved == "euclidean":
        delta = moving[:, None, :] - reference[None, :, :]
        return np.sqrt(np.sum(delta * delta, axis=2))
    if resolved == "manhattan":
        return np.sum(np.abs(moving[:, None, :] - reference[None, :, :]), axis=2)
    if resolved == "chebyshev":
        return np.max(np.abs(moving[:, None, :] - reference[None, :, :]), axis=2)
    if resolved == "cosine":
        return _row_cosine_distance(moving, reference)
    if resolved == "pearson":
        return _row_pearson_distance(moving, reference)
    ranked_moving = _rank_rows(moving)
    ranked_reference = _rank_rows(reference)
    return _row_pearson_distance(ranked_moving, ranked_reference)


def match_labels_by_centroid(
    reference_ids: np.ndarray,
    reference_centroids: np.ndarray,
    moving_ids: np.ndarray,
    moving_centroids: np.ndarray,
    *,
    metric: FeatureMatchMetric = "euclidean",
    standardize: FeatureMatchScale = "none",
    reference_volumes: Optional[np.ndarray] = None,
    moving_volumes: Optional[np.ndarray] = None,
    volume_weight: Optional[float] = None,
    location: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> Dict[int, int]:
    """
    Pair moving habitats to reference habitats by centroid distance.

    Default is raw Euclidean Hungarian (same-image intensity / spatial
    means, or any already-commensurate space). For cross-patient
    texture naming prefer :func:`match_labels_by_features`.

    Args:
        reference_ids: Habitat ids aligned with ``reference_centroids`` rows.
        reference_centroids: Feature centroids, shape
            ``(n_reference, n_features)``.
        moving_ids: Habitat ids aligned with ``moving_centroids`` rows.
        moving_centroids: Feature centroids, shape
            ``(n_moving, n_features)``.
        metric: Cost metric. Default ``"euclidean"``.
        standardize: ``"none"`` (default) or ``"zscore"``.
        reference_volumes: Optional volume fractions aligned with
            ``reference_ids`` (tie-break only).
        moving_volumes: Optional volume fractions aligned with
            ``moving_ids``.
        volume_weight: Weight on ``|Δvolume|``. ``None`` uses
            :data:`_DEFAULT_VOLUME_TIEBREAK` when volumes are given.
        location: Locked cohort mean. Requires ``scale``.
        scale: Locked cohort std.

    Returns:
        Mapping ``{moving_id: reference_id}`` for every assigned pair.
        Empty when either side has no habitats.

    Raises:
        ValueError: If centroid shapes, metrics, or locked stats are invalid.
    """
    return match_labels_by_features(
        reference_ids,
        reference_centroids,
        moving_ids,
        moving_centroids,
        metric=metric,
        standardize=standardize,
        reference_volumes=reference_volumes,
        moving_volumes=moving_volumes,
        volume_weight=volume_weight,
        location=location,
        scale=scale,
    )


def match_labels_by_features(
    reference_ids: np.ndarray,
    reference_features: np.ndarray,
    moving_ids: np.ndarray,
    moving_features: np.ndarray,
    *,
    metric: FeatureMatchMetric = "euclidean",
    standardize: FeatureMatchScale = "zscore",
    reference_volumes: Optional[np.ndarray] = None,
    moving_volumes: Optional[np.ndarray] = None,
    volume_weight: Optional[float] = None,
    location: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> Dict[int, int]:
    """
    Pair habitats by cohort-standardised texture summaries.

    Intended Stage-B naming operator:

    1. Rows are **unscaled** habitat means / medians (original textures).
    2. Columns are z-scored on the stacked reference+moving rows, or
       with a locked ``location`` / ``scale`` from
       :func:`fit_feature_match_scale` (full cohort).
    3. Hungarian assignment minimises Euclidean or ``1 - r`` cost.
    4. Volume fraction, if given, is a small additive tie-break.

    Do not pass per-tumour MinMax / z-score cluster centres: those
    axes change when one tumour's own min/max change.

    Args:
        reference_ids: Habitat ids aligned with ``reference_features`` rows.
        reference_features: Unscaled summaries, shape
            ``(n_reference, n_features)``.
        moving_ids: Habitat ids aligned with ``moving_features`` rows.
        moving_features: Unscaled summaries, shape
            ``(n_moving, n_features)``.
        metric: One of :data:`FEATURE_MATCH_METRICS`. Default Euclidean
            after z-score (equal feature weight). ``pearson`` /
            ``spearman`` compare profile shape (``1 - r``).
        standardize: ``"zscore"`` (default) or ``"none"``.
        reference_volumes: Optional fractions aligned with reference rows.
        moving_volumes: Optional fractions aligned with moving rows.
        volume_weight: Weight on ``|Δvolume|``. ``None`` uses a small
            default when both volume vectors are given, else ``0``.
        location: Locked cohort mean. Requires ``scale``.
        scale: Locked cohort std.

    Returns:
        Mapping ``{moving_id: reference_id}``. Empty when either side
        has no habitats. Hungarian is one-to-one: two moving ids never
        share a reference id.

    Raises:
        ValueError: If shapes, metric, or scaler inputs are invalid.
    """
    ref_ids = np.asarray(reference_ids, dtype=np.int64).reshape(-1)
    mov_ids = np.asarray(moving_ids, dtype=np.int64).reshape(-1)
    ref_feat = np.asarray(reference_features, dtype=np.float64)
    mov_feat = np.asarray(moving_features, dtype=np.float64)
    if ref_ids.size == 0 or mov_ids.size == 0:
        return {}
    if ref_feat.ndim != 2 or mov_feat.ndim != 2:
        raise ValueError(
            "match_labels_by_features: feature matrices must be 2-D; "
            f"got {ref_feat.ndim}D and {mov_feat.ndim}D."
        )
    if ref_feat.shape[0] != ref_ids.size or mov_feat.shape[0] != mov_ids.size:
        raise ValueError(
            "match_labels_by_features: feature rows must match ids; "
            f"got {ref_feat.shape[0]} vs {ref_ids.size} and "
            f"{mov_feat.shape[0]} vs {mov_ids.size}."
        )
    if ref_feat.shape[1] != mov_feat.shape[1]:
        raise ValueError(
            "match_labels_by_features: feature width mismatch; "
            f"got {ref_feat.shape[1]} vs {mov_feat.shape[1]}."
        )
    if not np.all(np.isfinite(ref_feat)) or not np.all(np.isfinite(mov_feat)):
        raise ValueError(
            "match_labels_by_features: feature matrices must be finite."
        )
    ref_scaled, mov_scaled = _standardize_pair(
        ref_feat,
        mov_feat,
        method=standardize,
        location=location,
        scale=scale,
    )
    cost = feature_match_cost_matrix(mov_scaled, ref_scaled, metric=metric)
    cost = _add_volume_tiebreak(
        cost,
        reference_volumes=reference_volumes,
        moving_volumes=moving_volumes,
        n_reference=int(ref_ids.size),
        n_moving=int(mov_ids.size),
        volume_weight=volume_weight,
    )
    rows, columns = linear_sum_assignment(cost)
    return {
        int(mov_ids[row]): int(ref_ids[column])
        for row, column in zip(rows.tolist(), columns.tolist())
    }


def _stack_feature_blocks(
    feature_blocks: Sequence[np.ndarray],
    *,
    caller: str,
) -> np.ndarray:
    """Stack habitat-row blocks after checking a shared feature width."""
    matrices: List[np.ndarray] = []
    n_features: Optional[int] = None
    for index, block in enumerate(feature_blocks):
        matrix = np.asarray(block, dtype=np.float64)
        if matrix.size == 0:
            continue
        if matrix.ndim != 2:
            raise ValueError(
                f"{caller}: feature block {index} must be 2-D; got {matrix.ndim}D."
            )
        if n_features is None:
            n_features = int(matrix.shape[1])
        elif int(matrix.shape[1]) != n_features:
            raise ValueError(
                f"{caller}: feature width mismatch at block {index}; "
                f"got {matrix.shape[1]} vs {n_features}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{caller}: feature block {index} must be finite.")
        matrices.append(matrix)
    if not matrices:
        raise ValueError(f"{caller}: no finite habitat rows to fit.")
    return np.vstack(matrices)


def _standardize_pair(
    reference_features: np.ndarray,
    moving_features: np.ndarray,
    *,
    method: FeatureMatchScale,
    location: Optional[np.ndarray],
    scale: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score both sides with one locked or stacked-pair scaler."""
    if location is not None or scale is not None:
        ref_scaled, _, _ = standardize_feature_rows(
            reference_features, method=method, location=location, scale=scale
        )
        mov_scaled, _, _ = standardize_feature_rows(
            moving_features, method=method, location=location, scale=scale
        )
        return ref_scaled, mov_scaled
    resolved = str(method).strip().lower()
    if resolved == "none":
        return reference_features, moving_features
    if resolved != "zscore":
        raise ValueError(
            "match_labels_by_features: standardize must be one of "
            f"{FEATURE_MATCH_SCALES}; got {method!r}."
        )
    stacked = np.vstack((reference_features, moving_features))
    loc, scl = fit_feature_match_scale((stacked,), method="zscore")
    return (reference_features - loc) / scl, (moving_features - loc) / scl


def _add_volume_tiebreak(
    cost: np.ndarray,
    *,
    reference_volumes: Optional[np.ndarray],
    moving_volumes: Optional[np.ndarray],
    n_reference: int,
    n_moving: int,
    volume_weight: Optional[float],
) -> np.ndarray:
    """Add |Δvolume| so volume only breaks near-equal feature costs."""
    have_volumes = reference_volumes is not None or moving_volumes is not None
    if not have_volumes:
        return cost
    if reference_volumes is None or moving_volumes is None:
        raise ValueError(
            "match_labels_by_features: reference_volumes and moving_volumes "
            "must be provided together."
        )
    ref_vf = np.asarray(reference_volumes, dtype=np.float64).reshape(-1)
    mov_vf = np.asarray(moving_volumes, dtype=np.float64).reshape(-1)
    if ref_vf.size != n_reference or mov_vf.size != n_moving:
        raise ValueError(
            "match_labels_by_features: volume lengths must match ids; "
            f"got {ref_vf.size} vs {n_reference} and "
            f"{mov_vf.size} vs {n_moving}."
        )
    weight = _DEFAULT_VOLUME_TIEBREAK if volume_weight is None else float(volume_weight)
    if weight == 0.0:
        return cost
    delta = np.abs(mov_vf[:, None] - ref_vf[None, :])
    return cost + weight * delta


def _rank_rows(matrix: np.ndarray) -> np.ndarray:
    """Average ranks along the feature axis (Spearman)."""
    ranked = np.empty_like(matrix, dtype=np.float64)
    for row in range(matrix.shape[0]):
        ranked[row] = rankdata(matrix[row], method="average")
    return ranked


def _row_cosine_distance(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """1 - cosine similarity; a zero vector is treated as orthogonal."""
    moving_norm = np.linalg.norm(moving, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    moving_ok = moving_norm[:, 0] >= _SCALE_FLOOR
    reference_ok = reference_norm[:, 0] >= _SCALE_FLOOR
    moving_unit = np.divide(moving, moving_norm, where=moving_norm >= _SCALE_FLOOR)
    reference_unit = np.divide(
        reference, reference_norm, where=reference_norm >= _SCALE_FLOOR
    )
    similarity = moving_unit @ reference_unit.T
    similarity[~moving_ok, :] = 0.0
    similarity[:, ~reference_ok] = 0.0
    return 1.0 - similarity


def _row_pearson_distance(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """1 - Pearson r of each pair of rows; a constant row costs 1."""
    moving_centered = moving - moving.mean(axis=1, keepdims=True)
    reference_centered = reference - reference.mean(axis=1, keepdims=True)
    moving_norm = np.linalg.norm(moving_centered, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference_centered, axis=1, keepdims=True)
    moving_ok = moving_norm[:, 0] >= _SCALE_FLOOR
    reference_ok = reference_norm[:, 0] >= _SCALE_FLOOR
    moving_unit = np.divide(
        moving_centered, moving_norm, where=moving_norm >= _SCALE_FLOOR
    )
    reference_unit = np.divide(
        reference_centered, reference_norm, where=reference_norm >= _SCALE_FLOOR
    )
    similarity = moving_unit @ reference_unit.T
    similarity[~moving_ok, :] = 0.0
    similarity[:, ~reference_ok] = 0.0
    return 1.0 - similarity


def _nonzero_union_labels(
    reference: np.ndarray,
    moving: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """1-D int64 labels on the union of non-background voxels.

    Background-only voxels never change present ids or the overlap table,
    so they are dropped before ``unique`` / ``bincount``. The compact
    vectors are equivalent to a full-volume scan for those quantities.

    Args:
        reference: Reference integer label image.
        moving: Moving integer label image, same number of elements.

    Returns:
        ``(ref_nz, mov_nz)`` 1-D int64 arrays, possibly empty.
    """
    ref_flat = np.asarray(reference).reshape(-1)
    mov_flat = np.asarray(moving).reshape(-1)
    keep = (ref_flat != 0) | (mov_flat != 0)
    if not np.any(keep):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return (
        np.asarray(ref_flat[keep], dtype=np.int64),
        np.asarray(mov_flat[keep], dtype=np.int64),
    )


def overlap_count_table(
    reference: np.ndarray,
    moving: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxel-overlap contingency used by Hungarian habitat matching.

    One linearised ``bincount`` replaces a nested ``(ref==id) & (mov==id)``
    scan of the full lattice. Counts are identical to that nested loop:
    only voxels where both sides are non-background contribute.

    Args:
        reference: Reference integer label image.
        moving: Moving integer label image, same shape as ``reference``.

    Returns:
        ``(ref_ids, mov_ids, overlap)`` where ``overlap`` is int64 with
        shape ``(n_moving, n_reference)`` and ``overlap[i, j]`` is the
        number of voxels labelled ``mov_ids[i]`` and ``ref_ids[j]``.
        Ids are sorted, matching :func:`present_habitat_ids`.

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
    ref_nz, mov_nz = _nonzero_union_labels(ref_labels, mov_labels)
    ref_ids = present_habitat_ids(ref_nz)
    mov_ids = present_habitat_ids(mov_nz)
    n_ref = int(ref_ids.size)
    n_mov = int(mov_ids.size)
    if n_ref == 0 or n_mov == 0:
        return ref_ids, mov_ids, np.zeros((n_mov, n_ref), dtype=np.int64)
    both = (ref_nz > 0) & (mov_nz > 0)
    if not np.any(both):
        return ref_ids, mov_ids, np.zeros((n_mov, n_ref), dtype=np.int64)
    # Compact ids so the table size is n_mov * n_ref, not max(id)^2.
    ref_idx = np.searchsorted(ref_ids, ref_nz[both])
    mov_idx = np.searchsorted(mov_ids, mov_nz[both])
    keys = mov_idx * n_ref + ref_idx
    counts = np.bincount(keys, minlength=n_mov * n_ref)
    overlap = counts.reshape(n_mov, n_ref).astype(np.int64, copy=False)
    return ref_ids, mov_ids, overlap


def _adjusted_rand_from_contingency(table: np.ndarray) -> float:
    """Hubert–Arabie ARI from a non-negative contingency table.

    Parameters
    ----------
    table : np.ndarray
        Counts, shape ``(n_moving, n_reference)``. Only jointly labelled
        voxels should be in the table (background already dropped).

    Returns
    -------
    float
        ARI in ``[-1, 1]``. ``NaN`` when fewer than two voxels remain.
        Degenerate partitions (every pair agrees) return ``1.0``.
    """
    counts = np.asarray(table, dtype=np.float64)
    n = float(counts.sum())
    if n < 2.0:
        return float("nan")
    sum_comb = float(np.sum(counts * (counts - 1.0)) / 2.0)
    row = counts.sum(axis=1)
    col = counts.sum(axis=0)
    sum_comb_row = float(np.sum(row * (row - 1.0)) / 2.0)
    sum_comb_col = float(np.sum(col * (col - 1.0)) / 2.0)
    comb_n = n * (n - 1.0) / 2.0
    expected = sum_comb_row * sum_comb_col / comb_n
    maximum = 0.5 * (sum_comb_row + sum_comb_col)
    numer = sum_comb - expected
    denom = maximum - expected
    if denom == 0.0:
        return 1.0 if numer == 0.0 else 0.0
    return float(numer / denom)


def adjusted_rand_index(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Chance-corrected partition agreement (Hubert–Arabie ARI).

    Compares two integer label maps on the same grid. Background ``0`` is
    ignored: only voxels labelled on **both** sides (and inside ``mask``,
    when given) enter the contingency table. The score does not need a
    Hungarian remapping; permuting habitat ids leaves ARI unchanged.

    The work is one ``bincount`` over those voxels plus an ``O(K^2)``
    reduction of the contingency table, so typical ROI sizes (10⁴–10⁶
    voxels, K ≤ 10) finish in milliseconds.

    Parameters
    ----------
    reference : np.ndarray
        Reference integer labels. ``0`` is background.
    moving : np.ndarray
        Moving integer labels, same shape as ``reference``.
    mask : Optional[np.ndarray]
        Optional boolean ROI. When set, voxels outside it are dropped
        before the contingency is built.

    Returns
    -------
    float
        ARI in ``[-1, 1]``. Random agreement is near ``0``; identical
        partitions (up to id permutation) are ``1``. ``NaN`` when fewer
        than two jointly labelled voxels remain.

    Raises
    ------
    ValueError
        If the arrays (or ``mask``) have different shapes.
    """
    ref_labels = np.asarray(reference)
    mov_labels = np.asarray(moving)
    if ref_labels.shape != mov_labels.shape:
        raise ValueError(
            "adjusted_rand_index: label shapes must match; "
            f"got {ref_labels.shape} vs {mov_labels.shape}."
        )
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != ref_labels.shape:
            raise ValueError(
                "adjusted_rand_index: mask shape must match labels; "
                f"got {keep.shape} vs {ref_labels.shape}."
            )
        ref_use = np.where(keep, ref_labels, 0)
        mov_use = np.where(keep, mov_labels, 0)
    else:
        ref_use = ref_labels
        mov_use = mov_labels
    _ref_ids, _mov_ids, overlap = overlap_count_table(ref_use, mov_use)
    return _adjusted_rand_from_contingency(overlap)


def habitat_dice_from_mapping(
    reference: np.ndarray,
    moving: np.ndarray,
    mapping: Mapping[int, int],
) -> List[Tuple[int, Optional[int], float, int, int]]:
    """
    Per-reference-habitat Dice for a ``{moving_id: reference_id}`` pairing.

    Formula matches :func:`~habit.precision.stability.habitat_stability`:
    ``2 * intersection / (n_reference + n_matched)``. Unmatched reference
    habitats score Dice 0 with ``matched_id`` set to ``None``. Counts are
    taken on the union of non-background voxels (equivalent to a full
    volume scan because habitat voxels are never 0).

    Args:
        reference: Reference integer label image.
        moving: Moving integer label image, same shape as ``reference``.
        mapping: Assignment ``{moving_id: reference_id}``.

    Returns:
        Rows ``(habitat_id, matched_id, dice, n_reference, n_matched)``
        in sorted reference-id order.

    Raises:
        ValueError: If the arrays have different shapes.
    """
    ref_labels = np.asarray(reference)
    mov_labels = np.asarray(moving)
    if ref_labels.shape != mov_labels.shape:
        raise ValueError(
            "habitat_dice_from_mapping: label shapes must match; "
            f"got {ref_labels.shape} vs {mov_labels.shape}."
        )
    ref_nz, mov_nz = _nonzero_union_labels(ref_labels, mov_labels)
    ref_ids = present_habitat_ids(ref_nz)
    matched_moving = {int(ref_id): int(mov_id) for mov_id, ref_id in mapping.items()}
    if ref_nz.size:
        ref_counts = np.bincount(ref_nz)
        mov_counts = np.bincount(mov_nz)
    else:
        ref_counts = np.zeros(1, dtype=np.int64)
        mov_counts = np.zeros(1, dtype=np.int64)
    both = (ref_nz > 0) & (mov_nz > 0)
    ref_both = ref_nz[both]
    mov_both = mov_nz[both]
    rows: List[Tuple[int, Optional[int], float, int, int]] = []
    for habitat_id in ref_ids.tolist():
        hid = int(habitat_id)
        n_reference = int(ref_counts[hid]) if hid < ref_counts.size else 0
        if hid not in matched_moving:
            rows.append((hid, None, 0.0, n_reference, 0))
            continue
        moved_id = int(matched_moving[hid])
        n_moved = int(mov_counts[moved_id]) if moved_id < mov_counts.size else 0
        if ref_both.size and n_reference > 0 and n_moved > 0:
            intersection = int(
                np.count_nonzero((ref_both == hid) & (mov_both == moved_id))
            )
        else:
            intersection = 0
        denom = n_reference + n_moved
        dice = (2.0 * intersection / denom) if denom > 0 else 0.0
        rows.append((hid, moved_id, float(dice), n_reference, n_moved))
    return rows


def match_labels_by_overlap(
    reference: np.ndarray,
    moving: np.ndarray,
) -> Dict[int, int]:
    """
    Pair moving habitats to reference habitats by maximal voxel overlap.

    This is the Prior 2024 Hungarian / ``munkres`` step. The assignment is
    the same pairing ``habitat_stability(..., method="overlap")`` uses.
    The overlap table is a one-pass ``bincount`` on non-background voxels,
    then the same ``linear_sum_assignment(-overlap)`` as before.

    Args:
        reference: Reference integer label image.
        moving: Moving integer label image, same shape as ``reference``.

    Returns:
        Mapping ``{moving_id: reference_id}`` for every assigned pair.

    Raises:
        ValueError: If the arrays have different shapes.
    """
    ref_ids, mov_ids, overlap = overlap_count_table(reference, moving)
    if ref_ids.size == 0 or mov_ids.size == 0:
        return {}
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

    A dense look-up table applies the completed mapping in one gather.
    That is equivalent to the previous two-pass shift (swap-safe because
    every present positive id is rewritten from the original array).

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

    # Look-up table is bit-identical to the two-pass shift: every present
    # positive id is in ``complete``, background 0 stays 0. One gather
    # replaces K full-volume equality scans.
    max_src = int(labels.max()) if labels.size else 0
    if max_src <= 0:
        return remapped
    lut = np.arange(max_src + 1, dtype=np.int32)
    for old_id, new_id in complete.items():
        src = int(old_id)
        if 0 < src <= max_src:
            lut[src] = np.int32(new_id)
    return lut[labels]


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
    metric: FeatureMatchMetric = "euclidean",
    standardize: Optional[FeatureMatchScale] = None,
    reduction: Literal["mean", "median"] = "mean",
    volume_tiebreak: bool = False,
    reference_volumes: Optional[np.ndarray] = None,
    moving_volumes: Optional[np.ndarray] = None,
    volume_weight: Optional[float] = None,
    location: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> Dict[int, int]:
    """
    Return ``{moving_id: reference_id}`` for one matching method.

    ``method="centroid"`` (default) is raw Euclidean Hungarian on
    explicit centroids, else mean intensity of ``image`` /
    ``moving_image``, else spatial means. ``method="features"`` is the
    Stage-B path: unscaled summaries, cohort z-score, then Hungarian.
    ``method="overlap"`` pairs by maximal voxel overlap.

    Args:
        reference: Reference integer label image.
        moving: Moving integer label image.
        image: Optional intensity / feature volume for the reference map.
            Also used for the moving map when ``moving_image`` is omitted.
            For ``method="features"`` this must be the **unscaled**
            texture volume.
        moving_image: Optional intensity / feature volume for the moving map.
        method: ``"centroid"`` (default), ``"features"``, or ``"overlap"``.
        reference_centroids: Optional explicit reference summaries.
        moving_centroids: Optional explicit moving summaries.
        reference_ids: Ids aligned with ``reference_centroids`` rows.
            Defaults to ``1 .. n_rows`` when centroids are given, else
            the present labels.
        moving_ids: Ids aligned with ``moving_centroids`` rows.
        metric: Feature cost. Ignored for ``overlap``.
        standardize: ``None`` follows the method default (``none`` for
            ``centroid``, ``zscore`` for ``features``).
        reduction: Mean or median when summaries are taken from ``image``.
        volume_tiebreak: If True and volumes are omitted, derive volume
            fractions from the label arrays.
        reference_volumes: Optional fractions aligned with reference rows.
        moving_volumes: Optional fractions aligned with moving rows.
        volume_weight: Weight on ``|Δvolume|``.
        location: Locked cohort mean. Requires ``scale``.
        scale: Locked cohort std.

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
    if resolved not in ("centroid", "features"):
        raise ValueError(
            "align_label_array: method must be 'centroid', 'features', "
            f"or 'overlap'; got {method!r}."
        )
    scale_method: FeatureMatchScale = (
        "zscore"
        if standardize is None and resolved == "features"
        else "none"
        if standardize is None
        else standardize
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
        ref_cent = np.asarray(reference_centroids, dtype=np.float64)
        mov_cent = np.asarray(moving_centroids, dtype=np.float64)
    elif image is not None:
        ref_image = np.asarray(image)
        mov_image = np.asarray(moving_image) if moving_image is not None else ref_image
        ref_ids, ref_cent = habitat_intensity_centroids(
            ref_image, ref_labels, reduction=reduction
        )
        mov_ids, mov_cent = habitat_intensity_centroids(
            mov_image, mov_labels, reduction=reduction
        )
    else:
        if resolved == "features":
            raise ValueError(
                "align_label_array: method='features' needs unscaled "
                "feature centroids or an unscaled image."
            )
        ref_ids, ref_cent = habitat_spatial_centroids(ref_labels)
        mov_ids, mov_cent = habitat_spatial_centroids(mov_labels)
    ref_vf = reference_volumes
    mov_vf = moving_volumes
    if volume_tiebreak and ref_vf is None and mov_vf is None:
        ref_vf = habitat_volume_fraction_vector(ref_labels, ref_ids)
        mov_vf = habitat_volume_fraction_vector(mov_labels, mov_ids)
    return match_labels_by_features(
        ref_ids,
        ref_cent,
        mov_ids,
        mov_cent,
        metric=metric,
        standardize=scale_method,
        reference_volumes=ref_vf,
        moving_volumes=mov_vf,
        volume_weight=volume_weight,
        location=location,
        scale=scale,
    )


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
    metric: FeatureMatchMetric = "euclidean",
    standardize: Optional[FeatureMatchScale] = None,
    reduction: Literal["mean", "median"] = "mean",
    volume_tiebreak: bool = False,
    reference_volumes: Optional[np.ndarray] = None,
    moving_volumes: Optional[np.ndarray] = None,
    volume_weight: Optional[float] = None,
    location: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Remap ``moving`` ids onto the ``reference`` id space.

    ``method="centroid"`` prefers explicit centroids, else per-habitat
    intensity of ``image`` / ``moving_image``, then spatial means.
    ``method="features"`` z-scores unscaled summaries before Hungarian.
    ``method="overlap"`` uses voxel overlap.

    Args:
        reference: Reference integer label image.
        moving: Moving integer label image.
        image: Optional intensity volume for the reference map. Also used
            for the moving map when ``moving_image`` is omitted.
        moving_image: Optional intensity volume for the moving map.
        method: ``"centroid"`` (default), ``"features"``, or ``"overlap"``.
        reference_centroids: Optional explicit reference centroids.
        moving_centroids: Optional explicit moving centroids.
        reference_ids: Ids aligned with ``reference_centroids`` rows.
            Defaults to ``1 .. n_rows`` when centroids are given, else
            the present labels.
        moving_ids: Ids aligned with ``moving_centroids`` rows.
        metric: Feature cost. Ignored for ``overlap``.
        standardize: ``None`` follows the method default.
        reduction: Mean or median when summaries are taken from ``image``.
        volume_tiebreak: Derive volume fractions from the label arrays
            when explicit volumes are omitted.
        reference_volumes: Optional fractions aligned with reference rows.
        moving_volumes: Optional fractions aligned with moving rows.
        volume_weight: Weight on ``|Δvolume|``.
        location: Locked cohort mean. Requires ``scale``.
        scale: Locked cohort std.

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
        metric=metric,
        standardize=standardize,
        reduction=reduction,
        volume_tiebreak=volume_tiebreak,
        reference_volumes=reference_volumes,
        moving_volumes=moving_volumes,
        volume_weight=volume_weight,
        location=location,
        scale=scale,
    )
    return remap_label_array(
        mov_labels,
        mapping,
        reserved_ids=present_habitat_ids(np.asarray(reference)),
    )
