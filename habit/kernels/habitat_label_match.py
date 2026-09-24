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
the second fit may be cluster 3 of the first. HABIT recovers the
correspondence in exactly one of two ways, chosen by what the two sides
share:

* **overlap** (:func:`match_labels_by_overlap`) -- the maps label the
  same voxels (retest, perturbation, another preprocessing chain,
  another reader). Hungarian assignment on voxel-overlap counts (the
  Prior 2024 ``munkres`` step used by habitat Dice).
* **prototypes** (:func:`match_rows_to_prototypes`) -- the maps label
  different voxels (different patients), so only habitat descriptions
  can be compared. Every subject is matched one-to-one onto shared
  prototypes (Stephens 2000 relabelling; k-means with a cannot-link
  constraint inside each subject). Two subjects are the special case
  of pairwise Hungarian on squared Euclidean distance. Prototypes can
  also be frozen to name new subjects with a trained definition.

Arrays in, arrays / dicts out. No HABIT types, no IO.
"""

from __future__ import annotations

from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "PROTOTYPE_METRICS",
    "adjusted_rand_index",
    "fit_feature_match_scale",
    "habitat_dice_from_mapping",
    "habitat_intensity_centroids",
    "match_labels_by_overlap",
    "match_rows_to_prototypes",
    "overlap_count_table",
    "PrototypeMatch",
    "present_habitat_ids",
    "remap_label_array",
]

FeatureMatchScale = Literal["none", "zscore"]

#: Distances accepted by :func:`match_rows_to_prototypes`. Each one is
#: paired with the prototype update that minimises it, so the alternating
#: search never increases its objective:
#:
#: * ``"sqeuclidean"`` -- squared Euclidean; prototype = mean (k-means).
#: * ``"manhattan"`` -- L1; prototype = per-feature median (k-medians).
#: * ``"cosine"`` -- ``1 - cos``; rows scaled to unit length, prototype =
#:   renormalised mean direction (spherical k-means). Ignores magnitude.
#: * ``"correlation"`` -- ``1 - Pearson r`` across features; rows centred
#:   on their own mean then treated as ``"cosine"``. Ignores offset and
#:   magnitude.
PROTOTYPE_METRICS: Tuple[str, ...] = (
    "sqeuclidean",
    "manhattan",
    "cosine",
    "correlation",
)

PrototypeMetric = Literal["sqeuclidean", "manhattan", "cosine", "correlation"]

#: Degenerate column / row scale replaced by 1 so a constant feature
#: becomes 0 after z-score instead of NaN. Also the zero-norm threshold
#: for cosine / correlation rows.
_SCALE_FLOOR: float = 1e-12


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
    Per-habitat summary of every non-background habitat.

    A scalar volume yields a centroid of shape ``(n_habitats, 1)``. A
    volume with a trailing feature axis ``(..., n_features)`` yields
    ``(n_habitats, n_features)`` — multimodal intensities, constructed
    maps, or texture channels are all valid as long as the spatial grid
    matches ``label_array``. Default reduction is the **mean** of voxels
    in that habitat (same quantity k-means stores as a centre).
    ``reduction="median"`` is the test-retest table convention.

    For cross-patient naming pass a feature volume whose values mean the
    same thing in every patient, then feed the rows to
    :func:`match_rows_to_prototypes`. Do not pass a per-tumour MinMax /
    z-score copy: those axes are not comparable across subjects.

    Args:
        image: Feature volume aligned with ``label_array``, or the same
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


def fit_feature_match_scale(
    feature_blocks: Sequence[np.ndarray],
    method: FeatureMatchScale = "zscore",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a locked cohort scaler on stacked unscaled habitat rows.

    Call this once on every patient's unscaled habitat summaries and
    apply ``(rows - location) / scale`` to every block before
    :func:`match_rows_to_prototypes`, so every subject uses the same
    ruler. Keep the pair to rescale new subjects that are named with
    frozen prototypes.

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
    if resolved not in ("none", "zscore"):
        raise ValueError(
            "fit_feature_match_scale: method must be 'none' or 'zscore'; "
            f"got {method!r}."
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


class PrototypeMatch(NamedTuple):
    """
    Result of :func:`match_rows_to_prototypes`.

    Attributes:
        assignments: One int64 vector per input block. Entry ``i`` is the
            prototype index (``0 .. K-1``) of row ``i``, or ``-1`` when the
            row stayed unmatched.
        distances: One float64 vector per block. Distance from each row to
            its prototype (``NaN`` for unmatched rows): Euclidean for
            ``"sqeuclidean"``, L1 for ``"manhattan"``, ``1 - cos`` for
            ``"cosine"``, ``1 - r`` for ``"correlation"``.
        prototypes: Prototypes in the metric's working space, shape
            ``(K, n_features)``: means (``"sqeuclidean"``), medians
            (``"manhattan"``), or unit vectors (``"cosine"``,
            ``"correlation"``). Fitted prototypes are sorted
            lexicographically so the numbering does not depend on which
            block seeded the search; frozen prototypes keep their order.
        objective: Sum of matched costs (squared distance for
            ``"sqeuclidean"``, the distance itself otherwise) plus the
            unmatched cost per unmatched row when ``max_distance`` is set.
            Lower is a tighter cohort grouping.
        n_iter: Assignment / update rounds run by the winning start
            (``1`` for frozen prototypes).
        converged: True when the winning start stopped because the
            assignments no longer changed (always True when frozen).
        init_block: Index of the block whose rows seeded the winning start,
            ``-1`` for frozen prototypes.
        metric: The metric used, one of :data:`PROTOTYPE_METRICS`.
    """

    assignments: Tuple[np.ndarray, ...]
    distances: Tuple[np.ndarray, ...]
    prototypes: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    init_block: int
    metric: str


def _metric_rows(rows: np.ndarray, metric: str, what: str) -> np.ndarray:
    """
    Move rows into the working space of ``metric``.

    ``"cosine"`` compares directions, so every row is scaled to unit
    length; ``"correlation"`` first subtracts each row's own mean across
    features (Pearson r is the cosine of centred rows). The other metrics
    use the rows as given. A zero-length row has no direction, so it is
    rejected instead of being given an arbitrary cost.

    Args:
        rows: Matrix ``(n_rows, n_features)``.
        metric: One of :data:`PROTOTYPE_METRICS`.
        what: Name used in the error message (for example ``"block 2"``).

    Returns:
        Float64 matrix of the same shape in the working space.

    Raises:
        ValueError: If a cosine / correlation row has zero length.
    """
    matrix = np.asarray(rows, dtype=np.float64)
    if metric not in ("cosine", "correlation") or matrix.shape[0] == 0:
        return matrix
    if metric == "correlation":
        matrix = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms[:, 0] < _SCALE_FLOOR):
        reason = (
            "is constant across features (Pearson r undefined; with one "
            "feature every row is constant)"
            if metric == "correlation"
            else "is all zeros (no direction)"
        )
        raise ValueError(
            f"match_rows_to_prototypes: a row of {what} {reason}; "
            f"metric={metric!r} cannot compare it."
        )
    return matrix / norms


def _metric_cost(rows: np.ndarray, prototypes: np.ndarray, metric: str) -> np.ndarray:
    """
    Assignment cost ``(n_rows, K)`` in the working space.

    ``"sqeuclidean"`` returns the squared distance (the quantity a mean
    minimises); ``"manhattan"`` the L1 distance (minimised by a median);
    ``"cosine"`` / ``"correlation"`` return ``1 - u . p`` for unit rows
    and unit prototypes (maximised agreement of directions).
    """
    if metric in ("cosine", "correlation"):
        return 1.0 - rows @ prototypes.T
    delta = rows[:, None, :] - prototypes[None, :, :]
    if metric == "manhattan":
        return np.sum(np.abs(delta), axis=2)
    return np.sum(delta * delta, axis=2)


def _assign_block_to_prototypes(
    rows: np.ndarray,
    prototypes: np.ndarray,
    metric: str,
    unmatched_cost: Optional[float],
) -> np.ndarray:
    """
    One-to-one assignment of a block's rows onto prototypes.

    The cost is the metric cost that the prototype update minimises, so
    the alternating search never increases the objective. A block
    contributes at most one row per prototype: two habitats of one
    tumour can never be given the same name. With more rows than
    prototypes (frozen prototypes only) the rows that lose stay
    unmatched.

    When ``unmatched_cost`` is set, every row also gets a private "leave
    unmatched" column at that cost. Hungarian then leaves a row unmatched
    exactly when every free prototype costs more; this is a partial
    assignment, not a post-hoc threshold.

    Args:
        rows: Block rows in the working space, shape ``(n_rows, n_features)``.
        prototypes: Current prototypes, shape ``(K, n_features)``.
        metric: One of :data:`PROTOTYPE_METRICS`.
        unmatched_cost: Cost charged for leaving a row unmatched, or
            ``None`` to match as many rows as possible.

    Returns:
        Int64 vector of prototype indices, ``-1`` for unmatched rows.
    """
    n_rows = int(rows.shape[0])
    assignment = np.full(n_rows, -1, dtype=np.int64)
    if n_rows == 0:
        return assignment
    cost = _metric_cost(rows, prototypes, metric)
    n_prototypes = int(prototypes.shape[0])
    if unmatched_cost is not None:
        # Diagonal dummy block: row i may only use its own dummy column.
        dummy = np.full((n_rows, n_rows), np.inf, dtype=np.float64)
        np.fill_diagonal(dummy, float(unmatched_cost))
        cost = np.hstack((cost, dummy))
    row_index, column_index = linear_sum_assignment(cost)
    for row, column in zip(row_index.tolist(), column_index.tolist()):
        if column < n_prototypes:
            assignment[row] = column
    return assignment


def _update_prototypes(
    blocks: Sequence[np.ndarray],
    assignments: Sequence[np.ndarray],
    previous: np.ndarray,
    metric: str,
) -> np.ndarray:
    """
    Move every prototype to the centre that minimises its metric cost.

    Mean for ``"sqeuclidean"``, per-feature median for ``"manhattan"``,
    renormalised mean for ``"cosine"`` / ``"correlation"``. A prototype
    with no assigned row (or whose unit rows cancel out exactly) keeps
    its previous value.
    """
    n_prototypes, n_features = previous.shape
    updated = previous.copy()
    if metric == "manhattan":
        members: List[List[np.ndarray]] = [[] for _ in range(n_prototypes)]
        for rows, assignment in zip(blocks, assignments):
            for row, prototype in zip(rows, assignment.tolist()):
                if prototype >= 0:
                    members[prototype].append(row)
        for prototype, rows in enumerate(members):
            if rows:
                updated[prototype] = np.median(np.vstack(rows), axis=0)
        return updated
    sums = np.zeros((n_prototypes, n_features), dtype=np.float64)
    counts = np.zeros(n_prototypes, dtype=np.int64)
    for rows, assignment in zip(blocks, assignments):
        matched = assignment >= 0
        if not np.any(matched):
            continue
        np.add.at(sums, assignment[matched], rows[matched])
        np.add.at(counts, assignment[matched], 1)
    filled = counts > 0
    if metric in ("cosine", "correlation"):
        norms = np.linalg.norm(sums, axis=1)
        filled &= norms >= _SCALE_FLOOR
        updated[filled] = sums[filled] / norms[filled, None]
        return updated
    updated[filled] = sums[filled] / counts[filled, None]
    return updated


def _prototype_objective(
    blocks: Sequence[np.ndarray],
    assignments: Sequence[np.ndarray],
    prototypes: np.ndarray,
    metric: str,
    unmatched_cost: Optional[float],
) -> float:
    """Sum of matched metric costs plus the unmatched penalty."""
    total = 0.0
    for rows, assignment in zip(blocks, assignments):
        matched = assignment >= 0
        if np.any(matched):
            cost = _metric_cost(rows[matched], prototypes, metric)
            total += float(np.sum(cost[np.arange(cost.shape[0]), assignment[matched]]))
        if unmatched_cost is not None:
            total += float(unmatched_cost) * int(np.count_nonzero(~matched))
    return total


def match_rows_to_prototypes(
    blocks: Sequence[np.ndarray],
    *,
    metric: PrototypeMetric = "sqeuclidean",
    max_distance: Optional[float] = None,
    max_iter: int = 100,
    prototypes: Optional[np.ndarray] = None,
) -> PrototypeMatch:
    """
    Name habitats across a cohort by matching them to shared prototypes.

    Pairwise matching to one reference subject biases the names toward
    that subject and is not transitive (A→B→C can disagree with A→C).
    This operator instead alternates two steps until the assignments stop
    changing (Stephens 2000 relabelling; equivalently k-means with a
    cannot-link constraint between rows of the same block, Wagstaff 2001):

    1. **Assign** -- every block is matched one-to-one onto the current
       prototypes by Hungarian assignment on the metric cost.
    2. **Update** -- every prototype moves to the centre that minimises
       that cost (see :data:`PROTOTYPE_METRICS`).

    Both steps can only lower the objective, so the loop terminates. The
    search is started once from every block that has exactly ``K`` rows
    and the lowest-objective result wins, so no single subject decides
    the outcome. Final prototypes are sorted lexicographically so the
    numbering is reproducible regardless of block order.

    ``K`` is the largest block row count. Every row of every block
    therefore finds a distinct prototype: no subject loses or merges a
    habitat, and a block with fewer rows simply leaves some prototypes
    empty. With two blocks and ``"sqeuclidean"`` the result is exactly
    pairwise Hungarian on squared Euclidean distance (a matched pair's
    within-group sum of squares is half its squared distance), found at
    the global optimum.

    Metric notes. ``"sqeuclidean"`` (default) and ``"manhattan"`` compare
    values, so a strongly and a weakly enhancing habitat differ.
    ``"cosine"`` compares only the direction of the feature vector:
    rows ``(0.5, 0.6)`` and ``(2.0, 2.4)`` are identical to it, and with
    one feature of constant sign every row looks the same.
    ``"correlation"`` compares only the profile shape across features:
    with two features every centred row is one of two directions, and
    with one feature it is undefined (rejected). Use cosine /
    correlation only when many features describe a habitat and their
    shape, not their level, is what defines it.

    ``prototypes`` freezes the definition: rows are assigned once to the
    given prototypes (in the order given), nothing is updated, and ``K``
    is ``prototypes.shape[0]``. Use it to name a validation cohort or a
    new patient with prototypes fitted elsewhere; refitting would let the
    new subjects change the definition. A block with more rows than
    ``K`` leaves the extra rows unmatched.

    Rows must already live in one comparable space; this kernel does not
    rescale columns. Arrays in, arrays out.

    Args:
        blocks: One ``(n_habitats, n_features)`` matrix per subject.
            Empty blocks (a subject without habitats) are allowed.
        metric: One of :data:`PROTOTYPE_METRICS`. Default
            ``"sqeuclidean"``.
        max_distance: Optional distance (in the units reported by
            ``distances``) above which a row is left unmatched instead of
            being forced onto a prototype. ``None`` (default) names every
            row that has a free prototype.
        max_iter: Upper bound on assignment / update rounds per start.
        prototypes: Optional frozen prototypes ``(K, n_features)`` in the
            same units as ``blocks`` (for cosine / correlation any
            positive scaling is fine; rows are normalised here).

    Returns:
        A :class:`PrototypeMatch`.

    Raises:
        ValueError: If blocks are not 2-D / finite / equal width, a
            cosine / correlation row has zero length, or a parameter is
            out of range.
    """
    resolved_metric = str(metric).strip().lower()
    if resolved_metric not in PROTOTYPE_METRICS:
        raise ValueError(
            "match_rows_to_prototypes: metric must be one of "
            f"{PROTOTYPE_METRICS}; got {metric!r}."
        )
    matrices: List[np.ndarray] = []
    n_features: Optional[int] = None
    for index, block in enumerate(blocks):
        matrix = np.asarray(block, dtype=np.float64)
        if matrix.size == 0:
            # Width is fixed below once a non-empty block reveals it.
            matrices.append(matrix)
            continue
        if matrix.ndim != 2:
            raise ValueError(
                "match_rows_to_prototypes: block "
                f"{index} must be 2-D; got {matrix.ndim}D."
            )
        if n_features is None:
            n_features = int(matrix.shape[1])
        elif int(matrix.shape[1]) != n_features:
            raise ValueError(
                "match_rows_to_prototypes: feature width mismatch at block "
                f"{index}; got {matrix.shape[1]} vs {n_features}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"match_rows_to_prototypes: block {index} must be finite."
            )
        matrices.append(_metric_rows(matrix, resolved_metric, f"block {index}"))
    if n_features is None:
        raise ValueError("match_rows_to_prototypes: no habitat rows to match.")
    matrices = [
        matrix if matrix.size else np.empty((0, n_features), dtype=np.float64)
        for matrix in matrices
    ]
    row_counts = [int(matrix.shape[0]) for matrix in matrices]
    if int(max_iter) < 1:
        raise ValueError(
            f"match_rows_to_prototypes: max_iter must be >= 1; got {max_iter}."
        )
    unmatched_cost: Optional[float] = None
    if max_distance is not None:
        if not np.isfinite(max_distance) or float(max_distance) <= 0:
            raise ValueError(
                "match_rows_to_prototypes: max_distance must be a positive "
                f"finite number; got {max_distance!r}."
            )
        # Only sqeuclidean reports a distance that is not its own cost.
        unmatched_cost = (
            float(max_distance) ** 2
            if resolved_metric == "sqeuclidean"
            else float(max_distance)
        )

    if prototypes is not None:
        frozen = np.asarray(prototypes, dtype=np.float64)
        if frozen.ndim != 2 or frozen.shape[0] == 0 or frozen.shape[1] != n_features:
            raise ValueError(
                "match_rows_to_prototypes: prototypes must have shape "
                f"(K, {n_features}); got {frozen.shape}."
            )
        if not np.all(np.isfinite(frozen)):
            raise ValueError("match_rows_to_prototypes: prototypes must be finite.")
        # Cosine / correlation prototypes must be unit rows in the working
        # space; means and medians are used as given.
        frozen = _metric_rows(frozen, resolved_metric, "prototypes")
        assignments = [
            _assign_block_to_prototypes(rows, frozen, resolved_metric, unmatched_cost)
            for rows in matrices
        ]
        best = PrototypeMatch(
            assignments=tuple(assignments),
            distances=(),
            prototypes=frozen,
            objective=_prototype_objective(
                matrices, assignments, frozen, resolved_metric, unmatched_cost
            ),
            n_iter=1,
            converged=True,
            init_block=-1,
            metric=resolved_metric,
        )
        order = np.arange(frozen.shape[0])
    else:
        # K = largest habitat count, so every habitat can get its own prototype.
        n_proto = max(row_counts)
        # Every subject with the largest habitat count seeds one start.
        seeds = [index for index, count in enumerate(row_counts) if count == n_proto]
        best_fit: Optional[PrototypeMatch] = None
        for seed in seeds:
            current = matrices[seed].copy()
            previous: Optional[List[np.ndarray]] = None
            converged = False
            n_iter = 0
            assignments = []
            for n_iter in range(1, int(max_iter) + 1):
                assignments = [
                    _assign_block_to_prototypes(
                        rows, current, resolved_metric, unmatched_cost
                    )
                    for rows in matrices
                ]
                current = _update_prototypes(
                    matrices, assignments, current, resolved_metric
                )
                if previous is not None and all(
                    np.array_equal(old, new) for old, new in zip(previous, assignments)
                ):
                    converged = True
                    break
                previous = assignments
            objective = _prototype_objective(
                matrices, assignments, current, resolved_metric, unmatched_cost
            )
            # Strict "<" keeps the earliest seed on ties, so results are
            # deterministic for a given block order.
            if best_fit is None or objective < best_fit.objective - 1e-12:
                best_fit = PrototypeMatch(
                    assignments=tuple(assignments),
                    distances=(),
                    prototypes=current,
                    objective=objective,
                    n_iter=n_iter,
                    converged=converged,
                    init_block=seed,
                    metric=resolved_metric,
                )
        assert best_fit is not None
        best = best_fit
        # Canonical numbering: lexicographic on feature columns, first
        # column most significant. np.lexsort treats the LAST key as primary.
        order = np.lexsort(best.prototypes.T[::-1])

    n_out = int(best.prototypes.shape[0])
    new_index = np.empty(n_out, dtype=np.int64)
    new_index[order] = np.arange(n_out, dtype=np.int64)
    final = best.prototypes[order]
    assignments_out: List[np.ndarray] = []
    distances_out: List[np.ndarray] = []
    for rows, assignment in zip(matrices, best.assignments):
        renamed = np.where(assignment >= 0, new_index[np.maximum(assignment, 0)], -1)
        distance = np.full(renamed.size, np.nan, dtype=np.float64)
        matched = renamed >= 0
        if np.any(matched):
            cost = _metric_cost(rows[matched], final, resolved_metric)
            picked = cost[np.arange(cost.shape[0]), renamed[matched]]
            if resolved_metric == "sqeuclidean":
                picked = np.sqrt(np.maximum(picked, 0.0))
            distance[matched] = picked
        assignments_out.append(renamed.astype(np.int64))
        distances_out.append(distance)
    return best._replace(
        assignments=tuple(assignments_out),
        distances=tuple(distances_out),
        prototypes=final,
    )


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
    the same pairing :func:`~habit.precision.habitat_stability` uses.
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


