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
"""L0 pure-math kernels for choosing the number of clusters.

This module is the single definition of HABIT's cluster-count selection rules,
cited when a reviewer asks how the habitat count was chosen. It is pure: score
sequences in, an index out -- no IO, no state, no logging, no configuration.

Both the v0.1 clustering stack and the v1.0 domain fitters call these
functions, which is what guarantees a habitat count cannot depend on which
code path computed it.

Selection rules
---------------
* ``maximize`` / ``minimize`` -- take the extreme score (silhouette,
  Calinski-Harabasz, gap / Davies-Bouldin, AIC, BIC).
* ``knee`` -- locate the knee of a convex, decreasing curve (inertia) with
  the Kneedle algorithm.

.. warning::
   **Breaking change in v1.0**: ``elbow`` now resolves to the same Kneedle
   knee-detection rule as ``kneedle``. HABIT v0.1 selected the ``elbow``
   point with a second-derivative rule (``argmax(diff2) + 1``), which
   frequently disagreed with Kneedle and is not the standard elbow criterion.
   Studies that ran with ``selection_method: elbow`` before v1.0 may obtain a
   different habitat count after upgrading; see the CHANGELOG.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

__all__ = [
    "SCORE_DIRECTIONS",
    "MAXIMIZE",
    "MINIMIZE",
    "KNEE",
    "score_direction",
    "knee_index",
    "best_index",
    "vote_best_index",
    "gap_statistic",
]

#: Selection rule: the best score is the largest one.
MAXIMIZE = "maximize"
#: Selection rule: the best score is the smallest one.
MINIMIZE = "minimize"
#: Selection rule: the best score sits at the knee of a decreasing curve.
KNEE = "knee"

#: Validation score -> selection rule. The rule is a property of the score
#: itself (a silhouette is always maximised), so it does not depend on which
#: clustering algorithm produced the curve; algorithms only differ in which
#: scores they *support*.
#:
#: ``elbow`` deliberately shares ``kneedle``'s rule -- see the module warning.
SCORE_DIRECTIONS: Mapping[str, str] = {
    "silhouette": MAXIMIZE,
    "calinski_harabasz": MAXIMIZE,
    "gap": MAXIMIZE,
    "davies_bouldin": MINIMIZE,
    "aic": MINIMIZE,
    "bic": MINIMIZE,
    "inertia": KNEE,
    "kneedle": KNEE,
    "elbow": KNEE,
}


def score_direction(method: str) -> str:
    """
    Return the selection rule for a validation score.

    Args:
        method: Validation score name, e.g. ``"silhouette"`` or ``"elbow"``.

    Returns:
        One of :data:`MAXIMIZE`, :data:`MINIMIZE` or :data:`KNEE`. Unknown
        names fall back to :data:`MAXIMIZE`, matching the v0.1 default.
    """
    return SCORE_DIRECTIONS.get(str(method), MAXIMIZE)


def knee_index(scores: Sequence[float]) -> int:
    """
    Locate the knee of a convex, decreasing score curve (Kneedle).

    Inertia curves fall steeply and then flatten; the knee is the point past
    which extra clusters buy little. Endpoints are never returned because the
    first and last candidate are not meaningful knees, and a curve too short
    or too flat for Kneedle falls back to its minimum.

    Args:
        scores: Score per candidate cluster count, in ascending count order.

    Returns:
        Index into ``scores`` of the selected candidate.
    """
    from kneed import KneeLocator

    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return 0
    if values.size < 3:
        # Too few points for a knee: the best achievable inertia wins.
        return int(np.argmin(values))
    if float(np.ptp(values)) == 0.0:
        # A flat curve has no knee. Short-circuit rather than let Kneedle
        # normalise by a zero range and warn about dividing by zero; the
        # midpoint is the same neutral choice made when no knee is found.
        return int(values.size // 2)

    locator = KneeLocator(
        np.arange(values.size, dtype=np.float64),
        values,
        curve="convex",
        direction="decreasing",
    )
    if locator.knee is None:
        # No knee detected: the midpoint is the stable neutral choice.
        return int(values.size // 2)

    index = int(locator.knee)
    # Clamp away from the endpoints, which are never meaningful knees.
    if index <= 0:
        index = 1
    if index >= values.size - 1:
        index = values.size - 2
    return index


def best_index(scores: Sequence[float], direction: str) -> int:
    """
    Return the index of the best score under the given selection rule.

    Args:
        scores: Score per candidate cluster count, in ascending count order.
        direction: :data:`MAXIMIZE`, :data:`MINIMIZE` or :data:`KNEE`.

    Returns:
        Index into ``scores`` of the selected candidate.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cluster selection requires at least one score.")
    if direction == KNEE:
        return knee_index(values)
    if direction == MINIMIZE:
        return int(np.argmin(values))
    return int(np.argmax(values))


def vote_best_index(
    scores_by_method: Mapping[str, Sequence[float]],
    methods: Sequence[str],
) -> int:
    """
    Combine several validation scores into one cluster-count choice.

    A single method decides on its own. Several methods each cast one vote
    for their own best candidate; the candidate with the most votes wins and
    ties are broken toward the smallest index, so a tie prefers the more
    parsimonious model.

    Args:
        scores_by_method: Validation score name -> score per candidate.
        methods: Names to consider, all of which must be present in
            ``scores_by_method``.

    Returns:
        Index of the selected candidate.

    Raises:
        ValueError: If ``methods`` is empty or names a score that was not
            computed.
    """
    names = [str(name) for name in methods]
    if not names:
        raise ValueError("At least one validation method is required.")
    missing = [name for name in names if name not in scores_by_method]
    if missing:
        raise ValueError(f"Unknown validation method(s): {', '.join(missing)}")

    if len(names) == 1:
        return best_index(scores_by_method[names[0]], score_direction(names[0]))

    votes: Dict[int, int] = {}
    for name in names:
        index = best_index(scores_by_method[name], score_direction(name))
        votes[index] = votes.get(index, 0) + 1
    most = max(votes.values())
    return min(index for index, count in votes.items() if count == most)


def gap_statistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_references: int = 10,
    random_state: int = 0,
) -> float:
    """
    Gap statistic of one clustering (Tibshirani, Walther & Hastie, 2001).

    Compares the achieved within-cluster dispersion against the dispersion
    expected from uniformly distributed noise spanning the same bounding box.
    A larger gap means the structure found is less likely to be an artefact
    of the data's extent, so the score is maximised.

    Args:
        features: Sample matrix of shape ``(n_samples, n_features)``.
        labels: Cluster label per sample, shape ``(n_samples,)``.
        n_references: Uniform reference datasets to average over.
        random_state: Seed for the reference datasets, making the score
            reproducible.

    Returns:
        ``log(E*[W_k]) - log(W_k)``; ``0.0`` when the clustering is
        degenerate (a single cluster or zero dispersion).

    Raises:
        ValueError: If ``features`` and ``labels`` disagree on sample count.
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"features must be 2-D; got shape {matrix.shape}.")
    label_array = np.asarray(labels)
    if label_array.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"features has {matrix.shape[0]} samples but labels has "
            f"{label_array.shape[0]}."
        )
    unique = np.unique(label_array)
    if unique.size < 2:
        return 0.0

    dispersion = _pooled_within_dispersion(matrix, label_array, unique)
    if dispersion <= 0.0:
        return 0.0

    rng = np.random.default_rng(random_state)
    low = matrix.min(axis=0)
    high = matrix.max(axis=0)
    reference_logs = np.empty(int(n_references), dtype=np.float64)
    for index in range(int(n_references)):
        sample = rng.uniform(low=low, high=high, size=matrix.shape)
        reference_logs[index] = np.log(
            max(_pooled_within_dispersion(sample, label_array, unique), 1e-300)
        )
    return float(reference_logs.mean() - np.log(dispersion))


def _pooled_within_dispersion(
    matrix: np.ndarray,
    labels: np.ndarray,
    unique_labels: np.ndarray,
) -> float:
    """
    Sum the within-cluster squared distances to each cluster mean.

    Args:
        matrix: Sample matrix of shape ``(n_samples, n_features)``.
        labels: Cluster label per sample.
        unique_labels: Distinct labels present, computed once by the caller.

    Returns:
        The pooled within-cluster dispersion ``W_k``.
    """
    total = 0.0
    for label in unique_labels:
        members = matrix[labels == label]
        if members.shape[0] == 0:
            continue
        centre = members.mean(axis=0)
        total += float(np.square(members - centre).sum())
    return total
