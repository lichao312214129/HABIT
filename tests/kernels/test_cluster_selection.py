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
"""Regression tests for the shared cluster-count selection kernel.

The reference curves below are the ones that exposed the v0.1 divergence
between the ``elbow`` and ``kneedle`` rules. Pinning their selected index here
is what stops the two code paths (v0.1 clustering, v1.0 domain fitters) from
drifting apart again.
"""

from __future__ import annotations

import numpy as np
import pytest

from habit.kernels.cluster_selection import (
    KNEE,
    MAXIMIZE,
    MINIMIZE,
    best_index,
    gap_statistic,
    knee_index,
    score_direction,
    vote_best_index,
)

#: name -> (curve, index the shared rule selects). Curves are inertia-like:
#: positive, decreasing, flattening. The indices are the observed output of
#: the v0.1 Kneedle rule, now the single rule, captured so any future change
#: to the selection code surfaces here instead of in a habitat count.
_REFERENCE_CURVES = {
    "exponential_decay": ([100.0, 50.0, 25.0, 12.5, 6.25, 3.125], 2),
    "sharp_knee_at_third": ([10.0, 3.0, 2.2, 2.0, 1.9], 1),
    "gentle_slope": ([10.0, 9.0, 8.0, 7.0, 6.0, 5.0], 3),
    "late_knee": ([100.0, 90.0, 80.0, 70.0, 20.0, 10.0], 1),
    "two_stage_drop": ([50.0, 20.0, 18.0, 17.0, 16.5, 16.2], 1),
    "near_linear": ([8.0, 6.5, 5.0, 3.5, 2.0], 2),
}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_REFERENCE_CURVES))
def test_knee_index_is_pinned_for_reference_curves(name: str) -> None:
    """Knee detection on the reference curves must not drift."""
    curve, expected = _REFERENCE_CURVES[name]
    assert knee_index(curve) == expected


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_REFERENCE_CURVES))
def test_elbow_and_kneedle_are_the_same_rule(name: str) -> None:
    """v1.0 breaking change: ``elbow`` resolves to Kneedle, like ``kneedle``."""
    curve, _ = _REFERENCE_CURVES[name]
    assert score_direction("elbow") == score_direction("kneedle") == KNEE
    assert best_index(curve, score_direction("elbow")) == best_index(
        curve, score_direction("kneedle")
    )
    assert best_index(curve, score_direction("elbow")) == knee_index(curve)


@pytest.mark.unit
def test_score_direction_covers_every_shipped_criterion() -> None:
    """Every criterion a config may name resolves to an explicit rule."""
    assert score_direction("silhouette") == MAXIMIZE
    assert score_direction("calinski_harabasz") == MAXIMIZE
    assert score_direction("gap") == MAXIMIZE
    assert score_direction("davies_bouldin") == MINIMIZE
    assert score_direction("aic") == MINIMIZE
    assert score_direction("bic") == MINIMIZE
    assert score_direction("inertia") == KNEE
    # Unknown names fall back to maximise, matching the v0.1 default.
    assert score_direction("no_such_score") == MAXIMIZE


@pytest.mark.unit
def test_knee_index_handles_degenerate_curves() -> None:
    """Short and flat curves resolve without warnings or endpoint picks."""
    assert knee_index([]) == 0
    # Two points cannot form a knee: the best inertia wins.
    assert knee_index([9.0, 4.0]) == 1
    # A flat curve has no knee; the midpoint is the neutral choice.
    assert knee_index([5.0, 5.0, 5.0]) == 1


@pytest.mark.unit
def test_knee_index_never_returns_an_endpoint_for_long_curves() -> None:
    """Endpoints are not meaningful knees and are clamped away."""
    for curve, _ in _REFERENCE_CURVES.values():
        index = knee_index(curve)
        assert 0 < index < len(curve) - 1


@pytest.mark.unit
def test_best_index_applies_the_requested_rule() -> None:
    """Maximise, minimise and knee each pick their own candidate."""
    scores = [0.1, 0.9, 0.4]
    assert best_index(scores, MAXIMIZE) == 1
    assert best_index(scores, MINIMIZE) == 0
    with pytest.raises(ValueError):
        best_index([], MAXIMIZE)


@pytest.mark.unit
def test_vote_best_index_uses_majority_then_prefers_fewer_clusters() -> None:
    """Votes are counted per criterion; ties prefer the smaller index."""
    scores = {
        "silhouette": [0.1, 0.9, 0.4],  # maximise -> index 1
        "calinski_harabasz": [1.0, 5.0, 2.0],  # maximise -> index 1
        "davies_bouldin": [0.2, 0.9, 0.4],  # minimise -> index 0
    }
    assert vote_best_index(scores, ["silhouette"]) == 1
    assert vote_best_index(scores, ["davies_bouldin"]) == 0
    assert (
        vote_best_index(scores, ["silhouette", "calinski_harabasz", "davies_bouldin"])
        == 1
    )
    # One vote each: the tie resolves to the more parsimonious candidate.
    assert vote_best_index(scores, ["silhouette", "davies_bouldin"]) == 0


@pytest.mark.unit
def test_vote_best_index_rejects_unusable_requests() -> None:
    """An empty or unknown criterion list is an error, never a silent default."""
    with pytest.raises(ValueError):
        vote_best_index({"silhouette": [0.1, 0.2]}, [])
    with pytest.raises(ValueError):
        vote_best_index({"silhouette": [0.1, 0.2]}, ["gap"])


@pytest.mark.unit
def test_gap_statistic_is_deterministic_and_rewards_real_structure() -> None:
    """The gap is reproducible and larger for clustered than shuffled labels."""
    rng = np.random.default_rng(0)
    clustered = np.vstack(
        [
            rng.normal(loc=-4.0, scale=0.3, size=(30, 2)),
            rng.normal(loc=4.0, scale=0.3, size=(30, 2)),
        ]
    )
    labels = np.repeat([0, 1], 30)
    gap = gap_statistic(clustered, labels)
    assert gap == pytest.approx(gap_statistic(clustered, labels))

    shuffled = rng.permutation(labels)
    assert gap > gap_statistic(clustered, shuffled)


@pytest.mark.unit
def test_gap_statistic_guards_degenerate_input() -> None:
    """A single cluster has no gap; shape mismatches are rejected."""
    matrix = np.zeros((6, 2), dtype=np.float64)
    assert gap_statistic(matrix, np.zeros(6, dtype=int)) == 0.0
    # Two clusters but zero dispersion (all points identical).
    assert gap_statistic(matrix, np.repeat([0, 1], 3)) == 0.0
    with pytest.raises(ValueError):
        gap_statistic(matrix, np.zeros(5, dtype=int))
    with pytest.raises(ValueError):
        gap_statistic(np.zeros(6, dtype=np.float64), np.zeros(6, dtype=int))
