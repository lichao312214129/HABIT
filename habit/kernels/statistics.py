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
# ---------------------------------------------------------------------------
# Third-party attribution
#
# The DeLong AUC comparison routines below (compute_midrank, fast_delong and
# helpers) are adapted from the VMAF project:
#
#     https://github.com/Netflix/vmaf
#     Copyright (c) 2020 Netflix, Inc.
#     Licensed under BSD-2-Clause-Patent
#
# The upstream notice is retained here and in the NOTICE file at the project
# root, as required by that license.
# ---------------------------------------------------------------------------
"""L0 pure-math kernels for model-evaluation statistics.

The DeLong, Hosmer-Lemeshow and Spiegelhalter tests are the statistical
backbone of HABIT's model comparison and calibration reporting. Keeping them
here as pure functions (arrays in, numbers out; no IO, no state, no logging)
makes the exact formulas independently reviewable and reusable outside the
domain layer. The implementations are numerically equivalent to the
established v0.1 routines in ``habit.core.machine_learning.statistics``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats

__all__ = [
    "compute_midrank",
    "fast_delong",
    "delong_roc_variance",
    "delong_roc_test",
    "delong_roc_ci",
    "hosmer_lemeshow_test",
    "spiegelhalter_z_test",
]


# ---------------------------------------------------------------------------
# DeLong's method for AUC variance and AUC comparison
# ---------------------------------------------------------------------------


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """
    Compute midranks (average ranks for tied values).

    Args:
        x: One-dimensional array of scores.

    Returns:
        Array of midranks, 1-based to match the AUC formula in the paper.
    """
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks_sorted = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks_sorted[i:j] = 0.5 * (i + j - 1)
        i = j
    midranks = np.empty(n, dtype=np.float64)
    # +1 converts to the 1-based ranks the AUC formula in the paper assumes.
    midranks[order] = midranks_sorted + 1
    return midranks


def fast_delong(
    predictions_sorted_transposed: np.ndarray,
    label_1_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute AUCs and the DeLong covariance for one or more classifiers.

    The fast version of DeLong's method for the covariance of unadjusted AUC
    (Sun & Xu, IEEE Signal Processing Letters 21(11), 2014).

    Args:
        predictions_sorted_transposed: Scores of shape
            ``(n_classifiers, n_examples)``, sorted so that the examples with
            label 1 come first.
        label_1_count: Number of positive examples (``m`` in the paper).

    Returns:
        Tuple ``(aucs, delong_covariance)`` with ``aucs`` of shape
        ``(n_classifiers,)`` and the covariance matrix of shape
        ``(n_classifiers, n_classifiers)`` (scalar when ``k == 1``).
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def _sorted_by_label(ground_truth: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return the label-descending order and the positive-example count."""
    if not np.array_equal(np.unique(ground_truth), [0, 1]):
        raise ValueError(
            "DeLong statistics require binary ground truth with both classes "
            "present (values {0, 1})."
        )
    order = (-ground_truth).argsort()
    return order, int(ground_truth.sum())


def delong_roc_variance(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute the ROC AUC and its DeLong variance for one score vector.

    Args:
        ground_truth: Binary labels (0/1), both classes present.
        predictions: Probability-of-class-1 scores aligned to the labels.

    Returns:
        Tuple ``(auc, delong_variance)``.
    """
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions, dtype=np.float64)
    order, label_1_count = _sorted_by_label(ground_truth)
    predictions_sorted = predictions[np.newaxis, order]
    aucs, delong_cov = fast_delong(predictions_sorted, label_1_count)
    return float(aucs[0]), float(delong_cov)


def delong_roc_test(
    ground_truth: np.ndarray,
    predictions_one: np.ndarray,
    predictions_two: np.ndarray,
) -> float:
    """
    Compute the p-value for the hypothesis that two ROC AUCs differ.

    Args:
        ground_truth: Binary labels (0/1), both classes present.
        predictions_one: Probability-of-class-1 scores of the first model.
        predictions_two: Probability-of-class-1 scores of the second model.

    Returns:
        Two-sided p-value of the paired DeLong test.
    """
    ground_truth = np.asarray(ground_truth)
    order, label_1_count = _sorted_by_label(ground_truth)
    stacked = np.vstack(
        (
            np.asarray(predictions_one, dtype=np.float64),
            np.asarray(predictions_two, dtype=np.float64),
        )
    )[:, order]
    aucs, delong_cov = fast_delong(stacked, label_1_count)
    contrast = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / (
        np.sqrt(np.dot(np.dot(contrast, delong_cov), contrast.T)) + 1e-8
    )
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return float(p_value[0])


def delong_roc_ci(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    alpha: float = 0.95,
) -> Tuple[float, np.ndarray]:
    """
    Compute the ROC AUC and its DeLong confidence interval.

    Args:
        ground_truth: Binary labels (0/1), both classes present.
        predictions: Probability-of-class-1 scores aligned to the labels.
        alpha: Confidence level, e.g. ``0.95``.

    Returns:
        Tuple ``(auc, ci)`` where ``ci`` is the ``(lower, upper)`` bound,
        clipped at 1.0.
    """
    auc, auc_cov = delong_roc_variance(ground_truth, predictions)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1
    return float(auc), ci


# ---------------------------------------------------------------------------
# Hosmer-Lemeshow goodness-of-fit test
# ---------------------------------------------------------------------------


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_groups: int = 10,
) -> Tuple[float, float]:
    """
    Perform the Hosmer-Lemeshow calibration test for binary outcomes.

    Subjects are grouped into ``n_groups`` quantile-based risk groups
    (right-closed intervals, lowest edge included, mirroring ``pd.qcut``),
    then the chi-square statistic comparing observed and expected event
    counts is evaluated against ``n_groups - 2`` degrees of freedom.

    Args:
        y_true: Binary outcomes (0/1).
        y_prob: Predicted event probabilities, aligned to ``y_true``.
        n_groups: Number of risk groups (classically 10, the decile test).

    Returns:
        Tuple ``(statistic, p_value)``.

    Raises:
        ValueError: If inputs are misaligned, non-binary, probabilities fall
            outside [0, 1], or the quantile edges are not unique (too many
            tied probabilities to form ``n_groups`` groups).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"y_true and y_prob must have the same shape; got "
            f"{y_true.shape} and {y_prob.shape}."
        )
    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("y_true must contain only 0 and 1.")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_prob must lie in [0, 1].")
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2; got {n_groups}.")

    edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_groups + 1))
    if np.unique(edges).size != edges.size:
        raise ValueError(
            "Cannot form the requested risk groups: quantile edges are not "
            "unique (too many tied predicted probabilities)."
        )
    # Right-closed bins (edge_i, edge_{i+1}], lowest edge included, matching
    # the pd.qcut grouping the v0.1 implementation relied on.
    bin_index = np.digitize(y_prob, edges[1:-1], right=True)

    statistic = 0.0
    for group in range(n_groups):
        in_group = bin_index == group
        observed_pos = float(y_true[in_group].sum())
        observed_neg = float(in_group.sum() - observed_pos)
        expected_pos = float(y_prob[in_group].sum())
        expected_neg = float(in_group.sum() - expected_pos)
        if expected_pos > 0:
            statistic += (observed_pos - expected_pos) ** 2 / expected_pos
        if expected_neg > 0:
            statistic += (observed_neg - expected_neg) ** 2 / expected_neg

    p_value = float(1 - stats.chi2.cdf(statistic, n_groups - 2))
    return float(statistic), p_value


# ---------------------------------------------------------------------------
# Spiegelhalter Z-test
# ---------------------------------------------------------------------------


def spiegelhalter_z_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float]:
    """
    Perform Spiegelhalter's Z-test of calibration for binary outcomes.

    The statistic compares observed minus expected event counts against the
    variance implied by the predicted probabilities:
    ``z = sum(y - p) / sqrt(sum(p * (1 - p)))``.

    Args:
        y_true: Binary outcomes (0/1).
        y_prob: Predicted event probabilities, aligned to ``y_true``.

    Returns:
        Tuple ``(z_statistic, p_value)`` (two-sided).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"y_true and y_prob must have the same shape; got "
            f"{y_true.shape} and {y_prob.shape}."
        )
    observed_minus_expected = y_true - y_prob
    variance = y_prob * (1.0 - y_prob)
    z = float(np.sum(observed_minus_expected) / np.sqrt(np.sum(variance)))
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    return z, p_value
