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
"""L3 statistics wrappers over the L0 evaluation kernels.

The kernels in :mod:`habit.kernels.statistics` and :mod:`habit.kernels.icc`
are deliberately minimal (arrays in, numbers out). The functions here are
the domain-layer surface the model-comparison and test-retest workflows
consume: they bundle the numbers a report actually needs into small frozen
result objects and operate on :class:`~habit.contracts.table.FeatureTable`
where the workflow is table-shaped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.api.exceptions import HABITAPIError
from habit.contracts.table import FeatureTable
from habit.kernels.icc import icc2_1, icc3_1
from habit.kernels.statistics import (
    delong_roc_ci,
    delong_roc_test,
    delong_roc_variance,
    hosmer_lemeshow_test,
    spiegelhalter_z_test,
)
from habit.utils.progress_utils import CustomTqdm

__all__ = [
    "DelongResult",
    "delong_test",
    "AucConfidenceInterval",
    "auc_confidence_interval",
    "CalibrationResult",
    "calibration_tests",
    "repeat_measurement_matrix",
    "icc_analysis",
]


# ---------------------------------------------------------------------------
# DeLong AUC comparison and confidence intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DelongResult:
    """Outcome of a paired DeLong AUC comparison."""

    #: AUC of the first model's scores.
    auc_a: float
    #: AUC of the second model's scores.
    auc_b: float
    #: Two-sided p-value of the hypothesis that the AUCs differ.
    p_value: float


def delong_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> DelongResult:
    """
    Compare two models' ROC AUCs on the same subjects (paired DeLong test).

    Args:
        y_true: Binary ground-truth labels (0/1), both classes present.
        scores_a: Probability-of-class-1 scores of the first model.
        scores_b: Probability-of-class-1 scores of the second model, aligned
            to ``scores_a``.

    Returns:
        The two AUCs and the two-sided p-value.
    """
    y_true = np.asarray(y_true)
    auc_a, _ = delong_roc_variance(y_true, np.asarray(scores_a, dtype=np.float64))
    auc_b, _ = delong_roc_variance(y_true, np.asarray(scores_b, dtype=np.float64))
    p_value = delong_roc_test(y_true, scores_a, scores_b)
    return DelongResult(auc_a=auc_a, auc_b=auc_b, p_value=p_value)


@dataclass(frozen=True)
class AucConfidenceInterval:
    """ROC AUC with its DeLong confidence interval."""

    #: Point estimate of the AUC.
    auc: float
    #: Lower bound of the interval (clipped at 0 by construction of the test).
    lower: float
    #: Upper bound of the interval (clipped at 1).
    upper: float
    #: Confidence level the interval was computed at, e.g. ``0.95``.
    alpha: float


def auc_confidence_interval(
    y_true: np.ndarray,
    scores: np.ndarray,
    alpha: float = 0.95,
) -> AucConfidenceInterval:
    """
    Compute the ROC AUC and its DeLong confidence interval.

    Args:
        y_true: Binary ground-truth labels (0/1), both classes present.
        scores: Probability-of-class-1 scores aligned to ``y_true``.
        alpha: Confidence level, e.g. ``0.95``.

    Returns:
        The AUC and its ``(lower, upper)`` confidence bounds.
    """
    auc, ci = delong_roc_ci(np.asarray(y_true), np.asarray(scores, dtype=np.float64), alpha=alpha)
    return AucConfidenceInterval(auc=auc, lower=float(ci[0]), upper=float(ci[1]), alpha=float(alpha))


# ---------------------------------------------------------------------------
# Calibration tests (Hosmer-Lemeshow + Spiegelhalter)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationResult:
    """The two calibration test outcomes reported together for a model."""

    #: Hosmer-Lemeshow chi-square statistic.
    hl_statistic: float
    #: Hosmer-Lemeshow p-value (high: calibration not rejected).
    hl_p_value: float
    #: Spiegelhalter Z statistic.
    spiegelhalter_z: float
    #: Spiegelhalter two-sided p-value (high: calibration not rejected).
    spiegelhalter_p_value: float


def calibration_tests(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_groups: int = 10,
) -> CalibrationResult:
    """
    Run both calibration tests HABIT reports for a binary model.

    Args:
        y_true: Binary outcomes (0/1).
        y_prob: Predicted event probabilities, aligned to ``y_true``.
        n_groups: Number of quantile-based risk groups for the
            Hosmer-Lemeshow test (classically 10).

    Returns:
        The Hosmer-Lemeshow and Spiegelhalter statistics with p-values.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    hl_statistic, hl_p = hosmer_lemeshow_test(y_true, y_prob, n_groups=n_groups)
    z, spiegelhalter_p = spiegelhalter_z_test(y_true, y_prob)
    return CalibrationResult(
        hl_statistic=hl_statistic,
        hl_p_value=hl_p,
        spiegelhalter_z=z,
        spiegelhalter_p_value=spiegelhalter_p,
    )


# ---------------------------------------------------------------------------
# ICC test-retest analysis over feature tables
# ---------------------------------------------------------------------------


def repeat_measurement_matrix(
    table: FeatureTable,
    repeat_tables: Sequence[FeatureTable],
    feature: str,
    *,
    owner: str,
) -> np.ndarray:
    """
    Build one feature's (subjects x sessions) repeat-measurement matrix.

    The primary ``table`` provides the first measurement session and the
    subject order; every table in ``repeat_tables`` contributes one further
    session, aligned to the primary rows by the identifier columns. Rows
    with a ``NaN`` in any session are dropped (pingouin's
    ``nan_policy="omit"``, which the v0.1 ICC analysis relied on).

    Args:
        table: Primary-measurement table carrying ``feature``.
        repeat_tables: One table per repeat session, each with identical
            ``id_columns`` and carrying ``feature``.
        feature: Feature column to extract.
        owner: Human-readable caller name for error messages.

    Returns:
        Array of shape ``(n_complete_subjects, 1 + len(repeat_tables))``.

    Raises:
        HABITAPIError: If a repeat table declares different identifier
            columns, lacks the feature, or contains unknown identifiers.
    """
    for repeat in repeat_tables:
        if tuple(repeat.id_columns) != tuple(table.id_columns):
            raise HABITAPIError(
                f"{owner} requires repeat tables with identical id_columns; "
                f"got {table.id_columns} and {repeat.id_columns}."
            )
        if feature not in repeat.feature_columns:
            raise HABITAPIError(
                f"{owner}: feature {feature!r} is missing from a "
                "repeat-measurement table."
            )
    columns = [table.frame[feature].to_numpy(dtype=np.float64)]
    primary_ids = table.frame[list(table.id_columns)]
    for repeat in repeat_tables:
        indexed = repeat.frame.set_index(list(repeat.id_columns))
        try:
            aligned = indexed.loc[
                pd.MultiIndex.from_frame(primary_ids)
                if len(table.id_columns) > 1
                else primary_ids[table.id_columns[0]]
            ]
        except KeyError as exc:
            raise HABITAPIError(
                f"{owner} requires repeat tables aligned to the primary "
                f"table by identifier columns; unknown id {exc}."
            ) from exc
        columns.append(aligned[feature].to_numpy(dtype=np.float64))
    matrix = np.column_stack(columns)
    # Omit subjects with a NaN in any session (pingouin nan_policy="omit").
    return matrix[~np.isnan(matrix).any(axis=1)]


def icc_analysis(
    table: FeatureTable,
    repeat_tables: Sequence[FeatureTable],
    icc_types: Tuple[str, ...] = ("icc2", "icc3"),
    min_subjects: int = 2,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute per-feature test-retest ICCs across measurement sessions.

    This is the v1 successor of the v0.1
    ``habit.core.machine_learning.feature_selectors.icc`` analysis: where
    that one merged CSVs from disk, this one consumes aligned feature
    tables directly and computes each ICC with the L0 kernels. Both the
    ICC(2,1) (two-way random, absolute agreement) and ICC(3,1) (two-way
    mixed, consistency) variants are reported by default, matching the v0.1
    reporting defaults.

    Args:
        table: Primary-measurement table; its feature columns and row order
            define the analysis.
        repeat_tables: One table per repeat measurement session, aligned to
            ``table`` by the identifier columns.
        icc_types: Which ICC variants to compute (``"icc2"`` and/or
            ``"icc3"``); one result column per variant.
        min_subjects: Minimum number of complete (NaN-free) subjects for a
            feature's ICC to be computed; fewer yields ``NaN``.
        verbose: Show a progress bar over the features.

    Returns:
        Frame with one row per feature and the columns ``feature`` plus one
        column per requested ICC variant (``NaN`` where undefined).

    Raises:
        HABITAPIError: If ``repeat_tables`` is empty or an unknown ICC
            variant is requested.
    """
    if not repeat_tables:
        raise HABITAPIError(
            "icc_analysis requires repeat_tables: one feature table per "
            "repeat measurement session, aligned by identifier columns."
        )
    kernels = {"icc2": icc2_1, "icc3": icc3_1}
    unknown = [name for name in icc_types if name not in kernels]
    if unknown:
        raise HABITAPIError(
            f"icc_analysis: unknown icc_types {unknown}; "
            f"supported: {sorted(kernels)}."
        )
    progress = CustomTqdm(
        total=len(table.feature_columns),
        desc="ICC analysis",
        disable=not verbose,
    )
    rows = []
    try:
        for feature in table.feature_columns:
            matrix = repeat_measurement_matrix(
                table, repeat_tables, feature, owner="icc_analysis"
            )
            row = {"feature": feature}
            for name in icc_types:
                if matrix.shape[0] < min_subjects:
                    row[name] = float("nan")
                else:
                    row[name] = float(kernels[name](matrix))
            rows.append(row)
            progress.update(1)
    finally:
        progress.close()
    return pd.DataFrame(rows, columns=["feature", *icc_types])
