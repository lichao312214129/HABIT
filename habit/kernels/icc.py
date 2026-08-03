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
"""L0 pure-math kernels for intraclass correlation coefficients (ICC).

The ICC quantifies how stable a measurement is across repeated observations
(test-retest scans, raters, sites). HABIT uses it to filter unstable imaging
features before modelling. The formulas are the classical Shrout & Fleiss /
McGraw & Wong two-way ANOVA definitions, numerically equivalent to
``pingouin.intraclass_corr`` for the single-measure ICC2/ICC3 rows, kept here
as pure functions so the exact definition is independently reviewable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["two_way_mean_squares", "icc3_1", "icc2_1"]


def two_way_mean_squares(data: np.ndarray) -> Tuple[float, float, float, int, int]:
    """
    Compute the row/column/error mean squares of a two-way layout.

    Args:
        data: Matrix of shape ``(n_targets, k_raters)`` with one observation
            per cell; NaNs are not allowed.

    Returns:
        Tuple ``(ms_rows, ms_columns, ms_error, n, k)``.

    Raises:
        ValueError: If the matrix is not 2-D, has fewer than two rows or
            columns, or contains NaN.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"ICC input must be a 2-D matrix; got ndim={data.ndim}.")
    n, k = data.shape
    if n < 2 or k < 2:
        raise ValueError(
            f"ICC requires at least 2 targets and 2 raters; got shape {data.shape}."
        )
    if np.isnan(data).any():
        raise ValueError("ICC input must not contain NaN values.")

    grand_mean = float(data.mean())
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ms_rows = k * float(np.sum((row_means - grand_mean) ** 2)) / (n - 1)
    ms_columns = n * float(np.sum((col_means - grand_mean) ** 2)) / (k - 1)
    residuals = data - row_means[:, None] - col_means[None, :] + grand_mean
    ms_error = float(np.sum(residuals**2)) / ((n - 1) * (k - 1))
    return ms_rows, ms_columns, ms_error, n, k


def icc3_1(data: np.ndarray) -> float:
    """
    Compute ICC(3,1): two-way mixed model, consistency, single measurement.

    This is the ICC variant HABIT uses by default for test-retest feature
    stability (the v0.1 ``icc3`` configuration default, pingouin's ``ICC3``
    row; McGraw & Wong notation ``ICC(C,1)``):
    ``(MS_R - MS_E) / (MS_R + (k - 1) * MS_E)``.

    Args:
        data: Matrix of shape ``(n_targets, k_raters)``.

    Returns:
        The ICC(3,1) value (can be negative when between-target variance is
        below the error variance).
    """
    ms_rows, _, ms_error, _, k = two_way_mean_squares(data)
    denominator = ms_rows + (k - 1) * ms_error
    if denominator == 0.0:
        # Degenerate (constant) measurement: no variance to attribute.
        return 0.0
    return float((ms_rows - ms_error) / denominator)


def icc2_1(data: np.ndarray) -> float:
    """
    Compute ICC(2,1): two-way random model, absolute agreement, single
    measurement (pingouin's ``ICC2`` row; McGraw & Wong ``ICC(A,1)``):
    ``(MS_R - MS_E) / (MS_R + (k - 1) * MS_E + k * (MS_C - MS_E) / n)``.

    Args:
        data: Matrix of shape ``(n_targets, k_raters)``.

    Returns:
        The ICC(2,1) value.
    """
    ms_rows, ms_columns, ms_error, n, k = two_way_mean_squares(data)
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    if denominator == 0.0:
        return 0.0
    return float((ms_rows - ms_error) / denominator)
