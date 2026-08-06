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
"""L0 kernels: voxel-level ICC point estimates with confidence limits.

Implements the two ICC flavours Prior et al. used to define "precise"
voxel-wise radiomics features (Radiol Artif Intell 2024;6(2):e230118,
Appendix E6, erratum-corrected):

* ``icc3a_1`` -- ICC(3A,1), two-way mixed, absolute agreement: repeatability
  across replications of the SAME acquisition/processing condition.
* ``icc3c_1`` -- ICC(3C,1), two-way mixed, consistency: reproducibility
  across CHANGING conditions (e.g. kernel radii or bin widths).

Both share the two-way mean-square decomposition of
:func:`habit.kernels.icc.two_way_mean_squares`. The erratum divides the
column mean square by ``n`` before it enters the absolute-agreement
correction term, i.e. the variance of the column means is used instead of
the ANOVA mean square (which grows with ``n``); with voxel tables
(``n`` in the thousands) the term vanishes either way, but the erratum form
stays well behaved for small ``n``. Per the erratum, the consistency
variant keeps the absolute-agreement confidence-interval formula.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats

from habit.kernels.icc import two_way_mean_squares

__all__ = ["ICCEstimate", "icc3a_1", "icc3c_1"]


class ICCEstimate(NamedTuple):
    """Point estimate and two-sided confidence limits of one ICC."""

    value: float
    lcl: float
    ucl: float


def _confidence_limits(
    ms_rows: float,
    ms_error: float,
    n: int,
    k: int,
    alpha: float,
) -> tuple:
    """
    F-distribution confidence limits of an ICC (Shrout-Fleiss convention).

    Args:
        ms_rows: Between-targets (voxel) mean square.
        ms_error: Residual mean square; must be positive.
        n: Number of targets (rows).
        k: Number of replications (columns).
        alpha: Two-sided significance level.

    Returns:
        ``(lcl, ucl)`` clipped to ``[0, 1]``.
    """
    f_ratio = ms_rows / ms_error
    f_crit = stats.f.ppf(1.0 - alpha / 2.0, n - 1, (n - 1) * (k - 1))
    f_lower = f_ratio / f_crit
    f_upper = f_ratio * f_crit
    lcl = (f_lower - 1.0) / (f_lower + (k - 1.0))
    ucl = (f_upper - 1.0) / (f_upper + (k - 1.0))
    return float(np.clip(lcl, 0.0, 1.0)), float(np.clip(ucl, 0.0, 1.0))


def _icc3(data: np.ndarray, alpha: float, consistency: bool) -> ICCEstimate:
    """
    Shared engine for ``icc3a_1`` and ``icc3c_1``.

    Args:
        data: ``(n, k)`` matrix, one row per target, one column per
            replication; NaN-free (enforced by ``two_way_mean_squares``).
        alpha: Two-sided significance level for the confidence limits.
        consistency: ``True`` drops the column-bias correction term
            (ICC(3C,1)); ``False`` keeps it (ICC(3A,1)).

    Returns:
        The estimate with its ``(1 - alpha)`` confidence limits, truncated
        to ``[0, 1]`` per the paper's convention.
    """
    ms_rows, ms_columns, ms_error, n, k = two_way_mean_squares(data)
    if ms_error == 0.0:
        # Identical columns: perfect agreement when targets do vary;
        # undefined when the whole matrix is constant -- a constant feature
        # carries no information and must not pass a precision screen, so
        # it is reported as 0 rather than 1.
        perfect = 1.0 if ms_rows > 0.0 else 0.0
        return ICCEstimate(perfect, perfect, perfect)
    denominator = ms_rows + (k - 1.0) * ms_error
    if not consistency:
        # Erratum: MSC enters divided by n, i.e. as the variance of the
        # column means, not as the ANOVA mean square.
        denominator += (k / n) * (ms_columns / n - ms_error)
    if denominator <= 0.0:
        # Pathological layout (e.g. zero between-target variance with a
        # negative bias correction): no reliability to speak of.
        return ICCEstimate(0.0, 0.0, 0.0)
    value = (ms_rows - ms_error) / denominator
    lcl, ucl = _confidence_limits(ms_rows, ms_error, n, k, alpha)
    # Paper convention: negative ICCs are truncated at 0. The value cannot
    # exceed 1 algebraically; the clip only absorbs float dust.
    return ICCEstimate(float(np.clip(value, 0.0, 1.0)), lcl, ucl)


def icc3a_1(data: np.ndarray, alpha: float = 0.05) -> ICCEstimate:
    """
    ICC(3A,1): two-way mixed effects, absolute agreement, single measurement.

    Args:
        data: ``(n, k)`` matrix, one row per voxel, one column per
            replication of the same condition. NaN handling is the caller's
            responsibility (pairwise-complete rows recommended).
        alpha: Two-sided significance level for the confidence limits.

    Returns:
        The estimate with its ``(1 - alpha)`` confidence limits, truncated
        to ``[0, 1]`` per the paper's convention.

    Raises:
        ValueError: If ``data`` is not a 2-D, NaN-free matrix with at least
            2 rows and 2 columns.
    """
    return _icc3(data, alpha, consistency=False)


def icc3c_1(data: np.ndarray, alpha: float = 0.05) -> ICCEstimate:
    """
    ICC(3C,1): two-way mixed effects, consistency, single measurement.

    Args:
        data: ``(n, k)`` matrix, one row per voxel, one column per condition
            (e.g. per kernel radius or per bin width). NaN handling is the
            caller's responsibility (pairwise-complete rows recommended).
        alpha: Two-sided significance level for the confidence limits.

    Returns:
        The estimate with its ``(1 - alpha)`` confidence limits, truncated
        to ``[0, 1]`` per the paper's convention.

    Raises:
        ValueError: If ``data`` is not a 2-D, NaN-free matrix with at least
            2 rows and 2 columns.
    """
    return _icc3(data, alpha, consistency=True)
