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
"""Regression-analysis figures.

Pure functions: arrays in, a matplotlib ``Figure`` out, no filesystem, all
text sanitised to ASCII. The set covers the standard regression diagnostics a
paper reports: predicted-vs-observed agreement, residual structure, residual
normality, and Bland-Altman limits of agreement.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from habit.exceptions import HABITAPIError
from habit.viz.labels import sanitize_label

__all__ = [
    "plot_predicted_vs_observed",
    "plot_residuals",
    "plot_residual_qq",
    "plot_bland_altman",
    "plot_coefficient_forest",
]


def _plt():
    """Return the pyplot module with the Agg canvas guaranteed headless."""
    import matplotlib

    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _check_pair(
    y_true: np.ndarray, y_pred: np.ndarray, owner: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the (y_true, y_pred) pair every regression figure shares."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise HABITAPIError(
            f"habit.viz.{owner}: y_true and y_pred must have the same shape; "
            f"got {y_true.shape} and {y_pred.shape}."
        )
    if y_true.size < 2:
        raise HABITAPIError(f"habit.viz.{owner}: need at least two samples.")
    return y_true, y_pred


def plot_predicted_vs_observed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    x_label: str = "Observed",
    y_label: str = "Predicted",
    identity: bool = True,
    fit_line: bool = True,
    annotate_r2: bool = True,
):
    """
    Predicted-against-observed scatter with the identity line.

    Points on the diagonal mean perfect agreement. Optionally overlays the
    least-squares fit and annotates R-squared, which together show both the
    bias (departure from the diagonal) and the spread.

    Args:
        y_true: Observed responses.
        y_pred: Predicted responses.
        x_label: X-axis label (sanitised).
        y_label: Y-axis label (sanitised).
        identity: Draw the y = x identity line.
        fit_line: Draw the least-squares regression of predicted on observed.
        annotate_r2: Annotate the coefficient of determination.

    Returns:
        The matplotlib ``Figure``.
    """
    plt = _plt()
    y_true, y_pred = _check_pair(y_true, y_pred, "plot_predicted_vs_observed")
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=14, alpha=0.7, color="#0072B2")

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    if identity:
        ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=0.8, label="identity")
    if fit_line:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        grid = np.linspace(lo, hi, 10)
        ax.plot(grid, slope * grid + intercept, color="#D55E00", lw=1.0, label="fit")
    if annotate_r2:
        from sklearn.metrics import r2_score

        r2 = r2_score(y_true, y_pred)
        ax.text(0.03, 0.97, f"R$^2$ = {r2:.3f}", transform=ax.transAxes, va="top")
    ax.set_xlabel(sanitize_label(x_label))
    ax.set_ylabel(sanitize_label(y_label))
    if identity or fit_line:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    x_label: str = "Fitted value",
    y_label: str = "Residual",
):
    """
    Residuals against fitted values, to expose heteroscedasticity and bias.

    A well-behaved model shows a mean-zero cloud with no funnel or trend;
    a curved or fan-shaped pattern signals a missing term or non-constant
    variance.

    Args:
        y_true: Observed responses.
        y_pred: Predicted responses.
        x_label: X-axis label (sanitised).
        y_label: Y-axis label (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    plt = _plt()
    y_true, y_pred = _check_pair(y_true, y_pred, "plot_residuals")
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, s=14, alpha=0.7, color="#009E73")
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    # A LOWESS-free smooth hint: binned means show systematic deviation.
    order = np.argsort(y_pred)
    bins = np.array_split(order, max(3, min(10, y_true.size // 3)))
    centres = [float(y_pred[b].mean()) for b in bins if b.size]
    means = [float(residuals[b].mean()) for b in bins if b.size]
    ax.plot(centres, means, "o-", color="#D55E00", lw=1.0, label="binned mean")
    ax.set_xlabel(sanitize_label(x_label))
    ax.set_ylabel(sanitize_label(y_label))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residual_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Normal Q-Q of residuals",
):
    """
    Quantile-quantile plot of the residuals against a normal reference.

    Residuals lying on the reference line are consistent with the normality
    that many regression inferential procedures assume; systematic S-curves
    or heavy tails flag a violation.

    Args:
        y_true: Observed responses.
        y_pred: Predicted responses.
        title: Figure title (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    plt = _plt()
    from scipy import stats

    y_true, y_pred = _check_pair(y_true, y_pred, "plot_residual_qq")
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, s=14, alpha=0.7, color="#0072B2")
    ax.plot(osm, slope * np.asarray(osm) + intercept, color="#D55E00", lw=1.0)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title(sanitize_label(title))
    fig.tight_layout()
    return fig


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    x_label: str = "Mean of observed and predicted",
    y_label: str = "Difference (observed - predicted)",
    sd_factor: float = 1.96,
):
    """
    Bland-Altman limits-of-agreement plot.

    The difference between two measurements is plotted against their mean,
    with the bias (mean difference) and the limits of agreement at
    ``sd_factor`` standard deviations. This is the standard way to show
    whether a predictor agrees with the reference across the whole range,
    rather than only on average.

    Args:
        y_true: Observed (reference) responses.
        y_pred: Predicted responses.
        x_label: X-axis label (sanitised).
        y_label: Y-axis label (sanitised).
        sd_factor: Multiplier for the limits of agreement (1.96 for ~95%).

    Returns:
        The matplotlib ``Figure``.
    """
    plt = _plt()
    y_true, y_pred = _check_pair(y_true, y_pred, "plot_bland_altman")
    mean = 0.5 * (y_true + y_pred)
    difference = y_true - y_pred
    bias = float(difference.mean())
    sd = float(difference.std(ddof=1))
    lower, upper = bias - sd_factor * sd, bias + sd_factor * sd

    fig, ax = plt.subplots()
    ax.scatter(mean, difference, s=14, alpha=0.7, color="#0072B2")
    ax.axhline(bias, color="#D55E00", lw=1.0, label=f"bias = {bias:.3g}")
    ax.axhline(upper, color="0.4", ls="--", lw=0.8, label=f"+{sd_factor:g} SD = {upper:.3g}")
    ax.axhline(lower, color="0.4", ls="--", lw=0.8, label=f"-{sd_factor:g} SD = {lower:.3g}")
    ax.set_xlabel(sanitize_label(x_label))
    ax.set_ylabel(sanitize_label(y_label))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_coefficient_forest(
    names: Sequence[str],
    coefficient: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    *,
    title: str = "Coefficients",
):
    """
    Forest-style plot of regression coefficients with optional CIs.

    Args:
        names: Covariate names (sanitised).
        coefficient: Point estimate per covariate.
        lower: Optional lower confidence bound per covariate.
        upper: Optional upper confidence bound per covariate.
        title: Figure title (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    plt = _plt()
    names = [sanitize_label(n) for n in names]
    coef = np.asarray(coefficient, dtype=np.float64)
    if coef.shape != (len(names),):
        raise HABITAPIError(
            "plot_coefficient_forest: coefficient must have one value per name."
        )
    has_ci = lower is not None and upper is not None
    if has_ci:
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        if lower.shape != coef.shape or upper.shape != coef.shape:
            raise HABITAPIError(
                "plot_coefficient_forest: lower/upper must align with coefficient."
            )

    fig_height = max(2.0, 0.35 * len(names) + 1.0)
    fig, ax = plt.subplots(figsize=(5.0, fig_height))
    y = np.arange(len(names))[::-1]
    if has_ci:
        ax.errorbar(
            coef,
            y,
            xerr=np.vstack([coef - lower, upper - coef]),
            fmt="o",
            color="#0072B2",
            ecolor="#0072B2",
            capsize=3,
        )
    else:
        ax.scatter(coef, y, color="#0072B2")
    ax.axvline(0.0, color="0.5", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Coefficient (95% CI)" if has_ci else "Coefficient")
    ax.set_title(sanitize_label(title))
    fig.tight_layout()
    return fig
