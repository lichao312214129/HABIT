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
"""Survival-analysis figures.

Each function is pure: contract objects or arrays in, a matplotlib ``Figure``
out, no filesystem. All text is sanitised to ASCII via
:func:`~habit.viz.labels.sanitize_label`. The KM estimator and the log-rank
test use lifelines (an optional ``analysis`` dependency), imported lazily so
the module loads without it; the time-dependent AUC and Brier curves use
scikit-survival on the arrays the caller supplies.

The figures cover the standard survival-paper set: Kaplan-Meier with a
numbers-at-risk table, the risk-score scatter + survival-function panel,
time-dependent AUC, calibration at a fixed horizon, and the Cox coefficient
forest plot.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.viz.labels import sanitize_label

__all__ = [
    "plot_kaplan_meier",
    "plot_risk_triptych",
    "plot_time_dependent_auc",
    "plot_survival_calibration",
    "plot_brier_curve",
    "plot_cox_forest",
]


def _plt():
    """Return the pyplot module with the Agg canvas guaranteed headless."""
    import matplotlib

    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _lifelines(owner: str):
    """Import lifelines or raise with the install hint attached."""
    try:
        import lifelines  # type: ignore
    except ImportError as exc:
        raise HABITAPIError(
            f"habit.viz.{owner} needs lifelines; install the 'analysis' "
            "extra (pip install HABIT[analysis])."
        ) from exc
    return lifelines


def _sksurv(owner: str):
    """Import scikit-survival's metrics or raise with the install hint."""
    try:
        from sksurv import metrics as _m  # type: ignore
    except ImportError as exc:
        raise HABITAPIError(
            f"habit.viz.{owner} needs scikit-survival; install the 'analysis' "
            "extra (pip install HABIT[analysis])."
        ) from exc
    return _m


def _check_survival_inputs(
    time: np.ndarray, event: np.ndarray, owner: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and normalise the (time, event) pair every figure shares."""
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.bool_)
    if time.shape != event.shape:
        raise HABITAPIError(
            f"habit.viz.{owner}: time and event must have the same shape; "
            f"got {time.shape} and {event.shape}."
        )
    if time.size == 0:
        raise HABITAPIError(f"habit.viz.{owner}: empty time/event arrays.")
    return time, event


# ---------------------------------------------------------------------------
# Kaplan-Meier with numbers-at-risk
# ---------------------------------------------------------------------------


def plot_kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
    group: Optional[np.ndarray] = None,
    *,
    group_names: Optional[Sequence[str]] = None,
    ci: bool = True,
    show_censoring: bool = True,
    risk_table: bool = True,
    log_rank: bool = True,
    title: str = "Kaplan-Meier",
    time_label: str = "Time",
    probability_label: str = "Survival probability",
):
    """
    Kaplan-Meier curves, optionally stratified, with a numbers-at-risk table.

    Args:
        time: Observed follow-up durations.
        event: Event indicators, True for observed events.
        group: Optional per-row stratum labels for 2+ KM curves; ``None``
            draws a single cohort curve.
        group_names: Display names for the strata, in sorted-label order;
            defaults to the labels themselves. Sanitised to ASCII.
        ci: Draw the pointwise confidence band.
        show_censoring: Mark censored observations on the curve.
        risk_table: Append a numbers-at-risk table beneath the axis.
        log_rank: Annotate the log-rank p-value (only with 2 strata).
        title: Figure title (sanitised).
        time_label: X-axis label (sanitised).
        probability_label: Y-axis label (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    _lifelines("plot_kaplan_meier")
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import logrank_test

    plt = _plt()
    time, event = _check_survival_inputs(time, event, "plot_kaplan_meier")
    fig, ax = plt.subplots()

    fitters = []
    if group is None:
        kmf = KaplanMeierFitter()
        kmf.fit(time, event, label="cohort")
        kmf.plot_survival_function(ax=ax, ci_show=ci, show_censors=show_censoring)
        fitters.append(kmf)
    else:
        group = np.asarray(group)
        strata = sorted(np.unique(group), key=lambda value: str(value))
        names = (
            [sanitize_label(value) for value in strata]
            if group_names is None
            else [sanitize_label(n) for n in group_names]
        )
        if len(names) != len(strata):
            raise HABITAPIError(
                f"plot_kaplan_meier: {len(strata)} strata but "
                f"{len(names)} group_names."
            )
        for stratum, name in zip(strata, names):
            mask = group == stratum
            kmf = KaplanMeierFitter()
            kmf.fit(time[mask], event[mask], label=name)
            kmf.plot_survival_function(
                ax=ax, ci_show=ci, show_censors=show_censoring
            )
            fitters.append(kmf)
        if log_rank and len(strata) == 2:
            left = group == strata[0]
            right = group == strata[1]
            result = logrank_test(
                time[left], time[right], event[left], event[right]
            )
            ax.text(
                0.03,
                0.05,
                f"log-rank p = {result.p_value:.3g}",
                transform=ax.transAxes,
            )

    ax.set_xlabel(sanitize_label(time_label))
    ax.set_ylabel(sanitize_label(probability_label))
    ax.set_title(sanitize_label(title))
    ax.set_ylim(0.0, 1.02)

    if risk_table and fitters:
        # lifelines lays the at-risk counts in a dedicated band under the axis.
        add_at_risk_counts(*fitters, ax=ax)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Risk-score triptych: scatter + survival function + event ribbon
# ---------------------------------------------------------------------------


def plot_risk_triptych(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    survival_probability: Optional[np.ndarray] = None,
    survival_times: Optional[np.ndarray] = None,
    time_label: str = "Time",
    risk_label: str = "Risk score",
):
    """
    Three-panel risk-stratification figure.

    Panel 1 ranks subjects by predicted risk and colours observed events;
    panel 2 overlays each subject's predicted survival curve (when supplied)
    against the cohort KM curve; panel 3 shows each subject's follow-up time
    and event status along the risk ranking. Together they show, at a glance,
    whether the score separates early events from long survivors.

    Args:
        time: Observed follow-up durations.
        event: Event indicators, True for observed events.
        risk: Per-subject risk scores (higher means shorter survival).
        survival_probability: Optional ``(n_subjects, n_times)`` predicted
            S(t|x) matrix for panel 2.
        survival_times: The times ``survival_probability`` columns align to.
        time_label: X-axis label (sanitised).
        risk_label: Risk-axis label (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    _lifelines("plot_risk_triptych")
    from lifelines import KaplanMeierFitter

    plt = _plt()
    time, event = _check_survival_inputs(time, event, "plot_risk_triptych")
    risk = np.asarray(risk, dtype=np.float64)
    if risk.shape != time.shape:
        raise HABITAPIError(
            "plot_risk_triptych: risk must align with time/event; got "
            f"{risk.shape} vs {time.shape}."
        )
    order = np.argsort(risk)
    rank = np.arange(time.size)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    # Panel 1: ranked risk, events marked.
    axes[0].scatter(
        rank[~event[order]],
        risk[order][~event[order]],
        s=10,
        label="censored",
    )
    axes[0].scatter(
        rank[event[order]],
        risk[order][event[order]],
        s=10,
        label="event",
    )
    axes[0].set_xlabel("Subjects by increasing risk")
    axes[0].set_ylabel(sanitize_label(risk_label))
    axes[0].legend()

    # Panel 2: predicted survival curves + cohort KM.
    if survival_probability is not None:
        probability = np.asarray(survival_probability, dtype=np.float64)
        if survival_times is None:
            raise HABITAPIError(
                "plot_risk_triptych: survival_times is required with "
                "survival_probability."
            )
        grid = np.asarray(survival_times, dtype=np.float64)
        if probability.shape != (time.size, grid.size):
            raise HABITAPIError(
                "plot_risk_triptych: survival_probability must have shape "
                f"(n_subjects, n_times) = {(time.size, grid.size)}; got "
                f"{probability.shape}."
            )
        for row in probability:
            axes[1].step(grid, row, color="0.75", lw=0.5, alpha=0.5)
    kmf = KaplanMeierFitter().fit(time, event, label="KM (cohort)")
    kmf.plot_survival_function(ax=axes[1], ci_show=False, color="#0072B2")
    axes[1].set_xlabel(sanitize_label(time_label))
    axes[1].set_ylabel("Survival probability")
    axes[1].set_ylim(0.0, 1.02)

    # Panel 3: follow-up along the risk ranking.
    axes[2].scatter(
        rank[~event[order]],
        time[order][~event[order]],
        s=10,
        label="censored",
    )
    axes[2].scatter(
        rank[event[order]],
        time[order][event[order]],
        s=10,
        label="event",
    )
    axes[2].set_xlabel("Subjects by increasing risk")
    axes[2].set_ylabel(sanitize_label(time_label))
    axes[2].legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Time-dependent AUC
# ---------------------------------------------------------------------------


def plot_time_dependent_auc(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    n_times: int = 50,
    time_label: str = "Time",
):
    """
    Uno's cumulative/dynamic AUC as a function of follow-up time.

    Args:
        time: Observed follow-up durations.
        event: Event indicators, True for observed events.
        risk: Per-subject risk scores (higher means shorter survival).
        n_times: Number of grid points across the evaluable range.
        time_label: X-axis label (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    metrics = _sksurv("plot_time_dependent_auc")
    plt = _plt()
    time, event = _check_survival_inputs(time, event, "plot_time_dependent_auc")
    risk = np.asarray(risk, dtype=np.float64)

    target = np.empty(time.size, dtype=[("event", np.bool_), ("time", np.float64)])
    target["event"] = event
    target["time"] = time
    event_times = time[event]
    lower = float(event_times.min()) if event_times.size else float(time.min())
    upper = float(time.max())
    step = (upper - lower) / max(n_times, 2)
    grid = np.linspace(lower, upper - 0.5 * step, n_times)
    auc_values, _ = metrics.cumulative_dynamic_auc(target, target, risk, grid)

    fig, ax = plt.subplots()
    ax.plot(grid, auc_values, color="#0072B2")
    ax.axhline(0.5, color="0.5", ls="--", lw=0.8, label="chance")
    ax.set_xlabel(sanitize_label(time_label))
    ax.set_ylabel("Time-dependent AUC")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Calibration at a fixed horizon
# ---------------------------------------------------------------------------


def plot_survival_calibration(
    time: np.ndarray,
    event: np.ndarray,
    predicted: np.ndarray,
    *,
    horizon: float,
    n_groups: int = 5,
    time_label: str = "Time",
):
    """
    Calibration of the predicted horizon survival against Kaplan-Meier truth.

    Subjects are binned by their predicted probability of surviving past
    ``horizon``; for each bin the mean prediction is plotted against the
    observed Kaplan-Meier survival at that horizon. Points on the diagonal
    mean the predicted and observed risks agree.

    Args:
        time: Observed follow-up durations.
        event: Event indicators, True for observed events.
        predicted: Per-subject predicted probability of surviving past
            ``horizon`` (values in ``[0, 1]``).
        horizon: The time horizon the predictions refer to.
        n_groups: Number of predicted-risk bins.
        time_label: Time units, used in axis annotation (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    _lifelines("plot_survival_calibration")
    from lifelines import KaplanMeierFitter

    plt = _plt()
    time, event = _check_survival_inputs(time, event, "plot_survival_calibration")
    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.shape != time.shape:
        raise HABITAPIError(
            "plot_survival_calibration: predicted must align with time/event."
        )
    if ((predicted < 0) | (predicted > 1)).any():
        raise HABITAPIError(
            "plot_survival_calibration: predicted probabilities must lie in [0, 1]."
        )

    edges = np.quantile(predicted, np.linspace(0.0, 1.0, n_groups + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    centres, observed = [], []
    for i in range(n_groups):
        mask = (predicted > edges[i]) & (predicted <= edges[i + 1])
        if not mask.any():
            continue
        kmf = KaplanMeierFitter().fit(time[mask], event[mask])
        # KM survival evaluated exactly at the horizon is the observed truth.
        observed.append(float(kmf.predict(horizon)))
        centres.append(float(predicted[mask].mean()))
    centres = np.asarray(centres)
    observed = np.asarray(observed)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=0.8, label="ideal")
    ax.plot(centres, observed, "o-", color="#D55E00", label="model")
    ax.set_xlabel(f"Predicted survival at {horizon:g} {sanitize_label(time_label)}")
    ax.set_ylabel(f"Observed survival at {horizon:g} {sanitize_label(time_label)}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Brier score curve
# ---------------------------------------------------------------------------


def plot_brier_curve(
    time: np.ndarray,
    event: np.ndarray,
    survival_probability: np.ndarray,
    times: np.ndarray,
    *,
    time_label: str = "Time",
):
    """
    Brier score of the predicted survival function across follow-up time.

    Args:
        time: Observed follow-up durations.
        event: Event indicators, True for observed events.
        survival_probability: ``(n_subjects, n_times)`` predicted S(t|x).
        times: The times the probability columns align to.
        time_label: X-axis label (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    metrics = _sksurv("plot_brier_curve")
    plt = _plt()
    time, event = _check_survival_inputs(time, event, "plot_brier_curve")
    probability = np.asarray(survival_probability, dtype=np.float64)
    grid = np.asarray(times, dtype=np.float64)
    if probability.shape != (time.size, grid.size):
        raise HABITAPIError(
            "plot_brier_curve: survival_probability must have shape "
            f"(n_subjects, n_times) = {(time.size, grid.size)}; got "
            f"{probability.shape}."
        )
    target = np.empty(time.size, dtype=[("event", np.bool_), ("time", np.float64)])
    target["event"] = event
    target["time"] = time
    _, scores = metrics.brier_score(target, target, probability, grid)

    fig, ax = plt.subplots()
    ax.plot(grid, scores, color="#009E73")
    ax.set_xlabel(sanitize_label(time_label))
    ax.set_ylabel("Brier score")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cox coefficient forest plot
# ---------------------------------------------------------------------------


def plot_cox_forest(
    names: Sequence[str],
    hazard_ratio: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    p_values: Optional[np.ndarray] = None,
    title: str = "Hazard ratios",
):
    """
    Forest plot of hazard ratios with 95% confidence intervals (log scale).

    Args:
        names: Covariate names (sanitised).
        hazard_ratio: Point estimate of each hazard ratio.
        lower: Lower confidence bound per covariate.
        upper: Upper confidence bound per covariate.
        p_values: Optional per-covariate p-value, annotated alongside.
        title: Figure title (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    plt = _plt()
    names = [sanitize_label(n) for n in names]
    hr = np.asarray(hazard_ratio, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (hr.shape == lower.shape == upper.shape == (len(names),)):
        raise HABITAPIError(
            "plot_cox_forest: hazard_ratio, lower and upper must each have "
            f"one value per name ({len(names)})."
        )

    fig_height = max(2.0, 0.35 * len(names) + 1.0)
    fig, ax = plt.subplots(figsize=(5.0, fig_height))
    y = np.arange(len(names))[::-1]
    ax.errorbar(
        hr,
        y,
        xerr=np.vstack([hr - lower, upper - hr]),
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        capsize=3,
    )
    ax.axvline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Hazard ratio (95% CI, log scale)")
    ax.set_title(sanitize_label(title))
    if p_values is not None:
        p_values = np.asarray(p_values, dtype=np.float64)
        if p_values.shape != hr.shape:
            raise HABITAPIError(
                "plot_cox_forest: p_values must align with hazard_ratio."
            )
        x_max = float(np.max(upper[np.isfinite(upper)])) if np.isfinite(upper).any() else 2.0
        for yi, p in zip(y, p_values):
            ax.text(x_max * 1.5, yi, f"p = {p:.3g}", va="center", fontsize=7)
    fig.tight_layout()
    return fig
