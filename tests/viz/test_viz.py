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
"""Tests for the habit.viz publication-figure package."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from habit.api.exceptions import HABITAPIError
from habit.viz import (
    StyleSpec,
    available_styles,
    get_style,
    plot_bland_altman,
    plot_brier_curve,
    plot_coefficient_forest,
    plot_cox_forest,
    plot_kaplan_meier,
    plot_predicted_vs_observed,
    plot_residual_qq,
    plot_residuals,
    plot_risk_triptych,
    plot_survival_calibration,
    plot_time_dependent_auc,
    sanitize_label,
    use_style,
)

pytestmark = pytest.mark.unit


def _survival_data(n: int = 60, seed: int = 0):
    """(time, event, risk) with a genuine risk-survival ordering."""
    rng = np.random.RandomState(seed)
    risk = rng.normal(size=n)
    time = np.clip(10.0 * np.exp(-0.7 * risk) * rng.uniform(0.5, 1.5, n), 0.5, None)
    event = (rng.rand(n) < 0.7).astype(bool)
    if not event.any():
        event[0] = True
    return time, event, risk


# ---------------------------------------------------------------------------
# sanitize_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("MSI_score", "MSI_score"),  # ASCII passes through unchanged
        ("CAFé", "CAFe"),  # accent stripped to the ASCII base
        ("T1加权", "T1?"),  # CJK becomes a visible placeholder
        ("CAFé–x°", "CAFe-x deg "),  # dash and degree folded to ASCII
    ],
)
def test_sanitize_label(raw: str, expected: str) -> None:
    """Labels become journal-safe ASCII, lossily but visibly."""
    assert sanitize_label(raw) == expected
    assert sanitize_label(raw).isascii()


def test_sanitize_label_handles_non_strings() -> None:
    """Numbers and None are stringified then sanitised."""
    assert sanitize_label(3.5) == "3.5"
    assert sanitize_label(None) == "None"


# ---------------------------------------------------------------------------
# styles
# ---------------------------------------------------------------------------


def test_builtin_styles_registered() -> None:
    """The four presets ship and resolve by name."""
    assert set(available_styles()) == {"default", "radiology", "nature", "lancet"}
    for name in available_styles():
        assert isinstance(get_style(name), StyleSpec)


def test_unknown_style_raises_with_hint() -> None:
    """An unregistered style names the fix (register_style)."""
    with pytest.raises(HABITAPIError, match="register_style"):
        get_style("bmj")


def test_style_figsize_matches_journal_columns() -> None:
    """89 mm single-column is the Radiology/Nature width (3.5 inches)."""
    spec = get_style("radiology")
    width_in, _ = spec.figsize(columns=1)
    assert abs(width_in - 89.0 / 25.4) < 1e-6


# ---------------------------------------------------------------------------
# Survival figures return Figures
# ---------------------------------------------------------------------------


def test_plot_kaplan_meier_single_cohort() -> None:
    time, event, _ = _survival_data()
    with use_style("radiology"):
        fig = plot_kaplan_meier(time, event)
    assert isinstance(fig, Figure)


def test_plot_kaplan_meier_stratified_with_risk_table() -> None:
    time, event, risk = _survival_data()
    group = (risk > np.median(risk)).astype(int)
    with use_style("nature"):
        fig = plot_kaplan_meier(time, event, group, group_names=["low", "high"])
    assert isinstance(fig, Figure)


def test_plot_risk_triptych() -> None:
    time, event, risk = _survival_data()
    grid = np.linspace(1.0, 15.0, 20)
    probability = np.clip(
        np.exp(-0.05 * np.outer(np.exp(risk), grid)), 0.0, 1.0
    )
    fig = plot_risk_triptych(
        time, event, risk, survival_probability=probability, survival_times=grid
    )
    assert isinstance(fig, Figure)


def test_plot_time_dependent_auc() -> None:
    time, event, risk = _survival_data()
    fig = plot_time_dependent_auc(time, event, risk)
    assert isinstance(fig, Figure)


def test_plot_survival_calibration() -> None:
    time, event, risk = _survival_data()
    horizon = 10.0
    predicted = np.clip(np.exp(-0.05 * np.exp(risk) * horizon), 0.0, 1.0)
    fig = plot_survival_calibration(time, event, predicted, horizon=horizon)
    assert isinstance(fig, Figure)


def test_plot_brier_curve() -> None:
    time, event, risk = _survival_data()
    grid = np.linspace(time[event].min(), time.max() - 1.0, 25)
    probability = np.clip(
        np.exp(-0.05 * np.outer(np.exp(risk), grid)), 0.0, 1.0
    )
    fig = plot_brier_curve(time, event, probability, grid)
    assert isinstance(fig, Figure)


def test_plot_cox_forest() -> None:
    fig = plot_cox_forest(
        ["f1", "f2"],
        np.array([1.8, 0.7]),
        np.array([1.1, 0.4]),
        np.array([2.9, 1.2]),
        p_values=np.array([0.02, 0.31]),
    )
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Regression figures return Figures
# ---------------------------------------------------------------------------


def _regression_data(n: int = 40, seed: int = 0):
    rng = np.random.RandomState(seed)
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(0.0, 0.2, n)
    return y_true, y_pred


def test_plot_predicted_vs_observed() -> None:
    y_true, y_pred = _regression_data()
    fig = plot_predicted_vs_observed(y_true, y_pred)
    assert isinstance(fig, Figure)


def test_plot_residuals() -> None:
    y_true, y_pred = _regression_data()
    fig = plot_residuals(y_true, y_pred)
    assert isinstance(fig, Figure)


def test_plot_residual_qq() -> None:
    y_true, y_pred = _regression_data()
    fig = plot_residual_qq(y_true, y_pred)
    assert isinstance(fig, Figure)


def test_plot_bland_altman() -> None:
    y_true, y_pred = _regression_data()
    fig = plot_bland_altman(y_true, y_pred)
    assert isinstance(fig, Figure)


def test_plot_coefficient_forest_with_and_without_ci() -> None:
    y_true, y_pred = _regression_data()
    names = ["a", "b", "c"]
    coef = np.array([0.5, -0.2, 0.1])
    fig_ci = plot_coefficient_forest(
        names, coef, lower=coef - 0.1, upper=coef + 0.1
    )
    fig_plain = plot_coefficient_forest(names, coef)
    assert isinstance(fig_ci, Figure)
    assert isinstance(fig_plain, Figure)


# ---------------------------------------------------------------------------
# Purity: no CJK on a figure, no filesystem writes
# ---------------------------------------------------------------------------


def test_no_cjk_leaks_onto_a_figure() -> None:
    """A CJK stratum name is sanitised before reaching the axis."""
    time, event, risk = _survival_data()
    group = (risk > np.median(risk)).astype(int)
    fig = plot_kaplan_meier(
        time, event, group, group_names=["低风险", "高风险"], risk_table=False
    )
    texts = []
    for ax in fig.axes:
        texts.append(ax.get_title())
        texts.append(ax.get_xlabel())
        texts.append(ax.get_ylabel())
        legend = ax.get_legend()
        if legend is not None:
            texts.extend(t.get_text() for t in legend.get_texts())
    joined = " ".join(texts)
    assert joined.isascii(), joined
    assert "低" not in joined and "高" not in joined


def test_no_figure_writes_to_disk(tmp_path, monkeypatch) -> None:
    """The viz package never calls savefig or show, even when called."""
    import matplotlib.pyplot as plt

    calls = {"savefig": 0, "show": 0}
    original_savefig = Figure.savefig
    original_show = plt.show

    def counting_savefig(self, *args, **kwargs):
        calls["savefig"] += 1
        return original_savefig(self, *args, **kwargs)

    def counting_show(*args, **kwargs):
        calls["show"] += 1
        return original_show(*args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", counting_savefig)
    monkeypatch.setattr(plt, "show", counting_show)

    time, event, risk = _survival_data()
    plot_kaplan_meier(time, event, risk_table=False)
    y_true, y_pred = _regression_data()
    plot_predicted_vs_observed(y_true, y_pred)

    assert calls["savefig"] == 0
    assert calls["show"] == 0
