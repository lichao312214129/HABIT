# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for habitat-feature contrast figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from matplotlib.collections import PolyCollection

from habit.contracts.table import FeatureTable
from habit.domain.habitat_features.compare import compare_habitat_features
from habit.viz import (
    plot_habitat_feature_bars,
    plot_habitat_feature_effect,
    plot_habitat_feature_heatmap,
    plot_habitat_feature_violin,
)
from habit.viz.habitat_features import _readable_feature_label, _zscore_columns

pytestmark = pytest.mark.unit


def _comparison(n_subjects: int = 10):
    """Small synthetic comparison for figure smoke tests."""
    rng = np.random.default_rng(4)
    rows = []
    for index in range(n_subjects):
        row = {"subject": f"s{index:03d}"}
        for hid, shift in ((1, 0.0), (2, 0.9)):
            row[f"has_habitat_{hid}"] = 1.0
            for feat_i in range(6):
                row[f"habitat_{hid}_tex_{feat_i:02d}_of_T2"] = float(
                    rng.normal(shift + 0.05 * feat_i, 0.25)
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    table = FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=tuple(c for c in frame.columns if c != "subject"),
    )
    return compare_habitat_features(table)


def _assert_ascii(fig: Figure) -> None:
    """Every drawn label must stay journal-safe ASCII."""
    for ax in fig.axes:
        for text in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            assert str(text.get_text()).isascii()
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            assert str(label.get_text()).isascii()


def test_heatmap_cohort_and_subject_return_figures() -> None:
    """Cohort mean and single-subject heatmaps both return a Figure."""
    comparison = _comparison()
    fig_c = plot_habitat_feature_heatmap(comparison, title="Cohort heatmap")
    fig_s = plot_habitat_feature_heatmap(
        comparison, subject_id="s000", title="Subject heatmap"
    )
    assert isinstance(fig_c, Figure)
    assert isinstance(fig_s, Figure)
    _assert_ascii(fig_c)
    _assert_ascii(fig_s)


def test_effect_violin_bars_return_figures() -> None:
    """The three detail figures return live Figures with ASCII text."""
    comparison = _comparison()
    fig_e = plot_habitat_feature_effect(comparison, pair=(2, 1), top_k=4)
    fig_v = plot_habitat_feature_violin(comparison, max_features=4)
    fig_b = plot_habitat_feature_bars(comparison, max_features=4)
    fig_one = plot_habitat_feature_bars(
        comparison, subject_id="s001", max_features=3
    )
    for fig in (fig_e, fig_v, fig_b, fig_one):
        assert isinstance(fig, Figure)
        _assert_ascii(fig)


def _mixed_scale_comparison(n_subjects: int = 10):
    """
    Cohort whose features cannot share one linear y-axis.

    ``original_firstorder_Energy_of_T2`` is ~1e9; ``volume_fraction`` is
    in ``[0, 1]``. Mean / Median / Kurtosis names must stay distinguishable
    after labelling.
    """
    rng = np.random.default_rng(7)
    feature_specs = (
        ("original_firstorder_Mean_of_T2", 120.0, 8.0),
        ("original_firstorder_Median_of_T2", 110.0, 7.0),
        ("original_firstorder_Energy_of_T2", 1.8e9, 2.0e8),
        ("original_firstorder_Kurtosis_of_T2", 3.2, 0.4),
        ("volume_fraction", 0.35, 0.08),
    )
    rows = []
    for index in range(n_subjects):
        row = {"subject": f"s{index:03d}"}
        for hid, shift in ((1, 0.0), (2, 0.35)):
            row[f"has_habitat_{hid}"] = 1.0
            for name, loc, scale in feature_specs:
                value = float(rng.normal(loc * (1.0 + 0.15 * shift), scale))
                if name == "volume_fraction":
                    value = float(np.clip(value, 0.05, 0.95))
                row[f"habitat_{hid}_{name}"] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    table = FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=tuple(c for c in frame.columns if c != "subject"),
    )
    return compare_habitat_features(table)


def _visible_axes(fig: Figure):
    """Return axes that still draw data (skip colorbars / hidden facets)."""
    return [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]


def test_readable_feature_labels_keep_mean_median_energy_kurtosis() -> None:
    """Wrapped labels stay unique and still name the statistic."""
    names = (
        "original_firstorder_Mean_of_T2",
        "original_firstorder_Median_of_T2",
        "original_firstorder_Energy_of_T2",
        "original_firstorder_Kurtosis_of_T2",
    )
    labels = [_readable_feature_label(name) for name in names]
    assert len(set(labels)) == len(names)
    for token, label in zip(("Mean", "Median", "Energy", "Kurtosis"), labels):
        assert token in label
        assert "..." not in label
        assert label.isascii()


def test_bars_incommensurable_features_keep_visible_heights() -> None:
    """Energy and volume_fraction each keep a y-scale that shows their bars."""
    comparison = _mixed_scale_comparison()
    features = (
        "original_firstorder_Energy_of_T2",
        "volume_fraction",
    )
    fig = plot_habitat_feature_bars(comparison, features=features)
    assert isinstance(fig, Figure)
    _assert_ascii(fig)
    axes = _visible_axes(fig)
    assert len(axes) >= 2
    heights_by_title: dict[str, list[float]] = {}
    for ax in axes:
        title = str(ax.get_title())
        heights = [float(patch.get_height()) for patch in ax.patches]
        if heights:
            heights_by_title[title] = heights
    energy_key = next(k for k in heights_by_title if "Energy" in k)
    frac_key = next(k for k in heights_by_title if "volume_fraction" in k)
    energy_max = max(heights_by_title[energy_key])
    frac_max = max(heights_by_title[frac_key])
    # Shared-axis bug: fraction bars sit at ~1e-9 of the Energy scale.
    assert energy_max > 1.0e8
    assert 0.05 <= frac_max <= 1.5
    energy_ylim = next(ax.get_ylim()[1] for ax in axes if "Energy" in ax.get_title())
    frac_ylim = next(
        ax.get_ylim()[1] for ax in axes if "volume_fraction" in ax.get_title()
    )
    assert frac_ylim < 5.0
    assert energy_ylim > 1.0e8


def test_heatmap_zscore_is_per_feature_across_habitats() -> None:
    """Heatmap cells are column-wise z-scores; cells stay near-square."""
    comparison = _mixed_scale_comparison()
    fig = plot_habitat_feature_heatmap(
        comparison,
        features=(
            "original_firstorder_Energy_of_T2",
            "volume_fraction",
        ),
    )
    fig.canvas.draw()
    ax = fig.axes[0]
    image = ax.images[0]
    shown = np.asarray(image.get_array(), dtype=np.float64)
    for col in range(shown.shape[1]):
        col_vals = shown[:, col]
        finite = col_vals[np.isfinite(col_vals)]
        assert finite.size >= 2
        assert abs(float(np.mean(finite))) < 1.0e-8
        assert abs(float(np.std(finite, ddof=0)) - 1.0) < 1.0e-8
    bbox = image.get_window_extent(fig.canvas.get_renderer())
    cell_w = bbox.width / shown.shape[1]
    cell_h = bbox.height / shown.shape[0]
    assert 0.55 <= (cell_w / cell_h) <= 1.80
    _ = _zscore_columns


def test_effect_xlim_includes_negative_ticks() -> None:
    """Lollipop x-axis is symmetric so negative delta keeps numeric ticks."""
    comparison = _mixed_scale_comparison()
    fig = plot_habitat_feature_effect(comparison, pair=(2, 1), top_k=5)
    ax = fig.axes[0]
    left, right = ax.get_xlim()
    assert left < 0.0
    assert right > 0.0
    assert abs(left + right) < 1.0e-6
    fig.canvas.draw()
    tick_values = [float(t) for t in ax.get_xticks() if t < 0]
    assert tick_values


def test_violin_uses_box_when_n_is_small() -> None:
    """n<5 per habitat: box + strip, not an angular KDE violin."""
    comparison = _mixed_scale_comparison(n_subjects=3)
    fig = plot_habitat_feature_violin(
        comparison,
        features=("volume_fraction",),
        max_features=1,
    )
    ax = _visible_axes(fig)[0]
    violin_bodies = [
        artist for artist in ax.collections if isinstance(artist, PolyCollection)
    ]
    assert violin_bodies == []
    assert ax.patches  # box faces
    assert ax.collections  # strip points
