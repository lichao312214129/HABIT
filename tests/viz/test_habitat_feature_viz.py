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
from habit.exceptions import HABITAPIError
from habit.viz import (
    plot_habitat_feature_bars,
    plot_habitat_feature_components,
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


def _three_habitat_comparison(n_subjects: int = 10, n_features: int = 6):
    """Cohort with three habitats so the all-pair heatmap has three columns."""
    rng = np.random.default_rng(11)
    rows = []
    for index in range(n_subjects):
        row = {"subject": f"s{index:03d}"}
        for hid, shift in ((1, 0.0), (2, 0.8), (3, -0.5)):
            row[f"has_habitat_{hid}"] = 1.0
            for feat_i in range(n_features):
                row[f"habitat_{hid}_tex_{feat_i:02d}_of_T2"] = float(
                    rng.normal(shift + 0.04 * feat_i, 0.22)
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    table = FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=tuple(c for c in frame.columns if c != "subject"),
    )
    return compare_habitat_features(table)


def _heatmap_data_ax(fig: Figure):
    """Return the effect-heatmap axes (skip the colorbar)."""
    for ax in fig.axes:
        if ax.images:
            return ax
    raise AssertionError("expected an imshow heatmap axes")


def test_effect_default_is_all_pair_heatmap() -> None:
    """Omitting pair draws one column per habitat pair, not a lollipop."""
    comparison = _three_habitat_comparison()
    fig = plot_habitat_feature_effect(comparison)
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    _assert_ascii(fig)
    ax = _heatmap_data_ax(fig)
    labels = [str(tick.get_text()) for tick in ax.get_xticklabels()]
    assert labels == ["H1-H2", "H1-H3", "H2-H3"]
    shown = np.asarray(ax.images[0].get_array())
    assert shown.shape[1] == 3
    assert shown.shape[0] == len(comparison.panel.feature_names)


def test_effect_habitats_two_ids_is_lollipop() -> None:
    """habitats=(a, b) is the same explicit-pair request as pair=(a, b)."""
    comparison = _three_habitat_comparison()
    fig = plot_habitat_feature_effect(comparison, habitats=(1, 3), top_k=4)
    fig.canvas.draw()
    ax = fig.axes[0]
    assert not ax.images
    xlabel = str(ax.get_xlabel())
    assert "H1 vs H3" in xlabel or "H3 vs H1" in xlabel
    _assert_ascii(fig)


def test_effect_explicit_pair_still_lollipop() -> None:
    """pair=(a, b) keeps the ranked forest / lollipop."""
    comparison = _comparison()
    fig = plot_habitat_feature_effect(comparison, pair=(2, 1), top_k=4)
    fig.canvas.draw()
    ax = fig.axes[0]
    assert not ax.images
    assert "H2 vs H1" in str(ax.get_xlabel()) or "H1 vs H2" in str(ax.get_xlabel())
    assert ax.collections  # scatter markers


def test_effect_heatmap_truncates_to_top_k_by_max_abs_delta() -> None:
    """More features than max_features: title states the silent-cap."""
    comparison = _three_habitat_comparison(n_features=20)
    fig = plot_habitat_feature_effect(comparison, max_features=5)
    fig.canvas.draw()
    ax = _heatmap_data_ax(fig)
    title = str(ax.get_title())
    assert "top 5 of 20" in title
    assert "max |delta|" in title
    y_labels = [str(tick.get_text()) for tick in ax.get_yticklabels()]
    assert len(y_labels) == 5
    assert len(set(y_labels)) == 5
    _assert_ascii(fig)


def test_effect_heatmap_feature_labels_stay_unique() -> None:
    """Wrapped radiomics names stay distinguishable on the delta heatmap."""
    comparison = _mixed_scale_comparison()
    fig = plot_habitat_feature_effect(comparison)
    fig.canvas.draw()
    ax = _heatmap_data_ax(fig)
    labels = [str(tick.get_text()) for tick in ax.get_yticklabels()]
    assert len(set(labels)) == len(labels)
    joined = "\n".join(labels)
    for token in ("Mean", "Median", "Energy", "Kurtosis"):
        assert token in joined


def _figure_title_text(fig: Figure) -> str:
    """Join suptitle + axes titles (ASCII contrast copy, not embeddings)."""
    parts: list[str] = []
    if getattr(fig, "_suptitle", None) is not None:
        parts.append(str(fig._suptitle.get_text()))
    for ax in fig.axes:
        parts.append(str(ax.get_title()))
    return "\n".join(parts)


def test_components_default_is_cva_habitat_contrast() -> None:
    """Default method is CVA; the hero is habitat contrast, not a scatter."""
    import inspect

    comparison = _three_habitat_comparison()
    signature = inspect.signature(plot_habitat_feature_components)
    assert signature.parameters["method"].default == "cva"
    assert signature.parameters["n_components"].default == 2
    fig = plot_habitat_feature_components(comparison)
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    _assert_ascii(fig)
    joined = _figure_title_text(fig).lower()
    assert "cva" in joined
    assert "contrast" in joined
    assert "embedding" not in joined
    assert "dimensionality" not in joined
    contrast_ax = fig.axes[0]
    tick_labels = [str(tick.get_text()) for tick in contrast_ax.get_xticklabels()]
    assert tick_labels == ["H1", "H2", "H3"]
    assert contrast_ax.patches  # habitat bars or box faces


def test_components_pca_and_cva_return_figures() -> None:
    """PCA and CVA both build contrast panels with habitat x-ticks."""
    comparison = _three_habitat_comparison()
    fig_pca = plot_habitat_feature_components(comparison, method="pca")
    fig_cva = plot_habitat_feature_components(comparison, method="cva")
    for fig in (fig_pca, fig_cva):
        assert isinstance(fig, Figure)
        fig.canvas.draw()
        _assert_ascii(fig)
        joined = _figure_title_text(fig).lower()
        assert "contrast" in joined
        assert "embedding" not in joined
        ticks = [str(tick.get_text()) for tick in fig.axes[0].get_xticklabels()]
        assert ticks == ["H1", "H2", "H3"]
    assert "PCA" in _figure_title_text(fig_pca)
    assert "CVA" in _figure_title_text(fig_cva)


def test_components_cva_uses_pca_when_p_exceeds_n() -> None:
    """p >= n - n_classes: title must say CVA (PCA-preprocessed)."""
    # 4 subjects x 2 habitats = 8 rows; 12 features => within-class rank 6.
    rng = np.random.default_rng(3)
    rows = []
    for index in range(4):
        row = {"subject": f"s{index:03d}"}
        for hid, shift in ((1, 0.0), (2, 0.9)):
            row[f"has_habitat_{hid}"] = 1.0
            for feat_i in range(12):
                row[f"habitat_{hid}_tex_{feat_i:02d}_of_T2"] = float(
                    rng.normal(shift, 0.25)
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    table = FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=tuple(c for c in frame.columns if c != "subject"),
    )
    wide = compare_habitat_features(table)
    fig = plot_habitat_feature_components(wide, method="cva")
    title = _figure_title_text(fig)
    assert "CVA (PCA-preprocessed)" in title
    assert "contrast" in title.lower()
    assert "embedding" not in title.lower()
    _assert_ascii(fig)


def test_components_loadings_keep_readable_feature_names() -> None:
    """Loadings wrap radiomics names; they must not mid-truncate with ..."""
    comparison = _mixed_scale_comparison()
    fig = plot_habitat_feature_components(comparison, method="cva")
    fig.canvas.draw()
    joined = "\n".join(
        str(tick.get_text())
        for ax in fig.axes
        for tick in list(ax.get_yticklabels()) + list(ax.get_xticklabels())
    )
    assert "..." not in joined
    for token in ("Mean", "Median", "Energy", "Kurtosis"):
        assert token in joined
    _assert_ascii(fig)


def test_components_cva_fails_clearly_when_n_equals_classes() -> None:
    """One subject x two habitats cannot form a within-class scatter."""
    comparison = _comparison(n_subjects=1)
    with pytest.raises(HABITAPIError, match="more \\(subject, habitat\\) rows"):
        plot_habitat_feature_components(comparison, method="cva")
