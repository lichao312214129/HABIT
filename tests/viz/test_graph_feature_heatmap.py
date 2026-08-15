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
"""Unit tests for ``plot_graph_feature_heatmap`` (synthetic tables only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from habit.exceptions import HABITAPIError
from habit.viz import plot_graph_feature_heatmap

pytestmark = pytest.mark.unit


def _synthetic_graph_table(n_subjects: int = 5) -> pd.DataFrame:
    """
    Build a wide subject x graph-feature frame with known variances.

    Args:
        n_subjects: Number of rows (people).

    Returns:
        pd.DataFrame: Identifier column ``subject_id`` plus graph columns.
    """
    rng = np.random.default_rng(11)
    rows = []
    for index in range(n_subjects):
        rows.append(
            {
                "subject_id": f"subj{index + 1:03d}",
                "single_h1_high_var": float(10.0 * index),
                "single_h1_low_var": float(1.0 + 0.01 * index),
                "single_h2_mid_var": float(index % 3),
                "pair_h1_h2_high_var": float(20.0 * index),
                "pair_h1_h2_low_var": float(0.5 + 0.001 * index),
                "pair_h2_h3_mid_var": float((index * 2) % 5),
                "graph_num_habitats": 4.0,
                "graph_num_nodes_total": float(80 + index),
            }
        )
    frame = pd.DataFrame(rows)
    for extra in range(6):
        frame[f"single_h3_extra_{extra:02d}"] = rng.normal(
            loc=float(extra), scale=0.2, size=n_subjects
        )
        frame[f"pair_h1_h3_extra_{extra:02d}"] = rng.normal(
            loc=float(extra), scale=0.15, size=n_subjects
        )
    return frame


def _assert_ascii(fig: Figure) -> None:
    """Every drawn label must stay journal-safe ASCII (no CJK / U+2212)."""
    for ax in fig.axes:
        for text in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            assert str(text.get_text()).isascii()
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            assert str(label.get_text()).isascii()


def _x_tick_names(fig: Figure) -> list[str]:
    """Return x-tick strings with wrapped newlines flattened to '_'."""
    fig.canvas.draw()
    return [
        str(label.get_text()).replace("\n", "_")
        for label in fig.axes[0].get_xticklabels()
    ]


def _y_tick_names(fig: Figure) -> list[str]:
    """Return y-tick strings (subject ids)."""
    fig.canvas.draw()
    return [str(label.get_text()) for label in fig.axes[0].get_yticklabels()]


def test_subjects_filter_orders_rows_and_rejects_missing() -> None:
    """``subjects`` keeps only the named people, in that order."""
    table = _synthetic_graph_table()
    fig = plot_graph_feature_heatmap(
        table,
        subjects=("subj005", "subj001"),
        features=("single_h1_high_var", "single_h2_mid_var"),
        zscore=True,
        title="Subject filter",
    )
    assert isinstance(fig, Figure)
    assert _y_tick_names(fig) == ["subj005", "subj001"]
    _assert_ascii(fig)
    with pytest.raises(HABITAPIError, match="not in the table"):
        plot_graph_feature_heatmap(
            table,
            subjects=("subj001", "missing_id"),
            features=("single_h1_high_var",),
            zscore=False,
        )


def test_explicit_features_override_group_and_cap() -> None:
    """An explicit ``features`` list is drawn as-is (caller order)."""
    table = _synthetic_graph_table()
    wanted = ("pair_h1_h2_low_var", "single_h1_high_var", "graph_num_habitats")
    fig = plot_graph_feature_heatmap(
        table,
        features=wanted,
        n_features=1,
        feature_group="single",
        select="variance",
        title="Explicit features",
    )
    assert _x_tick_names(fig) == list(wanted)
    _assert_ascii(fig)
    with pytest.raises(HABITAPIError, match="feature column"):
        plot_graph_feature_heatmap(
            table,
            features=("single_h1_high_var", "not_a_column"),
            zscore=True,
        )


def test_n_features_variance_vs_sample() -> None:
    """Variance picks the high-var column; sample is seeded and different."""
    table = _synthetic_graph_table()
    fig_var = plot_graph_feature_heatmap(
        table,
        n_features=1,
        feature_group="single",
        select="variance",
        title="Variance top-1",
    )
    assert _x_tick_names(fig_var) == ["single_h1_high_var"]
    fig_a = plot_graph_feature_heatmap(
        table,
        n_features=3,
        feature_group="single",
        select="sample",
        sample_seed=0,
        title="Sample seed 0",
    )
    fig_b = plot_graph_feature_heatmap(
        table,
        n_features=3,
        feature_group="single",
        select="sample",
        sample_seed=0,
        title="Sample seed 0 again",
    )
    fig_c = plot_graph_feature_heatmap(
        table,
        n_features=3,
        feature_group="single",
        select="sample",
        sample_seed=1,
        title="Sample seed 1",
    )
    names_a = _x_tick_names(fig_a)
    assert names_a == _x_tick_names(fig_b)
    assert names_a != _x_tick_names(fig_c)
    assert all(name.startswith("single_h") for name in names_a)
    assert "graph_num_habitats" not in names_a


def test_feature_group_single_vs_pair_excludes_graph_num() -> None:
    """``single`` / ``pair`` filter prefixes; ``graph_num_*`` needs ``all``."""
    table = _synthetic_graph_table()
    fig_single = plot_graph_feature_heatmap(
        table, n_features=8, feature_group="single", select="variance"
    )
    fig_pair = plot_graph_feature_heatmap(
        table, n_features=8, feature_group="pair", select="variance"
    )
    fig_all = plot_graph_feature_heatmap(
        table, n_features=20, feature_group="all", select="variance"
    )
    single_names = _x_tick_names(fig_single)
    pair_names = _x_tick_names(fig_pair)
    all_names = _x_tick_names(fig_all)
    assert all(name.startswith("single_h") for name in single_names)
    assert all(name.startswith("pair_h") for name in pair_names)
    assert not any(name.startswith("graph_num_") for name in single_names)
    assert not any(name.startswith("graph_num_") for name in pair_names)
    assert any(name.startswith("graph_num_") for name in all_names)


def test_zscore_columns_have_mean_near_zero() -> None:
    """Each drawn column is a z-score across the selected subjects."""
    table = _synthetic_graph_table()
    fig = plot_graph_feature_heatmap(
        table,
        features=("single_h1_high_var", "pair_h1_h2_high_var"),
        zscore=True,
        title="Z-score check",
    )
    fig.canvas.draw()
    shown = np.asarray(fig.axes[0].images[0].get_array(), dtype=np.float64)
    assert shown.shape[0] == 5
    for col in range(shown.shape[1]):
        finite = shown[:, col][np.isfinite(shown[:, col])]
        assert finite.size >= 2
        assert abs(float(np.mean(finite))) < 1.0e-8
        assert abs(float(np.std(finite, ddof=0)) - 1.0) < 1.0e-8
    with pytest.raises(HABITAPIError, match="at least 2"):
        plot_graph_feature_heatmap(
            table,
            subjects=("subj001",),
            features=("single_h1_high_var", "single_h2_mid_var"),
            zscore=True,
        )


def test_zscore_false_signed_delta_uses_diverging_scale() -> None:
    """A precomputed 5-minus-8 table draws raw signed values, not z-scores."""
    table = pd.DataFrame(
        {
            "subject_id": ["subj001", "subj002", "subj003"],
            "single_h1_n_nodes": [4.0, -2.0, 1.0],
            "single_h2_n_nodes": [-1.0, 3.0, 0.5],
        }
    )
    fig = plot_graph_feature_heatmap(
        table,
        features=("single_h1_n_nodes", "single_h2_n_nodes"),
        zscore=False,
        title="Graph features: 5-voxel minus 8-voxel",
    )
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    shown = np.asarray(fig.axes[0].images[0].get_array(), dtype=np.float64)
    assert shown.shape == (3, 2)
    np.testing.assert_allclose(shown[:, 0], [4.0, -2.0, 1.0])
    assert fig.axes[0].images[0].get_cmap().name == "RdBu_r"
    assert "5-voxel minus 8-voxel" in str(fig.axes[0].get_title())
    assert "Feature difference" in str(fig.axes[1].get_ylabel())
    _assert_ascii(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_title_and_axis_labels_are_english() -> None:
    """Default and custom titles stay English; axis labels are Subject / feature."""
    table = _synthetic_graph_table()
    fig = plot_graph_feature_heatmap(
        table,
        n_features=3,
        feature_group="single",
        title="Single-habitat graph features (column z-score)",
    )
    _assert_ascii(fig)
    ax = fig.axes[0]
    assert "Subject" in str(ax.get_ylabel())
    assert "Graph feature" in str(ax.get_xlabel())
    assert "Single-habitat" in str(ax.get_title())
    assert "Z-score" in str(fig.axes[1].get_ylabel())
    default = plot_graph_feature_heatmap(
        table, n_features=2, feature_group="pair", zscore=True
    )
    _assert_ascii(default)
    assert "Pairwise" in str(default.axes[0].get_title())
    assert "z-score" in str(default.axes[0].get_title()).lower()


def _paired_shift_tables(n_subjects: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build paired 5-voxel / 8-voxel tables with one shifted feature.

    Args:
        n_subjects: Number of paired rows (must be >= 3 for a t-test).

    Returns:
        Tuple of ``(table_5, table_8)``. ``single_h1_n_nodes`` has a
        large paired shift; ``single_h1_avg_degree`` is a null pair.
    """
    rng = np.random.default_rng(21)
    subjects = [f"subj{index + 1:03d}" for index in range(n_subjects)]
    baseline_shift = rng.normal(loc=10.0, scale=1.0, size=n_subjects)
    baseline_null = rng.normal(loc=2.0, scale=0.4, size=n_subjects)
    table_8 = pd.DataFrame(
        {
            "subject_id": subjects,
            "single_h1_n_nodes": baseline_shift,
            "single_h1_avg_degree": baseline_null,
        }
    )
    table_5 = pd.DataFrame(
        {
            "subject_id": subjects,
            # Large mean shift plus a little noise so the paired
            # difference is not constant (a constant column is skipped).
            "single_h1_n_nodes": baseline_shift
            + 4.0
            + rng.normal(loc=0.0, scale=0.12, size=n_subjects),
            "single_h1_avg_degree": baseline_null
            + rng.normal(loc=0.0, scale=0.02, size=n_subjects),
        }
    )
    return table_5, table_8


def test_star_significant_requires_reference() -> None:
    """A precomputed delta alone must not be starred (pairing unknown)."""
    table = _synthetic_graph_table()
    with pytest.raises(HABITAPIError, match="requires reference"):
        plot_graph_feature_heatmap(
            table,
            features=("single_h1_high_var", "single_h2_mid_var"),
            zscore=True,
            star_significant=True,
        )


def test_reference_zscore_and_star_marks_shifted_feature() -> None:
    """Column z-score of the raw delta; FDR star on the shifted feature only."""
    table_5, table_8 = _paired_shift_tables(n_subjects=8)
    fig = plot_graph_feature_heatmap(
        table_5,
        reference=table_8,
        features=("single_h1_n_nodes", "single_h1_avg_degree"),
        zscore=True,
        star_significant=True,
        star_alpha=0.05,
        star_test="ttest_rel",
        star_mtc="fdr_bh",
        title="Graph features: 5-voxel minus 8-voxel",
        cbar_label="Z-scored difference (5 - 8)",
    )
    assert isinstance(fig, Figure)
    ticks = _x_tick_names(fig)
    assert ticks[0] == "single_h1_n_nodes *"
    assert ticks[1] == "single_h1_avg_degree"
    fig.canvas.draw()
    shown = np.asarray(fig.axes[0].images[0].get_array(), dtype=np.float64)
    assert shown.shape == (8, 2)
    for col in range(shown.shape[1]):
        finite = shown[:, col][np.isfinite(shown[:, col])]
        assert finite.size >= 2
        assert abs(float(np.mean(finite))) < 1.0e-8
    assert fig.axes[0].images[0].get_cmap().name == "RdBu_r"
    assert "Z-scored difference (5 - 8)" in str(fig.axes[1].get_ylabel())
    _assert_ascii(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_star_skips_constant_column() -> None:
    """A constant paired difference is not starred (n>=3 still)."""
    subjects = [f"subj{index + 1:03d}" for index in range(6)]
    table_8 = pd.DataFrame(
        {
            "subject_id": subjects,
            "single_h1_n_nodes": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "single_h1_constant": [3.0] * 6,
        }
    )
    table_5 = pd.DataFrame(
        {
            "subject_id": subjects,
            "single_h1_n_nodes": [3.1, 4.0, 5.2, 5.9, 7.1, 8.0],
            "single_h1_constant": [3.0] * 6,
        }
    )
    fig = plot_graph_feature_heatmap(
        table_5,
        reference=table_8,
        features=("single_h1_n_nodes", "single_h1_constant"),
        zscore=True,
        star_significant=True,
    )
    ticks = _x_tick_names(fig)
    assert ticks[0] == "single_h1_n_nodes *"
    assert ticks[1] == "single_h1_constant"
    import matplotlib.pyplot as plt

    plt.close(fig)
