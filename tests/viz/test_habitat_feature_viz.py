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
"""Unit tests for habitat-feature contrast figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from habit.contracts.table import FeatureTable
from habit.domain.habitat_features.compare import compare_habitat_features
from habit.exceptions import HABITAPIError
from habit.viz import (
    plot_habitat_feature_bars,
    plot_habitat_feature_effect,
    plot_habitat_feature_heatmap,
    plot_habitat_feature_violin,
    plot_habitat_graph_pair_matrix,
)
from habit.viz.habitat_features import _short_feature_label

pytestmark = pytest.mark.unit


def _comparison(n_subjects: int = 12, *, missing_h2: bool = False):
    """
    Synthetic comparison for figure smoke tests.

    Habitat 2 is shifted so H2 vs H1 is detectably different. Optional
    ``missing_h2`` leaves one subject's habitat 2 as NaN (honest gap).
    """
    rng = np.random.default_rng(4)
    rows = []
    firstorder = (
        "original_firstorder_Mean_of_T2",
        "original_firstorder_Skewness_of_T2",
        "original_firstorder_Energy_of_T2",
        "original_firstorder_Entropy_of_T2",
        "voxel_count",
        "volume_fraction",
    )
    for index in range(n_subjects):
        row = {"subject": f"s{index:03d}"}
        for hid, shift in ((1, 0.0), (2, 1.1), (3, -0.4)):
            present = not (missing_h2 and hid == 2 and index == 0)
            row[f"has_habitat_{hid}"] = 1.0 if present else 0.0
            for feat_i, feat in enumerate(firstorder):
                value = float(rng.normal(shift + 0.08 * feat_i, 0.20))
                row[f"habitat_{hid}_{feat}"] = value if present else float("nan")
        rows.append(row)
    frame = pd.DataFrame(rows)
    table = FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=tuple(c for c in frame.columns if c != "subject"),
    )
    return compare_habitat_features(table)


def _graph_table(n_subjects: int = 8) -> FeatureTable:
    """Wide graph table with single_h* and pair_h* columns."""
    rng = np.random.default_rng(2)
    rows = []
    for index in range(n_subjects):
        row = {"subject": f"s{index:03d}"}
        for hid in (1, 2, 3):
            row[f"single_h{hid}_n_nodes"] = float(4 + hid + rng.normal(0, 0.3))
            row[f"single_h{hid}_edge_density"] = float(
                0.2 * hid + rng.normal(0, 0.05)
            )
        row["pair_h1_h2_contact_voxels_sum"] = float(rng.uniform(10, 40))
        row["pair_h1_h3_contact_voxels_sum"] = float(rng.uniform(5, 20))
        row["pair_h2_h3_contact_voxels_sum"] = float("nan") if index == 0 else float(
            rng.uniform(8, 25)
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=tuple(c for c in frame.columns if c != "subject"),
    )


def _assert_ascii(fig: Figure) -> None:
    """Every drawn label must stay journal-safe ASCII."""
    for ax in fig.axes:
        for text in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            assert str(text.get_text()).isascii()
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            assert str(label.get_text()).isascii()


def test_short_feature_label_drops_pyradiomics_boilerplate() -> None:
    """Axis labels must be readable, not original_firstorder_* stubs."""
    assert _short_feature_label("original_firstorder_Mean_of_LAP") == "Mean"
    assert _short_feature_label("original_glcm_Contrast_of_T2") == "GLCM Contrast"
    assert _short_feature_label("voxel_count") == "voxel count"
    assert _short_feature_label("n_nodes") == "n nodes"


def test_heatmap_cohort_and_subject_return_figures() -> None:
    """Cohort mean and single-subject heatmaps both return a Figure."""
    comparison = _comparison()
    fig_c = plot_habitat_feature_heatmap(comparison, title="Cohort heatmap")
    fig_s = plot_habitat_feature_heatmap(
        comparison, subject_id="s000", title="One case (s000)"
    )
    assert isinstance(fig_c, Figure)
    assert isinstance(fig_s, Figure)
    _assert_ascii(fig_c)
    _assert_ascii(fig_s)
    # Features as rows: y tick labels are shortened feature names.
    y_labels = [t.get_text() for t in fig_c.axes[0].get_yticklabels()]
    assert any(label == "Mean" for label in y_labels)
    assert not any("original_firstorder" in label for label in y_labels)


def test_heatmap_masks_missing_habitat_not_zero() -> None:
    """A NaN habitat x feature cell is masked, not drawn as zero."""
    comparison = _comparison(n_subjects=8, missing_h2=True)
    # Subject s000 has no habitat 2 -- the one-case heatmap must mask it.
    fig = plot_habitat_feature_heatmap(comparison, subject_id="s000")
    image = fig.axes[0].images[0]
    data = np.ma.asarray(image.get_array())
    assert np.ma.isMaskedArray(data)
    assert bool(np.ma.getmaskarray(data).any())


def test_effect_violin_bars_return_figures() -> None:
    """The three detail figures return live Figures with ASCII text."""
    comparison = _comparison()
    pair = comparison.strongest_pair()
    fig_e = plot_habitat_feature_effect(comparison, pair=pair, top_k=4)
    assert "q < 0.05" in fig_e.axes[0].get_xlabel()
    fig_v = plot_habitat_feature_violin(
        comparison, pair=pair, max_features=4, kind="box"
    )
    fig_b = plot_habitat_feature_bars(comparison, max_features=4)
    fig_one = plot_habitat_feature_bars(
        comparison, subject_id="s001", max_features=3
    )
    for fig in (fig_e, fig_v, fig_b, fig_one):
        assert isinstance(fig, Figure)
        _assert_ascii(fig)


def test_violin_defaults_to_the_contrasted_pair() -> None:
    """Without habitats=, only the strongest pair is drawn (one message)."""
    comparison = _comparison()
    pair = comparison.strongest_pair()
    fig = plot_habitat_feature_violin(comparison, max_features=2, kind="box")
    ax = fig.axes[0]
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == [f"H{pair[0]}", f"H{pair[1]}"]


def test_violin_rejects_unknown_kind() -> None:
    """kind= must be box or violin."""
    comparison = _comparison()
    with pytest.raises(HABITAPIError, match="kind"):
        plot_habitat_feature_violin(comparison, kind="hist")


def test_bars_omit_nan_instead_of_drawing_zero() -> None:
    """A missing habitat is not a zero-height bar."""
    comparison = _comparison(n_subjects=8, missing_h2=True)
    fig = plot_habitat_feature_bars(
        comparison, subject_id="s000", max_features=3, zscore=False
    )
    # Habitat 2 is absent for s000: three features x two habitats (H1, H3).
    rectangles = [
        patch
        for patch in fig.axes[0].patches
        if getattr(patch, "get_height", None) is not None
        and float(patch.get_width()) > 0
    ]
    assert len(rectangles) == 6


def test_graph_pair_matrix_returns_figure_and_masks_missing() -> None:
    """Pair contact matrix is symmetric; missing pairs stay masked."""
    table = _graph_table()
    fig = plot_habitat_graph_pair_matrix(table, metric="contact_voxels_sum")
    assert isinstance(fig, Figure)
    _assert_ascii(fig)
    image = fig.axes[0].images[0]
    data = np.ma.asarray(image.get_array())
    assert data.shape == (3, 3)
    mask = np.ma.getmaskarray(data)
    # Diagonal is never a pair; pair 2-3 is NaN for subject 0 but others
    # contribute, so only the diagonal must be masked.
    assert bool(mask[0, 0]) and bool(mask[1, 1]) and bool(mask[2, 2])
    assert not bool(mask[0, 1])
