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

from habit.contracts.table import FeatureTable
from habit.domain.habitat_features.compare import compare_habitat_features
from habit.viz import (
    plot_habitat_feature_bars,
    plot_habitat_feature_effect,
    plot_habitat_feature_heatmap,
    plot_habitat_feature_violin,
)

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
