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
"""Auxiliary recipe shells and 3D habitat viz."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from habit.recipes.auxiliary import merge_tables


@pytest.mark.unit
def test_merge_tables_joins_on_index(tmp_path: Path) -> None:
    """merge_tables mirrors the CLI merge-csv join semantics."""
    left = tmp_path / "a.csv"
    right = tmp_path / "b.csv"
    pd.DataFrame({"subject_id": ["s1", "s2"], "feat_a": [1.0, 2.0]}).to_csv(left, index=False)
    pd.DataFrame({"subject_id": ["s1", "s2"], "feat_b": [3.0, 4.0]}).to_csv(right, index=False)

    merged = merge_tables([str(left), str(right)], index_cols=["subject_id"])
    assert list(merged.index) == ["s1", "s2"]
    assert "feat_a" in merged.columns and "feat_b" in merged.columns


@pytest.mark.unit
def test_plot_habitat_clustering_pca_3d_returns_figure() -> None:
    """Static 3D PCA plot renders without touching the filesystem."""
    from habit.viz import plot_habitat_clustering_pca_3d

    features = np.random.default_rng(0).normal(size=(20, 4))
    labels = np.array([0, 0, 0, 1, 1] * 4)
    fig = plot_habitat_clustering_pca_3d(features, labels, n_clusters=2)
    assert fig.axes
