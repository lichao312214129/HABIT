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
"""Tests for habitat-clustering figures in ``habit.viz``."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from habit.viz import plot_habitat_clustering_pca_2d, use_style

pytestmark = pytest.mark.unit


def _synthetic_clustering(seed: int = 0, n_per_cluster: int = 12):
    """Return (features, labels, centers) for three separated 2D blobs."""
    rng = np.random.RandomState(seed)
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]], dtype=np.float64)
    blocks = []
    labels = []
    for cluster_id, center in enumerate(centers):
        points = center + rng.normal(scale=0.35, size=(n_per_cluster, 2))
        blocks.append(points)
        labels.append(np.full(n_per_cluster, cluster_id + 1, dtype=np.int64))
    features = np.vstack(blocks)
    return features, np.concatenate(labels), centers


def test_plot_habitat_clustering_pca_2d_returns_figure_and_saves(tmp_path) -> None:
    """Synthetic cohort units render to an ASCII-only PNG on disk."""
    features, labels, centers = _synthetic_clustering()
    with use_style("radiology"):
        fig = plot_habitat_clustering_pca_2d(
            features,
            labels,
            centers=centers,
            n_clusters=3,
        )
    assert isinstance(fig, Figure)

    output_path = tmp_path / "habitat_clustering_2D.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    texts: list[str] = []
    for ax in fig.axes:
        texts.append(ax.get_title())
        texts.append(ax.get_xlabel())
        texts.append(ax.get_ylabel())
        legend = ax.get_legend()
        if legend is not None:
            texts.extend(entry.get_text() for entry in legend.get_texts())
    joined = " ".join(texts)
    assert joined.isascii(), joined
    assert "Habitat" in joined
    assert "PC1" in joined or "Feature 1" in joined

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_clustering_pca_2d_never_calls_show(monkeypatch) -> None:
    """The viz function stays pure: no display hooks."""
    import matplotlib.pyplot as plt

    calls = {"show": 0}
    original_show = plt.show

    def counting_show(*args, **kwargs):
        calls["show"] += 1
        return original_show(*args, **kwargs)

    monkeypatch.setattr(plt, "show", counting_show)
    features, labels, _ = _synthetic_clustering()
    plot_habitat_clustering_pca_2d(features, labels)
    assert calls["show"] == 0
