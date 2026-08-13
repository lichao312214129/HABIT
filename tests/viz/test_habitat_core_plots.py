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
"""Unit tests for habitat-core visualization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from habit.kernels.habitat_metrics import (
    habitat_region_stats,
    habitat_volume_fractions,
    ith_score,
    spatial_interaction_matrix,
)
from habit.viz import (
    plot_cluster_validation_curves,
    plot_cluster_validation_from_report,
    plot_habitat_label_compare,
    plot_habitat_volume_fractions,
    plot_ith_summary,
    plot_msi_matrix,
    plot_partition_triptych,
)


@pytest.fixture
def toy_labels() -> np.ndarray:
    """Small 3D habitat map with two labels."""
    labels = np.zeros((12, 14, 16), dtype=np.int32)
    labels[3:8, 3:10, 3:12] = 1
    labels[5:9, 6:12, 7:14] = 2
    return labels


@pytest.mark.unit
def test_plot_cluster_validation_curves_returns_figure() -> None:
    """Validation curves accept a selection_report-shaped payload."""
    import matplotlib.pyplot as plt

    fig = plot_cluster_validation_curves(
        {"silhouette": [0.1, 0.4, 0.3], "inertia": [9.0, 4.0, 3.5]},
        [2, 3, 4],
        selected=3,
    )
    assert fig.get_axes()
    plt.close(fig)


@pytest.mark.unit
def test_plot_cluster_validation_from_report() -> None:
    """Report helper forwards candidates/scores/selected."""
    import matplotlib.pyplot as plt

    report = {
        "candidates": [2, 3, 4],
        "methods": ["silhouette"],
        "scores": {"silhouette": [0.2, 0.5, 0.4]},
        "selected": 3,
        "directions": {"silhouette": "maximize"},
    }
    fig = plot_cluster_validation_from_report(report)
    plt.close(fig)


@pytest.mark.unit
def test_volume_msi_ith_figures(toy_labels: np.ndarray) -> None:
    """Volume / MSI / ITH plotters accept kernel outputs."""
    import matplotlib.pyplot as plt

    ids = (1, 2)
    frac = habitat_volume_fractions(toy_labels, ids)
    # n_classes includes background row/column 0.
    matrix = spatial_interaction_matrix(toy_labels, n_classes=3)
    score = ith_score(toy_labels)
    stats = habitat_region_stats(toy_labels)

    plt.close(plot_habitat_volume_fractions(frac))
    plt.close(plot_msi_matrix(matrix, habitat_ids=ids))
    plt.close(plot_msi_matrix(matrix, habitat_ids=ids, scale="normalized"))
    plt.close(plot_msi_matrix(matrix, habitat_ids=ids, scale="raw"))
    plt.close(plot_msi_matrix(matrix, habitat_ids=ids, scale="log1p"))
    plt.close(plot_msi_matrix(matrix, habitat_ids=ids, scale="linear"))
    plt.close(plot_ith_summary(score, per_habitat=stats))


@pytest.mark.unit
def test_plot_msi_matrix_rejects_unknown_scale(toy_labels: np.ndarray) -> None:
    """Unknown MSI display scales fail fast."""
    matrix = spatial_interaction_matrix(toy_labels, n_classes=3)
    with pytest.raises(Exception, match="scale"):
        plot_msi_matrix(matrix, scale="rainbow")


@pytest.mark.unit
def test_plot_msi_matrix_linear_scale_ignores_diagonal() -> None:
    """Default linear scale is set by off-diagonal cells, not the diagonal."""
    import matplotlib.pyplot as plt

    matrix = np.array(
        [
            [1.0e6, 10.0, 20.0],
            [10.0, 1.0e6, 30.0],
            [20.0, 30.0, 1.0e6],
        ],
        dtype=np.float64,
    )
    fig = plot_msi_matrix(matrix, habitat_ids=(1, 2))
    image = fig.axes[0].images[0]
    vmin, vmax = image.get_clim()
    off = np.array([10.0, 20.0, 10.0, 20.0, 30.0, 30.0])
    expected_vmin = float(np.percentile(off, 2.0))
    expected_vmax = float(np.percentile(off, 98.0))
    assert vmin == pytest.approx(expected_vmin)
    assert vmax == pytest.approx(expected_vmax)
    assert vmax < 1000.0
    plt.close(fig)


@pytest.mark.unit
def test_plot_msi_matrix_raw_keeps_diagonal_in_scale() -> None:
    """scale='raw' colours the full matrix, including the diagonal."""
    import matplotlib.pyplot as plt

    matrix = np.array(
        [
            [100.0, 10.0],
            [10.0, 100.0],
        ],
        dtype=np.float64,
    )
    fig = plot_msi_matrix(matrix, habitat_ids=(1,), scale="raw")
    image = fig.axes[0].images[0]
    _vmin, vmax = image.get_clim()
    # 2nd–98th of {100, 10, 10, 100} still includes the large diagonal.
    assert vmax > 50.0
    plt.close(fig)


@pytest.mark.unit
def test_label_compare_and_triptych(toy_labels: np.ndarray) -> None:
    """Compare and two-step triptych share shape checks."""
    import matplotlib.pyplot as plt

    image = toy_labels.astype(np.float32) * 10.0
    labels_b = toy_labels.copy()
    labels_b[toy_labels == 2] = 1
    # Dense unique supervoxel ids inside the ROI (product-like fragmentation).
    sv = np.zeros_like(toy_labels)
    roi = toy_labels > 0
    sv[roi] = np.arange(1, int(roi.sum()) + 1, dtype=np.int32)

    plt.close(plot_habitat_label_compare(image, toy_labels, labels_b, axis=0))
    plt.close(plot_partition_triptych(image, sv, toy_labels, axis=0))
