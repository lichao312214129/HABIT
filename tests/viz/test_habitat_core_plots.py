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
    habitat_ith_dispersion,
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
    dispersion = habitat_ith_dispersion(toy_labels)

    plt.close(plot_habitat_volume_fractions(frac))
    plt.close(plot_msi_matrix(matrix, habitat_ids=ids))
    plt.close(plot_msi_matrix(matrix, habitat_ids=ids, scale="normalized"))
    plt.close(plot_msi_matrix(matrix, habitat_ids=ids, scale="raw"))
    plt.close(plot_msi_matrix(matrix, habitat_ids=ids, scale="log1p"))
    plt.close(plot_msi_matrix(matrix, habitat_ids=ids, scale="linear"))
    plt.close(plot_ith_summary(score, dispersion=dispersion))


@pytest.mark.unit
def test_plot_ith_summary_one_panel_ith_then_habitat_bars() -> None:
    """Single axes: ITH bar first, then n habitat bars, ylabel ITH."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    fig = plot_ith_summary(0.42, dispersion={1: 0.94, 2: 0.91, 3: 0.98})
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_ylabel() == "ITH"
    bars = ax.patches
    assert len(bars) == 4
    assert all(abs(bar.get_width() - 0.55) < 1e-6 for bar in bars)
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert labels[0] == "ITH"
    assert labels[1:] == ["H1", "H2", "H3"]
    assert bars[0].get_height() == pytest.approx(0.42)
    assert to_hex(bars[0].get_facecolor()).lower() == "#cc79a7"
    assert all(to_hex(bar.get_facecolor()).lower() == "#009e73" for bar in bars[1:])
    centers = [bar.get_x() + bar.get_width() / 2.0 for bar in bars]
    assert (centers[1] - centers[0]) > (centers[2] - centers[1])
    y0, y1 = ax.get_ylim()
    assert y0 == pytest.approx(0.0)
    assert y1 == pytest.approx(1.0)
    plt.close(fig)


@pytest.mark.unit
def test_plot_ith_summary_without_dispersion_is_one_ith_bar() -> None:
    """No dispersion: still one panel with a single ITH category."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    fig = plot_ith_summary(0.42)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_ylabel() == "ITH"
    assert len(ax.patches) == 1
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["ITH"]
    assert ax.patches[0].get_height() == pytest.approx(0.42)
    assert to_hex(ax.patches[0].get_facecolor()).lower() == "#cc79a7"
    y0, y1 = ax.get_ylim()
    assert y0 == pytest.approx(0.0)
    assert y1 == pytest.approx(1.0)
    plt.close(fig)


@pytest.mark.unit
def test_plot_ith_summary_rejects_old_region_count_mapping() -> None:
    """The old id → (n_regions, largest) mapping must not silently plot."""
    with pytest.raises(Exception, match="habitat_ith_dispersion"):
        plot_ith_summary(0.42, per_habitat={1: (3, 40), 2: (2, 25)})


@pytest.mark.unit
def test_plot_ith_summary_per_habitat_alias_warns() -> None:
    """``per_habitat`` still accepts id → float through v1.x, with a warning."""
    import matplotlib.pyplot as plt

    with pytest.warns(DeprecationWarning, match="dispersion="):
        fig = plot_ith_summary(0.42, per_habitat={1: 0.8, 2: 0.6})
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_ylabel() == "ITH"
    assert len(ax.patches) == 3
    assert [tick.get_text() for tick in ax.get_xticklabels()][0] == "ITH"
    plt.close(fig)


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

    compare = plot_habitat_label_compare(
        image, toy_labels, labels_b, axis=0, align_labels=False
    )
    compare.canvas.draw()
    compare_cbars = [ax for ax in compare.axes if not ax.images]
    assert compare_cbars
    assert any(ax.get_ylabel() == "Habitat" for ax in compare_cbars)
    plt.close(compare)

    triptych = plot_partition_triptych(image, sv, toy_labels, axis=0)
    triptych.canvas.draw()
    triptych_cbars = [ax for ax in triptych.axes if not ax.images]
    assert triptych_cbars
    assert any(ax.get_ylabel() == "Habitat" for ax in triptych_cbars)
    plt.close(triptych)


@pytest.mark.unit
def test_label_compare_aligns_permuted_ids_by_default() -> None:
    """Independent maps with swapped ids agree after default centroid align."""
    import matplotlib.pyplot as plt

    labels_a = np.zeros((8, 8, 8), dtype=np.int32)
    labels_a[0:4, 0:4, 0:4] = 1
    labels_a[4:8, 0:4, 0:4] = 2
    labels_b = np.zeros_like(labels_a)
    labels_b[0:4, 0:4, 0:4] = 2
    labels_b[4:8, 0:4, 0:4] = 1
    image = np.zeros((8, 8, 8), dtype=np.float64)
    image[labels_a == 1] = 1.0
    image[labels_a == 2] = 10.0
    fig = plot_habitat_label_compare(image, labels_a, labels_b, axis=0)
    fig.canvas.draw()
    plt.close(fig)
    from habit.kernels.habitat_label_match import align_label_array

    aligned = align_label_array(labels_a, labels_b, image=image, method="centroid")
    disagree = (aligned != labels_a) & ((aligned > 0) | (labels_a > 0))
    assert int(np.count_nonzero(disagree)) == 0


@pytest.mark.unit
def test_label_compare_skips_align_when_model_ids_match() -> None:
    """Shared model_id (apply-saved-model) is a no-op even if ids are swapped."""
    from habit.contracts import Geometry, HabitatMap, Provenance

    labels_a = np.zeros((6, 6, 6), dtype=np.int32)
    labels_a[0:3, 0:3, 0:3] = 1
    labels_a[3:6, 0:3, 0:3] = 2
    labels_b = np.zeros_like(labels_a)
    labels_b[0:3, 0:3, 0:3] = 2
    labels_b[3:6, 0:3, 0:3] = 1
    geometry = Geometry.from_array((6, 6, 6))
    provenance = Provenance.source("viz_test")
    map_a = HabitatMap(
        subject_id="P1",
        label_array=labels_a,
        geometry=geometry,
        model_id="shared-model",
        habitat_ids=(1, 2),
        provenance=provenance,
    )
    map_b = HabitatMap(
        subject_id="P1",
        label_array=labels_b,
        geometry=geometry,
        model_id="shared-model",
        habitat_ids=(1, 2),
        provenance=provenance,
    )
    image = np.zeros((6, 6, 6), dtype=np.float64)
    image[labels_a == 1] = 1.0
    image[labels_a == 2] = 10.0
    import matplotlib.pyplot as plt

    fig = plot_habitat_label_compare(image, map_a, map_b, axis=0)
    fig.canvas.draw()
    plt.close(fig)
    # Auto-skip leaves the permutation in place: disagreement is the full ROI.
    disagree = (labels_a != labels_b) & ((labels_a > 0) | (labels_b > 0))
    assert int(np.count_nonzero(disagree)) == int(np.count_nonzero(labels_a > 0))
