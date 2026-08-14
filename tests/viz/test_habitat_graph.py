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
"""Tests for habitat-graph figures in ``habit.viz.habitat_graph``."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from habit.kernels.habitat_graph import HabitatGraphFeatureOptions
from habit.viz import (
    plot_habitat_graph_network_2d,
    plot_habitat_graph_slice,
    render_habitat_graph_network_3d,
    render_habitat_graph_surface_3d,
    use_style,
)

pytestmark = pytest.mark.unit


def _synthetic_2d_labels() -> np.ndarray:
    """
    Build a readable 2D multi-habitat map with several disconnected regions.

    Returns:
        Integer label array with habitats 1--3 and background 0.
    """
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[4:12, 4:12] = 1
    labels[4:10, 18:26] = 1
    labels[16:26, 6:14] = 2
    labels[18:28, 18:28] = 2
    labels[12:16, 12:18] = 3
    return labels


def _synthetic_3d_labels() -> np.ndarray:
    """
    Build a compact 3D multi-habitat volume suitable for off-screen rendering.

    Returns:
        Integer label volume with habitats 1--3 and background 0.
    """
    labels = np.zeros((28, 28, 28), dtype=np.int32)
    labels[4:12, 4:12, 4:12] = 1
    labels[4:10, 16:24, 16:24] = 1
    labels[14:24, 6:14, 6:14] = 2
    labels[16:24, 16:24, 8:16] = 2
    labels[10:16, 10:18, 10:18] = 3
    return labels


def _viz_options() -> HabitatGraphFeatureOptions:
    """Return deterministic graph options that keep small synthetic nodes."""
    return HabitatGraphFeatureOptions(
        edge_method="centroid_distance",
        distance_threshold=12.0,
        erosion_radius=0,
        subdivide_region_voxels=0,
        include_extended_metrics=False,
    )


def test_plot_habitat_graph_slice_returns_figure_and_saves(tmp_path) -> None:
    """Slice overlay returns a matplotlib Figure and writes an inspectable PNG."""
    labels = _synthetic_2d_labels()
    with use_style("radiology"):
        fig = plot_habitat_graph_slice(labels)
    assert isinstance(fig, Figure)

    output_path = tmp_path / "habitat_graph_slice.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    image_axes = [ax for ax in fig.axes if ax.images]
    cbar_axes = [ax for ax in fig.axes if not ax.images]
    assert len(image_axes) == 1
    assert len(cbar_axes) == 1
    fig.canvas.draw()
    assert cbar_axes[0].get_ylabel() == "Habitat"
    titles = " ".join(ax.get_title() for ax in image_axes)
    assert titles.isascii()
    assert "Habitat" in titles

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_graph_network_2d_returns_figure_and_saves(tmp_path) -> None:
    """Network layout returns a multi-panel Figure with English titles."""
    labels = _synthetic_2d_labels()
    with use_style("nature"):
        fig = plot_habitat_graph_network_2d(labels, options=_viz_options())
    assert isinstance(fig, Figure)

    output_path = tmp_path / "habitat_graph_network_2d.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    image_axes = [ax for ax in fig.axes if ax.images]
    cbar_axes = [ax for ax in fig.axes if ax not in image_axes]
    assert image_axes
    assert cbar_axes
    for ax in image_axes:
        for image in ax.images:
            alpha = image.get_alpha()
            assert alpha is None or float(alpha) == pytest.approx(1.0)
        for line in ax.lines:
            assert float(line.get_alpha()) == pytest.approx(1.0)
        for collection in ax.collections:
            face_alpha = collection.get_alpha()
            if face_alpha is not None:
                assert float(face_alpha) == pytest.approx(1.0)
    fig.canvas.draw()
    cbar_labels = " ".join(ax.get_ylabel() for ax in cbar_axes)
    assert "Habitat" in cbar_labels
    panel_titles = [ax.get_title() for ax in image_axes]
    joined = " ".join(
        [fig._suptitle.get_text() if fig._suptitle is not None else ""]
        + panel_titles
    )
    assert joined.isascii()
    assert "graph" in joined.lower() or "Habitat" in joined
    # Per-panel titles drop the redundant "Intra-habitat graph" / "slice N"
    # (those live on the figure title) so neighbouring panels cannot collide.
    for title in panel_titles:
        assert "Intra-habitat graph" not in title
        assert "slice" not in title.lower()
    assert any(title.startswith("H") for title in panel_titles)
    assert "All habitats" in panel_titles

    import matplotlib.pyplot as plt

    plt.close(fig)


def _many_habitat_2d_labels(n_habitats: int = 8) -> np.ndarray:
    """
    Build a crowded 2D map so the network figure has many H1..Hn panels.

    Args:
        n_habitats: Number of distinct habitat labels to place.

    Returns:
        Integer label array with ``n_habitats`` blobs and background 0.
    """
    rows, cols = 4, 4
    cell = 12
    labels = np.zeros((rows * cell, cols * cell), dtype=np.int32)
    for habitat in range(1, n_habitats + 1):
        row = (habitat - 1) // cols
        col = (habitat - 1) % cols
        r0 = row * cell + 2
        c0 = col * cell + 2
        labels[r0 : r0 + 7, c0 : c0 + 7] = habitat
    return labels


def test_plot_habitat_graph_network_2d_titles_do_not_overlap() -> None:
    """Many-habitat grids keep short panel titles from colliding horizontally."""
    labels = _many_habitat_2d_labels(8)
    fig = plot_habitat_graph_network_2d(labels, options=_viz_options())
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    titled_axes = [ax for ax in fig.axes if ax.get_title()]
    assert len(titled_axes) >= 8
    bboxes = [ax.title.get_window_extent(renderer=renderer) for ax in titled_axes]
    for i, box_a in enumerate(bboxes):
        for j, box_b in enumerate(bboxes[i + 1 :], start=i + 1):
            assert not box_a.overlaps(box_b), (
                "subplot titles overlap: "
                f"{titled_axes[i].get_title()!r} vs "
                f"{titled_axes[j].get_title()!r}"
            )
    for ax in titled_axes:
        title = ax.get_title()
        assert title.isascii()
        assert "Intra-habitat graph" not in title
        assert "slice" not in title.lower()
    assert any(ax.get_title() == "All habitats" for ax in titled_axes)
    if fig._suptitle is not None:
        assert "slice" in fig._suptitle.get_text().lower()

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_graph_network_2d_returns_none_for_empty_map() -> None:
    """An all-background slice yields ``None`` (nothing to draw)."""
    empty = np.zeros((8, 8), dtype=np.int32)
    fig = plot_habitat_graph_network_2d(empty, options=_viz_options())
    assert fig is None


def test_3d_renderers_require_volume_not_2d() -> None:
    """3D APIs reject 2D arrays with a clear ValueError."""
    pytest.importorskip("pyvista")
    pytest.importorskip("skimage")
    labels_2d = _synthetic_2d_labels()
    with pytest.raises(ValueError, match="3D"):
        render_habitat_graph_surface_3d(labels_2d)
    with pytest.raises(ValueError, match="3D"):
        render_habitat_graph_network_3d(labels_2d, options=_viz_options())


def test_render_habitat_graph_surface_3d_returns_rgb_or_skips(tmp_path) -> None:
    """Surface renderer returns an RGB array when pyvista/skimage are present."""
    pytest.importorskip("pyvista")
    pytest.importorskip("skimage")

    labels = _synthetic_3d_labels()
    rgb = render_habitat_graph_surface_3d(
        labels,
        black_background=False,
        render_window=400,
        surface_smooth_iter=5,
    )
    assert rgb is not None
    assert isinstance(rgb, np.ndarray)
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert rgb.shape[0] == 400 and rgb.shape[1] == 400

    import matplotlib.pyplot as plt

    destination = tmp_path / "habitat_graph_surface_3d.png"
    plt.imsave(destination, rgb)
    assert destination.is_file() and destination.stat().st_size > 0


def test_render_habitat_graph_network_3d_returns_rgb_or_skips(tmp_path) -> None:
    """Network 3D renderer returns RGB and can be saved for inspection."""
    pytest.importorskip("pyvista")
    pytest.importorskip("skimage")

    labels = _synthetic_3d_labels()
    rgb = render_habitat_graph_network_3d(
        labels,
        options=_viz_options(),
        black_background=False,
        render_window=400,
    )
    assert rgb is not None
    assert isinstance(rgb, np.ndarray)
    assert rgb.ndim == 3 and rgb.shape[2] == 3

    import matplotlib.pyplot as plt

    destination = tmp_path / "habitat_graph_network_3d.png"
    plt.imsave(destination, rgb)
    assert destination.is_file() and destination.stat().st_size > 0
