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

import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from habit.kernels.habitat_graph import (
    HabitatGraphFeatureOptions,
    extract_graph_features,
)
from habit.viz import (
    plot_graph_feature_heatmap,
    plot_habitat_graph_network_2d,
    plot_habitat_graph_slice,
    render_habitat_graph_network_3d,
    render_habitat_graph_surface_3d,
    use_style,
)

pytestmark = pytest.mark.unit

#: H1 (n=..., e=...) — not H1-H2 pairwise titles.
_H_PANEL_TITLE = re.compile(r"^H\d+ \(n=")
#: H1-H2 (n=..., inter e=...)
_PAIR_PANEL_TITLE = re.compile(r"^H\d+-H\d+")


def _is_h_panel_title(title: str) -> bool:
    """Return True for a single-habitat panel title."""
    return bool(_H_PANEL_TITLE.match(title))


def _is_pair_panel_title(title: str) -> bool:
    """Return True for a pairwise inter-edge panel title."""
    return bool(_PAIR_PANEL_TITLE.match(title))


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
        node_method="component",
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
        title = ax.get_title()
        for image in ax.images:
            rgba = np.asarray(image.get_array())
            if rgba.ndim == 3 and rgba.shape[-1] == 4:
                alphas = rgba[..., 3]
                painted = alphas > 0
                if _is_h_panel_title(title) or _is_pair_panel_title(title):
                    assert painted.any()
                    assert np.allclose(alphas[painted], 1.0)
                    rgb = rgba[painted][..., :3]
                    chroma = rgb.max(axis=1) - rgb.min(axis=1)
                    # Featured habitat(s) are full colour; other foreground
                    # is a mid-dark gray wash (low chroma), not per-habitat
                    # gray and not a light wash that hides white strokes.
                    assert np.any(chroma > 0.15)
                    low_chroma = chroma <= 0.08
                    assert np.any(low_chroma)
                    gray_luma = rgb[low_chroma].mean(axis=1)
                    assert np.all(gray_luma < 0.65)
                    assert np.all(gray_luma > 0.25)
            else:
                alpha = image.get_alpha()
                assert alpha is None or float(alpha) == pytest.approx(1.0)
        for line in ax.lines:
            alpha = float(line.get_alpha())
            # Graph edges are opaque; the dashed lattice may be translucent.
            assert 0.0 < alpha <= 1.0 + 1e-9
        for collection in ax.collections:
            face_alpha = collection.get_alpha()
            if face_alpha is not None:
                assert float(face_alpha) == pytest.approx(1.0)
            if hasattr(collection, "get_sizes"):
                sizes = np.asarray(collection.get_sizes(), dtype=float)
                if sizes.size:
                    assert np.allclose(sizes, sizes[0])
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
    assert any(_is_h_panel_title(title) for title in panel_titles)
    pair_titles = [title for title in panel_titles if _is_pair_panel_title(title)]
    assert pair_titles
    for pair_title in pair_titles:
        assert "n=" in pair_title
        assert "inter e=" in pair_title
        assert "intra e=" not in pair_title
        assert "All habitats" not in pair_title

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_graph_network_2d_pair_panels_report_n_and_edges() -> None:
    """Pairwise titles list node and inter-edge counts; no All-habitats panel."""
    labels = _synthetic_2d_labels()
    fig = plot_habitat_graph_network_2d(labels, options=_viz_options())
    assert isinstance(fig, Figure)
    pair_axes = [ax for ax in fig.axes if _is_pair_panel_title(ax.get_title())]
    # Three habitats → three unordered pairs.
    assert len(pair_axes) == 3
    pair_heads = {ax.get_title().split(" (")[0] for ax in pair_axes}
    assert pair_heads == {"H1-H2", "H1-H3", "H2-H3"}
    for ax in pair_axes:
        title = ax.get_title()
        assert title.isascii()
        assert "n=" in title
        assert "inter e=" in title
        assert "intra e=" not in title
        assert "All habitats" not in title
    if fig.legends:
        legend_text = " ".join(t.get_text() for t in fig.legends[0].get_texts())
        assert "Other-habitat" not in legend_text
        assert "purple" not in legend_text.lower()
        assert "Node" in legend_text
        assert "Intra-habitat edge" in legend_text
        assert "Inter-habitat edge" in legend_text
    habitat_titles = [
        ax.get_title() for ax in fig.axes if _is_h_panel_title(ax.get_title())
    ]
    assert habitat_titles
    for habitat_title in habitat_titles:
        assert "n=" in habitat_title
        assert "e=" in habitat_title
        assert "intra e=" not in habitat_title
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_graph_network_2d_draws_dashed_uniform_grid() -> None:
    """Default uniform-grid nodes draw the same lattice as dashed lines."""
    labels = np.ones((16, 16), dtype=np.int32)
    fig = plot_habitat_graph_network_2d(labels)
    assert isinstance(fig, Figure)
    dashed = [
        line
        for ax in fig.axes
        for line in ax.lines
        if line.get_linestyle() == "--"
    ]
    assert dashed
    texts = [fig._suptitle.get_text() if fig._suptitle is not None else ""]
    texts.extend(ax.get_title() for ax in fig.axes)
    if fig.legends:
        texts.extend(t.get_text() for t in fig.legends[0].get_texts())
    joined = " ".join(texts)
    assert joined.isascii()
    assert "8-voxel cubes" in joined
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_block_size_8_vs_5_changes_n_nodes_and_delta_heatmap() -> None:
    """Same labels, only block_size 8 vs 5: node counts differ; delta plots."""
    labels_a = _synthetic_2d_labels()
    labels_b = _synthetic_2d_labels()
    labels_b[20:26, 4:10] = 3
    options_8 = HabitatGraphFeatureOptions(
        include_extended_metrics=False,
        edge_method="min_distance",
        distance_threshold=5.0,
        block_min_coverage=0.2,
        block_size=8,
    )
    options_5 = HabitatGraphFeatureOptions(
        include_extended_metrics=False,
        edge_method="min_distance",
        distance_threshold=5.0,
        block_min_coverage=0.2,
        block_size=5,
    )
    feats_a8 = extract_graph_features(labels_a, options=options_8)
    feats_a5 = extract_graph_features(labels_a, options=options_5)
    feats_b8 = extract_graph_features(labels_b, options=options_8)
    feats_b5 = extract_graph_features(labels_b, options=options_5)
    assert feats_a8["graph_num_nodes_total"] != feats_a5["graph_num_nodes_total"]
    assert feats_b8["graph_num_nodes_total"] != feats_b5["graph_num_nodes_total"]

    fig_8 = plot_habitat_graph_network_2d(
        labels_a, options=options_8, show_grid=True, block_size=8
    )
    fig_5 = plot_habitat_graph_network_2d(
        labels_a, options=options_5, show_grid=True, block_size=5
    )
    assert isinstance(fig_8, Figure)
    assert isinstance(fig_5, Figure)
    texts_8 = [fig_8._suptitle.get_text() if fig_8._suptitle is not None else ""]
    texts_5 = [fig_5._suptitle.get_text() if fig_5._suptitle is not None else ""]
    if fig_8.legends:
        texts_8.extend(item.get_text() for item in fig_8.legends[0].get_texts())
    if fig_5.legends:
        texts_5.extend(item.get_text() for item in fig_5.legends[0].get_texts())
    assert "8-voxel cubes" in " ".join(texts_8)
    assert "5-voxel cubes" in " ".join(texts_5)

    table_8 = pd.DataFrame(
        [
            {"subject_id": "subj001", **feats_a8},
            {"subject_id": "subj002", **feats_b8},
        ]
    )
    table_5 = pd.DataFrame(
        [
            {"subject_id": "subj001", **feats_a5},
            {"subject_id": "subj002", **feats_b5},
        ]
    )
    compare_features = tuple(
        name
        for name in table_8.columns
        if str(name).startswith("single_h")
    )[:8]
    fig_heat8 = plot_graph_feature_heatmap(
        table_8,
        subjects=("subj001", "subj002"),
        features=compare_features,
        zscore=True,
        title="Graph features: 8-voxel cubes (3D)",
    )
    fig_heat5 = plot_graph_feature_heatmap(
        table_5,
        subjects=("subj001", "subj002"),
        features=compare_features,
        zscore=True,
        title="Graph features: 5-voxel cubes (3D)",
    )
    fig_delta = plot_graph_feature_heatmap(
        table_5,
        reference=table_8,
        subjects=("subj001", "subj002"),
        features=compare_features,
        zscore=True,
        star_significant=True,
        title="Graph features: 5-voxel minus 8-voxel",
    )
    assert isinstance(fig_heat8, Figure)
    assert isinstance(fig_heat5, Figure)
    assert isinstance(fig_delta, Figure)
    assert "8-voxel cubes (3D)" in str(fig_heat8.axes[0].get_title())
    assert "5-voxel cubes (3D)" in str(fig_heat5.axes[0].get_title())
    assert "5-voxel minus 8-voxel" in str(fig_delta.axes[0].get_title())
    assert str(fig_delta.axes[0].get_title()).isascii()
    import matplotlib.pyplot as plt

    plt.close(fig_8)
    plt.close(fig_5)
    plt.close(fig_heat8)
    plt.close(fig_heat5)
    plt.close(fig_delta)


def test_graph_compare_extracts_full_volume_not_slice() -> None:
    """8-vs-5 compare tables must come from 3D labels, not one axial slice."""
    labels_3d = _synthetic_3d_labels()
    options = HabitatGraphFeatureOptions(
        include_extended_metrics=False,
        edge_method="min_distance",
        distance_threshold=5.0,
        block_min_coverage=0.2,
        block_size=8,
    )
    slice_index = int(
        (labels_3d > 0).reshape(labels_3d.shape[0], -1).sum(axis=1).argmax()
    )
    feats_slice = extract_graph_features(labels_3d[slice_index], options=options)
    feats_3d = extract_graph_features(labels_3d, options=options)
    assert feats_3d["graph_num_nodes_total"] != feats_slice["graph_num_nodes_total"]
    assert feats_3d["graph_num_nodes_total"] > feats_slice["graph_num_nodes_total"]


def test_plot_habitat_graph_network_2d_display_block_size_overrides() -> None:
    """Plot ``block_size`` / linestyle override extraction options for drawing."""
    labels = np.ones((16, 16), dtype=np.int32)
    fig = plot_habitat_graph_network_2d(
        labels,
        block_size=4,
        grid_linestyle=":",
    )
    assert isinstance(fig, Figure)
    dotted = [
        line
        for ax in fig.axes
        for line in ax.lines
        if line.get_linestyle() == ":"
    ]
    assert dotted
    texts = [fig._suptitle.get_text() if fig._suptitle is not None else ""]
    if fig.legends:
        texts.extend(t.get_text() for t in fig.legends[0].get_texts())
    assert "4-voxel cubes" in " ".join(texts)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_graph_network_2d_uses_unified_node_and_edge_sizes() -> None:
    """H panels and pairwise panels share node size and edge width."""
    labels = _synthetic_2d_labels()
    options = HabitatGraphFeatureOptions(
        edge_method="centroid_distance",
        distance_threshold=30.0,
        erosion_radius=0,
        node_method="component",
        subdivide_region_voxels=0,
        include_extended_metrics=False,
    )
    fig = plot_habitat_graph_network_2d(labels, options=options, show_grid=False)
    assert isinstance(fig, Figure)
    habitat_sizes: list[float] = []
    pair_sizes: list[float] = []
    habitat_widths: list[float] = []
    pair_widths: list[float] = []
    for ax in fig.axes:
        title = ax.get_title()
        sizes = [
            float(size)
            for collection in ax.collections
            if hasattr(collection, "get_sizes")
            for size in np.asarray(collection.get_sizes(), dtype=float)
        ]
        widths = [float(line.get_linewidth()) for line in ax.lines]
        if _is_pair_panel_title(title):
            pair_sizes.extend(sizes)
            pair_widths.extend(widths)
        elif _is_h_panel_title(title):
            habitat_sizes.extend(sizes)
            habitat_widths.extend(widths)
    assert habitat_sizes and pair_sizes
    assert np.allclose(habitat_sizes, habitat_sizes[0])
    assert np.allclose(pair_sizes, habitat_sizes[0])
    assert habitat_widths and pair_widths
    assert np.allclose(habitat_widths, habitat_widths[0])
    assert np.allclose(pair_widths, habitat_widths[0])
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_graph_network_2d_featured_panel_omits_other_edges() -> None:
    """H panels draw only that habitat's white intra-edges; pairs are white inter."""
    from matplotlib.colors import to_hex, to_rgb

    labels = _synthetic_2d_labels()
    options = HabitatGraphFeatureOptions(
        edge_method="centroid_distance",
        distance_threshold=30.0,
        erosion_radius=0,
        node_method="component",
        subdivide_region_voxels=0,
        include_extended_metrics=False,
    )
    fig = plot_habitat_graph_network_2d(labels, options=options, show_grid=False)
    assert isinstance(fig, Figure)
    white = np.asarray(to_rgb("#FFFFFF"))
    outline = np.asarray(to_rgb("#1A1A1A"))
    habitat_axes = [ax for ax in fig.axes if _is_h_panel_title(ax.get_title())]
    assert habitat_axes
    for ax in habitat_axes:
        title = ax.get_title()
        edge_count = int(title.split("e=")[-1].rstrip(")"))
        assert len(ax.lines) == edge_count
        for line in ax.lines:
            assert float(line.get_alpha()) == pytest.approx(1.0)
            hex_color = to_hex(line.get_color()).lower()
            assert hex_color == "#ffffff"
            assert hex_color not in {"#9aa0a6", "#8e44ad", "#c5c8cc"}
        for collection in ax.collections:
            faces = np.asarray(collection.get_facecolors(), dtype=float)
            if faces.size:
                assert np.allclose(faces[:, :3], white, atol=1e-6)
            edges = np.asarray(collection.get_edgecolors(), dtype=float)
            if edges.size:
                assert np.allclose(edges[:, :3], outline, atol=1e-6)
            if hasattr(collection, "get_linewidths"):
                lws = np.asarray(collection.get_linewidths(), dtype=float)
                if lws.size:
                    assert np.all(lws > 0.0)
    pair_axes = [ax for ax in fig.axes if _is_pair_panel_title(ax.get_title())]
    assert pair_axes
    for ax in pair_axes:
        title = ax.get_title()
        n_inter = int(title.split("inter e=")[-1].rstrip(")"))
        # One white stroke per inter-edge (not two-tone halves).
        assert len(ax.lines) == n_inter
        for line in ax.lines:
            hex_color = to_hex(line.get_color()).lower()
            assert hex_color == "#ffffff"
            assert hex_color != "#8e44ad"
        for collection in ax.collections:
            faces = np.asarray(collection.get_facecolors(), dtype=float)
            if faces.size:
                assert np.allclose(faces[:, :3], white, atol=1e-6)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_network_2d_layout_four_habitats_uses_2x3_pair_grid() -> None:
    """Four habitats: H1--H4 on one row, six pair panels in a 2x3 grid."""
    from habit.viz.habitat_graph import _network_2d_layout

    h_rows, h_cols, pair_rows, pair_cols = _network_2d_layout(4, 6, max_cols=4)
    assert (h_rows, h_cols) == (1, 4)
    assert (pair_rows, pair_cols) == (2, 3)


def test_plot_habitat_graph_network_2d_shared_window_and_panel_size() -> None:
    """Every H and pair panel shares one ROI window and one physical size."""
    labels = _many_habitat_2d_labels(4)
    fig = plot_habitat_graph_network_2d(labels, options=_viz_options())
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    panel_axes = [
        ax
        for ax in fig.axes
        if _is_h_panel_title(ax.get_title()) or _is_pair_panel_title(ax.get_title())
    ]
    assert len(panel_axes) == 10
    xlims = [ax.get_xlim() for ax in panel_axes]
    ylims = [ax.get_ylim() for ax in panel_axes]
    assert all(np.allclose(xlim, xlims[0]) for xlim in xlims)
    assert all(np.allclose(ylim, ylims[0]) for ylim in ylims)
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_window_extent(renderer=renderer) for ax in panel_axes]
    widths = np.asarray([box.width for box in boxes], dtype=float)
    heights = np.asarray([box.height for box in boxes], dtype=float)
    assert np.allclose(widths, widths[0], rtol=0.08, atol=2.0)
    assert np.allclose(heights, heights[0], rtol=0.08, atol=2.0)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_graph_network_2d_four_habitats_six_pair_panels() -> None:
    """Four habitats draw six pairwise inter-edge panels and no All-habitats hairball."""
    labels = _many_habitat_2d_labels(4)
    fig = plot_habitat_graph_network_2d(labels, options=_viz_options())
    assert isinstance(fig, Figure)
    h_titles = [ax.get_title() for ax in fig.axes if _is_h_panel_title(ax.get_title())]
    pair_titles = [
        ax.get_title() for ax in fig.axes if _is_pair_panel_title(ax.get_title())
    ]
    assert len(h_titles) == 4
    assert len(pair_titles) == 6
    assert {title.split(" (")[0] for title in pair_titles} == {
        "H1-H2",
        "H1-H3",
        "H1-H4",
        "H2-H3",
        "H2-H4",
        "H3-H4",
    }
    assert not any("All habitats" in ax.get_title() for ax in fig.axes)
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
    assert any(_is_pair_panel_title(ax.get_title()) for ax in titled_axes)
    assert not any("All habitats" in ax.get_title() for ax in titled_axes)
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
