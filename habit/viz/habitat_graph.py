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
"""Habitat graph (topology) figures.

Pure functions following the ``habit.viz`` contract: label arrays in, a
matplotlib ``Figure`` (2D) or an RGB render array (3D, PyVista) out, no
filesystem and no ``show``. Where a figure ends up is the caller's decision.

The graph nodes and edges use the SAME construction algorithms as the numeric
feature extractor (:func:`habit.kernels.habitat_graph.extract_habitat_nodes`
with erosion / subdivision / connectivity, and
:func:`~habit.kernels.habitat_graph.build_centroid_distance_graph` /
:func:`~habit.kernels.habitat_graph.build_min_distance_graph` /
:func:`~habit.kernels.habitat_graph.build_adjacency_graph` with the configured
parameters), so the drawn graph matches the measured features. The 2D network
figure is built from the representative cross-section so the overlay matches
the slice figure; describe this slice-local 2D graph separately in
publications when 3D volumetric features are also reported.

:func:`plot_graph_feature_heatmap` is a different figure: rows are
subjects and columns are graph-feature values (``single_h*`` /
``pair_h*``), not habitats x texture features. Cap the column count
(default 40) and let the caller choose people and features; do not dump
the full ~400-column bank onto one axes.

All text drawn on the figures is English-only.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from habit.kernels.habitat_graph import (
    HabitatGraphFeatureOptions,
    HabitatGraphNode,
    HabitatNodeExtractionResult,
    build_adjacency_graph,
    build_centroid_distance_graph,
    build_min_distance_graph,
    extract_habitat_nodes,
    iter_label_pairs,
)
from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz.colorbar import (
    ColorbarSpec,
    DEFAULT_HABITAT_CBAR_LABEL,
    add_discrete_habitat_colorbar,
    colorbar_is_enabled,
    discrete_habitat_mappable,
)
from habit.viz.labels import sanitize_label
from habit.viz.palette import habitat_hex_colors
from habit.viz.style import use_style

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

__all__ = [
    "plot_habitat_graph_slice",
    "plot_habitat_graph_network_2d",
    "plot_graph_feature_heatmap",
    "render_habitat_graph_surface_3d",
    "render_habitat_graph_network_3d",
]

#: What habit.viz needs matplotlib for.
_VIZ_PURPOSE = "habitat graph topology figures"
#: What the 3D renderers need PyVista / scikit-image for.
_VIEW_PURPOSE = "3D habitat graph rendering"
#: Default column cap so a ~400-feature graph table stays readable.
_DEFAULT_HEATMAP_FEATURES: int = 40
#: Single-habitat columns look like ``single_h1_avg_degree``.
_SINGLE_FEATURE_RE = re.compile(r"^single_h\d+_")
#: Pairwise columns look like ``pair_h1_h2_edge_density``.
_PAIR_FEATURE_RE = re.compile(r"^pair_h\d+_h\d+_")
#: Subject-level counts (``graph_num_habitats``, ``graph_num_nodes_total``).
_GRAPH_NUM_RE = re.compile(r"^graph_num_")
#: Heatmap typography (GitHub Pages / gallery readability).
_HEATMAP_TITLE_FONTSIZE: float = 11.0
_HEATMAP_LABEL_FONTSIZE: float = 10.0
_HEATMAP_TICK_FONTSIZE: float = 8.0
_HEATMAP_CBAR_FONTSIZE: float = 9.0

#: 3D renderer only: inter-habitat tubes stay a single accent. The 2D
#: graph overlay is white (nodes + edges) on the coloured habitat fill.
_INTER_EDGE_COLOR = "#8E44AD"
#: 3D renderer only: intra-habitat tubes. 2D intra-edges are white.
_INTRA_EDGE_COLOR = "#9AA0A6"
_BACKGROUND_COLOR = "#D9DCE1"
#: 2D graph overlay: solid white nodes and white edges on the fill.
_GRAPH_NODE_COLOR = "#FFFFFF"
_GRAPH_EDGE_COLOR = "#FFFFFF"
#: Thin dark rim so a white dot stays visible on pink / light habitats.
_NODE_OUTLINE_COLOR = "#1A1A1A"


#: Type alias for an undirected edge expressed as a pair of node ids.
_EdgePair = Tuple[str, str]


def _plt():
    """
    Return the pyplot module with the Agg canvas guaranteed headless.

    Returns:
        The ``matplotlib.pyplot`` module, with a non-interactive backend
        already active.

    Raises:
        OptionalDependencyError: When matplotlib is not installed.
    """
    matplotlib = require("matplotlib", extra="viz", purpose=_VIZ_PURPOSE)
    if matplotlib.get_backend().lower() not in (
        "agg",
        "module://matplotlib_inline.backend_inline",
    ):
        matplotlib.use("Agg")
    return require("matplotlib.pyplot", extra="viz", purpose=_VIZ_PURPOSE)


# ---------------------------------------------------------------------------
# Array preparation helpers
# ---------------------------------------------------------------------------


def _as_label_array(label_array: np.ndarray) -> np.ndarray:
    """
    Coerce input to a 2D or 3D integer label array.

    Args:
        label_array: Habitat label map (background encoded as 0).

    Returns:
        np.ndarray: Integer label array with ndim in ``{2, 3}``.

    Raises:
        ValueError: When the array is not 2D or 3D.
    """
    array = np.asarray(label_array)
    if array.ndim not in (2, 3):
        raise ValueError(
            f"label_array must be 2D or 3D; got shape {tuple(array.shape)}."
        )
    return array.astype(np.int32, copy=False)


def _crop_to_foreground(label_array: np.ndarray, pad: int = 3) -> np.ndarray:
    """Crop an array to the non-background bounding box with padding."""
    mask = label_array > 0
    if not mask.any():
        return label_array
    slices = []
    for axis, axis_size in enumerate(label_array.shape):
        axis_mask = mask.any(axis=tuple(i for i in range(label_array.ndim) if i != axis))
        idx = np.where(axis_mask)[0]
        start = max(0, int(idx[0]) - pad)
        stop = min(axis_size, int(idx[-1]) + 1 + pad)
        slices.append(slice(start, stop))
    return label_array[tuple(slices)]


def _largest_slice_index(label_array: np.ndarray) -> int:
    """Return the axis-0 slice index with the most non-background voxels."""
    counts = (label_array > 0).reshape(label_array.shape[0], -1).sum(axis=1)
    if not counts.any():
        return label_array.shape[0] // 2
    return int(np.argmax(counts))


def _representative_slice(
    label_array: np.ndarray,
    slice_index: Optional[int],
) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
    """
    Return the 2D slice to draw, its index, and the cropped 3D volume.

    Args:
        label_array: 2D or 3D habitat label map.
        slice_index: Explicit axis-0 slice for 3D input; ``None`` selects the
            largest cross-section. Ignored for 2D input.

    Returns:
        Tuple: ``(label_2d, slice_index, cropped_3d_or_None)``.
    """
    if label_array.ndim == 3:
        cropped = _crop_to_foreground(label_array)
        index = _largest_slice_index(cropped) if slice_index is None else int(slice_index)
        return cropped[index], index, cropped
    return _crop_to_foreground(label_array), 0, None


# ---------------------------------------------------------------------------
# Feature-aligned graph construction (mirrors the numeric extractor)
# ---------------------------------------------------------------------------


def _extract_nodes(
    label_array: np.ndarray, options: HabitatGraphFeatureOptions
) -> HabitatNodeExtractionResult:
    """Extract graph nodes using the exact feature-extractor options."""
    return extract_habitat_nodes(
        label_array=label_array,
        connectivity=options.connectivity,
        min_region_voxels=options.min_region_voxels,
        erosion_radius=options.erosion_radius,
        subdivide_region_voxels=options.subdivide_region_voxels,
        block_size=options.block_size,
        block_min_coverage=options.block_min_coverage,
        node_method=options.node_method,
    )


def _single_intra_edges(
    nodes: Sequence[HabitatGraphNode],
    label: int,
    options: HabitatGraphFeatureOptions,
    node_result: Optional[HabitatNodeExtractionResult] = None,
) -> List[_EdgePair]:
    """Build within-habitat edges for one habitat (same as the feature graph)."""
    if options.edge_method == "adjacency" and node_result is not None:
        graph = build_adjacency_graph(
            node_result=node_result,
            labels=(label,),
            graph_kind="single",
            adjacency_connectivity=options.adjacency_connectivity,
            adjacency_min_voxels=options.adjacency_min_voxels,
            edge_weight=options.edge_weight,
        )
    elif options.edge_method == "min_distance" and node_result is not None:
        graph = build_min_distance_graph(
            node_result=node_result,
            labels=(label,),
            graph_kind="single",
            distance_threshold=options.distance_threshold,
            edge_weight=options.edge_weight,
        )
    else:
        graph = build_centroid_distance_graph(
            nodes=nodes,
            labels=(label,),
            graph_kind="single",
            distance_threshold=options.distance_threshold,
            edge_weight=options.edge_weight,
        )
    return [(edge.source, edge.target) for edge in graph.edges]


def _pair_inter_edges(
    node_result: HabitatNodeExtractionResult,
    label_a: int,
    label_b: int,
    options: HabitatGraphFeatureOptions,
) -> List[_EdgePair]:
    """Build between-habitat edges for one pair (same as the feature graph)."""
    if options.edge_method == "adjacency":
        graph = build_adjacency_graph(
            node_result=node_result,
            labels=(label_a, label_b),
            graph_kind="pairwise",
            adjacency_connectivity=options.adjacency_connectivity,
            adjacency_min_voxels=options.adjacency_min_voxels,
            edge_weight=options.edge_weight,
            include_intra_edges=False,
        )
    elif options.edge_method == "min_distance":
        graph = build_min_distance_graph(
            node_result=node_result,
            labels=(label_a, label_b),
            graph_kind="pairwise",
            distance_threshold=options.distance_threshold,
            edge_weight=options.edge_weight,
            include_intra_edges=False,
        )
    else:
        pair_nodes = list(node_result.nodes_by_habitat.get(label_a, [])) + list(
            node_result.nodes_by_habitat.get(label_b, [])
        )
        graph = build_centroid_distance_graph(
            nodes=pair_nodes,
            labels=(label_a, label_b),
            graph_kind="pairwise",
            distance_threshold=options.distance_threshold,
            edge_weight=options.edge_weight,
            include_intra_edges=False,
        )
    return [
        (edge.source, edge.target)
        for edge in graph.edges
        if edge.edge_type != "intra"
    ]


def _combined_graph(
    node_result: HabitatNodeExtractionResult,
    options: HabitatGraphFeatureOptions,
) -> Tuple[Dict[str, HabitatGraphNode], List[_EdgePair], List[_EdgePair]]:
    """
    Assemble the feature-aligned combined graph.

    Returns:
        Tuple: (node lookup by id, intra edges, inter edges). Intra edges are
        the union of every habitat's single-habitat graph; inter edges are the
        union of every habitat pair's between-habitat edges.
    """
    labels = sorted(node_result.nodes_by_habitat.keys())
    id_to_node: Dict[str, HabitatGraphNode] = {}
    for label in labels:
        for node in node_result.nodes_by_habitat[label]:
            id_to_node[node.node_id] = node

    intra_edges: List[_EdgePair] = []
    if options.include_single_habitat_graph:
        for label in labels:
            intra_edges.extend(
                _single_intra_edges(
                    node_result.nodes_by_habitat[label], label, options, node_result
                )
            )
    inter_edges: List[_EdgePair] = []
    if options.include_pairwise_habitat_graph:
        for label_a, label_b in iter_label_pairs(labels):
            inter_edges.extend(_pair_inter_edges(node_result, label_a, label_b, options))
    return id_to_node, intra_edges, inter_edges


def _pair_panel_title(
    label_a: int,
    label_b: int,
    n_nodes: int,
    n_inter_edges: int,
) -> str:
    """
    Build the English pairwise-panel title.

    Args:
        label_a: First habitat id of the unordered pair.
        label_b: Second habitat id of the unordered pair.
        n_nodes: Nodes drawn on this panel (both habitats).
        n_inter_edges: Inter-habitat edges drawn on this panel.

    Returns:
        Title such as ``H1-H2 (n=26, inter e=12)``. Figure text stays
        English-only.
    """
    return (
        f"H{int(label_a)}-H{int(label_b)} "
        f"(n={int(n_nodes)}, inter e={int(n_inter_edges)})"
    )


def _network_2d_layout(
    n_habitats: int,
    n_pairs: int,
    max_cols: int,
) -> Tuple[int, int, int, int]:
    """
    Compute the habitat-row and pairwise-panel grid shape.

    Habitat panels use up to ``max_cols`` so H1--H4 sit on the first row
    when ``K <= 4``. Six pairwise panels (four habitats) use a 2x3
    placement inside an equal-cell parent grid so every panel is the
    same physical size.

    Args:
        n_habitats: Number of single-habitat (H) panels.
        n_pairs: Number of unordered habitat-pair panels.
        max_cols: Maximum H-panel columns (also used for large pair grids).

    Returns:
        Tuple[int, int, int, int]:
        ``(h_rows, h_cols, pair_rows, pair_cols)``.
    """
    n_habitats = int(n_habitats)
    n_pairs = int(n_pairs)
    max_cols = max(1, int(max_cols))
    h_cols = min(max_cols, max(1, n_habitats))
    h_rows = int(np.ceil(n_habitats / h_cols)) if n_habitats else 0
    if n_pairs <= 0:
        return h_rows, h_cols, 0, 1
    if n_pairs == 6:
        pair_cols = 3
    elif n_pairs <= 3:
        pair_cols = n_pairs
    else:
        pair_cols = min(max_cols, n_pairs)
    pair_rows = int(np.ceil(n_pairs / pair_cols))
    return h_rows, h_cols, pair_rows, pair_cols


# ---------------------------------------------------------------------------
# Color and styling helpers
# ---------------------------------------------------------------------------


def _habitat_colors(labels: Sequence[int]) -> Dict[int, str]:
    """Map habitat labels to stable Radiology-safe colours (no 8-colour wrap)."""
    ordered = sorted({int(value) for value in labels if int(value) > 0})
    if not ordered:
        return {}
    hexes = habitat_hex_colors(len(ordered))
    return {label: hexes[index] for index, label in enumerate(ordered)}


#: Matplotlib scatter area (points^2). H1--Hk and pairwise panels share
#: this small solid-dot size.
_DEFAULT_NODE_SIZE: float = 28.0
#: Shared graph-edge stroke (points) on every 2D panel (H intra-edges
#: and pairwise inter-edges use the same width).
_DEFAULT_EDGE_WIDTH: float = 0.60
#: Thin dark outline (points) around each solid white node.
_NODE_EDGE_WIDTH: float = 0.55
#: Publication-readable type sizes for the multi-panel 2D network figure.
_PANEL_TITLE_FONTSIZE: float = 11.5
_AXIS_TEXT_FONTSIZE: float = 10.5
_FIG_LEGEND_FONTSIZE: float = 10.5
#: Mid-dark gray ROI silhouette behind a featured-habitat panel so
#: white nodes / edges stay visible when they cross other habitats.
#: One wash for the whole non-featured foreground — not per-habitat
#: gray fills, and not a light wash that swallows white strokes.
_ROI_BACKDROP_COLOR: str = "#7A7E84"
#: Dashed lattice overlay (same origin / block_size as node extraction).
_GRID_LINE_COLOR: str = "#6B7280"
_GRID_LINE_ALPHA: float = 0.45
_GRID_LINE_WIDTH: float = 0.55
#: Default matplotlib linestyle for the display lattice.
_DEFAULT_GRID_LINESTYLE: str = "--"
#: Shared pad (voxels) around the union-of-habitats bbox so every 2D
#: graph panel uses the same spatial window.
_SHARED_WINDOW_PAD: float = 3.0


def _centroid_xy_display(node: HabitatGraphNode) -> Tuple[float, float]:
    """
    Return (x, y) image coordinates for 2D network plotting.

    For 2D label maps the centroid is ``(row, col)``; for 3D maps it is
    ``(z, row, col)`` and the z axis is dropped so the graph is projected
    onto the representative cross-section plane.
    """
    centroid = node.centroid
    if centroid.shape[0] == 2:
        return float(centroid[1]), float(centroid[0])
    return float(centroid[2]), float(centroid[1])


def _phys_xyz(
    node: HabitatGraphNode, spacing: Tuple[float, float, float]
) -> np.ndarray:
    """Convert a (z, y, x) centroid to physical (x, y, z) coordinates."""
    sz, sy, sx = spacing
    if node.centroid.shape[0] == 2:
        cy, cx = float(node.centroid[0]), float(node.centroid[1])
        cz = 0.0
    else:
        cz, cy, cx = (
            float(node.centroid[0]),
            float(node.centroid[1]),
            float(node.centroid[2]),
        )
    return np.array([cx * sx, cy * sy, cz * sz], dtype=float)


def _draw_background_2d(
    ax,
    label_2d: np.ndarray,
    colors: Optional[Dict[int, str]],
    show_background: bool,
    featured_labels: Optional[Sequence[int]] = None,
) -> None:
    """
    Draw spatial context behind 2D network graphs.

    When ``colors`` is provided, each habitat partition is painted with the
    same palette as the slice figure. Background ``0`` is left transparent
    (not drawn). Pass ``featured_labels`` so those habitats stay full
    colour and every other foreground voxel is a single mid-dark gray
    wash (``_ROI_BACKDROP_COLOR``) — not per-habitat gray. H panels pass
    one id; pairwise panels pass the two ids of that pair.

    Args:
        ax: Target matplotlib axes.
        label_2d: 2D habitat label map (background encoded as 0).
        colors: Habitat label to hex colour mapping, or ``None`` for a
            gray tissue silhouette.
        show_background: When ``False``, nothing is drawn.
        featured_labels: Habitat ids to keep in full colour. ``None``
            paints every habitat at alpha 1.
    """
    if not show_background:
        return
    from matplotlib.colors import ListedColormap, to_rgba

    if colors:
        ordered = sorted(colors)
        if not ordered:
            return
        rgba = np.zeros((*label_2d.shape, 4), dtype=float)
        if featured_labels is None:
            for label in ordered:
                mask = label_2d == label
                if np.any(mask):
                    rgba[mask] = to_rgba(colors[label], alpha=1.0)
        else:
            featured = {int(value) for value in featured_labels}
            roi = label_2d > 0
            rgba[roi] = to_rgba(_ROI_BACKDROP_COLOR, alpha=1.0)
            for label in featured:
                featured_mask = label_2d == label
                if label in colors and np.any(featured_mask):
                    rgba[featured_mask] = to_rgba(colors[label], alpha=1.0)
        ax.imshow(rgba, interpolation="nearest", zorder=0)
        return
    silhouette = np.where(label_2d > 0, 1.0, np.nan)
    ax.imshow(
        silhouette,
        cmap=ListedColormap([_BACKGROUND_COLOR]),
        interpolation="nearest",
        alpha=1.0,
        zorder=0,
    )


def _display_block_size(
    options: HabitatGraphFeatureOptions,
    block_size: Optional[int],
) -> int:
    """
    Resolve the cube edge length used only for drawing the lattice.

    Args:
        options: Extraction options (nodes / edges still use these).
        block_size: Plot-function override. ``None`` uses
            ``options.block_size`` (library default 8 voxels).

    Returns:
        int: Cube edge length in voxels, ``>= 1``.

    Raises:
        ValueError: When the resolved size is less than 1.
    """
    size = int(options.block_size) if block_size is None else int(block_size)
    if size < 1:
        raise ValueError("block_size must be >= 1.")
    return size


def _display_grid_origin(
    label_2d: np.ndarray,
    node_result: HabitatNodeExtractionResult,
    display_size: int,
) -> Optional[Tuple[int, ...]]:
    """
    Return the lattice origin for the drawn cubes.

    When the display size matches the extraction lattice, reuse
    ``node_result.grid_origin`` so the overlay sits on the same cubes
    that became graph nodes. Otherwise recompute the VOI-min origin
    (same rule as :func:`~habit.kernels.habitat_graph.extract_habitat_nodes`).

    Args:
        label_2d: Slice used to recover a VOI origin if needed.
        node_result: Node extraction result (may carry ``grid_origin``).
        display_size: Cube edge length that will be drawn.

    Returns:
        Inclusive voxel-index origin, or ``None`` when the slice is empty.
    """
    if (
        node_result.grid_origin is not None
        and node_result.grid_block_size == int(display_size)
    ):
        return tuple(int(v) for v in node_result.grid_origin)
    coords = np.argwhere(np.asarray(label_2d) > 0)
    if coords.size == 0:
        return None
    return tuple(int(v) for v in coords.min(axis=0))


def _grid_caption(block_size: int) -> str:
    """
    English label that states the displayed cube size.

    Args:
        block_size: Cube edge length in voxels.

    Returns:
        Caption such as ``\"8-voxel cubes\"``.
    """
    return f"{int(block_size)}-voxel cubes"


def _should_draw_grid(
    options: HabitatGraphFeatureOptions,
    show_grid: bool,
    block_size: Optional[int],
) -> bool:
    """
    Whether the display lattice should be drawn.

    Default: on for ``uniform_grid``. An explicit ``block_size`` overlay
    is also drawn in ``component`` mode so the caller can show cubes
    without rebuilding extraction options.

    Args:
        options: Extraction options (``node_method``).
        show_grid: Caller on/off switch.
        block_size: Plot-function override, or ``None``.

    Returns:
        bool: ``True`` when dashed (or styled) lattice lines should appear.
    """
    if not show_grid:
        return False
    return options.node_method == "uniform_grid" or block_size is not None


def _draw_grid_2d(
    ax,
    label_2d: np.ndarray,
    grid_origin: Optional[Tuple[int, ...]],
    block_size: Optional[int],
    *,
    linestyle: str = _DEFAULT_GRID_LINESTYLE,
    color: str = _GRID_LINE_COLOR,
    alpha: float = _GRID_LINE_ALPHA,
    linewidth: float = _GRID_LINE_WIDTH,
) -> None:
    """
    Draw the node-extraction lattice on a 2D habitat panel.

    The origin and ``block_size`` should match
    :class:`~habit.kernels.habitat_graph.HabitatNodeExtractionResult` when
    the caller has not overridden the display size, so the overlay matches
    the cubes that became graph nodes. Coordinates are image indices:
    x = column, y = row.

    Args:
        ax: Target matplotlib axes.
        label_2d: Slice used only for axis extent.
        grid_origin: Inclusive voxel-index origin of the lattice.
        block_size: Cube edge length in voxels.
        linestyle: Matplotlib line style (default dashed ``\"--\"``).
        color: Lattice colour.
        alpha: Lattice opacity.
        linewidth: Lattice stroke width in points.
    """
    if grid_origin is None or block_size is None or int(block_size) < 1:
        return
    origin = tuple(int(v) for v in grid_origin)
    if len(origin) == 2:
        row0, col0 = origin
    elif len(origin) >= 3:
        row0, col0 = origin[-2], origin[-1]
    else:
        return
    n_rows, n_cols = int(label_2d.shape[0]), int(label_2d.shape[1])
    step = int(block_size)
    style = str(linestyle) if linestyle else _DEFAULT_GRID_LINESTYLE
    x = col0
    while x <= n_cols:
        if x >= 0:
            ax.axvline(
                x - 0.5,
                color=color,
                linestyle=style,
                linewidth=float(linewidth),
                alpha=float(alpha),
                zorder=1,
            )
        x += step
    y = row0
    while y <= n_rows:
        if y >= 0:
            ax.axhline(
                y - 0.5,
                color=color,
                linestyle=style,
                linewidth=float(linewidth),
                alpha=float(alpha),
                zorder=1,
            )
        y += step


def _shared_axis_window_2d(
    label_2d: np.ndarray,
    pad: float = _SHARED_WINDOW_PAD,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Shared image-coordinate window for every 2D habitat-graph panel.

    The window is the bounding box of all foreground voxels on this
    slice (union of every habitat), plus ``pad`` voxels on each side.
    Pairwise panels must not crop or zoom to the two featured habitats.

    Args:
        label_2d: Representative slice (background encoded as 0).
        pad: Extra voxels around the union bbox (default 3).

    Returns:
        ``(xlim, ylim)`` with ``ylim`` inverted so row 0 is at the top.
    """
    array = np.asarray(label_2d)
    n_rows = int(array.shape[0])
    n_cols = int(array.shape[1])
    full_xlim = (-0.5, float(n_cols) - 0.5)
    full_ylim = (float(n_rows) - 0.5, -0.5)
    coords = np.argwhere(array > 0)
    if coords.size == 0:
        return full_xlim, full_ylim
    row0, col0 = (int(value) for value in coords.min(axis=0))
    row1, col1 = (int(value) for value in coords.max(axis=0))
    pad = float(pad)
    x0 = max(full_xlim[0], float(col0) - pad - 0.5)
    x1 = min(full_xlim[1], float(col1) + pad + 0.5)
    y_bottom = min(full_ylim[0], float(row1) + pad + 0.5)
    y_top = max(full_ylim[1], float(row0) - pad - 0.5)
    return (x0, x1), (y_bottom, y_top)


def _apply_shared_axis_window(
    ax: Any,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    """
    Lock one axes to the shared ROI window and equal image aspect.

    ``adjustable='box'`` plus disabled autoscale keep constrained
    layout from expanding data limits on wider pair-row cells.

    Args:
        ax: Target matplotlib axes.
        xlim: Shared column limits (image coordinates).
        ylim: Shared row limits, inverted (image coordinates).
    """
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_autoscale_on(False)


def _fill_unused_grid_cells(
    fig: Any,
    gs: Any,
    n_rows: int,
    n_cols: int,
    occupied: set,
) -> None:
    """
    Occupy unused GridSpec cells so constrained_layout cannot collapse
    empty slots and make pair-row panels wider than H panels.

    Args:
        fig: Parent figure.
        gs: Equal-cell GridSpec.
        n_rows: Number of grid rows.
        n_cols: Number of grid columns (including a reserved colorbar
            column when present).
        occupied: ``(row, col)`` cells that already have an axes.
    """
    for row in range(int(n_rows)):
        for col in range(int(n_cols)):
            if (row, col) in occupied:
                continue
            spacer = fig.add_subplot(gs[row, col])
            spacer.set_axis_off()
            spacer.set_frame_on(False)


def _style_axis_2d(
    ax: Any,
    label_2d: np.ndarray,
    title: str,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title_fontsize: Optional[float] = None,
    title_pad: Optional[float] = None,
) -> None:
    """
    Apply consistent journal styling to a 2D graph/image axis.

    Args:
        ax: Target matplotlib axes.
        label_2d: Slice used to derive axis limits when ``xlim`` /
            ``ylim`` are omitted.
        title: English panel title.
        xlim: Shared column limits. ``None`` uses the full-ROI window
            of ``label_2d``.
        ylim: Shared inverted row limits. ``None`` uses the full-ROI
            window of ``label_2d``.
        title_fontsize: Optional override in points. ``None`` keeps the
            active style preset (single-panel figures stay unchanged).
        title_pad: Optional title padding in points.
    """
    title_kwargs: Dict[str, Any] = {}
    if title_fontsize is not None:
        title_kwargs["fontsize"] = title_fontsize
    if title_pad is not None:
        title_kwargs["pad"] = title_pad
    ax.set_title(title, **title_kwargs)
    if xlim is None or ylim is None:
        xlim, ylim = _shared_axis_window_2d(label_2d)
    _apply_shared_axis_window(ax, xlim, ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_edges_2d(
    ax,
    id_to_node: Dict[str, HabitatGraphNode],
    edges: Sequence[_EdgePair],
    color: Any,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    """Draw a set of 2D edges given node-id pairs (centroid projected to x/y)."""
    for source, target in edges:
        node_a = id_to_node.get(source)
        node_b = id_to_node.get(target)
        if node_a is None or node_b is None:
            continue
        x_a, y_a = _centroid_xy_display(node_a)
        x_b, y_b = _centroid_xy_display(node_b)
        ax.plot(
            (x_a, x_b),
            (y_a, y_b),
            color=color,
            linewidth=linewidth,
            alpha=float(alpha),
            zorder=zorder,
            solid_capstyle="round",
        )


def _draw_nodes_2d(
    ax,
    nodes: Sequence[HabitatGraphNode],
    node_size: float = _DEFAULT_NODE_SIZE,
    linewidths: float = _NODE_EDGE_WIDTH,
) -> None:
    """
    Scatter solid white nodes with a thin dark outline.

    Colour lives on the habitat fill, not on the graph layer. Every
    marker is the same small filled dot so H panels and pairwise
    panels share ``node_size``. Region voxel count is not encoded in
    marker size.

    Args:
        ax: Target matplotlib axes.
        nodes: Nodes to draw.
        node_size: Matplotlib scatter area in points squared. The same
            value is applied to every node.
        linewidths: Dark marker-rim width in points (default thin
            outline so white dots stay visible on light fills).
    """
    if not nodes:
        return
    xs = [_centroid_xy_display(n)[0] for n in nodes]
    ys = [_centroid_xy_display(n)[1] for n in nodes]
    ax.scatter(
        xs,
        ys,
        s=float(node_size),
        c=_GRAPH_NODE_COLOR,
        edgecolors=_NODE_OUTLINE_COLOR,
        linewidths=float(linewidths),
        alpha=1.0,
        zorder=4,
    )


# ---------------------------------------------------------------------------
# 2D figure builders (matplotlib)
# ---------------------------------------------------------------------------


def _apply_display_grid(
    ax: Any,
    label_2d: np.ndarray,
    node_result: HabitatNodeExtractionResult,
    options: HabitatGraphFeatureOptions,
    *,
    show_grid: bool,
    block_size: Optional[int],
    grid_linestyle: str,
    grid_color: str,
    grid_alpha: float,
    grid_linewidth: float,
) -> int:
    """
    Draw the display lattice when requested and return the cube size.

    Args:
        ax: Target matplotlib axes.
        label_2d: Slice used for extent and VOI origin fallback.
        node_result: Extraction result (nodes still come from ``options``).
        options: Extraction options.
        show_grid: Caller on/off switch.
        block_size: Display override, or ``None`` to follow ``options``.
        grid_linestyle: Matplotlib line style.
        grid_color: Lattice colour.
        grid_alpha: Lattice opacity.
        grid_linewidth: Lattice stroke width in points.

    Returns:
        int: Display cube edge length in voxels (also used in captions).
    """
    display_size = _display_block_size(options, block_size)
    if _should_draw_grid(options, show_grid, block_size):
        origin = _display_grid_origin(label_2d, node_result, display_size)
        _draw_grid_2d(
            ax,
            label_2d,
            origin,
            display_size,
            linestyle=grid_linestyle,
            color=grid_color,
            alpha=grid_alpha,
            linewidth=grid_linewidth,
        )
    return display_size


def plot_habitat_graph_slice(
    label_array: np.ndarray,
    *,
    options: HabitatGraphFeatureOptions = HabitatGraphFeatureOptions(),
    slice_index: Optional[int] = None,
    show_grid: bool = True,
    block_size: Optional[int] = None,
    grid_linestyle: str = _DEFAULT_GRID_LINESTYLE,
    grid_color: str = _GRID_LINE_COLOR,
    grid_alpha: float = _GRID_LINE_ALPHA,
    grid_linewidth: float = _GRID_LINE_WIDTH,
    panel_size: float = 3.2,
    colorbar: ColorbarSpec = True,
    colorbar_label: str = DEFAULT_HABITAT_CBAR_LABEL,
) -> "Figure":
    """
    Draw the colored habitat map at the largest cross-section (2D).

    Display knobs (``show_grid``, ``block_size``, ``grid_linestyle``, …)
    override ``options`` for drawing only. Node extraction still uses
    ``options``. Default lattice: ``block_size=None`` →
    ``options.block_size`` (library default 8 voxels), dashed lines.

    Args:
        label_array: 2D or 3D habitat label map (background encoded as 0).
        options: Graph construction options shared with the extractor.
        slice_index: Explicit axis-0 slice for 3D input; ``None`` selects the
            largest cross-section. Ignored for 2D input.
        show_grid: Draw the node lattice (default ``True``).
        block_size: Display cube edge in voxels. ``None`` (default) uses
            ``options.block_size`` so the lattice matches the nodes.
        grid_linestyle: Matplotlib line style (default ``\"--\"`` dashed).
        grid_color: Lattice colour.
        grid_alpha: Lattice opacity.
        grid_linewidth: Lattice stroke width in points.
        panel_size: Base panel edge length in inches.
        colorbar: Discrete habitat-ID colorbar (default ``True``). Background
            ``0`` is omitted from the bar. Pass ``False`` to hide it.
        colorbar_label: Colorbar label (English default ``\"Habitat\"``).

    Returns:
        The matplotlib ``Figure``; the caller decides where it goes.

    Raises:
        OptionalDependencyError: When matplotlib is not installed.
    """
    plt = _plt()
    from matplotlib.colors import ListedColormap

    labels_array = _as_label_array(label_array)
    label_2d, index, _ = _representative_slice(labels_array, slice_index)
    colors = _habitat_colors(np.unique(label_2d[label_2d > 0]))

    ordered = sorted(colors)
    display = np.zeros_like(label_2d)
    for new_value, label in enumerate(ordered, start=1):
        display[label_2d == label] = new_value
    cmap = ListedColormap(["#FFFFFF"] + [colors[label] for label in ordered])
    node_result = _extract_nodes(label_2d, options)
    display_size = _display_block_size(options, block_size)
    with use_style("radiology"):
        fig, ax = plt.subplots(
            figsize=(panel_size * 1.4, panel_size * 1.4),
            constrained_layout=True,
        )
        ax.imshow(
            display,
            cmap=cmap,
            vmin=0,
            vmax=len(ordered),
            interpolation="nearest",
        )
        _apply_display_grid(
            ax,
            label_2d,
            node_result,
            options,
            show_grid=show_grid,
            block_size=block_size,
            grid_linestyle=grid_linestyle,
            grid_color=grid_color,
            grid_alpha=grid_alpha,
            grid_linewidth=grid_linewidth,
        )
        _style_axis_2d(
            ax,
            label_2d,
            (
                f"Habitat map (max cross-section, slice {index}; "
                f"{_grid_caption(display_size)})"
            ),
        )
        add_discrete_habitat_colorbar(
            ax,
            ordered,
            [colors[label] for label in ordered],
            colorbar=colorbar,
            label=colorbar_label,
        )
    return fig


def plot_habitat_graph_network_2d(
    label_array: np.ndarray,
    *,
    options: HabitatGraphFeatureOptions = HabitatGraphFeatureOptions(),
    slice_index: Optional[int] = None,
    show_background: bool = True,
    show_grid: bool = True,
    block_size: Optional[int] = None,
    grid_linestyle: str = _DEFAULT_GRID_LINESTYLE,
    grid_color: str = _GRID_LINE_COLOR,
    grid_alpha: float = _GRID_LINE_ALPHA,
    grid_linewidth: float = _GRID_LINE_WIDTH,
    panel_size: float = 3.2,
    max_cols: int = 4,
    node_size: float = _DEFAULT_NODE_SIZE,
    colorbar: ColorbarSpec = True,
    colorbar_label: str = DEFAULT_HABITAT_CBAR_LABEL,
) -> Optional["Figure"]:
    """
    Draw the intra/inter habitat graphs built from the 2D slice habitat map.

    Node extraction and edge construction reuse the same configured algorithms
    as the feature extractor, but the input is the representative cross-section
    rather than the full 3D volume. Display knobs override ``options`` for
    drawing only (extraction still uses ``options``).

    Each H1--Hk panel fills only that habitat in its palette colour and
    overlays white intra-edges plus white nodes (solid dots, thin dark
    outline). Other habitats use one mid-dark gray wash for shape
    context (not a light fill that hides white strokes). Each unordered
    habitat pair gets its own panel: those two habitats stay in palette
    colours, other habitats use the same gray wash, and only **white
    inter-edges between that pair** are drawn (no intra-edges, no
    edges to other habitats, no purple). H1--H4 sit on the first row
    when ``K <= 4``; six pairs (K=4) use a 2x3 placement in an
    equal-cell parent grid so every panel is the same physical size.
    All panels share one spatial window (union of every habitat on the
    slice plus a shared pad), the same solid-dot node size, and the
    same edge linewidth.

    Args:
        label_array: 2D or 3D habitat label map (background encoded as 0).
        options: Graph construction options shared with the feature extractor.
        slice_index: Explicit axis-0 slice for 3D input; ``None`` selects the
            largest cross-section. Ignored for 2D input.
        show_background: Whether to draw habitat partitions behind the
            graph (default ``True``). Featured habitats (one on an H
            panel, two on a pair panel) are full colour; other
            foreground is a mid-dark gray wash; background 0 stays
            undrawn. Graph nodes and edges are white.
        show_grid: Draw the uniform-grid lattice (default ``True``).
            Also draws when ``block_size`` is passed in ``component`` mode.
        block_size: Display cube edge in voxels. ``None`` (default) uses
            ``options.block_size`` (library default 8 voxels) so the lattice
            matches the nodes.
        grid_linestyle: Matplotlib line style (default ``\"--\"`` dashed).
        grid_color: Lattice colour.
        grid_alpha: Lattice opacity.
        grid_linewidth: Lattice stroke width in points.
        panel_size: Base panel edge length in inches.
        max_cols: Maximum number of H-panel columns (H1--H4 on the first
            row when ``K <= 4``). Six pair panels use a 2x3 placement
            inside an equal-cell parent grid.
        node_size: Matplotlib scatter area in points squared applied to
            every panel (default ``28``). H1--Hk and pairwise panels
            share this size. Voxel count does not scale markers.
        colorbar: Discrete habitat-ID colorbar in a reserved column so
            it does not shrink any H or pair panel (default ``True``).
            Pass ``False`` to hide it.
        colorbar_label: Colorbar label (English default ``\"Habitat\"``).

    Returns:
        The matplotlib ``Figure``, or ``None`` when the slice holds no habitat
        nodes to draw.

    Raises:
        OptionalDependencyError: When matplotlib is not installed.
    """
    plt = _plt()
    from matplotlib.lines import Line2D

    labels_array = _as_label_array(label_array)
    label_2d, index, _ = _representative_slice(labels_array, slice_index)
    colors = _habitat_colors(np.unique(label_2d[label_2d > 0]))

    node_result = _extract_nodes(label_2d, options)
    labels = sorted(node_result.nodes_by_habitat.keys())
    if not labels:
        return None
    id_to_node: Dict[str, HabitatGraphNode] = {
        node.node_id: node
        for label in labels
        for node in node_result.nodes_by_habitat[label]
    }
    pairs = (
        list(iter_label_pairs(labels))
        if options.include_pairwise_habitat_graph
        else []
    )
    display_size = _display_block_size(options, block_size)
    grid_kwargs = dict(
        show_grid=show_grid,
        block_size=block_size,
        grid_linestyle=grid_linestyle,
        grid_color=grid_color,
        grid_alpha=grid_alpha,
        grid_linewidth=grid_linewidth,
    )

    h_rows, h_cols, pair_rows, pair_cols = _network_2d_layout(
        len(labels), len(pairs), max_cols
    )
    shared_xlim, shared_ylim = _shared_axis_window_2d(label_2d)
    # One parent grid: every H / pair cell is the same size. A thin
    # trailing column holds the colorbar so it cannot steal width from
    # a single panel. Unused cells get spacer axes so constrained
    # layout cannot collapse them and widen the pair row.
    n_cols = max(h_cols, pair_cols)
    n_rows = max(1, h_rows + pair_rows)
    draw_cbar = colorbar_is_enabled(colorbar) and bool(colors)
    gs_cols = n_cols + (1 if draw_cbar else 0)
    width_ratios = [1.0] * n_cols + ([0.08] if draw_cbar else [])
    # Extra width + bottom room so short titles and the larger figlegend
    # cannot collide when many habitats share one row.
    fig_width = n_cols * panel_size * 1.18 + (0.55 if draw_cbar else 0.0)
    fig_height = n_rows * panel_size + 1.65
    with use_style("radiology"):
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        fig.set_constrained_layout_pads(
            w_pad=0.05,
            h_pad=0.12,
            wspace=0.12,
            hspace=0.28,
        )
        gs = fig.add_gridspec(n_rows, gs_cols, width_ratios=width_ratios)
        occupied_cells: set = set()

        habitat_axes: List[Any] = []
        for index_h, label in enumerate(labels):
            cell = (index_h // h_cols, index_h % h_cols)
            ax = fig.add_subplot(gs[cell[0], cell[1]])
            occupied_cells.add(cell)
            habitat_axes.append(ax)
            sub = node_result.nodes_by_habitat[label]
            _draw_background_2d(
                ax,
                label_2d,
                colors,
                show_background,
                featured_labels=(int(label),),
            )
            _apply_display_grid(
                ax, label_2d, node_result, options, **grid_kwargs
            )
            # Featured habitat only: white intra-edges, no other-habitat edges.
            edges = _single_intra_edges(sub, label, options, node_result)
            _draw_edges_2d(
                ax,
                id_to_node,
                edges,
                _GRAPH_EDGE_COLOR,
                _DEFAULT_EDGE_WIDTH,
                1.0,
                2,
            )
            _draw_nodes_2d(
                ax,
                sub,
                node_size=node_size,
                linewidths=_NODE_EDGE_WIDTH,
            )
            # Slice index lives on the figure title; keep panel titles short
            # so neighbouring H1 / H2 headings cannot run into each other.
            _style_axis_2d(
                ax,
                label_2d,
                f"H{label} (n={len(sub)}, e={len(edges)})",
                xlim=shared_xlim,
                ylim=shared_ylim,
                title_fontsize=_PANEL_TITLE_FONTSIZE,
                title_pad=4.0,
            )

        pair_axes: List[Any] = []
        for index_p, (label_a, label_b) in enumerate(pairs):
            pair_row, pair_col = divmod(index_p, pair_cols)
            cell = (h_rows + pair_row, pair_col)
            ax = fig.add_subplot(gs[cell[0], cell[1]])
            occupied_cells.add(cell)
            pair_axes.append(ax)
            nodes_a = list(node_result.nodes_by_habitat.get(label_a, []))
            nodes_b = list(node_result.nodes_by_habitat.get(label_b, []))
            pair_nodes = nodes_a + nodes_b
            inter_edges = _pair_inter_edges(
                node_result, label_a, label_b, options
            )
            _draw_background_2d(
                ax,
                label_2d,
                colors,
                show_background,
                featured_labels=(int(label_a), int(label_b)),
            )
            _apply_display_grid(
                ax, label_2d, node_result, options, **grid_kwargs
            )
            # Pair only: white inter-edges, no intra-edges.
            _draw_edges_2d(
                ax,
                id_to_node,
                inter_edges,
                _GRAPH_EDGE_COLOR,
                _DEFAULT_EDGE_WIDTH,
                1.0,
                3,
            )
            _draw_nodes_2d(
                ax,
                pair_nodes,
                node_size=node_size,
                linewidths=_NODE_EDGE_WIDTH,
            )
            _style_axis_2d(
                ax,
                label_2d,
                _pair_panel_title(
                    label_a, label_b, len(pair_nodes), len(inter_edges)
                ),
                xlim=shared_xlim,
                ylim=shared_ylim,
                title_fontsize=_PANEL_TITLE_FONTSIZE,
                title_pad=4.0,
            )

        if draw_cbar:
            # Dedicated column: do not attach to a panel axes (that
            # shrinks only that subplot and breaks equal panel size).
            # Occupy every row of the column so pair rows stay the same
            # width as the H row.
            cbar_host = fig.add_subplot(gs[0, n_cols])
            occupied_cells.add((0, n_cols))
            ordered = sorted(colors)
            mappable, ticks, ticklabels = discrete_habitat_mappable(
                ordered,
                [colors[label] for label in ordered],
            )
            n_habitats = len(ticks)
            boundaries = [0.5 + float(index) for index in range(n_habitats + 1)]
            cbar = fig.colorbar(
                mappable,
                cax=cbar_host,
                ticks=ticks,
                boundaries=boundaries,
                spacing="uniform",
                drawedges=True,
            )
            cbar.set_label(colorbar_label)
            from matplotlib.ticker import FixedFormatter, FixedLocator

            cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
            cbar.ax.yaxis.set_major_formatter(
                FixedFormatter([str(text) for text in ticklabels])
            )
            cbar.minorticks_off()
            cbar.ax.tick_params(
                which="major",
                length=3.0,
                width=0.6,
                labelsize=_AXIS_TEXT_FONTSIZE,
            )
            cbar.ax.yaxis.label.set_size(_AXIS_TEXT_FONTSIZE)
        _fill_unused_grid_cells(fig, gs, n_rows, gs_cols, occupied_cells)
        for ax in habitat_axes + pair_axes:
            _apply_shared_axis_window(ax, shared_xlim, shared_ylim)
        # Dark legend frame so solid-white node / edge swatches stay visible.
        node_handle = Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=6.5,
            markerfacecolor=_GRAPH_NODE_COLOR,
            markeredgecolor=_NODE_OUTLINE_COLOR,
            markeredgewidth=0.8,
            label="Node",
        )
        intra_handle = Line2D(
            [0],
            [0],
            color=_GRAPH_EDGE_COLOR,
            lw=1.6,
            label="Intra-habitat edge",
        )
        inter_handle = Line2D(
            [0],
            [0],
            color=_GRAPH_EDGE_COLOR,
            lw=1.6,
            label="Inter-habitat edge",
        )
        grid_handle = Line2D(
            [0],
            [0],
            color="#D1D5DB",
            lw=1.2,
            linestyle=grid_linestyle,
            label=_grid_caption(display_size),
        )
        fig.legend(
            handles=[node_handle, intra_handle, inter_handle, grid_handle],
            labels=[
                "Node",
                "Intra-habitat edge",
                "Inter-habitat edge",
                _grid_caption(display_size),
            ],
            loc="lower center",
            ncol=4,
            fontsize=_FIG_LEGEND_FONTSIZE,
            facecolor="#4B5563",
            edgecolor="#1A1A1A",
            labelcolor="white",
            framealpha=0.95,
        )
        fig.suptitle(
            (
                f"2D habitat graphs from representative cross-section "
                f"(slice {index}; {_grid_caption(display_size)})"
            ),
            fontsize=_PANEL_TITLE_FONTSIZE,
        )
    return fig


# ---------------------------------------------------------------------------
# Subject x graph-feature heatmap (matplotlib)
# ---------------------------------------------------------------------------


def _ascii_minus_on_ticks(fig: "Figure") -> None:
    """
    Persist ASCII '-' on numeric axes (including colorbars).

    ``use_style`` restores rcParams on exit, so a later ``get_ticklabels``
    would otherwise regenerate U+2212. A FuncFormatter stays on the axes.
    Categorical ticks (subject ids, feature names) keep their FixedFormatter.
    """
    matplotlib = require("matplotlib", extra="viz", purpose=_VIZ_PURPOSE)
    ticker = matplotlib.ticker
    formatter = ticker.FuncFormatter(
        lambda value, _pos: f"{value:g}".replace("\u2212", "-")
    )
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            current = axis.get_major_formatter()
            if isinstance(current, ticker.ScalarFormatter):
                axis.set_major_formatter(formatter)


def _heatmap_color_scale(
    finite: np.ndarray,
    *,
    zscore: bool,
) -> Tuple[Optional[float], Optional[float], str, str]:
    """
    Choose colormap limits and the colorbar label for a graph heatmap.

    Z-scored columns (including a column-z-scored ``table - reference``
    difference) share a zero-centered diverging map. Unsigned raw
    values keep a sequential map because mixed graph units are not a
    single signed scale. Signed ``zscore=False`` values also use the
    diverging map.

    Args:
        finite: Finite entries of the drawn matrix (may be empty).
        zscore: Whether the matrix was column-wise z-scored.

    Returns:
        Tuple of ``(vmin, vmax, cmap_name, colorbar_label)``. ``vmin`` /
        ``vmax`` are ``None`` when matplotlib should autoscale.
    """
    if zscore and finite.size:
        vmax = float(np.nanmax(np.abs(finite)))
        vmax = 1.0 if vmax == 0.0 else vmax
        return -vmax, vmax, "RdBu_r", "Z-score (across subjects)"
    signed = (
        finite.size > 0
        and float(np.nanmin(finite)) < 0.0
        and float(np.nanmax(finite)) > 0.0
    )
    if signed:
        vmax = float(np.nanmax(np.abs(finite)))
        vmax = 1.0 if vmax == 0.0 else vmax
        return -vmax, vmax, "RdBu_r", "Feature difference"
    return None, None, "cividis", "Feature value (mixed units)"


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """
    Z-score each feature (column) across subjects; NaN-safe.

    Args:
        matrix: Subject x feature values.

    Returns:
        np.ndarray: Same shape; columns with fewer than two finite
        values become NaN; zero-variance columns become 0.
    """
    out = np.asarray(matrix, dtype=np.float64).copy()
    for col in range(out.shape[1]):
        values = out[:, col]
        finite = np.isfinite(values)
        if int(finite.sum()) < 2:
            out[:, col] = np.nan
            continue
        mu = float(np.mean(values[finite]))
        sd = float(np.std(values[finite], ddof=0))
        if sd == 0.0:
            out[:, col] = 0.0
        else:
            scaled = (values - mu) / sd
            scaled[~finite] = np.nan
            out[:, col] = scaled
    return out


def _is_graph_feature_column(name: str) -> bool:
    """Return True for graph-family columns (single / pair / graph_num)."""
    text = str(name)
    return bool(
        _SINGLE_FEATURE_RE.match(text)
        or _PAIR_FEATURE_RE.match(text)
        or _GRAPH_NUM_RE.match(text)
    )


def _columns_for_feature_group(
    columns: Sequence[str],
    feature_group: Literal["single", "pair", "all"],
) -> List[str]:
    """
    Filter graph-feature column names by family.

    ``single`` / ``pair`` keep only ``single_h*`` / ``pair_h*``.
    ``graph_num_*`` is excluded from those two groups and included only
    when ``feature_group='all'``.
    """
    group = str(feature_group)
    if group not in ("single", "pair", "all"):
        raise HABITAPIError(
            "plot_graph_feature_heatmap: feature_group must be "
            f"'single', 'pair', or 'all'; got {feature_group!r}."
        )
    selected: List[str] = []
    for name in columns:
        text = str(name)
        if group == "single" and _SINGLE_FEATURE_RE.match(text):
            selected.append(text)
        elif group == "pair" and _PAIR_FEATURE_RE.match(text):
            selected.append(text)
        elif group == "all" and _is_graph_feature_column(text):
            selected.append(text)
    return selected


def _graph_feature_tick(name: str) -> str:
    """ASCII tick: keep the habitat prefix, wrap the metric on a second line."""
    label = sanitize_label(str(name))
    single = re.match(r"^(single_h\d+)_(.+)$", label)
    if single:
        return f"{single.group(1)}\n{single.group(2)}"
    pair = re.match(r"^(pair_h\d+_h\d+)_(.+)$", label)
    if pair:
        return f"{pair.group(1)}\n{pair.group(2)}"
    return label


def _resolve_subject_frame(
    table: "pd.DataFrame",
    *,
    subjects: Optional[Sequence[str]],
    subject_col: str,
) -> "pd.DataFrame":
    """Restrict and reorder rows to the requested subject ids."""
    if subject_col not in table.columns:
        hint = ""
        if "subject" in table.columns and subject_col != "subject":
            hint = (
                " FeatureTable frames from GraphHabitatFeatures use "
                "subject_col='subject'."
            )
        raise HABITAPIError(
            "plot_graph_feature_heatmap: subject column "
            f"{subject_col!r} is not in the table. Available columns: "
            f"{list(table.columns)}.{hint}"
        )
    frame = table.copy()
    frame[subject_col] = frame[subject_col].astype(str)
    if subjects is None:
        if frame.empty:
            raise HABITAPIError(
                "plot_graph_feature_heatmap: table has no subject rows."
            )
        return frame
    wanted = [str(item) for item in subjects]
    if not wanted:
        raise HABITAPIError(
            "plot_graph_feature_heatmap: subjects is an empty sequence."
        )
    available = set(frame[subject_col].tolist())
    missing = [item for item in wanted if item not in available]
    if missing:
        raise HABITAPIError(
            "plot_graph_feature_heatmap: subject id(s) not in the table: "
            f"{missing}. Available: {sorted(available)}."
        )
    order = {item: index for index, item in enumerate(wanted)}
    frame = frame.loc[frame[subject_col].isin(wanted)].copy()
    frame["_habit_subject_order"] = frame[subject_col].map(order)
    frame = frame.sort_values("_habit_subject_order", kind="stable")
    return frame.drop(columns=["_habit_subject_order"])


def _coerce_numeric_column(column: Any) -> Any:
    """Coerce one series to float, turning non-numeric cells into NaN."""
    import pandas as pd

    return pd.to_numeric(column, errors="coerce")


def _select_heatmap_features(
    frame: "pd.DataFrame",
    *,
    features: Optional[Sequence[str]],
    n_features: int,
    feature_group: Literal["single", "pair", "all"],
    select: Literal["variance", "sample"],
    sample_seed: int,
    subject_col: str,
) -> List[str]:
    """
    Choose which graph-feature columns to draw.

    An explicit ``features`` list wins (must exist) and ignores
    ``n_features`` / ``select`` / ``feature_group``.
    """
    if features is not None:
        wanted = [str(name) for name in features]
        if not wanted:
            raise HABITAPIError(
                "plot_graph_feature_heatmap: features is an empty sequence."
            )
        missing = [name for name in wanted if name not in frame.columns]
        if missing:
            raise HABITAPIError(
                "plot_graph_feature_heatmap: feature column(s) not in "
                f"the table: {missing}."
            )
        return wanted
    if int(n_features) < 1:
        raise HABITAPIError(
            "plot_graph_feature_heatmap: n_features must be >= 1; "
            f"got {n_features}."
        )
    method = str(select)
    if method not in ("variance", "sample"):
        raise HABITAPIError(
            "plot_graph_feature_heatmap: select must be 'variance' or "
            f"'sample'; got {select!r}."
        )
    candidates = _columns_for_feature_group(
        [str(name) for name in frame.columns if str(name) != subject_col],
        feature_group,
    )
    if not candidates:
        raise HABITAPIError(
            "plot_graph_feature_heatmap: no columns matched "
            f"feature_group={feature_group!r}. Expected names such as "
            "single_h1_avg_degree, pair_h1_h2_edge_density, or "
            "graph_num_habitats (group='all' only)."
        )
    cap = min(int(n_features), len(candidates))
    if cap >= len(candidates):
        return list(candidates)
    numeric = frame[candidates].apply(
        lambda col: _coerce_numeric_column(col), axis=0
    )
    matrix = np.asarray(numeric, dtype=np.float64)
    if method == "sample":
        rng = np.random.default_rng(int(sample_seed))
        chosen = rng.choice(len(candidates), size=cap, replace=False)
        return [candidates[int(index)] for index in chosen]
    variances = np.empty(len(candidates), dtype=np.float64)
    for index in range(len(candidates)):
        values = matrix[:, index]
        finite = values[np.isfinite(values)]
        variances[index] = (
            float(np.var(finite, ddof=0)) if finite.size else -1.0
        )
    order = np.argsort(-variances, kind="stable")
    return [candidates[int(index)] for index in order[:cap]]


def _align_reference_frame(
    table: "pd.DataFrame",
    reference: "pd.DataFrame",
    *,
    subjects: Optional[Sequence[str]],
    subject_col: str,
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """
    Align ``table`` and ``reference`` on subjects and shared features.

    Args:
        table: Minuend frame (for example the 5-voxel extract).
        reference: Subtrahend frame (for example the 8-voxel extract).
        subjects: Optional subject-id order applied to both frames.
        subject_col: Identifier column shared by both frames.

    Returns:
        Tuple of ``(left, right, delta)``. ``delta`` is
        ``left - right`` on the intersecting feature columns, with
        ``subject_col`` restored. Subject order follows ``table``
        (or ``subjects``).

    Raises:
        HABITAPIError: Missing subject column, empty intersection, or
            subjects present in ``table`` but not in ``reference``.
    """
    import pandas as pd

    if not isinstance(reference, pd.DataFrame):
        raise HABITAPIError(
            "plot_graph_feature_heatmap: reference must be a pandas "
            f"DataFrame; got {type(reference)!r}."
        )
    left = _resolve_subject_frame(
        table, subjects=subjects, subject_col=subject_col
    )
    left_ids = [str(item) for item in left[subject_col]]
    right = _resolve_subject_frame(
        reference, subjects=left_ids, subject_col=subject_col
    )
    feature_cols = [
        name
        for name in left.columns
        if name != subject_col and name in right.columns
    ]
    if not feature_cols:
        raise HABITAPIError(
            "plot_graph_feature_heatmap: table and reference share no "
            "feature columns to subtract."
        )
    left_num = left.set_index(subject_col)[feature_cols].apply(
        lambda col: _coerce_numeric_column(col), axis=0
    )
    right_num = right.set_index(subject_col)[feature_cols].apply(
        lambda col: _coerce_numeric_column(col), axis=0
    )
    # Reindex right to left's subject order (already aligned, belt-and-braces).
    right_num = right_num.reindex(left_num.index)
    delta = (left_num - right_num).reset_index()
    return left, right, delta


def _paired_ttest_pvalues(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """
    Paired t-test p-value per feature column (subjects x features).

    Equivalent to a one-sample t-test of ``left - right`` against 0.
    A column is skipped (NaN p-value, no star) when fewer than 3
    finite pairs remain or the paired difference is constant.

    Args:
        left: Minuend matrix, shape ``(n_subjects, n_features)``.
        right: Subtrahend matrix, same shape.

    Returns:
        np.ndarray: Length ``n_features``; NaN means "do not star".
    """
    from scipy.stats import ttest_rel

    matrix_l = np.asarray(left, dtype=np.float64)
    matrix_r = np.asarray(right, dtype=np.float64)
    if matrix_l.shape != matrix_r.shape:
        raise HABITAPIError(
            "plot_graph_feature_heatmap: paired tables must have the "
            f"same aligned shape; got {matrix_l.shape} vs {matrix_r.shape}."
        )
    n_features = int(matrix_l.shape[1]) if matrix_l.ndim == 2 else 0
    pvalues = np.full(n_features, np.nan, dtype=np.float64)
    for index in range(n_features):
        values_l = matrix_l[:, index]
        values_r = matrix_r[:, index]
        finite = np.isfinite(values_l) & np.isfinite(values_r)
        if int(finite.sum()) < 3:
            continue
        paired = values_l[finite] - values_r[finite]
        if float(np.std(paired, ddof=1)) == 0.0:
            continue
        result = ttest_rel(values_l[finite], values_r[finite])
        pvalue = float(result.pvalue)
        if np.isfinite(pvalue):
            pvalues[index] = pvalue
    return pvalues


def _adjust_pvalues(
    pvalues: np.ndarray,
    method: Literal["fdr_bh", "bonferroni", "none"],
) -> np.ndarray:
    """
    Multiple-testing correction over the plotted (finite) p-values.

    ``fdr_bh`` prefers :func:`scipy.stats.false_discovery_control`
    (Benjamini-Hochberg), then ``statsmodels.stats.multitest``, then
    Bonferroni. Columns that were skipped (NaN) stay NaN.

    Args:
        pvalues: Per-feature p-values; NaN entries are not tested.
        method: ``'fdr_bh'``, ``'bonferroni'``, or ``'none'``.

    Returns:
        np.ndarray: Adjusted p-values, same shape.
    """
    out = np.asarray(pvalues, dtype=np.float64).copy()
    if method not in ("fdr_bh", "bonferroni", "none"):
        raise HABITAPIError(
            "plot_graph_feature_heatmap: star_mtc must be 'fdr_bh', "
            f"'bonferroni', or 'none'; got {method!r}."
        )
    finite = np.isfinite(out)
    tested = out[finite]
    if tested.size == 0 or method == "none":
        return out
    if method == "bonferroni":
        out[finite] = np.minimum(tested * float(tested.size), 1.0)
        return out
    try:
        from scipy.stats import false_discovery_control

        out[finite] = np.asarray(
            false_discovery_control(tested, method="bh"), dtype=np.float64
        )
        return out
    except (ImportError, TypeError, ValueError):
        pass
    try:
        from statsmodels.stats.multitest import multipletests

        _rejected, adjusted, _alphac_sidak, _alphac_bonf = multipletests(
            tested, method="fdr_bh"
        )
        out[finite] = np.asarray(adjusted, dtype=np.float64)
        return out
    except ImportError:
        out[finite] = np.minimum(tested * float(tested.size), 1.0)
        return out


def _feature_tick_with_star(name: str, starred: bool) -> str:
    """Wrapped feature tick; append ASCII `` *`` when the column is significant."""
    label = _graph_feature_tick(name)
    if starred:
        return f"{label} *"
    return label


def plot_graph_feature_heatmap(
    table: "pd.DataFrame",
    *,
    subjects: Optional[Sequence[str]] = None,
    features: Optional[Sequence[str]] = None,
    n_features: int = _DEFAULT_HEATMAP_FEATURES,
    feature_group: Literal["single", "pair", "all"] = "single",
    select: Literal["variance", "sample"] = "variance",
    sample_seed: int = 0,
    zscore: bool = True,
    reference: Optional["pd.DataFrame"] = None,
    star_significant: bool = False,
    star_alpha: float = 0.05,
    star_test: Literal["ttest_rel"] = "ttest_rel",
    star_mtc: Literal["fdr_bh", "bonferroni", "none"] = "fdr_bh",
    cbar_label: Optional[str] = None,
    subject_col: str = "subject_id",
    title: Optional[str] = None,
    ax: Optional[Any] = None,
) -> "Figure":
    """
    Draw a subject x graph-feature heatmap (not habitat x texture).

    Columns in a graph table mix incompatible units (counts, ratios,
    path lengths). ``zscore=True`` (default) standardizes each selected
    feature across the **selected subjects** so a row is a relative
    profile, not a raw magnitude. Raw mixed units are not comparable;
    pass ``zscore=False`` only when every drawn column already shares
    one scale. Signed ``zscore=False`` values use a zero-centered
    diverging map.

    For a lattice comparison (for example 5-voxel minus 8-voxel), pass
    the 5-voxel frame as ``table`` and the 8-voxel frame as
    ``reference``. The function aligns subjects and shared feature
    columns, plots ``table - reference``, and (when ``zscore=True``)
    column-z-scores that **raw** difference. Do not pass a precomputed
    (or already z-scored) delta as ``table`` together with
    ``reference`` — that would subtract twice.

    ``star_significant=True`` marks **features** (x-tick labels), not
    cells. Each plotted column gets a paired t-test
    (``scipy.stats.ttest_rel`` of the two source tables; equivalent to
    a one-sample t of the raw difference against 0). Columns with
    fewer than 3 finite pairs or a constant difference are skipped.
    Multiple testing defaults to Benjamini-Hochberg FDR across the
    **plotted** features (``scipy.stats.false_discovery_control``, then
    statsmodels, then Bonferroni). Significant names get a trailing
    `` *``. Starring requires ``reference``; a lone precomputed delta
    cannot reconstruct the pairing. Default ``star_significant=False``
    so generic heatmaps stay unmarked.

    Visualization parameters (who / which features / how many) are
    first-class: pass ``subjects`` and either an explicit ``features``
    list or ``n_features`` + ``feature_group`` + ``select``. The default
    cap is 40 columns so the full ~400-feature bank is never dumped
    onto one figure.

    This is a different figure from
    :func:`~habit.viz.plot_habitat_feature_heatmap` (habitats x
    radiomics features).

    Args:
        table: Wide frame, one row per subject. Identifier column
            defaults to ``subject_id``. Domain FeatureTable frames use
            ``subject`` — pass ``subject_col='subject'``. When
            ``reference`` is set this is the minuend (e.g. 5-voxel).
        subjects: Subject ids to show, in y-axis order. ``None`` keeps
            every row. Missing ids raise :class:`~habit.exceptions.HABITAPIError`.
        features: Exact column list. When set, it overrides
            ``n_features``, ``select``, and ``feature_group``.
        n_features: Column cap when ``features`` is omitted (default 40).
        feature_group: ``'single'`` (``single_h*``), ``'pair'``
            (``pair_h*``), or ``'all'`` (those plus ``graph_num_*``).
            ``graph_num_*`` is excluded from ``single`` / ``pair``.
        select: When ``features`` is omitted, take the top-k columns by
            cross-subject variance (``'variance'``) or a reproducible
            random subset (``'sample'``, seeded by ``sample_seed``).
            With ``reference``, variance is of the raw difference.
        sample_seed: RNG seed for ``select='sample'``.
        zscore: Column-wise z-score across the selected subjects
            (default ``True``). Requires at least two subjects.
            ``False`` draws the (possibly subtracted) values as-is.
        reference: Optional paired frame (e.g. 8-voxel). When set,
            the plotted matrix is aligned ``table - reference``.
            Required when ``star_significant=True``.
        star_significant: If True, append `` *`` to x-tick labels of
            features that stay significant after ``star_mtc``.
            Default False; ignored pairing is never inferred from a
            precomputed delta alone.
        star_alpha: Significance threshold after correction (default
            0.05).
        star_test: Paired test. Only ``'ttest_rel'`` is supported.
        star_mtc: Multiple-testing method over plotted features:
            ``'fdr_bh'`` (default), ``'bonferroni'``, or ``'none'``.
        cbar_label: Optional colorbar label. ``None`` uses a default
            from the scale (``Z-scored difference`` when
            ``reference`` and ``zscore`` are both set).
        subject_col: Identifier column name (default ``'subject_id'``).
        title: Optional English figure title. ``None`` builds one from
            the group and whether values are z-scored.
        ax: Optional existing axes. ``None`` creates a new figure.

    Returns:
        The matplotlib ``Figure``. The caller decides whether to save it.

    Raises:
        HABITAPIError: Missing subjects / columns, empty selection,
            invalid knobs, ``star_significant=True`` without
            ``reference``, or ``zscore=True`` with fewer than two rows.
        OptionalDependencyError: When matplotlib is not installed.
    """
    import pandas as pd

    if not isinstance(table, pd.DataFrame):
        raise HABITAPIError(
            "plot_graph_feature_heatmap: table must be a pandas "
            f"DataFrame; got {type(table)!r}."
        )
    left_frame: Optional["pd.DataFrame"] = None
    right_frame: Optional["pd.DataFrame"] = None
    is_difference = False
    if reference is not None:
        left_frame, right_frame, frame = _align_reference_frame(
            table,
            reference,
            subjects=subjects,
            subject_col=subject_col,
        )
        is_difference = True
    else:
        frame = _resolve_subject_frame(
            table, subjects=subjects, subject_col=subject_col
        )
    chosen = _select_heatmap_features(
        frame,
        features=features,
        n_features=n_features,
        feature_group=feature_group,
        select=select,
        sample_seed=sample_seed,
        subject_col=subject_col,
    )
    subject_ids = [sanitize_label(str(item)) for item in frame[subject_col]]
    if zscore and len(subject_ids) < 2:
        raise HABITAPIError(
            "plot_graph_feature_heatmap: zscore=True needs at least 2 "
            "subjects (column-wise standardization across people). "
            "Pass zscore=False for a single row, or include more "
            "people via subjects=..."
        )
    if star_significant:
        if left_frame is None or right_frame is None:
            raise HABITAPIError(
                "plot_graph_feature_heatmap: star_significant=True "
                "requires reference= (the paired table). A "
                "precomputed delta cannot reconstruct pairing."
            )
        if str(star_test) != "ttest_rel":
            raise HABITAPIError(
                "plot_graph_feature_heatmap: star_test must be "
                f"'ttest_rel'; got {star_test!r}."
            )
        alpha = float(star_alpha)
        if not (0.0 < alpha <= 1.0):
            raise HABITAPIError(
                "plot_graph_feature_heatmap: star_alpha must be in "
                f"(0, 1]; got {star_alpha!r}."
            )
        left_matrix = np.asarray(
            left_frame[chosen].apply(
                lambda col: pd.to_numeric(col, errors="coerce")
            ),
            dtype=np.float64,
        )
        right_matrix = np.asarray(
            right_frame[chosen].apply(
                lambda col: pd.to_numeric(col, errors="coerce")
            ),
            dtype=np.float64,
        )
        raw_p = _paired_ttest_pvalues(left_matrix, right_matrix)
        adjusted = _adjust_pvalues(raw_p, star_mtc)
        starred = [
            bool(np.isfinite(value) and float(value) < alpha)
            for value in adjusted
        ]
    else:
        starred = [False] * len(chosen)
    matrix = np.asarray(
        frame[chosen].apply(lambda col: pd.to_numeric(col, errors="coerce")),
        dtype=np.float64,
    )
    shown = _zscore_columns(matrix) if zscore else matrix

    plt = _plt()
    n_feat = max(len(chosen), 1)
    n_subj = max(len(subject_ids), 1)
    cell_in = 0.32
    left_in, right_in, top_in, bottom_in = 1.15, 1.05, 0.55, 1.85
    fig_w = min(14.0, max(6.4, left_in + right_in + cell_in * n_feat))
    fig_h = min(7.2, max(3.6, top_in + bottom_in + 0.42 * n_subj))
    group_titles = {
        "single": "Single-habitat graph features",
        "pair": "Pairwise graph features",
        "all": "Graph features",
    }
    if title is not None:
        resolved = title
    else:
        scale = "column z-score" if zscore else "raw values"
        if is_difference:
            resolved = f"Graph feature difference ({scale})"
        elif features is not None:
            resolved = f"Graph features ({scale})"
        else:
            resolved = f"{group_titles[str(feature_group)]} ({scale})"

    with use_style("radiology") as style:
        axes: Any = ax
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(fig_w, fig_h), layout=None)
            fig.subplots_adjust(
                left=left_in / fig_w,
                right=1.0 - right_in / fig_w,
                top=1.0 - top_in / fig_h,
                bottom=bottom_in / fig_h,
            )
        else:
            fig = axes.figure
        finite = shown[np.isfinite(shown)]
        vmin, vmax, cmap, default_cbar = _heatmap_color_scale(
            finite, zscore=zscore
        )
        if cbar_label is not None:
            cbar_text = cbar_label
        elif is_difference and zscore:
            cbar_text = "Z-scored difference"
        else:
            cbar_text = default_cbar
        image = axes.imshow(
            shown,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axes.set_yticks(np.arange(len(subject_ids)))
        axes.set_yticklabels(subject_ids, fontsize=_HEATMAP_TICK_FONTSIZE)
        axes.set_xticks(np.arange(len(chosen)))
        axes.set_xticklabels(
            [
                _feature_tick_with_star(name, flag)
                for name, flag in zip(chosen, starred)
            ],
            rotation=90,
            ha="center",
            va="top",
            fontsize=_HEATMAP_TICK_FONTSIZE,
        )
        axes.set_xlabel(
            sanitize_label("Graph feature"), fontsize=_HEATMAP_LABEL_FONTSIZE
        )
        axes.set_ylabel(sanitize_label("Subject"), fontsize=_HEATMAP_LABEL_FONTSIZE)
        cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
        cbar.set_label(
            sanitize_label(cbar_text),
            fontsize=_HEATMAP_CBAR_FONTSIZE,
        )
        cbar.ax.tick_params(labelsize=_HEATMAP_TICK_FONTSIZE)
        axes.set_title(sanitize_label(resolved), fontsize=_HEATMAP_TITLE_FONTSIZE)
        axes.tick_params(axis="both", labelsize=_HEATMAP_TICK_FONTSIZE)
        axes.xaxis.label.set_size(_HEATMAP_LABEL_FONTSIZE)
        axes.yaxis.label.set_size(_HEATMAP_LABEL_FONTSIZE)
        _ = style
        _ascii_minus_on_ticks(fig)
    return fig


# ---------------------------------------------------------------------------
# 3D renderers (PyVista; return RGB arrays, never touch the filesystem)
# ---------------------------------------------------------------------------


def _check_3d_dependencies() -> None:
    """
    Verify the optional packages required for PyVista 3D rendering.

    Raises:
        OptionalDependencyError: When pyvista or scikit-image is missing.
    """
    require("pyvista", extra="view", purpose=_VIEW_PURPOSE)
    require("skimage", extra="slic", purpose=_VIEW_PURPOSE)


def _new_plotter(render_window: int, black_background: bool):
    """Create an off-screen PyVista plotter with journal styling."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(render_window, render_window))
    plotter.set_background("black" if black_background else "white")
    return plotter


def _add_habitat_surfaces(
    plotter,
    label_3d: np.ndarray,
    colors: Dict[int, str],
    spacing: Tuple[float, float, float],
    surface_smooth_iter: int,
) -> bool:
    """
    Add opaque marching-cubes habitat surfaces to a PyVista plotter.

    Args:
        plotter: Target PyVista plotter.
        label_3d: Cropped 3D habitat label map.
        colors: Habitat label to display color mapping.
        spacing: Voxel spacing as ``(sz, sy, sx)``.
        surface_smooth_iter: Laplacian smoothing iterations; 0 disables.

    Returns:
        bool: ``True`` when at least one habitat surface was added.
    """
    import pyvista as pv
    from skimage import measure

    drew_any = False
    for label in sorted(colors):
        binary = (label_3d == label).astype(np.float32)
        if binary.sum() < 8:
            continue
        padded = np.pad(binary, 1, mode="constant")
        try:
            verts, faces, _, _ = measure.marching_cubes(
                padded, level=0.5, spacing=spacing
            )
        except Exception:
            continue
        verts = verts - np.array(spacing)  # undo the 1-voxel pad
        verts_xyz = verts[:, [2, 1, 0]]  # (z, y, x) -> (x, y, z)
        faces_pv = np.hstack(
            [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
        ).ravel()
        mesh = pv.PolyData(verts_xyz, faces_pv)
        if surface_smooth_iter > 0:
            mesh = mesh.smooth(n_iter=surface_smooth_iter)
        plotter.add_mesh(
            mesh,
            color=colors[label],
            smooth_shading=True,
            ambient=0.25,
            diffuse=0.7,
            specular=0.4,
            specular_power=15,
        )
        drew_any = True
    return drew_any


def _screenshot_rgb(plotter) -> np.ndarray:
    """Render the scene off-screen and return it as an RGB array."""
    plotter.camera_position = "iso"
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return np.asarray(image)


def render_habitat_graph_surface_3d(
    label_array: np.ndarray,
    *,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    black_background: bool = True,
    render_window: int = 1600,
    surface_smooth_iter: int = 30,
) -> Optional[np.ndarray]:
    """
    Render a 3D surface view of the habitat volume (PyVista).

    Args:
        label_array: 3D habitat label map (background encoded as 0).
        spacing: Voxel spacing as ``(sz, sy, sx)``.
        black_background: Whether the scene uses a black background.
        render_window: Off-screen render window edge length in pixels.
        surface_smooth_iter: Laplacian smoothing iterations; 0 disables.

    Returns:
        Optional[np.ndarray]: RGB render of shape ``(H, W, 3)``, or ``None``
        when the volume holds no renderable habitat surface.

    Raises:
        ValueError: When ``label_array`` is not 3D.
        OptionalDependencyError: When pyvista or scikit-image is missing.
    """
    _check_3d_dependencies()
    labels_array = _as_label_array(label_array)
    if labels_array.ndim != 3:
        raise ValueError(
            f"render_habitat_graph_surface_3d requires a 3D volume; "
            f"got shape {tuple(labels_array.shape)}."
        )
    cropped = _crop_to_foreground(labels_array)
    colors = _habitat_colors(np.unique(cropped[cropped > 0]))

    plotter = _new_plotter(render_window, black_background)
    if not _add_habitat_surfaces(plotter, cropped, colors, spacing, surface_smooth_iter):
        plotter.close()
        return None
    plotter.add_axes(color="white" if black_background else "black")
    return _screenshot_rgb(plotter)


def render_habitat_graph_network_3d(
    label_array: np.ndarray,
    *,
    options: HabitatGraphFeatureOptions = HabitatGraphFeatureOptions(),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    black_background: bool = True,
    render_window: int = 1600,
) -> Optional[np.ndarray]:
    """
    Render the 3D feature-aligned spatial graph on a dark scene (PyVista).

    Nodes are spheres at region centroids colored by habitat; intra-habitat
    edges are thin gray tubes and inter-habitat edges are thicker accent tubes.

    Args:
        label_array: 3D habitat label map (background encoded as 0).
        options: Graph construction options shared with the feature extractor.
        spacing: Voxel spacing as ``(sz, sy, sx)``.
        black_background: Whether the scene uses a black background.
        render_window: Off-screen render window edge length in pixels.

    Returns:
        Optional[np.ndarray]: RGB render of shape ``(H, W, 3)``, or ``None``
        when the volume yields no graph nodes.

    Raises:
        ValueError: When ``label_array`` is not 3D.
        OptionalDependencyError: When pyvista or scikit-image is missing.
    """
    _check_3d_dependencies()
    import pyvista as pv

    labels_array = _as_label_array(label_array)
    if labels_array.ndim != 3:
        raise ValueError(
            f"render_habitat_graph_network_3d requires a 3D volume; "
            f"got shape {tuple(labels_array.shape)}."
        )
    cropped = _crop_to_foreground(labels_array)
    colors = _habitat_colors(np.unique(cropped[cropped > 0]))

    node_result = _extract_nodes(cropped, options)
    labels = sorted(node_result.nodes_by_habitat.keys())
    id_to_node, intra_edges, inter_edges = _combined_graph(node_result, options)
    if len(id_to_node) < 1:
        return None

    # Approximate node radius from the median region size and the scene scale.
    all_points = np.array([_phys_xyz(n, spacing) for n in id_to_node.values()])
    scene_extent = float(np.ptp(all_points, axis=0).max()) if len(all_points) > 1 else 10.0
    node_radius = max(scene_extent * 0.012, 0.5)
    intra_radius = node_radius * 0.25
    inter_radius = node_radius * 0.4

    plotter = _new_plotter(render_window, black_background)

    def _add_tubes(edges: Sequence[_EdgePair], color: str, radius: float) -> None:
        points: List[np.ndarray] = []
        lines: List[int] = []
        for source, target in edges:
            node_a = id_to_node.get(source)
            node_b = id_to_node.get(target)
            if node_a is None or node_b is None:
                continue
            index = len(points)
            points.append(_phys_xyz(node_a, spacing))
            points.append(_phys_xyz(node_b, spacing))
            lines.extend([2, index, index + 1])
        if not points:
            return
        poly = pv.PolyData()
        poly.points = np.asarray(points, dtype=float)
        poly.lines = np.asarray(lines, dtype=np.int64)
        tube = poly.tube(radius=radius, n_sides=12)
        plotter.add_mesh(tube, color=color, smooth_shading=True, specular=0.3)

    _add_tubes(intra_edges, _INTRA_EDGE_COLOR, intra_radius)
    _add_tubes(inter_edges, _INTER_EDGE_COLOR, inter_radius)

    for label in labels:
        nodes = node_result.nodes_by_habitat[label]
        if not nodes:
            continue
        pts = np.array([_phys_xyz(n, spacing) for n in nodes])
        cloud = pv.PolyData(pts)
        glyphs = cloud.glyph(geom=pv.Sphere(radius=node_radius), scale=False, orient=False)
        plotter.add_mesh(
            glyphs, color=colors[label], smooth_shading=True,
            specular=0.5, specular_power=20, ambient=0.3, diffuse=0.7,
        )
    plotter.add_axes(color="white" if black_background else "black")
    return _screenshot_rgb(plotter)
