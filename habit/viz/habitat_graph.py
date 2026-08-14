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

All text drawn on the figures is English-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

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
from habit.utils.optional_deps import require
from habit.viz.colorbar import (
    ColorbarSpec,
    DEFAULT_HABITAT_CBAR_LABEL,
    add_discrete_habitat_colorbar,
)
from habit.viz.palette import habitat_hex_colors
from habit.viz.style import use_style

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "plot_habitat_graph_slice",
    "plot_habitat_graph_network_2d",
    "render_habitat_graph_surface_3d",
    "render_habitat_graph_network_3d",
]

#: What habit.viz needs matplotlib for.
_VIZ_PURPOSE = "habitat graph topology figures"
#: What the 3D renderers need PyVista / scikit-image for.
_VIEW_PURPOSE = "3D habitat graph rendering"

#: Accent color for inter-habitat edges; intra-habitat edges use neutral gray.
_INTER_EDGE_COLOR = "#8E44AD"
_INTRA_EDGE_COLOR = "#9AA0A6"
_BACKGROUND_COLOR = "#D9DCE1"


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


def _all_habitats_title(
    labels: Sequence[int],
    nodes_by_habitat: Dict[int, Sequence[HabitatGraphNode]],
    intra_edges: Sequence[_EdgePair],
    inter_edges: Sequence[_EdgePair],
) -> str:
    """
    Build the English All-habitats panel title with node and edge counts.

    Args:
        labels: Sorted habitat IDs present on the drawn slice.
        nodes_by_habitat: Nodes grouped by habitat label.
        intra_edges: Combined intra-habitat edges drawn on the All panel.
        inter_edges: Combined inter-habitat edges drawn on the All panel.

    Returns:
        Multiline title, for example::

            All habitats
            n=60 (H1=12, H2=14, H3=20, H4=14)
            intra e=78, inter e=12

        Per-habitat counts wrap every four habitats so crowded grids stay
        readable. Figure text stays English-only.
    """
    counts = [len(nodes_by_habitat[int(label)]) for label in labels]
    total = int(sum(counts))
    parts = [
        f"H{int(label)}={count}" for label, count in zip(labels, counts)
    ]
    # Wrap every four habitats so an 8-habitat grid does not overflow.
    wrapped: List[str] = []
    for start in range(0, len(parts), 4):
        wrapped.append(", ".join(parts[start : start + 4]))
    if len(wrapped) == 1:
        n_line = f"n={total} ({wrapped[0]})"
    else:
        n_line = f"n={total} ({wrapped[0]},\n  " + ",\n  ".join(wrapped[1:]) + ")"
    return (
        f"All habitats\n"
        f"{n_line}\n"
        f"intra e={len(intra_edges)}, inter e={len(inter_edges)}"
    )


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


#: Matplotlib scatter area (points^2) for every 2D graph node.
_DEFAULT_NODE_SIZE: float = 64.0
#: Publication-readable type sizes for the multi-panel 2D network figure.
_PANEL_TITLE_FONTSIZE: float = 11.5
_AXIS_TEXT_FONTSIZE: float = 10.5
_FIG_LEGEND_FONTSIZE: float = 10.5
#: Other-habitat fill on per-habitat (H1--Hk) panels: opaque gray, not
#: the original colour at a low alpha. Featured habitat stays full
#: colour at alpha 1.0; the All-habitats panel stays fully coloured.
_OTHER_HABITAT_FILL: str = "#C5C8CC"
_OTHER_HABITAT_ALPHA: float = 1.0
#: Intra-habitat edges of non-featured habitats on an H1--Hk panel.
_OTHER_INTRA_EDGE_ALPHA: float = 0.28
_OTHER_INTRA_EDGE_WIDTH: float = 0.55
#: Dashed lattice overlay (same origin / block_size as node extraction).
_GRID_LINE_COLOR: str = "#6B7280"
_GRID_LINE_ALPHA: float = 0.45
_GRID_LINE_WIDTH: float = 0.55
#: Default matplotlib linestyle for the display lattice.
_DEFAULT_GRID_LINESTYLE: str = "--"


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
    featured_label: Optional[int] = None,
) -> None:
    """
    Draw spatial context behind 2D network graphs.

    When ``colors`` is provided, each habitat partition is painted with the
    same palette as the slice figure. Background ``0`` is left transparent
    (not drawn). On a per-habitat panel, pass ``featured_label`` so that
    habitat stays full colour (alpha=1) while every other habitat is
    filled with opaque gray (``_OTHER_HABITAT_FILL``). The All-habitats
    panel omits ``featured_label`` so every habitat stays fully coloured.

    Args:
        ax: Target matplotlib axes.
        label_2d: 2D habitat label map (background encoded as 0).
        colors: Habitat label to hex colour mapping, or ``None`` for a
            gray tissue silhouette.
        show_background: When ``False``, nothing is drawn.
        featured_label: Habitat id to keep opaque on a per-habitat panel.
            ``None`` paints every habitat at alpha 1 (All-habitats panel).
    """
    if not show_background:
        return
    from matplotlib.colors import ListedColormap, to_rgba

    if colors:
        ordered = sorted(colors)
        if not ordered:
            return
        rgba = np.zeros((*label_2d.shape, 4), dtype=float)
        for label in ordered:
            mask = label_2d == label
            if not np.any(mask):
                continue
            if featured_label is None or int(label) == int(featured_label):
                rgba[mask] = to_rgba(colors[label], alpha=1.0)
            else:
                rgba[mask] = to_rgba(_OTHER_HABITAT_FILL, alpha=_OTHER_HABITAT_ALPHA)
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
            ``options.block_size`` (library default 5 voxels).

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
        Caption such as ``\"5-voxel cubes\"``.
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


def _style_axis_2d(
    ax: Any,
    label_2d: np.ndarray,
    title: str,
    *,
    title_fontsize: Optional[float] = None,
    title_pad: Optional[float] = None,
) -> None:
    """
    Apply consistent journal styling to a 2D graph/image axis.

    Args:
        ax: Target matplotlib axes.
        label_2d: Slice used only for axis limits (image coordinates).
        title: English panel title.
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
    ax.set_xlim(-0.5, label_2d.shape[1] - 0.5)
    ax.set_ylim(label_2d.shape[0] - 0.5, -0.5)  # image coordinates
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_edges_2d(
    ax,
    id_to_node: Dict[str, HabitatGraphNode],
    edges: Sequence[_EdgePair],
    color: str,
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
    colors: Dict[int, str],
    node_size: float = _DEFAULT_NODE_SIZE,
) -> None:
    """
    Scatter 2D graph nodes at projected region centroids, colored by habitat.

    Every marker uses the same filled-dot area. Region voxel count is not
    encoded in marker size so small blocks stay as readable as large ones.

    Args:
        ax: Target matplotlib axes.
        nodes: Nodes to draw.
        colors: Habitat label to hex colour mapping.
        node_size: Matplotlib scatter area in points squared. The same
            value is applied to every node.
    """
    if not nodes:
        return
    xs = [_centroid_xy_display(n)[0] for n in nodes]
    ys = [_centroid_xy_display(n)[1] for n in nodes]
    node_colors = [colors.get(int(n.habitat_label), "#444444") for n in nodes]
    ax.scatter(
        xs,
        ys,
        s=float(node_size),
        c=node_colors,
        # Dark rim, not white: a white edge on a small marker reads as a
        # pale / hollow node even when face alpha is 1.0.
        edgecolors="#1A1A1A",
        linewidths=0.5,
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
    ``options.block_size`` (library default 5 voxels), dashed lines.

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

    On each H1--Hk panel the featured habitat's intra-edges stay opaque;
    intra-edges of the other habitats are drawn gray and more transparent
    so the full intra-habitat graphs remain visible as context.

    Args:
        label_array: 2D or 3D habitat label map (background encoded as 0).
        options: Graph construction options shared with the feature extractor.
        slice_index: Explicit axis-0 slice for 3D input; ``None`` selects the
            largest cross-section. Ignored for 2D input.
        show_background: Whether to draw habitat partitions behind the
            graph (default ``True``). On H1--Hk panels the featured
            habitat is full colour and other habitats are opaque gray;
            background 0 stays undrawn. The All-habitats panel stays
            fully coloured. Featured-habitat nodes stay solid.
        show_grid: Draw the uniform-grid lattice (default ``True``).
            Also draws when ``block_size`` is passed in ``component`` mode.
        block_size: Display cube edge in voxels. ``None`` (default) uses
            ``options.block_size`` (library default 5 voxels) so the lattice
            matches the nodes.
        grid_linestyle: Matplotlib line style (default ``\"--\"`` dashed).
        grid_color: Lattice colour.
        grid_alpha: Lattice opacity.
        grid_linewidth: Lattice stroke width in points.
        panel_size: Base panel edge length in inches.
        max_cols: Maximum number of panels per row in the grid.
        node_size: Matplotlib scatter area in points squared applied to
            every node (default ``64``). Voxel count does not scale markers.
        colorbar: Discrete habitat-ID colorbar on the combined-graph panel
            (default ``True``). Pass ``False`` to hide it.
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
    id_to_node, intra_edges, inter_edges = _combined_graph(node_result, options)
    display_size = _display_block_size(options, block_size)
    grid_kwargs = dict(
        show_grid=show_grid,
        block_size=block_size,
        grid_linestyle=grid_linestyle,
        grid_color=grid_color,
        grid_alpha=grid_alpha,
        grid_linewidth=grid_linewidth,
    )

    n_panels = len(labels) + 1
    cols = min(max_cols, max(1, n_panels))
    rows = int(np.ceil(n_panels / cols))
    # Extra width + bottom room so short titles and the larger figlegend
    # cannot collide when many habitats share one row.
    fig_width = cols * panel_size * 1.18
    # Extra height for the All-panel multiline title (n / intra / inter).
    fig_height = rows * panel_size + 1.65
    with use_style("radiology"):
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(fig_width, fig_height),
            squeeze=False,
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(
            w_pad=0.05,
            h_pad=0.12,
            wspace=0.12,
            hspace=0.28,
        )
        flat = [ax for row in axes for ax in row]
        for ax in flat[n_panels:]:
            ax.axis("off")
        axes_list = flat[:n_panels]

        for ax, label in zip(axes_list, labels):
            sub = node_result.nodes_by_habitat[label]
            _draw_background_2d(
                ax, label_2d, colors, show_background, featured_label=int(label)
            )
            _apply_display_grid(
                ax, label_2d, node_result, options, **grid_kwargs
            )
            # Context: other habitats' intra-edges, gray and translucent.
            for other_label in labels:
                if int(other_label) == int(label):
                    continue
                other_nodes = node_result.nodes_by_habitat[other_label]
                other_edges = _single_intra_edges(
                    other_nodes, other_label, options, node_result
                )
                _draw_edges_2d(
                    ax,
                    id_to_node,
                    other_edges,
                    _INTRA_EDGE_COLOR,
                    _OTHER_INTRA_EDGE_WIDTH,
                    _OTHER_INTRA_EDGE_ALPHA,
                    1,
                )
            edges = _single_intra_edges(sub, label, options, node_result)
            # Featured intra-habitat edges stay opaque so the panel is readable.
            _draw_edges_2d(ax, id_to_node, edges, _INTRA_EDGE_COLOR, 0.7, 1.0, 2)
            _draw_nodes_2d(ax, sub, colors, node_size=node_size)
            # Slice index lives on the figure title; keep panel titles short
            # so neighbouring H1 / H2 headings cannot run into each other.
            _style_axis_2d(
                ax,
                label_2d,
                f"H{label} (n={len(sub)}, e={len(edges)})",
                title_fontsize=_PANEL_TITLE_FONTSIZE,
                title_pad=4.0,
            )
        ax_cross = axes_list[-1]
        _draw_background_2d(ax_cross, label_2d, colors, show_background)
        _apply_display_grid(
            ax_cross, label_2d, node_result, options, **grid_kwargs
        )
        _draw_edges_2d(
            ax_cross, id_to_node, intra_edges, _INTRA_EDGE_COLOR, 0.7, 1.0, 2
        )
        _draw_edges_2d(
            ax_cross, id_to_node, inter_edges, _INTER_EDGE_COLOR, 1.1, 1.0, 3
        )
        all_nodes = [
            node for label in labels for node in node_result.nodes_by_habitat[label]
        ]
        _draw_nodes_2d(ax_cross, all_nodes, colors, node_size=node_size)
        _style_axis_2d(
            ax_cross,
            label_2d,
            _all_habitats_title(
                labels,
                node_result.nodes_by_habitat,
                intra_edges,
                inter_edges,
            ),
            title_fontsize=_PANEL_TITLE_FONTSIZE,
            title_pad=6.0,
        )
        cbar = add_discrete_habitat_colorbar(
            ax_cross,
            sorted(colors),
            [colors[label] for label in sorted(colors)],
            colorbar=colorbar,
            label=colorbar_label,
        )
        if cbar is not None:
            cbar.ax.tick_params(labelsize=_AXIS_TEXT_FONTSIZE)
            cbar.ax.yaxis.label.set_size(_AXIS_TEXT_FONTSIZE)
        handles = [
            Line2D(
                [0], [0], color=_INTER_EDGE_COLOR, lw=1.6, label="Inter-habitat edge"
            ),
            Line2D(
                [0], [0], color=_INTRA_EDGE_COLOR, lw=1.2, label="Intra-habitat edge"
            ),
            Line2D(
                [0],
                [0],
                color=_INTRA_EDGE_COLOR,
                lw=1.0,
                alpha=_OTHER_INTRA_EDGE_ALPHA,
                label="Other-habitat edge",
            ),
            Line2D(
                [0],
                [0],
                color=grid_color,
                lw=1.2,
                linestyle=grid_linestyle,
                label=_grid_caption(display_size),
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            fontsize=_FIG_LEGEND_FONTSIZE,
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
