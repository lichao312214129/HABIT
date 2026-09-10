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
"""Teaching diagrams: how data ENTERS HABIT and how features are chained.

Gallery pages that teach an ingestion route (directory tree, SimpleITK
objects, NumPy arrays) or the feature-preprocessing chains get a thumbnail
that shows the ROUTE, not another bare slice: a stylised badge on the left,
an arrow, and a real cropped data panel on the right. The data panel reuses
the habitat-overlay rendering internals, so colours and orientation match
:func:`~habit.viz.plot_habitat_overlay` exactly.

The badges are HABIT-drawn schematics. They are NOT the official SimpleITK
or NumPy logos (trademark); they only evoke a voxel cube / a matrix so the
ingestion route is recognisable at thumbnail size.

Pure functions: contract objects or arrays in, a matplotlib ``Figure`` out,
no filesystem and no ``show``. All text is ASCII (journal-safe).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz._crop import bbox_slices
from habit.viz.habitat_overlay import (
    _blend_overlay,
    _draw_label_contour,
    _habitat_color_lookup,
    _normalize_grey,
    _positive_habitat_ids,
    _slice_index,
    _take_slice,
)
from habit.viz.labels import sanitize_label
from habit.viz.orientation import (
    DEFAULT_DISPLAY_CONVENTION,
    direction_matrix as _parse_direction_matrix,
    imshow_physical_extent,
    orient_slice_for_display,
    resolve_display_geometry,
)
from habit.viz.style import use_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_directory_ingest",
    "plot_simpleitk_ingest",
    "plot_numpy_ingest",
    "plot_feature_preprocessing_chain",
]

#: What habit.viz needs matplotlib for.
_VIZ_PURPOSE = "ingestion / preprocessing teaching diagrams"

#: ROI fill colour, matching the cyan ROI contour used across habit.viz.
_ROI_FILL = "#00E5FF"

#: Directory-tree entry kinds mapped to colours (Okabe-Ito hues).
_TREE_KIND_COLORS = {
    "folder": "#0072B2",  # blue
    "image": "#009E73",  # bluish green
    "mask": "#E69F00",  # orange
}

#: Soft categorical ramp for the preprocessing-chain boxes (Tableau-like).
_CHAIN_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#B07AA1",
)

#: Default directory tree drawn by :func:`plot_directory_ingest`. Each entry
#: is ``(depth, name, kind)`` with kind in ``_TREE_KIND_COLORS``.
_DEFAULT_TREE: Tuple[Tuple[int, str, str], ...] = (
    (0, "DATA/", "folder"),
    (1, "images/", "folder"),
    (2, "subj001/", "folder"),
    (3, "LAP/", "folder"),
    (4, "image.nrrd", "image"),
    (1, "masks/", "folder"),
    (2, "subj001/", "folder"),
    (3, "LAP/", "folder"),
    (4, "mask.nrrd", "mask"),
)


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


def _volume_array(obj: object, *, name: str) -> np.ndarray:
    """
    Coerce a contract object or plain array to a 2D/3D NumPy volume.

    Accepts objects with ``.label_array`` (``HabitatMap`` / supervoxel maps),
    objects with ``.data`` (``ImageVolume`` / ``MaskVolume``), or bare arrays.

    Args:
        obj: Candidate volume-like object.
        name: Name used in error messages.

    Returns:
        Array with ndim in ``{2, 3}``.

    Raises:
        HABITAPIError: When the object cannot be read as a 2D/3D volume.
    """
    if hasattr(obj, "label_array"):
        arr = getattr(obj, "label_array")
    elif hasattr(obj, "data"):
        arr = getattr(obj, "data")
    else:
        arr = obj
    volume = np.asarray(arr)
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = np.squeeze(volume, axis=0)
    if volume.ndim == 4:
        # Multi-channel volumes: average channels for display only.
        volume = np.mean(volume, axis=-1) if volume.shape[-1] <= 4 else volume[0]
    if volume.ndim not in (2, 3):
        raise HABITAPIError(
            f"{name} must be a 2D or 3D volume; got shape {tuple(volume.shape)}."
        )
    if volume.size == 0:
        raise HABITAPIError(f"{name} must not be empty.")
    return volume


def _draw_arrow(ax: "Axes") -> None:
    """Draw a single bold right arrow centred in a spacer axes."""
    ax.axis("off")
    ax.annotate(
        "",
        xy=(0.82, 0.5),
        xytext=(0.18, 0.5),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 3.0,
            "color": "#555555",
            "mutation_scale": 32,
        },
    )


def _draw_data_panel(
    ax: "Axes",
    image: object,
    *,
    labels: Optional[object] = None,
    roi_mask: Optional[object] = None,
    pad: int = 8,
    panel_title: Optional[str] = None,
) -> None:
    """
    Draw one cropped, display-oriented axial data slice on ``ax``.

    With ``labels`` the panel is an opaque habitat-colour overlay identical
    in palette and orientation to :func:`plot_habitat_overlay`; with only
    ``roi_mask`` the ROI is painted as a cyan fill. The view is zoomed to
    the label / ROI bounding box so the tumour fills the panel.

    Args:
        ax: Target axes.
        image: Anatomy volume (array or ``ImageVolume``).
        labels: Optional habitat label map (array or ``HabitatMap``).
        roi_mask: Optional ROI mask (array or ``MaskVolume``).
        pad: Context voxels around the bounding box.
        panel_title: Optional panel title (ASCII-sanitised).

    Raises:
        HABITAPIError: When neither labels nor roi_mask is given, or shapes
            disagree.
    """
    if labels is None and roi_mask is None:
        raise HABITAPIError(
            "diagrams: a data panel needs labels or roi_mask; got neither."
        )
    image_vol = _volume_array(image, name="image")
    label_vol = (
        _volume_array(labels, name="labels").astype(np.int32)
        if labels is not None
        else None
    )
    roi_vol = (
        _volume_array(roi_mask, name="roi_mask") if roi_mask is not None else None
    )
    target = label_vol if label_vol is not None else roi_vol
    if target is None or image_vol.shape != target.shape:
        raise HABITAPIError(
            "diagrams: image and labels/roi_mask must share the same shape; "
            f"got image {image_vol.shape} vs {None if target is None else target.shape}."
        )

    # Zoom to the foreground bounding box, then pick the densest axial slice
    # inside the cropped volume (same rule as the main overlay plotter).
    crop = bbox_slices(target, pad, caller="diagrams", mask_name="labels/roi_mask")
    image_vol = image_vol[crop]
    if label_vol is not None:
        label_vol = label_vol[crop]
    if roi_vol is not None:
        roi_vol = roi_vol[crop]
    target = label_vol if label_vol is not None else roi_vol

    resolved_direction, resolved_spacing = resolve_display_geometry(
        image, labels if labels is not None else roi_mask
    )
    direction = _parse_direction_matrix(resolved_direction, ndim=image_vol.ndim)
    spacing_xyz = (
        tuple(1.0 for _ in range(image_vol.ndim))
        if resolved_spacing is None
        else tuple(float(v) for v in resolved_spacing)
    )

    axis_id = 0
    slice_index = _slice_index(target, axis_id, None)
    grey = _normalize_grey(_take_slice(image_vol, axis_id, slice_index))
    mask_2d = _take_slice(target, axis_id, slice_index)
    grey = orient_slice_for_display(
        grey, slice_axis=axis_id, direction=direction
    )
    mask_2d = orient_slice_for_display(
        mask_2d, slice_axis=axis_id, direction=direction
    )

    if label_vol is not None:
        # Volume-level ID->colour so the panel matches multi-panel overlays.
        habitat_ids = _positive_habitat_ids(label_vol)
        id_to_color = _habitat_color_lookup(habitat_ids)
        rgb = _blend_overlay(
            grey,
            mask_2d,
            alpha=1.0,
            colors=(),
            id_to_color=id_to_color,
        )
    else:
        rgb = np.stack([grey, grey, grey], axis=-1)
        fill = np.array(
            (
                int(_ROI_FILL[1:3], 16) / 255.0,
                int(_ROI_FILL[3:5], 16) / 255.0,
                int(_ROI_FILL[5:7], 16) / 255.0,
            )
        )
        inside = mask_2d > 0
        for channel in range(3):
            plane = rgb[..., channel]
            plane[inside] = 0.2 * plane[inside] + 0.8 * fill[channel]
            rgb[..., channel] = plane

    extent = imshow_physical_extent(
        (int(rgb.shape[0]), int(rgb.shape[1])),
        spacing_xyz,
        slice_axis=axis_id,
        ndim=image_vol.ndim,
        direction=direction,
        convention=DEFAULT_DISPLAY_CONVENTION,
    )
    ax.imshow(
        rgb,
        interpolation="nearest",
        origin="upper",
        extent=extent,
        aspect="equal",
    )
    _draw_label_contour(ax, mask_2d, extent=extent)
    ax.set_aspect("equal", adjustable="box")
    if panel_title:
        ax.set_title(sanitize_label(panel_title))
    ax.axis("off")


def _draw_directory_tree(
    ax: "Axes", tree: Sequence[Tuple[int, str, str]]
) -> None:
    """
    Draw a colour-coded directory tree on ``ax`` (monospace, ASCII only).

    Args:
        ax: Target axes.
        tree: ``(depth, name, kind)`` entries; kind is one of
            ``_TREE_KIND_COLORS`` (folder / image / mask).
    """
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    n_lines = len(tree)
    line_height = 0.86 / max(n_lines, 1)
    y_top = 0.94
    for row, (depth, name, kind) in enumerate(tree):
        y = y_top - row * line_height
        x = 0.04 + 0.09 * depth
        color = _TREE_KIND_COLORS.get(kind, "#666666")
        # Colour swatch in front of every entry so kinds survive greyscale
        # printing and colour-vision deficiency.
        ax.add_patch(
            _rectangle((x, y - 0.028), 0.035, 0.05, color)
        )
        ax.text(
            x + 0.055,
            y,
            sanitize_label(name),
            family="monospace",
            fontsize=10.5,
            color="#222222",
            va="center",
            ha="left",
        )


def _rectangle(xy: Tuple[float, float], width: float, height: float, color: str):
    """Build one filled Rectangle patch (kept small for tree swatches)."""
    from matplotlib.patches import Rectangle

    return Rectangle(xy, width, height, facecolor=color, edgecolor="none")


def _draw_simpleitk_badge(ax: "Axes") -> None:
    """
    Draw a stylised voxel-cube badge with a ``SimpleITK`` caption on ``ax``.

    HABIT-drawn schematic (an isometric gridded cube evoking an ITK image
    object); NOT the official SimpleITK logo.
    """
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    # Front face square and the depth offset that makes it a cube.
    x0, y0, side = 0.20, 0.30, 0.40
    dx, dy = 0.14, 0.14
    front = [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)]
    back = [(x + dx, y + dy) for x, y in front]
    faces = (
        # (polygon vertices, face colour) — paint back-to-front.
        ([front[1], back[1], back[2], front[2]], "#6BAED6"),  # right side
        ([front[2], back[2], back[3], front[3]], "#C6DBEF"),  # top
        (front, "#9ECAE1"),  # front
    )
    from matplotlib.patches import Polygon

    for vertices, color in faces:
        ax.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor=color,
                edgecolor="#2171B5",
                linewidth=1.6,
            )
        )
    # Voxel lattice on the front face: 4x4 grid lines.
    for step in range(1, 4):
        frac = step / 4.0
        ax.plot(
            [x0 + side * frac, x0 + side * frac],
            [y0, y0 + side],
            color="#2171B5",
            linewidth=0.8,
            alpha=0.75,
        )
        ax.plot(
            [x0, x0 + side],
            [y0 + side * frac, y0 + side * frac],
            color="#2171B5",
            linewidth=0.8,
            alpha=0.75,
        )
    ax.text(
        0.5,
        0.16,
        "SimpleITK",
        fontsize=14,
        fontweight="bold",
        color="#08519C",
        ha="center",
        va="center",
    )
    ax.text(
        0.5,
        0.06,
        "sitk.Image",
        family="monospace",
        fontsize=9,
        color="#555555",
        ha="center",
        va="center",
    )


def _downsample_to_matrix(slice_2d: np.ndarray, cells: int = 10) -> np.ndarray:
    """
    Block-average a 2D slice into a small ``cells x cells`` display matrix.

    Args:
        slice_2d: Cropped 2D slice (already display-oriented).
        cells: Number of rows/columns of the badge matrix.

    Returns:
        ``(cells, cells)`` float array for ``imshow``.
    """
    data = np.asarray(slice_2d, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros((cells, cells))
    data = np.nan_to_num(data, nan=float(np.min(finite)))
    # Pad to a whole number of blocks, then average each block.
    pad_rows = (-data.shape[0]) % cells
    pad_cols = (-data.shape[1]) % cells
    padded = np.pad(
        data,
        ((0, pad_rows), (0, pad_cols)),
        mode="edge",
    )
    block_rows = padded.shape[0] // cells
    block_cols = padded.shape[1] // cells
    return padded.reshape(cells, block_rows, cells, block_cols).mean(axis=(1, 3))


def _draw_numpy_badge(ax: "Axes", matrix: np.ndarray) -> None:
    """
    Draw the NumPy badge on ``ax``: the data itself as a small colour matrix.

    Args:
        ax: Target axes.
        matrix: Small 2D array (e.g. from :func:`_downsample_to_matrix`).
    """
    ax.set_title("")
    ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    # White cell borders make the matrix read as an array, not an image.
    n_rows, n_cols = matrix.shape
    for row in range(n_rows + 1):
        ax.axhline(row - 0.5, color="white", linewidth=0.6, alpha=0.7)
    for col in range(n_cols + 1):
        ax.axvline(col - 0.5, color="white", linewidth=0.6, alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("NumPy ndarray  (z, y, x)", fontsize=11, fontweight="bold")


def _draw_chain(ax: "Axes", steps: Sequence[str]) -> None:
    """
    Draw the feature-preprocessing chain as coloured boxes joined by arrows.

    Args:
        ax: Target axes.
        steps: Step names in pipeline order (e.g. ``("raw", "winsorize",
            "minmax", "zscore", "binning")``).
    """
    from matplotlib.patches import FancyBboxPatch

    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    n_steps = len(steps)
    gap = 0.045
    box_width = (1.0 - gap * (n_steps - 1)) / n_steps
    box_height = 0.34
    y0 = 0.5 - box_height / 2.0
    fontsize = 10.0 if n_steps <= 4 else 8.5
    for index, step in enumerate(steps):
        x0 = index * (box_width + gap)
        color = _CHAIN_COLORS[index % len(_CHAIN_COLORS)]
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                box_width,
                box_height,
                boxstyle="round,pad=0.012",
                facecolor=color,
                edgecolor="#333333",
                linewidth=0.8,
            )
        )
        ax.text(
            x0 + box_width / 2.0,
            0.5,
            sanitize_label(str(step)),
            fontsize=fontsize,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )
        if index < n_steps - 1:
            ax.annotate(
                "",
                xy=(x0 + box_width + gap * 0.85, 0.5),
                xytext=(x0 + box_width + gap * 0.15, 0.5),
                xycoords="axes fraction",
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 1.6,
                    "color": "#555555",
                },
            )


def _ingest_figure(
    badge,
    image: object,
    *,
    labels: Optional[object] = None,
    roi_mask: Optional[object] = None,
    title: str,
    panel_title: str,
    crop_pad: int,
    badge_ratio: float = 1.0,
) -> "Figure":
    """
    Shared three-column layout: badge | arrow | cropped data panel.

    Args:
        badge: Callable drawing the left-hand badge on an axes.
        image: Anatomy volume for the right-hand data panel.
        labels: Optional habitat map for the data panel.
        roi_mask: Optional ROI mask for the data panel.
        title: Figure title (ASCII-sanitised); drawn as a small caption
            under the badge so it cannot collide with the panel title.
        panel_title: Data-panel title (ASCII-sanitised).
        crop_pad: Context voxels around the foreground bounding box.
        badge_ratio: Width ratio of the badge column relative to the data
            column (which is fixed at ratio 1.15).

    Returns:
        A matplotlib ``Figure``.
    """
    plt = _plt()
    with use_style("radiology"):
        fig = plt.figure(figsize=(7.2, 3.5), constrained_layout=True)
        grid = fig.add_gridspec(
            1, 3, width_ratios=[badge_ratio, 0.18, 1.15]
        )
        ax_badge = fig.add_subplot(grid[0, 0])
        ax_arrow = fig.add_subplot(grid[0, 1])
        ax_data = fig.add_subplot(grid[0, 2])
        badge(ax_badge)
        # Caption under the badge doubles as the figure title: a suptitle
        # would sit on top of the data-panel title at this compact height.
        ax_badge.set_title(sanitize_label(title), fontsize=10, pad=10)
        _draw_arrow(ax_arrow)
        _draw_data_panel(
            ax_data,
            image,
            labels=labels,
            roi_mask=roi_mask,
            pad=crop_pad,
            panel_title=panel_title,
        )
    return fig


def plot_directory_ingest(
    image: object,
    roi_mask: object,
    *,
    tree: Optional[Sequence[Tuple[int, str, str]]] = None,
    title: str = "From directory to Subject",
    panel_title: str = "Subject anatomy + ROI",
    crop_pad: int = 8,
) -> "Figure":
    """
    Diagram for the directory-ingestion route: tree -> Subject with ROI.

    Args:
        image: Anatomy volume (array or ``ImageVolume``) of one subject.
        roi_mask: ROI mask (array or ``MaskVolume``), same grid as ``image``.
        tree: Optional ``(depth, name, kind)`` entries overriding the default
            preprocessed-tree schematic; kind is ``"folder"`` / ``"image"`` /
            ``"mask"``.
        title: Figure title (ASCII-sanitised).
        panel_title: Title of the cropped data panel.
        crop_pad: Context voxels around the ROI bounding box (default ``8``).

    Returns:
        A matplotlib ``Figure`` with the directory tree on the left and the
        cropped anatomy + cyan ROI fill on the right.
    """
    entries = _DEFAULT_TREE if tree is None else tuple(tree)
    return _ingest_figure(
        lambda ax: _draw_directory_tree(ax, entries),
        image,
        roi_mask=roi_mask,
        title=title,
        panel_title=panel_title,
        crop_pad=crop_pad,
    )


def plot_simpleitk_ingest(
    image: object,
    labels: object,
    *,
    title: str = "From SimpleITK to habitats",
    panel_title: str = "Habitats",
    crop_pad: int = 8,
) -> "Figure":
    """
    Diagram for the SimpleITK route: voxel-cube badge -> habitat overlay.

    Args:
        image: Anatomy volume (array or ``ImageVolume``) of one subject.
        labels: Habitat label map (array or ``HabitatMap``) on the same grid.
        title: Figure title (ASCII-sanitised).
        panel_title: Title of the cropped data panel.
        crop_pad: Context voxels around the habitat bounding box
            (default ``8``).

    Returns:
        A matplotlib ``Figure`` with the stylised SimpleITK badge on the
        left and the cropped habitat overlay on the right.
    """
    return _ingest_figure(
        _draw_simpleitk_badge,
        image,
        labels=labels,
        title=title,
        panel_title=panel_title,
        crop_pad=crop_pad,
    )


def plot_numpy_ingest(
    image: object,
    roi_mask: Optional[object] = None,
    *,
    labels: Optional[object] = None,
    title: str = "From NumPy arrays to HABIT",
    panel_title: str = "Subject anatomy + ROI",
    crop_pad: int = 8,
    matrix_cells: int = 10,
) -> "Figure":
    """
    Diagram for the array route: the data itself drawn as a colour matrix.

    The badge matrix is a block-averaged miniature of the cropped ROI slice
    of ``image`` -- the badge IS the data, not a stock icon.

    Args:
        image: Anatomy volume (array or ``ImageVolume``) of one subject.
        roi_mask: Optional ROI mask (array or ``MaskVolume``); painted as a
            cyan fill on the data panel when ``labels`` is not given.
        labels: Optional habitat label map (array or ``HabitatMap``); when
            given, the data panel shows the habitat overlay instead.
        title: Figure title (ASCII-sanitised).
        panel_title: Title of the cropped data panel.
        crop_pad: Context voxels around the foreground bounding box
            (default ``8``).
        matrix_cells: Rows/columns of the badge matrix (default ``10``).

    Returns:
        A matplotlib ``Figure`` with the matrix badge on the left and the
        cropped data panel on the right.

    Raises:
        HABITAPIError: When neither ``roi_mask`` nor ``labels`` is given.
    """
    if roi_mask is None and labels is None:
        raise HABITAPIError(
            "plot_numpy_ingest: pass roi_mask or labels so the data panel "
            "has a foreground to zoom to."
        )
    # Build the badge matrix from the same cropped slice the panel shows.
    image_vol = _volume_array(image, name="image")
    target = _volume_array(
        labels if labels is not None else roi_mask, name="labels/roi_mask"
    )
    crop = bbox_slices(
        target, crop_pad, caller="plot_numpy_ingest", mask_name="labels/roi_mask"
    )
    cropped_image = image_vol[crop]
    cropped_target = target[crop]
    slice_index = _slice_index(cropped_target, 0, None)
    matrix = _downsample_to_matrix(
        _take_slice(cropped_image, 0, slice_index), cells=matrix_cells
    )
    return _ingest_figure(
        lambda ax: _draw_numpy_badge(ax, matrix),
        image,
        labels=labels,
        roi_mask=roi_mask,
        title=title,
        panel_title=panel_title,
        crop_pad=crop_pad,
    )


def plot_feature_preprocessing_chain(
    steps: Sequence[str],
    image: Optional[object] = None,
    labels: Optional[object] = None,
    *,
    title: str = "Feature preprocessing",
    panel_title: str = "Habitats after the chain",
    crop_pad: int = 8,
) -> "Figure":
    """
    Diagram of the clustering-feature preprocessing chain.

    Args:
        steps: Step names in pipeline order, e.g. ``("raw", "winsorize",
            "minmax", "zscore", "binning")``.
        image: Optional anatomy volume; with ``labels`` the right-hand panel
            shows the cropped habitat map produced after the chain.
        labels: Optional habitat label map (array or ``HabitatMap``).
        title: Figure title (ASCII-sanitised).
        panel_title: Title of the cropped data panel.
        crop_pad: Context voxels around the habitat bounding box
            (default ``8``).

    Returns:
        A matplotlib ``Figure``: the chain of step boxes, an arrow, and --
        when ``image`` and ``labels`` are given -- the resulting habitats.

    Raises:
        HABITAPIError: When ``steps`` is empty or only one of
            ``image`` / ``labels`` is given.
    """
    if not steps:
        raise HABITAPIError(
            "plot_feature_preprocessing_chain: steps must not be empty."
        )
    if (image is None) != (labels is None):
        raise HABITAPIError(
            "plot_feature_preprocessing_chain: image and labels must be "
            "given together (the data panel needs both)."
        )
    plt = _plt()
    with use_style("radiology"):
        if image is None:
            # Chain-only figure: one wide axes, no data panel.
            fig, ax = plt.subplots(
                1, 1, figsize=(7.2, 1.6), constrained_layout=True
            )
            _draw_chain(ax, steps)
            fig.suptitle(sanitize_label(title))
            return fig
        return _ingest_figure(
            lambda ax: _draw_chain(ax, steps),
            image,
            labels=labels,
            title=title,
            panel_title=panel_title,
            crop_pad=crop_pad,
            badge_ratio=1.6,
        )
