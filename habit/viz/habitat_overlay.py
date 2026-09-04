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
"""Habitat label maps drawn on top of the source image.

Pure functions: image and label arrays in, a matplotlib ``Figure`` out, no
filesystem and no ``show``. Background label ``0`` stays greyscale anatomy;
habitat IDs ``>= 1`` are painted with a colour-blind-friendly categorical
palette (opaque by default). Pass ``alpha<1`` only for an explicit blend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz.colorbar import (
    ColorbarSpec,
    DEFAULT_HABITAT_CBAR_LABEL,
    add_discrete_habitat_colorbar,
)
from habit.viz.palette import habitat_rgb_colors
from habit.viz.labels import sanitize_label
from habit.viz.style import use_style
from habit.viz.orientation import (
    DEFAULT_DISPLAY_CONVENTION,
    DEFAULT_NATIVE_DIRECTION,
    DEFAULT_RAS_DIRECTION,
    DisplayConvention,
    array_axis_lps_direction,
    array_from_display_input,
    desired_screen_directions,
    direction_matrix as _parse_direction_matrix,
    imshow_physical_extent,
    normalize_display_convention,
    orient_slice_for_display,
    plane_spacings_mm,
    resolve_display_geometry,
    slice_row_col_axes,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_habitat_overlay"]

# Re-export under the historical private names so existing unit tests keep working.
_DEFAULT_RAS_DIRECTION = DEFAULT_RAS_DIRECTION
_DEFAULT_NATIVE_DIRECTION = DEFAULT_NATIVE_DIRECTION


#: What habit.viz needs matplotlib for.
_VIZ_PURPOSE = "habitat overlay figures (source image + habitat labels)"

#: Designed habitat bank as RGB (Okabe–Ito + Tol extras). Prefer
#: :func:`habitat_rgb_colors` so K>8 does not wrap to a duplicate.
_HABITAT_COLORS: Tuple[Tuple[float, float, float], ...] = tuple(
    habitat_rgb_colors(16)
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


def _as_volume(array: object, name: str) -> np.ndarray:
    """
    Coerce ``array`` to a 2D or 3D float/int volume (drop singleton leading axes).

    Accepts a NumPy array, :class:`~habit.api.image.ImageVolume` (``.data``),
    or a habitat / supervoxel map (``.label_array``).

    Args:
        array: Candidate image or label array / volume object.
        name: Name used in error messages.

    Returns:
        Array with ndim in ``{2, 3}``.

    Raises:
        HABITAPIError: When the array cannot be interpreted as a volume.
    """
    volume = array_from_display_input(array)
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = np.squeeze(volume, axis=0)
    if volume.ndim == 4:
        # Multi-channel volumes: average channels for display only.
        volume = np.mean(volume, axis=-1) if volume.shape[-1] <= 4 else volume[0]
    if volume.ndim not in (2, 3):
        raise HABITAPIError(
            f"plot_habitat_overlay: {name} must be 2D or 3D after squeeze; "
            f"got shape {tuple(volume.shape)}."
        )
    if volume.size == 0:
        raise HABITAPIError(f"plot_habitat_overlay: {name} must not be empty.")
    return volume


def _normalize_grey(slice_2d: np.ndarray) -> np.ndarray:
    """
    Scale a 2D slice to ``[0, 1]`` for display using robust percentiles.

    Args:
        slice_2d: Single greyscale slice.

    Returns:
        Float32 array in ``[0, 1]``.
    """
    data = np.asarray(slice_2d, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.float32)
    low, high = np.percentile(finite, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high <= low:
            return np.zeros(data.shape, dtype=np.float32)
    scaled = (data - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _draw_label_contour(
    ax,
    labels: np.ndarray,
    *,
    extent: Tuple[float, float, float, float],
    color: str = "#00E5FF",
    linewidth: float = 1.35,
) -> None:
    """
    Outline non-background habitat voxels (label ``> 0``).

    Args:
        ax: Matplotlib axes already showing the overlay.
        labels: 2D integer label map (already display-oriented).
        extent: Same physical ``imshow`` extent as the underlay.
        color: Contour colour (default cyan; English figures only).
        linewidth: Contour line width in points.
    """
    binary = (np.asarray(labels) > 0).astype(np.float64)
    if not np.any(binary):
        return
    ax.contour(
        binary,
        levels=[0.5],
        colors=[color],
        linewidths=float(linewidth),
        origin="upper",
        extent=extent,
    )


def _positive_habitat_ids(labels: np.ndarray) -> List[int]:
    """
    Return sorted unique habitat IDs, excluding background ``0``.

    Args:
        labels: Integer label array (any shape).

    Returns:
        Sorted positive integer IDs.
    """
    return sorted(
        {int(value) for value in np.unique(np.asarray(labels)) if int(value) > 0}
    )


def _habitat_color_lookup(
    habitat_ids: Sequence[int],
    colors: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Map each habitat ID to a stable RGB triple.

    Colour index follows the sorted-ID order so every panel that shares
    the same ID list paints habitat ``k`` with the same colour (a slice
    that happens to miss an ID does not re-index the palette).

    Args:
        habitat_ids: Positive integer habitat IDs (already unique).
        colors: Optional RGB triples. When omitted, HABIT assigns one
            distinct colour per ID from the Radiology-safe bank (no
            silent 8-colour wrap). A caller-supplied list still cycles
            if it is shorter than the ID list.

    Returns:
        ``habitat_id → (r, g, b)`` in ``[0, 1]``.
    """
    ordered = [int(habitat_id) for habitat_id in habitat_ids]
    if not ordered:
        return {}
    if colors is None:
        face = habitat_rgb_colors(len(ordered))
    else:
        bank = list(colors)
        if not bank:
            face = habitat_rgb_colors(len(ordered))
        elif len(bank) >= len(ordered):
            face = [bank[index] for index in range(len(ordered))]
        else:
            # Caller chose a short custom list; cycling is explicit.
            face = [bank[index % len(bank)] for index in range(len(ordered))]
    return {
        habitat_id: face[index] for index, habitat_id in enumerate(ordered)
    }


def _habitat_color_list(
    habitat_ids: Sequence[int],
    colors: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> List[Tuple[float, float, float]]:
    """Return palette colours aligned with ``habitat_ids`` (for the colorbar)."""
    lookup = _habitat_color_lookup(habitat_ids, colors)
    return [lookup[int(habitat_id)] for habitat_id in habitat_ids]


def _blend_overlay(
    grey: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
    colors: Sequence[Tuple[float, float, float]],
    id_to_color: Optional[Mapping[int, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """
    Paint habitat colours onto a greyscale slice (label 0 stays anatomy).

    ``alpha=1`` replaces habitat voxels (opaque). Values in ``(0, 1)`` blend
    as an explicit option. Default callers pass ``alpha=1.0``.

    Args:
        grey: 2D float array in ``[0, 1]``.
        labels: 2D integer label map, same shape as ``grey``.
        alpha: Opacity of habitat colours in ``(0, 1]``.
        colors: RGB triples used when ``id_to_color`` is omitted.
        id_to_color: Optional ID-keyed RGB map (volume-level, so orthogonal
            slices share colours). When omitted, colours are assigned from
            the IDs present on this slice.

    Returns:
        RGB float array of shape ``(H, W, 3)`` in ``[0, 1]``.
    """
    rgb = np.stack([grey, grey, grey], axis=-1)
    overlay = rgb.copy()
    habitat_ids = _positive_habitat_ids(labels)
    lookup = (
        dict(id_to_color)
        if id_to_color is not None
        else _habitat_color_lookup(habitat_ids, colors)
    )
    for habitat_id in habitat_ids:
        mask = labels == habitat_id
        if not np.any(mask):
            continue
        color = lookup.get(habitat_id)
        if color is None:
            color = colors[(habitat_id - 1) % len(colors)]
        for channel, value in enumerate(color):
            channel_plane = overlay[..., channel]
            channel_plane[mask] = (1.0 - alpha) * channel_plane[mask] + alpha * value
            overlay[..., channel] = channel_plane
    return np.clip(overlay, 0.0, 1.0)


def _slice_index(
    labels: np.ndarray,
    axis: int,
    index: Optional[int],
) -> int:
    """
    Return a valid slice index along ``axis``.

    When ``index`` is omitted, pick the slice with the most non-background
    habitat voxels. Tumours are often off-centre, so a geometric mid-slice
    frequently shows no overlay at all; the densest-label slice is what a
    user expects from ``habit view``.

    Args:
        labels: Integer label volume (2D or 3D).
        axis: Axis along which to choose the slice.
        index: Explicit slice index, or ``None`` for auto selection.

    Returns:
        Slice index in ``[0, length)``.
    """
    if labels.ndim == 2:
        length = 1
    else:
        length = int(labels.shape[axis])
    if length <= 0:
        raise HABITAPIError("plot_habitat_overlay: volume axis length must be > 0.")
    if index is not None:
        if index < 0 or index >= length:
            raise HABITAPIError(
                f"plot_habitat_overlay: slice index {index} is out of range "
                f"for axis length {length}."
            )
        return int(index)
    if labels.ndim == 2 or length == 1:
        return 0
    other_axes = tuple(i for i in range(labels.ndim) if i != axis)
    counts = np.sum(np.asarray(labels) > 0, axis=other_axes)
    if int(np.max(counts)) == 0:
        return length // 2
    return int(np.argmax(counts))


def _take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract a 2D slice from a 2D/3D volume."""
    if volume.ndim == 2:
        return volume
    return np.take(volume, index, axis=axis)


def _direction_matrix(
    direction: Optional[Sequence[float]],
    *,
    ndim: int,
) -> Optional[np.ndarray]:
    """Parse SimpleITK direction; wrap shared helper with overlay-prefixed errors."""
    try:
        return _parse_direction_matrix(direction, ndim=ndim)
    except HABITAPIError as exc:
        raise HABITAPIError(f"plot_habitat_overlay: {exc}") from exc


def _array_axis_lps_direction(direction: np.ndarray, array_axis: int) -> np.ndarray:
    """LPS-world unit direction of increasing ``array_axis`` (see orientation)."""
    return array_axis_lps_direction(direction, array_axis)


def _slice_row_col_axes(slice_axis: int) -> Tuple[int, int]:
    """Return ``(row_array_axis, col_array_axis)`` for a ``np.take`` plane."""
    return slice_row_col_axes(slice_axis)


def _spacing_xyz(
    spacing: Optional[Sequence[float]],
    *,
    ndim: int,
) -> Tuple[float, ...]:
    """
    Parse SimpleITK spacing ``(x, y[, z])``; default to isotropic 1 mm.

    Args:
        spacing: Physical voxel sizes in SimpleITK axis order, or ``None``.
        ndim: Array dimensionality (2 or 3).

    Returns:
        Spacing tuple of length ``ndim``.

    Raises:
        HABITAPIError: When length or values are invalid.
    """
    if spacing is None:
        return tuple(1.0 for _ in range(ndim))
    values = tuple(float(v) for v in spacing)
    if len(values) != ndim:
        raise HABITAPIError(
            f"plot_habitat_overlay: spacing must have {ndim} values "
            f"(SimpleITK x,y[,z]); got {len(values)}."
        )
    if any(not np.isfinite(v) or v <= 0.0 for v in values):
        raise HABITAPIError(
            "plot_habitat_overlay: spacing values must be finite and > 0."
        )
    return values


def _array_axis_spacing(spacing_xyz: Sequence[float], array_axis: int) -> float:
    """
    Physical size along a NumPy ``(z, y, x)`` array axis.

    SimpleITK spacing is ``(x, y, z)`` while ``ImageVolume.data`` is
    ``(z, y, x)``, so array axis ``0/1/2`` maps to spacing index ``2/1/0``.
    """
    sitk_axis = (2, 1, 0)[int(array_axis)]
    return float(spacing_xyz[sitk_axis])


def _imshow_aspect(
    spacing_xyz: Sequence[float],
    *,
    slice_axis: int,
    ndim: int,
    direction: Optional[np.ndarray] = None,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> float:
    """
    Matplotlib data-unit aspect: ``(physical size of one row) / (one column)``.

    This is ``spacing_along_row / spacing_along_col`` after display orientation
    (including an SI transpose when superior lies along the extract columns).
    Prefer :func:`_imshow_physical_extent` + ``aspect='equal'`` for drawing so
    layout code cannot silently re-square anisotropic voxels.
    """
    spacing_row, spacing_col = plane_spacings_mm(
        spacing_xyz,
        slice_axis=slice_axis,
        ndim=ndim,
        direction=direction,
        convention=convention,
    )
    return spacing_row / spacing_col


def _plane_spacings(
    spacing_xyz: Sequence[float],
    *,
    slice_axis: int,
    ndim: int,
    direction: Optional[np.ndarray] = None,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> Tuple[float, float]:
    """Return ``(spacing_row_mm, spacing_col_mm)`` for a display plane."""
    return plane_spacings_mm(
        spacing_xyz,
        slice_axis=slice_axis,
        ndim=ndim,
        direction=direction,
        convention=convention,
    )


def _imshow_physical_extent(
    shape_hw: Tuple[int, int],
    spacing_xyz: Sequence[float],
    *,
    slice_axis: int,
    ndim: int,
    direction: Optional[np.ndarray] = None,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> Tuple[float, float, float, float]:
    """
    ``imshow`` extent in millimetres so ``aspect='equal'`` is physically true.

    Delegates to :func:`habit.viz.orientation.imshow_physical_extent`
    (non-inverted ylim; see that docstring for the coronal/sagittal flip).
    """
    try:
        return imshow_physical_extent(
            shape_hw,
            spacing_xyz,
            slice_axis=slice_axis,
            ndim=ndim,
            direction=direction,
            convention=convention,
        )
    except HABITAPIError as exc:
        raise HABITAPIError(f"plot_habitat_overlay: {exc}") from exc


def _desired_screen_directions(
    slice_axis: int,
    *,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> Tuple[np.ndarray, np.ndarray]:
    """Screen ``(up, left)`` LPS directions for ``convention`` (see orientation)."""
    return desired_screen_directions(slice_axis, convention=convention)


def _orient_slice_for_display(
    slice_2d: np.ndarray,
    *,
    slice_axis: int,
    direction: Optional[np.ndarray],
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> np.ndarray:
    """Flip a 2D slice for matplotlib ``imshow`` under ``convention``."""
    return orient_slice_for_display(
        slice_2d,
        slice_axis=slice_axis,
        direction=direction,
        convention=convention,
    )


def _prepare_overlay_slice(
    image_vol: np.ndarray,
    label_int: np.ndarray,
    *,
    axis_id: int,
    slice_index: int,
    alpha: float,
    direction: Optional[np.ndarray],
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
    id_to_color: Optional[Mapping[int, Tuple[float, float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize, orient, and paint one orthogonal slice; return RGB + labels."""
    grey = _normalize_grey(_take_slice(image_vol, axis_id, slice_index))
    labs = _take_slice(label_int, axis_id, slice_index)
    grey = _orient_slice_for_display(
        grey, slice_axis=axis_id, direction=direction, convention=convention
    )
    labs = _orient_slice_for_display(
        labs, slice_axis=axis_id, direction=direction, convention=convention
    )
    rgb = _blend_overlay(
        grey,
        labs,
        alpha=float(alpha),
        colors=_HABITAT_COLORS,
        id_to_color=id_to_color,
    )
    return rgb, labs


def plot_habitat_overlay(
    image: object,
    labels: object,
    *,
    alpha: float = 1.0,
    title: Optional[str] = None,
    axis: Optional[int] = None,
    index: Optional[int] = None,
    direction: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
    display_convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
    contour: bool = True,
    colorbar: ColorbarSpec = True,
    colorbar_label: str = DEFAULT_HABITAT_CBAR_LABEL,
) -> "Figure":
    """
    Draw habitat labels as an opaque colour overlay on the source image.

    For 3D volumes the default is a three-panel figure (orthogonal slices in
    NumPy axis order ``0 / 1 / 2``, i.e. SimpleITK ``(z, y, x)``). Each panel
    uses the slice with the most non-background habitat voxels so the overlay
    is visible even when the tumour is off-centre. Pass ``axis`` / ``index`` to
    pin a specific slice. Label ``0`` is treated as background and is not
    coloured.

    Slices are oriented using ``direction`` (SimpleITK flattened 3x3) and
    ``display_convention`` (default ``\"radiological\"``). When ``direction``
    is omitted, HABIT reads it from an ``ImageVolume`` / ``HabitatMap`` if
    you pass those objects rather than bare arrays; otherwise LPS identity
    is assumed — the same default as
    :class:`~habit.api.image.ImageVolume` — not RAS.

    Panel aspect ratios follow ``spacing`` (SimpleITK ``(x, y, z)``) so thick
    slices are not squashed into square pixels on coronal / sagittal views.
    Pass the volume object (not ``.data``) so coronal/sagittal superior-up
    and left-right match ITK-SNAP / 3D Slicer. Override with
    ``display_convention=\"native\"`` to skip display flips, or
    ``\"neurological\"`` for patient-left on the viewer's left.

    Args:
        image: Source image array (2D or 3D; SimpleITK/NumPy ``(z, y, x)``
            order) or an :class:`~habit.api.image.ImageVolume`.
        labels: Habitat label map with the same shape as ``image``, or a
            :class:`~habit.contracts.habitat.HabitatMap`.
        alpha: Habitat colour opacity (default ``1.0`` = opaque inside
            habitat voxels; anatomy stays grey outside). Use ``(0, 1)``
            only for an explicit translucent blend.
        contour: When True, outline non-background habitat voxels.
        title: Optional figure title (ASCII-sanitised).
        axis: If set, draw only this axis (``0``, ``1``, or ``2``).
        index: Slice index along ``axis``; densest habitat slice when omitted.
        direction: Optional SimpleITK direction cosines (9 floats). Same layout
            as ``ImageVolume.direction``. Controls anterior/posterior,
            superior/inferior, and left/right flips per panel. Inferred from
            ``image`` / ``labels`` when omitted.
        spacing: Optional SimpleITK voxel spacing ``(x, y[, z])`` in mm. Same
            layout as ``ImageVolume.spacing``. Controls true physical aspect
            per panel; inferred from the volume object, else isotropic ``1.0``.
        display_convention: ``\"radiological\"`` (default), ``\"neurological\"``,
            or ``\"native\"`` (no display flips). See
            :mod:`habit.viz.orientation`.
        colorbar: Draw a discrete habitat-ID colorbar (default ``True``).
            One tick / colour per positive ID; background ``0`` is omitted.
            Pass ``False`` to hide it, or a mapping of colorbar style
            kwargs (``shrink``, ``pad``, ``fraction``, ``aspect``,
            ``label``, ...).
        colorbar_label: Colorbar label (English default ``\"Habitat\"``).

    Returns:
        A matplotlib ``Figure``. The caller owns persistence / display.

    Raises:
        HABITAPIError: On shape / parameter errors.
        OptionalDependencyError: When matplotlib is not installed.

    See Also
    --------
    habit.contracts.HabitatMap : Label image this function overlays.
    """
    if not (0.0 < float(alpha) <= 1.0):
        raise HABITAPIError(
            f"plot_habitat_overlay: alpha must be in (0, 1]; got {alpha!r}."
        )

    try:
        convention = normalize_display_convention(display_convention)
    except HABITAPIError as exc:
        raise HABITAPIError(f"plot_habitat_overlay: {exc}") from exc

    image_vol = _as_volume(image, "image")
    label_vol = _as_volume(labels, "labels")
    if image_vol.shape != label_vol.shape:
        raise HABITAPIError(
            "plot_habitat_overlay: image and labels must share the same shape; "
            f"got image {image_vol.shape} vs labels {label_vol.shape}."
        )

    plt = _plt()
    label_int = np.asarray(label_vol, dtype=np.int32)
    resolved_direction, resolved_spacing = resolve_display_geometry(
        image, labels, direction=direction, spacing=spacing
    )
    direction_matrix = _direction_matrix(resolved_direction, ndim=image_vol.ndim)
    spacing_xyz = _spacing_xyz(resolved_spacing, ndim=image_vol.ndim)
    # Volume-level ID→colour so orthogonal slices and the colorbar match.
    habitat_ids = _positive_habitat_ids(label_int)
    id_to_color = _habitat_color_lookup(habitat_ids)

    with use_style("radiology"):
        if image_vol.ndim == 2 or axis is not None:
            axis_id = 0 if image_vol.ndim == 2 else int(axis)
            if image_vol.ndim == 3 and axis_id not in (0, 1, 2):
                raise HABITAPIError(
                    f"plot_habitat_overlay: axis must be 0, 1, or 2; got {axis_id}."
                )
            slice_index = _slice_index(label_int, axis_id, index)
            rgb, labs = _prepare_overlay_slice(
                image_vol,
                label_int,
                axis_id=axis_id,
                slice_index=slice_index,
                alpha=float(alpha),
                direction=direction_matrix,
                convention=convention,
                id_to_color=id_to_color,
            )
            extent = _imshow_physical_extent(
                (int(rgb.shape[0]), int(rgb.shape[1])),
                spacing_xyz,
                slice_axis=axis_id,
                ndim=image_vol.ndim,
                direction=direction_matrix,
                convention=convention,
            )

            fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5), constrained_layout=True)
            # Physical mm extent + equal aspect: 1 mm row == 1 mm col on screen.
            # adjustable='box' shrinks the axes, never re-squares the voxels.
            ax.imshow(
                rgb,
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
            )
            if contour:
                _draw_label_contour(ax, labs, extent=extent)
            ax.set_aspect("equal", adjustable="box")
            axis_name = (
                ("axis-0", "axis-1", "axis-2")[axis_id]
                if image_vol.ndim == 3
                else "2D"
            )
            ax.set_title(
                sanitize_label(
                    title
                    if title is not None
                    else f"Habitat overlay ({axis_name}, index={slice_index})"
                )
            )
            ax.axis("off")
            add_discrete_habitat_colorbar(
                ax,
                habitat_ids,
                _habitat_color_list(habitat_ids),
                colorbar=colorbar,
                label=colorbar_label,
            )
            return fig

        # 3D default: three orthogonal slices through the densest habitat region.
        # Taller figsize so coronal/sagittal panels (wide FOV, thick-slice height)
        # are not cramped; each axes uses physical mm extent + equal aspect.
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 6.5), constrained_layout=True)
        panel_names = (
            "Axis 0 (axial-like)",
            "Axis 1 (coronal-like)",
            "Axis 2 (sagittal-like)",
        )
        for axis_id, ax in enumerate(axes):
            slice_index = _slice_index(label_int, axis_id, None)
            rgb, labs = _prepare_overlay_slice(
                image_vol,
                label_int,
                axis_id=axis_id,
                slice_index=slice_index,
                alpha=float(alpha),
                direction=direction_matrix,
                convention=convention,
                id_to_color=id_to_color,
            )
            extent = _imshow_physical_extent(
                (int(rgb.shape[0]), int(rgb.shape[1])),
                spacing_xyz,
                slice_axis=axis_id,
                ndim=image_vol.ndim,
                direction=direction_matrix,
                convention=convention,
            )
            ax.imshow(
                rgb,
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
            )
            if contour:
                _draw_label_contour(ax, labs, extent=extent)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(sanitize_label(f"{panel_names[axis_id]} @ {slice_index}"))
            ax.axis("off")

        # Shared discrete bar on the last panel (same IDs on every view).
        add_discrete_habitat_colorbar(
            axes[2],
            habitat_ids,
            _habitat_color_list(habitat_ids),
            colorbar=colorbar,
            label=colorbar_label,
        )
        if title is not None:
            fig.suptitle(sanitize_label(title))
        else:
            fig.suptitle(sanitize_label("Habitat overlay on source image"))
        return fig
