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
filesystem and no ``show``. Background label ``0`` stays transparent so the
underlying greyscale anatomy remains visible; habitat IDs ``>= 1`` are tinted
with a colour-blind-friendly categorical palette.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz.labels import sanitize_label
from habit.viz.orientation import (
    DEFAULT_DISPLAY_CONVENTION,
    DEFAULT_NATIVE_DIRECTION,
    DEFAULT_RAS_DIRECTION,
    DisplayConvention,
    array_axis_lps_direction,
    desired_screen_directions,
    direction_matrix as _parse_direction_matrix,
    normalize_display_convention,
    orient_slice_for_display,
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

#: Colour-blind-friendly RGB colours for habitat IDs (cycled if needed).
#: Matches the radiology style palette so overlays stay journal-safe in
#: greyscale printouts.
_HABITAT_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (0.00, 0.45, 0.70),  # blue
    (0.90, 0.60, 0.00),  # orange
    (0.00, 0.62, 0.45),  # green
    (0.80, 0.40, 0.00),  # vermillion
    (0.80, 0.47, 0.65),  # reddish purple
    (0.95, 0.90, 0.25),  # yellow
    (0.35, 0.70, 0.90),  # sky blue
    (0.60, 0.60, 0.60),  # grey
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


def _as_volume(array: np.ndarray, name: str) -> np.ndarray:
    """
    Coerce ``array`` to a 2D or 3D float/int volume (drop singleton leading axes).

    Args:
        array: Candidate image or label array.
        name: Name used in error messages.

    Returns:
        Array with ndim in ``{2, 3}``.

    Raises:
        HABITAPIError: When the array cannot be interpreted as a volume.
    """
    volume = np.asarray(array)
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = np.squeeze(volume, axis=0)
    if volume.ndim == 4:
        # Multi-channel volumes: average channels for display only.
        volume = np.mean(volume, axis=-1) if volume.shape[-1] <= 4 else volume[0]
    if volume.ndim not in (2, 3):
        raise HABITAPIError(
            f"plot_habitat_overlay: {name} must be 2D or 3D after squeeze; "
            f"got shape {tuple(np.asarray(array).shape)}."
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


def _blend_overlay(
    grey: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
    colors: Sequence[Tuple[float, float, float]],
) -> np.ndarray:
    """
    Blend habitat colours onto a greyscale slice (label 0 stays transparent).

    Args:
        grey: 2D float array in ``[0, 1]``.
        labels: 2D integer label map, same shape as ``grey``.
        alpha: Opacity of habitat colours in ``(0, 1]``.
        colors: RGB triples cycled across habitat IDs.

    Returns:
        RGB float array of shape ``(H, W, 3)`` in ``[0, 1]``.
    """
    rgb = np.stack([grey, grey, grey], axis=-1)
    overlay = rgb.copy()
    habitat_ids = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    for index, habitat_id in enumerate(habitat_ids):
        mask = labels == habitat_id
        if not np.any(mask):
            continue
        color = colors[index % len(colors)]
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
) -> float:
    """
    Matplotlib data-unit aspect: ``(physical size of one row) / (one column)``.

    This is ``spacing_along_row / spacing_along_col`` after any display flips
    (``flipud`` / ``fliplr`` do not swap which array axis is row vs column).
    Prefer :func:`_imshow_physical_extent` + ``aspect='equal'`` for drawing so
    layout code cannot silently re-square anisotropic voxels.
    """
    if ndim == 2:
        # 2D arrays are ``(y, x)`` with SimpleITK spacing ``(x, y)``.
        return float(spacing_xyz[1]) / float(spacing_xyz[0])
    row_axis, col_axis = _slice_row_col_axes(slice_axis)
    spacing_row = _array_axis_spacing(spacing_xyz, row_axis)
    spacing_col = _array_axis_spacing(spacing_xyz, col_axis)
    return spacing_row / spacing_col


def _plane_spacings(
    spacing_xyz: Sequence[float],
    *,
    slice_axis: int,
    ndim: int,
) -> Tuple[float, float]:
    """
    Return ``(spacing_row_mm, spacing_col_mm)`` for a display plane.

    Args:
        spacing_xyz: SimpleITK spacing ``(x, y[, z])``.
        slice_axis: NumPy axis removed by ``np.take`` (ignored when ``ndim==2``).
        ndim: Array dimensionality.

    Returns:
        Physical size of one array row and one array column in millimetres.
    """
    if ndim == 2:
        return float(spacing_xyz[1]), float(spacing_xyz[0])
    row_axis, col_axis = _slice_row_col_axes(slice_axis)
    return (
        _array_axis_spacing(spacing_xyz, row_axis),
        _array_axis_spacing(spacing_xyz, col_axis),
    )


def _imshow_physical_extent(
    shape_hw: Tuple[int, int],
    spacing_xyz: Sequence[float],
    *,
    slice_axis: int,
    ndim: int,
) -> Tuple[float, float, float, float]:
    """
    ``imshow`` extent in millimetres so ``aspect='equal'`` is physically true.

    Uses ``origin='upper'`` convention: ``(left, right, bottom, top)`` with
    ``top < bottom`` so row 0 stays at the top of the axes after radiological
    flips. Thick-slice (large row spacing) then occupies more vertical millimetres
    and appears longer on screen, not flatter.
    """
    nrows, ncols = int(shape_hw[0]), int(shape_hw[1])
    if nrows <= 0 or ncols <= 0:
        raise HABITAPIError(
            "plot_habitat_overlay: slice shape must be positive for extent."
        )
    spacing_row, spacing_col = _plane_spacings(
        spacing_xyz, slice_axis=slice_axis, ndim=ndim
    )
    # left, right, bottom, top — top=0 keeps superior/anterior at the top edge
    # after _orient_slice_for_display; bottom is the full physical height.
    return (
        0.0,
        float(ncols) * spacing_col,
        float(nrows) * spacing_row,
        0.0,
    )


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
) -> np.ndarray:
    """Normalize, orient, and blend one orthogonal slice to RGB."""
    grey = _normalize_grey(_take_slice(image_vol, axis_id, slice_index))
    labs = _take_slice(label_int, axis_id, slice_index)
    grey = _orient_slice_for_display(
        grey, slice_axis=axis_id, direction=direction, convention=convention
    )
    labs = _orient_slice_for_display(
        labs, slice_axis=axis_id, direction=direction, convention=convention
    )
    return _blend_overlay(grey, labs, alpha=float(alpha), colors=_HABITAT_COLORS)


def plot_habitat_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 0.45,
    title: Optional[str] = None,
    axis: Optional[int] = None,
    index: Optional[int] = None,
    direction: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
    display_convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> "Figure":
    """
    Draw habitat labels as a translucent colour overlay on the source image.

    For 3D volumes the default is a three-panel figure (orthogonal slices in
    NumPy axis order ``0 / 1 / 2``, i.e. SimpleITK ``(z, y, x)``). Each panel
    uses the slice with the most non-background habitat voxels so the overlay
    is visible even when the tumour is off-centre. Pass ``axis`` / ``index`` to
    pin a specific slice. Label ``0`` is treated as background and is not
    coloured.

    Slices are oriented using ``direction`` (SimpleITK flattened 3x3) and
    ``display_convention`` (default ``\"radiological\"``). When ``direction``
    is omitted, LPS identity is assumed — the same default as
    :class:`~habit.api.image.ImageVolume` — not RAS.

    Panel aspect ratios follow ``spacing`` (SimpleITK ``(x, y, z)``) so thick
    slices are not squashed into square pixels on coronal / sagittal views.

    Args:
        image: Source image array (2D or 3D; SimpleITK/NumPy ``(z, y, x)`` order).
        labels: Habitat label map with the same shape as ``image``.
        alpha: Habitat colour opacity in ``(0, 1]``.
        title: Optional figure title (ASCII-sanitised).
        axis: If set, draw only this axis (``0``, ``1``, or ``2``).
        index: Slice index along ``axis``; densest habitat slice when omitted.
        direction: Optional SimpleITK direction cosines (9 floats). Same layout
            as ``ImageVolume.direction``. Controls anterior/posterior,
            superior/inferior, and left/right flips per panel.
        spacing: Optional SimpleITK voxel spacing ``(x, y[, z])`` in mm. Same
            layout as ``ImageVolume.spacing``. Controls true physical aspect
            per panel; defaults to isotropic ``1.0`` when omitted.
        display_convention: ``\"radiological\"`` (default), ``\"neurological\"``,
            or ``\"native\"`` (no display flips). See
            :mod:`habit.viz.orientation`.

    Returns:
        A matplotlib ``Figure``. The caller owns persistence / display.

    Raises:
        HABITAPIError: On shape / parameter errors.
        OptionalDependencyError: When matplotlib is not installed.
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
    direction_matrix = _direction_matrix(direction, ndim=image_vol.ndim)
    spacing_xyz = _spacing_xyz(spacing, ndim=image_vol.ndim)

    if image_vol.ndim == 2 or axis is not None:
        axis_id = 0 if image_vol.ndim == 2 else int(axis)
        if image_vol.ndim == 3 and axis_id not in (0, 1, 2):
            raise HABITAPIError(
                f"plot_habitat_overlay: axis must be 0, 1, or 2; got {axis_id}."
            )
        slice_index = _slice_index(label_int, axis_id, index)
        rgb = _prepare_overlay_slice(
            image_vol,
            label_int,
            axis_id=axis_id,
            slice_index=slice_index,
            alpha=float(alpha),
            direction=direction_matrix,
            convention=convention,
        )
        extent = _imshow_physical_extent(
            (int(rgb.shape[0]), int(rgb.shape[1])),
            spacing_xyz,
            slice_axis=axis_id,
            ndim=image_vol.ndim,
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
        ax.set_aspect("equal", adjustable="box")
        axis_name = ("axis-0", "axis-1", "axis-2")[axis_id] if image_vol.ndim == 3 else "2D"
        ax.set_title(
            sanitize_label(
                title
                if title is not None
                else f"Habitat overlay ({axis_name}, index={slice_index})"
            )
        )
        ax.axis("off")
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
        rgb = _prepare_overlay_slice(
            image_vol,
            label_int,
            axis_id=axis_id,
            slice_index=slice_index,
            alpha=float(alpha),
            direction=direction_matrix,
            convention=convention,
        )
        extent = _imshow_physical_extent(
            (int(rgb.shape[0]), int(rgb.shape[1])),
            spacing_xyz,
            slice_axis=axis_id,
            ndim=image_vol.ndim,
        )
        ax.imshow(
            rgb,
            interpolation="nearest",
            origin="upper",
            extent=extent,
            aspect="equal",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(sanitize_label(f"{panel_names[axis_id]} @ {slice_index}"))
        ax.axis("off")

    if title is not None:
        fig.suptitle(sanitize_label(title))
    else:
        fig.suptitle(sanitize_label("Habitat overlay on source image"))
    return fig
