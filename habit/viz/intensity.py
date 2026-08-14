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
"""Greyscale anatomy / intensity-slice figures for image preprocessing.

Whole-image geometry and intensity steps (reorient, resample, N4, denoise,
histogram, CLAHE, z-score, …) must be drawn as **full-FOV greyscale**
anatomy. This plotter never crops the display to an ROI and never uses a
sequential colourmap on MR/CT intensities.

Each panel is windowed independently in native intensity units; an
independent colorbar shows those limits so an affine change (z-score)
is visible as a scale change even when greyscale contrast looks similar.

Optional ``roi_mask`` is a **contour overlay on anatomy only**, for the
rare teaching case where the same transform follows the mask
(registration; optionally resample/reorient). It does not hide voxels
outside the mask.

Pure functions: arrays or :class:`~habit.api.image.ImageVolume` in, a
matplotlib ``Figure`` out, no filesystem and no ``show``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz.colorbar import ColorbarSpec, add_image_colorbar_from_spec, colorbar_is_enabled
from habit.viz.labels import sanitize_label
from habit.viz.orientation import (
    DEFAULT_DISPLAY_CONVENTION,
    DisplayConvention,
    array_from_display_input,
    direction_matrix as _parse_direction_matrix,
    imshow_physical_extent,
    normalize_display_convention,
    orient_slice_for_display,
    resolve_display_geometry,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_intensity_slice"]

#: What habit.viz needs matplotlib for.
_VIZ_PURPOSE = "greyscale anatomy / intensity-slice figures"

#: Cyan outline when an ROI contour is requested (English figures only).
_ROI_CONTOUR_COLOR = "#00E5FF"

#: Histogram size used to find a dominant low-end (air / padding) mode.
_CLIM_HIST_BINS: int = 256
#: A histogram bin is treated as background when it holds more than this
#: fraction of finite voxels (ImageJ Auto uses ``pixelCount / 10``).
_CLIM_BG_BIN_FRACTION: float = 0.10
#: Only look for that background mode in the lowest quarter of the range,
#: so a mid-grey tissue peak is not dropped.
_CLIM_BG_SEARCH_FRACTION: float = 0.25
#: After dropping air / padding, window the remaining tissue from this
#: lower percentile to ``_CLIM_TISSUE_P_HIGH`` so a long contrast tail
#: does not reopen the window, while organs are not clipped to white.
_CLIM_TISSUE_P_LOW: float = 2.0
_CLIM_TISSUE_P_HIGH: float = 90.0
#: Need at least this many voxels after dropping the background mode.
_CLIM_MIN_TISSUE: int = 16


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
    Coerce ``array`` to a 2D or 3D volume (drop singleton leading axes).

    Args:
        array: Candidate image array or volume object (``.data``).
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
        volume = np.mean(volume, axis=-1) if volume.shape[-1] <= 4 else volume[0]
    if volume.ndim not in (2, 3):
        raise HABITAPIError(
            f"plot_intensity_slice: {name} must be 2D or 3D after squeeze; "
            f"got shape {tuple(volume.shape)}."
        )
    if volume.size == 0:
        raise HABITAPIError(f"plot_intensity_slice: {name} must not be empty.")
    return np.asarray(volume)


def _percentile_window(sample: np.ndarray) -> Tuple[float, float]:
    """Return the 1st–99th percentile window, or min/max if too small."""
    if sample.size >= 4:
        vmin = float(np.percentile(sample, 1.0))
        vmax = float(np.percentile(sample, 99.0))
    else:
        vmin = float(np.min(sample))
        vmax = float(np.max(sample))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax < vmin:
        vmin = float(np.min(sample))
        vmax = float(np.max(sample))
    if vmax < vmin:
        vmax = vmin
    return vmin, vmax


def _drop_background_peak(finite: np.ndarray) -> np.ndarray:
    """
    Drop the dominant low-end histogram mode (air / padding).

    Contrast-enhanced MR/CT slices often have a huge near-zero peak plus a
    long bright tail. Percentile windows on all voxels then map parenchyma
    to near-black. Only the lowest quarter of the intensity range is
    searched, so a mid-grey tissue mode is kept.

    Args:
        finite: 1-D finite intensities from one slice.

    Returns:
        Intensities with the background mode removed, or ``finite`` itself
        when no dominant low-end peak is found / too few voxels remain.
    """
    n = int(finite.size)
    if n < _CLIM_MIN_TISSUE:
        return finite
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        return finite
    n_bins = int(min(_CLIM_HIST_BINS, n))
    hist, edges = np.histogram(finite, bins=n_bins, range=(lo, hi))
    search_n = max(1, int(np.ceil(n_bins * _CLIM_BG_SEARCH_FRACTION)))
    peak = int(np.argmax(hist[:search_n]))
    if float(hist[peak]) <= (n * _CLIM_BG_BIN_FRACTION):
        return finite
    cut = float(edges[peak + 1])
    kept = finite[finite > cut]
    if kept.size < _CLIM_MIN_TISSUE:
        return finite
    return kept


def _tissue_window(sample: np.ndarray) -> Tuple[float, float]:
    """
    Return a robust tissue window on voxels that already exclude air.

    ``median ± 3·MAD`` is too tight when the remaining tissue is still
    bimodal (dark body wall + brighter organs): the median sits in the
    dark cluster and parenchyma clips to white. Using the 2nd–90th
    percentiles of the remaining voxels keeps organs visible and still
    clips the brightest contrast / vessel tail.

    Args:
        sample: 1-D finite intensities (typically after background drop).

    Returns:
        Tuple of ``(vmin, vmax)``. ``vmax >= vmin``; both finite.
    """
    if sample.size < 4:
        return _percentile_window(sample)
    vmin = float(np.percentile(sample, _CLIM_TISSUE_P_LOW))
    vmax = float(np.percentile(sample, _CLIM_TISSUE_P_HIGH))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        return _percentile_window(sample)
    return vmin, vmax


def _panel_clim(slice_2d: np.ndarray) -> Tuple[float, float]:
    """
    Return independent ``(vmin, vmax)`` in native intensity units.

    Contrast-enhanced volumes are right-skewed: air/padding pile up near
    zero and a few hot voxels sit far above parenchyma. A 1st–99th
    percentile of *all* voxels therefore still windows to the bright
    tail and the anatomy looks black. ``median ± 3·MAD`` after dropping
    air is the opposite failure: the median stays in dark body wall and
    organs clip to white.

    The window is therefore:

    1. Drop a dominant low-end histogram mode (air / padding), if any.
    2. Use the 2nd–90th percentiles of the remaining tissue.

    A constant slice falls back to min/max (honest: no fake stretch).
    Callers must put these same numbers on the colorbar — never remap
    the array to ``[0, 1]``, which would hide an affine intensity change
    such as z-score.

    Args:
        slice_2d: Single greyscale slice in native units.

    Returns:
        Tuple of ``(vmin, vmax)``. ``vmax >= vmin``; both finite.
    """
    data = np.asarray(slice_2d, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    tissue = _drop_background_peak(finite)
    return _tissue_window(tissue)


def _clim_extend(
    slice_2d: np.ndarray, vmin: float, vmax: float
) -> str:
    """
    Return a colorbar ``extend`` mode for voxels outside ``[vmin, vmax]``.

    Args:
        slice_2d: Slice whose finite voxels were windowed.
        vmin: Lower colour limit shown on the colorbar.
        vmax: Upper colour limit shown on the colorbar.

    Returns:
        ``\"neither\"``, ``\"min\"``, ``\"max\"``, or ``\"both\"``.
    """
    finite = np.asarray(slice_2d, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "neither"
    tol = 1e-12 * (1.0 + abs(vmax) + abs(vmin))
    below = bool(np.any(finite < (vmin - tol)))
    above = bool(np.any(finite > (vmax + tol)))
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def _take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract a 2D slice from a 2D/3D volume."""
    if volume.ndim == 2:
        return volume
    return np.take(volume, index, axis=axis)


def _slice_index(
    volume: np.ndarray,
    axis: int,
    index: Optional[int],
    roi_mask: Optional[np.ndarray] = None,
) -> int:
    """
    Return a valid slice index along ``axis``.

    When ``index`` is omitted: prefer the densest ROI slice if a mask is
    given (contour teaching figures); otherwise pick the plane with the
    most above-threshold anatomy voxels so empty edge slices are avoided.

    Args:
        volume: Intensity volume used for anatomy-mass fallback.
        axis: Axis along which to choose the slice.
        index: Explicit slice index, or ``None`` for auto selection.
        roi_mask: Optional ROI; used only to pick a plane, never to crop.

    Returns:
        Slice index in ``[0, length)``.

    Raises:
        HABITAPIError: When the axis length is invalid or ``index`` is OOB.
    """
    if volume.ndim == 2:
        length = 1
    else:
        length = int(volume.shape[axis])
    if length <= 0:
        raise HABITAPIError(
            "plot_intensity_slice: volume axis length must be > 0."
        )
    if index is not None:
        if index < 0 or index >= length:
            raise HABITAPIError(
                f"plot_intensity_slice: slice index {index} is out of range "
                f"for axis length {length}."
            )
        return int(index)
    if volume.ndim == 2 or length == 1:
        return 0
    other_axes = tuple(i for i in range(volume.ndim) if i != axis)
    if roi_mask is not None:
        counts = np.sum(np.asarray(roi_mask) > 0, axis=other_axes)
        if int(np.max(counts)) > 0:
            return int(np.argmax(counts))
    finite = np.where(np.isfinite(volume), np.abs(volume), 0.0)
    positive = finite[finite > 0]
    threshold = float(np.percentile(positive, 10.0)) if positive.size else 0.0
    mass = np.sum(finite > threshold, axis=other_axes)
    if int(np.max(mass)) == 0:
        return length // 2
    return int(np.argmax(mass))


def _spacing_xyz(
    spacing: Optional[Sequence[float]],
    *,
    ndim: int,
) -> Tuple[float, ...]:
    """Parse SimpleITK spacing ``(x, y[, z])``; default to isotropic 1 mm."""
    if spacing is None:
        return tuple(1.0 for _ in range(ndim))
    values = tuple(float(v) for v in spacing)
    if len(values) != ndim:
        raise HABITAPIError(
            f"plot_intensity_slice: spacing must have {ndim} values "
            f"(SimpleITK x,y[,z]); got {len(values)}."
        )
    if any(not np.isfinite(v) or v <= 0.0 for v in values):
        raise HABITAPIError(
            "plot_intensity_slice: spacing values must be finite and > 0."
        )
    return values


def _imshow_extent(
    shape_hw: Tuple[int, int],
    spacing_xyz: Sequence[float],
    *,
    slice_axis: int,
    ndim: int,
    direction: Optional[np.ndarray] = None,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> Tuple[float, float, float, float]:
    """``imshow`` extent in millimetres so ``aspect='equal'`` is physical."""
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
        raise HABITAPIError(f"plot_intensity_slice: {exc}") from exc


def _draw_roi_contour(
    ax,
    roi_slice: Optional[np.ndarray],
    *,
    extent: Tuple[float, float, float, float],
) -> None:
    """Draw the ROI as a closed outline on anatomy (never a filled crop)."""
    if roi_slice is None:
        return
    binary = (np.asarray(roi_slice) > 0).astype(np.float64)
    if not np.any(binary):
        return
    ax.contour(
        binary,
        levels=[0.5],
        colors=[_ROI_CONTOUR_COLOR],
        linewidths=1.35,
        origin="upper",
        extent=extent,
    )


def _show_grey_panel(
    ax,
    volume: np.ndarray,
    *,
    axis_id: int,
    slice_index: int,
    direction,
    convention: DisplayConvention,
    spacing_xyz: Sequence[float],
    cmap: str,
    panel_title: str,
    roi_slice: Optional[np.ndarray],
    colorbar: ColorbarSpec,
    colorbar_label: str,
    symmetric_clim: bool = False,
) -> None:
    """Draw one oriented greyscale slice; optional contour, never ROI crop."""
    # Keep native units so the colorbar can show raw intensity vs z-score.
    grey = orient_slice_for_display(
        _take_slice(volume, axis_id, slice_index),
        slice_axis=axis_id,
        direction=direction,
        convention=convention,
    )
    vmin, vmax = _panel_clim(grey)
    if symmetric_clim:
        limit = max(abs(vmin), abs(vmax))
        if limit == 0.0:
            limit = 1.0
        vmin, vmax = -limit, limit
    extent = _imshow_extent(
        (int(grey.shape[0]), int(grey.shape[1])),
        spacing_xyz,
        slice_axis=axis_id,
        ndim=volume.ndim,
        direction=direction,
        convention=convention,
    )
    image = ax.imshow(
        grey,
        cmap=str(cmap),
        interpolation="nearest",
        origin="upper",
        extent=extent,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    if roi_slice is not None:
        oriented_roi = orient_slice_for_display(
            roi_slice,
            slice_axis=axis_id,
            direction=direction,
            convention=convention,
        )
        _draw_roi_contour(ax, oriented_roi, extent=extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(sanitize_label(panel_title))
    ax.axis("off")
    ax.set_facecolor("white")
    add_image_colorbar_from_spec(
        image,
        colorbar,
        ax=ax,
        label=colorbar_label,
        extend=_clim_extend(grey, vmin, vmax),
    )


def plot_intensity_slice(
    image: object,
    *,
    before: Optional[object] = None,
    roi_mask: Optional[object] = None,
    axis: Optional[int] = None,
    index: Optional[int] = None,
    cmap: str = "gray",
    title: Optional[str] = None,
    image_label: str = "Processed",
    before_label: str = "Original",
    direction: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
    display_convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
    roi_contour: bool = False,
    colorbar: ColorbarSpec = True,
    colorbar_label: str = "Intensity",
    before_colorbar_label: str = "Intensity",
    before_cmap: Optional[str] = None,
    symmetric_clim: bool = False,
) -> "Figure":
    """
    Display a whole-FOV greyscale anatomy / intensity slice.

    Use this for image-preprocessing teaching figures. Do **not** use
    :func:`~habit.viz.plot_voxel_texture_slice` for MR/CT intensities: that
    plotter is a voxel-texture map viewer (sequential colormap, ROI crop).

    Pass ``before=`` for a two-panel original | processed figure when both
    volumes share a grid (z-score, N4, histogram, CLAHE). After resample /
    reorient the grid often changes — omit ``before`` and show the processed
    volume alone.

    Each panel is windowed independently in **native units**: drop a
    dominant low-end histogram mode (air / padding), then the 2nd–90th
    percentiles of the remaining tissue. The colorbar shows those same
    limits, so a z-score (approximately :math:`N(0,1)`) is
    distinguishable from raw MR/CT intensity even when ``cmap='gray'``
    greyscale contrast looks similar.
    Do not share ``vmin``/``vmax`` across a z-score before/after pair:
    that would hide the affine change again.

    ``roi_mask`` is drawn only when ``roi_contour=True``, and then only as a
    cyan outline on the anatomy. Outside-ROI voxels stay visible. Whole-image
    steps should omit the mask.

    Args:
        image: Processed (or only) intensity volume. Array or
            :class:`~habit.api.image.ImageVolume`.
        before: Optional original volume, same shape as ``image``.
        roi_mask: Optional ROI (``> 0`` inside). Contour overlay only;
            never used to crop the display.
        axis: If set, draw only this NumPy axis (``0``, ``1``, or ``2``).
            Default for 3D is a single axial-like panel (``axis=0``).
        index: Slice index along ``axis``; auto (anatomy mass / densest ROI)
            when omitted.
        cmap: Matplotlib colormap. Default ``\"gray\"`` for MR/CT anatomy.
        title: Optional figure title (ASCII-sanitised).
        image_label: Right-hand (or only) panel title.
        before_label: Left-hand panel title when ``before`` is set.
        direction: Optional SimpleITK direction cosines (9 floats).
        spacing: Optional SimpleITK voxel spacing ``(x, y[, z])`` in mm.
        display_convention: ``\"radiological\"`` (default), ``\"neurological\"``,
            or ``\"native\"``.
        roi_contour: When ``True`` and ``roi_mask`` is set, outline the ROI
            on every anatomy panel.
        colorbar: Draw an independent colorbar per panel (default ``True``).
            Pass ``False`` to hide it, or a mapping of colorbar style
            kwargs (``shrink``, ``pad``, ``fraction``, ``aspect``,
            ``ticks``, ``label``, ...) to override the short default bar.
        colorbar_label: Colorbar label for the processed (or only) panel.
        before_colorbar_label: Colorbar label for the original panel when
            ``before`` is set.
        before_cmap: Colormap for the original panel. Defaults to ``cmap``.
            Use ``\"gray\"`` with ``cmap=\"RdBu_r\"`` so a z-score panel can
            show signed values while anatomy stays greyscale.
        symmetric_clim: When ``True``, the processed (or only) panel is
            windowed symmetrically about zero. Use this for z-score so the
            colorbar reads as ``[-a, a]`` rather than an asymmetric
            percentile window that hides the signed scale.

    Returns:
        A matplotlib ``Figure``. The caller owns persistence / display.

    Raises:
        HABITAPIError: On shape / parameter errors.
        OptionalDependencyError: When matplotlib is not installed.
    """
    try:
        convention = normalize_display_convention(display_convention)
    except HABITAPIError as exc:
        raise HABITAPIError(f"plot_intensity_slice: {exc}") from exc

    image_vol = _as_volume(image, "image")
    before_vol: Optional[np.ndarray] = None
    if before is not None:
        before_vol = _as_volume(before, "before")
        if before_vol.shape != image_vol.shape:
            raise HABITAPIError(
                "plot_intensity_slice: before and image must share the same "
                f"shape; got before {before_vol.shape} vs image "
                f"{image_vol.shape}. Omit before= when resample/reorient "
                "changed the grid."
            )

    roi_vol: Optional[np.ndarray] = None
    if roi_mask is not None:
        roi_vol = _as_volume(roi_mask, "roi_mask")
        if roi_vol.shape != image_vol.shape:
            raise HABITAPIError(
                "plot_intensity_slice: roi_mask and image must share the "
                f"same shape; got roi {roi_vol.shape} vs image "
                f"{image_vol.shape}."
            )

    resolved_direction, resolved_spacing = resolve_display_geometry(
        image, before, roi_mask, direction=direction, spacing=spacing
    )
    try:
        direction_matrix = _parse_direction_matrix(
            resolved_direction, ndim=image_vol.ndim
        )
    except HABITAPIError as exc:
        raise HABITAPIError(f"plot_intensity_slice: {exc}") from exc
    spacing_xyz = _spacing_xyz(resolved_spacing, ndim=image_vol.ndim)

    axis_id = 0 if image_vol.ndim == 2 else (0 if axis is None else int(axis))
    if image_vol.ndim == 3 and axis_id not in (0, 1, 2):
        raise HABITAPIError(
            f"plot_intensity_slice: axis must be 0, 1, or 2; got {axis_id}."
        )
    # Pick the plane from the original anatomy when a before/after pair
    # is shown. After z-score, |intensity| of air is no longer small, so
    # mass-based selection on the processed volume prefers empty slices.
    slice_index = _slice_index(
        before_vol if before_vol is not None else image_vol,
        axis_id,
        index,
        roi_mask=roi_vol if roi_contour else None,
    )
    roi_slice = (
        _take_slice(roi_vol, axis_id, slice_index)
        if (roi_contour and roi_vol is not None)
        else None
    )

    plt = _plt()
    n_panels = 2 if before_vol is not None else 1
    draw_cbar = colorbar_is_enabled(colorbar)
    panel_width = 6.2 if draw_cbar else 5.4
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(panel_width * n_panels, 5.0),
        constrained_layout=True,
        facecolor="white",
    )
    if n_panels == 1:
        axes = [axes]

    original_cmap = str(before_cmap) if before_cmap is not None else str(cmap)
    if before_vol is not None:
        _show_grey_panel(
            axes[0],
            before_vol,
            axis_id=axis_id,
            slice_index=slice_index,
            direction=direction_matrix,
            convention=convention,
            spacing_xyz=spacing_xyz,
            cmap=original_cmap,
            panel_title=before_label,
            roi_slice=roi_slice,
            colorbar=colorbar,
            colorbar_label=before_colorbar_label,
        )
        _show_grey_panel(
            axes[1],
            image_vol,
            axis_id=axis_id,
            slice_index=slice_index,
            direction=direction_matrix,
            convention=convention,
            spacing_xyz=spacing_xyz,
            cmap=cmap,
            panel_title=image_label,
            roi_slice=roi_slice,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            symmetric_clim=bool(symmetric_clim),
        )
    else:
        _show_grey_panel(
            axes[0],
            image_vol,
            axis_id=axis_id,
            slice_index=slice_index,
            direction=direction_matrix,
            convention=convention,
            spacing_xyz=spacing_xyz,
            cmap=cmap,
            panel_title=image_label if title is None else image_label,
            roi_slice=roi_slice,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            symmetric_clim=bool(symmetric_clim),
        )

    if title is not None:
        fig.suptitle(sanitize_label(title))
    fig.patch.set_facecolor("white")
    return fig
