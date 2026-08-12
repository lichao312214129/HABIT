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
"""Voxel-level texture / feature-map figures.

Pure functions: dense feature volumes (or a :class:`VoxelFeatureField`) in, a
matplotlib ``Figure`` out, no filesystem and no ``show``. Typical inputs are
local-entropy maps from :func:`habit.kernels.voxel_texture.local_entropy_map`
or per-voxel radiomics maps densified from a field; outside-ROI voxels stay
transparent (NaN / masked).

All text drawn on the figures is English-only. This module ships excellent 2D
multi-panel slices first; optional 3D volume rendering is intentionally out of
scope so the API stays consistent with the matplotlib ``[viz]`` stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz.labels import sanitize_label
from habit.viz.orientation import (
    DEFAULT_DISPLAY_CONVENTION,
    DisplayConvention,
    direction_matrix as _parse_direction_matrix,
    normalize_display_convention,
    orient_slice_for_display,
    slice_row_col_axes,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from habit.contracts.habitat import VoxelFeatureField

__all__ = [
    "dense_voxel_feature_map",
    "plot_voxel_texture_slice",
]

#: What habit.viz needs matplotlib for.
_VIZ_PURPOSE = "voxel texture / feature-map figures"

#: Supported panel layouts for :func:`plot_voxel_texture_slice`.
LayoutMode = Literal["side_by_side", "overlay", "feature_only"]

#: Default perceptually uniform colormap (journal-safe on print).
_DEFAULT_CMAP = "magma"


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
    Coerce ``array`` to a 2D or 3D volume (drop singleton leading axes).

    Args:
        array: Candidate image or feature array.
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
            f"plot_voxel_texture_slice: {name} must be 2D or 3D after squeeze; "
            f"got shape {tuple(np.asarray(array).shape)}."
        )
    if volume.size == 0:
        raise HABITAPIError(
            f"plot_voxel_texture_slice: {name} must not be empty."
        )
    return volume


def _coerce_array(
    value: object,
    *,
    name: str,
) -> np.ndarray:
    """
    Accept a NumPy array or an object with a ``.data`` volume (ImageVolume).

    Args:
        value: Array-like or volume contract.
        name: Name used in error messages.

    Returns:
        Dense NumPy array.

    Raises:
        HABITAPIError: When ``value`` cannot be coerced.
    """
    if value is None:
        raise HABITAPIError(
            f"plot_voxel_texture_slice: {name} must not be None."
        )
    data_attr = getattr(value, "data", None)
    if data_attr is not None and not isinstance(value, np.ndarray):
        return _as_volume(np.asarray(data_attr), name)
    return _as_volume(np.asarray(value), name)


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


def _take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract a 2D slice from a 2D/3D volume."""
    if volume.ndim == 2:
        return volume
    return np.take(volume, index, axis=axis)


def _slice_index_from_mask(
    mask: np.ndarray,
    axis: int,
    index: Optional[int],
) -> int:
    """
    Return a valid slice index along ``axis``.

    When ``index`` is omitted, pick the slice with the most positive mask
    voxels (ROI / finite feature support). Falls back to the geometric mid
    slice when the mask is empty.

    Args:
        mask: Boolean or numeric volume; values ``> 0`` count as support.
        axis: Axis along which to choose the slice.
        index: Explicit slice index, or ``None`` for auto selection.

    Returns:
        Slice index in ``[0, length)``.

    Raises:
        HABITAPIError: When the axis length is invalid or ``index`` is OOB.
    """
    if mask.ndim == 2:
        length = 1
    else:
        length = int(mask.shape[axis])
    if length <= 0:
        raise HABITAPIError(
            "plot_voxel_texture_slice: volume axis length must be > 0."
        )
    if index is not None:
        if index < 0 or index >= length:
            raise HABITAPIError(
                f"plot_voxel_texture_slice: slice index {index} is out of "
                f"range for axis length {length}."
            )
        return int(index)
    if mask.ndim == 2 or length == 1:
        return 0
    other_axes = tuple(i for i in range(mask.ndim) if i != axis)
    counts = np.sum(np.asarray(mask) > 0, axis=other_axes)
    if int(np.max(counts)) == 0:
        return length // 2
    return int(np.argmax(counts))


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
            f"plot_voxel_texture_slice: spacing must have {ndim} values "
            f"(SimpleITK x,y[,z]); got {len(values)}."
        )
    if any(not np.isfinite(v) or v <= 0.0 for v in values):
        raise HABITAPIError(
            "plot_voxel_texture_slice: spacing values must be finite and > 0."
        )
    return values


def _array_axis_spacing(spacing_xyz: Sequence[float], array_axis: int) -> float:
    """Physical size along a NumPy ``(z, y, x)`` array axis."""
    sitk_axis = (2, 1, 0)[int(array_axis)]
    return float(spacing_xyz[sitk_axis])


def _plane_spacings(
    spacing_xyz: Sequence[float],
    *,
    slice_axis: int,
    ndim: int,
) -> Tuple[float, float]:
    """Return ``(spacing_row_mm, spacing_col_mm)`` for a display plane."""
    if ndim == 2:
        return float(spacing_xyz[1]), float(spacing_xyz[0])
    row_axis, col_axis = slice_row_col_axes(slice_axis)
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
    """``imshow`` extent in millimetres so ``aspect='equal'`` is physical."""
    nrows, ncols = int(shape_hw[0]), int(shape_hw[1])
    if nrows <= 0 or ncols <= 0:
        raise HABITAPIError(
            "plot_voxel_texture_slice: slice shape must be positive for extent."
        )
    spacing_row, spacing_col = _plane_spacings(
        spacing_xyz, slice_axis=slice_axis, ndim=ndim
    )
    return (
        0.0,
        float(ncols) * spacing_col,
        float(nrows) * spacing_row,
        0.0,
    )


def _resolve_feature_column(
    field: "VoxelFeatureField",
    feature: Optional[Union[str, int]],
) -> Tuple[int, str]:
    """
    Resolve a feature name or column index on a :class:`VoxelFeatureField`.

    Args:
        field: Sparse per-voxel feature table.
        feature: Column name, integer index, or ``None`` when the field has
            exactly one feature column.

    Returns:
        ``(column_index, feature_name)``.

    Raises:
        HABITAPIError: When the feature cannot be resolved.
    """
    names = list(field.feature_names)
    if not names:
        raise HABITAPIError(
            "plot_voxel_texture_slice: VoxelFeatureField has no feature columns."
        )
    if feature is None:
        if len(names) != 1:
            raise HABITAPIError(
                "plot_voxel_texture_slice: feature must be set when the field "
                f"has {len(names)} columns; names={names!r}."
            )
        return 0, str(names[0])
    if isinstance(feature, int):
        if feature < 0 or feature >= len(names):
            raise HABITAPIError(
                f"plot_voxel_texture_slice: feature index {feature} out of "
                f"range for {len(names)} columns."
            )
        return int(feature), str(names[feature])
    name = str(feature)
    if name not in names:
        raise HABITAPIError(
            f"plot_voxel_texture_slice: feature {name!r} not in "
            f"feature_names={names!r}."
        )
    return names.index(name), name


def dense_voxel_feature_map(
    field: "VoxelFeatureField",
    feature: Optional[Union[str, int]] = None,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Scatter one :class:`VoxelFeatureField` column into a dense volume.

    Voxels outside the field's ``voxel_index`` receive ``fill_value`` (default
    NaN) so matplotlib can leave them transparent.

    Args:
        field: Sparse ROI feature table with ``(z, y, x)`` indices.
        feature: Column name or index; optional when the field has one column.
        fill_value: Value written outside the ROI (prefer NaN for display).

    Returns:
        Float array with shape ``field.geometry.shape``.

    Raises:
        HABITAPIError: On unresolved feature names or invalid geometry.
    """
    # Local import keeps ``import habit.viz`` light for callers that only need
    # style helpers and never touch habitat contracts.
    from habit.contracts.habitat import VoxelFeatureField as _VoxelFeatureField

    if not isinstance(field, _VoxelFeatureField):
        raise HABITAPIError(
            "dense_voxel_feature_map: field must be a VoxelFeatureField; "
            f"got {type(field).__name__}."
        )
    column, _name = _resolve_feature_column(field, feature)
    shape = tuple(int(v) for v in field.geometry.shape)
    if len(shape) not in (2, 3):
        raise HABITAPIError(
            "dense_voxel_feature_map: geometry.shape must be 2D or 3D; "
            f"got {shape}."
        )
    volume = np.full(shape, float(fill_value), dtype=np.float64)
    index = np.asarray(field.voxel_index, dtype=np.int64)
    values = np.asarray(field.values[:, column], dtype=np.float64)
    if index.shape[0] == 0:
        return volume
    if shape == 2:
        # Geometry may be (y, x); indices still carry a leading z column of 0.
        volume[index[:, 1], index[:, 2]] = values
    else:
        volume[index[:, 0], index[:, 1], index[:, 2]] = values
    return volume


def _feature_display_limits(
    feature_slice: np.ndarray,
    vmin: Optional[float],
    vmax: Optional[float],
) -> Tuple[float, float]:
    """
    Resolve colourscale limits from finite feature values.

    Args:
        feature_slice: 2D feature values (may contain NaN).
        vmin: Explicit lower bound, or ``None`` for the 2nd percentile.
        vmax: Explicit upper bound, or ``None`` for the 98th percentile.

    Returns:
        ``(vmin, vmax)`` suitable for ``imshow``.
    """
    finite = np.asarray(feature_slice, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(finite, 2.0)) if vmin is None else float(vmin)
    high = float(np.percentile(finite, 98.0)) if vmax is None else float(vmax)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high <= low:
            high = low + 1.0
    return low, high


def _masked_feature_slice(
    feature_slice: np.ndarray,
    roi_slice: Optional[np.ndarray],
) -> np.ma.MaskedArray:
    """
    Mask non-finite values and optional outside-ROI voxels for ``imshow``.

    Args:
        feature_slice: 2D feature values.
        roi_slice: Optional 2D ROI mask (``> 0`` keeps the voxel).

    Returns:
        Masked array whose invalid entries stay transparent.
    """
    data = np.asarray(feature_slice, dtype=np.float64)
    mask = ~np.isfinite(data)
    if roi_slice is not None:
        mask = mask | (np.asarray(roi_slice) <= 0)
    return np.ma.array(data, mask=mask)


def _draw_single_axis_figure(
    *,
    anatomy: Optional[np.ndarray],
    feature: np.ndarray,
    roi_mask: Optional[np.ndarray],
    axis_id: int,
    slice_index: int,
    mode: LayoutMode,
    cmap: str,
    alpha: float,
    vmin: Optional[float],
    vmax: Optional[float],
    title: Optional[str],
    feature_label: str,
    direction: Optional[np.ndarray],
    convention: DisplayConvention,
    spacing_xyz: Sequence[float],
) -> "Figure":
    """Build a one- or two-panel figure for a single orthogonal slice."""
    plt = _plt()
    feat_slice = _take_slice(feature, axis_id, slice_index)
    feat_slice = orient_slice_for_display(
        feat_slice,
        slice_axis=axis_id,
        direction=direction,
        convention=convention,
    )
    roi_slice = None
    if roi_mask is not None:
        roi_slice = orient_slice_for_display(
            _take_slice(roi_mask, axis_id, slice_index),
            slice_axis=axis_id,
            direction=direction,
            convention=convention,
        )

    extent = _imshow_physical_extent(
        (int(feat_slice.shape[0]), int(feat_slice.shape[1])),
        spacing_xyz,
        slice_axis=axis_id,
        ndim=feature.ndim,
    )
    clim = _feature_display_limits(feat_slice, vmin, vmax)
    masked = _masked_feature_slice(feat_slice, roi_slice)
    axis_name = (
        ("axis-0", "axis-1", "axis-2")[axis_id] if feature.ndim == 3 else "2D"
    )
    default_title = (
        f"{feature_label} ({axis_name}, index={slice_index})"
    )

    if mode == "feature_only" or anatomy is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5), constrained_layout=True)
        image = ax.imshow(
            masked,
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            interpolation="nearest",
            origin="upper",
            extent=extent,
            aspect="equal",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            sanitize_label(title if title is not None else default_title)
        )
        ax.axis("off")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(sanitize_label(feature_label))
        return fig

    anat_slice = orient_slice_for_display(
        _normalize_grey(_take_slice(anatomy, axis_id, slice_index)),
        slice_axis=axis_id,
        direction=direction,
        convention=convention,
    )

    if mode == "overlay":
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5), constrained_layout=True)
        ax.imshow(
            anat_slice,
            cmap="gray",
            interpolation="nearest",
            origin="upper",
            extent=extent,
            aspect="equal",
        )
        image = ax.imshow(
            masked,
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            alpha=float(alpha),
            interpolation="nearest",
            origin="upper",
            extent=extent,
            aspect="equal",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            sanitize_label(title if title is not None else default_title)
        )
        ax.axis("off")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(sanitize_label(feature_label))
        return fig

    # side_by_side
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), constrained_layout=True)
    axes[0].imshow(
        anat_slice,
        cmap="gray",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        aspect="equal",
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title(sanitize_label("Anatomy"))
    axes[0].axis("off")

    image = axes[1].imshow(
        masked,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        interpolation="nearest",
        origin="upper",
        extent=extent,
        aspect="equal",
    )
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title(sanitize_label(feature_label))
    axes[1].axis("off")
    cbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label(sanitize_label(feature_label))
    if title is not None:
        fig.suptitle(sanitize_label(title))
    else:
        fig.suptitle(sanitize_label(default_title))
    return fig


def _draw_triptych(
    *,
    anatomy: Optional[np.ndarray],
    feature: np.ndarray,
    roi_mask: Optional[np.ndarray],
    support: np.ndarray,
    mode: LayoutMode,
    cmap: str,
    alpha: float,
    vmin: Optional[float],
    vmax: Optional[float],
    title: Optional[str],
    feature_label: str,
    direction: Optional[np.ndarray],
    convention: DisplayConvention,
    spacing_xyz: Sequence[float],
) -> "Figure":
    """Three orthogonal panels through the densest support region."""
    plt = _plt()
    panel_names = (
        "Axis 0 (axial-like)",
        "Axis 1 (coronal-like)",
        "Axis 2 (sagittal-like)",
    )
    n_cols = 2 if mode == "side_by_side" and anatomy is not None else 1
    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(5.5 * n_cols, 14.0),
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = np.asarray(axes).reshape(3, 1)

    for axis_id in range(3):
        slice_index = _slice_index_from_mask(support, axis_id, None)
        feat_slice = orient_slice_for_display(
            _take_slice(feature, axis_id, slice_index),
            slice_axis=axis_id,
            direction=direction,
            convention=convention,
        )
        roi_slice = None
        if roi_mask is not None:
            roi_slice = orient_slice_for_display(
                _take_slice(roi_mask, axis_id, slice_index),
                slice_axis=axis_id,
                direction=direction,
                convention=convention,
            )
        extent = _imshow_physical_extent(
            (int(feat_slice.shape[0]), int(feat_slice.shape[1])),
            spacing_xyz,
            slice_axis=axis_id,
            ndim=3,
        )
        clim = _feature_display_limits(feat_slice, vmin, vmax)
        masked = _masked_feature_slice(feat_slice, roi_slice)
        row_title = f"{panel_names[axis_id]} @ {slice_index}"

        if mode == "side_by_side" and anatomy is not None:
            anat_slice = orient_slice_for_display(
                _normalize_grey(_take_slice(anatomy, axis_id, slice_index)),
                slice_axis=axis_id,
                direction=direction,
                convention=convention,
            )
            axes[axis_id, 0].imshow(
                anat_slice,
                cmap="gray",
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
            )
            axes[axis_id, 0].set_aspect("equal", adjustable="box")
            axes[axis_id, 0].set_title(sanitize_label(f"Anatomy — {row_title}"))
            axes[axis_id, 0].axis("off")
            image = axes[axis_id, 1].imshow(
                masked,
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
            )
            axes[axis_id, 1].set_aspect("equal", adjustable="box")
            axes[axis_id, 1].set_title(
                sanitize_label(f"{feature_label} — {row_title}")
            )
            axes[axis_id, 1].axis("off")
            fig.colorbar(image, ax=axes[axis_id, 1], fraction=0.046, pad=0.04)
        elif mode == "overlay" and anatomy is not None:
            anat_slice = orient_slice_for_display(
                _normalize_grey(_take_slice(anatomy, axis_id, slice_index)),
                slice_axis=axis_id,
                direction=direction,
                convention=convention,
            )
            axes[axis_id, 0].imshow(
                anat_slice,
                cmap="gray",
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
            )
            image = axes[axis_id, 0].imshow(
                masked,
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                alpha=float(alpha),
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
            )
            axes[axis_id, 0].set_aspect("equal", adjustable="box")
            axes[axis_id, 0].set_title(
                sanitize_label(f"{feature_label} — {row_title}")
            )
            axes[axis_id, 0].axis("off")
            fig.colorbar(image, ax=axes[axis_id, 0], fraction=0.046, pad=0.04)
        else:
            image = axes[axis_id, 0].imshow(
                masked,
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
            )
            axes[axis_id, 0].set_aspect("equal", adjustable="box")
            axes[axis_id, 0].set_title(
                sanitize_label(f"{feature_label} — {row_title}")
            )
            axes[axis_id, 0].axis("off")
            fig.colorbar(image, ax=axes[axis_id, 0], fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(sanitize_label(title))
    else:
        fig.suptitle(sanitize_label(f"{feature_label} (orthogonal slices)"))
    return fig


def plot_voxel_texture_slice(
    feature_map: Union[np.ndarray, "VoxelFeatureField", object],
    *,
    anatomy: Optional[Union[np.ndarray, object]] = None,
    roi_mask: Optional[Union[np.ndarray, object]] = None,
    feature: Optional[Union[str, int]] = None,
    axis: Optional[int] = None,
    index: Optional[int] = None,
    mode: LayoutMode = "side_by_side",
    cmap: str = _DEFAULT_CMAP,
    alpha: float = 0.55,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    feature_label: Optional[str] = None,
    direction: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
    display_convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> "Figure":
    """
    Display a voxel-level texture / feature map as 2D publication panels.

    Accepts either a dense 2D/3D map (e.g. output of
    :func:`~habit.kernels.local_entropy_map`) or a sparse
    :class:`~habit.contracts.habitat.VoxelFeatureField` (e.g. from the
    ``local_entropy`` / ``voxel_radiomics`` extractors). Optional anatomy is
    shown side-by-side or as a translucent underlay. Outside-ROI / non-finite
    voxels stay transparent.

    For 3D volumes the default is three orthogonal panels through the densest
    ROI (or densest finite-feature) slice. Pass ``axis`` / ``index`` to pin one
    plane (``index=None`` still auto-selects the densest slice on that axis).

    This API is **2D-slice only** (matplotlib ``[viz]``). It does not provide
    3D volume rendering; use external viewers if you need full volumetric
    browsing of a texture map.

    Args:
        feature_map: Dense feature volume, or a ``VoxelFeatureField``. Objects
            with a ``.data`` attribute (e.g. ``ImageVolume``) are also accepted
            as dense maps.
        anatomy: Optional greyscale underlay / companion panel (same shape).
        roi_mask: Optional ROI mask (``> 0`` inside). Used for auto slice
            selection and to hide outside-ROI feature values.
        feature: Column name or index when ``feature_map`` is a field.
        axis: If set, draw only this NumPy axis (``0``, ``1``, or ``2``).
        index: Slice index along ``axis``; densest support when omitted.
        mode: ``\"side_by_side\"`` (default), ``\"overlay\"``, or
            ``\"feature_only\"``.
        cmap: Matplotlib colormap name for the feature values.
        alpha: Feature opacity when ``mode=\"overlay\"`` (in ``(0, 1]``).
        vmin: Optional colourscale lower bound (else 2nd percentile).
        vmax: Optional colourscale upper bound (else 98th percentile).
        title: Optional figure title (ASCII-sanitised).
        feature_label: Colourbar / panel label; defaults to the feature name
            or ``\"Voxel texture\"``.
        direction: Optional SimpleITK direction cosines (9 floats).
        spacing: Optional SimpleITK voxel spacing ``(x, y[, z])`` in mm.
        display_convention: ``\"radiological\"`` (default), ``\"neurological\"``,
            or ``\"native\"``.

    Returns:
        A matplotlib ``Figure``. The caller owns persistence / display.

    Raises:
        HABITAPIError: On shape / parameter errors.
        OptionalDependencyError: When matplotlib is not installed.
    """
    if mode not in ("side_by_side", "overlay", "feature_only"):
        raise HABITAPIError(
            "plot_voxel_texture_slice: mode must be 'side_by_side', "
            f"'overlay', or 'feature_only'; got {mode!r}."
        )
    if not (0.0 < float(alpha) <= 1.0):
        raise HABITAPIError(
            f"plot_voxel_texture_slice: alpha must be in (0, 1]; got {alpha!r}."
        )

    try:
        convention = normalize_display_convention(display_convention)
    except HABITAPIError as exc:
        raise HABITAPIError(f"plot_voxel_texture_slice: {exc}") from exc

    from habit.contracts.habitat import VoxelFeatureField as _VoxelFeatureField

    resolved_label = feature_label
    if isinstance(feature_map, _VoxelFeatureField):
        _column, name = _resolve_feature_column(feature_map, feature)
        feature_vol = dense_voxel_feature_map(feature_map, feature)
        if resolved_label is None:
            resolved_label = name
        if spacing is None:
            spacing = tuple(float(v) for v in feature_map.geometry.spacing)
        if direction is None:
            direction = tuple(float(v) for v in feature_map.geometry.direction)
    else:
        feature_vol = _coerce_array(feature_map, name="feature_map")
        if feature is not None:
            raise HABITAPIError(
                "plot_voxel_texture_slice: feature= is only valid when "
                "feature_map is a VoxelFeatureField."
            )
        if resolved_label is None:
            resolved_label = "Voxel texture"

    anatomy_vol: Optional[np.ndarray] = None
    if anatomy is not None:
        anatomy_vol = _coerce_array(anatomy, name="anatomy")
        if anatomy_vol.shape != feature_vol.shape:
            raise HABITAPIError(
                "plot_voxel_texture_slice: anatomy and feature_map must share "
                f"the same shape; got anatomy {anatomy_vol.shape} vs feature "
                f"{feature_vol.shape}."
            )

    roi_vol: Optional[np.ndarray] = None
    if roi_mask is not None:
        roi_vol = _coerce_array(roi_mask, name="roi_mask")
        if roi_vol.shape != feature_vol.shape:
            raise HABITAPIError(
                "plot_voxel_texture_slice: roi_mask and feature_map must share "
                f"the same shape; got roi {roi_vol.shape} vs feature "
                f"{feature_vol.shape}."
            )

    # Support mask for auto slice selection: ROI if given, else finite voxels.
    if roi_vol is not None:
        support = (np.asarray(roi_vol) > 0).astype(np.int8)
    else:
        support = np.isfinite(feature_vol).astype(np.int8)

    try:
        direction_matrix = _parse_direction_matrix(
            direction, ndim=feature_vol.ndim
        )
    except HABITAPIError as exc:
        raise HABITAPIError(f"plot_voxel_texture_slice: {exc}") from exc
    spacing_xyz = _spacing_xyz(spacing, ndim=feature_vol.ndim)

    if feature_vol.ndim == 2 or axis is not None:
        axis_id = 0 if feature_vol.ndim == 2 else int(axis)
        if feature_vol.ndim == 3 and axis_id not in (0, 1, 2):
            raise HABITAPIError(
                f"plot_voxel_texture_slice: axis must be 0, 1, or 2; "
                f"got {axis_id}."
            )
        slice_index = _slice_index_from_mask(support, axis_id, index)
        return _draw_single_axis_figure(
            anatomy=anatomy_vol,
            feature=feature_vol,
            roi_mask=roi_vol,
            axis_id=axis_id,
            slice_index=slice_index,
            mode=mode,
            cmap=str(cmap),
            alpha=float(alpha),
            vmin=vmin,
            vmax=vmax,
            title=title,
            feature_label=str(resolved_label),
            direction=direction_matrix,
            convention=convention,
            spacing_xyz=spacing_xyz,
        )

    return _draw_triptych(
        anatomy=anatomy_vol,
        feature=feature_vol,
        roi_mask=roi_vol,
        support=support,
        mode=mode,
        cmap=str(cmap),
        alpha=float(alpha),
        vmin=vmin,
        vmax=vmax,
        title=title,
        feature_label=str(resolved_label),
        direction=direction_matrix,
        convention=convention,
        spacing_xyz=spacing_xyz,
    )
