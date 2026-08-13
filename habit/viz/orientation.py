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
"""Display-orientation helpers for habitat visualization.

Shared by the matplotlib overlay (:mod:`habit.viz.habitat_overlay`) and the
optional napari viewer (:mod:`habit.viz.habitat_napari`). **All** display
flip / convention decisions for habitat image viewing live here; backends
must not invent their own axis flips.

``ImageVolume.data`` / SimpleITK arrays are NumPy ``(z, y, x)`` while
``direction`` columns are SimpleITK index axes ``(x, y, z)`` in an LPS world
(+x Left, +y Posterior, +z Superior). Raw array order therefore often shows
posterior/inferior toward the top of a canvas that uses image coordinates
(row 0 at the top). These helpers compute the per-axis flips **and** the in-plane transpose
that put axial / coronal / sagittal panels into the requested display
convention. A raw ``np.take`` extract uses array-axis order, so when
superior lies along the column axis the slice must be transposed before
``flipud`` / ``fliplr`` — otherwise SI stays horizontal.

Default display convention
--------------------------
``radiological`` (clinical radiology):

- Axial: anterior up; patient's right on the viewer's left.
- Coronal / sagittal: superior up; coronal uses the same L-R rule as axial;
  sagittal puts anterior on the viewer's left.

``neurological`` keeps the same superior/anterior "up" rules but reverses
left/right (patient's left on the viewer's left). ``native`` applies no
display flips (array order as stored).

Missing ``direction``
---------------------
When callers omit geometry, HABIT assumes **LPS identity** — the same default
as :meth:`habit.api.image.ImageVolume.from_array`. It does **not** silently
assume RAS (that previously mis-flipped A-P on LPS demo NRRDs).

Napari vs matplotlib (axial index)
----------------------------------
Matplotlib orients each 2D extract independently, so coronal/sagittal panels
can flip superior/inferior **within the panel** without remapping axial slice
indices. Napari applies whole-volume flips; by default axis 0 (``z``) is
**not** flipped so the axial slider keeps file / ITK-SNAP index semantics.
In-plane A-P / L-R still follow the same convention as matplotlib.
"""

from __future__ import annotations

import warnings
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError

DisplayConvention = Literal["radiological", "neurological", "native"]

__all__ = [
    "DEFAULT_DISPLAY_CONVENTION",
    "DEFAULT_NATIVE_DIRECTION",
    "DEFAULT_RAS_DIRECTION",
    "DisplayConvention",
    "apply_radiological_flips",
    "array_axis_lps_direction",
    "array_from_display_input",
    "desired_screen_directions",
    "direction_matrix",
    "display_array_axis_flips",
    "display_geometry_from_input",
    "display_slice_row_col_axes",
    "imshow_physical_extent",
    "normalize_display_convention",
    "orient_slice_for_display",
    "plane_spacings_mm",
    "radiological_array_axis_flips",
    "resolve_display_geometry",
    "slice_row_col_axes",
    "volume_display_flips",
]

# SimpleITK / ITK physical axis meanings (LPS world): +x Left, +y Posterior, +z Superior.
_LPS_RIGHT = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
_LPS_LEFT = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_LPS_ANTERIOR = np.array([0.0, -1.0, 0.0], dtype=np.float64)
_LPS_SUPERIOR = np.array([0.0, 0.0, 1.0], dtype=np.float64)

#: Default when callers pass arrays without geometry. Matches
#: :meth:`habit.api.image.ImageVolume.from_array` (LPS identity), **not** RAS.
DEFAULT_NATIVE_DIRECTION: Tuple[float, ...] = (
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
)

#: Explicit RAS direction (common for NIfTI via SimpleITK). Use when geometry
#: is known to be RAS; do not treat omitted ``direction`` as this value.
DEFAULT_RAS_DIRECTION: Tuple[float, ...] = (
    -1.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    1.0,
)

#: Product default for habitat image display (clinical radiology layout).
DEFAULT_DISPLAY_CONVENTION: DisplayConvention = "radiological"

_VALID_CONVENTIONS: Tuple[DisplayConvention, ...] = (
    "radiological",
    "neurological",
    "native",
)


def normalize_display_convention(
    convention: Optional[str],
) -> DisplayConvention:
    """
    Validate and normalize a display-convention name.

    Args:
        convention: ``\"radiological\"``, ``\"neurological\"``, ``\"native\"``,
            or ``None`` (uses :data:`DEFAULT_DISPLAY_CONVENTION`).

    Returns:
        Normalized convention key.

    Raises:
        HABITAPIError: When ``convention`` is not a supported value.
    """
    if convention is None:
        return DEFAULT_DISPLAY_CONVENTION
    key = str(convention).strip().lower()
    if key not in _VALID_CONVENTIONS:
        raise HABITAPIError(
            "display_convention must be one of "
            f"{list(_VALID_CONVENTIONS)}; got {convention!r}."
        )
    return key  # type: ignore[return-value]


def direction_matrix(
    direction: Optional[Sequence[float]],
    *,
    ndim: int,
) -> Optional[np.ndarray]:
    """
    Parse a SimpleITK flattened direction into a square matrix.

    Args:
        direction: Flattened row-major direction cosines (``ndim**2`` values),
            in SimpleITK index order ``(x, y, z)``. ``None`` uses
            :data:`DEFAULT_NATIVE_DIRECTION` (LPS identity), matching
            :class:`~habit.api.image.ImageVolume` defaults — not RAS.
        ndim: Array dimensionality (2 or 3). Orientation flips apply only to 3D.

    Returns:
        ``(3, 3)`` float matrix for 3D volumes, or ``None`` when ``ndim != 3``.

    Raises:
        HABITAPIError: When ``direction`` has the wrong length.
    """
    if ndim != 3:
        return None
    values = (
        DEFAULT_NATIVE_DIRECTION
        if direction is None
        else tuple(float(v) for v in direction)
    )
    expected = 9
    if len(values) != expected:
        raise HABITAPIError(
            f"direction must have {expected} values "
            f"(SimpleITK 3x3 flattened); got {len(values)}."
        )
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def array_axis_lps_direction(direction: np.ndarray, array_axis: int) -> np.ndarray:
    """
    LPS-world unit direction of increasing ``array_axis`` in a ``(z, y, x)`` volume.

    ``ImageVolume.data`` / ``sitk.GetArrayFromImage`` use NumPy order ``(z, y, x)``
    while ``direction`` columns are SimpleITK index axes ``(x, y, z)``.
    """
    sitk_axis = (2, 1, 0)[int(array_axis)]
    vector = np.asarray(direction[:, sitk_axis], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise HABITAPIError("direction matrix has a zero-length axis.")
    return vector / norm


def slice_row_col_axes(slice_axis: int) -> Tuple[int, int]:
    """Return ``(row_array_axis, col_array_axis)`` for ``np.take(..., axis=slice_axis)``."""
    if slice_axis == 0:
        return 1, 2
    if slice_axis == 1:
        return 0, 2
    if slice_axis == 2:
        return 0, 1
    raise HABITAPIError(f"axis must be 0, 1, or 2; got {slice_axis}.")


def display_slice_row_col_axes(
    slice_axis: int,
    *,
    direction: Optional[np.ndarray] = None,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> Tuple[int, int]:
    """
    Return display ``(row_array_axis, col_array_axis)`` after any SI transpose.

    ``np.take(..., axis=slice_axis)`` always yields axes in array order
    (coronal-like ``(z, x)``, sagittal-like ``(z, y)``). When the direction
    matrix puts the desired screen-up vector (superior on coronal/sagittal,
    anterior on axial) along the **column** axis, the 2D extract must be
    transposed so that axis becomes rows — otherwise SI (or A-P) is drawn
    horizontally.

    Args:
        slice_axis: NumPy axis removed by ``np.take``.
        direction: ``(3, 3)`` LPS direction matrix, or ``None`` (no transpose).
        convention: Display convention. ``\"native\"`` never transposes.

    Returns:
        Array axes that should be rows and columns after orientation.
    """
    row_axis, col_axis = slice_row_col_axes(slice_axis)
    convention = normalize_display_convention(convention)
    if convention == "native" or direction is None:
        return row_axis, col_axis
    row_dir = array_axis_lps_direction(direction, row_axis)
    col_dir = array_axis_lps_direction(direction, col_axis)
    desired_up, _desired_left = desired_screen_directions(
        slice_axis, convention=convention
    )
    if abs(float(np.dot(col_dir, desired_up))) > abs(
        float(np.dot(row_dir, desired_up))
    ):
        return col_axis, row_axis
    return row_axis, col_axis


def desired_screen_directions(
    slice_axis: int,
    *,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Screen ``(up, left)`` directions in LPS world for a slice axis.

    Radiological (default):

    - Axial-like (axis 0): anterior up, patient right on viewer's left.
    - Coronal-like (axis 1): superior up, patient right on viewer's left.
    - Sagittal-like (axis 2): superior up, anterior on viewer's left.

    Neurological keeps the same "up" vectors but reverses left/right on
    axial and coronal (patient left on viewer's left). Sagittal left/right
    is unchanged (still anterior on the viewer's left).
    """
    convention = normalize_display_convention(convention)
    if convention == "native":
        raise HABITAPIError(
            "desired_screen_directions is undefined for convention='native'."
        )

    if slice_axis == 0:
        left = _LPS_LEFT if convention == "neurological" else _LPS_RIGHT
        return _LPS_ANTERIOR.copy(), left.copy()
    if slice_axis == 1:
        left = _LPS_LEFT if convention == "neurological" else _LPS_RIGHT
        return _LPS_SUPERIOR.copy(), left.copy()
    if slice_axis == 2:
        # Sagittal: "left" on screen is anterior for both conventions.
        return _LPS_SUPERIOR.copy(), _LPS_ANTERIOR.copy()
    raise HABITAPIError(f"axis must be 0, 1, or 2; got {slice_axis}.")


def orient_slice_for_display(
    slice_2d: np.ndarray,
    *,
    slice_axis: int,
    direction: Optional[np.ndarray],
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> np.ndarray:
    """
    Transpose and flip a 2D slice so image coordinates match ``convention``.

    Used by matplotlib per orthogonal extract. Does not remap which volume
    index was selected — only the 2D panel layout.

    Order of operations:

    1. Transpose when the desired screen-up vector (superior on coronal /
       sagittal) lies along the extract's column axis, so SI becomes rows.
    2. ``flipud`` / ``fliplr`` so row 0 is screen-up and col 0 is screen-left
       under ``origin='upper'``.
    """
    data = np.asarray(slice_2d)
    convention = normalize_display_convention(convention)
    if convention == "native" or direction is None or data.ndim != 2:
        return data

    raw_row, raw_col = slice_row_col_axes(slice_axis)
    row_axis, col_axis = display_slice_row_col_axes(
        slice_axis, direction=direction, convention=convention
    )
    if (row_axis, col_axis) != (raw_row, raw_col):
        # np.take yields (raw_row, raw_col); swap so desired-up is rows.
        data = np.transpose(data)

    row_dir = array_axis_lps_direction(direction, row_axis)
    col_dir = array_axis_lps_direction(direction, col_axis)
    desired_up, desired_left = desired_screen_directions(
        slice_axis, convention=convention
    )

    # Image origin='upper': row 0 at top, col 0 at left. Screen-up is therefore
    # -row_dir and screen-left is -col_dir. Flip when the increasing index axis
    # already points the wrong way relative to the target.
    if float(np.dot(row_dir, desired_up)) > 0.0:
        data = np.flipud(data)
    if float(np.dot(col_dir, desired_left)) > 0.0:
        data = np.fliplr(data)
    return data


def display_array_axis_flips(
    direction: Optional[np.ndarray],
    *,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> Tuple[bool, bool, bool]:
    """
    Per-array-axis flips that make all three orthogonal views match ``convention``.

    For each orthogonal plane the matplotlib helper may flipud / fliplr. Those
    decisions are consistent across planes for a given array axis, so a single
    ``(flip_z, flip_y, flip_x)`` triple applied to the full volume matches
    :func:`orient_slice_for_display` on every plane.

    Args:
        direction: ``(3, 3)`` LPS direction matrix, or ``None`` (no flips).
        convention: Display convention (see module docstring).

    Returns:
        Booleans for NumPy axes ``(0, 1, 2)`` i.e. ``(z, y, x)``.
    """
    convention = normalize_display_convention(convention)
    if convention == "native" or direction is None:
        return (False, False, False)

    flips = [False, False, False]
    for slice_axis in (0, 1, 2):
        row_axis, col_axis = display_slice_row_col_axes(
            slice_axis, direction=direction, convention=convention
        )
        row_dir = array_axis_lps_direction(direction, row_axis)
        col_dir = array_axis_lps_direction(direction, col_axis)
        desired_up, desired_left = desired_screen_directions(
            slice_axis, convention=convention
        )
        if float(np.dot(row_dir, desired_up)) > 0.0:
            flips[row_axis] = True
        if float(np.dot(col_dir, desired_left)) > 0.0:
            flips[col_axis] = True
    return (flips[0], flips[1], flips[2])


def radiological_array_axis_flips(
    direction: Optional[np.ndarray],
) -> Tuple[bool, bool, bool]:
    """
    Per-array-axis flips for the default radiological convention.

    Equivalent to :func:`display_array_axis_flips` with
    ``convention=\"radiological\"``.
    """
    return display_array_axis_flips(direction, convention="radiological")


def volume_display_flips(
    direction: Optional[Sequence[float]],
    *,
    ndim: int,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
    preserve_axial_index: bool = True,
) -> Tuple[bool, ...]:
    """
    Whole-volume flips for interactive viewers (napari).

    Args:
        direction: SimpleITK flattened 3x3 direction, or ``None`` (LPS identity
            for 3D via :func:`direction_matrix`).
        ndim: Array dimensionality (2 or 3).
        convention: Display convention.
        preserve_axial_index: When ``True`` (default), never flip array axis 0
            (``z``). Axial slider indices then match file order / ITK-SNAP /
            matplotlib axis-0 indices. In-plane A-P and L-R still follow
            ``convention``. Set ``False`` only when a caller explicitly wants
            the full three-plane volume remap (including inverted z indices).

    Returns:
        Booleans for each array axis (length ``ndim``).
    """
    convention = normalize_display_convention(convention)
    if ndim != 3:
        return tuple(False for _ in range(ndim))
    if convention == "native":
        return (False, False, False)

    matrix = direction_matrix(direction, ndim=ndim)
    flips = list(display_array_axis_flips(matrix, convention=convention))
    if preserve_axial_index:
        flips[0] = False
    return (flips[0], flips[1], flips[2])


def apply_radiological_flips(
    volume: np.ndarray,
    flips: Sequence[bool],
) -> np.ndarray:
    """
    Flip ``volume`` along axes marked ``True`` in ``flips``.

    Args:
        volume: 2D or 3D array (``(y, x)`` or ``(z, y, x)``).
        flips: Booleans aligned with array axes (length ``>= volume.ndim``;
            only the first ``volume.ndim`` entries are used).

    Returns:
        Flipped array (a view when possible; callers that need a writeable
        buffer should copy explicitly).
    """
    data = np.asarray(volume)
    if data.ndim == 0:
        return data
    axes = [axis for axis, flip in enumerate(flips[: data.ndim]) if flip]
    if not axes:
        return data
    return np.flip(data, axis=tuple(axes))


def array_from_display_input(value: object) -> np.ndarray:
    """
    Unwrap a display volume from an array, ``ImageVolume``, or label map.

    ``plot_habitat_overlay`` callers often pass ``subject.image(...).data``
    and drop ``direction`` / ``spacing``. Accepting the volume object itself
    (or a :class:`~habit.contracts.habitat.HabitatMap`) keeps geometry
    attached so coronal / sagittal panels can be oriented correctly.

    Args:
        value: NumPy array, object with ``.data`` (``ImageVolume``), or
            object with ``.label_array`` (``HabitatMap`` / supervoxel map).

    Returns:
        The voxel array (not yet squeezed to 2D/3D).

    Raises:
        HABITAPIError: When ``value`` is ``None``.
    """
    if value is None:
        raise HABITAPIError("display input must not be None.")
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    data_attr = getattr(value, "data", None)
    if data_attr is not None:
        return np.asarray(data_attr)
    label_attr = getattr(value, "label_array", None)
    if label_attr is not None:
        return np.asarray(label_attr)
    return np.asarray(value)


def display_geometry_from_input(
    value: object,
) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
    """
    Read SimpleITK ``(direction, spacing)`` from a volume-like object.

    Prefers ``.geometry.direction`` / ``.geometry.spacing`` (contracts
    :class:`~habit.contracts.geometry.Geometry`), then ``.direction`` /
    ``.spacing`` on :class:`~habit.api.image.ImageVolume`. Arrays have no
    geometry; the caller then falls back to LPS identity.

    Args:
        value: Array or volume-like object.

    Returns:
        ``(direction, spacing)`` tuples, or ``(None, None)`` when absent.
    """
    if value is None or isinstance(value, np.ndarray):
        return None, None
    geometry = getattr(value, "geometry", None)
    if geometry is not None:
        direction = getattr(geometry, "direction", None)
        spacing = getattr(geometry, "spacing", None)
        if direction is not None and spacing is not None:
            return (
                tuple(float(v) for v in direction),
                tuple(float(v) for v in spacing),
            )
    direction = getattr(value, "direction", None)
    spacing = getattr(value, "spacing", None)
    if direction is not None and spacing is not None:
        return (
            tuple(float(v) for v in direction),
            tuple(float(v) for v in spacing),
        )
    return None, None


def _is_label_like_volume(value: object) -> bool:
    """
    Return True when ``value`` looks like a mask / habitat / label map.

    Used to break image-vs-mask direction conflicts: the labelled anatomy
    (ITK-SNAP / 3D Slicer mask) is the geometry that matches what the user
    drew, even when the intensity volume's direction cosines disagree.
    """
    if value is None or isinstance(value, np.ndarray):
        return False
    if getattr(value, "label_array", None) is not None:
        return True
    if getattr(value, "roi_name", None) is not None:
        return True
    labels = getattr(value, "labels", None)
    return isinstance(labels, (tuple, list))


def _directions_disagree(
    first: Sequence[float],
    second: Sequence[float],
    *,
    atol: float = 1e-5,
) -> bool:
    """Return True when two flattened direction cosine tuples differ."""
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        return True
    return not np.allclose(a, b, atol=atol)


def resolve_display_geometry(
    *volumes: object,
    direction: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
    """
    Resolve display ``direction`` / ``spacing``, preferring explicit kwargs.

    Walks ``volumes`` and collects attached geometry. Explicit ``direction``
    / ``spacing`` always win so callers can override stored metadata.

    When two volumes disagree on ``direction`` (common on demo LAP: the
    intensity NRRD claims LPS identity while the mask has ``+z = Inferior``
    and matches the anatomy), a warning is emitted and the **mask / label**
    geometry is used. That pairing puts superior toward row 0 on coronal
    and sagittal after :func:`orient_slice_for_display`.

    Args:
        volumes: Volume-like objects that may carry geometry.
        direction: Optional SimpleITK flattened direction cosines.
        spacing: Optional SimpleITK spacing ``(x, y[, z])``.

    Returns:
        ``(direction, spacing)`` each either a tuple or ``None``.
    """
    resolved_direction: Optional[Tuple[float, ...]] = (
        tuple(float(v) for v in direction) if direction is not None else None
    )
    resolved_spacing: Optional[Tuple[float, ...]] = (
        tuple(float(v) for v in spacing) if spacing is not None else None
    )
    if resolved_direction is not None and resolved_spacing is not None:
        return resolved_direction, resolved_spacing

    candidates: List[Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]], bool]] = []
    for volume in volumes:
        found_dir, found_sp = display_geometry_from_input(volume)
        if found_dir is None and found_sp is None:
            continue
        candidates.append((found_dir, found_sp, _is_label_like_volume(volume)))

    if resolved_direction is None:
        dirs = [item[0] for item in candidates if item[0] is not None]
        if len(dirs) >= 2 and any(
            _directions_disagree(dirs[0], other) for other in dirs[1:]
        ):
            warnings.warn(
                "Display geometry conflict: image/anatomy direction does not "
                "match mask/label direction. Using the mask/label direction "
                "so coronal/sagittal superior-up follows the labelled "
                "anatomy. Pass direction= to override.",
                UserWarning,
                stacklevel=2,
            )
            label_dirs = [item[0] for item in candidates if item[2] and item[0] is not None]
            resolved_direction = label_dirs[-1] if label_dirs else dirs[0]
        elif dirs:
            resolved_direction = dirs[0]

    if resolved_spacing is None:
        # Prefer spacing from the same volume that won the direction choice
        # (label-like when directions conflict), else the first spacing found.
        if resolved_direction is not None:
            for found_dir, found_sp, is_label in candidates:
                if (
                    found_dir is not None
                    and found_sp is not None
                    and not _directions_disagree(found_dir, resolved_direction)
                ):
                    if is_label or resolved_spacing is None:
                        resolved_spacing = found_sp
                        if is_label:
                            break
        if resolved_spacing is None:
            for _found_dir, found_sp, _is_label in candidates:
                if found_sp is not None:
                    resolved_spacing = found_sp
                    break
    return resolved_direction, resolved_spacing


def plane_spacings_mm(
    spacing_xyz: Sequence[float],
    *,
    slice_axis: int,
    ndim: int,
    direction: Optional[np.ndarray] = None,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> Tuple[float, float]:
    """
    Return ``(spacing_row_mm, spacing_col_mm)`` for a display plane.

    When ``direction`` is set, row/col follow :func:`display_slice_row_col_axes`
    (including an SI transpose) so physical extent matches the oriented slice.

    Args:
        spacing_xyz: SimpleITK spacing ``(x, y[, z])``.
        slice_axis: NumPy axis removed by ``np.take`` (ignored when ``ndim==2``).
        ndim: Array dimensionality.
        direction: Optional ``(3, 3)`` LPS direction matrix.
        convention: Display convention used for the SI-transpose decision.

    Returns:
        Physical size of one array row and one array column in millimetres.
    """
    if ndim == 2:
        return float(spacing_xyz[1]), float(spacing_xyz[0])
    row_axis, col_axis = display_slice_row_col_axes(
        slice_axis, direction=direction, convention=convention
    )
    sitk_row = (2, 1, 0)[int(row_axis)]
    sitk_col = (2, 1, 0)[int(col_axis)]
    return float(spacing_xyz[sitk_row]), float(spacing_xyz[sitk_col])


def imshow_physical_extent(
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

    Returns ``(left, right, bottom, top)`` with **bottom < top** (``0`` to
    physical height). Combined with ``origin='upper'``, matplotlib maps
    array row 0 to ``top`` — after :func:`orient_slice_for_display` that is
    superior (coronal/sagittal) or anterior (axial).

    An inverted extent (``top < bottom``) used to keep row 0 at the top of
    an inverted y-axis. That fights ``ax.set_aspect('equal')`` on tall
    coronal / sagittal panels (thick-slice z) and can silently flip
    superior/inferior on screen while axial (near-square) still looks right.

    Args:
        shape_hw: ``(n_rows, n_cols)`` of the 2D extract after orientation.
        spacing_xyz: SimpleITK spacing ``(x, y[, z])``.
        slice_axis: NumPy axis removed by ``np.take``.
        ndim: Array dimensionality of the parent volume.
        direction: Optional ``(3, 3)`` LPS direction (same as the orient call)
            so a transposed coronal/sagittal keeps the correct mm spacing.
        convention: Display convention used for the SI-transpose decision.

    Returns:
        Extent ``(left, right, bottom, top)`` in millimetres.

    Raises:
        HABITAPIError: When the slice shape is not positive.
    """
    nrows, ncols = int(shape_hw[0]), int(shape_hw[1])
    if nrows <= 0 or ncols <= 0:
        raise HABITAPIError("slice shape must be positive for imshow extent.")
    spacing_row, spacing_col = plane_spacings_mm(
        spacing_xyz,
        slice_axis=slice_axis,
        ndim=ndim,
        direction=direction,
        convention=convention,
    )
    return (
        0.0,
        float(ncols) * spacing_col,
        0.0,
        float(nrows) * spacing_row,
    )
