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
(row 0 at the top). These helpers compute the per-axis flips that put
axial / coronal / sagittal panels into the requested display convention.

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
**not** flipped so the axial slider keeps file / ITK-SNAP index semantics
(demo basal tip ~slice 110 stays usable). In-plane A-P / L-R still follow
the same convention as matplotlib.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple

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
    "desired_screen_directions",
    "direction_matrix",
    "display_array_axis_flips",
    "normalize_display_convention",
    "orient_slice_for_display",
    "radiological_array_axis_flips",
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
    Flip a 2D slice so image coordinates (origin upper-left) match ``convention``.

    Used by matplotlib per orthogonal extract. Does not remap which volume
    index was selected — only the 2D panel layout.
    """
    data = np.asarray(slice_2d)
    convention = normalize_display_convention(convention)
    if convention == "native" or direction is None or data.ndim != 2:
        return data

    row_axis, col_axis = slice_row_col_axes(slice_axis)
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
        row_axis, col_axis = slice_row_col_axes(slice_axis)
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
