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
"""Radiological orientation helpers for habitat visualization.

Shared by the matplotlib overlay (:mod:`habit.viz.habitat_overlay`) and the
optional napari viewer (:mod:`habit.viz.habitat_napari`).

``ImageVolume.data`` / SimpleITK arrays are NumPy ``(z, y, x)`` while
``direction`` columns are SimpleITK index axes ``(x, y, z)`` in an LPS world
(+x Left, +y Posterior, +z Superior). Raw array order therefore often shows
posterior/inferior toward the top of a canvas that uses image coordinates
(row 0 at the top). These helpers compute the per-axis flips that put
axial / coronal / sagittal panels into standard radiological layout.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError

__all__ = [
    "DEFAULT_RAS_DIRECTION",
    "array_axis_lps_direction",
    "apply_radiological_flips",
    "desired_screen_directions",
    "direction_matrix",
    "orient_slice_for_display",
    "radiological_array_axis_flips",
    "slice_row_col_axes",
]

# SimpleITK / ITK physical axis meanings (LPS world): +x Left, +y Posterior, +z Superior.
_LPS_RIGHT = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
_LPS_ANTERIOR = np.array([0.0, -1.0, 0.0], dtype=np.float64)
_LPS_SUPERIOR = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Default when callers pass arrays without geometry: NIfTI via SimpleITK is usually RAS.
# Flattened row-major 3x3, same layout as ``ImageVolume.direction`` / ``sitk.GetDirection``.
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


def direction_matrix(
    direction: Optional[Sequence[float]],
    *,
    ndim: int,
) -> Optional[np.ndarray]:
    """
    Parse a SimpleITK flattened direction into a square matrix.

    Args:
        direction: Flattened row-major direction cosines (``ndim**2`` values),
            in SimpleITK index order ``(x, y, z)``. ``None`` keeps the RAS
            default used for bare NumPy arrays from NIfTI.
        ndim: Array dimensionality (2 or 3). Orientation flips apply only to 3D.

    Returns:
        ``(3, 3)`` float matrix for 3D volumes, or ``None`` when ``ndim != 3``.

    Raises:
        HABITAPIError: When ``direction`` has the wrong length.
    """
    if ndim != 3:
        return None
    values = (
        DEFAULT_RAS_DIRECTION if direction is None else tuple(float(v) for v in direction)
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


def desired_screen_directions(slice_axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radiological screen ``(up, left)`` directions in LPS world for a slice axis.

    - Axial-like (axis 0): anterior up, patient right on viewer's left.
    - Coronal-like (axis 1): superior up, patient right on viewer's left.
    - Sagittal-like (axis 2): superior up, anterior on viewer's left.
    """
    if slice_axis == 0:
        return _LPS_ANTERIOR.copy(), _LPS_RIGHT.copy()
    if slice_axis == 1:
        return _LPS_SUPERIOR.copy(), _LPS_RIGHT.copy()
    if slice_axis == 2:
        return _LPS_SUPERIOR.copy(), _LPS_ANTERIOR.copy()
    raise HABITAPIError(f"axis must be 0, 1, or 2; got {slice_axis}.")


def orient_slice_for_display(
    slice_2d: np.ndarray,
    *,
    slice_axis: int,
    direction: Optional[np.ndarray],
) -> np.ndarray:
    """
    Flip a 2D slice so image coordinates (origin upper-left) match radiology.

    Without this step, RAS volumes from SimpleITK show posterior/inferior toward
    the top of the panel and patient-left on the viewer's left, because array
    index 0 is drawn at the top-left while anatomical +Y/+X increase anterior /
    right.
    """
    data = np.asarray(slice_2d)
    if direction is None or data.ndim != 2:
        return data

    row_axis, col_axis = slice_row_col_axes(slice_axis)
    row_dir = array_axis_lps_direction(direction, row_axis)
    col_dir = array_axis_lps_direction(direction, col_axis)
    desired_up, desired_left = desired_screen_directions(slice_axis)

    # Image origin='upper': row 0 at top, col 0 at left. Screen-up is therefore
    # -row_dir and screen-left is -col_dir. Flip when the increasing index axis
    # already points the wrong way relative to the radiological target.
    if float(np.dot(row_dir, desired_up)) > 0.0:
        data = np.flipud(data)
    if float(np.dot(col_dir, desired_left)) > 0.0:
        data = np.fliplr(data)
    return data


def radiological_array_axis_flips(
    direction: Optional[np.ndarray],
) -> Tuple[bool, bool, bool]:
    """
    Per-array-axis flips that make all three orthogonal views radiological.

    For each orthogonal plane the matplotlib helper may flipud / fliplr. Those
    decisions are consistent across planes for a given array axis, so a single
    ``(flip_z, flip_y, flip_x)`` triple applied to the full volume matches
    :func:`orient_slice_for_display` on every plane. Napari can then show the
    flipped volume with ordinary image coordinates (row 0 at top).

    Args:
        direction: ``(3, 3)`` LPS direction matrix, or ``None`` (no flips).

    Returns:
        Booleans for NumPy axes ``(0, 1, 2)`` i.e. ``(z, y, x)``.
    """
    if direction is None:
        return (False, False, False)

    flips = [False, False, False]
    for slice_axis in (0, 1, 2):
        row_axis, col_axis = slice_row_col_axes(slice_axis)
        row_dir = array_axis_lps_direction(direction, row_axis)
        col_dir = array_axis_lps_direction(direction, col_axis)
        desired_up, desired_left = desired_screen_directions(slice_axis)
        if float(np.dot(row_dir, desired_up)) > 0.0:
            flips[row_axis] = True
        if float(np.dot(col_dir, desired_left)) > 0.0:
            flips[col_axis] = True
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
