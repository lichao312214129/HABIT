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
"""Unit tests for habit.viz.orientation display conventions and flip policy."""

from __future__ import annotations

import numpy as np
import pytest

from habit.exceptions import HABITAPIError
from habit.viz.habitat_napari import napari_display_flips
from habit.viz.orientation import (
    DEFAULT_NATIVE_DIRECTION,
    DEFAULT_RAS_DIRECTION,
    apply_radiological_flips,
    direction_matrix,
    display_array_axis_flips,
    normalize_display_convention,
    orient_slice_for_display,
    radiological_array_axis_flips,
    volume_display_flips,
)


_LPS = DEFAULT_NATIVE_DIRECTION
_RAS = DEFAULT_RAS_DIRECTION


def test_direction_none_defaults_to_lps_identity_not_ras() -> None:
    """Omitted direction must match ImageVolume.from_array (LPS identity)."""
    matrix = direction_matrix(None, ndim=3)
    assert matrix is not None
    np.testing.assert_allclose(matrix, np.eye(3))
    np.testing.assert_allclose(
        matrix.ravel(), np.asarray(DEFAULT_NATIVE_DIRECTION)
    )
    assert not np.allclose(matrix.ravel(), np.asarray(DEFAULT_RAS_DIRECTION))


def test_lps_radiological_full_volume_flips_z_only() -> None:
    """LPS identity: full three-plane remap flips z for coronal/sagittal S-up.

    Axial in-plane needs no A-P / L-R flip. Older comments claiming 'no
    radiological flips' for LPS were misleading — z still flips when the
    full-volume remap is requested.
    """
    matrix = direction_matrix(_LPS, ndim=3)
    assert radiological_array_axis_flips(matrix) == (True, False, False)
    assert display_array_axis_flips(matrix, convention="radiological") == (
        True,
        False,
        False,
    )


def test_ras_radiological_flips_all_axes() -> None:
    matrix = direction_matrix(_RAS, ndim=3)
    assert radiological_array_axis_flips(matrix) == (True, True, True)


def test_volume_display_flips_preserve_axial_index_for_lps_and_ras() -> None:
    """Napari default: clear z flip so axial slider matches file indices."""
    assert volume_display_flips(
        _LPS, ndim=3, convention="radiological", preserve_axial_index=True
    ) == (False, False, False)
    assert volume_display_flips(
        _RAS, ndim=3, convention="radiological", preserve_axial_index=True
    ) == (False, True, True)
    # Explicit full remap still available for callers that opt in.
    assert volume_display_flips(
        _LPS, ndim=3, convention="radiological", preserve_axial_index=False
    ) == (True, False, False)


def test_direction_none_uses_lps_volume_flips() -> None:
    """None direction → LPS identity → no in-plane napari flips (preserve z)."""
    assert napari_display_flips(None, ndim=3) == (False, False, False)
    assert volume_display_flips(None, ndim=3) == (False, False, False)


def test_native_convention_never_flips() -> None:
    matrix = direction_matrix(_RAS, ndim=3)
    assert display_array_axis_flips(matrix, convention="native") == (
        False,
        False,
        False,
    )
    assert volume_display_flips(_RAS, ndim=3, convention="native") == (
        False,
        False,
        False,
    )
    slice_2d = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = orient_slice_for_display(
        slice_2d, slice_axis=0, direction=matrix, convention="native"
    )
    np.testing.assert_array_equal(out, slice_2d)


def test_neurological_reverses_axial_lr_vs_radiological() -> None:
    """Neurological keeps A-P up rule but reverses L-R on axial."""
    matrix = direction_matrix(_RAS, ndim=3)
    # Patient-right marker at max x under RAS.
    slice_lr = np.zeros((10, 10), dtype=np.float32)
    slice_lr[5, -1] = 1.0
    rad = orient_slice_for_display(
        slice_lr, slice_axis=0, direction=matrix, convention="radiological"
    )
    neuro = orient_slice_for_display(
        slice_lr, slice_axis=0, direction=matrix, convention="neurological"
    )
    _, col_r = np.argwhere(rad == 1.0)[0]
    _, col_n = np.argwhere(neuro == 1.0)[0]
    assert col_r < 5, "radiological: patient-right on viewer's left"
    assert col_n > 5, "neurological: patient-right on viewer's right"
    # Anterior marker stays upper-half for both.
    slice_ap = np.zeros((10, 10), dtype=np.float32)
    slice_ap[-1, 5] = 1.0
    row_r = np.argwhere(
        orient_slice_for_display(
            slice_ap, slice_axis=0, direction=matrix, convention="radiological"
        )
        == 1.0
    )[0, 0]
    row_n = np.argwhere(
        orient_slice_for_display(
            slice_ap, slice_axis=0, direction=matrix, convention="neurological"
        )
        == 1.0
    )[0, 0]
    assert row_r < 5 and row_n < 5


def test_lps_axial_radiological_keeps_ap_unflipped() -> None:
    """LPS axial: posterior↑ index already puts anterior toward top after image coords?"""
    # Under LPS, +y is Posterior. Row 0 at top → top is anterior side of the
    # array (low y). Marker at max y (posterior) must stay in the lower half.
    matrix = direction_matrix(_LPS, ndim=3)
    slice_2d = np.zeros((10, 10), dtype=np.float32)
    slice_2d[-1, 5] = 1.0  # max y = posterior
    out = orient_slice_for_display(
        slice_2d, slice_axis=0, direction=matrix, convention="radiological"
    )
    row, _ = np.argwhere(out == 1.0)[0]
    assert row > 5, "LPS posterior marker stays inferior on screen (anterior up)"
    # And the oriented slice equals the raw extract (no in-plane flip).
    np.testing.assert_array_equal(out, slice_2d)


def test_matplotlib_napari_axial_ap_policy_consistent_for_lps_and_ras() -> None:
    """In-plane A-P for axial must match between matplotlib orient and napari flips."""
    for direction in (_LPS, _RAS):
        matrix = direction_matrix(direction, ndim=3)
        volume = np.zeros((4, 10, 10), dtype=np.float32)
        # Anterior under RAS = max y; under LPS anterior is low y — place both.
        volume[2, -1, 5] = 1.0
        volume[2, 0, 5] = 2.0

        mpl = orient_slice_for_display(
            volume[2], slice_axis=0, direction=matrix, convention="radiological"
        )
        flips = volume_display_flips(
            direction,
            ndim=3,
            convention="radiological",
            preserve_axial_index=True,
        )
        assert flips[0] is False, "axial index must be preserved for napari"
        nap = apply_radiological_flips(volume[2], (flips[1], flips[2]))
        np.testing.assert_array_equal(nap, mpl)

        # Same slice index still contains the markers (z not remapped).
        flipped_vol = apply_radiological_flips(volume, flips)
        np.testing.assert_array_equal(flipped_vol[2], nap)


def test_normalize_display_convention_rejects_unknown() -> None:
    with pytest.raises(HABITAPIError, match="display_convention"):
        normalize_display_convention("diagonal")
    assert normalize_display_convention(None) == "radiological"
    assert normalize_display_convention("Radiological") == "radiological"
