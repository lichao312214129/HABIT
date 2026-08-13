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
"""Tests for the L0 image perturbation kernels.

The geometric kernels are pinned down with a bright-voxel probe: a
translation must move the probe by exactly the requested physical offset,
and a +90-degree rotation about z must move a probe at +x to +y (content
rotation, counterclockwise looking down the positive axis).
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from habit.kernels.image_perturbation import (
    add_gaussian_noise,
    estimate_noise_sigma,
    rigid_transform_image,
    rotate_image,
    translate_image,
)


def _probe_image(size: int = 21, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """21^3 float64 image with a single bright voxel at (z, y, x) = (10, 10, 15)."""
    array = np.zeros((size, size, size), dtype=np.float64)
    array[size // 2, size // 2, size // 2 + 5] = 100.0
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    return image


def _centre_of_mass_index(image: sitk.Image) -> np.ndarray:
    """Centre of mass of the array in (z, y, x) index coordinates."""
    array = sitk.GetArrayFromImage(image)
    total = array.sum()
    grids = np.indices(array.shape)
    return np.array([(array * grids[axis]).sum() / total for axis in range(3)])


class TestEstimateNoiseSigma:
    def test_chang_recovers_known_sigma(self) -> None:
        rng = np.random.default_rng(0)
        array = rng.normal(100.0, 5.0, size=(16, 64, 64))
        sigma = estimate_noise_sigma(array, method="chang")
        assert sigma == pytest.approx(5.0, rel=0.2)

    def test_chang_constant_image_is_zero(self) -> None:
        assert estimate_noise_sigma(np.full((8, 9, 9), 42.0)) == pytest.approx(0.0)

    def test_chang_accepts_odd_sizes(self) -> None:
        # Edge padding pulls the median slightly low; on a reasonably sized
        # array the effect stays well inside the estimator's own variability.
        rng = np.random.default_rng(1)
        array = rng.normal(0.0, 3.0, size=(9, 65, 63))
        sigma = estimate_noise_sigma(array)
        assert sigma == pytest.approx(3.0, rel=0.2)

    def test_chang_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="at least two axes"):
            estimate_noise_sigma(np.zeros(16))

    def test_roi_std(self) -> None:
        rng = np.random.default_rng(2)
        array = rng.normal(0.0, 4.0, size=(10, 20, 20))
        mask = np.zeros_like(array, dtype=np.int64)
        mask[2:8, 5:15, 5:15] = 1
        sigma = estimate_noise_sigma(array, mask=mask, method="roi_std")
        assert sigma == pytest.approx(np.std(array[mask > 0]), rel=1e-12)

    def test_roi_std_empty_mask_raises(self) -> None:
        with pytest.raises(ValueError, match="no voxels"):
            estimate_noise_sigma(
                np.zeros((4, 4, 4)), mask=np.zeros((4, 4, 4)), method="roi_std"
            )

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            estimate_noise_sigma(np.zeros((4, 4, 4)), method="bogus")


class TestAddGaussianNoise:
    def test_zero_sigma_returns_unmodified_copy(self) -> None:
        array = np.arange(12.0).reshape(3, 4)
        result = add_gaussian_noise(array, 0.0, np.random.default_rng(0))
        np.testing.assert_array_equal(result, array)
        assert result is not array

    def test_seeded_noise_is_reproducible(self) -> None:
        array = np.zeros((5, 6, 7))
        first = add_gaussian_noise(array, 2.0, np.random.default_rng(42))
        second = add_gaussian_noise(array, 2.0, np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)

    def test_empirical_sigma_matches(self) -> None:
        rng = np.random.default_rng(3)
        array = np.full((32, 64, 64), 50.0)
        noisy = add_gaussian_noise(array, 7.0, rng)
        assert float(np.std(noisy - array)) == pytest.approx(7.0, rel=0.05)

    def test_mask_restricts_noise(self) -> None:
        array = np.zeros((6, 6, 6))
        mask = np.zeros((6, 6, 6), dtype=np.int64)
        mask[0, 0, 0] = 1
        noisy = add_gaussian_noise(array, 5.0, np.random.default_rng(4), mask=mask)
        assert noisy[0, 0, 0] != 0.0
        np.testing.assert_array_equal(noisy[mask == 0], array[mask == 0])

    def test_round_to_int(self) -> None:
        array = np.full((4, 5, 6), 10.0)
        noisy = add_gaussian_noise(
            array, 3.0, np.random.default_rng(5), round_to_int=True
        )
        np.testing.assert_array_equal(noisy, np.rint(noisy))


class TestTranslateImage:
    def test_content_moves_by_requested_physical_offset(self) -> None:
        spacing = (2.0, 3.0, 4.0)
        image = _probe_image(spacing=spacing)
        shifted = translate_image(image, (0.5, 0.0, 0.0), interpolator="linear")
        before = _centre_of_mass_index(image)
        after = _centre_of_mass_index(shifted)
        # (z, y, x) index space: a +0.5-voxel x shift moves the probe by
        # +0.5 along the LAST array axis.
        np.testing.assert_allclose(after - before, (0.0, 0.0, 0.5), atol=1e-6)

    def test_grid_is_preserved(self) -> None:
        image = _probe_image(spacing=(2.0, 3.0, 4.0))
        shifted = translate_image(image, (0.3, -0.2, 0.1))
        assert shifted.GetSize() == image.GetSize()
        assert shifted.GetSpacing() == image.GetSpacing()
        assert shifted.GetOrigin() == image.GetOrigin()

    def test_bad_shift_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="3 components"):
            translate_image(_probe_image(), (1.0, 2.0))

    def test_nearest_keeps_label_values(self) -> None:
        image = _probe_image()
        shifted = translate_image(image, (0.4, 0.0, 0.0), interpolator="nearest")
        values = np.unique(sitk.GetArrayFromImage(shifted))
        assert set(values.tolist()) <= {0.0, 100.0}

    def test_unknown_interpolator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown interpolator"):
            translate_image(_probe_image(), (0.1, 0.0, 0.0), interpolator="cubic")


class TestRotateImage:
    def test_plus_90_degrees_about_z_moves_plus_x_to_plus_y(self) -> None:
        image = _probe_image()
        rotated = rotate_image(image, 90.0, axis="z", interpolator="nearest")
        array = sitk.GetArrayFromImage(rotated)
        peak = np.unravel_index(np.argmax(array), array.shape)
        # Probe started at (z, y, x) = (10, 10, 15): +x content rotated by
        # +90 degrees about z lands at +y, i.e. index (10, 15, 10).
        assert peak == (10, 15, 10)

    def test_zero_angle_is_identity(self) -> None:
        # Linear interpolation is interpolating (passes through the sample
        # points), so a zero rotation must reproduce the image exactly;
        # bspline is approximating and would leave float dust.
        image = _probe_image()
        rotated = rotate_image(image, 0.0, axis="z", interpolator="linear")
        np.testing.assert_allclose(
            sitk.GetArrayFromImage(rotated), sitk.GetArrayFromImage(image)
        )

    def test_grid_is_preserved(self) -> None:
        image = _probe_image(spacing=(2.0, 3.0, 4.0))
        rotated = rotate_image(image, 0.5, axis="y")
        assert rotated.GetSize() == image.GetSize()
        assert rotated.GetSpacing() == image.GetSpacing()
        assert rotated.GetOrigin() == image.GetOrigin()

    def test_bad_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            rotate_image(_probe_image(), 1.0, axis="w")


class TestRigidTransformImage:
    def test_pure_translation_matches_translate_image(self) -> None:
        image = _probe_image()
        shift = (0.0, 0.0, 1.0)
        sequential = translate_image(image, shift, interpolator="linear")
        composed = rigid_transform_image(
            image, shift, angle_degrees=0.0, interpolator="linear"
        )
        np.testing.assert_allclose(
            sitk.GetArrayFromImage(composed),
            sitk.GetArrayFromImage(sequential),
            atol=1e-6,
        )

    def test_grid_is_preserved(self) -> None:
        image = _probe_image(spacing=(2.0, 3.0, 4.0))
        moved = rigid_transform_image(
            image, (0.5, -0.25, 0.0), angle_degrees=0.5, interpolator="linear"
        )
        assert moved.GetSize() == image.GetSize()
        assert moved.GetSpacing() == image.GetSpacing()
        assert moved.GetOrigin() == image.GetOrigin()

    def test_bad_shift_raises(self) -> None:
        with pytest.raises(ValueError, match="3 components"):
            rigid_transform_image(_probe_image(), (1.0, 2.0), 0.5)
