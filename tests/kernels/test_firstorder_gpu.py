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
"""
Parity tests: TorchRadiomicsFirstOrder (GPU) vs radiomics.firstorder.

Feature values must match the PyRadiomics reference within tight float64
tolerance (reduction order differs between numpy and torch, so the
comparison is allclose, not exact equality). Runs on CPU torch always and
on CUDA when available.
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from radiomics import firstorder

from habit.kernels.radiomics.torchradiomics.TorchRadiomicsFirstOrder import (
    TorchRadiomicsFirstOrder,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# All non-deprecated first-order features.
FEATURES = [
    "Energy",
    "TotalEnergy",
    "Entropy",
    "Minimum",
    "10Percentile",
    "90Percentile",
    "Maximum",
    "Mean",
    "Median",
    "InterquartileRange",
    "Range",
    "MeanAbsoluteDeviation",
    "RobustMeanAbsoluteDeviation",
    "RootMeanSquared",
    "Skewness",
    "Kurtosis",
    "Variance",
    "Uniformity",
]


def _random_case(rng: np.random.Generator, shape, mask_frac: float):
    """Random float image and a random boolean mask."""
    image = rng.normal(loc=300.0, scale=80.0, size=shape)
    mask = rng.random(shape) < mask_frac
    # Guarantee at least a few masked voxels.
    if mask.sum() < 4:
        mask.ravel()[:4] = True
    return image, mask


def _make_pair(image, mask, device, voxel_based: bool, kernel_radius: int = 3):
    """Build the PyRadiomics reference and the torch calculator with identical settings."""
    sitk_img = sitk.GetImageFromArray(image)
    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    kwargs = dict(binWidth=25.0, voxelArrayShift=300)
    if voxel_based:
        kwargs.update(voxelBased=True, kernelRadius=kernel_radius)
    ref = firstorder.RadiomicsFirstOrder(sitk_img, sitk_mask, **kwargs)
    new = TorchRadiomicsFirstOrder(
        sitk_img, sitk_mask, device=device, dtype=torch.float64, **kwargs
    )
    return ref, new


def _assert_features_same(ref, new, coords, features=FEATURES):
    """Run one calculation batch on both calculators and compare every feature."""
    ref._initCalculation(coords)
    new._initCalculation(coords)
    for name in features:
        v_ref = np.asarray(
            getattr(ref, "get%sFeatureValue" % name)(), dtype=np.float64
        ).reshape(-1)
        v_new = np.asarray(
            getattr(new, "get%sFeatureValue" % name)(), dtype=np.float64
        ).reshape(-1)
        assert v_ref.shape == v_new.shape, (
            f"{name}: shape {v_new.shape} vs reference {v_ref.shape}"
        )
        assert np.allclose(v_new, v_ref, rtol=1e-9, atol=1e-9, equal_nan=True), (
            f"{name}: max abs diff "
            f"{np.nanmax(np.abs(v_new - v_ref)) if v_ref.size else 0.0}"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_segment_mode(device):
    rng = np.random.default_rng(42)
    image, mask = _random_case(rng, (8, 9, 10), 0.5)
    ref, new = _make_pair(image, mask, device, voxel_based=False)
    _assert_features_same(ref, new, None)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel_radius", [1, 2, 3])
def test_voxel_mode_radii(device, kernel_radius):
    rng = np.random.default_rng(43)
    image, mask = _random_case(rng, (10, 11, 12), 0.5)
    ref, new = _make_pair(image, mask, device, True, kernel_radius)
    coords = np.array(np.where(mask), dtype=np.int64)
    _assert_features_same(ref, new, coords)


@pytest.mark.parametrize("device", DEVICES)
def test_voxel_mode_mask_touches_border(device):
    """Mask at the array border forces kernel windows into the NaN padding."""
    rng = np.random.default_rng(44)
    image, mask = _random_case(rng, (8, 8, 8), 0.0)
    mask[:3, :3, :3] = True
    ref, new = _make_pair(image, mask, device, True, 2)
    coords = np.array(np.where(mask), dtype=np.int64)
    _assert_features_same(ref, new, coords)


@pytest.mark.parametrize("device", DEVICES)
def test_voxel_mode_flat_region(device):
    """Constant-intensity ROI hits the m2 == 0 guards in Skewness/Kurtosis."""
    rng = np.random.default_rng(45)
    image = rng.normal(loc=300.0, scale=80.0, size=(9, 9, 9))
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[3:6, 3:6, 3:6] = True
    image[mask] = 500.0  # flat region
    ref, new = _make_pair(image, mask, device, True, 1)
    coords = np.array(np.where(mask), dtype=np.int64)
    _assert_features_same(ref, new, coords)


@pytest.mark.parametrize("device", DEVICES)
def test_external_target_assignment(device):
    """supervoxel_batch assigns targetVoxelArray directly; the setter must
    upload it so getters reflect the new array (histogram features use the
    p_i from the last _initCalculation in both implementations)."""
    rng = np.random.default_rng(46)
    image, mask = _random_case(rng, (8, 9, 10), 0.5)
    ref, new = _make_pair(image, mask, device, voxel_based=False)
    ref._initCalculation(None)
    new._initCalculation(None)

    n_roi = ref.targetVoxelArray.shape[1]
    batch = np.full((2, n_roi + 3), np.nan)
    batch[0, :n_roi] = ref.targetVoxelArray[0]
    batch[1, :n_roi] = ref.targetVoxelArray[0] * 1.5
    ref.targetVoxelArray = batch
    new.targetVoxelArray = batch

    for name in FEATURES:
        v_ref = np.asarray(
            getattr(ref, "get%sFeatureValue" % name)(), dtype=np.float64
        ).reshape(-1)
        v_new = np.asarray(
            getattr(new, "get%sFeatureValue" % name)(), dtype=np.float64
        ).reshape(-1)
        assert np.allclose(v_new, v_ref, rtol=1e-9, atol=1e-9, equal_nan=True), name


@pytest.mark.parametrize("device", DEVICES)
def test_voxel_mode_multiple_batches(device):
    """Two consecutive batches must give the same values as one full pass."""
    rng = np.random.default_rng(47)
    image, mask = _random_case(rng, (10, 10, 10), 0.4)
    ref, new = _make_pair(image, mask, device, True, 2)
    coords = np.array(np.where(mask), dtype=np.int64)
    half = coords.shape[1] // 2

    ref._initCalculation(coords)
    new._initCalculation(coords)
    full_ref = np.asarray(ref.getMeanFeatureValue(), dtype=np.float64)
    full_new = np.asarray(new.getMeanFeatureValue(), dtype=np.float64)

    new._initCalculation(coords[:, :half])
    first_half = np.asarray(new.getMeanFeatureValue(), dtype=np.float64)
    new._initCalculation(coords[:, half:])
    second_half = np.asarray(new.getMeanFeatureValue(), dtype=np.float64)

    assert np.allclose(full_new, full_ref, rtol=1e-9, atol=1e-9)
    assert np.allclose(
        np.concatenate([first_half, second_half]), full_ref, rtol=1e-9, atol=1e-9
    )
