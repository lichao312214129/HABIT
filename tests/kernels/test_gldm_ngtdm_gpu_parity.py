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
Parity tests: gpumatrices GLDM / NGTDM vs radiomics.cMatrices.

GLDM counts are small integers, compared with strict equality.
NGTDM column 0 (n_i) and column 2 (gray-level label) are exact; column 1
(s_i) is a float64 sum of abs differences and is compared with a tight
allclose (C and torch may differ in the last ulp of a reduction).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from radiomics import cMatrices

from habit.kernels.radiomics.gpumatrices import calculate_gldm, calculate_ngtdm

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _random_case(rng: np.random.Generator, shape, ng: int, mask_frac: float):
    """Random discretised image (levels 1..ng) and a random boolean mask."""
    image = rng.integers(1, ng + 1, size=shape).astype(np.int32)
    mask = rng.random(shape) < mask_frac
    if mask.sum() < 4:
        mask.ravel()[:4] = True
    return image, mask


def _assert_gldm(image, mask, distances, ng, alpha, force2D, force2Ddimension,
                 kernel_radius, voxel_coords, device):
    """Run C extension and GPU GLDM, require identical integer counts."""
    args = [
        image, mask, np.asarray(distances, dtype=np.int32), ng, alpha,
        force2D, force2Ddimension,
    ]
    if voxel_coords is not None:
        args += [kernel_radius, voxel_coords]
    p_c = cMatrices.calculate_gldm(*args)
    p_g = calculate_gldm(
        image, mask, np.asarray(distances, dtype=np.int32), ng,
        alpha=alpha, force2D=bool(force2D), force2Ddimension=force2Ddimension,
        kernelRadius=kernel_radius, voxelCoordinates=voxel_coords,
        device=device, dtype=torch.float64,
    ).cpu().numpy()
    assert p_g.shape == p_c.shape, f"GLDM shape {p_g.shape} vs {p_c.shape}"
    assert np.array_equal(p_g, p_c), (
        f"GLDM mismatch: max abs {np.abs(p_g - p_c).max()}, "
        f"nnz diff {(p_g != p_c).sum()}"
    )


def _assert_ngtdm(image, mask, distances, ng, force2D, force2Ddimension,
                  kernel_radius, voxel_coords, device):
    """Run C extension and GPU NGTDM; n_i / labels exact, s_i tight allclose."""
    args = [
        image, mask, np.asarray(distances, dtype=np.int32), ng,
        force2D, force2Ddimension,
    ]
    if voxel_coords is not None:
        args += [kernel_radius, voxel_coords]
    p_c = cMatrices.calculate_ngtdm(*args)
    p_g = calculate_ngtdm(
        image, mask, np.asarray(distances, dtype=np.int32), ng,
        force2D=bool(force2D), force2Ddimension=force2Ddimension,
        kernelRadius=kernel_radius, voxelCoordinates=voxel_coords,
        device=device, dtype=torch.float64,
    ).cpu().numpy()
    assert p_g.shape == p_c.shape, f"NGTDM shape {p_g.shape} vs {p_c.shape}"
    assert np.array_equal(p_g[:, :, 0], p_c[:, :, 0]), "NGTDM n_i mismatch"
    assert np.array_equal(p_g[:, :, 2], p_c[:, :, 2]), "NGTDM gray-level mismatch"
    assert np.allclose(p_g[:, :, 1], p_c[:, :, 1], rtol=0, atol=1e-12), (
        f"NGTDM s_i mismatch: max abs {np.abs(p_g[:, :, 1] - p_c[:, :, 1]).max()}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_gldm_segment(device):
    rng = np.random.default_rng(42)
    image, mask = _random_case(rng, (6, 7, 8), ng=4, mask_frac=0.6)
    _assert_gldm(image, mask, [1], 4, 0, 0, 0, 0, None, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("alpha", [0, 1, 2])
def test_gldm_voxel_alpha(device, alpha):
    rng = np.random.default_rng(7)
    image, mask = _random_case(rng, (8, 9, 10), ng=5, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_gldm(image, mask, [1], 5, alpha, 0, 0, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_gldm_voxel_force2d(device):
    rng = np.random.default_rng(13)
    image, mask = _random_case(rng, (7, 8, 9), ng=4, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_gldm(image, mask, [1], 4, 0, 1, 0, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_gldm_border_and_multi_distance(device):
    rng = np.random.default_rng(17)
    image = rng.integers(1, 4, size=(5, 6, 7)).astype(np.int32)
    mask = np.ones(image.shape, dtype=bool)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_gldm(image, mask, [1, 2], 3, 1, 0, 0, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_ngtdm_segment(device):
    rng = np.random.default_rng(42)
    image, mask = _random_case(rng, (6, 7, 8), ng=4, mask_frac=0.6)
    _assert_ngtdm(image, mask, [1], 4, 0, 0, 0, None, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel_radius", [1, 2, 3])
def test_ngtdm_voxel_radii(device, kernel_radius):
    rng = np.random.default_rng(7)
    image, mask = _random_case(rng, (9, 10, 11), ng=5, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_ngtdm(image, mask, [1], 5, 0, 0, kernel_radius, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_ngtdm_voxel_force2d(device):
    rng = np.random.default_rng(13)
    image, mask = _random_case(rng, (7, 8, 9), ng=4, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_ngtdm(image, mask, [1], 4, 1, 2, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_ngtdm_isolated_voxels(device):
    """A centre with no masked neighbour must still increment n_i, s_i = 0."""
    image = np.ones((4, 4, 4), dtype=np.int32)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    mask[3, 3, 3] = True  # too far to be a neighbour at distance 1
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_ngtdm(image, mask, [1], 1, 0, 0, 1, coords, device)
    _assert_gldm(image, mask, [1], 1, 0, 0, 0, 1, coords, device)
