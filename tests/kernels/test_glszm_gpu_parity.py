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
Parity tests: gpumatrices.calculate_glszm vs radiomics.cMatrices.

Zone counts are small integers, compared with strict equality. The size
axis is cropped to the largest zone actually found (C wrapper behaviour).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from radiomics import cMatrices

from habit.kernels.radiomics.gpumatrices import calculate_glszm

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _random_case(rng: np.random.Generator, shape, ng: int, mask_frac: float):
    """Random discretised image (levels 1..ng) and a random boolean mask."""
    image = rng.integers(1, ng + 1, size=shape).astype(np.int32)
    mask = rng.random(shape) < mask_frac
    if mask.sum() < 4:
        mask.ravel()[:4] = True
    return image, mask


def _assert_glszm(image, mask, ng, force2D, force2Ddimension,
                  kernel_radius, voxel_coords, device):
    """Run C extension and GPU GLSZM, require identical integer counts."""
    ns = int(mask.sum())
    args = [image, mask, ng, ns, force2D, force2Ddimension]
    if voxel_coords is not None:
        args += [kernel_radius, voxel_coords]
    p_c = cMatrices.calculate_glszm(*args)
    p_g = calculate_glszm(
        image, mask, ng, ns,
        force2D=bool(force2D), force2Ddimension=force2Ddimension,
        kernelRadius=kernel_radius, voxelCoordinates=voxel_coords,
        device=device, dtype=torch.float64,
    ).cpu().numpy()
    assert p_g.shape == p_c.shape, f"GLSZM shape {p_g.shape} vs {p_c.shape}"
    assert np.array_equal(p_g, p_c), (
        f"GLSZM mismatch: max abs {np.abs(p_g - p_c).max()}, "
        f"nnz diff {(p_g != p_c).sum()}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_glszm_segment(device):
    rng = np.random.default_rng(42)
    image, mask = _random_case(rng, (6, 7, 8), ng=4, mask_frac=0.6)
    _assert_glszm(image, mask, 4, 0, 0, 0, None, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel_radius", [1, 2, 3])
def test_glszm_voxel_radii(device, kernel_radius):
    rng = np.random.default_rng(7)
    image, mask = _random_case(rng, (8, 9, 10), ng=5, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_glszm(image, mask, 5, 0, 0, kernel_radius, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_glszm_voxel_force2d(device):
    rng = np.random.default_rng(13)
    image, mask = _random_case(rng, (7, 8, 9), ng=4, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_glszm(image, mask, 4, 1, 0, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_glszm_border(device):
    rng = np.random.default_rng(17)
    image = rng.integers(1, 4, size=(5, 6, 7)).astype(np.int32)
    mask = np.ones(image.shape, dtype=bool)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_glszm(image, mask, 3, 0, 0, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_glszm_single_voxel(device):
    image = np.ones((4, 4, 4), dtype=np.int32)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_glszm(image, mask, 1, 0, 0, 1, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_glszm_uniform_block(device):
    """A 2x2x2 block of the same gray level is one zone of size 8."""
    image = np.ones((5, 5, 5), dtype=np.int32)
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_glszm(image, mask, 1, 0, 0, 2, coords, device)
