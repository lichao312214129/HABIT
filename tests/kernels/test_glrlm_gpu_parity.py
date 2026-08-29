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
Parity tests: gpumatrices.calculate_glrlm vs radiomics.cMatrices.

Run counts are small integers, compared with strict equality. Angles
must also match the C extension (mono-directional, distance 1).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from radiomics import cMatrices

from habit.kernels.radiomics.gpumatrices import calculate_glrlm

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _random_case(rng: np.random.Generator, shape, ng: int, mask_frac: float):
    """Random discretised image (levels 1..ng) and a random boolean mask."""
    image = rng.integers(1, ng + 1, size=shape).astype(np.int32)
    mask = rng.random(shape) < mask_frac
    if mask.sum() < 4:
        mask.ravel()[:4] = True
    return image, mask


def _assert_glrlm(image, mask, ng, force2D, force2Ddimension,
                  kernel_radius, voxel_coords, device):
    """Run C extension and GPU GLRLM, require identical counts and angles."""
    nr = int(max(image.shape))
    args = [image, mask, ng, nr, force2D, force2Ddimension]
    if voxel_coords is not None:
        args += [kernel_radius, voxel_coords]
    p_c, a_c = cMatrices.calculate_glrlm(*args)
    p_g, a_g = calculate_glrlm(
        image, mask, ng, nr,
        force2D=bool(force2D), force2Ddimension=force2Ddimension,
        kernelRadius=kernel_radius, voxelCoordinates=voxel_coords,
        device=device, dtype=torch.float64,
    )
    p_g = p_g.cpu().numpy()
    a_g = a_g.cpu().numpy().astype(np.int64)
    assert p_g.shape == p_c.shape, f"GLRLM shape {p_g.shape} vs {p_c.shape}"
    assert np.array_equal(a_g, a_c), f"angles mismatch:\n{a_g}\nvs\n{a_c}"
    assert np.array_equal(p_g, p_c), (
        f"GLRLM mismatch: max abs {np.abs(p_g - p_c).max()}, "
        f"nnz diff {(p_g != p_c).sum()}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_glrlm_segment(device):
    rng = np.random.default_rng(42)
    image, mask = _random_case(rng, (6, 7, 8), ng=4, mask_frac=0.6)
    _assert_glrlm(image, mask, 4, 0, 0, 0, None, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel_radius", [1, 2, 3])
def test_glrlm_voxel_radii(device, kernel_radius):
    rng = np.random.default_rng(7)
    image, mask = _random_case(rng, (8, 9, 10), ng=5, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_glrlm(image, mask, 5, 0, 0, kernel_radius, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_glrlm_voxel_force2d(device):
    rng = np.random.default_rng(13)
    image, mask = _random_case(rng, (7, 8, 9), ng=4, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_glrlm(image, mask, 4, 1, 0, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_glrlm_border(device):
    rng = np.random.default_rng(17)
    image = rng.integers(1, 4, size=(5, 6, 7)).astype(np.int32)
    mask = np.ones(image.shape, dtype=bool)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_glrlm(image, mask, 3, 0, 0, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_glrlm_single_voxel_wiped(device):
    """A lone masked voxel has multiElement==0, so C zeros the matrix."""
    image = np.ones((4, 4, 4), dtype=np.int32)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_glrlm(image, mask, 1, 0, 0, 1, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_glrlm_aligned_pair(device):
    """Two masked voxels on a ray keep their runs (multiElement==1)."""
    image = np.ones((5, 5, 5), dtype=np.int32)
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    mask[2, 2, 3] = True
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_glrlm(image, mask, 1, 0, 0, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_glrlm_forced_sort_chunks_match_c(device):
    """Length-1 wipe must not erase bins written by another sort chunk."""
    rng = np.random.default_rng(29)
    image, mask = _random_case(rng, (16, 16, 16), ng=6, mask_frac=0.85)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    # Several dozen listed voxels at radius 3 overflow a 2^14 sort budget
    # (max_pts ≈ 1260; one window is up to 343 points).
    n_keep = min(int(coords.shape[1]), 64)
    coords = coords[:, :n_keep]
    ng = 6
    nr = int(max(image.shape))
    kernel_radius = 3
    p_c, a_c = cMatrices.calculate_glrlm(
        image, mask, ng, nr, 0, 0, kernel_radius, coords
    )
    from habit.kernels.radiomics.gpumatrices._geom import prepare_centre_grid
    from habit.kernels.radiomics.gpumatrices.glrlm import _accumulate_glrlm

    grid = prepare_centre_grid(
        image=image,
        mask=mask,
        distances=np.asarray([1], dtype=np.int32),
        force2D=False,
        force2Ddimension=0,
        kernelRadius=kernel_radius,
        voxelCoordinates=coords,
        device=device,
        bidirectional=False,
    )
    p_g, a_g = _accumulate_glrlm(
        grid, ng, nr, torch.float64, max_sort_elems=1 << 14
    )
    p_g_np = p_g.cpu().numpy()
    a_g_np = a_g.cpu().numpy().astype(np.int64)
    assert np.array_equal(a_g_np, a_c)
    assert np.array_equal(p_g_np, p_c), (
        f"chunked GLRLM mismatch: max abs {np.abs(p_g_np - p_c).max()}, "
        f"nnz diff {(p_g_np != p_c).sum()}"
    )
