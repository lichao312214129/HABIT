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
Bit-exactness tests: gpumatrices.calculate_glcm vs radiomics.cMatrices.

The GPU path must reproduce the C extension's count matrices exactly
(counts are small integers, exactly representable in float64, so the
comparison is strict equality, not allclose). Runs on CPU torch always and
on CUDA when available.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from radiomics import cMatrices

from habit.kernels.radiomics.gpumatrices import calculate_glcm
from habit.kernels.radiomics.gpumatrices.angles import build_angles, get_angle_count

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _random_case(rng: np.random.Generator, shape, ng: int, mask_frac: float):
    """Random discretised image (levels 1..ng) and a random boolean mask."""
    image = rng.integers(1, ng + 1, size=shape).astype(np.int32)
    mask = rng.random(shape) < mask_frac
    # Guarantee at least a few masked voxels.
    if mask.sum() < 4:
        mask.ravel()[:4] = True
    return image, mask


def _assert_same(image, mask, distances, ng, force2D, force2Ddimension,
                 kernel_radius, voxel_coords, device):
    """Run C extension and GPU implementation, require identical outputs."""
    p_c, a_c = cMatrices.calculate_glcm(
        image, mask, np.asarray(distances, dtype=np.int32), ng,
        force2D, force2Ddimension, kernel_radius, voxel_coords,
    )
    p_g, a_g = calculate_glcm(
        image, mask, np.asarray(distances, dtype=np.int32), ng,
        force2D=bool(force2D), force2Ddimension=force2Ddimension,
        kernelRadius=kernel_radius, voxelCoordinates=voxel_coords,
        device=device, dtype=torch.float64,
    )
    p_g = p_g.cpu().numpy()
    a_g = a_g.cpu().numpy().astype(np.int64)

    assert p_g.shape == p_c.shape, f"shape mismatch: {p_g.shape} vs {p_c.shape}"
    assert np.array_equal(a_g, a_c), f"angles mismatch:\n{a_g}\nvs\n{a_c}"
    assert np.array_equal(p_g, p_c), (
        f"matrix mismatch: max abs diff "
        f"{np.abs(p_g - p_c).max()}, nnz diff {(p_g != p_c).sum()}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_segment_mode(device):
    rng = np.random.default_rng(42)
    image, mask = _random_case(rng, (6, 7, 8), ng=4, mask_frac=0.6)
    _assert_same(image, mask, [1], 4, 0, 0, 0, None, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel_radius", [1, 2, 3])
def test_voxel_mode_radii(device, kernel_radius):
    rng = np.random.default_rng(7)
    image, mask = _random_case(rng, (9, 10, 11), ng=5, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)  # (3, Nvox)
    _assert_same(image, mask, [1], 5, 0, 0, kernel_radius, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_voxel_mode_multi_distance(device):
    rng = np.random.default_rng(11)
    image, mask = _random_case(rng, (8, 9, 10), ng=3, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_same(image, mask, [1, 2], 3, 0, 0, 3, coords, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("f2d_dim", [0, 1, 2])
def test_voxel_mode_force2d(device, f2d_dim):
    rng = np.random.default_rng(13)
    image, mask = _random_case(rng, (7, 8, 9), ng=4, mask_frac=0.5)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_same(image, mask, [1], 4, 1, f2d_dim, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_mask_touches_image_border(device):
    """Mask extends to image edges: exercises the per-voxel bb clipping."""
    rng = np.random.default_rng(17)
    image = rng.integers(1, 4, size=(5, 6, 7)).astype(np.int32)
    mask = np.ones(image.shape, dtype=bool)  # border voxels included
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_same(image, mask, [1], 3, 0, 0, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_distance_exceeds_image_dim(device):
    """Distance larger than a dimension: angle count clamps like the C code."""
    rng = np.random.default_rng(19)
    image, mask = _random_case(rng, (3, 4, 5), ng=3, mask_frac=0.7)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    _assert_same(image, mask, [1, 4], 3, 0, 0, 2, coords, device)


@pytest.mark.parametrize("device", DEVICES)
def test_voxel_subset_batch(device):
    """A subset of voxels (a voxel_batch slice) matches the C extension."""
    rng = np.random.default_rng(23)
    image, mask = _random_case(rng, (8, 8, 8), ng=6, mask_frac=0.4)
    coords = np.array(np.nonzero(mask), dtype=np.int32)
    coords = coords[:, : max(1, coords.shape[1] // 3)].copy()
    _assert_same(image, mask, [1], 6, 0, 0, 3, coords, device)


def test_angle_count_and_order_match_c():
    """Angles must equal cMatrices.generate_angles exactly (set and order)."""
    rng = np.random.default_rng(29)
    for _ in range(20):
        size = rng.integers(2, 12, size=3).tolist()
        # Distances must be unique: duplicate entries double-count in
        # get_angle_count but are stored once in build_angles, which
        # deadlocks the C extension as well — out of scope for parity.
        distances = sorted(set(rng.integers(1, 4, size=int(rng.integers(1, 3))).tolist()))
        for bidirectional in (0, 1):
            a_c = cMatrices.generate_angles(
                np.asarray(size, dtype=np.int32),
                np.asarray(distances, dtype=np.int32),
                bidirectional, 0, 0,
            )
            a_g = build_angles(size, distances, force2Ddimension=-1,
                               bidirectional=bool(bidirectional))
            assert a_g.shape == a_c.shape
            assert np.array_equal(a_g, a_c)
            assert get_angle_count(size, distances, -1, bool(bidirectional)) == a_c.shape[0]


@pytest.mark.parametrize("device", DEVICES)
def test_feature_values_match_pyradiomics_class(device):
    """
    End-to-end: TorchRadiomicsGLCM with use_gpu_matrices produces the same
    per-voxel feature values as the CPU PyRadiomics RadiomicsGLCM class.
    """
    import SimpleITK as sitk
    from radiomics import glcm as py_glcm

    from habit.kernels.radiomics.torchradiomics.TorchRadiomicsGLCM import (
        TorchRadiomicsGLCM,
    )

    rng = np.random.default_rng(31)
    shape = (10, 11, 12)
    # Continuous intensities; binning (binWidth) discretises them.
    image_arr = rng.normal(100.0, 30.0, size=shape)
    mask_arr = (rng.random(shape) < 0.5).astype(np.uint8)
    if mask_arr.sum() < 8:
        mask_arr.ravel()[:8] = 1

    image_sitk = sitk.GetImageFromArray(image_arr)
    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk.CopyInformation(image_sitk)

    settings = {
        "binWidth": 5,
        "label": 1,
        "voxelBased": True,
        "kernelRadius": 2,
        "voxelBatch": 16,
        "distances": [1],
    }

    ref = py_glcm.RadiomicsGLCM(image_sitk, mask_sitk, **settings)
    gpu = TorchRadiomicsGLCM(
        image_sitk, mask_sitk, device=device, dtype=torch.float64,
        use_gpu_matrices=True, **settings,
    )

    # Enable the voxel-safe feature subset (HABIT's production default):
    # MCC/Imc1/Imc2 crash TorchRadiomics on degenerate small-kernel GLCMs
    # (pre-existing limitation, unrelated to the GPU matrix path).
    from habit.utils.radiomics_params_utils import VOXEL_SAFE_GLCM_FEATURES

    for cls in (ref, gpu):
        for feature in VOXEL_SAFE_GLCM_FEATURES:
            cls.enableFeatureByName(feature, True)

    ref_values = ref.execute()
    gpu_values = gpu.execute()

    for name, ref_val in ref_values.items():
        gpu_val = gpu_values[name]
        ref_a = np.asarray(ref_val, dtype=np.float64)
        gpu_a = np.asarray(gpu_val, dtype=np.float64)
        assert ref_a.shape == gpu_a.shape, f"{name}: shape mismatch"
        # Matrices are bit-identical; formulas differ only by floating-point
        # summation order, so allow a tight numerical tolerance plus NaNs.
        assert np.allclose(np.nan_to_num(gpu_a), np.nan_to_num(ref_a),
                           rtol=1e-9, atol=1e-12), (
            f"{name}: max abs diff "
            f"{np.abs(np.nan_to_num(gpu_a) - np.nan_to_num(ref_a)).max()}"
        )
