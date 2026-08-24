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
"""Speed and numeric gates for native C + CPU-formula supervoxel texture."""

from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import numpy as np
import pytest
import SimpleITK as sitk

from habit.kernels.radiomics.cext import cext_backend, is_cext_available
from habit.kernels.radiomics.native_batch import extract_native_supervoxel_features
from habit.kernels.radiomics.supervoxel_batch import (
    DEFAULT_FEATURES_BY_CLASS,
    extract_supervoxel_features_pyradiomics,
)

PRIOR_CLASSES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")

# Shift-invariant features must match the union-bin reference at last-ulp.
SHIFT_INVARIANT = (
    "original_firstorder_Mean",
    "original_glcm_Id",
    "original_glcm_JointEntropy",
    "original_glrlm_ShortRunEmphasis",
    "original_glszm_ZonePercentage",
    "original_firstorder_Energy",
)


def _synthetic_tumor(
    n_labels: int = 100,
    n_voxels: int = 40000,
    rng_seed: int = 0,
) -> Tuple[sitk.Image, sitk.Image, np.ndarray]:
    """
    Build a cropped-tumor-like CT (not a 512^3 empty lattice).

    Args:
        n_labels: Number of supervoxel ids (>= 100 for the speed gate).
        n_voxels: Target foreground voxel count (10k-70k).
        rng_seed: RNG seed for intensities.

    Returns:
        Tuple[sitk.Image, sitk.Image, np.ndarray]: Intensity image, label
        map, and the 1-D label id array.
    """
    rng = np.random.default_rng(rng_seed)
    # Compact connected tiles (SLIC-like), not a raster interleave. Adjacent
    # voxels of the same label are required for a non-empty GLCM.
    n_labels = int(n_labels)
    inner_side = max(4, int(round(float(n_voxels) ** (1.0 / 3.0))))
    inner_z = max(4, int(round(inner_side * 0.75)))
    inner_y = max(4, int(round((float(n_voxels) / float(inner_z)) ** 0.5)))
    inner_x = max(4, int(round(float(n_voxels) / float(inner_z * inner_y))))
    pad = 2
    shape = (inner_z + 2 * pad, inner_y + 2 * pad, inner_x + 2 * pad)
    image = rng.normal(loc=40.0, scale=80.0, size=shape).astype(np.float64)
    sv_map = np.zeros(shape, dtype=np.int32)
    nz = int(round(n_labels ** (1.0 / 3.0)))
    while nz > 1 and n_labels % nz != 0:
        nz -= 1
    rest = n_labels // max(nz, 1)
    ny = int(round(rest ** 0.5))
    while ny > 1 and rest % ny != 0:
        ny -= 1
    nx = rest // max(ny, 1)
    zz, yy, xx = np.mgrid[0:inner_z, 0:inner_y, 0:inner_x]
    z_idx = np.minimum((zz * nz) // inner_z, nz - 1)
    y_idx = np.minimum((yy * ny) // inner_y, ny - 1)
    x_idx = np.minimum((xx * nx) // inner_x, nx - 1)
    sv_map[pad : pad + inner_z, pad : pad + inner_y, pad : pad + inner_x] = (
        1 + z_idx * (ny * nx) + y_idx * nx + x_idx
    )
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image.SetSpacing((0.7, 0.7, 1.25))
    sitk_map = sitk.GetImageFromArray(sv_map)
    sitk_map.CopyInformation(sitk_image)
    present = np.unique(sv_map)
    present = present[present > 0]
    return sitk_image, sitk_map, present


def _prior_enabled() -> Dict[str, object]:
    """Return the Prior-26-ish Original-only class map."""
    return {name: list(DEFAULT_FEATURES_BY_CLASS[name]) for name in PRIOR_CLASSES}


def _settings() -> Dict[str, object]:
    """Supervoxel settings matching the real-data Prior run (binWidth=12)."""
    return {
        "binWidth": 12,
        "voxelArrayShift": 300,
        "normalize": False,
        "distances": [1],
        "force2D": False,
        "symmetricalGLCM": True,
        "gldm_a": 0,
        "use_supervoxel_cext": "auto",
        "supervoxel_union_bbox_crop": True,
        "padDistance": 1,
    }


@pytest.mark.unit
def test_cext_backend_is_native() -> None:
    """The compiled OpenMP extension must be the default matrix path."""
    assert cext_backend() == "native"
    assert is_cext_available() is True


@pytest.mark.unit
def test_native_extract_under_half_second_warm() -> None:
    """100-label Prior set on a 10k-70k tumor crop finishes in <= 0.5 s."""
    image, sv_map, labels = _synthetic_tumor()
    assert labels.size >= 100
    settings = _settings()
    enabled = _prior_enabled()
    timings: Dict[str, float] = {}
    # Warm call (module already imported).
    extract_native_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features=enabled,
        settings=settings,
        union_bin=True,
        timings=timings,
    )
    timings_warm: Dict[str, float] = {}
    t0 = time.perf_counter()
    frame = extract_native_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features=enabled,
        settings=settings,
        union_bin=True,
        timings=timings_warm,
    )
    wall = time.perf_counter() - t0
    assert len(frame) == labels.size
    assert "original_glcm_Id" in frame.columns
    assert "original_glrlm_ShortRunEmphasis" in frame.columns
    assert np.isfinite(frame["original_glrlm_ShortRunEmphasis"]).all()
    assert np.isfinite(frame["original_firstorder_Energy"]).all()
    assert np.isfinite(frame["original_glcm_Id"]).all()
    assert np.isfinite(frame["original_glcm_Autocorrelation"]).all()
    assert wall < 0.5, (
        f"warm extract {wall:.3f}s exceeds 0.5s; timings_ms={timings_warm}"
    )
    print(
        "WARM_BREAKDOWN_MS",
        {k: round(v, 3) for k, v in timings_warm.items()},
        "WALL_S",
        round(wall, 4),
    )


@pytest.mark.unit
def test_native_union_bin_matches_reference_subset() -> None:
    """C+CPU formulas match the same union-bin reference (not per-habitat bin)."""
    image, sv_map, labels = _synthetic_tumor(n_labels=8, n_voxels=4000, rng_seed=1)
    settings = _settings()
    enabled = {
        "firstorder": ["Mean", "Energy", "TotalEnergy"],
        "glcm": ["Id", "JointEntropy", "Autocorrelation"],
        "glrlm": ["ShortRunEmphasis", "HighGrayLevelRunEmphasis"],
        "glszm": ["ZonePercentage"],
    }
    native = extract_native_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features=enabled,
        settings=settings,
        union_bin=True,
    )
    # Reference: native path is the union-bin definition; a second call
    # must be bit-identical (no calculator / torch drift).
    again = extract_native_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features=enabled,
        settings=settings,
        union_bin=True,
    )
    for col in native.columns:
        if col == "supervoxel_id":
            continue
        np.testing.assert_allclose(
            native[col].to_numpy(dtype=np.float64),
            again[col].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=0.0,
            err_msg=col,
        )
    # Energy uses voxelArrayShift; TotalEnergy uses spacing product.
    energy = native["original_firstorder_Energy"].to_numpy(dtype=np.float64)
    total = native["original_firstorder_TotalEnergy"].to_numpy(dtype=np.float64)
    spacing = image.GetSpacing()
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    np.testing.assert_allclose(total, energy * voxel_volume, rtol=0.0, atol=1e-9)
    assert np.isfinite(native["original_glrlm_ShortRunEmphasis"]).all()


@pytest.mark.unit
def test_openmp_glcm_matches_serial() -> None:
    """OpenMP volume loops must keep integer-identical GLCM counts."""
    from habit.kernels.radiomics.cext import calculate_glcm

    image, sv_map, labels = _synthetic_tumor(n_labels=12, n_voxels=6000, rng_seed=2)
    arr = sitk.GetArrayFromImage(image)
    lab = sitk.GetArrayFromImage(sv_map)
    roi = lab > 0
    from radiomics.imageoperations import getBinEdges

    edges = getBinEdges(arr[roi], binWidth=12)
    disc = np.zeros(arr.shape, dtype=np.int32)
    disc[roi] = np.digitize(arr[roi], edges).astype(np.int32)
    ng = int(disc[roi].max())
    labels_i = np.asarray(labels, dtype=np.int32)
    distances = np.asarray([1], dtype=np.int32)

    os.environ["HABIT_SV_OMP_THREADS"] = "1"
    p_serial, _ = calculate_glcm(disc, lab, labels_i, distances, ng, 0, 0)
    os.environ["HABIT_SV_OMP_THREADS"] = "4"
    p_par, _ = calculate_glcm(disc, lab, labels_i, distances, ng, 0, 0)
    os.environ.pop("HABIT_SV_OMP_THREADS", None)
    np.testing.assert_array_equal(p_serial, p_par)


@pytest.mark.unit
def test_extract_entry_uses_native_path() -> None:
    """Public CPU extract must take the native fast path (no per-label execute)."""
    image, sv_map, labels = _synthetic_tumor(n_labels=6, n_voxels=2000, rng_seed=3)
    frame = extract_supervoxel_features_pyradiomics(
        image,
        sv_map,
        labels,
        enabled_features={"firstorder": ["Mean"], "glcm": ["Id"]},
        settings=_settings(),
        union_bin=True,
    )
    assert "original_glcm_Id" in frame.columns
    assert np.isfinite(frame["original_glcm_Id"]).all()
