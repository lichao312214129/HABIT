#!/usr/bin/env python3
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
"""Wall-clock benchmark for native supervoxel texture (not a Sphinx page)."""

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk

from habit.kernels.radiomics.cext import cext_backend
from habit.kernels.radiomics.native_batch import extract_native_supervoxel_features
from habit.kernels.radiomics.supervoxel_batch import DEFAULT_FEATURES_BY_CLASS


def _synthetic_tumor(
    n_labels: int = 100,
    n_voxels: int = 40000,
) -> Tuple[sitk.Image, sitk.Image, np.ndarray]:
    """Build a 10k-70k voxel tumor-like crop with ``n_labels`` supervoxels."""
    rng = np.random.default_rng(0)
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


def _run_once(tag: str) -> Dict[str, float]:
    """Extract once; print wall time and millisecond breakdown."""
    image, sv_map, labels = _synthetic_tumor()
    enabled = {
        name: list(DEFAULT_FEATURES_BY_CLASS[name])
        for name in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
    }
    settings = {
        "binWidth": 12,
        "voxelArrayShift": 300,
        "use_supervoxel_cext": "auto",
        "padDistance": 1,
    }
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    frame = extract_native_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features=enabled,
        settings=settings,
        union_bin=True,
        timings=timings,
    )
    wall = time.perf_counter() - t0
    timings["wall_s"] = wall
    sre = frame["original_glrlm_ShortRunEmphasis"].to_numpy()
    print(
        f"{tag}: wall={wall:.4f}s backend={cext_backend()} rows={len(frame)} "
        f"SRE_finite={bool(np.isfinite(sre).all())} "
        f"breakdown_ms={{{', '.join(f'{k}={v:.2f}' for k, v in timings.items() if k.endswith('_ms'))}}}"
    )
    return timings


def main() -> int:
    """Cold extract is the first call in this process; optional warm repeat."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-warm", action="store_true")
    args = parser.parse_args()
    print(f"cext_backend={cext_backend()}")
    first = _run_once("cold")
    if args.repeat_warm:
        _run_once("warm")
    return 0 if first["wall_s"] < 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
