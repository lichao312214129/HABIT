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
GPU (torch) implementation of the PyRadiomics NGTDM matrix calculation.

Drop-in replacement for ``radiomics.cMatrices.calculate_ngtdm`` (v3.0.1).

C semantics (``calculate_ngtdm`` in ``cmatrices.c``):

- Angles are the bidirectional set (both directions of every offset).
- For every masked centre in the bounding box, collect neighbours that
  are inside the same box and masked. Let ``count`` be their number and
  ``sum`` the sum of their gray levels (C accumulates both as ``double``).
- ``diff = 0`` if ``count == 0``, else ``abs(image[centre] - sum / count)``.
- ``P[gl, 0] += 1`` (n_i), ``P[gl, 1] += diff`` (s_i), ``P[gl, 2] = gl + 1``
  (filled for every gray level up front, even those that never appear).
- Isolated centres still increment n_i and add 0 to s_i.

``s_i`` is a sum of floating-point differences, so it is accumulated in
float64 (matching C ``double``) and cast to ``dtype`` only at the end.
``n_i`` is an integer count.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from ._geom import CentreGrid, coords_in_image, flat_index, prepare_centre_grid


def calculate_ngtdm(
    image: np.ndarray,
    mask: np.ndarray,
    distances: np.ndarray,
    Ng: int,
    force2D: bool = False,
    force2Ddimension: int = 0,
    kernelRadius: int = 0,
    voxelCoordinates: Optional[np.ndarray] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float64,
    max_chunk_elems: int = 1 << 24,
) -> torch.Tensor:
    """
    GPU drop-in replacement for ``radiomics.cMatrices.calculate_ngtdm``.

    Args:
        image: Discretised image (gray levels 1..Ng inside the mask).
        mask: Boolean array, same shape as ``image``.
        distances: Integer infinity-norm distances for angle generation.
        Ng: Number of gray levels (max gray level inside the ROI).
        force2D: Restrict angles and the kernel window to a 2D plane.
        force2Ddimension: Out-of-plane dimension when ``force2D`` is set.
        kernelRadius: Voxel-kernel radius; must be > 0 in voxel-based mode.
        voxelCoordinates: ``(Nd, Nvox)`` listed voxels, or ``None`` for
            segment-based mode.
        device: Torch device for the calculation and the output.
        dtype: Floating dtype of the returned matrix.
        max_chunk_elems: Upper bound on ``Na * Kc * Nvox`` processed in
            one vectorised chunk; bounds transient GPU memory.

    Returns:
        torch.Tensor: ``P_ngtdm`` of shape ``(Nvox, Ng, 3)`` and ``dtype``,
        on ``device``. Column 0 is n_i, column 1 is s_i, column 2 is the
        gray-level index (1..Ng).
    """
    grid = prepare_centre_grid(
        image=image,
        mask=mask,
        distances=distances,
        force2D=force2D,
        force2Ddimension=force2Ddimension,
        kernelRadius=kernelRadius,
        voxelCoordinates=voxelCoordinates,
        device=device,
        bidirectional=True,
    )
    return _accumulate_ngtdm(grid, int(Ng), dtype, max_chunk_elems)


def _accumulate_ngtdm(
    grid: CentreGrid,
    ng: int,
    dtype: torch.dtype,
    max_chunk_elems: int,
) -> torch.Tensor:
    """Vectorised NGTDM accumulation over kernel-window centres."""
    na = int(grid.angles_t.shape[0])
    n_off = int(grid.offsets_t.shape[0])
    n_vox = grid.n_vox
    device = grid.device

    # Column 2 is the gray-level label, filled for every i even if n_i == 0.
    p = torch.zeros(grid.n_matrices, ng, 3, dtype=torch.float64, device=device)
    p[:, :, 2] = torch.arange(1, ng + 1, dtype=torch.float64, device=device)
    if n_vox == 0 or n_off == 0:
        return p.to(dtype=dtype)

    # n_i counts centres; bounded by the kernel window volume, so int16.
    n_flat = torch.zeros(grid.n_matrices * ng, dtype=torch.int16, device=device)
    s_flat = torch.zeros(grid.n_matrices * ng, dtype=torch.float64, device=device)

    chunk = max(1, int(max_chunk_elems) // max(1, na * n_vox))

    for start in range(0, n_off, chunk):
        o_c = grid.offsets_t[start : start + chunk]
        centres = grid.base_coords[None, :, :] + o_c[:, None, :]
        centre_in = coords_in_image(centres, grid.size_t)
        centre_flat = flat_index(centres, grid.strides_t, grid.n_elements)
        centre_valid = centre_in & grid.mask_flat[centre_flat]
        if not bool(centre_valid.any()):
            continue

        neighbours = centres[None, :, :, :] + grid.angles_t[:, None, None, :]
        neigh_in = coords_in_image(neighbours, grid.size_t)
        if grid.voxel_based:
            in_window = (
                (neighbours - grid.base_coords[None, None, :, :]).abs() <= grid.kernel_radius
            ).all(dim=-1)
            neigh_in = neigh_in & in_window
        neigh_flat = flat_index(neighbours, grid.strides_t, grid.n_elements)
        neigh_valid = neigh_in & grid.mask_flat[neigh_flat]

        gi = grid.img_flat[centre_flat]  # (Kc, Nvox)
        gj = grid.img_flat[neigh_flat].to(torch.float64)  # (Na, Kc, Nvox)
        valid_f = neigh_valid.to(torch.float64)
        count = valid_f.sum(dim=0)  # (Kc, Nvox)
        neigh_sum = (gj * valid_f).sum(dim=0)
        # C: if (count == 0) diff = 0; else abs(image[i] - sum / count)
        # with count/sum stored as double.
        diff = torch.where(
            count == 0,
            torch.zeros_like(count),
            (gi.to(torch.float64) - neigh_sum / count).abs(),
        )

        gi0 = (gi - 1).clamp_(0, ng - 1)
        flat_idx = grid.matrix_ids[None, :] * ng + gi0
        selected = flat_idx[centre_valid]
        n_flat.index_add_(
            0, selected, torch.ones(selected.numel(), dtype=torch.int16, device=device)
        )
        s_flat.index_add_(0, selected, diff[centre_valid])

    p[:, :, 0] = n_flat.reshape(grid.n_matrices, ng).to(torch.float64)
    p[:, :, 1] = s_flat.reshape(grid.n_matrices, ng)
    return p.to(dtype=dtype)
