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
GPU (torch) implementation of the PyRadiomics GLDM matrix calculation.

Drop-in replacement for ``radiomics.cMatrices.calculate_gldm`` (v3.0.1).

C semantics (``calculate_gldm`` in ``cmatrices.c``):

- Angles are the *bidirectional* set (both directions of every offset).
- For every masked centre in the bounding box (full image in segment
  mode; the image-clipped ``[-r, r]`` kernel in voxel mode), count the
  neighbours that are inside the same box, masked, and satisfy
  ``|image[centre] - image[neighbour]| <= alpha``.
- Increment ``P[grey, dep]`` once per centre, where ``dep`` is that
  neighbour count (0 if the centre has no dependent neighbour).
- Output shape is ``(Nvox, Ng, Na * 2 + 1)``. The extra columns are a
  leftover of the C wrapper treating ``Na`` as mono-directional; they
  stay zero. We keep the same allocation so the array is bit-identical
  to the C extension before empty-size deletion.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from ._geom import (
    CentreGrid,
    coords_in_image,
    flat_index,
    prepare_centre_grid,
    window_cells,
)


def calculate_gldm(
    image: np.ndarray,
    mask: np.ndarray,
    distances: np.ndarray,
    Ng: int,
    alpha: int = 0,
    force2D: bool = False,
    force2Ddimension: int = 0,
    kernelRadius: int = 0,
    voxelCoordinates: Optional[np.ndarray] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float64,
    max_chunk_elems: int = 1 << 24,
) -> torch.Tensor:
    """
    GPU drop-in replacement for ``radiomics.cMatrices.calculate_gldm``.

    Args:
        image: Discretised image (gray levels 1..Ng inside the mask).
        mask: Boolean array, same shape as ``image``.
        distances: Integer infinity-norm distances for angle generation.
        Ng: Number of gray levels (max gray level inside the ROI).
        alpha: Dependence cutoff; a neighbour is dependent iff
            ``|i - j| <= alpha``. Integer, matching the C ``int alpha``.
        force2D: Restrict angles and the kernel window to a 2D plane.
        force2Ddimension: Out-of-plane dimension when ``force2D`` is set.
        kernelRadius: Voxel-kernel radius; must be > 0 in voxel-based mode.
        voxelCoordinates: ``(Nd, Nvox)`` listed voxels, or ``None`` for
            segment-based mode.
        device: Torch device for the calculation and the output.
        dtype: Floating dtype of the returned matrix (counts are exact
            small integers in either float32 or float64).
        max_chunk_elems: Upper bound on ``Na * Kc * Nvox`` processed in
            one vectorised chunk; bounds transient GPU memory.

    Returns:
        torch.Tensor: ``P_gldm`` of shape ``(Nvox, Ng, Na * 2 + 1)`` and
        ``dtype``, on ``device``.
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
    na = int(grid.angles_t.shape[0])
    # C wrapper: "Na angels, 2 directions and +1 for no dependency" — even
    # though angles are already bidirectional, so the upper half is unused.
    n_dep = na * 2 + 1
    return _accumulate_gldm(grid, int(Ng), int(alpha), n_dep, dtype, max_chunk_elems)


def _accumulate_gldm(
    grid: CentreGrid,
    ng: int,
    alpha: int,
    n_dep: int,
    dtype: torch.dtype,
    max_chunk_elems: int,
) -> torch.Tensor:
    """Vectorised GLDM accumulation over kernel-window centres."""
    na = int(grid.angles_t.shape[0])
    n_off = int(grid.offsets_t.shape[0])
    n_vox = grid.n_vox
    device = grid.device

    # int16 is exact: a GLDM entry counts centres, bounded by the kernel
    # window volume (2r+1)^Nd, far below 2^15 for any realistic radius.
    p_flat = torch.zeros(grid.n_matrices * ng * n_dep, dtype=torch.int16, device=device)
    if n_vox == 0 or n_off == 0:
        return p_flat.reshape(grid.n_matrices, ng, n_dep).to(dtype=dtype)

    # One chunk covers ``Na * Kc * Nvox`` neighbour slots.
    chunk = max(1, int(max_chunk_elems) // max(1, na * n_vox))
    # Voxel mode reads centres and neighbours out of the same (No, Nvox)
    # tables instead of building (Na, Kc, Nvox, Nd) coordinates; see
    # _geom.WindowCells. Segment mode has no offset window, so a neighbour
    # is not an enumerated offset and the coordinate path stays.
    cells = window_cells(grid) if grid.voxel_based else None

    for start in range(0, n_off, chunk):
        if cells is not None:
            o_slice = slice(start, start + chunk)
            centre_valid = cells.cell_valid[o_slice]  # (Kc, Nvox)
            gi = cells.cell_gray[o_slice]
            neigh_o = cells.neigh_off[:, o_slice]  # (Na, Kc)
            neigh_valid = (neigh_o >= 0)[:, :, None] & cells.cell_valid[
                neigh_o.clamp(min=0)
            ]
            gj = cells.cell_gray[neigh_o.clamp(min=0)]  # (Na, Kc, Nvox)
        else:
            o_c = grid.offsets_t[start : start + chunk]  # (Kc, Nd)
            centres = grid.base_coords[None, :, :] + o_c[:, None, :]  # (Kc, Nvox, Nd)
            centre_in = coords_in_image(centres, grid.size_t)
            centre_flat = flat_index(centres, grid.strides_t, grid.n_elements)
            centre_valid = centre_in & grid.mask_flat[centre_flat]
            # Neighbours of every centre along every bidirectional angle.
            # Shape: (Na, Kc, Nvox, Nd).
            neighbours = centres[None, :, :, :] + grid.angles_t[:, None, None, :]
            neigh_in = coords_in_image(neighbours, grid.size_t)
            neigh_flat = flat_index(neighbours, grid.strides_t, grid.n_elements)
            neigh_valid = neigh_in & grid.mask_flat[neigh_flat]
            gi = grid.img_flat[centre_flat]  # (Kc, Nvox)
            gj = grid.img_flat[neigh_flat]  # (Na, Kc, Nvox)

        dependent = neigh_valid & ((gi[None, :, :] - gj).abs() <= alpha)
        # Isolated centres (no valid neighbour) keep dep == 0, matching C.
        dep = dependent.to(torch.int32).sum(dim=0)  # (Kc, Nvox)

        gi0 = (gi.to(torch.int64) - 1).clamp_(0, ng - 1)
        flat_idx = (grid.matrix_ids[None, :] * ng + gi0) * n_dep + dep
        selected = flat_idx[centre_valid]
        p_flat.index_add_(
            0, selected, torch.ones(selected.numel(), dtype=torch.int16, device=device)
        )

    return p_flat.reshape(grid.n_matrices, ng, n_dep).to(dtype=dtype)
