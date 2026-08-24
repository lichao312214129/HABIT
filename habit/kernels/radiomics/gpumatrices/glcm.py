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
GPU (torch) implementation of the PyRadiomics GLCM matrix calculation.

``calculate_glcm`` is a drop-in replacement for
``radiomics.cMatrices.calculate_glcm`` (PyRadiomics v3.0.1 semantics) that
runs fully on a torch device and returns torch tensors, avoiding both the
single-threaded C voxel loop and the host-to-device copy of the (potentially
very large) per-voxel matrix array.

Semantics replicated from ``radiomics/src/cmatrices.c`` and
``radiomics/src/_cmatrices.c``:

- Angles come from the exact port in :mod:`.angles` (mono-directional set,
  PyRadiomics emission order).
- Segment-based mode (``voxelCoordinates=None``): one matrix for the whole
  ROI; every masked voxel is a centre; the neighbour must lie inside the
  image and be masked.
- Voxel-based mode: one matrix per listed voxel ``v``. Centres range over
  the ``(2*kernelRadius+1)^Nd`` window around ``v`` (clipped to the image;
  collapsed to the voxel's own slice along ``force2Ddimension`` when
  ``force2D`` is set). A pair ``(centre, neighbour)`` is counted when both
  voxels are inside the window, inside the image, and masked. The neighbour
  sits at ``centre + angle``.

Counts are accumulated in floating point with values that are exact small
integers, so the result is bit-identical to the C ``double`` accumulation
regardless of atomic ordering on the GPU.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

from ._geom import coords_in_image, flat_index, prepare_centre_grid, window_cells


def calculate_glcm(
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU drop-in replacement for ``radiomics.cMatrices.calculate_glcm``.

    Args:
        image: Discretised image array (gray levels 1..Ng inside the mask),
            any dimensionality, numpy order (e.g. ``(z, y, x)``).
        mask: Boolean array, same shape as ``image``.
        distances: Integer infinity-norm distances for angle generation.
        Ng: Number of gray levels (max gray level inside the ROI).
        force2D: Restrict angles and the kernel window to a 2D plane.
        force2Ddimension: Out-of-plane dimension when ``force2D`` is set.
        kernelRadius: Voxel-kernel radius; must be > 0 in voxel-based mode.
        voxelCoordinates: ``(Nd, Nvox)`` integer array of centre voxels;
            ``None`` selects segment-based mode (one matrix for the ROI).
        device: Torch device for the calculation and the outputs.
        dtype: Floating dtype of the returned matrix (counts are exact
            small integers in either float32 or float64).
        max_chunk_elems: Upper bound on ``pairs x voxels`` processed in one
            vectorised chunk; bounds transient GPU memory.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(P_glcm, angles)`` on
        ``device``. ``P_glcm`` has shape ``(Nvox, Ng, Ng, Na)`` and
        ``dtype``; ``angles`` has shape ``(Na, Nd)`` and ``dtype`` (matching
        what ``TorchRadiomicsBase.tensor`` would produce from the C output).
    """
    grid = prepare_centre_grid(
        image=image,
        mask=mask,
        distances=np.asarray(distances),
        force2D=force2D,
        force2Ddimension=force2Ddimension,
        kernelRadius=kernelRadius,
        voxelCoordinates=voxelCoordinates,
        device=device,
        bidirectional=False,
    )
    device = grid.device
    na = int(grid.angles_t.shape[0])
    n_matrices = grid.n_matrices
    n_vox = grid.n_vox

    # Accumulate counts in int16: a (Nvox, Ng, Ng, Na) GLCM entry is hit at
    # most once per kernel offset, far below 2^15 for any realistic radius,
    # so integer atomics are exact and the largest allocation is halved
    # against int32. Cast to ``dtype`` once at the end; the values are small
    # integers, exactly representable in float32/64.
    p_flat = torch.zeros(n_matrices * Ng * Ng * na, dtype=torch.int16, device=device)
    angles_out = grid.angles_t.to(dtype=dtype)
    if n_vox == 0:
        return p_flat.reshape(n_matrices, Ng, Ng, na).to(dtype=dtype), angles_out

    cells = window_cells(grid) if grid.voxel_based else None
    if cells is not None:
        # A (angle, offset) pair contributes only if the neighbour stays
        # inside the kernel window, which is exactly where the neighbour is
        # itself an enumerated offset.
        pair_a, pair_o = torch.nonzero(cells.neigh_off >= 0, as_tuple=True)
    else:
        # Segment-based mode: a single matrix; every masked voxel is a centre
        # and the only neighbour constraint is the image bounds.
        pair_a = torch.arange(na, dtype=torch.long, device=device)
        pair_o = torch.zeros(na, dtype=torch.long, device=device)

    n_pairs = int(pair_a.shape[0])
    chunk = max(1, int(max_chunk_elems) // max(1, n_vox))

    for start in range(0, n_pairs, chunk):
        a_c = pair_a[start : start + chunk]  # (Kc,) angle index per pair
        o_c = pair_o[start : start + chunk]  # (Kc,) kernel-offset index per pair

        if cells is not None:
            o2 = cells.neigh_off[a_c, o_c]  # (Kc,), all >= 0 by construction
            valid = cells.cell_valid[o_c] & cells.cell_valid[o2]  # (Kc, Nvox)
            gi = (cells.cell_gray[o_c].to(torch.int64) - 1).clamp_(0, Ng - 1)
            gj = (cells.cell_gray[o2].to(torch.int64) - 1).clamp_(0, Ng - 1)
        else:
            # Centres and neighbours for every (pair, voxel) combination.
            centres = grid.base_coords[None, :, :] + grid.offsets_t[o_c][:, None, :]
            neighbours = centres + grid.angles_t[a_c][:, None, :]  # (Kc, Nvox, Nd)
            # Both voxels must lie inside the image (the per-voxel kernel
            # window is the image-clipped [-r, r]^Nd box, so bounds checks
            # here implement the same clipping as the C bb handling).
            valid = coords_in_image(centres, grid.size_t) & coords_in_image(
                neighbours, grid.size_t
            )
            # Out-of-bounds coordinates produce out-of-range flat indices;
            # ``flat_index`` clamps them -- the gathered values at those
            # positions are don't-care because ``valid`` drops them.
            centre_flat = flat_index(centres, grid.strides_t, grid.n_elements)
            neighbour_flat = flat_index(neighbours, grid.strides_t, grid.n_elements)
            # Both voxels must be part of the ROI (mask[i] and mask[j] in C).
            valid = (
                valid
                & grid.mask_flat[centre_flat]
                & grid.mask_flat[neighbour_flat]
            )
            # Gray levels are in [1, Ng] on masked voxels; the clamp only
            # sanitises don't-care values at invalid positions.
            gi = (grid.img_flat[centre_flat] - 1).clamp_(0, Ng - 1)
            gj = (grid.img_flat[neighbour_flat] - 1).clamp_(0, Ng - 1)

        # C-order flat index of element [v, i, j, a] in (Nmat, Ng, Ng, Na).
        flat_idx = ((grid.matrix_ids[None, :] * Ng + gi) * Ng + gj) * na + a_c[:, None]
        selected = flat_idx[valid]
        p_flat.index_add_(
            0, selected, torch.ones(selected.numel(), dtype=torch.int16, device=device)
        )

    p_glcm = p_flat.reshape(n_matrices, Ng, Ng, na).to(dtype=dtype)
    return p_glcm, angles_out
