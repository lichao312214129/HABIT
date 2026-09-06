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

from ._geom import CentreGrid, coords_in_image, flat_index, prepare_centre_grid, window_cells
from .sparse_coo import SparseGLCM, unique_counts, unpack_glcm_key


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
    angles_out = grid.angles_t.to(dtype=dtype)
    if n_vox == 0:
        p_empty = torch.zeros(
            n_matrices, Ng, Ng, na, dtype=dtype, device=device
        )
        return p_empty, angles_out

    selected = _glcm_event_keys(grid, int(Ng), int(max_chunk_elems))
    p_flat = torch.zeros(n_matrices * Ng * Ng * na, dtype=torch.int16, device=device)
    if selected.numel() > 0:
        p_flat.index_add_(
            0,
            selected,
            torch.ones(selected.numel(), dtype=torch.int16, device=device),
        )
    p_glcm = p_flat.reshape(n_matrices, Ng, Ng, na).to(dtype=dtype)
    return p_glcm, angles_out


def calculate_glcm_coo(
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
) -> Tuple[SparseGLCM, torch.Tensor]:
    """
    Sparse-COO GLCM. Never allocates a dense ``(B, Ng, Ng, Na)`` store.

    Same pair generation as :func:`calculate_glcm`. Counts are reduced
    with ``unique`` + ``bincount``. Default ``dtype`` is float64; float32
    is opt-in and is not value-safe for ClusterShade.

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
        device: Torch device for the calculation and the outputs.
        dtype: Count / feature compute dtype (float64 default).
        max_chunk_elems: Same pair-chunk cap as the dense kernel.

    Returns:
        Tuple[SparseGLCM, torch.Tensor]: COO counts and the ``(Na, Nd)``
        angle table on ``device``.
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
    n_vox = grid.n_vox
    angles_out = grid.angles_t.to(dtype=dtype)
    empty = torch.empty(0, dtype=torch.long, device=device)
    if n_vox == 0:
        return (
            SparseGLCM(
                voxel=empty,
                i_idx=empty,
                j_idx=empty,
                angle=empty,
                count=torch.empty(0, dtype=dtype, device=device),
                n_vox=int(grid.n_matrices),
                ng=int(Ng),
                na=na,
            ),
            angles_out,
        )
    selected = _glcm_event_keys(grid, int(Ng), int(max_chunk_elems))
    uniq, counts = unique_counts(selected, dtype)
    voxel, i_idx, j_idx, angle = unpack_glcm_key(uniq, int(Ng), na)
    return (
        SparseGLCM(
            voxel=voxel,
            i_idx=i_idx,
            j_idx=j_idx,
            angle=angle,
            count=counts,
            n_vox=int(grid.n_matrices),
            ng=int(Ng),
            na=na,
        ),
        angles_out,
    )


def _glcm_event_keys(
    grid: CentreGrid,
    ng: int,
    max_chunk_elems: int,
) -> torch.Tensor:
    """
    Flat C-order ``(v, i, j, a)`` keys for every valid GLCM pair.

    Shared by the dense gold scatter and the sparse COO unique. The
    working store here is the event list, not ``P``.

    Args:
        grid: Prepared :class:`CentreGrid`.
        ng: Gray-level axis length.
        max_chunk_elems: Upper bound on ``pairs x voxels`` per chunk.

    Returns:
        torch.Tensor: 1-D int64 keys, possibly empty.
    """
    device = grid.device
    na = int(grid.angles_t.shape[0])
    n_vox = grid.n_vox
    cells = window_cells(grid) if grid.voxel_based else None
    if cells is not None:
        pair_a, pair_o = torch.nonzero(cells.neigh_off >= 0, as_tuple=True)
    else:
        pair_a = torch.arange(na, dtype=torch.long, device=device)
        pair_o = torch.zeros(na, dtype=torch.long, device=device)

    n_pairs = int(pair_a.shape[0])
    chunk = max(1, int(max_chunk_elems) // max(1, n_vox))
    parts = []
    for start in range(0, n_pairs, chunk):
        a_c = pair_a[start : start + chunk]
        o_c = pair_o[start : start + chunk]
        if cells is not None:
            o2 = cells.neigh_off[a_c, o_c]
            valid = cells.cell_valid[o_c] & cells.cell_valid[o2]
            gi = (cells.cell_gray[o_c].to(torch.int64) - 1).clamp_(0, ng - 1)
            gj = (cells.cell_gray[o2].to(torch.int64) - 1).clamp_(0, ng - 1)
        else:
            centres = grid.base_coords[None, :, :] + grid.offsets_t[o_c][:, None, :]
            neighbours = centres + grid.angles_t[a_c][:, None, :]
            valid = coords_in_image(centres, grid.size_t) & coords_in_image(
                neighbours, grid.size_t
            )
            centre_flat = flat_index(centres, grid.strides_t, grid.n_elements)
            neighbour_flat = flat_index(neighbours, grid.strides_t, grid.n_elements)
            valid = (
                valid
                & grid.mask_flat[centre_flat]
                & grid.mask_flat[neighbour_flat]
            )
            gi = (grid.img_flat[centre_flat] - 1).clamp_(0, ng - 1)
            gj = (grid.img_flat[neighbour_flat] - 1).clamp_(0, ng - 1)
        flat_idx = ((grid.matrix_ids[None, :] * ng + gi) * ng + gj) * na + a_c[:, None]
        parts.append(flat_idx[valid])
    if not parts:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(parts, dim=0)
