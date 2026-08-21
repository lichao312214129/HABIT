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

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .angles import build_angles
from ._geom import element_strides as _element_strides
from ._geom import kernel_offsets as _kernel_offsets


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
    size = [int(s) for s in image.shape]
    nd = len(size)
    f2d_dim = int(force2Ddimension) if force2D else -1

    angles_np = build_angles(size, distances, force2Ddimension=f2d_dim, bidirectional=False)
    na = angles_np.shape[0]

    device = torch.device(device)
    size_t = torch.as_tensor(size, dtype=torch.long, device=device)
    strides_t = torch.as_tensor(_element_strides(size), dtype=torch.long, device=device)
    angles_t = torch.as_tensor(angles_np, dtype=torch.long, device=device)

    # C PyArray_FORCECAST truncates toward zero when casting the (possibly
    # float) binned image to int; torch.long conversion does the same.
    img_flat = torch.as_tensor(np.ascontiguousarray(image)).to(device=device, dtype=torch.long).reshape(-1)
    mask_flat = torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.bool, device=device).reshape(-1)

    if voxelCoordinates is None:
        # Segment-based mode: a single matrix; every masked voxel is a centre
        # and the only neighbour constraint is the image bounds.
        base_coords = torch.nonzero(mask_flat.reshape(tuple(size)), as_tuple=False)
        matrix_ids = torch.zeros(base_coords.shape[0], dtype=torch.long, device=device)
        offsets_np = np.zeros((1, nd), dtype=np.int64)
        # No kernel window: every angle pairs with the single zero offset.
        pair_a = torch.arange(na, dtype=torch.long, device=device)
        pair_o = torch.zeros(na, dtype=torch.long, device=device)
        n_matrices = 1
    else:
        coords = torch.as_tensor(np.ascontiguousarray(voxelCoordinates)).to(
            device=device, dtype=torch.long
        )
        if coords.ndim != 2 or coords.shape[0] != nd:
            raise ValueError(
                f"voxelCoordinates must have shape (Nd, Nvox) = ({nd}, N); "
                f"got {tuple(coords.shape)}"
            )
        base_coords = coords.t().contiguous()  # (Nvox, Nd)
        matrix_ids = torch.arange(base_coords.shape[0], dtype=torch.long, device=device)
        if kernelRadius <= 0:
            raise ValueError(f"kernelRadius must be > 0 in voxel-based mode; got {kernelRadius}")
        offsets_np = _kernel_offsets(nd, int(kernelRadius), f2d_dim)
        offsets_t = torch.as_tensor(offsets_np, dtype=torch.long, device=device)
        # A (angle, offset) pair can contribute only if the neighbour stays
        # inside the kernel window: |offset + angle| <= radius per dimension.
        neighbour_in_window = (
            (offsets_t[None, :, :] + angles_t[:, None, :]).abs() <= int(kernelRadius)
        ).all(dim=2)  # (Na, No)
        pair_a, pair_o = torch.nonzero(neighbour_in_window, as_tuple=True)
        n_matrices = base_coords.shape[0]

    offsets_t = torch.as_tensor(offsets_np, dtype=torch.long, device=device)

    # Accumulate counts in int16: per-entry counts are bounded by the number
    # of angles (<= 26 < 2^15), integer atomics are exact and use a quarter
    # of the memory traffic of float64. Cast to ``dtype`` once at the end;
    # the values are small integers, exactly representable in float32/64.
    # Accumulate counts in int16: a (Nvox, Ng, Ng, Na) GLCM entry can be
    # hit at most Na times (once per angle), and Na <= 26 < 2^15, so int16
    # is exact and halves the largest allocation vs int32. Cast to ``dtype``
    # once at the end.
    p_flat = torch.zeros(n_matrices * Ng * Ng * na, dtype=torch.int16, device=device)
    n_vox = base_coords.shape[0]
    n_pairs = int(pair_a.shape[0])
    chunk = max(1, int(max_chunk_elems) // max(1, n_vox))

    for start in range(0, n_pairs, chunk):
        a_c = pair_a[start : start + chunk]  # (Kc,) angle index per pair
        o_c = pair_o[start : start + chunk]  # (Kc,) kernel-offset index per pair

        # Centres and neighbours for every (pair, voxel) combination.
        centres = base_coords[None, :, :] + offsets_t[o_c][:, None, :]  # (Kc, Nvox, Nd)
        neighbours = centres + angles_t[a_c][:, None, :]  # (Kc, Nvox, Nd)

        # Both voxels must lie inside the image (the per-voxel kernel window
        # is the image-clipped [-r, r]^Nd box, so bounds checks here implement
        # the same clipping as the C bb handling).
        valid = (
            (centres >= 0).all(dim=2)
            & (centres < size_t).all(dim=2)
            & (neighbours >= 0).all(dim=2)
            & (neighbours < size_t).all(dim=2)
        )
        if not bool(valid.any()):
            continue

        centre_flat = (centres * strides_t).sum(dim=2)  # (Kc, Nvox)
        neighbour_flat = (neighbours * strides_t).sum(dim=2)
        # Out-of-bounds coordinates produce out-of-range (or silently
        # negative-wrapping) flat indices; clamp them — the gathered values
        # at those positions are don't-care because ``valid`` drops them.
        n_elements = mask_flat.shape[0]
        centre_flat.clamp_(0, n_elements - 1)
        neighbour_flat.clamp_(0, n_elements - 1)

        # Both voxels must be part of the ROI (mask[i] and mask[j] in C).
        valid = valid & mask_flat[centre_flat] & mask_flat[neighbour_flat]
        if not bool(valid.any()):
            continue

        # Gray levels are guaranteed in [1, Ng] on masked voxels; the clamp
        # only sanitises don't-care values at invalid (unmasked) positions.
        gi = (img_flat[centre_flat] - 1).clamp_(0, Ng - 1)
        gj = (img_flat[neighbour_flat] - 1).clamp_(0, Ng - 1)

        # C-order flat index of element [v, i, j, a] in (Nmat, Ng, Ng, Na).
        flat_idx = ((matrix_ids[None, :] * Ng + gi) * Ng + gj) * na + a_c[:, None]
        selected = flat_idx[valid]
        p_flat.index_add_(
            0, selected, torch.ones(selected.numel(), dtype=torch.int16, device=device)
        )

    p_glcm = p_flat.reshape(n_matrices, Ng, Ng, na).to(dtype=dtype)
    angles_out = angles_t.to(dtype=dtype)
    return p_glcm, angles_out
