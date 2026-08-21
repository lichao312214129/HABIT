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
GPU (torch) implementation of the PyRadiomics GLSZM matrix calculation.

Drop-in replacement for ``radiomics.cMatrices.calculate_glszm`` (v3.0.1).

C semantics (``calculate_glszm`` / ``fill_glszm`` in ``cmatrices.c``):

- Angles are the *bidirectional* set at distance 1 (26-connected in 3D,
  8-connected in 2D). The C wrapper hard-codes ``distances = [1]``.
- For every masked, not-yet-processed voxel in the bounding box, a
  flood-fill grows the zone through in-box, masked neighbours of the
  same gray level. The zone is recorded as ``(gray, size)``.
- The output is cropped to ``maxRegion`` (largest zone found, at least
  1): ``P[v, i, j]`` is the number of zones with gray ``i+1`` and size
  ``j+1``. Empty gray levels stay and are dropped later in Python.

Voxel-based kernels are independent (one column per listed voxel) and
use label-propagation on the regular ``[-r, r]`` offset grid. Segment
mode uses the same propagation on the sparse list of masked voxels.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

from ._geom import CentreGrid, coords_in_image, element_strides, flat_index, prepare_centre_grid


def calculate_glszm(
    image: np.ndarray,
    mask: np.ndarray,
    Ng: int,
    Ns: int,
    force2D: bool = False,
    force2Ddimension: int = 0,
    kernelRadius: int = 0,
    voxelCoordinates: Optional[np.ndarray] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    GPU drop-in replacement for ``radiomics.cMatrices.calculate_glszm``.

    Args:
        image: Discretised image (gray levels 1..Ng inside the mask).
        mask: Boolean array, same shape as ``image``.
        Ng: Number of gray levels (max gray level inside the ROI).
        Ns: C temp-buffer bound (``mask.sum()``). Unused here; the
            returned size axis is cropped to the largest zone found,
            matching the C wrapper.
        force2D: Restrict angles and the kernel window to a 2D plane.
        force2Ddimension: Out-of-plane dimension when ``force2D`` is set.
        kernelRadius: Voxel-kernel radius; must be > 0 in voxel-based mode.
        voxelCoordinates: ``(Nd, Nvox)`` listed voxels, or ``None`` for
            segment-based mode.
        device: Torch device for the calculation and the output.
        dtype: Floating dtype of the returned matrix (counts are exact
            small integers).

    Returns:
        torch.Tensor: ``P_glszm`` of shape ``(Nvox, Ng, maxRegion)`` on
        ``device``. ``maxRegion`` is the largest zone actually found
        (1 if the ROI is empty).
    """
    del Ns  # C temp-buffer size; we crop to the observed max zone.
    distances = np.asarray([1], dtype=np.int32)
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
    if grid.voxel_based:
        return _accumulate_glszm_voxel(grid, int(Ng), dtype)
    return _accumulate_glszm_segment(grid, int(Ng), dtype)


def _offset_lookup(offsets: torch.Tensor, radius: int, nd: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map a packed local offset in ``[-r, r]^Nd`` to its row in ``offsets``.

    Args:
        offsets: ``(No, Nd)`` kernel offsets.
        radius: Kernel radius ``r``.
        nd: Dimensionality.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(lookup, pack_strides)``.
        ``lookup[pack(local)]`` is the offset index, or ``-1`` if that
        local vector is not an enumerated offset (force2D collapsed dim).
    """
    extent = 2 * radius + 1
    pack_strides = torch.as_tensor(
        element_strides([extent] * nd), dtype=torch.long, device=offsets.device
    )
    packed = ((offsets + radius) * pack_strides).sum(dim=-1)
    lookup = torch.full((extent ** nd,), -1, dtype=torch.long, device=offsets.device)
    lookup[packed] = torch.arange(offsets.shape[0], dtype=torch.long, device=offsets.device)
    return lookup, pack_strides


def _propagate_labels(
    label: torch.Tensor,
    valid: torch.Tensor,
    gl: torch.Tensor,
    neigh_idx: torch.Tensor,
    neigh_ok: torch.Tensor,
) -> torch.Tensor:
    """
    One label-propagation sweep: each voxel takes ``min(self, neighbours)``
    among same-gray, valid, in-box neighbours.

    Args:
        label: Current labels, ``(No, Nvox)`` or ``(N,)``.
        valid: Boolean mask of participating voxels, same leading shape
            as ``label`` without the neighbour axis.
        gl: Gray levels, same shape as ``label``.
        neigh_idx: Neighbour gather index, ``(Na, No)`` or ``(Na, N)``;
            ``-1`` marks an invalid neighbour.
        neigh_ok: Boolean extra constraint (in-image / in-window), same
            shape as ``neigh_idx`` broadcast against ``label``.

    Returns:
        torch.Tensor: Updated labels, same shape as ``label``.
    """
    safe = neigh_idx.clamp(min=0)
    if label.ndim == 2:
        # Voxel path: label is (No, Nvox); neigh_idx is (Na, No).
        label_n = label[safe]  # (Na, No, Nvox)
        gl_n = gl[safe]
        valid_n = valid[safe]
        ok = (
            neigh_ok
            & (neigh_idx[:, :, None] >= 0)
            & valid[None, :, :]
            & valid_n
            & (gl[None, :, :] == gl_n)
            & (label[None, :, :] > 0)
            & (label_n > 0)
        )
        huge = torch.iinfo(label.dtype).max
        cand = torch.where(ok, label_n, torch.full_like(label_n, huge))
        return torch.minimum(label, cand.min(dim=0).values)
    # Segment path: label is (N,); neigh_idx is (Na, N).
    label_n = label[safe]
    gl_n = gl[safe]
    valid_n = valid[safe]
    ok = (
        neigh_ok
        & (neigh_idx >= 0)
        & valid[None, :]
        & valid_n
        & (gl[None, :] == gl_n)
        & (label[None, :] > 0)
        & (label_n > 0)
    )
    huge = torch.iinfo(label.dtype).max
    cand = torch.where(ok, label_n, torch.full_like(label_n, huge))
    return torch.minimum(label, cand.min(dim=0).values)


def _histogram_zones(
    matrix_ids: torch.Tensor,
    labels: torch.Tensor,
    gl: torch.Tensor,
    valid: torch.Tensor,
    n_matrices: int,
    ng: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Turn per-voxel component labels into a ``(Nvox, Ng, maxRegion)`` GLSZM.

    Args:
        matrix_ids: Matrix index of every labelled voxel, 1-D.
        labels: Component labels (unique per matrix), 1-D.
        gl: Gray levels, 1-D.
        valid: Boolean mask; only ``True`` entries form zones.
        n_matrices: Number of output matrices.
        ng: Gray-level axis length.
        dtype: Output floating dtype.

    Returns:
        torch.Tensor: GLSZM cropped to the largest observed zone size.
    """
    device = labels.device
    if not bool(valid.any()):
        return torch.zeros(n_matrices, ng, 1, dtype=dtype, device=device)

    mid = matrix_ids[valid]
    lab = labels[valid]
    grey = (gl[valid] - 1).clamp(0, ng - 1).to(torch.int64)
    lab_min = int(lab.min())
    lab_span = int(lab.max()) - lab_min + 1
    zone_key = mid * lab_span + (lab - lab_min)

    if device.type == "cuda":
        # torch.unique / scatter_reduce on CUDA are correct for this many
        # int64 keys (the CPU torch kernels are not; see glrlm._lexsort),
        # so only the CPU path pays for the numpy host round-trip.
        uniq, inv = torch.unique(zone_key, return_inverse=True)
        sizes = torch.bincount(inv).to(torch.int64)
        z_mid = torch.div(uniq, lab_span, rounding_mode="floor")
        # Every voxel of a zone shares one gray level, so amin is exact.
        z_gl = torch.full((uniq.numel(),), ng, dtype=torch.int64, device=device)
        z_gl.scatter_reduce_(0, inv, grey, reduce="amin", include_self=True)
        max_region = max(int(sizes.max()), 1)
        rl = sizes - 1
        in_range = (rl >= 0) & (rl < max_region)
        p = torch.zeros(n_matrices * ng * max_region, dtype=torch.int32, device=device)
        flat = (z_mid[in_range] * ng + z_gl[in_range]) * max_region + rl[in_range]
        p.index_add_(0, flat, torch.ones(flat.numel(), dtype=torch.int32, device=device))
        return p.reshape(n_matrices, ng, max_region).to(dtype=dtype)

    mid_np = mid.detach().cpu().numpy().astype(np.int64)
    grey_np = grey.detach().cpu().numpy().astype(np.int64)
    zone_key_np = zone_key.detach().cpu().numpy().astype(np.int64)
    _uniq, first, inv = np.unique(zone_key_np, return_index=True, return_inverse=True)
    sizes = np.bincount(inv).astype(np.int64)
    z_mid = mid_np[first]
    z_gl = grey_np[first]
    max_region = max(int(sizes.max()), 1)
    p = np.zeros((n_matrices, ng, max_region), dtype=np.int32)
    rl = sizes - 1
    in_range = (rl >= 0) & (rl < max_region)
    np.add.at(p, (z_mid[in_range], z_gl[in_range], rl[in_range]), 1)
    return torch.as_tensor(p, dtype=dtype, device=device)


def _accumulate_glszm_voxel(grid: CentreGrid, ng: int, dtype: torch.dtype) -> torch.Tensor:
    """26/8-connected zones on the regular per-voxel kernel grid."""
    device = grid.device
    n_off = int(grid.offsets_t.shape[0])
    n_vox = grid.n_vox
    na = int(grid.angles_t.shape[0])
    if n_vox == 0 or n_off == 0:
        return torch.zeros(grid.n_matrices, ng, 1, dtype=dtype, device=device)

    centres = grid.base_coords[None, :, :] + grid.offsets_t[:, None, :]
    centre_in = coords_in_image(centres, grid.size_t)
    in_window = (
        (centres - grid.base_coords[None, :, :]).abs() <= grid.kernel_radius
    ).all(dim=-1)
    centre_flat = flat_index(centres, grid.strides_t, grid.n_elements)
    valid = centre_in & in_window & grid.mask_flat[centre_flat]
    gl = grid.img_flat[centre_flat]
    # Unique label per offset inside one kernel; columns (listed voxels)
    # never mix, so the same 1..No ids can be reused across columns.
    # int32 is enough (labels <= No = kernel volume) and halves the
    # gather traffic in every propagation sweep.
    label = torch.arange(1, n_off + 1, dtype=torch.int32, device=device).unsqueeze(1)
    label = label.expand(n_off, n_vox).clone()
    label = torch.where(valid, label, torch.zeros_like(label))

    lookup, pack_strides = _offset_lookup(grid.offsets_t, grid.kernel_radius, grid.nd)
    extent = 2 * grid.kernel_radius + 1
    # Neighbour offset index and in-image flag for every angle × offset.
    local_n = grid.offsets_t[None, :, :] + grid.angles_t[:, None, :]  # (Na, No, Nd)
    in_win = (local_n.abs() <= grid.kernel_radius).all(dim=-1)  # (Na, No)
    packed = ((local_n + grid.kernel_radius) * pack_strides).sum(dim=-1)
    packed_ok = (packed >= 0) & (packed < extent ** grid.nd) & in_win
    neigh_o = torch.where(
        packed_ok, lookup[packed.clamp(0, extent ** grid.nd - 1)], torch.full_like(packed, -1)
    )
    # in-image depends on the listed voxel (window may be clipped).
    neigh_coords = centres[None, :, :, :] + grid.angles_t[:, None, None, :]
    in_img = coords_in_image(neigh_coords, grid.size_t)  # (Na, No, Nvox)
    neigh_ok = packed_ok[:, :, None] & in_img

    # Diameter of a (2r+1)^Nd 26-connected set is small; stop early.
    for _ in range(n_off):
        new_label = _propagate_labels(label, valid, gl, neigh_o, neigh_ok)
        # Pointer jumping: each labelled voxel adopts the label stored at
        # its own label's position (labels index offsets within the same
        # column), doubling the distance the component minimum travels per
        # sweep. The connected partition -- and thus the zone sizes -- is
        # identical to pure propagation; only convergence is faster.
        # gather indices must be int64; the (No, Nvox) cast is cheap.
        jumped = torch.gather(new_label, 0, (new_label.to(torch.int64) - 1).clamp(min=0))
        new_label = torch.where(new_label > 0, jumped, new_label)
        if bool(torch.equal(new_label, label)):
            break
        label = new_label

    matrix_ids = grid.matrix_ids.expand(n_off, n_vox)
    return _histogram_zones(
        matrix_ids.reshape(-1),
        label.reshape(-1),
        gl.reshape(-1),
        valid.reshape(-1),
        grid.n_matrices,
        ng,
        dtype,
    )


def _accumulate_glszm_segment(grid: CentreGrid, ng: int, dtype: torch.dtype) -> torch.Tensor:
    """26/8-connected zones over the whole-ROI masked voxel list."""
    device = grid.device
    n = grid.n_vox
    if n == 0:
        return torch.zeros(1, ng, 1, dtype=dtype, device=device)

    coords = grid.base_coords  # (N, Nd), already the masked voxels
    flat = flat_index(coords, grid.strides_t, grid.n_elements)
    gl = grid.img_flat[flat]
    valid = torch.ones(n, dtype=torch.bool, device=device)
    # int32 labels halve the gather traffic; fall back to int64 if the
    # voxel count could ever exceed int32 (paranoia guard).
    lab_dtype = torch.int32 if n < 2 ** 31 - 1 else torch.int64
    label = torch.arange(1, n + 1, dtype=lab_dtype, device=device)

    lookup = torch.full((grid.n_elements,), -1, dtype=torch.long, device=device)
    lookup[flat] = torch.arange(n, dtype=torch.long, device=device)

    neigh_coords = coords[None, :, :] + grid.angles_t[:, None, :]  # (Na, N, Nd)
    in_img = coords_in_image(neigh_coords, grid.size_t)
    neigh_flat = flat_index(neigh_coords, grid.strides_t, grid.n_elements)
    neigh_i = torch.where(in_img, lookup[neigh_flat], torch.full_like(neigh_flat, -1))
    neigh_ok = in_img & (neigh_i >= 0)

    for _ in range(n):
        new_label = _propagate_labels(label, valid, gl, neigh_i, neigh_ok)
        # Pointer jumping (see the voxel path); same partition, fewer sweeps.
        jumped = new_label[(new_label.to(torch.int64) - 1).clamp(min=0)]
        new_label = torch.where(new_label > 0, jumped, new_label)
        if bool(torch.equal(new_label, label)):
            break
        label = new_label

    matrix_ids = torch.zeros(n, dtype=torch.long, device=device)
    return _histogram_zones(matrix_ids, label, gl, valid, 1, ng, dtype)
