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

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from ._geom import CentreGrid, coords_in_image, element_strides, flat_index, prepare_centre_grid
from .sparse_coo import SparseGLSZM, unique_counts, voxel_ns_cap

# Reused across voxel batches of the same (No, Nvox) shape. empty_cache is
# not the product: these tensors stay live and are zero-filled in place.
_LABEL_BUF: Dict[str, torch.Tensor] = {}


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


def calculate_glszm_coo(
    image: np.ndarray,
    mask: np.ndarray,
    Ng: int,
    Ns: int = 0,
    force2D: bool = False,
    force2Ddimension: int = 0,
    kernelRadius: int = 0,
    voxelCoordinates: Optional[np.ndarray] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float64,
) -> SparseGLSZM:
    """
    Sparse-COO GLSZM. Never allocates a dense ``(B, Ng, maxRegion)`` store.

    Voxel mode: zone sizes are bounded by ``(2r+1)^Nd``. Segment mode
    stores only the zones that exist (protocol ``Ns`` is unused, matching
    the dense kernel).

    Args:
        image: Discretised image (gray levels 1..Ng inside the mask).
        mask: Boolean array, same shape as ``image``.
        Ng: Number of gray levels.
        Ns: C temp-buffer bound; unused (kept for call-site compatibility).
        force2D: Restrict angles and the kernel window to a 2D plane.
        force2Ddimension: Out-of-plane dimension when ``force2D`` is set.
        kernelRadius: Voxel-kernel radius; must be > 0 in voxel-based mode.
        voxelCoordinates: ``(Nd, Nvox)`` listed voxels, or ``None`` for
            segment-based mode.
        device: Torch device.
        dtype: Count / feature compute dtype (float64 default).

    Returns:
        SparseGLSZM: COO zone counts on ``device``.
    """
    del Ns
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
        return _accumulate_glszm_voxel_coo(grid, int(Ng), dtype)
    return _accumulate_glszm_segment_coo(grid, int(Ng), dtype)


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


def _zone_links(
    valid: torch.Tensor,
    gl: torch.Tensor,
    neigh_idx: torch.Tensor,
    neigh_ok: torch.Tensor,
) -> torch.Tensor:
    """
    Boolean "these two voxels belong to the same zone" mask, once per call.

    A link exists when the neighbour is enumerated, in the box/image, both
    endpoints are masked, and both carry the same gray level. None of that
    depends on the current labels, so the mask is a loop invariant of the
    propagation: recomputing it every sweep gathered the int64 gray levels
    and re-evaluated five boolean terms over the whole ``(Na, No, Nvox)``
    neighbour tensor for nothing.

    The two label terms of the original predicate (``label > 0`` and
    ``label_n > 0``) are dropped because they are equivalent to ``valid``
    and ``valid_n``: labels start as ``valid ? id : 0``, and both the min
    sweep and the pointer jump keep a masked voxel positive and an unmasked
    one at zero. The propagation therefore visits exactly the same links.

    Args:
        valid: Masked-voxel flag, ``(No, Nvox)`` or ``(N,)``.
        gl: Gray levels, same shape as ``valid``.
        neigh_idx: Neighbour gather index, ``(Na, No)`` or ``(Na, N)``;
            ``-1`` marks an invalid neighbour.
        neigh_ok: In-image / in-window flag, broadcastable to the
            neighbour tensor shape.

    Returns:
        torch.Tensor: Boolean mask, ``(Na, No, Nvox)`` or ``(Na, N)``.
    """
    safe = neigh_idx.clamp(min=0)
    if valid.ndim == 2:
        # Voxel path: valid is (No, Nvox); neigh_idx is (Na, No).
        return (
            neigh_ok
            & (neigh_idx[:, :, None] >= 0)
            & valid[None, :, :]
            & valid[safe]
            & (gl[None, :, :] == gl[safe])
        )
    # Segment path: valid is (N,); neigh_idx is (Na, N).
    return (
        neigh_ok
        & (neigh_idx >= 0)
        & valid[None, :]
        & valid[safe]
        & (gl[None, :] == gl[safe])
    )


def _propagate_labels(
    label: torch.Tensor,
    neigh_idx: torch.Tensor,
    links: torch.Tensor,
) -> torch.Tensor:
    """
    One label-propagation sweep: each voxel takes ``min(self, neighbours)``
    over its zone links.

    Args:
        label: Current labels, ``(No, Nvox)`` or ``(N,)``.
        neigh_idx: Neighbour gather index, ``(Na, No)`` or ``(Na, N)``.
        links: Same-zone mask from :func:`_zone_links`.

    Returns:
        torch.Tensor: Updated labels, same shape as ``label``.
    """
    label_n = label[neigh_idx.clamp(min=0)]
    # A 0-dim fill value broadcasts, so ``where`` does not need a second
    # full-size tensor just to hold the sentinel.
    huge = torch.tensor(
        torch.iinfo(label.dtype).max, dtype=label.dtype, device=label.device
    )
    cand = torch.where(links, label_n, huge)
    return torch.minimum(label, cand.amin(dim=0))


def _histogram_zones(
    matrix_ids: torch.Tensor,
    labels: torch.Tensor,
    gl: torch.Tensor,
    valid: torch.Tensor,
    n_matrices: int,
    ng: int,
    dtype: torch.dtype,
    label_bound: int,
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
        label_bound: Exclusive-plus-one upper bound on ``labels``
            (``n_off`` on the voxel path, ``n`` on the segment path).
            Used as a static key span so we do not sync ``lab.min/max``.

    Returns:
        torch.Tensor: GLSZM cropped to the largest observed zone size.
    """
    z_mid, z_gl, sizes = _zone_table(
        matrix_ids, labels, gl, valid, ng, label_bound
    )
    device = labels.device
    if z_mid.numel() == 0:
        return torch.zeros(n_matrices, ng, 1, dtype=dtype, device=device)
    max_region = max(int(sizes.max().item()), 1)
    rl = sizes - 1
    in_range = (rl >= 0) & (rl < max_region)
    p = torch.zeros(n_matrices * ng * max_region, dtype=torch.int32, device=device)
    flat = (z_mid[in_range] * ng + z_gl[in_range]) * max_region + rl[in_range]
    p.index_add_(0, flat, torch.ones(flat.numel(), dtype=torch.int32, device=device))
    return p.reshape(n_matrices, ng, max_region).to(dtype=dtype)


def _zone_table(
    matrix_ids: torch.Tensor,
    labels: torch.Tensor,
    gl: torch.Tensor,
    valid: torch.Tensor,
    ng: int,
    label_bound: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One row per connected zone: ``(matrix, gray_idx, size)``.

    Args:
        matrix_ids: Matrix index of every labelled voxel, 1-D.
        labels: Component labels (unique per matrix), 1-D.
        gl: Gray levels, 1-D.
        valid: Boolean mask; only ``True`` entries form zones.
        ng: Gray-level axis length.
        label_bound: Exclusive-plus-one upper bound on ``labels``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ``z_mid``, ``z_gl`` (0-based), ``sizes`` (voxel count).
    """
    device = labels.device
    empty = torch.empty(0, dtype=torch.int64, device=device)
    if not bool(valid.any()):
        return empty, empty, empty
    mid = matrix_ids[valid]
    lab = labels[valid]
    grey = (gl[valid] - 1).clamp(0, ng - 1).to(torch.int64)
    lab_span = int(label_bound) + 1
    zone_key = mid * lab_span + lab
    if device.type == "cuda":
        uniq, inv = torch.unique(zone_key, return_inverse=True)
        sizes = torch.bincount(inv).to(torch.int64)
        z_mid = torch.div(uniq, lab_span, rounding_mode="floor")
        z_gl = torch.full((uniq.numel(),), ng, dtype=torch.int64, device=device)
        z_gl.scatter_reduce_(0, inv, grey, reduce="amin", include_self=True)
        return z_mid, z_gl, sizes
    mid_np = mid.detach().cpu().numpy().astype(np.int64)
    grey_np = grey.detach().cpu().numpy().astype(np.int64)
    zone_key_np = zone_key.detach().cpu().numpy().astype(np.int64)
    _uniq, first, inv = np.unique(zone_key_np, return_index=True, return_inverse=True)
    sizes = np.bincount(inv).astype(np.int64)
    return (
        torch.as_tensor(mid_np[first], dtype=torch.int64, device=device),
        torch.as_tensor(grey_np[first], dtype=torch.int64, device=device),
        torch.as_tensor(sizes, dtype=torch.int64, device=device),
    )


def _zones_to_coo(
    z_mid: torch.Tensor,
    z_gl: torch.Tensor,
    sizes: torch.Tensor,
    n_matrices: int,
    ng: int,
    ns_cap: int,
    dtype: torch.dtype,
) -> SparseGLSZM:
    """Collapse zones that share ``(voxel, gray, size)`` into COO counts."""
    device = z_mid.device
    empty = torch.empty(0, dtype=torch.long, device=device)
    if z_mid.numel() == 0:
        return SparseGLSZM(
            voxel=empty,
            gray=empty,
            size=empty,
            count=torch.empty(0, dtype=dtype, device=device),
            n_vox=n_matrices,
            ng=ng,
            ns_cap=ns_cap,
        )
    rl = (sizes - 1).clamp(min=0)
    if ns_cap > 0:
        in_range = rl < int(ns_cap)
        z_mid = z_mid[in_range]
        z_gl = z_gl[in_range]
        rl = rl[in_range]
    ns_used = max(int(ns_cap), int(rl.max().item()) + 1 if rl.numel() else 1)
    key = (z_mid * ng + z_gl) * ns_used + rl
    uniq, counts = unique_counts(key, dtype)
    ns_i = ns_used
    voxel = torch.div(uniq, ng * ns_i, rounding_mode="floor")
    rem = uniq - voxel * (ng * ns_i)
    gray_u = torch.div(rem, ns_i, rounding_mode="floor")
    size_u = rem - gray_u * ns_i
    return SparseGLSZM(
        voxel=voxel,
        gray=gray_u,
        size=size_u,
        count=counts,
        n_vox=n_matrices,
        ng=ng,
        ns_cap=ns_used,
    )


def _reuse_label(n_off: int, n_vox: int, device: torch.device) -> torch.Tensor:
    """Reuse the ``(No, Nvox)`` label buffer across equal-shaped batches."""
    tag = "voxel_label"
    shape = (n_off, n_vox)
    prev = _LABEL_BUF.get(tag)
    if (
        prev is None
        or prev.shape != shape
        or prev.dtype != torch.int32
        or prev.device != device
    ):
        prev = torch.empty(shape, dtype=torch.int32, device=device)
        _LABEL_BUF[tag] = prev
    ids = torch.arange(1, n_off + 1, dtype=torch.int32, device=device).unsqueeze(1)
    prev.copy_(ids.expand(n_off, n_vox))
    return prev


def _accumulate_glszm_voxel(grid: CentreGrid, ng: int, dtype: torch.dtype) -> torch.Tensor:
    """26/8-connected zones on the regular per-voxel kernel grid."""
    fields = _voxel_zone_fields(grid)
    if fields is None:
        return torch.zeros(grid.n_matrices, ng, 1, dtype=dtype, device=grid.device)
    matrix_ids, label, gl, valid, n_off = fields
    return _histogram_zones(
        matrix_ids.reshape(-1),
        label.reshape(-1),
        gl.reshape(-1),
        valid.reshape(-1),
        grid.n_matrices,
        ng,
        dtype,
        n_off,
    )


def _accumulate_glszm_voxel_coo(
    grid: CentreGrid,
    ng: int,
    dtype: torch.dtype,
) -> SparseGLSZM:
    """Voxel GLSZM as COO. Zone axis cap is ``(2r+1)^Nd``."""
    ns_cap = voxel_ns_cap(grid.kernel_radius, grid.nd)
    fields = _voxel_zone_fields(grid)
    if fields is None:
        empty = torch.empty(0, dtype=torch.long, device=grid.device)
        return SparseGLSZM(
            voxel=empty,
            gray=empty,
            size=empty,
            count=torch.empty(0, dtype=dtype, device=grid.device),
            n_vox=grid.n_matrices,
            ng=ng,
            ns_cap=ns_cap,
        )
    matrix_ids, label, gl, valid, n_off = fields
    z_mid, z_gl, sizes = _zone_table(
        matrix_ids.reshape(-1),
        label.reshape(-1),
        gl.reshape(-1),
        valid.reshape(-1),
        ng,
        n_off,
    )
    return _zones_to_coo(z_mid, z_gl, sizes, grid.n_matrices, ng, ns_cap, dtype)


def _voxel_zone_fields(
    grid: CentreGrid,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Label-propagate one voxel-kernel batch. Shared by dense and COO."""
    device = grid.device
    n_off = int(grid.offsets_t.shape[0])
    n_vox = grid.n_vox
    if n_vox == 0 or n_off == 0:
        return None

    centres = grid.base_coords[None, :, :] + grid.offsets_t[:, None, :]
    centre_in = coords_in_image(centres, grid.size_t)
    in_window = (
        (centres - grid.base_coords[None, :, :]).abs() <= grid.kernel_radius
    ).all(dim=-1)
    centre_flat = flat_index(centres, grid.strides_t, grid.n_elements)
    valid = centre_in & in_window & grid.mask_flat[centre_flat]
    gl = grid.img_flat[centre_flat]
    label = _reuse_label(n_off, n_vox, device)
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

    # Blind sweeps first (no device sync). Pointer jumping is idempotent,
    # but 8 rounds is not always enough on a full 7^3 window, so we then
    # poll torch.equal until the partition stops changing -- same stop
    # condition as before, fewer syncs on the typical short path.
    radius = max(int(grid.kernel_radius), 1)
    n_blind = min(n_off, max(8, int(math.ceil(math.log2(max(2 * radius, 2)))) + 4))
    links = _zone_links(valid, gl, neigh_o, neigh_ok)

    def _step(current: torch.Tensor) -> torch.Tensor:
        new_label = _propagate_labels(current, neigh_o, links)
        jumped = torch.gather(new_label, 0, (new_label.to(torch.int64) - 1).clamp(min=0))
        return torch.where(new_label > 0, jumped, new_label)

    for _ in range(n_blind):
        label = _step(label)
    for _ in range(n_off - n_blind):
        new_label = _step(label)
        if bool(torch.equal(new_label, label)):
            break
        label = new_label

    matrix_ids = grid.matrix_ids.expand(n_off, n_vox)
    return matrix_ids, label, gl, valid, n_off


def _accumulate_glszm_segment_coo(
    grid: CentreGrid,
    ng: int,
    dtype: torch.dtype,
) -> SparseGLSZM:
    """Segment GLSZM as COO over zones that actually exist."""
    p = _accumulate_glszm_segment(grid, ng, dtype)
    # Segment mode is one matrix; convert the (already cropped) dense
    # gold histogram to COO so feature code is unified. The dense P here
    # is ``(1, Ng, maxRegion)`` with maxRegion = largest zone, not a
    # per-voxel Ng * Ns bomb.
    n_matrices, ng_i, ns = int(p.shape[0]), int(p.shape[1]), int(p.shape[2])
    nz = torch.nonzero(p, as_tuple=False)
    if nz.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=p.device)
        return SparseGLSZM(
            voxel=empty,
            gray=empty,
            size=empty,
            count=torch.empty(0, dtype=dtype, device=p.device),
            n_vox=n_matrices,
            ng=ng_i,
            ns_cap=ns,
        )
    return SparseGLSZM(
        voxel=nz[:, 0].to(torch.long),
        gray=nz[:, 1].to(torch.long),
        size=nz[:, 2].to(torch.long),
        count=p[nz[:, 0], nz[:, 1], nz[:, 2]].to(dtype=dtype),
        n_vox=n_matrices,
        ng=ng_i,
        ns_cap=ns,
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

    n_blind = min(n, max(8, int(math.ceil(math.log2(max(n, 2)))) + 4))
    links = _zone_links(valid, gl, neigh_i, neigh_ok)

    def _step_seg(current: torch.Tensor) -> torch.Tensor:
        new_label = _propagate_labels(current, neigh_i, links)
        jumped = new_label[(new_label.to(torch.int64) - 1).clamp(min=0)]
        return torch.where(new_label > 0, jumped, new_label)

    for _ in range(n_blind):
        label = _step_seg(label)
    for _ in range(n - n_blind):
        new_label = _step_seg(label)
        if bool(torch.equal(new_label, label)):
            break
        label = new_label

    matrix_ids = torch.zeros(n, dtype=torch.long, device=device)
    return _histogram_zones(matrix_ids, label, gl, valid, 1, ng, dtype, n)
