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
GPU (torch) implementation of the PyRadiomics GLRLM matrix calculation.

Drop-in replacement for ``radiomics.cMatrices.calculate_glrlm`` (v3.0.1).

C semantics (``calculate_glrlm`` in ``cmatrices.c``):

- Angles are the *mono-directional* set at distance 1 (the C wrapper
  does not take a distances argument; it hard-codes ``[1]``).
- For each angle the C code walks every ray that starts on an incoming
  face of the bounding box. Consecutive masked voxels with the same
  gray level form a run of length ``L`` stored at run-length index
  ``L - 1``. An unmasked voxel (or leaving the box) breaks the run.
- After all rays of an angle, if no ray contained more than one masked
  voxel (``multiElement == 0``), every length-1 count for that angle is
  zeroed. This drops degenerate 2-D / single-voxel angles.

Vectorised equivalent: keep only masked in-box voxels, assign each a
ray key (invariants of ``x - t * angle``) and a position ``t`` along
the walk, sort, and run-length-encode. A gap ``Δt != |angle[d0]|`` is
the same break as an unmasked voxel (the box is convex, so every gap
on a ray is an in-box unmasked voxel).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ._geom import CentreGrid, coords_in_image, flat_index, prepare_centre_grid


def calculate_glrlm(
    image: np.ndarray,
    mask: np.ndarray,
    Ng: int,
    Nr: int,
    force2D: bool = False,
    force2Ddimension: int = 0,
    kernelRadius: int = 0,
    voxelCoordinates: Optional[np.ndarray] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU drop-in replacement for ``radiomics.cMatrices.calculate_glrlm``.

    Args:
        image: Discretised image (gray levels 1..Ng inside the mask).
        mask: Boolean array, same shape as ``image``.
        Ng: Number of gray levels (max gray level inside the ROI).
        Nr: Run-length axis length; PyRadiomics passes
            ``max(image.shape)``. Runs longer than ``Nr`` are not
            counted (C would raise). Empty trailing columns stay zero.
        force2D: Restrict angles and the kernel window to a 2D plane.
        force2Ddimension: Out-of-plane dimension when ``force2D`` is set.
        kernelRadius: Voxel-kernel radius; must be > 0 in voxel-based mode.
        voxelCoordinates: ``(Nd, Nvox)`` listed voxels, or ``None`` for
            segment-based mode.
        device: Torch device for the calculation and the outputs.
        dtype: Floating dtype of the returned matrix (counts are exact
            small integers).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(P_glrlm, angles)`` on
        ``device``. ``P_glrlm`` has shape ``(Nvox, Ng, Nr, Na)``;
        ``angles`` has shape ``(Na, Nd)``.
    """
    # C wrapper passes distances=NULL → [1], bidirectional=0.
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
        bidirectional=False,
    )
    return _accumulate_glrlm(grid, int(Ng), int(Nr), dtype)


def _lexsort(keys: List[torch.Tensor]) -> torch.Tensor:
    """
    Multi-key argsort, ``keys[0]`` least significant (numpy lexsort order).

    Packs the keys into a single int64 so one ``argsort`` is enough.
    A chained stable-argsort was bit-exact on CUDA but silently wrong
    on CPU once the point count grew (kernelRadius=3, a few hundred
    listed voxels): CPU ``argsort(stable=True)`` does not preserve
    earlier key order for duplicate values of the current key.

    Args:
        keys: 1-D integer tensors of equal length, least-significant first.

    Returns:
        torch.Tensor: Permutation indices of length ``keys[0].shape[0]``.

    Raises:
        RuntimeError: If the packed key would overflow int64.
    """
    packed = keys[-1].to(torch.int64)
    for key in keys[-2::-1]:
        key64 = key.to(torch.int64)
        kmin = int(key64.min().item()) if key64.numel() else 0
        kmax = int(key64.max().item()) if key64.numel() else 0
        span = int(kmax - kmin + 1)
        # packed * span + (key - kmin) must stay inside signed int64.
        if span <= 0:
            span = 1
        packed_min = int(packed.min().item()) if packed.numel() else 0
        packed_max = int(packed.max().item()) if packed.numel() else 0
        limit = (2 ** 63 - 1) // span
        if packed_max > limit or packed_min < -limit:
            raise RuntimeError(
                "GLRLM sort-key pack would overflow int64; "
                f"packed range [{packed_min}, {packed_max}], span={span}."
            )
        packed = packed * span + (key64 - kmin)
    # torch.argsort on CPU int64 is not a permutation for n around 3e4
    # (duplicate indices, missing values) on the torch builds we test;
    # CUDA argsort(stable=True) is a correct stable sort, so only the CPU
    # path pays for the numpy host round-trip.
    if packed.is_cuda:
        return torch.argsort(packed, stable=True)
    order = np.argsort(packed.detach().cpu().numpy(), kind="stable")
    return torch.as_tensor(order, dtype=torch.long, device=packed.device)


def _ray_invariants_and_t(
    coords: torch.Tensor,
    angle: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Ray identity and walk position for one angle.

    Two voxels lie on the same directed ray iff
    ``x - t * angle`` is the same intercept. The intercept is encoded
    as the ``nd`` invariants ``x[d] * a[d0] - x[d0] * a[d]`` (zero on
    the reference moving dim) together with the coordinates of the
    frozen dimensions (``a[d] == 0``). ``t = x[d0] * sign(a[d0])``
    increases along the C walk (``j += angle``).

    Args:
        coords: ``(N, Nd)`` integer voxel coordinates.
        angle: ``(Nd,)`` integer offset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: ``(invariants, t, step)``
        where ``invariants`` is ``(N, Nd)``, ``t`` is ``(N,)``, and
        ``step`` is ``|angle[d0]|`` (adjacent voxels differ by ``step``).
    """
    moving = torch.nonzero(angle != 0, as_tuple=False).reshape(-1)
    d0 = int(moving[0].item())
    a0 = int(angle[d0].item())
    sign0 = 1 if a0 > 0 else -1
    t = coords[:, d0] * sign0
    # invariants[:, d] = x[d] * a[d0] - x[d0] * a[d]
    # Frozen dims (a[d] == 0): this equals x[d] * a[d0], which still
    # uniquely tags the ray because a[d0] != 0. We store x[d] itself
    # for those dims so the key stays small and obvious.
    inv = coords * angle[d0] - coords[:, d0 : d0 + 1] * angle[None, :]
    frozen = angle == 0
    if bool(frozen.any()):
        inv = inv.clone()
        inv[:, frozen] = coords[:, frozen]
    return inv, t, abs(a0)


def _sort_order_static(
    t: torch.Tensor,
    inv: torch.Tensor,
    matrix_ids: torch.Tensor,
    n_matrices: int,
    size_max: int,
) -> Optional[torch.Tensor]:
    """
    Argsort of the packed ``(matrix, inv..., t)`` key with static bounds.

    Distance-1 angles and in-image coords give ``|t| <= size_max`` and
    ``|inv| <= 2 * size_max``, so the packing needs no measured min/max
    (which would force one GPU sync per key). Returns ``None`` when the
    static packing could overflow int64; the caller then falls back to
    the measured-bound ``_lexsort``.
    """
    nd = int(inv.shape[1])
    span_t = 2 * size_max + 1
    span_inv = 4 * size_max + 1
    # Host-side overflow check with python ints (no device traffic).
    if n_matrices * (span_inv ** nd) * span_t > 2 ** 63 - 1:
        return None
    packed = matrix_ids.to(torch.int64)
    for dim in range(nd):
        packed = packed * span_inv + (inv[:, dim].to(torch.int64) + 2 * size_max)
    packed = packed * span_t + (t.to(torch.int64) + size_max)
    if packed.is_cuda:
        return torch.argsort(packed, stable=True)
    order = np.argsort(packed.detach().cpu().numpy(), kind="stable")
    return torch.as_tensor(order, dtype=torch.long, device=packed.device)


def _next_start(lengths_ok: torch.Tensor, n_pts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Own-group start and next-group start for every sorted position.

    ``lengths_ok`` marks the first position of every group (run or ray).
    The group length at position ``i`` is ``next_start[i] - own_start[i]``.
    Pure fixed-shape ops (cummax / flipped cummin), so no GPU sync --
    unlike ``nonzero`` + ``item()`` on the group count.
    """
    arange_n = torch.arange(n_pts, device=lengths_ok.device)
    own_start = torch.cummax(torch.where(lengths_ok, arange_n, 0), 0).values
    # Reverse cummin gives the next start at OR AFTER i; shifting left by
    # one turns it into the next start strictly after i (sentinel n_pts).
    nxt = torch.where(lengths_ok, arange_n, n_pts)
    rcm = torch.flip(torch.cummin(torch.flip(nxt, [0]), 0).values, [0])
    nxt = torch.cat([rcm[1:], torch.full_like(rcm[:1], n_pts)])
    return own_start, nxt


def _accumulate_glrlm(
    grid: CentreGrid,
    ng: int,
    nr: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorised GLRLM: per-angle ray RLE over masked in-box centres."""
    device = grid.device
    na = int(grid.angles_t.shape[0])
    n_off = int(grid.offsets_t.shape[0])
    n_vox = grid.n_vox
    nd = grid.nd

    # int16 is exact: a run count is bounded by the kernel window volume.
    # One extra "dumpster" slot absorbs masked-out contributions, which
    # keeps every scatter fixed-shape (no nonzero / .item() syncs).
    dump = grid.n_matrices * ng * nr * na
    p_flat = torch.zeros(dump + 1, dtype=torch.int16, device=device)
    angles_out = grid.angles_t.to(dtype=dtype)
    if n_vox == 0 or n_off == 0:
        return p_flat[:dump].reshape(grid.n_matrices, ng, nr, na).to(dtype=dtype), angles_out

    # All in-box centres (masked later). Voxel mode: kernel window;
    # segment mode: one zero offset, base_coords already = masked voxels.
    centres = grid.base_coords[None, :, :] + grid.offsets_t[:, None, :]
    centre_in = coords_in_image(centres, grid.size_t)
    if grid.voxel_based:
        in_window = (
            (centres - grid.base_coords[None, :, :]).abs() <= grid.kernel_radius
        ).all(dim=-1)
        centre_in = centre_in & in_window
    centre_flat = flat_index(centres, grid.strides_t, grid.n_elements)
    centre_valid = centre_in & grid.mask_flat[centre_flat]
    if not bool(centre_valid.any()):
        return p_flat[:dump].reshape(grid.n_matrices, ng, nr, na).to(dtype=dtype), angles_out

    coords = centres[centre_valid]  # (N, Nd)
    matrix_ids = grid.matrix_ids.expand(n_off, n_vox)[centre_valid]
    gl = grid.img_flat[centre_flat][centre_valid]
    n_pts = int(coords.shape[0])
    size_max = max(grid.size)
    one16 = torch.ones((), dtype=torch.int16, device=device)
    zero16 = torch.zeros((), dtype=torch.int16, device=device)
    g_all = torch.arange(ng, device=device)
    v_all = torch.arange(grid.n_matrices, device=device)

    for a_idx in range(na):
        angle = grid.angles_t[a_idx]
        inv, t, step = _ray_invariants_and_t(coords, angle)
        order = _sort_order_static(t, inv, matrix_ids, grid.n_matrices, size_max)
        if order is None:
            # Static packing would overflow: measured-bound fallback.
            keys: List[torch.Tensor] = [t]
            for dim in range(nd - 1, -1, -1):
                keys.append(inv[:, dim])
            keys.append(matrix_ids)
            order = _lexsort(keys)
        mid_s = matrix_ids[order]
        inv_s = inv[order]
        t_s = t[order]
        gl_s = gl[order]

        is_new = torch.ones(n_pts, dtype=torch.bool, device=device)
        if n_pts > 1:
            same_ray = (mid_s[1:] == mid_s[:-1]) & (inv_s[1:] == inv_s[:-1]).all(dim=1)
            adjacent = (t_s[1:] - t_s[:-1]) == step
            same_gl = gl_s[1:] == gl_s[:-1]
            is_new[1:] = ~(same_ray & adjacent & same_gl)

        # Run length at every sorted position; only run starts contribute.
        own_start, next_start = _next_start(is_new, n_pts)
        rl = (next_start - own_start) - 1  # C stores a length-L run at index L-1
        run_gl = (gl_s - 1).clamp_(0, ng - 1)
        valid_run = is_new & (rl < nr)
        flat_idx = ((mid_s * ng + run_gl) * nr + rl) * na + a_idx
        p_flat.index_add_(
            0,
            torch.where(valid_run, flat_idx, dump),
            torch.where(valid_run, one16, zero16),
        )

        # multiElement: any ray of this matrix has 2+ masked voxels.
        ray_new = torch.ones(n_pts, dtype=torch.bool, device=device)
        if n_pts > 1:
            ray_new[1:] = ~((mid_s[1:] == mid_s[:-1]) & (inv_s[1:] == inv_s[:-1]).all(dim=1))
        own_ray, next_ray = _next_start(ray_new, n_pts)
        big = ray_new & ((next_ray - own_ray) >= 2)
        # int32: a matrix can own more rays than int16 holds.
        multi_counts = torch.zeros(grid.n_matrices + 1, dtype=torch.int32, device=device)
        multi_counts.index_add_(
            0,
            torch.where(big, mid_s, grid.n_matrices),
            torch.where(big, 1, 0).to(torch.int32),
        )
        multi = multi_counts[: grid.n_matrices] > 0
        # C: if (!multiElement) zero run-length index 0 for every gray
        # level of this angle. Fixed-shape gather/where/scatter: no sync.
        idx0 = ((v_all[:, None] * ng + g_all[None, :]) * nr + 0) * na + a_idx
        cur = p_flat[idx0]
        p_flat[idx0] = torch.where(~multi[:, None], zero16, cur)

    return p_flat[:dump].reshape(grid.n_matrices, ng, nr, na).to(dtype=dtype), angles_out
