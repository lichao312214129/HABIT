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


def _ray_invariants_and_t_batch(
    coords: torch.Tensor,
    angles: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ray identity and walk position for every angle at once.

    Two voxels lie on the same directed ray iff ``x - t * angle`` is the
    same intercept. The intercept is encoded as the ``nd`` invariants
    ``x[d] * a[d0] - x[d0] * a[d]`` (zero on the reference moving dim)
    together with the coordinates of the frozen dimensions (``a[d] == 0``).
    ``t = x[d0] * sign(a[d0])`` increases along the C walk (``j += angle``),
    and adjacent voxels on a ray differ by ``step = |a[d0]|``, which is 1
    for the distance-1 angle set. No Python loop and no ``.item()`` syncs.

    Args:
        coords: ``(N, Nd)`` integer voxel coordinates.
        angles: ``(Na, Nd)`` integer offsets.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ``invariants`` ``(Na, N, Nd)``, ``t`` ``(Na, N)``, ``step``
        ``(Na,)`` = ``|angle[d0]|``.
    """
    # First nonzero axis per angle (PyRadiomics distance-1 set is never 0).
    d0 = (angles != 0).to(torch.long).argmax(dim=1)
    a0 = angles.gather(1, d0.unsqueeze(1)).squeeze(1)
    sign0 = torch.where(a0 > 0, torch.ones_like(a0), -torch.ones_like(a0))
    # coords[:, d0] is (N, Na); t[a, i] = coords[i, d0[a]] * sign(a0[a]).
    t = (coords[:, d0] * sign0.unsqueeze(0)).permute(1, 0)
    coords_d0 = coords[:, d0].permute(1, 0)
    inv = (
        coords.unsqueeze(0) * a0[:, None, None]
        - coords_d0[:, :, None] * angles[:, None, :]
    )
    frozen = (angles == 0)[:, None, :]
    inv = torch.where(frozen, coords.unsqueeze(0), inv)
    return inv, t, a0.abs()


def _sort_order_static(
    t: torch.Tensor,
    inv: torch.Tensor,
    matrix_ids: torch.Tensor,
    n_matrices: int,
    size_max: int,
    angle_ids: Optional[torch.Tensor] = None,
    n_angles: int = 1,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Argsort of the packed ``(matrix, angle, inv..., t)`` key with static bounds.

    Distance-1 angles and in-image coords give ``|t| <= size_max`` and
    ``|inv| <= 2 * size_max``, so the packing needs no measured min/max
    (which would force one GPU sync per key). Returns ``(None, None)`` when
    the static packing could overflow int64; the caller then falls back to
    the measured-bound ``_lexsort``.

    The packed key is returned as well: it holds ``(matrix, angle, inv, t)``
    in that significance order, so the caller can read every one of those
    fields back out of the sorted key with integer arithmetic instead of
    gathering five more permuted arrays.

    Args:
        t: Walk position, shape ``(M,)``.
        inv: Ray invariants, shape ``(M, Nd)``.
        matrix_ids: Listed-voxel / matrix index, shape ``(M,)``.
        n_matrices: Number of output matrices (upper bound on ``matrix_ids``).
        size_max: ``max(image.shape)``; static packing bound.
        angle_ids: Optional angle index, shape ``(M,)``. Omitted for a
            single-angle sort (legacy).
        n_angles: Angle-axis length when ``angle_ids`` is set.

    Returns:
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        ``(order, packed)``, both ``None`` on potential overflow.
    """
    nd = int(inv.shape[1])
    span_t = 2 * size_max + 1
    span_inv = 4 * size_max + 1
    n_groups = int(n_matrices) * int(n_angles)
    # Host-side overflow check with python ints (no device traffic).
    if n_groups * (span_inv ** nd) * span_t > 2 ** 63 - 1:
        return None, None
    packed = matrix_ids.to(torch.int64)
    if angle_ids is not None:
        packed = packed * int(n_angles) + angle_ids.to(torch.int64)
    for dim in range(nd):
        packed = packed * span_inv + (inv[:, dim].to(torch.int64) + 2 * size_max)
    packed = packed * span_t + (t.to(torch.int64) + size_max)
    if packed.is_cuda:
        return torch.argsort(packed, stable=True), packed
    order = np.argsort(packed.detach().cpu().numpy(), kind="stable")
    return torch.as_tensor(order, dtype=torch.long, device=packed.device), packed


def _next_start(lengths_ok: torch.Tensor, n_pts: int) -> torch.Tensor:
    """
    Next-group start for every sorted position.

    ``lengths_ok`` marks the first position of every group (run or ray). At
    a group start ``i`` the group length is ``next_start[i] - i``, which is
    all callers need -- the own-group start is ``i`` there by definition, so
    no ``cummax`` is required. Pure fixed-shape ops (flipped cummin), so no
    GPU sync -- unlike ``nonzero`` + ``item()`` on the group count.

    Args:
        lengths_ok: Boolean group-start mask, shape ``(n_pts,)``.
        n_pts: Length of ``lengths_ok``; also the past-the-end sentinel.

    Returns:
        torch.Tensor: Index of the next group start strictly after each
        position, ``n_pts`` when there is none.
    """
    arange_n = torch.arange(n_pts, device=lengths_ok.device)
    # Reverse cummin gives the next start at OR AFTER i; shifting left by
    # one turns it into the next start strictly after i (sentinel n_pts).
    nxt = torch.where(lengths_ok, arange_n, n_pts)
    rcm = torch.flip(torch.cummin(torch.flip(nxt, [0]), 0).values, [0])
    return torch.cat([rcm[1:], torch.full_like(rcm[:1], n_pts)])


def _accumulate_glrlm(
    grid: CentreGrid,
    ng: int,
    nr: int,
    dtype: torch.dtype,
    max_sort_elems: int = 1 << 23,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorised GLRLM: one batched ray RLE over all angles."""
    device = grid.device
    na = int(grid.angles_t.shape[0])
    n_off = int(grid.offsets_t.shape[0])
    n_vox = grid.n_vox

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
    coords = centres[centre_valid]
    n_pts = int(coords.shape[0])
    if n_pts == 0:
        return p_flat[:dump].reshape(grid.n_matrices, ng, nr, na).to(dtype=dtype), angles_out

    matrix_ids = grid.matrix_ids.expand(n_off, n_vox)[centre_valid]
    gl = grid.img_flat[centre_flat][centre_valid]
    size_max = max(grid.size)
    # Keep every point of one matrix in the same sort so runs are not split.
    max_pts = max(1, int(max_sort_elems) // max(na, 1))
    if n_pts <= max_pts:
        _rle_all_angles(
            coords, matrix_ids, gl, grid.angles_t, p_flat, dump,
            grid.n_matrices, ng, nr, na, size_max,
        )
    else:
        counts = torch.bincount(matrix_ids, minlength=grid.n_matrices)
        counts_np = counts.detach().cpu().numpy()
        start = 0
        acc = 0
        for mid, count in enumerate(counts_np.tolist()):
            count_i = int(count)
            if acc > 0 and acc + count_i > max_pts:
                sl = (matrix_ids >= start) & (matrix_ids < mid)
                _rle_all_angles(
                    coords[sl], matrix_ids[sl], gl[sl], grid.angles_t, p_flat,
                    dump, grid.n_matrices, ng, nr, na, size_max,
                )
                start = mid
                acc = 0
            acc += count_i
        sl = matrix_ids >= start
        _rle_all_angles(
            coords[sl], matrix_ids[sl], gl[sl], grid.angles_t, p_flat, dump,
            grid.n_matrices, ng, nr, na, size_max,
        )
    return p_flat[:dump].reshape(grid.n_matrices, ng, nr, na).to(dtype=dtype), angles_out


def _rle_all_angles(
    coords: torch.Tensor,
    matrix_ids: torch.Tensor,
    gl: torch.Tensor,
    angles: torch.Tensor,
    p_flat: torch.Tensor,
    dump: int,
    n_matrices: int,
    ng: int,
    nr: int,
    na: int,
    size_max: int,
) -> None:
    """
    Sort every angle together and scatter run-length counts into ``p_flat``.

    Same C rules as the old per-angle loop: a run is consecutive same-gray
    voxels on one directed ray of one matrix; ``multiElement`` zeros
    length-1 bins when no ray of that ``(matrix, angle)`` has 2+ voxels.

    Args:
        coords: Masked in-box centres, ``(N, Nd)``.
        matrix_ids: Matrix index per centre, ``(N,)``.
        gl: Gray levels per centre, ``(N,)``.
        angles: ``(Na, Nd)`` integer offsets.
        p_flat: Flat GLRLM plus dumpster slot, mutated in place.
        dump: Dumpster index (``len(p_flat) - 1``).
        n_matrices: Output matrix count.
        ng: Gray-level axis.
        nr: Run-length axis.
        na: Angle axis.
        size_max: Static packing bound (``max(image.shape)``).
    """
    n_pts = int(coords.shape[0])
    if n_pts == 0 or na == 0:
        return
    device = coords.device
    nd = int(coords.shape[1])
    inv, t, step = _ray_invariants_and_t_batch(coords, angles)
    n_all = na * n_pts
    inv_b = inv.reshape(n_all, nd)
    t_b = t.reshape(n_all)
    angle_ids = (
        torch.arange(na, device=device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(na, n_pts)
        .reshape(n_all)
    )
    mid_b = matrix_ids.unsqueeze(0).expand(na, n_pts).reshape(n_all)
    gl_b = gl.unsqueeze(0).expand(na, n_pts).reshape(n_all)
    order, packed = _sort_order_static(
        t_b, inv_b, mid_b, n_matrices, size_max,
        angle_ids=angle_ids, n_angles=na,
    )
    if order is None:
        keys: List[torch.Tensor] = [t_b]
        for dim in range(nd - 1, -1, -1):
            keys.append(inv_b[:, dim])
        keys.append(angle_ids)
        keys.append(mid_b)
        order = _lexsort(keys)

    if packed is not None:
        # The sorted key already carries (matrix, angle, inv, t); unpacking it
        # costs a few elementwise ops, while gathering mid / ang / inv / t
        # through ``order`` would move Nd + 3 permuted int64 arrays of length
        # na * n_pts. Same fields, same run/ray predicates, same counts.
        span_t = 2 * size_max + 1
        packed_s = packed[order]
        ray_key = packed_s // span_t
        t_s = packed_s - ray_key * span_t - size_max
        ang_mid = ray_key // ((4 * size_max + 1) ** nd)
        ang_s = ang_mid % na
        mid_s = ang_mid // na
        same_ray_core = None if n_all <= 1 else (ray_key[1:] == ray_key[:-1])
    else:
        mid_s = mid_b[order]
        ang_s = angle_ids[order]
        t_s = t_b[order]
        inv_s = inv_b[order]
        same_ray_core = (
            None
            if n_all <= 1
            else (
                (mid_s[1:] == mid_s[:-1])
                & (ang_s[1:] == ang_s[:-1])
                & (inv_s[1:] == inv_s[:-1]).all(dim=1)
            )
        )
    gl_s = gl_b[order]

    ray_new = torch.ones(n_all, dtype=torch.bool, device=device)
    is_new = torch.ones(n_all, dtype=torch.bool, device=device)
    if same_ray_core is not None:
        # step is |angle[d0]|, one value per angle; a break is only tested
        # between two points of the same ray, so either side's angle serves.
        step_s = step.to(torch.int64)[ang_s]
        adjacent = (t_s[1:] - t_s[:-1]) == step_s[1:]
        same_gl = gl_s[1:] == gl_s[:-1]
        ray_new[1:] = ~same_ray_core
        is_new[1:] = ~(same_ray_core & adjacent & same_gl)

    # At a group start the own-group start is the position itself, so the
    # length is next_start - position.
    arange_all = torch.arange(n_all, device=device)
    rl = (_next_start(is_new, n_all) - arange_all) - 1
    run_gl = (gl_s - 1).clamp_(0, ng - 1)
    valid_run = is_new & (rl < nr)
    flat_idx = ((mid_s * ng + run_gl) * nr + rl) * na + ang_s
    one16 = torch.ones((), dtype=torch.int16, device=device)
    zero16 = torch.zeros((), dtype=torch.int16, device=device)
    p_flat.index_add_(
        0,
        torch.where(valid_run, flat_idx, dump),
        torch.where(valid_run, one16, zero16),
    )

    # A ray starting at i holds 2+ points exactly when i + 1 is not itself a
    # ray start, which replaces a second cummax / cummin pass over n_all.
    last_false = torch.zeros(1, dtype=torch.bool, device=device)
    big = ray_new & torch.cat([~ray_new[1:], last_false])
    n_ma = n_matrices * na
    multi_counts = torch.zeros(n_ma + 1, dtype=torch.int32, device=device)
    pair_idx = mid_s * na + ang_s
    multi_counts.index_add_(
        0,
        torch.where(big, pair_idx, n_ma),
        torch.where(big, 1, 0).to(torch.int32),
    )
    multi = (multi_counts[:n_ma] > 0).reshape(n_matrices, na)
    v_all = torch.arange(n_matrices, device=device)
    g_all = torch.arange(ng, device=device)
    a_all = torch.arange(na, device=device)
    idx0 = ((v_all[:, None, None] * ng + g_all[None, :, None]) * nr + 0) * na + a_all[
        None, None, :
    ]
    cur = p_flat[idx0]
    p_flat[idx0] = torch.where(~multi[:, None, :], zero16, cur)
