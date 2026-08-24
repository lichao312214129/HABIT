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
Shared geometry helpers for GPU texture-matrix builders.

Replicates the bounding-box / kernel-window / angle setup of
``radiomics/src/_cmatrices.c`` (``set_bb``, ``build_angles_arr``) so every
feature class (GLCM, GLDM, NGTDM, GLRLM, GLSZM) sees the same centres,
neighbours and angle order as the C extension.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .angles import build_angles

# Reuse uploaded image / mask / kernel-offset tensors across GLCM, GLRLM,
# GLSZM, GLDM and NGTDM in one voxel execute(). The scientific matrices
# stay identical; only the host-to-device copies are skipped. Evict when
# the table is full so a long cohort cannot pin many GPU buffers.
_CENTRE_GRID_CACHE: Dict[Tuple[object, ...], "CentreGrid"] = {}
# Enough for one voxel execute: ~15 batches x 2 angle sets (uni / bi).
_CENTRE_GRID_CACHE_MAX = 64


def _array_md5(array: np.ndarray) -> str:
    """Stable fingerprint of array bytes for the centre-grid cache."""
    contig = np.ascontiguousarray(array)
    return hashlib.md5(contig.tobytes()).hexdigest()


def element_strides(size: Sequence[int]) -> List[int]:
    """C-order element strides for a contiguous array of shape ``size``."""
    strides = [1] * len(size)
    for dim in range(len(size) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * int(size[dim + 1])
    return strides


def kernel_offsets(nd: int, radius: int, force2Ddimension: int) -> np.ndarray:
    """
    All centre offsets of the voxel kernel window.

    Returns the ``(2*radius+1)^Nd`` offsets of the window ``[-r, r]^Nd``;
    along ``force2Ddimension`` the window is collapsed to offset 0 only,
    matching ``set_bb`` in ``_cmatrices.c``.
    """
    axes = [
        range(-radius, radius + 1) if dim != force2Ddimension else range(0, 1)
        for dim in range(nd)
    ]
    return np.asarray(list(itertools.product(*axes)), dtype=np.int64)


@dataclass
class CentreGrid:
    """
    Device-side geometry shared by every texture-matrix builder.

    ``base_coords`` is one row per output matrix: the listed voxel (voxel
    mode) or every masked voxel (segment mode). Adding ``offsets_t[o]``
    yields a candidate centre; adding an angle to that centre yields a
    neighbour. A neighbour is inside the C bounding box iff it is inside
    the image and ``|neighbour - base| <= kernel_radius`` on every
    in-plane dimension (the image-clipped ``[-r, r]`` box of ``set_bb``).
    """

    device: torch.device
    size: List[int]
    nd: int
    size_t: torch.Tensor
    strides_t: torch.Tensor
    img_flat: torch.Tensor
    mask_flat: torch.Tensor
    n_elements: int
    angles_t: torch.Tensor
    angles_np: np.ndarray
    base_coords: torch.Tensor
    matrix_ids: torch.Tensor
    offsets_t: torch.Tensor
    n_matrices: int
    n_vox: int
    kernel_radius: int
    voxel_based: bool


def prepare_centre_grid(
    image: np.ndarray,
    mask: np.ndarray,
    distances: np.ndarray,
    force2D: bool,
    force2Ddimension: int,
    kernelRadius: int,
    voxelCoordinates: Optional[np.ndarray],
    device: Union[str, torch.device],
    bidirectional: bool,
) -> CentreGrid:
    """
    Build the shared centre / angle / stride tensors for one matrix call.

    Args:
        image: Discretised image (gray levels 1..Ng inside the mask).
        mask: Boolean array, same shape as ``image``.
        distances: Integer infinity-norm distances for angle generation.
        force2D: Restrict angles and the kernel window to a 2D plane.
        force2Ddimension: Out-of-plane dimension when ``force2D`` is set.
        kernelRadius: Voxel-kernel radius; ignored in segment mode.
        voxelCoordinates: ``(Nd, Nvox)`` listed voxels, or ``None`` for
            segment-based mode (one matrix over the whole ROI).
        device: Torch device that will hold the tensors.
        bidirectional: Keep both directions of every angle (GLDM / NGTDM
            / GLSZM); GLCM / GLRLM pass ``False``.

    Returns:
        CentreGrid: Device tensors ready for vectorised pair enumeration.
    """
    distances_t = tuple(int(value) for value in np.asarray(distances).ravel().tolist())
    # Do not key on id(voxelCoordinates): PyRadiomics passes a view of
    # labelledVoxelCoordinates per batch, and CPython can recycle that
    # view's id after the previous batch is freed. A recycled id would
    # return a grid with the wrong Nvox (e.g. 1000 vs the last 660).
    # Image / mask ids are safe: the feature class holds those arrays.
    if voxelCoordinates is None:
        coords_key: Tuple[object, ...] = ("segment",)
    else:
        coords_arr = np.ascontiguousarray(voxelCoordinates)
        coords_key = (
            tuple(int(size) for size in coords_arr.shape),
            _array_md5(coords_arr),
        )
    cache_key = (
        id(image),
        id(mask),
        coords_key,
        distances_t,
        bool(force2D),
        int(force2Ddimension),
        int(kernelRadius),
        str(torch.device(device)),
        bool(bidirectional),
    )
    cached = _CENTRE_GRID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    size = [int(s) for s in image.shape]
    nd = len(size)
    f2d_dim = int(force2Ddimension) if force2D else -1
    device_t = torch.device(device)

    angles_np = build_angles(
        size, distances, force2Ddimension=f2d_dim, bidirectional=bidirectional
    )
    size_t = torch.as_tensor(size, dtype=torch.long, device=device_t)
    strides_t = torch.as_tensor(element_strides(size), dtype=torch.long, device=device_t)
    angles_t = torch.as_tensor(angles_np, dtype=torch.long, device=device_t)

    # C PyArray_FORCECAST truncates toward zero when casting the (possibly
    # float) binned image to int; torch.long conversion does the same.
    img_flat = (
        torch.as_tensor(np.ascontiguousarray(image))
        .to(device=device_t, dtype=torch.long)
        .reshape(-1)
    )
    mask_flat = torch.as_tensor(
        np.ascontiguousarray(mask), dtype=torch.bool, device=device_t
    ).reshape(-1)

    if voxelCoordinates is None:
        base_coords = torch.nonzero(mask_flat.reshape(tuple(size)), as_tuple=False)
        matrix_ids = torch.zeros(base_coords.shape[0], dtype=torch.long, device=device_t)
        offsets_np = np.zeros((1, nd), dtype=np.int64)
        n_matrices = 1
        kernel_radius = 0
        voxel_based = False
    else:
        coords = torch.as_tensor(np.ascontiguousarray(voxelCoordinates)).to(
            device=device_t, dtype=torch.long
        )
        if coords.ndim != 2 or coords.shape[0] != nd:
            raise ValueError(
                f"voxelCoordinates must have shape (Nd, Nvox) = ({nd}, N); "
                f"got {tuple(coords.shape)}"
            )
        base_coords = coords.t().contiguous()
        matrix_ids = torch.arange(base_coords.shape[0], dtype=torch.long, device=device_t)
        if kernelRadius <= 0:
            raise ValueError(
                f"kernelRadius must be > 0 in voxel-based mode; got {kernelRadius}"
            )
        offsets_np = kernel_offsets(nd, int(kernelRadius), f2d_dim)
        n_matrices = int(base_coords.shape[0])
        kernel_radius = int(kernelRadius)
        voxel_based = True

    grid = CentreGrid(
        device=device_t,
        size=size,
        nd=nd,
        size_t=size_t,
        strides_t=strides_t,
        img_flat=img_flat,
        mask_flat=mask_flat,
        n_elements=int(mask_flat.shape[0]),
        angles_t=angles_t,
        angles_np=angles_np,
        base_coords=base_coords,
        matrix_ids=matrix_ids,
        offsets_t=torch.as_tensor(offsets_np, dtype=torch.long, device=device_t),
        n_matrices=n_matrices,
        n_vox=int(base_coords.shape[0]),
        kernel_radius=kernel_radius,
        voxel_based=voxel_based,
    )
    if len(_CENTRE_GRID_CACHE) >= _CENTRE_GRID_CACHE_MAX:
        _CENTRE_GRID_CACHE.clear()
    _CENTRE_GRID_CACHE[cache_key] = grid
    return grid


@dataclass
class WindowCells:
    """
    Per-(offset, listed voxel) tables for one voxel-based matrix call.

    Every voxel a texture builder touches is ``base + offset`` for some
    enumerated kernel offset, and -- this is the point -- so is every
    *neighbour*: a neighbour is inside the C bounding box exactly when
    ``offset + angle`` still satisfies ``|.| <= radius``, i.e. when it is
    itself an enumerated offset. So the whole ``(Na, No, Nvox, Nd)``
    neighbour-coordinate tensor (hundreds of MB of int64 at radius 3) is
    redundant: gray level and validity of a neighbour are just row
    ``neigh_off[angle, offset]`` of the same ``(No, Nvox)`` tables the
    centres use.

    Attributes:
        cell_flat: ``(No, Nvox)`` clamped flat index of ``base + offset``.
        cell_valid: ``(No, Nvox)`` in-image and masked.
        cell_gray: ``(No, Nvox)`` gray level; don't-care where invalid.
        neigh_off: ``(Na, No)`` offset index of ``offset + angle``, ``-1``
            when that lands outside the kernel window.
    """

    cell_flat: torch.Tensor
    cell_valid: torch.Tensor
    cell_gray: torch.Tensor
    neigh_off: torch.Tensor


def window_cells(grid: CentreGrid) -> WindowCells:
    """
    Build the shared centre / neighbour lookup tables for a voxel-based grid.

    Args:
        grid: Voxel-based :class:`CentreGrid` (``grid.voxel_based`` true).

    Returns:
        WindowCells: Tables described on that class.

    Raises:
        ValueError: If ``grid`` is segment-based, where a neighbour is not
            constrained to the offset window and the mapping does not hold.
    """
    if not grid.voxel_based:
        raise ValueError("window_cells requires a voxel-based CentreGrid")

    centres = grid.base_coords[None, :, :] + grid.offsets_t[:, None, :]
    cell_in = coords_in_image(centres, grid.size_t)
    cell_flat = flat_index(centres, grid.strides_t, grid.n_elements)
    cell_valid = cell_in & grid.mask_flat[cell_flat]
    # int32 halves the traffic of the (Na, No, Nvox) neighbour gather; gray
    # levels are bounded by Ng.
    cell_gray = grid.img_flat[cell_flat].to(torch.int32)

    radius = int(grid.kernel_radius)
    extent = 2 * radius + 1
    n_local = extent ** grid.nd
    pack_strides = torch.as_tensor(
        element_strides([extent] * grid.nd), dtype=torch.long, device=grid.device
    )
    lookup = torch.full((n_local,), -1, dtype=torch.long, device=grid.device)
    packed_own = ((grid.offsets_t + radius) * pack_strides).sum(dim=-1)
    lookup[packed_own] = torch.arange(
        grid.offsets_t.shape[0], dtype=torch.long, device=grid.device
    )

    local_n = grid.offsets_t[None, :, :] + grid.angles_t[:, None, :]  # (Na, No, Nd)
    in_window = (local_n.abs() <= radius).all(dim=-1)
    packed_n = ((local_n + radius) * pack_strides).sum(dim=-1)
    packed_ok = in_window & (packed_n >= 0) & (packed_n < n_local)
    neigh_off = torch.where(
        packed_ok,
        lookup[packed_n.clamp(0, n_local - 1)],
        torch.full_like(packed_n, -1),
    )
    return WindowCells(
        cell_flat=cell_flat,
        cell_valid=cell_valid,
        cell_gray=cell_gray,
        neigh_off=neigh_off,
    )


def coords_in_image(coords: torch.Tensor, size_t: torch.Tensor) -> torch.Tensor:
    """Boolean mask: every dimension of ``coords`` lies in ``[0, size)``."""
    return (coords >= 0).all(dim=-1) & (coords < size_t).all(dim=-1)


def flat_index(coords: torch.Tensor, strides_t: torch.Tensor, n_elements: int) -> torch.Tensor:
    """
    C-order flat index of ``coords``, clamped to ``[0, n_elements)``.

    Out-of-image coordinates are don't-care: callers must AND the gathered
    values with a validity mask. Clamping avoids a device-side OOB gather.
    """
    return (coords * strides_t).sum(dim=-1).clamp_(0, n_elements - 1)
