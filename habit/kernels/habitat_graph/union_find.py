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
"""Union-find connected-component labeling on painted coordinates.

Used by ``uniform_grid`` node extraction to replace ``scipy.ndimage.label``
on each cube. The neighbourhood matches ``ndi.generate_binary_structure``:

* ``connectivity='full'`` -- 8-connected in 2-D / 26-connected in 3-D
* ``connectivity='face'`` -- 4-connected in 2-D / 6-connected in 3-D

Component *partitions* match ``ndi.label`` (same voxels together).
Component *ids* may differ.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "label_painted_components",
    "neighborhood_offsets",
]

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional accelerator
    njit = None
    _HAS_NUMBA = False


def neighborhood_offsets(ndim: int, connectivity: str) -> np.ndarray:
    """
    Integer neighbour offsets for one connectivity rule.

    Args:
        ndim: ``2`` or ``3``.
        connectivity: ``'face'`` or ``'full'``.

    Returns:
        np.ndarray: Shape ``(n_offsets, ndim)``, ``int32``, excluding
        the zero offset.

    Raises:
        ValueError: If ``ndim`` or ``connectivity`` is unsupported.
    """
    if int(ndim) not in (2, 3):
        raise ValueError("ndim must be 2 or 3.")
    if connectivity not in ("face", "full"):
        raise ValueError("connectivity must be 'face' or 'full'.")
    ndim_i = int(ndim)
    offsets = []
    if ndim_i == 2:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                if connectivity == "face" and (abs(dy) + abs(dx) != 1):
                    continue
                offsets.append((dy, dx))
    else:
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    if connectivity == "face" and (abs(dz) + abs(dy) + abs(dx) != 1):
                        continue
                    offsets.append((dz, dy, dx))
    return np.asarray(offsets, dtype=np.int32)


def label_painted_components(
    coords: np.ndarray,
    volume_shape: Tuple[int, ...],
    connectivity: str,
) -> Tuple[np.ndarray, int]:
    """
    Label connected components of painted local coordinates.

    Args:
        coords: Voxel coordinates, shape ``(n, ndim)``, integer, already
            clipped to ``volume_shape`` (as in a cube's local frame).
        volume_shape: Inclusive-exclusive local volume, length ``ndim``.
        connectivity: ``'face'`` or ``'full'``.

    Returns:
        ``(labels, n_components)``. ``labels`` is ``int32`` of length
        ``n`` with ids in ``1 .. n_components`` (``0`` unused). Empty
        input returns ``(empty, 0)``.
    """
    points = np.asarray(coords, dtype=np.int32)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.empty(0, dtype=np.int32), 0
    ndim = int(points.shape[1])
    shape = tuple(int(v) for v in volume_shape)
    if len(shape) != ndim:
        raise ValueError("volume_shape length must match coords.ndim.")
    offsets = neighborhood_offsets(ndim, connectivity)
    if _HAS_NUMBA and _label_numba is not None:
        labels, n_comp = _label_numba(
            np.ascontiguousarray(points, dtype=np.int32),
            np.asarray(shape, dtype=np.int32),
            np.ascontiguousarray(offsets, dtype=np.int32),
        )
        return labels, int(n_comp)
    return _label_python(points, shape, offsets)


def _label_python(
    coords: np.ndarray,
    shape: Tuple[int, ...],
    offsets: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Python union-find on painted coordinates."""
    n_vox = int(coords.shape[0])
    ndim = int(coords.shape[1])
    parent = np.arange(n_vox, dtype=np.int32)
    # Linear index -> slot. Background / unpainted stays -1.
    lookup_size = 1
    for axis in shape:
        lookup_size *= int(axis)
    lookup = np.full(lookup_size, -1, dtype=np.int32)
    strides = np.empty(ndim, dtype=np.int32)
    strides[ndim - 1] = 1
    for axis in range(ndim - 2, -1, -1):
        strides[axis] = strides[axis + 1] * int(shape[axis + 1])
    linear = np.zeros(n_vox, dtype=np.int32)
    for slot in range(n_vox):
        index = 0
        for axis in range(ndim):
            index += int(coords[slot, axis]) * int(strides[axis])
        linear[slot] = index
        lookup[index] = slot

    def _find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != root:
            nxt = int(parent[node])
            parent[node] = root
            node = nxt
        return root

    def _union(a: int, b: int) -> None:
        ra = _find(a)
        rb = _find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    n_off = int(offsets.shape[0])
    for slot in range(n_vox):
        for off in range(n_off):
            inside = True
            index = 0
            for axis in range(ndim):
                coord = int(coords[slot, axis]) + int(offsets[off, axis])
                if coord < 0 or coord >= int(shape[axis]):
                    inside = False
                    break
                index += coord * int(strides[axis])
            if not inside:
                continue
            other = int(lookup[index])
            if other >= 0:
                _union(slot, other)

    labels = np.empty(n_vox, dtype=np.int32)
    remap = np.full(n_vox, -1, dtype=np.int32)
    n_comp = 0
    for slot in range(n_vox):
        root = _find(slot)
        assigned = int(remap[root])
        if assigned < 0:
            n_comp += 1
            assigned = n_comp
            remap[root] = assigned
        labels[slot] = assigned
    return labels, n_comp


if _HAS_NUMBA:

    @njit(cache=True)
    def _label_numba(
        coords: np.ndarray,
        shape: np.ndarray,
        offsets: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Compiled union-find; same neighbourhood as the Python path."""
        n_vox = coords.shape[0]
        ndim = coords.shape[1]
        parent = np.empty(n_vox, dtype=np.int32)
        for slot in range(n_vox):
            parent[slot] = slot
        lookup_size = 1
        for axis in range(ndim):
            lookup_size *= shape[axis]
        lookup = np.empty(lookup_size, dtype=np.int32)
        for index in range(lookup_size):
            lookup[index] = -1
        strides = np.empty(ndim, dtype=np.int32)
        strides[ndim - 1] = 1
        for axis in range(ndim - 2, -1, -1):
            strides[axis] = strides[axis + 1] * shape[axis + 1]
        for slot in range(n_vox):
            index = 0
            for axis in range(ndim):
                index += coords[slot, axis] * strides[axis]
            lookup[index] = slot

        for slot in range(n_vox):
            node = slot
            while parent[node] != node:
                node = parent[node]
            root_a = node
            walk = slot
            while parent[walk] != root_a:
                nxt = parent[walk]
                parent[walk] = root_a
                walk = nxt
            n_off = offsets.shape[0]
            for off in range(n_off):
                inside = True
                index = 0
                for axis in range(ndim):
                    coord = coords[slot, axis] + offsets[off, axis]
                    if coord < 0 or coord >= shape[axis]:
                        inside = False
                        break
                    index += coord * strides[axis]
                if not inside:
                    continue
                other = lookup[index]
                if other < 0:
                    continue
                node_b = other
                while parent[node_b] != node_b:
                    node_b = parent[node_b]
                root_b = node_b
                walk_b = other
                while parent[walk_b] != root_b:
                    nxt_b = parent[walk_b]
                    parent[walk_b] = root_b
                    walk_b = nxt_b
                if root_a != root_b:
                    if root_a < root_b:
                        parent[root_b] = root_a
                    else:
                        parent[root_a] = root_b
                        root_a = root_b

        labels = np.empty(n_vox, dtype=np.int32)
        remap = np.empty(n_vox, dtype=np.int32)
        for slot in range(n_vox):
            remap[slot] = -1
        n_comp = 0
        for slot in range(n_vox):
            node = slot
            while parent[node] != node:
                node = parent[node]
            root = node
            walk = slot
            while parent[walk] != root:
                nxt = parent[walk]
                parent[walk] = root
                walk = nxt
            assigned = remap[root]
            if assigned < 0:
                n_comp += 1
                assigned = n_comp
                remap[root] = assigned
            labels[slot] = assigned
        return labels, n_comp

else:  # pragma: no cover - no numba
    _label_numba = None
