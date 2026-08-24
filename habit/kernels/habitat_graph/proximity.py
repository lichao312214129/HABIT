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
"""Exact proximity-graph construction for ``min_distance`` edges.

Default ``block_size=8`` / ``distance_threshold=5`` is the common case, but
both are user-set. The search radius on the dual lattice is derived from
the cube-separation lower bound, not hard-coded to 26-neighbours.

A pair of axis-aligned cubes of edge ``B`` whose lattice indices differ by
``?`` (integer vector) has closest-voxel distance at least the Euclidean
norm of the per-axis gaps ``g_a = 0`` if ``?_a = 0`` else
``(?_a - 1) B + 1``. An edge with threshold ``T`` is therefore impossible
unless that lower bound is ``<= T``. The cheapest (single-axis) case gives
the Chebyshev radius ``R = 0`` when ``T < 1``, else
``R = floor((T - 1) / B) + 1``.

Nodes that span several cubes (residual / component leftovers) are inserted
into every overlapped cell. When ``R`` is so large that a lattice walk
would visit more cells than a centroid range query, the implementation
falls back to a kd-tree on centroids with radius
``T + extent_i + extent_j`` (still a complete candidate set).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from habit.kernels.habitat_graph.models import (
    HabitatGraphNode,
    HabitatNodeExtractionResult,
)

__all__ = [
    "lattice_chebyshev_radius",
    "cube_separation_lower_bound",
    "collect_coords_by_node_id",
    "candidate_node_pairs",
    "min_distances_for_pairs",
    "uses_uniform_grid",
    "volume_sweep_worthwhile",
    "owner_volume",
    "volume_sweep_min_distances",
]

# Volume-neighbour sweep is exact and cheaper than node-pair clouds while
# (2*ceil(T)+1)**ndim * n_voxels stays in this budget.
_SWEEP_MAX_OPS: float = 4.0e8

# Lattice walk is cheaper only while the offset window stays modest.
# ``9**ndim`` is Chebyshev radius 4 in 3D (729 cells) -- beyond that a
# centroid ball query produces fewer, tighter candidates.
_MAX_LATTICE_WINDOW: int = 9 ** 3
# Brute-force set-separation is faster than a kd-tree only for tiny clouds.
# Larger pairs use a kd-tree with ``distance_upper_bound=T`` so non-edges
# exit without a full Cartesian product.
_BRUTE_PRODUCT: int = 256

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional accelerator
    njit = None
    prange = None
    _HAS_NUMBA = False


def lattice_chebyshev_radius(block_size: int, threshold: float) -> int:
    """
    Maximum lattice Chebyshev distance that can still host an edge.

    Args:
        block_size: Cube edge ``B`` in voxels (``>= 1``).
        threshold: Closest-voxel threshold ``T`` (``>= 0``).

    Returns:
        int: ``R`` such that only pairs with ``||?block||_? <= R`` can
        have ``d_min <= T``. ``0`` means same cell only.

    Raises:
        ValueError: If ``block_size < 1`` or ``threshold < 0``.
    """
    if int(block_size) < 1:
        raise ValueError("block_size must be >= 1.")
    if float(threshold) < 0.0:
        raise ValueError("distance_threshold must be >= 0.")
    if float(threshold) < 1.0:
        return 0
    return int((float(threshold) - 1.0) // int(block_size)) + 1


def cube_separation_lower_bound(
    delta_blocks: Sequence[int],
    block_size: int,
) -> float:
    """
    Euclidean lower bound on closest-voxel distance between two cubes.

    Args:
        delta_blocks: Absolute lattice-index difference per axis.
        block_size: Cube edge in voxels.

    Returns:
        float: ``||g||_2`` with ``g_a = 0`` if ``?_a = 0`` else
        ``(?_a - 1) B + 1``.
    """
    gap_sq = 0.0
    size = int(block_size)
    for raw in delta_blocks:
        delta = abs(int(raw))
        if delta <= 0:
            continue
        gap = float((delta - 1) * size + 1)
        gap_sq += gap * gap
    return float(gap_sq ** 0.5)


def uses_uniform_grid(node_result: HabitatNodeExtractionResult) -> bool:
    """
    Return True when nodes sit on the ``uniform_grid`` lattice.

    Voxel-neighbour sweep and dual-lattice range search are only valid
    accelerations for that tessellation. ``component`` nodes have no
    ``grid_origin`` / ``grid_block_size`` and keep the all-pairs
    closest-voxel walk.

    Args:
        node_result: Node extraction result.

    Returns:
        bool: Lattice metadata is present and ``block_size >= 1``.
    """
    origin = node_result.grid_origin
    block_size = node_result.grid_block_size
    return origin is not None and block_size is not None and int(block_size) >= 1


def volume_sweep_worthwhile(
    n_voxels: int,
    threshold: float,
    ndim: int,
) -> bool:
    """
    Whether a cropped-volume neighbour sweep is cheaper than node-pair clouds.

    Args:
        n_voxels: Non-empty voxels in the VOI crop.
        threshold: Closest-voxel threshold ``T``.
        ndim: 2 or 3.

    Returns:
        bool: ``True`` when the Chebyshev window times ``n_voxels`` is
        below ``_SWEEP_MAX_OPS``.
    """
    if float(threshold) < 0.0:
        return False
    radius = int(np.ceil(float(threshold)))
    window = (2 * radius + 1) ** int(ndim)
    return float(window) * float(max(int(n_voxels), 1)) <= _SWEEP_MAX_OPS


def owner_volume(
    node_result: HabitatNodeExtractionResult,
    nodes: Sequence[HabitatGraphNode],
) -> np.ndarray:
    """
    Paint node indices onto the cropped component lattice.

    Habitats partition voxels, so maps are merged without conflict.
    ``-1`` is background / unassigned.

    Args:
        node_result: Extraction whose ``component_maps`` match the crop.
        nodes: Node list; painted values are indices into this sequence.

    Returns:
        np.ndarray: ``int32`` volume, same shape as ``label_array``.
    """
    shape = tuple(int(v) for v in node_result.label_array.shape)
    owner = np.full(shape, -1, dtype=np.int32)
    index_of: Dict[Tuple[int, int], int] = {
        (int(node.habitat_label), int(node.component_id)): slot
        for slot, node in enumerate(nodes)
    }
    by_habitat: Dict[int, List[HabitatGraphNode]] = {}
    for node in nodes:
        by_habitat.setdefault(int(node.habitat_label), []).append(node)
    for habitat_label, group in by_habitat.items():
        component_map = node_result.component_maps.get(habitat_label)
        if component_map is None:
            continue
        max_id = int(component_map.max()) if component_map.size else 0
        if max_id <= 0:
            continue
        lookup = np.full(max_id + 1, -1, dtype=np.int32)
        for node in group:
            key = (int(node.habitat_label), int(node.component_id))
            slot = index_of.get(key)
            if slot is None:
                continue
            cid = int(node.component_id)
            if 0 <= cid <= max_id:
                lookup[cid] = int(slot)
        painted = lookup[component_map]
        assigned = painted >= 0
        owner[assigned] = painted[assigned]
    return owner


def volume_sweep_min_distances(
    owner: np.ndarray,
    threshold: float,
    n_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact ``d_min`` for every node pair whose voxels lie within ``T``.

    Scans a Chebyshev window of radius ``ceil(T)`` around every painted
    voxel and keeps the Euclidean minimum. Translation-invariant, so the
    cropped lattice is enough.

    Args:
        owner: Node-index volume from :func:`owner_volume`.
        threshold: Closest-voxel threshold.
        n_nodes: Node count (matrix side).

    Returns:
        ``(index_a, index_b, distance)`` with ``index_a < index_b`` and
        ``distance <= T``.
    """
    if n_nodes < 2 or owner.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    radius = int(np.ceil(float(threshold)))
    mins = _sweep_min_matrix(
        np.asarray(owner, dtype=np.int32),
        int(radius),
        float(threshold),
        int(n_nodes),
    )
    rows, cols = np.nonzero(np.isfinite(mins) & (mins <= float(threshold)))
    keep = rows < cols
    rows = rows[keep]
    cols = cols[keep]
    return (
        rows.astype(np.int64, copy=False),
        cols.astype(np.int64, copy=False),
        mins[rows, cols].astype(np.float64, copy=False),
    )


def collect_coords_by_node_id(
    node_result: HabitatNodeExtractionResult,
    nodes: Sequence[HabitatGraphNode],
) -> Dict[str, np.ndarray]:
    """
    Gather voxel-index clouds for ``nodes`` with one pass per habitat map.

    ``component_maps`` live on the cropped lattice; ``crop_offset`` is added
    so coordinates match stored centroids / bboxes (original index space).

    Args:
        node_result: Node extraction with per-habitat component maps.
        nodes: Nodes whose clouds should be collected.

    Returns:
        Dict[str, np.ndarray]: ``node_id -> (n_voxels, ndim)`` float array.
        Missing or empty components yield an empty array.
    """
    ndim = int(node_result.label_array.ndim)
    wanted: Dict[Tuple[int, int], str] = {
        (int(node.habitat_label), int(node.component_id)): node.node_id
        for node in nodes
    }
    coords: Dict[str, np.ndarray] = {
        node.node_id: np.empty((0, ndim), dtype=np.float64) for node in nodes
    }
    offset = (
        np.asarray(node_result.crop_offset, dtype=np.float64)
        if node_result.crop_offset is not None
        else None
    )
    for habitat_label, component_map in node_result.component_maps.items():
        hits = np.argwhere(component_map > 0)
        if hits.size == 0:
            continue
        component_ids = component_map[tuple(hits.T)].astype(np.int32, copy=False)
        points = hits.astype(np.float64, copy=False)
        if offset is not None:
            points = points + offset
        order = np.argsort(component_ids, kind="stable")
        component_ids = component_ids[order]
        points = points[order]
        breaks = np.flatnonzero(component_ids[1:] != component_ids[:-1]) + 1
        starts = np.concatenate((np.asarray([0], dtype=np.int64), breaks))
        stops = np.concatenate((breaks, np.asarray([component_ids.size], dtype=np.int64)))
        habitat = int(habitat_label)
        for start, stop in zip(starts.tolist(), stops.tolist()):
            node_id = wanted.get((habitat, int(component_ids[start])))
            if node_id is None:
                continue
            coords[node_id] = points[start:stop]
    return coords


def candidate_node_pairs(
    nodes: Sequence[HabitatGraphNode],
    node_result: HabitatNodeExtractionResult,
    distance_threshold: float,
    *,
    allow_same_label: bool = True,
    allow_cross_label: bool = True,
) -> List[Tuple[int, int]]:
    """
    Complete candidate index pairs that may have ``d_min <= threshold``.

    Uses the dual-lattice walk when ``grid_origin`` / ``grid_block_size``
    are present and the Chebyshev window is small; otherwise a centroid
    ball that is still complete (radius ``T + extent_i + extent_j``).

    Args:
        nodes: Node list; returned indices refer to this sequence.
        node_result: Extraction result (lattice metadata optional).
        distance_threshold: Closest-voxel threshold ``T``.
        allow_same_label: Include intra-habitat pairs.
        allow_cross_label: Include inter-habitat pairs.

    Returns:
        List[Tuple[int, int]]: Undirected pairs with ``i < j``.
    """
    n_nodes = len(nodes)
    if n_nodes < 2 or (not allow_same_label and not allow_cross_label):
        return []
    origin = node_result.grid_origin
    block_size = node_result.grid_block_size
    ndim = int(node_result.label_array.ndim)
    use_lattice = uses_uniform_grid(node_result)
    if use_lattice:
        radius = lattice_chebyshev_radius(int(block_size), float(distance_threshold))
        window = (2 * radius + 1) ** ndim
        if window <= _MAX_LATTICE_WINDOW:
            pairs = _lattice_candidate_pairs(
                nodes,
                np.asarray(origin, dtype=np.int64),
                int(block_size),
                radius,
                float(distance_threshold),
                ndim,
            )
            return _filter_label_pairs(
                nodes, pairs, allow_same_label, allow_cross_label
            )
    pairs = _centroid_candidate_pairs(nodes, float(distance_threshold))
    return _filter_label_pairs(nodes, pairs, allow_same_label, allow_cross_label)


def min_distances_for_pairs(
    nodes: Sequence[HabitatGraphNode],
    coords_by_id: Dict[str, np.ndarray],
    pairs: Sequence[Tuple[int, int]],
    distance_threshold: float,
) -> np.ndarray:
    """
    Closest-voxel distance for each candidate pair (``inf`` if ``> T``).

    Tiny clouds use a compiled brute-force kernel; large leftover /
    component clouds use a kd-tree. BBox separation is applied first.

    Args:
        nodes: Node list matching ``pairs`` indices.
        coords_by_id: Voxel clouds from :func:`collect_coords_by_node_id`.
        pairs: Candidate index pairs.
        distance_threshold: Keep only distances ``<= T``.

    Returns:
        np.ndarray: Shape ``(len(pairs),)``. ``inf`` means no edge.
    """
    n_pairs = len(pairs)
    out = np.full(n_pairs, np.inf, dtype=np.float64)
    if n_pairs == 0:
        return out
    ndim = int(nodes[0].centroid.size)
    clouds = [
        np.asarray(coords_by_id.get(node.node_id, np.empty((0, ndim))), dtype=np.float64)
        for node in nodes
    ]
    sizes = np.asarray([int(cloud.shape[0]) for cloud in clouds], dtype=np.int64)
    bbox_lo, bbox_hi = _bbox_arrays(nodes, ndim)
    threshold = float(distance_threshold)

    small_pos: List[int] = []
    small_i: List[int] = []
    small_j: List[int] = []
    large: List[Tuple[int, int, int]] = []
    for position, (index_a, index_b) in enumerate(pairs):
        if sizes[index_a] == 0 or sizes[index_b] == 0:
            continue
        gap = _bbox_gap_arrays(bbox_lo, bbox_hi, index_a, index_b, ndim)
        if gap > threshold:
            continue
        product = int(sizes[index_a]) * int(sizes[index_b])
        if product <= _BRUTE_PRODUCT:
            small_pos.append(position)
            small_i.append(index_a)
            small_j.append(index_b)
        else:
            large.append((position, index_a, index_b))

    if small_i:
        packed, indptr = _pack_clouds(clouds, ndim)
        distances = _brute_pair_distances(
            packed,
            indptr,
            np.asarray(small_i, dtype=np.int64),
            np.asarray(small_j, dtype=np.int64),
            int(ndim),
            float(threshold),
        )
        for slot, distance in zip(small_pos, distances.tolist()):
            if distance <= threshold:
                out[slot] = float(distance)

    trees: Dict[int, cKDTree] = {}

    def _tree(index: int) -> cKDTree:
        tree = trees.get(index)
        if tree is None:
            tree = cKDTree(clouds[index])
            trees[index] = tree
        return tree

    cap = float(threshold) + 1.0e-9
    for position, index_a, index_b in large:
        cloud_a = clouds[index_a]
        cloud_b = clouds[index_b]
        if cloud_a.shape[0] <= cloud_b.shape[0]:
            nearest = _tree(index_b).query(cloud_a, k=1, distance_upper_bound=cap)[0]
        else:
            nearest = _tree(index_a).query(cloud_b, k=1, distance_upper_bound=cap)[0]
        distance = float(np.min(nearest))
        if np.isfinite(distance) and distance <= threshold:
            out[position] = distance
    return out


def _filter_label_pairs(
    nodes: Sequence[HabitatGraphNode],
    pairs: Sequence[Tuple[int, int]],
    allow_same_label: bool,
    allow_cross_label: bool,
) -> List[Tuple[int, int]]:
    """Drop pairs that the caller does not want (intra / inter)."""
    kept: List[Tuple[int, int]] = []
    for index_a, index_b in pairs:
        same = int(nodes[index_a].habitat_label) == int(nodes[index_b].habitat_label)
        if same and not allow_same_label:
            continue
        if (not same) and not allow_cross_label:
            continue
        kept.append((index_a, index_b))
    return kept


def _lattice_offsets(
    radius: int,
    ndim: int,
    block_size: int,
    threshold: float,
) -> List[Tuple[int, ...]]:
    """Lattice offsets whose cube-separation lower bound is ``<= T``."""
    offsets: List[Tuple[int, ...]] = []
    for offset in product(range(-int(radius), int(radius) + 1), repeat=int(ndim)):
        if cube_separation_lower_bound(offset, block_size) <= threshold + 1.0e-12:
            offsets.append(tuple(int(v) for v in offset))
    return offsets


def _bbox_cells(
    bbox: Tuple[int, ...],
    origin: np.ndarray,
    block_size: int,
    ndim: int,
) -> List[Tuple[int, ...]]:
    """Lattice cells overlapped by a half-open voxel bbox."""
    low = np.asarray(bbox[:ndim], dtype=np.int64)
    high = np.asarray(bbox[ndim:], dtype=np.int64)
    last = np.maximum(high - 1, low)
    start = (low - origin) // int(block_size)
    stop = (last - origin) // int(block_size)
    ranges = [range(int(a), int(b) + 1) for a, b in zip(start.tolist(), stop.tolist())]
    return [tuple(int(v) for v in cell) for cell in product(*ranges)]


def _lattice_candidate_pairs(
    nodes: Sequence[HabitatGraphNode],
    origin: np.ndarray,
    block_size: int,
    radius: int,
    threshold: float,
    ndim: int,
) -> List[Tuple[int, int]]:
    """Hash nodes into overlapped cubes and emit neighbour-cell pairs."""
    cell_to_nodes: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    node_cells: List[List[Tuple[int, ...]]] = []
    for index, node in enumerate(nodes):
        cells = _bbox_cells(node.bbox, origin, block_size, ndim)
        if not cells:
            cells = [tuple(int(v) for v in ((np.asarray(node.centroid) - origin) // block_size))]
        node_cells.append(cells)
        for cell in cells:
            cell_to_nodes[cell].append(index)

    offsets = _lattice_offsets(radius, ndim, block_size, threshold)
    n_nodes = len(nodes)
    # Boolean upper triangle: cheaper than a Python set at a few thousand nodes.
    marked = np.zeros((n_nodes, n_nodes), dtype=np.bool_)
    for index, cells in enumerate(node_cells):
        for cell in cells:
            for offset in offsets:
                neighbour = tuple(int(cell[axis] + offset[axis]) for axis in range(ndim))
                for other in cell_to_nodes.get(neighbour, ()):
                    if index < other:
                        marked[index, other] = True
    rows, cols = np.nonzero(marked)
    return [(int(i), int(j)) for i, j in zip(rows.tolist(), cols.tolist())]


def _centroid_candidate_pairs(
    nodes: Sequence[HabitatGraphNode],
    threshold: float,
) -> List[Tuple[int, int]]:
    """Complete candidates via centroid balls of radius ``T + ext_i + ext_j``."""
    n_nodes = len(nodes)
    if n_nodes < 2:
        return []
    centroids = np.asarray([node.centroid for node in nodes], dtype=np.float64)
    extents = np.asarray([_bbox_half_extent(node) for node in nodes], dtype=np.float64)
    max_extent = float(extents.max()) if extents.size else 0.0
    tree = cKDTree(centroids)
    seen: Set[Tuple[int, int]] = set()
    for index, centroid in enumerate(centroids):
        radius = float(threshold) + float(extents[index]) + max_extent
        for other in tree.query_ball_point(centroid, r=radius):
            if index < other:
                seen.add((index, int(other)))
    return list(seen)


def _bbox_half_extent(node: HabitatGraphNode) -> float:
    """Half the Euclidean bbox diagonal (conservative cloud radius)."""
    ndim = len(node.bbox) // 2
    low = np.asarray(node.bbox[:ndim], dtype=np.float64)
    high = np.asarray(node.bbox[ndim:], dtype=np.float64)
    span = np.maximum(high - low, 0.0)
    return 0.5 * float(np.linalg.norm(span))


def _bbox_arrays(
    nodes: Sequence[HabitatGraphNode],
    ndim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack half-open bboxes as ``(n, ndim)`` low / high arrays."""
    low = np.zeros((len(nodes), ndim), dtype=np.float64)
    high = np.zeros((len(nodes), ndim), dtype=np.float64)
    for index, node in enumerate(nodes):
        low[index] = np.asarray(node.bbox[:ndim], dtype=np.float64)
        high[index] = np.asarray(node.bbox[ndim:], dtype=np.float64)
    return low, high


def _bbox_gap_arrays(
    low: np.ndarray,
    high: np.ndarray,
    index_a: int,
    index_b: int,
    ndim: int,
) -> float:
    """Closest-point gap between two half-open boxes (0 if they overlap)."""
    gap_sq = 0.0
    for axis in range(ndim):
        if high[index_a, axis] <= low[index_b, axis]:
            gap = float(low[index_b, axis] - high[index_a, axis] + 1.0)
        elif high[index_b, axis] <= low[index_a, axis]:
            gap = float(low[index_a, axis] - high[index_b, axis] + 1.0)
        else:
            gap = 0.0
        gap_sq += gap * gap
    return float(gap_sq ** 0.5)


def _pack_clouds(
    clouds: Sequence[np.ndarray],
    ndim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate clouds into ``(n_voxels, ndim)`` plus CSR-style ``indptr``."""
    counts = [int(cloud.shape[0]) for cloud in clouds]
    indptr = np.zeros(len(clouds) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
    packed = np.empty((int(indptr[-1]), ndim), dtype=np.float64)
    for index, cloud in enumerate(clouds):
        start = int(indptr[index])
        stop = int(indptr[index + 1])
        if stop > start:
            packed[start:stop] = cloud
    return packed, indptr


def _brute_pair_distances(
    packed: np.ndarray,
    indptr: np.ndarray,
    index_a: np.ndarray,
    index_b: np.ndarray,
    ndim: int,
    threshold: float,
) -> np.ndarray:
    """Set-separation for many small pairs (compiled when numba is present)."""
    if _HAS_NUMBA and _brute_pairs_numba is not None:
        return _brute_pairs_numba(
            packed, indptr, index_a, index_b, int(ndim), float(threshold)
        )
    cap_sq = float(threshold) * float(threshold)
    out = np.empty(index_a.shape[0], dtype=np.float64)
    for slot in range(index_a.shape[0]):
        start_a = int(indptr[index_a[slot]])
        stop_a = int(indptr[index_a[slot] + 1])
        start_b = int(indptr[index_b[slot]])
        stop_b = int(indptr[index_b[slot] + 1])
        best_sq = np.inf
        for ia in range(start_a, stop_a):
            for ib in range(start_b, stop_b):
                dist_sq = 0.0
                for axis in range(ndim):
                    delta = packed[ia, axis] - packed[ib, axis]
                    dist_sq += delta * delta
                if dist_sq < best_sq:
                    best_sq = dist_sq
                    if best_sq <= 1.0:
                        break
            if best_sq <= 1.0:
                break
        if best_sq <= cap_sq:
            out[slot] = float(best_sq ** 0.5)
        else:
            out[slot] = np.inf
    return out


if _HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _brute_pairs_numba(
        packed: np.ndarray,
        indptr: np.ndarray,
        index_a: np.ndarray,
        index_b: np.ndarray,
        ndim: int,
        threshold: float,
    ) -> np.ndarray:
        """Parallel closest-point distance over packed voxel clouds."""
        n_pairs = index_a.shape[0]
        out = np.empty(n_pairs, dtype=np.float64)
        cap_sq = threshold * threshold
        for slot in prange(n_pairs):
            start_a = indptr[index_a[slot]]
            stop_a = indptr[index_a[slot] + 1]
            start_b = indptr[index_b[slot]]
            stop_b = indptr[index_b[slot] + 1]
            best_sq = 1.0e300
            done = False
            for ia in range(start_a, stop_a):
                for ib in range(start_b, stop_b):
                    dist_sq = 0.0
                    for axis in range(ndim):
                        delta = packed[ia, axis] - packed[ib, axis]
                        dist_sq += delta * delta
                    if dist_sq < best_sq:
                        best_sq = dist_sq
                        if best_sq <= 1.0:
                            done = True
                            break
                if done:
                    break
            if best_sq <= cap_sq:
                out[slot] = best_sq ** 0.5
            else:
                out[slot] = np.inf
        return out

else:  # pragma: no cover - no numba
    _brute_pairs_numba = None


def _sweep_min_matrix(
    owner: np.ndarray,
    radius: int,
    threshold: float,
    n_nodes: int,
) -> np.ndarray:
    """Fill an ``n x n`` matrix of closest-voxel distances (``inf`` if none)."""
    painted = np.argwhere(np.asarray(owner) >= 0)
    if painted.size == 0:
        return np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    owner_i = np.asarray(owner, dtype=np.int32)
    if _HAS_NUMBA and _sweep_painted_2d_numba is not None:
        if owner.ndim == 2:
            return _sweep_painted_2d_numba(
                owner_i,
                np.ascontiguousarray(painted[:, 0], dtype=np.int32),
                np.ascontiguousarray(painted[:, 1], dtype=np.int32),
                int(radius),
                float(threshold),
                int(n_nodes),
            )
        return _sweep_painted_3d_numba(
            owner_i,
            np.ascontiguousarray(painted[:, 0], dtype=np.int32),
            np.ascontiguousarray(painted[:, 1], dtype=np.int32),
            np.ascontiguousarray(painted[:, 2], dtype=np.int32),
            int(radius),
            float(threshold),
            int(n_nodes),
        )
    return _sweep_painted_python(owner, painted, radius, threshold, n_nodes)


def _sweep_painted_python(
    owner: np.ndarray,
    painted: np.ndarray,
    radius: int,
    threshold: float,
    n_nodes: int,
) -> np.ndarray:
    """Sweep only painted voxels (empty space is skipped)."""
    mins = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    cap_sq = float(threshold) * float(threshold)
    if owner.ndim == 2:
        height, width = owner.shape
        for y, x in painted:
            node_a = int(owner[y, x])
            y0 = max(0, int(y) - radius)
            y1 = min(height, int(y) + radius + 1)
            x0 = max(0, int(x) - radius)
            x1 = min(width, int(x) + radius + 1)
            for yy in range(y0, y1):
                dy = float(yy - y)
                for xx in range(x0, x1):
                    node_b = int(owner[yy, xx])
                    if node_b < 0 or node_b == node_a:
                        continue
                    dist_sq = dy * dy + float(xx - x) ** 2
                    if dist_sq > cap_sq:
                        continue
                    lo, hi = (node_a, node_b) if node_a < node_b else (node_b, node_a)
                    dist = dist_sq ** 0.5
                    if dist < mins[lo, hi]:
                        mins[lo, hi] = dist
        return mins
    depth, height, width = owner.shape
    for z, y, x in painted:
        node_a = int(owner[z, y, x])
        z0 = max(0, int(z) - radius)
        z1 = min(depth, int(z) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(height, int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(width, int(x) + radius + 1)
        for zz in range(z0, z1):
            dz = float(zz - z)
            for yy in range(y0, y1):
                dy = float(yy - y)
                for xx in range(x0, x1):
                    node_b = int(owner[zz, yy, xx])
                    if node_b < 0 or node_b == node_a:
                        continue
                    dist_sq = dz * dz + dy * dy + float(xx - x) ** 2
                    if dist_sq > cap_sq:
                        continue
                    lo, hi = (node_a, node_b) if node_a < node_b else (node_b, node_a)
                    dist = dist_sq ** 0.5
                    if dist < mins[lo, hi]:
                        mins[lo, hi] = dist
    return mins


if _HAS_NUMBA:

    @njit(cache=True)
    def _sweep_painted_2d_numba(
        owner: np.ndarray,
        py: np.ndarray,
        px: np.ndarray,
        radius: int,
        threshold: float,
        n_nodes: int,
    ) -> np.ndarray:
        """Compiled 2-D sweep over painted voxels only."""
        mins = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
        cap_sq = threshold * threshold
        height = owner.shape[0]
        width = owner.shape[1]
        n_painted = py.shape[0]
        for slot in range(n_painted):
            y = py[slot]
            x = px[slot]
            node_a = owner[y, x]
            y0 = y - radius
            if y0 < 0:
                y0 = 0
            y1 = y + radius + 1
            if y1 > height:
                y1 = height
            x0 = x - radius
            if x0 < 0:
                x0 = 0
            x1 = x + radius + 1
            if x1 > width:
                x1 = width
            for yy in range(y0, y1):
                dy = float(yy - y)
                for xx in range(x0, x1):
                    node_b = owner[yy, xx]
                    if node_b < 0 or node_b == node_a:
                        continue
                    dist_sq = dy * dy + float(xx - x) * float(xx - x)
                    if dist_sq > cap_sq:
                        continue
                    if node_a < node_b:
                        lo = node_a
                        hi = node_b
                    else:
                        lo = node_b
                        hi = node_a
                    dist = dist_sq ** 0.5
                    if dist < mins[lo, hi]:
                        mins[lo, hi] = dist
        return mins

    @njit(cache=True)
    def _sweep_painted_3d_numba(
        owner: np.ndarray,
        pz: np.ndarray,
        py: np.ndarray,
        px: np.ndarray,
        radius: int,
        threshold: float,
        n_nodes: int,
    ) -> np.ndarray:
        """Compiled 3-D sweep over painted voxels only."""
        mins = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
        cap_sq = threshold * threshold
        depth = owner.shape[0]
        height = owner.shape[1]
        width = owner.shape[2]
        n_painted = pz.shape[0]
        for slot in range(n_painted):
            z = pz[slot]
            y = py[slot]
            x = px[slot]
            node_a = owner[z, y, x]
            z0 = z - radius
            if z0 < 0:
                z0 = 0
            z1 = z + radius + 1
            if z1 > depth:
                z1 = depth
            y0 = y - radius
            if y0 < 0:
                y0 = 0
            y1 = y + radius + 1
            if y1 > height:
                y1 = height
            x0 = x - radius
            if x0 < 0:
                x0 = 0
            x1 = x + radius + 1
            if x1 > width:
                x1 = width
            for zz in range(z0, z1):
                dz = float(zz - z)
                for yy in range(y0, y1):
                    dy = float(yy - y)
                    for xx in range(x0, x1):
                        node_b = owner[zz, yy, xx]
                        if node_b < 0 or node_b == node_a:
                            continue
                        dist_sq = dz * dz + dy * dy + float(xx - x) * float(xx - x)
                        if dist_sq > cap_sq:
                            continue
                        if node_a < node_b:
                            lo = node_a
                            hi = node_b
                        else:
                            lo = node_b
                            hi = node_a
                        dist = dist_sq ** 0.5
                        if dist < mins[lo, hi]:
                            mins[lo, hi] = dist
        return mins

else:  # pragma: no cover - no numba
    _sweep_painted_2d_numba = None
    _sweep_painted_3d_numba = None
