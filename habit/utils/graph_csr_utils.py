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
"""CSR graph primitives used by habitat-graph metrics.

Compressed Sparse Row stores an undirected simple graph as two integer
arrays: ``indptr`` (row starts, length ``n+1``) and ``indices``
(neighbours). Habitat-graph hop / clustering / component / assortativity
features run on this layout so the extract path never builds a NetworkX
object for the default (non-extended) columns.

Clustering matches NetworkX ``average_clustering`` (zeros for degree
< 2). Assortativity matches the Newman mixing-matrix definition.
Louvain is a CSR Blondel sweep; the partition can differ from
NetworkX ``seed=0``, so the modularity scalar may move slightly.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from habit.utils.graph_brandes_utils import csr_from_edge_arrays

__all__ = [
    "connected_components_csr",
    "induce_csr",
    "degrees_csr",
    "average_clustering_csr",
    "degree_assortativity_csr",
    "attribute_assortativity_csr",
    "louvain_modularity_csr",
]

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional accelerator
    njit = None
    _HAS_NUMBA = False


def degrees_csr(indptr: np.ndarray) -> np.ndarray:
    """Return integer degrees from a CSR row pointer.

    Args:
        indptr: CSR row pointer, length ``n_nodes + 1``.

    Returns:
        np.ndarray: Degree of each node, ``int64``.
    """
    pointer = np.asarray(indptr, dtype=np.int64)
    return np.diff(pointer)


def connected_components_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_nodes: int,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Label connected components of an undirected CSR graph.

    Isolated vertices are size-1 components, matching NetworkX.

    Args:
        indptr: CSR row pointer.
        indices: CSR neighbours.
        n_nodes: Vertex count.

    Returns:
        ``(labels, n_components, sizes)``. ``labels[i]`` is the component
        id of node ``i``; ``sizes[c]`` is that component's node count.
    """
    n_nodes = int(n_nodes)
    labels = np.full(n_nodes, -1, dtype=np.int32)
    if n_nodes == 0:
        return labels, 0, np.empty(0, dtype=np.int64)
    if _HAS_NUMBA and _components_numba is not None:
        labels, n_comp, sizes = _components_numba(
            np.asarray(indptr, dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
            n_nodes,
        )
        return labels, int(n_comp), sizes[: int(n_comp)]
    return _components_python(indptr, indices, n_nodes)


def _components_python(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_nodes: int,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Python BFS connected components."""
    labels = np.full(n_nodes, -1, dtype=np.int32)
    sizes = np.zeros(n_nodes, dtype=np.int64)
    n_comp = 0
    for start in range(n_nodes):
        if labels[start] >= 0:
            continue
        labels[start] = n_comp
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            row0 = int(indptr[node])
            row1 = int(indptr[node + 1])
            for slot in range(row0, row1):
                neighbour = int(indices[slot])
                if labels[neighbour] < 0:
                    labels[neighbour] = n_comp
                    stack.append(neighbour)
        sizes[n_comp] = size
        n_comp += 1
    return labels, n_comp, sizes[:n_comp]


def induce_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    keep: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Induce a subgraph on the nodes where ``keep`` is true.

    Args:
        indptr: Parent CSR row pointer.
        indices: Parent CSR neighbours.
        keep: Boolean mask, length ``n_nodes``.

    Returns:
        ``(sub_indptr, sub_indices, old_ids)``. ``old_ids[k]`` is the
        parent index of subgraph node ``k``.
    """
    keep_b = np.asarray(keep, dtype=bool)
    old_ids = np.flatnonzero(keep_b).astype(np.int64, copy=False)
    n_sub = int(old_ids.size)
    if n_sub == 0:
        return np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int64), old_ids
    remap = np.full(keep_b.shape[0], -1, dtype=np.int64)
    remap[old_ids] = np.arange(n_sub, dtype=np.int64)
    n_parent = int(indptr.shape[0] - 1)
    src = np.repeat(np.arange(n_parent, dtype=np.int64), np.diff(indptr))
    dst = np.asarray(indices, dtype=np.int64)
    kept_edge = keep_b[src] & keep_b[dst] & (src < dst)
    sub_src = remap[src[kept_edge]]
    sub_dst = remap[dst[kept_edge]]
    sub_indptr, sub_indices = csr_from_edge_arrays(n_sub, sub_src, sub_dst)
    return sub_indptr, sub_indices, old_ids


def average_clustering_csr(indptr: np.ndarray, indices: np.ndarray) -> float:
    """Mean local clustering (NetworkX ``average_clustering``).

    Args:
        indptr: CSR row pointer.
        indices: CSR neighbours (each row should be unique neighbours).

    Returns:
        float: Mean of per-node clustering; 0 when the graph is empty.
    """
    n_nodes = int(indptr.shape[0] - 1)
    if n_nodes <= 0:
        return 0.0
    if _HAS_NUMBA and _clustering_numba is not None:
        value = float(
            _clustering_numba(
                np.asarray(indptr, dtype=np.int64),
                np.asarray(indices, dtype=np.int64),
                n_nodes,
            )
        )
        return value if np.isfinite(value) else 0.0
    return _clustering_python(indptr, indices, n_nodes)


def _clustering_python(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_nodes: int,
) -> float:
    """Python triangle count; each node with degree < 2 contributes 0."""
    total = 0.0
    for node in range(n_nodes):
        start = int(indptr[node])
        stop = int(indptr[node + 1])
        degree = stop - start
        if degree < 2:
            continue
        neighbours = np.asarray(indices[start:stop], dtype=np.int64)
        neighbour_set = set(int(v) for v in neighbours.tolist())
        triangles = 0
        for slot_a in range(degree):
            peer_a = int(neighbours[slot_a])
            row0 = int(indptr[peer_a])
            row1 = int(indptr[peer_a + 1])
            for slot_b in range(row0, row1):
                peer_b = int(indices[slot_b])
                if peer_b in neighbour_set and peer_a < peer_b:
                    triangles += 1
        total += 2.0 * float(triangles) / float(degree * (degree - 1))
    return float(total / float(n_nodes))


def degree_assortativity_csr(indptr: np.ndarray, indices: np.ndarray) -> float:
    """Newman degree assortativity of an undirected CSR graph.

    Args:
        indptr: CSR row pointer.
        indices: CSR neighbours.

    Returns:
        float: Assortativity in ``[-1, 1]``, or 0 when undefined.
    """
    degrees = degrees_csr(indptr).astype(np.float64, copy=False)
    n_nodes = int(degrees.size)
    if n_nodes == 0 or int(indptr[-1]) == 0:
        return 0.0
    src = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr))
    dst = np.asarray(indices, dtype=np.int64)
    undirected = src < dst
    src = src[undirected]
    dst = dst[undirected]
    if src.size == 0:
        return 0.0
    deg_u = degrees[src]
    deg_v = degrees[dst]
    return _newman_numeric_assortativity(deg_u, deg_v)


def attribute_assortativity_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    attributes: np.ndarray,
) -> float:
    """Newman attribute assortativity (discrete labels).

    Args:
        indptr: CSR row pointer.
        indices: CSR neighbours.
        attributes: One discrete label per node.

    Returns:
        float: Assortativity, or 0 when undefined.
    """
    n_nodes = int(indptr.shape[0] - 1)
    if n_nodes == 0 or int(indptr[-1]) == 0:
        return 0.0
    labels = np.asarray(attributes)
    src = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr))
    dst = np.asarray(indices, dtype=np.int64)
    undirected = src < dst
    src = src[undirected]
    dst = dst[undirected]
    if src.size == 0:
        return 0.0
    left = labels[src]
    right = labels[dst]
    unique = np.unique(np.concatenate((left, right)))
    index = {value: slot for slot, value in enumerate(unique.tolist())}
    n_types = int(unique.size)
    mix = np.zeros((n_types, n_types), dtype=np.float64)
    for value_a, value_b in zip(left.tolist(), right.tolist()):
        slot_a = index[value_a]
        slot_b = index[value_b]
        mix[slot_a, slot_b] += 1.0
        mix[slot_b, slot_a] += 1.0
    total = float(mix.sum())
    if total <= 0.0:
        return 0.0
    mix /= total
    trace = float(np.trace(mix))
    expected = float(np.sum(mix.sum(axis=0) ** 2))
    denom = 1.0 - expected
    if abs(denom) < 1e-15:
        return 0.0
    value = (trace - expected) / denom
    return float(value) if np.isfinite(value) else 0.0


def _newman_numeric_assortativity(values_u: np.ndarray, values_v: np.ndarray) -> float:
    """Newman r for one numeric attribute observed on undirected edges."""
    n_edges = float(values_u.size)
    if n_edges <= 0.0:
        return 0.0
    sum_u = float(values_u.sum() + values_v.sum())
    sum_sq = float(np.square(values_u).sum() + np.square(values_v).sum())
    sum_prod = float((values_u * values_v).sum())
    numerator = (sum_prod / n_edges) - (sum_u / (2.0 * n_edges)) ** 2
    denominator = (sum_sq / (2.0 * n_edges)) - (sum_u / (2.0 * n_edges)) ** 2
    if abs(denominator) < 1e-15:
        return 0.0
    value = numerator / denominator
    return float(value) if np.isfinite(value) else 0.0


def louvain_modularity_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
) -> float:
    """
    Louvain modularity on an undirected CSR graph.

    Uses a CSR Blondel sweep (not NetworkX ``seed=0``). The partition
    can therefore differ slightly from NetworkX.

    Args:
        indptr: CSR row pointer.
        indices: CSR neighbours (both directions).
        weights: Per-directed-slot weights aligned with ``indices``.
            For an unweighted graph pass ones.
        n_nodes: Vertex count.

    Returns:
        float: Modularity of the found partition, or 0 when undefined.
    """
    n_nodes = int(n_nodes)
    if n_nodes <= 0 or int(indptr[-1]) == 0:
        return 0.0
    src = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr))
    dst = np.asarray(indices, dtype=np.int64)
    undirected = src < dst
    src_u = src[undirected]
    dst_u = dst[undirected]
    if src_u.size == 0:
        return 0.0
    weight_u = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weight_u.size == int(indptr[-1]):
        weight_u = weight_u[undirected]
    elif weight_u.size != src_u.size:
        weight_u = np.ones(src_u.size, dtype=np.float64)
    communities = _louvain_partition_python(indptr, indices, n_nodes)
    return _modularity_of_partition(src_u, dst_u, weight_u, communities, n_nodes)


def _louvain_partition_python(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """One-level greedy community assignment (Blondel local moves)."""
    community = np.arange(n_nodes, dtype=np.int32)
    degrees = degrees_csr(indptr).astype(np.float64, copy=False)
    two_m = float(degrees.sum())
    if two_m <= 0.0:
        return community
    strength = degrees.copy()
    moved = True
    n_pass = 0
    while moved and n_pass < n_nodes:
        moved = False
        n_pass += 1
        for node in range(n_nodes):
            current = int(community[node])
            best = current
            best_gain = 0.0
            start = int(indptr[node])
            stop = int(indptr[node + 1])
            seen: dict = {}
            for slot in range(start, stop):
                neighbour = int(indices[slot])
                comm = int(community[neighbour])
                seen[comm] = seen.get(comm, 0.0) + 1.0
            k_i = float(stop - start)
            for comm, k_i_in in seen.items():
                tot = float(strength[comm])
                if comm == current:
                    tot -= k_i
                gain = k_i_in - (tot * k_i) / two_m
                if comm == current:
                    continue
                if gain > best_gain + 1e-15:
                    best_gain = gain
                    best = comm
            if best != current and best_gain > 0.0:
                community[node] = best
                strength[current] -= k_i
                strength[best] += k_i
                moved = True
    return community


def _modularity_of_partition(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    community: np.ndarray,
    n_nodes: int,
) -> float:
    """Newman modularity of one partition on an undirected edge list."""
    two_m = 2.0 * float(weights.sum())
    if two_m <= 0.0:
        return 0.0
    degrees = np.zeros(n_nodes, dtype=np.float64)
    for node_a, node_b, weight in zip(src.tolist(), dst.tolist(), weights.tolist()):
        degrees[int(node_a)] += float(weight)
        degrees[int(node_b)] += float(weight)
    quality = 0.0
    for node_a, node_b, weight in zip(src.tolist(), dst.tolist(), weights.tolist()):
        if int(community[int(node_a)]) != int(community[int(node_b)]):
            continue
        quality += float(weight) - (degrees[int(node_a)] * degrees[int(node_b)]) / two_m
    # Each undirected edge counted once; Newman Q uses 1/(2m) * sum_ij.
    value = 2.0 * quality / two_m
    return float(value) if np.isfinite(value) else 0.0


if _HAS_NUMBA:

    @njit(cache=True)
    def _components_numba(
        indptr: np.ndarray,
        indices: np.ndarray,
        n_nodes: int,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Compiled BFS connected components."""
        labels = np.empty(n_nodes, dtype=np.int32)
        sizes = np.zeros(n_nodes, dtype=np.int64)
        for node in range(n_nodes):
            labels[node] = -1
        n_comp = 0
        stack = np.empty(n_nodes, dtype=np.int32)
        for start in range(n_nodes):
            if labels[start] >= 0:
                continue
            labels[start] = n_comp
            stack[0] = start
            depth = 1
            size = 0
            while depth > 0:
                depth -= 1
                node = stack[depth]
                size += 1
                row0 = indptr[node]
                row1 = indptr[node + 1]
                for slot in range(row0, row1):
                    neighbour = indices[slot]
                    if labels[neighbour] < 0:
                        labels[neighbour] = n_comp
                        stack[depth] = neighbour
                        depth += 1
            sizes[n_comp] = size
            n_comp += 1
        return labels, n_comp, sizes

    @njit(cache=True)
    def _clustering_numba(
        indptr: np.ndarray,
        indices: np.ndarray,
        n_nodes: int,
    ) -> float:
        """Compiled mean local clustering with a neighbour mark array."""
        mark = np.zeros(n_nodes, dtype=np.int32)
        stamp = 1
        total = 0.0
        for node in range(n_nodes):
            start = indptr[node]
            stop = indptr[node + 1]
            degree = stop - start
            if degree < 2:
                continue
            if stamp == 2147483647:
                for reset in range(n_nodes):
                    mark[reset] = 0
                stamp = 1
            for slot in range(start, stop):
                mark[indices[slot]] = stamp
            triangles = 0
            for slot in range(start, stop):
                peer = indices[slot]
                row0 = indptr[peer]
                row1 = indptr[peer + 1]
                for other in range(row0, row1):
                    neighbour = indices[other]
                    if mark[neighbour] == stamp:
                        triangles += 1
            # Each triangle is counted twice (once from each endpoint in N(v)).
            total += float(triangles) / float(degree * (degree - 1))
            stamp += 1
        return total / float(n_nodes)

else:  # pragma: no cover - no numba
    _components_numba = None
    _clustering_numba = None
