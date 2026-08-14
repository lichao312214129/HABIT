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
"""Edge-building strategies for habitat graph features."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from habit.kernels.habitat_graph.models import (
    EdgeWeightMode,
    HabitatGraph,
    HabitatGraphEdge,
    HabitatGraphNode,
    HabitatNodeExtractionResult,
)

__all__ = [
    "build_centroid_distance_graph",
    "build_adjacency_graph",
    "iter_label_pairs",
    "iter_cross_label_nodes",
]


def _nodes_to_dict(nodes: Iterable[HabitatGraphNode]) -> Dict[str, HabitatGraphNode]:
    """Return nodes keyed by stable node id."""
    return {node.node_id: node for node in nodes}


def _distance_weight(distance: float, edge_weight: EdgeWeightMode) -> float:
    """Convert a centroid distance to the requested edge weight."""
    if edge_weight == "distance":
        return float(distance)
    if edge_weight == "inverse_distance":
        return float(1.0 / (distance + 1e-6))
    return 1.0


def _contact_weight(contact_voxels: int, edge_weight: EdgeWeightMode) -> float:
    """Convert a contact voxel count to the requested edge weight."""
    if edge_weight == "contact_voxels":
        return float(contact_voxels)
    return 1.0


def _intra_label_edges(
    nodes: Sequence[HabitatGraphNode],
    distance_threshold: float,
    edge_weight: EdgeWeightMode,
) -> List[HabitatGraphEdge]:
    """
    Build same-label proximity edges within one node group.

    These ``"intra"`` edges encode each habitat's own spatial continuity. They
    are added to pairwise graphs so that whole-graph metrics (modularity,
    class-based assortativity, betweenness) reflect real tissue organization
    instead of a degenerate bipartite structure.

    Args:
        nodes: Nodes that all share the same habitat label.
        distance_threshold: Maximum Euclidean centroid distance for an edge.
        edge_weight: Optional distance-derived edge weighting mode.

    Returns:
        List[HabitatGraphEdge]: Intra-label edges tagged with ``edge_type='intra'``.
    """
    edges: List[HabitatGraphEdge] = []
    if len(nodes) < 2:
        return edges
    coords = np.asarray([node.centroid for node in nodes], dtype=float)
    tree = cKDTree(coords)
    for index_a, index_b in tree.query_pairs(r=distance_threshold):
        node_a = nodes[index_a]
        node_b = nodes[index_b]
        distance = float(np.linalg.norm(node_a.centroid - node_b.centroid))
        edges.append(
            HabitatGraphEdge(
                source=node_a.node_id,
                target=node_b.node_id,
                edge_type="intra",
                distance=distance,
                contact_voxels=None,
                weight=_distance_weight(distance, edge_weight),
            )
        )
    return edges


def build_centroid_distance_graph(
    nodes: Sequence[HabitatGraphNode],
    labels: Tuple[int, ...],
    graph_kind: str,
    distance_threshold: float,
    edge_weight: EdgeWeightMode = "none",
    include_intra_edges: bool = False,
) -> HabitatGraph:
    """
    Build a graph by connecting nodes whose centroid distance is within threshold.

    Args:
        nodes: Nodes to include in the graph. For pairwise graphs this must
            contain nodes from both habitat labels.
        labels: One label for a single-habitat graph or two labels for a pair.
        graph_kind: ``"single"`` or ``"pairwise"``.
        distance_threshold: Maximum Euclidean centroid distance in pixel units.
        edge_weight: Optional distance-derived edge weighting mode.
        include_intra_edges: For pairwise graphs, also add same-label proximity
            edges within each habitat. This matches the source PathPrism
            multi-tissue graph, where whole-graph metrics use both inter- and
            intra-class edges while interface metrics use inter-class edges only.

    Returns:
        HabitatGraph: Lightweight graph with all input nodes and inferred edges.
    """
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be >= 0.")
    if len(labels) not in (1, 2):
        raise ValueError("labels must contain one or two habitat labels.")

    graph_nodes = _nodes_to_dict(nodes)
    edges: List[HabitatGraphEdge] = []

    if len(nodes) < 2:
        return HabitatGraph(
            graph_kind=graph_kind,  # type: ignore[arg-type]
            labels=labels,
            nodes=graph_nodes,
            edges=edges,
        )

    if len(labels) == 1:
        coords = np.asarray([node.centroid for node in nodes], dtype=float)
        tree = cKDTree(coords)
        for index_a, index_b in tree.query_pairs(r=distance_threshold):
            node_a = nodes[index_a]
            node_b = nodes[index_b]
            distance = float(np.linalg.norm(node_a.centroid - node_b.centroid))
            edges.append(
                HabitatGraphEdge(
                    source=node_a.node_id,
                    target=node_b.node_id,
                    edge_type="centroid_distance",
                    distance=distance,
                    contact_voxels=None,
                    weight=_distance_weight(distance, edge_weight),
                )
            )
    else:
        label_a, label_b = labels
        nodes_a = [node for node in nodes if node.habitat_label == label_a]
        nodes_b = [node for node in nodes if node.habitat_label == label_b]
        if nodes_a and nodes_b:
            coords_b = np.asarray([node.centroid for node in nodes_b], dtype=float)
            tree_b = cKDTree(coords_b)
            for node_a in nodes_a:
                matches = tree_b.query_ball_point(node_a.centroid, r=distance_threshold)
                for index_b in matches:
                    node_b = nodes_b[index_b]
                    distance = float(np.linalg.norm(node_a.centroid - node_b.centroid))
                    edges.append(
                        HabitatGraphEdge(
                            source=node_a.node_id,
                            target=node_b.node_id,
                            edge_type="inter",
                            distance=distance,
                            contact_voxels=None,
                            weight=_distance_weight(distance, edge_weight),
                        )
                    )
        if include_intra_edges:
            edges.extend(
                _intra_label_edges(nodes_a, distance_threshold, edge_weight)
            )
            edges.extend(
                _intra_label_edges(nodes_b, distance_threshold, edge_weight)
            )

    return HabitatGraph(
        graph_kind=graph_kind,  # type: ignore[arg-type]
        labels=labels,
        nodes=graph_nodes,
        edges=edges,
    )


def _half_offsets(ndim: int, adjacency_connectivity: str) -> List[Tuple[int, ...]]:
    """
    Return the minimal set of offset vectors that covers all unique neighbor pairs
    without duplication (first non-zero component is always +1).

    Args:
        ndim: Number of array dimensions (2 or 3).
        adjacency_connectivity: ``"face"`` for axis-aligned neighbors only (6-conn
            in 3D / 4-conn in 2D), ``"edge"`` to additionally include edge-sharing
            neighbors (18-conn in 3D / 8-conn in 2D), or ``"corner"`` to include
            all diagonal neighbors (26-conn in 3D / 8-conn in 2D, same as ``"edge"``
            in 2D since there is no third dimension to distinguish edge from corner).

    Returns:
        List[Tuple[int, ...]]: Half-space offset vectors, one per unique direction.
    """
    n_nonzero_limit = {
        "face": 1,
        "edge": 2,
        "corner": ndim,
    }.get(adjacency_connectivity, ndim)

    offsets: List[Tuple[int, ...]] = []
    for bits in product(range(-1, 2), repeat=ndim):
        if all(b == 0 for b in bits):
            continue
        n_nonzero = sum(b != 0 for b in bits)
        if n_nonzero > n_nonzero_limit:
            continue
        # Keep only the "positive" representative: first non-zero must be +1.
        for val in bits:
            if val != 0:
                if val == 1:
                    offsets.append(bits)
                break
    return offsets


def _shifted_pair(
    arr: np.ndarray,
    offset: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return two aligned array views differing by one voxel step along ``offset``.

    For offset ``(+1, -1, 0)`` the first view removes the last row and first
    column while the second view removes the first row and last column, so that
    ``view_a[i, j, k]`` and ``view_b[i, j, k]`` are offset-neighbors in the
    original array.

    Args:
        arr: Source array.
        offset: Per-dimension offset values (each in {-1, 0, +1}).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (view_a, view_b) aligned neighbor views.
    """
    slices_a: List[slice] = []
    slices_b: List[slice] = []
    for d in offset:
        if d > 0:
            slices_a.append(slice(None, -d))
            slices_b.append(slice(d, None))
        elif d < 0:
            slices_a.append(slice(-d, None))
            slices_b.append(slice(None, d))
        else:
            slices_a.append(slice(None))
            slices_b.append(slice(None))
    return arr[tuple(slices_a)], arr[tuple(slices_b)]


def _build_node_id_map(
    node_result: HabitatNodeExtractionResult,
    labels: Sequence[int],
) -> Tuple[np.ndarray, Dict[int, "HabitatGraphNode"]]:
    """
    Assign a unique positive integer ID to every voxel that belongs to a node.

    Background voxels and voxels outside the requested labels receive ID 0.

    Args:
        node_result: Output from connected-region node extraction.
        labels: Habitat labels whose nodes should be included.

    Returns:
        Tuple[np.ndarray, Dict[int, HabitatGraphNode]]: ``(node_id_array,
        id_to_node)`` where ``id_to_node`` maps integer ID to the node object.
    """
    node_id_array = np.zeros(node_result.label_array.shape, dtype=np.int32)
    id_to_node: Dict[int, HabitatGraphNode] = {}
    next_id = 1
    for label in labels:
        comp_map = node_result.component_maps.get(int(label))
        nodes = node_result.nodes_by_habitat.get(int(label), [])
        if comp_map is None:
            continue
        for node in nodes:
            mask = comp_map == node.component_id
            node_id_array[mask] = next_id
            id_to_node[next_id] = node
            next_id += 1
    return node_id_array, id_to_node


def _count_adjacency_pairs(
    node_id_array: np.ndarray,
    offsets: List[Tuple[int, ...]],
) -> "Counter[Tuple[int, int]]":
    """
    Count the number of touching voxel pairs for every adjacent node pair.

    Iterates over half-space offsets and counts voxel-neighbor hits so each
    pair (A, B) is counted once regardless of direction.

    Args:
        node_id_array: Array whose non-zero entries carry a unique node integer ID.
        offsets: Half-space offset vectors (each covers one unique neighbor direction).

    Returns:
        Counter[Tuple[int, int]]: ``(id_a, id_b)`` → number of adjacent voxel pairs,
        where ``id_a < id_b`` by construction.
    """
    counts: Counter[Tuple[int, int]] = Counter()
    for offset in offsets:
        view_a, view_b = _shifted_pair(node_id_array, offset)
        mask = (view_a > 0) & (view_b > 0) & (view_a != view_b)
        if not np.any(mask):
            continue
        ids_a = view_a[mask]
        ids_b = view_b[mask]
        # Enforce canonical ordering so (A, B) == (B, A).
        lo = np.minimum(ids_a, ids_b)
        hi = np.maximum(ids_a, ids_b)
        for a_int, b_int in zip(lo.tolist(), hi.tolist()):
            counts[(int(a_int), int(b_int))] += 1
    return counts


def build_adjacency_graph(
    node_result: HabitatNodeExtractionResult,
    labels: Tuple[int, ...],
    graph_kind: str,
    adjacency_connectivity: str = "corner",
    adjacency_min_voxels: int = 10,
    edge_weight: EdgeWeightMode = "none",
    include_intra_edges: bool = False,
) -> HabitatGraph:
    """
    Build a graph by connecting spatially adjacent habitat-region nodes.

    Two nodes are connected when they share at least ``adjacency_min_voxels``
    neighboring voxel pairs under the requested connectivity rule.

    Handles both single-habitat (intra) and pairwise (inter) graphs, and
    supports face, edge, and corner connectivity.

    Args:
        node_result: Output from connected-region node extraction.
        labels: One label for a single-habitat graph or two labels for a pairwise
            graph.
        graph_kind: ``"single"`` or ``"pairwise"``.
        adjacency_connectivity: Neighbor definition. Default ``"corner"``
            is 8-conn in 2D / 26-conn in 3D. ``"face"`` is 4/6-conn;
            ``"edge"`` is 8/18-conn.
        adjacency_min_voxels: Minimum adjacent voxel pair count required to
            create an edge. Must be >= 1. Default ``10``: an edge exists only
            when two regions are adjacent and share at least 10 contact voxels.
        edge_weight: ``"contact_voxels"`` stores the voxel-pair count as the edge
            weight; ``"none"`` keeps an unweighted binary graph.
        include_intra_edges: For pairwise graphs, also connect same-label node
            pairs that are spatially adjacent.  Ignored for single-label graphs.

    Returns:
        HabitatGraph: Graph with adjacency-derived edges.

    Raises:
        ValueError: If ``adjacency_min_voxels < 1`` or ``labels`` is empty.
    """
    if adjacency_min_voxels < 1:
        raise ValueError("adjacency_min_voxels must be >= 1.")
    if len(labels) not in (1, 2):
        raise ValueError("labels must contain one or two habitat labels.")

    active_labels = list(labels)

    node_id_array, id_to_node = _build_node_id_map(node_result, active_labels)
    offsets = _half_offsets(node_id_array.ndim, adjacency_connectivity)
    pair_counts = _count_adjacency_pairs(node_id_array, offsets)

    # Collect all nodes that belong to the requested labels.
    all_nodes: List[HabitatGraphNode] = []
    for label in active_labels:
        all_nodes.extend(node_result.nodes_by_habitat.get(int(label), []))
    graph_nodes = _nodes_to_dict(all_nodes)

    label_a = int(labels[0])
    label_b = int(labels[1]) if len(labels) == 2 else None

    edges: List[HabitatGraphEdge] = []
    for (int_id_a, int_id_b), contact_voxels in pair_counts.items():
        if contact_voxels < adjacency_min_voxels:
            continue
        node_a = id_to_node.get(int_id_a)
        node_b = id_to_node.get(int_id_b)
        if node_a is None or node_b is None:
            continue

        hab_a = node_a.habitat_label
        hab_b = node_b.habitat_label
        same_label = hab_a == hab_b

        # For pairwise graphs, decide whether to include this edge.
        if label_b is not None:
            if same_label and not include_intra_edges:
                continue
            if not same_label and not (
                (hab_a == label_a and hab_b == label_b)
                or (hab_a == label_b and hab_b == label_a)
            ):
                continue

        edge_type = "intra" if same_label else "inter"
        distance = float(np.linalg.norm(node_a.centroid - node_b.centroid))
        weight = _contact_weight(int(contact_voxels), edge_weight)
        edges.append(
            HabitatGraphEdge(
                source=node_a.node_id,
                target=node_b.node_id,
                edge_type=edge_type,
                distance=distance,
                contact_voxels=int(contact_voxels),
                weight=weight,
            )
        )

    return HabitatGraph(
        graph_kind=graph_kind,  # type: ignore[arg-type]
        labels=labels,
        nodes=graph_nodes,
        edges=edges,
    )


def iter_label_pairs(labels: Sequence[int]) -> Iterable[Tuple[int, int]]:
    """Yield stable pairwise habitat label combinations."""
    return combinations(sorted(int(label) for label in labels), 2)


def iter_cross_label_nodes(
    nodes_a: Sequence[HabitatGraphNode],
    nodes_b: Sequence[HabitatGraphNode],
) -> Iterable[Tuple[HabitatGraphNode, HabitatGraphNode]]:
    """Yield explicit cross-label node pairs for callers that need them."""
    return product(nodes_a, nodes_b)
