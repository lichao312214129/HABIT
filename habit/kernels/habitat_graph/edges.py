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
    MinDistanceEdgeTable,
)
from habit.kernels.habitat_graph.proximity import (
    candidate_node_pairs,
    collect_coords_by_node_id,
    lattice_chebyshev_radius,
    min_distances_for_pairs,
    owner_volume,
    uses_uniform_grid,
    volume_sweep_min_distances,
    volume_sweep_worthwhile,
)

__all__ = [
    "as_intra_edge",
    "compose_pairwise_graph",
    "build_centroid_distance_graph",
    "build_centroid_inter_edges",
    "build_min_distance_inter_edges",
    "build_min_distance_edge_table",
    "build_min_distance_edges",
    "build_min_distance_graph",
    "build_adjacency_graph",
    "iter_label_pairs",
    "iter_cross_label_nodes",
    "lattice_chebyshev_radius",
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
            edges within each habitat. PathPrism source
            (``multi_graph_process.py``) adds those intra edges even though
            the STAR Methods text said inter-only. Whole-graph metrics
            (modularity, assortativity, betweenness) use the full graph;
            interface metrics (isolated ratio, ``avg_h*_per_h*``, pair
            degree family) count inter-class neighbors only.

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


def as_intra_edge(edge: HabitatGraphEdge) -> HabitatGraphEdge:
    """Copy an edge and tag it ``intra`` (pairwise reuse of single-habitat edges)."""
    if edge.edge_type == "intra":
        return edge
    return HabitatGraphEdge(
        source=edge.source,
        target=edge.target,
        edge_type="intra",
        distance=edge.distance,
        contact_voxels=edge.contact_voxels,
        weight=edge.weight,
    )


def compose_pairwise_graph(
    nodes: Sequence[HabitatGraphNode],
    labels: Tuple[int, int],
    inter_edges: Sequence[HabitatGraphEdge],
    intra_edges: Sequence[HabitatGraphEdge],
) -> HabitatGraph:
    """
    Assemble a pairwise graph from inter edges plus reused intra edges.

    Intra edges must already be tagged ``intra`` (use :func:`as_intra_edge`).
    Node membership and stored distances are unchanged from a from-scratch
    pairwise build; only the closest-voxel (or centroid) queries are skipped
    for pairs that a single-habitat graph already measured.
    """
    if len(labels) != 2:
        raise ValueError("compose_pairwise_graph requires exactly two labels.")
    return HabitatGraph(
        graph_kind="pairwise",
        labels=(int(labels[0]), int(labels[1])),
        nodes=_nodes_to_dict(nodes),
        edges=list(inter_edges) + list(intra_edges),
    )


def _node_voxel_coords(
    node_result: HabitatNodeExtractionResult,
    node: HabitatGraphNode,
) -> np.ndarray:
    """
    Return voxel-index coordinates for one connected-region node.

    Coordinates are integer array indices (row/col or z/row/col), the same
    units used by ``centroid_distance``. Physical spacing is not applied.

    Args:
        node_result: Node extraction result that holds per-habitat component maps.
        node: Node whose voxels should be collected.

    Returns:
        np.ndarray: Coordinate array of shape ``(n_voxels, ndim)``. Empty when
        the component map is missing or the component id is absent.
    """
    component_map = node_result.component_maps.get(int(node.habitat_label))
    if component_map is None:
        return np.empty((0, node_result.label_array.ndim), dtype=float)
    coords = np.argwhere(component_map == int(node.component_id))
    if coords.size == 0:
        return np.empty((0, component_map.ndim), dtype=float)
    values = coords.astype(float, copy=False)
    if node_result.crop_offset is not None:
        values = values + np.asarray(node_result.crop_offset, dtype=float)
    return values


def _min_voxel_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    Return the closest-point Euclidean distance between two voxel sets.

    This is the set-separation (minimum pairwise) distance
    ``min_{a in A, b in B} ||a-b||``. It is not the Hausdorff distance,
    which uses a max-of-mins. Default is a CPU kd-tree; pass
    ``device="cuda"`` on :func:`habit.utils.torch_graph_utils.min_voxel_distance`
    only when a single large pair should use ``cdist``.

    Args:
        coords_a: Voxel coordinates of region A, shape ``(n_a, ndim)``.
        coords_b: Voxel coordinates of region B, shape ``(n_b, ndim)``.

    Returns:
        float: Minimum Euclidean distance in voxel-index units, or ``inf``
        when either set is empty.
    """
    from habit.utils.torch_graph_utils import min_voxel_distance

    return min_voxel_distance(coords_a, coords_b, device="cpu")


def _bbox_min_distance(bbox_a: Tuple[int, ...], bbox_b: Tuple[int, ...]) -> float:
    """
    Euclidean lower bound on closest-voxel distance from two half-open boxes.

    Each bbox is ``(min_0, ..., min_{d-1}, max_0, ..., max_{d-1})`` with
    exclusive upper corners. Occupied voxels run through ``max - 1``, so
    adjacent boxes such as ``[0, 8)`` and ``[8, 16)`` have gap 1.

    Args:
        bbox_a: Half-open box of region A.
        bbox_b: Half-open box of region B.

    Returns:
        Lower bound on ``min ||a-b||``. Zero when the boxes overlap.
    """
    n_dim = len(bbox_a) // 2
    gap_sq = 0.0
    for axis in range(n_dim):
        min_a = bbox_a[axis]
        max_a = bbox_a[n_dim + axis]
        min_b = bbox_b[axis]
        max_b = bbox_b[n_dim + axis]
        if max_a <= min_b:
            gap = float(min_b - max_a + 1)
        elif max_b <= min_a:
            gap = float(min_a - max_b + 1)
        else:
            gap = 0.0
        gap_sq += gap * gap
    return float(gap_sq ** 0.5)


def _min_distance_edges_for_pairs(
    nodes_a: Sequence[HabitatGraphNode],
    nodes_b: Sequence[HabitatGraphNode],
    coords_by_id: Dict[str, np.ndarray],
    distance_threshold: float,
    edge_weight: EdgeWeightMode,
    edge_type: str,
) -> List[HabitatGraphEdge]:
    """
    Connect node pairs whose closest voxels are within ``distance_threshold``.

    When ``nodes_a`` and ``nodes_b`` are the same sequence (intra-label), each
    unordered pair is considered once (``i < j``).

    Args:
        nodes_a: First node group.
        nodes_b: Second node group. May be the same object as ``nodes_a``.
        coords_by_id: Precomputed voxel coordinates keyed by node id.
        distance_threshold: Maximum closest-point distance for an edge.
        edge_weight: Optional distance-derived edge weighting mode.
        edge_type: Stored ``HabitatGraphEdge.edge_type`` (``inter`` / ``intra``
            / ``min_distance``).

    Returns:
        List[HabitatGraphEdge]: Edges whose minimum voxel distance is ``<=``
        the threshold.
    """
    edges: List[HabitatGraphEdge] = []
    same_group = nodes_a is nodes_b
    trees: Dict[str, cKDTree] = {}

    def _tree(node_id: str, coords: np.ndarray) -> cKDTree:
        tree = trees.get(node_id)
        if tree is None:
            tree = cKDTree(coords)
            trees[node_id] = tree
        return tree

    for index_a, node_a in enumerate(nodes_a):
        coords_a = coords_by_id.get(node_a.node_id)
        if coords_a is None or coords_a.size == 0:
            continue
        start_b = index_a + 1 if same_group else 0
        for node_b in nodes_b[start_b:]:
            if node_a.node_id == node_b.node_id:
                continue
            coords_b = coords_by_id.get(node_b.node_id)
            if coords_b is None or coords_b.size == 0:
                continue
            if _bbox_min_distance(node_a.bbox, node_b.bbox) > distance_threshold:
                continue
            tree_a = _tree(node_a.node_id, coords_a)
            tree_b = _tree(node_b.node_id, coords_b)
            if coords_a.shape[0] <= coords_b.shape[0]:
                distance = float(np.min(tree_b.query(coords_a, k=1)[0]))
            else:
                distance = float(np.min(tree_a.query(coords_b, k=1)[0]))
            if distance > distance_threshold:
                continue
            edges.append(
                HabitatGraphEdge(
                    source=node_a.node_id,
                    target=node_b.node_id,
                    edge_type=edge_type,
                    distance=distance,
                    contact_voxels=None,
                    weight=_distance_weight(distance, edge_weight),
                )
            )
    return edges


def _empty_edge_table() -> MinDistanceEdgeTable:
    """Return a zero-length closest-voxel table."""
    empty_i = np.empty(0, dtype=np.int64)
    empty_h = np.empty(0, dtype=np.int32)
    return MinDistanceEdgeTable(
        index_a=empty_i,
        index_b=empty_i,
        distance=np.empty(0, dtype=np.float64),
        habitat_a=empty_h,
        habitat_b=empty_h,
    )


def _table_from_index_pairs(
    node_list: Sequence[HabitatGraphNode],
    index_a: np.ndarray,
    index_b: np.ndarray,
    distances: np.ndarray,
    *,
    allow_same_label: bool,
    allow_cross_label: bool,
) -> MinDistanceEdgeTable:
    """Pack surviving index pairs into a :class:`MinDistanceEdgeTable`."""
    habitats = np.asarray(
        [int(node.habitat_label) for node in node_list], dtype=np.int32
    )
    src = np.asarray(index_a, dtype=np.int64)
    dst = np.asarray(index_b, dtype=np.int64)
    dist = np.asarray(distances, dtype=np.float64)
    if src.size == 0:
        return _empty_edge_table()
    habitat_a = habitats[src]
    habitat_b = habitats[dst]
    same = habitat_a == habitat_b
    keep = np.ones(src.shape[0], dtype=bool)
    if not allow_same_label:
        keep &= ~same
    if not allow_cross_label:
        keep &= same
    return MinDistanceEdgeTable(
        index_a=src[keep],
        index_b=dst[keep],
        distance=dist[keep],
        habitat_a=habitat_a[keep],
        habitat_b=habitat_b[keep],
    )


def _edges_from_table(
    node_list: Sequence[HabitatGraphNode],
    table: MinDistanceEdgeTable,
    edge_weight: EdgeWeightMode,
) -> List[HabitatGraphEdge]:
    """Materialize dataclass edges (public / viz path only)."""
    edges: List[HabitatGraphEdge] = []
    for slot in range(table.index_a.shape[0]):
        node_a = node_list[int(table.index_a[slot])]
        node_b = node_list[int(table.index_b[slot])]
        distance = float(table.distance[slot])
        same = int(table.habitat_a[slot]) == int(table.habitat_b[slot])
        edges.append(
            HabitatGraphEdge(
                source=node_a.node_id,
                target=node_b.node_id,
                edge_type="min_distance" if same else "inter",
                distance=distance,
                contact_voxels=None,
                weight=_distance_weight(distance, edge_weight),
            )
        )
    return edges


def build_min_distance_edge_table(
    node_result: HabitatNodeExtractionResult,
    nodes: Sequence[HabitatGraphNode],
    distance_threshold: float,
    *,
    allow_same_label: bool = True,
    allow_cross_label: bool = True,
) -> MinDistanceEdgeTable:
    """
    Exact ``min_distance`` edges as integer arrays into ``nodes``.

    Same geometry as :func:`build_min_distance_edges`, without allocating
    one Python object per edge.

    Args:
        node_result: Node extraction (component maps + optional lattice).
        nodes: Nodes to connect; table indices refer to this sequence.
        distance_threshold: Maximum closest-voxel distance.
        allow_same_label: Keep intra-habitat pairs.
        allow_cross_label: Keep inter-habitat pairs.

    Returns:
        MinDistanceEdgeTable: Surviving pairs with ``index_a < index_b``.

    Raises:
        ValueError: If ``distance_threshold < 0``.
    """
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be >= 0.")
    node_list = list(nodes)
    if len(node_list) < 2:
        return _empty_edge_table()
    n_voxels = int(np.count_nonzero(node_result.label_array))
    if uses_uniform_grid(node_result) and volume_sweep_worthwhile(
        n_voxels, float(distance_threshold), int(node_result.label_array.ndim)
    ):
        owner = owner_volume(node_result, node_list)
        index_a, index_b, distances = volume_sweep_min_distances(
            owner, float(distance_threshold), len(node_list)
        )
        return _table_from_index_pairs(
            node_list,
            index_a,
            index_b,
            distances,
            allow_same_label=allow_same_label,
            allow_cross_label=allow_cross_label,
        )
    if not uses_uniform_grid(node_result):
        coords_by_id = collect_coords_by_node_id(node_result, node_list)
        raw = _min_distance_edges_for_pairs(
            node_list,
            node_list,
            coords_by_id,
            float(distance_threshold),
            "none",
            "min_distance",
        )
        index = {node.node_id: slot for slot, node in enumerate(node_list)}
        src: List[int] = []
        dst: List[int] = []
        dist: List[float] = []
        for edge in raw:
            slot_a = index[edge.source]
            slot_b = index[edge.target]
            if slot_a == slot_b:
                continue
            if slot_a > slot_b:
                slot_a, slot_b = slot_b, slot_a
            src.append(slot_a)
            dst.append(slot_b)
            dist.append(float(edge.distance) if edge.distance is not None else np.inf)
        return _table_from_index_pairs(
            node_list,
            np.asarray(src, dtype=np.int64),
            np.asarray(dst, dtype=np.int64),
            np.asarray(dist, dtype=np.float64),
            allow_same_label=allow_same_label,
            allow_cross_label=allow_cross_label,
        )
    pairs = candidate_node_pairs(
        node_list,
        node_result,
        float(distance_threshold),
        allow_same_label=allow_same_label,
        allow_cross_label=allow_cross_label,
    )
    if not pairs:
        return _empty_edge_table()
    coords_by_id = collect_coords_by_node_id(node_result, node_list)
    pair_a = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    pair_b = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    distances = min_distances_for_pairs(
        node_list, coords_by_id, pairs, float(distance_threshold)
    )
    finite = np.isfinite(distances) & (distances <= float(distance_threshold))
    return _table_from_index_pairs(
        node_list,
        pair_a[finite],
        pair_b[finite],
        np.asarray(distances[finite], dtype=np.float64),
        allow_same_label=True,
        allow_cross_label=True,
    )


def build_min_distance_edges(
    node_result: HabitatNodeExtractionResult,
    nodes: Sequence[HabitatGraphNode],
    distance_threshold: float,
    edge_weight: EdgeWeightMode = "none",
    *,
    allow_same_label: bool = True,
    allow_cross_label: bool = True,
) -> List[HabitatGraphEdge]:
    """
    Exact ``min_distance`` edges among ``nodes``.

    ``uniform_grid`` uses a voxel-neighbour sweep when the Chebyshev
    window times painted voxels is cheap, otherwise a lattice range
    search (or a centroid-ball envelope when that window is huge).
    ``component`` nodes have no lattice metadata and keep the all-pairs
    closest-voxel walk. Distances are true closest-voxel values.

    Args:
        node_result: Node extraction (component maps + optional lattice).
        nodes: Nodes to connect.
        distance_threshold: Maximum closest-voxel distance.
        edge_weight: Optional distance-derived weight.
        allow_same_label: Emit intra-habitat edges (type ``min_distance``).
        allow_cross_label: Emit inter-habitat edges (type ``inter``).

    Returns:
        List[HabitatGraphEdge]: Undirected proximity edges.

    Raises:
        ValueError: If ``distance_threshold < 0``.
    """
    table = build_min_distance_edge_table(
        node_result,
        nodes,
        distance_threshold,
        allow_same_label=allow_same_label,
        allow_cross_label=allow_cross_label,
    )
    return _edges_from_table(list(nodes), table, edge_weight)


def build_min_distance_inter_edges(
    node_result: HabitatNodeExtractionResult,
    labels: Tuple[int, int],
    distance_threshold: float,
    edge_weight: EdgeWeightMode = "none",
) -> List[HabitatGraphEdge]:
    """Closest-voxel edges between two habitats only (no intra pairs)."""
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be >= 0.")
    label_a, label_b = int(labels[0]), int(labels[1])
    nodes_a = list(node_result.nodes_by_habitat.get(label_a, []))
    nodes_b = list(node_result.nodes_by_habitat.get(label_b, []))
    if not nodes_a or not nodes_b:
        return []
    return build_min_distance_edges(
        node_result,
        [*nodes_a, *nodes_b],
        distance_threshold,
        edge_weight,
        allow_same_label=False,
        allow_cross_label=True,
    )


def build_centroid_inter_edges(
    nodes_a: Sequence[HabitatGraphNode],
    nodes_b: Sequence[HabitatGraphNode],
    distance_threshold: float,
    edge_weight: EdgeWeightMode = "none",
) -> List[HabitatGraphEdge]:
    """Centroid-proximity edges between two habitats only (no intra pairs)."""
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be >= 0.")
    edges: List[HabitatGraphEdge] = []
    if not nodes_a or not nodes_b:
        return edges
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
    return edges


def build_min_distance_graph(
    node_result: HabitatNodeExtractionResult,
    labels: Tuple[int, ...],
    graph_kind: str,
    distance_threshold: float,
    edge_weight: EdgeWeightMode = "none",
    include_intra_edges: bool = False,
) -> HabitatGraph:
    """
    Build a graph by connecting regions whose closest voxels are within threshold.

    Unlike :func:`build_centroid_distance_graph`, the distance is the minimum
    Euclidean distance between any voxel of region A and any voxel of region B
    (closest-voxel / set-separation distance), not the distance between
    centroids and not the Hausdorff distance. Units are voxel indices,
    matching ``centroid_distance``. The same ``d_min`` is stored on the
    edge as ``distance`` and is what ``avg_edge_distance`` summarizes.

    An undirected edge exists when ``min_{a in A, b in B} ||a-b|| <= threshold``.

    Args:
        node_result: Output from connected-region node extraction. Voxel
            coordinates are read from the component maps.
        labels: One label for a single-habitat graph or two labels for a pair.
        graph_kind: ``"single"`` or ``"pairwise"``.
        distance_threshold: Maximum closest-point Euclidean distance in voxel
            index units. Reuses the same field as ``centroid_distance``.
        edge_weight: Optional distance-derived edge weighting mode.
        include_intra_edges: For pairwise graphs, also add same-label
            closest-point edges within each habitat.

    Returns:
        HabitatGraph: Graph with closest-point edges.

    Raises:
        ValueError: If ``distance_threshold < 0`` or ``labels`` is empty.
    """
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be >= 0.")
    if len(labels) not in (1, 2):
        raise ValueError("labels must contain one or two habitat labels.")

    all_nodes: List[HabitatGraphNode] = []
    for label in labels:
        all_nodes.extend(node_result.nodes_by_habitat.get(int(label), []))
    graph_nodes = _nodes_to_dict(all_nodes)
    edges: List[HabitatGraphEdge] = []

    if len(all_nodes) < 2:
        return HabitatGraph(
            graph_kind=graph_kind,  # type: ignore[arg-type]
            labels=labels,
            nodes=graph_nodes,
            edges=edges,
        )

    raw_edges = build_min_distance_edges(
        node_result,
        all_nodes,
        distance_threshold,
        edge_weight,
        allow_same_label=(len(labels) == 1) or bool(include_intra_edges),
        allow_cross_label=len(labels) == 2,
    )
    if len(labels) == 1:
        edges.extend(raw_edges)
    else:
        for edge in raw_edges:
            if edge.edge_type == "min_distance":
                edges.append(as_intra_edge(edge))
            else:
                edges.append(edge)

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
