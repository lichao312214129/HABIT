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
"""Build :class:`GraphArrays` and run hop metrics on CSR without NetworkX."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from habit.kernels.habitat_graph.models import (
    EdgeWeightMode,
    GraphArrays,
    GraphKind,
    HabitatGraph,
    HabitatGraphNode,
    MinDistanceEdgeTable,
)
from habit.kernels.habitat_graph.traversal import HopMetricResult
from habit.utils.graph_brandes_utils import csr_from_edge_arrays, hop_metrics_csr
from habit.utils.graph_csr_utils import connected_components_csr, induce_csr

__all__ = [
    "habitat_graph_to_arrays",
    "graph_arrays_from_table",
    "hop_from_graph_arrays",
    "edge_weight_from_distance",
]


def edge_weight_from_distance(
    distance: np.ndarray,
    edge_weight: EdgeWeightMode,
) -> np.ndarray:
    """Vectorized edge weights from closest-voxel distances.

    Args:
        distance: Finite distances, shape ``(m,)``.
        edge_weight: Weight rule.

    Returns:
        np.ndarray: Weights aligned with ``distance``.
    """
    values = np.asarray(distance, dtype=np.float64)
    if edge_weight == "distance":
        return values
    if edge_weight == "inverse_distance":
        return 1.0 / (values + 1e-6)
    return np.ones(values.shape[0], dtype=np.float64)


def habitat_graph_to_arrays(graph: HabitatGraph) -> GraphArrays:
    """Convert a :class:`HabitatGraph` to integer arrays.

    Args:
        graph: Lightweight habitat graph.

    Returns:
        GraphArrays: Same nodes and undirected edges.
    """
    node_ids = tuple(graph.nodes.keys())
    index = {node_id: slot for slot, node_id in enumerate(node_ids)}
    n_nodes = len(node_ids)
    habitats = np.empty(n_nodes, dtype=np.int32)
    voxels = np.empty(n_nodes, dtype=np.float64)
    if n_nodes == 0:
        centroids = np.empty((0, 1), dtype=np.float64)
    else:
        first = next(iter(graph.nodes.values()))
        centroids = np.empty((n_nodes, int(first.centroid.size)), dtype=np.float64)
    for slot, node_id in enumerate(node_ids):
        node = graph.nodes[node_id]
        habitats[slot] = int(node.habitat_label)
        voxels[slot] = float(node.voxel_count)
        centroids[slot] = np.asarray(node.centroid, dtype=np.float64)
    src: list = []
    dst: list = []
    distance: list = []
    weight: list = []
    is_inter: list = []
    contact: list = []
    for edge in graph.edges:
        slot_a = index[edge.source]
        slot_b = index[edge.target]
        if slot_a == slot_b:
            continue
        if slot_a > slot_b:
            slot_a, slot_b = slot_b, slot_a
        src.append(slot_a)
        dst.append(slot_b)
        distance.append(
            float(edge.distance) if edge.distance is not None else np.nan
        )
        weight.append(float(edge.weight))
        is_inter.append(edge.edge_type != "intra" and habitats[slot_a] != habitats[slot_b])
        contact.append(
            float(edge.contact_voxels) if edge.contact_voxels is not None else np.nan
        )
    return GraphArrays(
        graph_kind=graph.graph_kind,
        labels=graph.labels,
        node_ids=node_ids,
        habitats=habitats,
        voxels=voxels,
        centroids=centroids,
        src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64),
        distance=np.asarray(distance, dtype=np.float64),
        weight=np.asarray(weight, dtype=np.float64),
        is_inter=np.asarray(is_inter, dtype=bool),
        contact=np.asarray(contact, dtype=np.float64),
    )


def graph_arrays_from_table(
    all_nodes: Sequence[HabitatGraphNode],
    table: MinDistanceEdgeTable,
    keep_labels: Sequence[int],
    graph_kind: GraphKind,
    *,
    include_intra: bool = True,
    edge_weight: EdgeWeightMode = "none",
) -> GraphArrays:
    """
    Slice a global closest-voxel table to one habitat or one pair.

    Args:
        all_nodes: Node sequence used when the table was built.
        table: Global edge table.
        keep_labels: Habitat ids to keep.
        graph_kind: ``single`` or ``pairwise``.
        include_intra: Keep same-habitat edges (pairwise interface-only
            when False).
        edge_weight: Distance-derived weight rule.

    Returns:
        GraphArrays: Locally reindexed subgraph.
    """
    wanted = {int(label) for label in keep_labels}
    node_list = list(all_nodes)
    keep_slots = [
        slot
        for slot, node in enumerate(node_list)
        if int(node.habitat_label) in wanted
    ]
    local_nodes = [node_list[slot] for slot in keep_slots]
    node_ids = tuple(node.node_id for node in local_nodes)
    n_nodes = len(local_nodes)
    habitats = np.asarray(
        [int(node.habitat_label) for node in local_nodes], dtype=np.int32
    )
    voxels = np.asarray(
        [float(node.voxel_count) for node in local_nodes], dtype=np.float64
    )
    if n_nodes == 0:
        centroids = np.empty((0, 1), dtype=np.float64)
    else:
        centroids = np.stack(
            [np.asarray(node.centroid, dtype=np.float64) for node in local_nodes],
            axis=0,
        )
    remap = np.full(len(node_list), -1, dtype=np.int64)
    if keep_slots:
        remap[np.asarray(keep_slots, dtype=np.int64)] = np.arange(
            n_nodes, dtype=np.int64
        )
    if table.index_a.size == 0 or n_nodes < 2:
        empty = np.empty(0, dtype=np.int64)
        return GraphArrays(
            graph_kind=graph_kind,
            labels=tuple(int(label) for label in keep_labels),
            node_ids=node_ids,
            habitats=habitats,
            voxels=voxels,
            centroids=centroids,
            src=empty,
            dst=empty,
            distance=np.empty(0, dtype=np.float64),
            weight=np.empty(0, dtype=np.float64),
            is_inter=np.empty(0, dtype=bool),
            contact=np.empty(0, dtype=np.float64),
        )
    mask = np.isin(table.habitat_a, list(wanted)) & np.isin(
        table.habitat_b, list(wanted)
    )
    if not include_intra:
        mask &= table.habitat_a != table.habitat_b
    src = remap[table.index_a[mask]]
    dst = remap[table.index_b[mask]]
    valid = (src >= 0) & (dst >= 0)
    src = src[valid]
    dst = dst[valid]
    distance = np.asarray(table.distance[mask][valid], dtype=np.float64)
    is_inter = habitats[src] != habitats[dst]
    return GraphArrays(
        graph_kind=graph_kind,
        labels=tuple(int(label) for label in keep_labels),
        node_ids=node_ids,
        habitats=habitats,
        voxels=voxels,
        centroids=centroids,
        src=src.astype(np.int64, copy=False),
        dst=dst.astype(np.int64, copy=False),
        distance=distance,
        weight=edge_weight_from_distance(distance, edge_weight),
        is_inter=is_inter,
        contact=np.full(src.shape[0], np.nan, dtype=np.float64),
    )


def hop_from_graph_arrays(
    arrays: GraphArrays,
    *,
    largest_component: bool,
) -> Tuple[HopMetricResult, int]:
    """
    Brandes / path metrics on ``arrays``.

    Args:
        arrays: Integer-indexed graph.
        largest_component: If True, hop metrics use only the largest
            component (single-habitat PathPrism convention). Pairwise
            betweenness uses the full graph (False).

    Returns:
        ``(hop, n_components)``. ``hop.n_nodes`` is 0 when undefined.
        Isolated / edgeless graphs return a zero hop result.
    """
    n_nodes = len(arrays.node_ids)
    if n_nodes == 0:
        return (
            HopMetricResult(0, {}, {}, 0.0, 0.0),
            0,
        )
    indptr, indices = csr_from_edge_arrays(n_nodes, arrays.src, arrays.dst)
    labels, n_comp, sizes = connected_components_csr(indptr, indices, n_nodes)
    if n_nodes <= 1 or arrays.src.size == 0:
        return HopMetricResult(0, {}, {}, 0.0, 0.0), int(n_comp)
    work_indptr = indptr
    work_indices = indices
    work_n = n_nodes
    old_ids = np.arange(n_nodes, dtype=np.int64)
    if largest_component:
        keep = labels == int(np.argmax(sizes))
        if int(keep.sum()) <= 1:
            return HopMetricResult(int(keep.sum()), {}, {}, 0.0, 0.0), int(n_comp)
        work_indptr, work_indices, old_ids = induce_csr(indptr, indices, keep)
        work_n = int(old_ids.size)
    hop_arr = hop_metrics_csr(work_indptr, work_indices, work_n, device="auto")
    bc = hop_arr.betweenness
    cc = hop_arr.closeness
    avg_path = hop_arr.avg_path_length
    diameter = hop_arr.diameter
    betweenness = {}
    closeness = {}
    for slot in range(work_n):
        node_id = arrays.node_ids[int(old_ids[slot])]
        betweenness[node_id] = float(bc[slot])
        closeness[node_id] = float(cc[slot])
    return (
        HopMetricResult(
            n_nodes=work_n,
            betweenness=betweenness,
            closeness=closeness,
            avg_path_length=float(avg_path),
            diameter=float(diameter) if work_n > 1 else 0.0,
        ),
        int(n_comp),
    )
