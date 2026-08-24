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
"""Graph metric calculation for habitat graph features."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple
import warnings

import networkx as nx
from networkx.algorithms import community as nx_community
import numpy as np
from scipy.spatial import cKDTree

from habit.kernels.habitat_graph.extended_metrics import (
    compute_extended_graph_metrics,
    compute_extended_pairwise_metrics,
)
from habit.kernels.habitat_graph.array_graph import (
    habitat_graph_to_arrays,
    hop_from_graph_arrays,
)
from habit.kernels.habitat_graph.traversal import (
    component_summary,
    hop_metrics,
    mean_betweenness,
)
from habit.utils.graph_brandes_utils import csr_from_edge_arrays
from habit.utils.graph_csr_utils import (
    attribute_assortativity_csr,
    average_clustering_csr,
    degree_assortativity_csr,
    degrees_csr,
    louvain_modularity_csr,
)
from habit.utils.igraph_graph_utils import (
    average_clustering_igraph,
    igraph_is_available,
    modularity_igraph,
)
from habit.kernels.habitat_graph.models import (
    GraphArrays,
    HabitatGraph,
    HabitatGraphEdge,
    HabitatGraphNode,
    label_token,
    pair_feature_prefix,
    single_feature_prefix,
)

__all__ = [
    "calculate_single_graph_metrics",
    "calculate_pairwise_graph_metrics",
]


def _safe_mean(values: Sequence[float]) -> float:
    """Return a numeric mean with an empty-sequence fallback."""
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    """Return a numeric standard deviation with an empty-sequence fallback."""
    return float(np.std(values)) if values else 0.0


def _finite_or_zero(value: float) -> float:
    """Return finite numeric values and replace NaN/Inf with zero."""
    return float(value) if np.isfinite(value) else 0.0


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numeric values and return zero when the denominator is invalid."""
    if denominator <= 0:
        return 0.0
    return _finite_or_zero(float(numerator) / float(denominator))


def _coefficient_of_variation(values: Sequence[float]) -> float:
    """Return standard deviation divided by mean with zero-safe behavior."""
    if not values:
        return 0.0
    mean_value = float(np.mean(values))
    if abs(mean_value) < 1e-12:
        return 0.0
    return float(np.std(values) / mean_value)


def _entropy(values: Sequence[int]) -> float:
    """
    Calculate entropy of a discrete value distribution.

    Args:
        values: Discrete values such as node degrees.

    Returns:
        float: Shannon entropy in bits.
    """
    if not values:
        return 0.0
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))


def _to_networkx(graph: HabitatGraph) -> nx.Graph:
    """Convert the lightweight graph model to a NetworkX graph."""
    nx_graph = nx.Graph()
    for node_id, node in graph.nodes.items():
        nx_graph.add_node(
            node_id,
            habitat_label=node.habitat_label,
            voxel_count=node.voxel_count,
            centroid=node.centroid,
        )
    for edge in graph.edges:
        nx_graph.add_edge(
            edge.source,
            edge.target,
            weight=edge.weight,
            distance=edge.distance,
            contact_voxels=edge.contact_voxels,
            edge_type=edge.edge_type,
        )
    return nx_graph


def _edge_distances(graph: HabitatGraph) -> List[float]:
    """
    Return the distance stored on each edge.

    The stored length depends on the edge method: closest-voxel
    ``d_min`` for ``min_distance`` (the library default), centroid
    Euclidean distance for ``centroid_distance`` and ``adjacency``.
    """
    return [
        float(edge.distance)
        for edge in graph.edges
        if edge.distance is not None
    ]


def _inter_edges(graph: HabitatGraph) -> List[HabitatGraphEdge]:
    """
    Return inter-class edges of a pairwise graph.

    Intra-class edges (``edge_type == 'intra'``) are excluded so that interface
    metrics describe only connections between the two habitat labels, matching
    the source PathPrism multi-tissue feature definitions.
    """
    return [edge for edge in graph.edges if edge.edge_type != "intra"]


def _contact_voxels(edges: Iterable[HabitatGraphEdge]) -> List[int]:
    """Return contact voxel counts from the supplied edge collection."""
    return [
        int(edge.contact_voxels)
        for edge in edges
        if edge.contact_voxels is not None
    ]


def _local_contact_scales(
    graph: HabitatGraph,
    edges: Iterable[HabitatGraphEdge],
) -> List[float]:
    """
    Normalize each contact count by the smaller node's local area scale.

    A contact count is an interface-like quantity.  For one edge between
    nodes with voxel counts ``v_i`` and ``v_j``, its natural local scale is
    ``min(v_i, v_j)**((d-1)/d)`` rather than the whole-ROI interface scale.
    This makes the mean and maximum contact summaries comparable across local
    node sizes.  The value is not constrained to ``[0, 1]`` because thin,
    elongated nodes and diagonal adjacency can create more adjacent voxel
    pairs than the compact-shape approximation.

    Args:
        graph: Graph containing node voxel counts and centroid dimensionality.
        edges: Contact-bearing edges to normalize.

    Returns:
        List[float]: One finite locally scaled contact value per valid edge.
    """
    if not graph.nodes:
        return []
    first_node = next(iter(graph.nodes.values()))
    ndim = int(first_node.centroid.size)
    if ndim <= 0:
        return []

    values: List[float] = []
    exponent = (ndim - 1.0) / ndim
    for edge in edges:
        if edge.contact_voxels is None:
            continue
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None:
            continue
        local_volume = float(min(source.voxel_count, target.voxel_count))
        local_area_scale = local_volume**exponent if local_volume > 0 else 0.0
        if local_area_scale > 0:
            values.append(float(edge.contact_voxels) / local_area_scale)
    return values


def _spatial_dispersion(nodes: Iterable[HabitatGraphNode]) -> float:
    """
    Summarize how broadly node centroids spread in the habitat map.

    The value is the mean standard deviation across coordinate dimensions.
    """
    positions = np.asarray([node.centroid for node in nodes], dtype=float)
    if positions.size == 0 or positions.shape[0] < 2:
        return 0.0
    return float(np.mean(np.std(positions, axis=0)))


def _nx_int_edges(
    nx_graph: nx.Graph,
) -> Tuple[int, List[Tuple[int, int]], List[float]]:
    """Integer endpoints and weights in NetworkX insertion order."""
    nodes = list(nx_graph.nodes())
    index = {node_id: slot for slot, node_id in enumerate(nodes)}
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for source, target, data in nx_graph.edges(data=True):
        edges.append((index[source], index[target]))
        weights.append(float(data.get("weight", 1.0)))
    return len(nodes), edges, weights


def _resolve_metric_backend(backend: str) -> str:
    """Return ``igraph`` or ``networkx`` for one metric call."""
    requested = str(backend).strip().lower()
    if requested == "igraph":
        return "igraph"
    if requested in {"auto", ""}:
        return "igraph" if igraph_is_available() else "networkx"
    return "networkx"


def _average_clustering(nx_graph: nx.Graph, backend: str = "auto") -> float:
    """Mean local clustering; igraph when the optional extra is selected."""
    if nx_graph.number_of_nodes() == 0:
        return 0.0
    if _resolve_metric_backend(backend) == "igraph":
        n_nodes, edges, _weights = _nx_int_edges(nx_graph)
        return average_clustering_igraph(n_nodes, edges)
    return float(nx.average_clustering(nx_graph))


def _modularity(nx_graph: nx.Graph, backend: str = "auto") -> float:
    """
    Calculate Louvain-community modularity of a graph.

    Modularity measures how well the graph splits into densely connected
    communities. It is undefined without edges, so empty or edgeless graphs
    return 0.0.

    Args:
        nx_graph: NetworkX graph converted from a habitat graph.

    Returns:
        float: Modularity in roughly [-0.5, 1.0]; 0.0 when undefined.
    """
    if nx_graph.number_of_edges() == 0:
        return 0.0
    if _resolve_metric_backend(backend) == "igraph":
        n_nodes, edges, weights = _nx_int_edges(nx_graph)
        return _finite_or_zero(
            modularity_igraph(n_nodes, edges, weights=weights)
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Louvain with a fixed seed mirrors the source PathPrism modularity.
            communities = nx_community.louvain_communities(
                nx_graph, weight="weight", seed=0
            )
            return _finite_or_zero(
                nx_community.modularity(nx_graph, communities, weight="weight")
            )
    except Exception:
        return 0.0


def _nearest_neighbor_ratio(nodes: Sequence[HabitatGraphNode]) -> float:
    """
    Calculate the Clark-Evans nearest-neighbor ratio of node centroids.

    The ratio ``R`` divides the observed mean nearest-neighbor centroid
    distance by the distance expected under complete spatial randomness (CSR)
    given the same point density. ``R < 1`` indicates clustering, ``R = 1``
    indicates random spacing, and ``R > 1`` indicates regular spacing. The
    study-region measure uses the total occupied voxel count of the involved
    regions (note: the source PathPrism uses the centroid bounding-box area
    instead, so absolute values are not directly comparable across the two).
    A 3D-aware CSR expectation is used so the ratio is valid for volumetric
    habitat maps, not only 2D slides.

    Args:
        nodes: Graph nodes carrying centroid coordinates and voxel counts.

    Returns:
        float: Nearest-neighbor ratio; 0.0 when undefined.
    """
    node_list = list(nodes)
    n_nodes = len(node_list)
    if n_nodes < 2:
        return 0.0

    coords = np.asarray([node.centroid for node in node_list], dtype=float)
    ndim = int(coords.shape[1])
    region_measure = float(sum(node.voxel_count for node in node_list))
    if region_measure <= 0:
        return 0.0

    tree = cKDTree(coords)
    # k=2 because the closest point to each node is the node itself (distance 0).
    distances, _ = tree.query(coords, k=2)
    observed = float(np.mean(distances[:, 1]))

    density = n_nodes / region_measure
    if density <= 0:
        return 0.0
    if ndim == 2:
        expected = 1.0 / (2.0 * np.sqrt(density))
    elif ndim == 3:
        # Mean CSR nearest-neighbor distance in 3D: Gamma(4/3) / (4/3*pi*rho)^(1/3).
        expected = 0.5539602785 / (density ** (1.0 / 3.0))
    else:
        return 0.0
    if expected <= 0:
        return 0.0
    return _finite_or_zero(observed / expected)


def _csr_from_arrays(arrays: GraphArrays):
    """Undirected CSR of ``arrays``."""
    return csr_from_edge_arrays(len(arrays.node_ids), arrays.src, arrays.dst)


def _single_features_from_arrays(
    arrays: GraphArrays,
    nodes: Sequence[HabitatGraphNode],
    *,
    graph_metric_backend: str = "networkx",
) -> Dict[str, float]:
    """Default (non-extended) single-habitat columns from CSR arrays."""
    if len(arrays.labels) != 1:
        raise ValueError("single graph metrics require exactly one label.")
    prefix = single_feature_prefix(arrays.labels[0])
    n_nodes = len(arrays.node_ids)
    n_edges = int(arrays.src.size)
    hop, n_components = hop_from_graph_arrays(
        arrays,
        largest_component=True,
        backend=graph_metric_backend,
    )
    hop_ok = hop.n_nodes > 1
    max_edges = n_nodes * (n_nodes - 1) / 2
    degree_scale = float(n_nodes - 1)
    indptr, indices = _csr_from_arrays(arrays)
    degrees = degrees_csr(indptr).astype(int).tolist()
    edge_distances = [
        float(value) for value in arrays.distance.tolist() if np.isfinite(value)
    ]
    node_voxels = arrays.voxels.astype(float).tolist()
    weights = (
        arrays.weight
        if arrays.weight.size == int(indptr[-1])
        else np.ones(int(indptr[-1]), dtype=np.float64)
    )
    # CSR stores both directions; modularity helper unique-s the upper triangle.
    if arrays.weight.size == n_edges:
        directed_w = np.ones(int(indptr[-1]), dtype=np.float64)
        src = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr))
        dst = indices
        # Scatter undirected weights onto both directed slots.
        weight_map = {
            (int(a), int(b)): float(w)
            for a, b, w in zip(arrays.src.tolist(), arrays.dst.tolist(), arrays.weight.tolist())
        }
        directed_w = np.ones(src.size, dtype=np.float64)
        for slot, (node_a, node_b) in enumerate(zip(src.tolist(), dst.tolist())):
            key = (node_a, node_b) if node_a < node_b else (node_b, node_a)
            directed_w[slot] = weight_map.get(key, 1.0)
        weights = directed_w
    features: Dict[str, float] = {
        f"{prefix}_n_nodes": float(n_nodes),
        f"{prefix}_n_edges": float(n_edges),
        f"{prefix}_edge_density": float(n_edges / max_edges) if max_edges else 0.0,
        f"{prefix}_connected_components": float(n_components),
        f"{prefix}_avg_degree": _safe_mean(degrees),
        f"{prefix}_max_degree": float(max(degrees)) if degrees else 0.0,
        f"{prefix}_min_degree": float(min(degrees)) if degrees else 0.0,
        f"{prefix}_avg_degree_norm": _safe_divide(_safe_mean(degrees), degree_scale),
        f"{prefix}_max_degree_norm": _safe_divide(
            float(max(degrees)) if degrees else 0.0,
            degree_scale,
        ),
        f"{prefix}_min_degree_norm": _safe_divide(
            float(min(degrees)) if degrees else 0.0,
            degree_scale,
        ),
        f"{prefix}_degree_cv": _coefficient_of_variation(degrees),
        f"{prefix}_degree_entropy": _entropy(degrees),
        f"{prefix}_avg_edge_distance": _safe_mean(edge_distances),
        f"{prefix}_std_edge_distance": _safe_std(edge_distances),
        f"{prefix}_avg_node_voxels": _safe_mean(node_voxels),
        f"{prefix}_std_node_voxels": _safe_std(node_voxels),
        f"{prefix}_node_voxels_cv": _coefficient_of_variation(node_voxels),
        f"{prefix}_spatial_dispersion": _spatial_dispersion(nodes),
        f"{prefix}_connected_components_ratio": (
            float(n_components / n_nodes) if n_nodes else 0.0
        ),
        f"{prefix}_nearest_neighbor_ratio": _nearest_neighbor_ratio(nodes),
        f"{prefix}_modularity": louvain_modularity_csr(
            indptr, indices, weights, n_nodes, backend=graph_metric_backend
        ),
    }
    if n_nodes == 0:
        features[f"{prefix}_largest_component_ratio"] = 0.0
        features[f"{prefix}_avg_clustering"] = 0.0
    else:
        lcc_size = hop.n_nodes if hop.n_nodes > 0 else 1
        features[f"{prefix}_largest_component_ratio"] = float(lcc_size / n_nodes)
        features[f"{prefix}_avg_clustering"] = average_clustering_csr(indptr, indices)
    if hop_ok:
        path_scale = float(hop.n_nodes - 1)
        features[f"{prefix}_avg_path_length"] = hop.avg_path_length
        features[f"{prefix}_diameter"] = hop.diameter
        features[f"{prefix}_avg_path_length_norm"] = _safe_divide(
            features[f"{prefix}_avg_path_length"],
            path_scale,
        )
        features[f"{prefix}_diameter_norm"] = _safe_divide(
            features[f"{prefix}_diameter"],
            path_scale,
        )
        features[f"{prefix}_avg_betweenness"] = mean_betweenness(hop.betweenness)
        features[f"{prefix}_avg_closeness"] = _safe_mean(list(hop.closeness.values()))
    else:
        features[f"{prefix}_avg_path_length"] = 0.0
        features[f"{prefix}_diameter"] = 0.0
        features[f"{prefix}_avg_path_length_norm"] = 0.0
        features[f"{prefix}_diameter_norm"] = 0.0
        features[f"{prefix}_avg_betweenness"] = 0.0
        features[f"{prefix}_avg_closeness"] = 0.0
    features[f"{prefix}_degree_assortativity"] = _finite_or_zero(
        degree_assortativity_csr(indptr, indices)
    )
    return features


def calculate_single_graph_metrics(
    graph: HabitatGraph,
    *,
    include_extended_metrics: bool = False,
    extended_min_nodes: int = 10,
    small_world_nrand: int = 100,
    small_world_niter: int = 100,
    rich_club_q: int = 100,
    graph_null_sampler: str = "analytic",
    graph_null_device: str = "auto",
    graph_metric_backend: str = "networkx",
) -> Dict[str, float]:
    """
    Calculate graph features for one habitat label.

    Args:
        graph: Single-habitat graph.
        include_extended_metrics: Also compute efficiency / small-world /
            rich-club / node-distribution summaries. Default False because
            those metrics dominate runtime on large graphs.
        extended_min_nodes: Minimum node count for either small-world sigma.
        small_world_nrand: Degree-preserving ensemble size for
            ``small_world_sigma_rewire``.
        small_world_niter: Rewires per edge when ``graph_null_sampler``
            is ``rewire``.
        rich_club_q: Mixing floor for the ``rewire`` sampler.
        graph_null_sampler: ``analytic`` (default Humphries ER), ``config``,
            or ``rewire``.
        graph_null_device: Batched-metric device for the null ensemble.
        graph_metric_backend: ``networkx`` (default), ``igraph``, or
            ``auto`` (igraph when the optional extra is installed).

    Returns:
        Dict[str, float]: Feature names mapped to numeric values.
    """
    if len(graph.labels) != 1:
        raise ValueError("single graph metrics require exactly one label.")

    arrays = habitat_graph_to_arrays(graph)
    features = _single_features_from_arrays(
        arrays,
        list(graph.nodes.values()),
        graph_metric_backend=graph_metric_backend,
    )
    prefix = single_feature_prefix(graph.labels[0])

    if include_extended_metrics:
        hop, _n_comp = hop_from_graph_arrays(
            arrays, largest_component=True, backend=graph_metric_backend
        )
        nx_graph = _to_networkx(graph)
        _n_components, largest = component_summary(nx_graph)
        n_nodes = nx_graph.number_of_nodes()
        extended_graph = largest if n_nodes else nx_graph
        reused_bc = hop.betweenness if hop.n_nodes > 1 else None
        features.update(
            compute_extended_graph_metrics(
                extended_graph,
                prefix,
                extended_min_nodes=extended_min_nodes,
                small_world_nrand=small_world_nrand,
                small_world_niter=small_world_niter,
                rich_club_q=rich_club_q,
                graph_null_sampler=graph_null_sampler,
                graph_null_device=graph_null_device,
                betweenness=reused_bc,
                avg_path_length=None if hop.n_nodes <= 1 else hop.avg_path_length,
            )
        )

    return features


def _nodes_for_label(graph: HabitatGraph, label: int) -> List[str]:
    """Return node ids belonging to one habitat label."""
    return [
        node_id
        for node_id, node in graph.nodes.items()
        if node.habitat_label == label
    ]


def _cross_degree_values(
    nx_graph: nx.Graph,
    node_ids: Sequence[str],
    neighbor_label: int,
) -> List[int]:
    """Count cross-label neighbors for each node."""
    values: List[int] = []
    for node_id in node_ids:
        count = 0
        for neighbor_id in nx_graph.neighbors(node_id):
            if nx_graph.nodes[neighbor_id]["habitat_label"] == neighbor_label:
                count += 1
        values.append(count)
    return values


def _local_contact_from_arrays(arrays: GraphArrays) -> List[float]:
    """Locally scaled contact on inter-class edges (same rule as HabitatGraph)."""
    if arrays.centroids.size == 0 or arrays.src.size == 0:
        return []
    ndim = int(arrays.centroids.shape[1])
    if ndim <= 0:
        return []
    exponent = (ndim - 1.0) / ndim
    values: List[float] = []
    for slot in range(arrays.src.size):
        if not bool(arrays.is_inter[slot]):
            continue
        contact = float(arrays.contact[slot])
        if not np.isfinite(contact):
            continue
        voxels_a = float(arrays.voxels[int(arrays.src[slot])])
        voxels_b = float(arrays.voxels[int(arrays.dst[slot])])
        scale = min(voxels_a, voxels_b) ** exponent
        if scale <= 0.0:
            continue
        values.append(contact / scale)
    return values


def _pairwise_features_from_arrays(
    arrays: GraphArrays,
    *,
    graph_metric_backend: str = "networkx",
) -> Dict[str, float]:
    """Default (non-extended) pairwise columns from CSR arrays."""
    if len(arrays.labels) != 2:
        raise ValueError("pairwise graph metrics require exactly two labels.")
    label_a, label_b = arrays.labels
    prefix = pair_feature_prefix(label_a, label_b)
    mask_a = arrays.habitats == int(label_a)
    mask_b = arrays.habitats == int(label_b)
    nodes_a = [arrays.node_ids[slot] for slot in np.flatnonzero(mask_a).tolist()]
    nodes_b = [arrays.node_ids[slot] for slot in np.flatnonzero(mask_b).tolist()]
    n_nodes_a = len(nodes_a)
    n_nodes_b = len(nodes_b)
    inter = arrays.is_inter
    n_edges = int(np.count_nonzero(inter))
    max_edges = n_nodes_a * n_nodes_b
    inter_dist = [
        float(value)
        for value, flag in zip(arrays.distance.tolist(), inter.tolist())
        if flag and np.isfinite(value)
    ]
    contact_values = [
        int(value)
        for value, flag in zip(arrays.contact.tolist(), inter.tolist())
        if flag and np.isfinite(value)
    ]
    local_contact_values = _local_contact_from_arrays(arrays)
    n_nodes = len(arrays.node_ids)
    indptr, indices = _csr_from_arrays(arrays)
    degrees = degrees_csr(indptr)
    cross = np.zeros(n_nodes, dtype=np.int64)
    for node in range(n_nodes):
        start = int(indptr[node])
        stop = int(indptr[node + 1])
        habitat = int(arrays.habitats[node])
        for slot in range(start, stop):
            neighbour = int(indices[slot])
            if int(arrays.habitats[neighbour]) != habitat:
                cross[node] += 1
    cross_a = cross[mask_a].astype(int).tolist()
    cross_b = cross[mask_b].astype(int).tolist()
    total_a = degrees[mask_a].astype(int).tolist()
    total_b = degrees[mask_b].astype(int).tolist()
    isolated_a = sum(1 for value in cross_a if value == 0)
    isolated_b = sum(1 for value in cross_b if value == 0)
    total_nodes = n_nodes_a + n_nodes_b
    graph_degree_scale = float(total_nodes - 1)
    avg_cross_a = _safe_mean(cross_a)
    avg_cross_b = _safe_mean(cross_b)
    avg_degree_a = _safe_mean(total_a)
    avg_degree_b = _safe_mean(total_b)
    hop_full, n_components = hop_from_graph_arrays(
        arrays, largest_component=False, backend=graph_metric_backend
    )
    hop_ok = hop_full.n_nodes > 1 and int(arrays.src.size) > 0
    weights = np.ones(int(indptr[-1]), dtype=np.float64)
    if arrays.weight.size == int(arrays.src.size):
        weight_map = {
            (int(a), int(b)): float(w)
            for a, b, w in zip(
                arrays.src.tolist(), arrays.dst.tolist(), arrays.weight.tolist()
            )
        }
        src = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr))
        dst = indices
        for slot, (node_a, node_b) in enumerate(zip(src.tolist(), dst.tolist())):
            key = (node_a, node_b) if node_a < node_b else (node_b, node_a)
            weights[slot] = weight_map.get(key, 1.0)
    features: Dict[str, float] = {
        f"{prefix}_n_nodes_1": float(n_nodes_a),
        f"{prefix}_n_nodes_2": float(n_nodes_b),
        f"{prefix}_n_edges": float(n_edges),
        f"{prefix}_edge_density": float(n_edges / max_edges) if max_edges else 0.0,
        f"{prefix}_avg_edge_distance": _safe_mean(inter_dist),
        f"{prefix}_std_edge_distance": _safe_std(inter_dist),
        f"{prefix}_contact_voxels_sum": float(sum(contact_values)),
        f"{prefix}_contact_voxels_mean": _safe_mean(contact_values),
        f"{prefix}_contact_voxels_max": float(max(contact_values)) if contact_values else 0.0,
        f"{prefix}_contact_voxels_mean_norm": _safe_mean(local_contact_values),
        f"{prefix}_contact_voxels_max_norm": (
            float(max(local_contact_values)) if local_contact_values else 0.0
        ),
        f"{prefix}_isolated_ratio_1": (
            float(isolated_a / n_nodes_a) if n_nodes_a else 0.0
        ),
        f"{prefix}_isolated_ratio_2": (
            float(isolated_b / n_nodes_b) if n_nodes_b else 0.0
        ),
        f"{prefix}_avg_{label_token(label_b)}_per_{label_token(label_a)}": avg_cross_a,
        f"{prefix}_avg_{label_token(label_b)}_per_{label_token(label_a)}_norm": (
            _safe_divide(avg_cross_a, float(n_nodes_b))
        ),
        f"{prefix}_avg_{label_token(label_a)}_per_{label_token(label_b)}": avg_cross_b,
        f"{prefix}_avg_{label_token(label_a)}_per_{label_token(label_b)}_norm": (
            _safe_divide(avg_cross_b, float(n_nodes_a))
        ),
        f"{prefix}_avg_degree_1": avg_degree_a,
        f"{prefix}_avg_degree_1_norm": _safe_divide(avg_degree_a, graph_degree_scale),
        f"{prefix}_avg_degree_2": avg_degree_b,
        f"{prefix}_avg_degree_2_norm": _safe_divide(avg_degree_b, graph_degree_scale),
        f"{prefix}_degree_cv_1": _coefficient_of_variation(total_a),
        f"{prefix}_degree_cv_2": _coefficient_of_variation(total_b),
        f"{prefix}_degree_entropy_1": _entropy(total_a),
        f"{prefix}_degree_entropy_2": _entropy(total_b),
        f"{prefix}_connected_components": float(n_components),
        f"{prefix}_connected_components_norm": _safe_divide(
            float(n_components),
            float(total_nodes),
        ),
        f"{prefix}_modularity": louvain_modularity_csr(
            indptr, indices, weights, n_nodes, backend=graph_metric_backend
        ),
    }
    if hop_ok:
        features[f"{prefix}_betweenness_mean_1"] = mean_betweenness(
            hop_full.betweenness, nodes_a
        )
        features[f"{prefix}_betweenness_mean_2"] = mean_betweenness(
            hop_full.betweenness, nodes_b
        )
        features[f"{prefix}_habitat_assortativity"] = _finite_or_zero(
            attribute_assortativity_csr(indptr, indices, arrays.habitats)
        )
    else:
        features[f"{prefix}_betweenness_mean_1"] = 0.0
        features[f"{prefix}_betweenness_mean_2"] = 0.0
        features[f"{prefix}_habitat_assortativity"] = 0.0
    return features


def calculate_pairwise_graph_metrics(
    graph: HabitatGraph,
    *,
    include_extended_metrics: bool = False,
    extended_min_nodes: int = 10,
    small_world_nrand: int = 100,
    small_world_niter: int = 100,
    rich_club_q: int = 100,
    graph_null_sampler: str = "analytic",
    graph_null_device: str = "auto",
    graph_metric_backend: str = "networkx",
) -> Dict[str, float]:
    """
    Calculate graph features for one pair of habitat labels.

    Degree statistics use two complementary bases. Interface features
    (``avg_h*_per_h*`` = R21/R12 and the isolated-node ratios) count only
    other-class neighbors, while ``avg_degree_*`` (AD1/AD2), ``degree_cv_*``,
    and ``degree_entropy_*`` use the full node degree (intra + inter edges) so
    that the average/variability/entropy triple shares one degree basis.

    Args:
        graph: Pairwise inter-habitat graph.
        include_extended_metrics: Also compute efficiency / small-world /
            rich-club / node-distribution summaries. Default False because
            those metrics dominate runtime on large graphs.
        extended_min_nodes: Minimum node count for either small-world sigma.
        small_world_nrand: Degree-preserving ensemble size for
            ``small_world_sigma_rewire``.
        small_world_niter: Rewires per edge when ``graph_null_sampler``
            is ``rewire``.
        rich_club_q: Mixing floor for the ``rewire`` sampler.
        graph_null_sampler: ``analytic`` (default Humphries ER), ``config``,
            or ``rewire``.
        graph_null_device: Batched-metric device for the null ensemble.
        graph_metric_backend: ``networkx`` (default), ``igraph``, or
            ``auto`` (igraph when the optional extra is installed).

    Returns:
        Dict[str, float]: Feature names mapped to numeric values.
    """
    if len(graph.labels) != 2:
        raise ValueError("pairwise graph metrics require exactly two labels.")

    arrays = habitat_graph_to_arrays(graph)
    features = _pairwise_features_from_arrays(
        arrays, graph_metric_backend=graph_metric_backend
    )
    label_a, label_b = graph.labels
    prefix = pair_feature_prefix(label_a, label_b)

    if include_extended_metrics:
        hop_full, n_components = hop_from_graph_arrays(
            arrays, largest_component=False, backend=graph_metric_backend
        )
        nx_graph = _to_networkx(graph)
        nodes_a = _nodes_for_label(graph, label_a)
        nodes_b = _nodes_for_label(graph, label_b)
        features.update(
            compute_extended_pairwise_metrics(
                nx_graph,
                nodes_a,
                nodes_b,
                prefix,
                extended_min_nodes=extended_min_nodes,
                small_world_nrand=small_world_nrand,
                small_world_niter=small_world_niter,
                rich_club_q=rich_club_q,
                graph_null_sampler=graph_null_sampler,
                graph_null_device=graph_null_device,
                betweenness=None if hop_full.n_nodes <= 1 else hop_full.betweenness,
                avg_path_length=(
                    hop_full.avg_path_length
                    if hop_full.n_nodes > 1 and n_components == 1
                    else None
                ),
            )
        )

    return features
