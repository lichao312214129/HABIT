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
from typing import Dict, Iterable, List, Sequence
import warnings

import networkx as nx
from networkx.algorithms import community as nx_community
import numpy as np
from scipy.spatial import cKDTree

from habit.kernels.habitat_graph.extended_metrics import (
    compute_extended_graph_metrics,
    compute_extended_pairwise_metrics,
)
from habit.kernels.habitat_graph.models import (
    HabitatGraph,
    HabitatGraphEdge,
    HabitatGraphNode,
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


def _largest_component(nx_graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a subgraph copy."""
    if nx_graph.number_of_nodes() == 0:
        return nx_graph.copy()
    largest_nodes = max(nx.connected_components(nx_graph), key=len)
    return nx_graph.subgraph(largest_nodes).copy()


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


def _modularity(nx_graph: nx.Graph) -> float:
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


def calculate_single_graph_metrics(
    graph: HabitatGraph,
    *,
    include_extended_metrics: bool = True,
    extended_min_nodes: int = 10,
) -> Dict[str, float]:
    """
    Calculate graph features for one habitat label.

    Args:
        graph: Single-habitat graph.
        include_extended_metrics: Also compute efficiency / small-world /
            rich-club / node-distribution summaries.
        extended_min_nodes: Minimum node count for the small-world sigma.

    Returns:
        Dict[str, float]: Feature names mapped to numeric values.
    """
    if len(graph.labels) != 1:
        raise ValueError("single graph metrics require exactly one label.")

    label = graph.labels[0]
    prefix = single_feature_prefix(label)
    nx_graph = _to_networkx(graph)
    n_nodes = nx_graph.number_of_nodes()
    n_edges = nx_graph.number_of_edges()
    max_edges = n_nodes * (n_nodes - 1) / 2
    degree_scale = float(n_nodes - 1)
    degrees = [int(degree) for _, degree in nx_graph.degree()]
    edge_distances = _edge_distances(graph)
    node_voxels = [float(node.voxel_count) for node in graph.nodes.values()]

    features: Dict[str, float] = {
        f"{prefix}_n_nodes": float(n_nodes),
        f"{prefix}_n_edges": float(n_edges),
        f"{prefix}_edge_density": float(n_edges / max_edges) if max_edges else 0.0,
        f"{prefix}_connected_components": (
            float(nx.number_connected_components(nx_graph)) if n_nodes else 0.0
        ),
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
        f"{prefix}_spatial_dispersion": _spatial_dispersion(graph.nodes.values()),
        f"{prefix}_connected_components_ratio": (
            float(nx.number_connected_components(nx_graph) / n_nodes)
            if n_nodes
            else 0.0
        ),
        f"{prefix}_nearest_neighbor_ratio": _nearest_neighbor_ratio(
            graph.nodes.values()
        ),
        f"{prefix}_modularity": _modularity(nx_graph),
    }

    if n_nodes:
        largest = _largest_component(nx_graph)
        features[f"{prefix}_largest_component_ratio"] = (
            float(largest.number_of_nodes() / n_nodes)
        )
        features[f"{prefix}_avg_clustering"] = float(nx.average_clustering(nx_graph))
    else:
        largest = nx_graph
        features[f"{prefix}_largest_component_ratio"] = 0.0
        features[f"{prefix}_avg_clustering"] = 0.0

    if largest.number_of_nodes() > 1:
        path_scale = float(largest.number_of_nodes() - 1)
        features[f"{prefix}_avg_path_length"] = float(
            nx.average_shortest_path_length(largest)
        )
        features[f"{prefix}_diameter"] = float(nx.diameter(largest))
        # Path length and graph diameter are hop counts, so they are normalized
        # by the largest possible hop distance inside the component, not by
        # physical VOI size.
        features[f"{prefix}_avg_path_length_norm"] = _safe_divide(
            features[f"{prefix}_avg_path_length"],
            path_scale,
        )
        features[f"{prefix}_diameter_norm"] = _safe_divide(
            features[f"{prefix}_diameter"],
            path_scale,
        )
        betweenness = nx.betweenness_centrality(largest)
        closeness = nx.closeness_centrality(largest)
        features[f"{prefix}_avg_betweenness"] = _safe_mean(list(betweenness.values()))
        features[f"{prefix}_avg_closeness"] = _safe_mean(list(closeness.values()))
    else:
        features[f"{prefix}_avg_path_length"] = 0.0
        features[f"{prefix}_diameter"] = 0.0
        features[f"{prefix}_avg_path_length_norm"] = 0.0
        features[f"{prefix}_diameter_norm"] = 0.0
        features[f"{prefix}_avg_betweenness"] = 0.0
        features[f"{prefix}_avg_closeness"] = 0.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            features[f"{prefix}_degree_assortativity"] = _finite_or_zero(
                nx.degree_assortativity_coefficient(nx_graph)
            )
    except Exception:
        features[f"{prefix}_degree_assortativity"] = 0.0

    if include_extended_metrics:
        extended_graph = largest if n_nodes else nx_graph
        features.update(
            compute_extended_graph_metrics(
                extended_graph,
                prefix,
                extended_min_nodes=extended_min_nodes,
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


def calculate_pairwise_graph_metrics(
    graph: HabitatGraph,
    *,
    include_extended_metrics: bool = True,
    extended_min_nodes: int = 10,
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
            rich-club / node-distribution summaries.
        extended_min_nodes: Minimum node count for the small-world sigma.

    Returns:
        Dict[str, float]: Feature names mapped to numeric values.
    """
    if len(graph.labels) != 2:
        raise ValueError("pairwise graph metrics require exactly two labels.")

    label_a, label_b = graph.labels
    prefix = pair_feature_prefix(label_a, label_b)
    # Full graph (inter + optional intra edges) drives whole-graph metrics such
    # as modularity, class assortativity, betweenness, and components.
    nx_graph = _to_networkx(graph)
    nodes_a = _nodes_for_label(graph, label_a)
    nodes_b = _nodes_for_label(graph, label_b)
    n_nodes_a = len(nodes_a)
    n_nodes_b = len(nodes_b)
    # Interface metrics use inter-class edges only, matching the source.
    inter_edges = _inter_edges(graph)
    n_edges = len(inter_edges)
    max_edges = n_nodes_a * n_nodes_b
    edge_distances = [
        float(edge.distance) for edge in inter_edges if edge.distance is not None
    ]
    contact_values = _contact_voxels(inter_edges)
    local_contact_values = _local_contact_scales(graph, inter_edges)
    # Cross degree counts only other-class neighbors; it drives the interface
    # metrics R21/R12 and the isolated-node ratios.
    cross_degrees_a = _cross_degree_values(nx_graph, nodes_a, label_b)
    cross_degrees_b = _cross_degree_values(nx_graph, nodes_b, label_a)
    # Total degree counts every neighbor (intra + inter edges). It backs the
    # AD/CV/EN degree-statistics triple so all three share one degree basis,
    # matching the Table S3 wording "neighbors per region".
    total_degrees_a = [int(degree) for _, degree in nx_graph.degree(nodes_a)]
    total_degrees_b = [int(degree) for _, degree in nx_graph.degree(nodes_b)]

    isolated_a = sum(1 for value in cross_degrees_a if value == 0)
    isolated_b = sum(1 for value in cross_degrees_b if value == 0)
    total_nodes = n_nodes_a + n_nodes_b
    graph_degree_scale = float(total_nodes - 1)
    avg_cross_a = _safe_mean(cross_degrees_a)
    avg_cross_b = _safe_mean(cross_degrees_b)
    avg_degree_a = _safe_mean(total_degrees_a)
    avg_degree_b = _safe_mean(total_degrees_b)
    connected_components = (
        float(nx.number_connected_components(nx_graph))
        if nx_graph.number_of_nodes()
        else 0.0
    )

    features: Dict[str, float] = {
        f"{prefix}_n_nodes_1": float(n_nodes_a),
        f"{prefix}_n_nodes_2": float(n_nodes_b),
        f"{prefix}_n_edges": float(n_edges),
        f"{prefix}_edge_density": float(n_edges / max_edges) if max_edges else 0.0,
        f"{prefix}_avg_edge_distance": _safe_mean(edge_distances),
        f"{prefix}_std_edge_distance": _safe_std(edge_distances),
        f"{prefix}_contact_voxels_sum": float(sum(contact_values)),
        f"{prefix}_contact_voxels_mean": _safe_mean(contact_values),
        f"{prefix}_contact_voxels_max": float(max(contact_values)) if contact_values else 0.0,
        # Sum is normalized later by whole-ROI area scale. Mean / max are
        # local edge summaries, so use each edge's smaller-node area scale.
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
        # R21/R12: average number of other-class neighbors per node (cross degree).
        f"{prefix}_avg_h{label_b}_per_h{label_a}": avg_cross_a,
        f"{prefix}_avg_h{label_b}_per_h{label_a}_norm": _safe_divide(
            avg_cross_a,
            float(n_nodes_b),
        ),
        f"{prefix}_avg_h{label_a}_per_h{label_b}": avg_cross_b,
        f"{prefix}_avg_h{label_a}_per_h{label_b}_norm": _safe_divide(
            avg_cross_b,
            float(n_nodes_a),
        ),
        # AD1/AD2: average total degree (intra + inter neighbors) per class.
        f"{prefix}_avg_degree_1": avg_degree_a,
        f"{prefix}_avg_degree_1_norm": _safe_divide(
            avg_degree_a,
            graph_degree_scale,
        ),
        f"{prefix}_avg_degree_2": avg_degree_b,
        f"{prefix}_avg_degree_2_norm": _safe_divide(
            avg_degree_b,
            graph_degree_scale,
        ),
        # CV/EN are computed on the total-degree distribution so the AD/CV/EN
        # triple is internally consistent.
        f"{prefix}_degree_cv_1": _coefficient_of_variation(total_degrees_a),
        f"{prefix}_degree_cv_2": _coefficient_of_variation(total_degrees_b),
        f"{prefix}_degree_entropy_1": _entropy(total_degrees_a),
        f"{prefix}_degree_entropy_2": _entropy(total_degrees_b),
        f"{prefix}_connected_components": connected_components,
        f"{prefix}_connected_components_norm": _safe_divide(
            connected_components,
            float(total_nodes),
        ),
        f"{prefix}_modularity": _modularity(nx_graph),
    }

    if nx_graph.number_of_nodes() > 1 and nx_graph.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(nx_graph)
        features[f"{prefix}_betweenness_mean_1"] = _safe_mean(
            [float(betweenness[node_id]) for node_id in nodes_a]
        )
        features[f"{prefix}_betweenness_mean_2"] = _safe_mean(
            [float(betweenness[node_id]) for node_id in nodes_b]
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                features[f"{prefix}_habitat_assortativity"] = _finite_or_zero(
                    nx.attribute_assortativity_coefficient(nx_graph, "habitat_label")
                )
        except Exception:
            features[f"{prefix}_habitat_assortativity"] = 0.0
    else:
        features[f"{prefix}_betweenness_mean_1"] = 0.0
        features[f"{prefix}_betweenness_mean_2"] = 0.0
        features[f"{prefix}_habitat_assortativity"] = 0.0

    if include_extended_metrics:
        features.update(
            compute_extended_pairwise_metrics(
                nx_graph,
                nodes_a,
                nodes_b,
                prefix,
                extended_min_nodes=extended_min_nodes,
            )
        )

    return features
