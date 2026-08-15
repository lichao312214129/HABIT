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
"""Extended graph metrics: efficiency, small-world, rich-club, node distributions."""

from __future__ import annotations

from typing import Dict, List, Sequence
import warnings

import networkx as nx
import numpy as np
from scipy.stats import skew

__all__ = [
    "compute_extended_graph_metrics",
    "compute_extended_pairwise_metrics",
]


def _finite_or_zero(value: float) -> float:
    """Return finite numeric values and replace NaN/Inf with zero."""
    return float(value) if np.isfinite(value) else 0.0


def _copy_normalized_betweenness(value: float, n_nodes: int) -> float:
    """
    Reuse NetworkX-normalized betweenness; do not divide again.

    NetworkX ``betweenness_centrality(..., normalized=True)`` already
    divides raw pair-counts by ``(n-1)*(n-2)/2`` on undirected graphs,
    so the values lie in ``[0, 1]`` and are comparable across graph
    sizes. Companions named ``*_norm`` must copy that value. Dividing
    by the same factor a second time is double normalization and is
    incorrect.

    Args:
        value: Already-normalized betweenness summary (max or std).
        n_nodes: Node count of the graph used to compute that betweenness.

    Returns:
        float: ``value`` when ``n_nodes >= 3`` (betweenness defined);
        otherwise ``0.0``.
    """
    if n_nodes < 3:
        return 0.0
    return float(value)


def _analysis_subgraph(nx_graph: nx.Graph) -> nx.Graph:
    """
    Return the graph used for extended integration metrics.

    When disconnected, the largest connected component is used so efficiency
    and small-world estimates are not dominated by isolated nodes.
    """
    if nx_graph.number_of_nodes() == 0:
        return nx_graph
    if nx_graph.number_of_edges() == 0:
        return nx_graph
    if nx.is_connected(nx_graph):
        return nx_graph
    largest_nodes = max(nx.connected_components(nx_graph), key=len)
    return nx_graph.subgraph(largest_nodes).copy()


def _node_local_efficiency_values(nx_graph: nx.Graph) -> List[float]:
    """
    Compute per-node local efficiency values.

    For each node, local efficiency is the global efficiency of the subgraph
    induced by its neighbors.

    Args:
        nx_graph: NetworkX graph.

    Returns:
        List[float]: One local efficiency value per node.
    """
    values: List[float] = []
    for node in nx_graph.nodes():
        neighbors = list(nx_graph.neighbors(node))
        if len(neighbors) < 2:
            values.append(0.0)
            continue
        subgraph = nx_graph.subgraph(neighbors)
        if subgraph.number_of_edges() == 0:
            values.append(0.0)
            continue
        try:
            values.append(float(nx.global_efficiency(subgraph)))
        except Exception:
            values.append(0.0)
    return values


def _rich_club_coefficient_mean(nx_graph: nx.Graph) -> float:
    """
    Summarize the normalized rich-club curve as a single scalar.

    Args:
        nx_graph: NetworkX graph.

    Returns:
        float: Mean finite normalized rich-club coefficient across degree bins.
    """
    if nx_graph.number_of_edges() == 0:
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coefficients = nx.rich_club_coefficient(nx_graph, normalized=True)
        finite = [
            float(value)
            for value in coefficients.values()
            if np.isfinite(value)
        ]
        return float(np.mean(finite)) if finite else 0.0
    except Exception:
        return 0.0


def _small_world_sigma(
    nx_graph: nx.Graph,
    min_nodes: int,
    seed: int = 0,
) -> float:
    """
    Compute the small-world sigma coefficient relative to random graphs.

    Args:
        nx_graph: NetworkX graph (should be connected for stable estimates).
        min_nodes: Minimum node count required; smaller graphs return 0.
        seed: Random seed for the null-model comparison.

    Returns:
        float: Small-world sigma; 0 when undefined or below ``min_nodes``.
    """
    n_nodes = nx_graph.number_of_nodes()
    if n_nodes < min_nodes or nx_graph.number_of_edges() < 1:
        return 0.0
    if not nx.is_connected(nx_graph):
        return 0.0
    try:
        from networkx.algorithms import smallworld as nx_smallworld

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sigma = nx_smallworld.sigma(
                nx_graph,
                niter=5,
                nrand=3,
                seed=seed,
            )
        return _finite_or_zero(float(sigma))
    except Exception:
        return 0.0


def _append_betweenness_distribution(
    features: Dict[str, float],
    prefix: str,
    nx_graph: nx.Graph,
) -> None:
    """
    Add graph-level betweenness distribution summaries.

    ``betweenness_max`` and ``betweenness_std`` use NetworkX's default
    normalized betweenness (already in ``[0, 1]``). The ``*_norm``
    companions copy those values when ``n >= 3``; they are not divided
    by ``(n-1)*(n-2)/2`` a second time.

    Args:
        features: Feature dictionary updated in place.
        prefix: Column prefix (e.g. ``single_h1``).
        nx_graph: Graph on which betweenness is computed.
    """
    n_nodes = nx_graph.number_of_nodes()
    if n_nodes > 1 and nx_graph.number_of_edges() > 0:
        try:
            betweenness = nx.betweenness_centrality(nx_graph)
            bc_values = list(betweenness.values())
            bc_max = float(max(bc_values)) if bc_values else 0.0
            bc_std = float(np.std(bc_values)) if bc_values else 0.0
        except Exception:
            bc_max = 0.0
            bc_std = 0.0
    else:
        bc_max = 0.0
        bc_std = 0.0

    features[f"{prefix}_betweenness_max"] = bc_max
    features[f"{prefix}_betweenness_std"] = bc_std
    features[f"{prefix}_betweenness_max_norm"] = _copy_normalized_betweenness(
        bc_max, n_nodes
    )
    features[f"{prefix}_betweenness_std_norm"] = _copy_normalized_betweenness(
        bc_std, n_nodes
    )


def compute_extended_graph_metrics(
    nx_graph: nx.Graph,
    prefix: str,
    *,
    extended_min_nodes: int = 10,
) -> Dict[str, float]:
    """
    Compute extended graph metrics for a single-habitat or full pairwise graph.

    Args:
        nx_graph: NetworkX graph converted from a habitat graph.
        prefix: Feature name prefix.
        extended_min_nodes: Minimum nodes required for small-world sigma.

    Returns:
        Dict[str, float]: Extended feature columns (dimensionless scalars).
    """
    features: Dict[str, float] = {}
    analysis_graph = _analysis_subgraph(nx_graph)
    n_analysis = analysis_graph.number_of_nodes()

    if n_analysis >= 2 and analysis_graph.number_of_edges() > 0:
        try:
            features[f"{prefix}_global_efficiency"] = float(
                nx.global_efficiency(analysis_graph)
            )
        except Exception:
            features[f"{prefix}_global_efficiency"] = 0.0
        try:
            features[f"{prefix}_local_efficiency"] = float(
                nx.local_efficiency(analysis_graph)
            )
        except Exception:
            features[f"{prefix}_local_efficiency"] = 0.0
        features[f"{prefix}_small_world_sigma"] = _small_world_sigma(
            analysis_graph,
            min_nodes=extended_min_nodes,
        )
        features[f"{prefix}_rich_club_coefficient"] = _rich_club_coefficient_mean(
            analysis_graph
        )
    else:
        features[f"{prefix}_global_efficiency"] = 0.0
        features[f"{prefix}_local_efficiency"] = 0.0
        features[f"{prefix}_small_world_sigma"] = 0.0
        features[f"{prefix}_rich_club_coefficient"] = 0.0

    _append_betweenness_distribution(features, prefix, analysis_graph)

    degrees = [int(degree) for _, degree in analysis_graph.degree()]
    if len(degrees) >= 3:
        features[f"{prefix}_degree_skewness"] = float(skew(degrees, bias=False))
    else:
        features[f"{prefix}_degree_skewness"] = 0.0

    local_eff_values = _node_local_efficiency_values(analysis_graph)
    if local_eff_values:
        features[f"{prefix}_node_local_efficiency_min"] = float(min(local_eff_values))
        features[f"{prefix}_node_local_efficiency_std"] = float(
            np.std(local_eff_values)
        )
    else:
        features[f"{prefix}_node_local_efficiency_min"] = 0.0
        features[f"{prefix}_node_local_efficiency_std"] = 0.0

    return features


def compute_extended_pairwise_metrics(
    nx_graph: nx.Graph,
    nodes_a: Sequence[str],
    nodes_b: Sequence[str],
    prefix: str,
    *,
    extended_min_nodes: int = 10,
) -> Dict[str, float]:
    """
    Compute extended metrics for a pairwise habitat graph.

    Whole-graph efficiency/small-world/rich-club are computed on the full graph.
    Class-specific hub summaries use per-class betweenness maxima and degree
    skewness.

    Args:
        nx_graph: Full pairwise NetworkX graph (intra + inter edges).
        nodes_a: Node ids belonging to habitat *a* (suffix ``_1``).
        nodes_b: Node ids belonging to habitat *b* (suffix ``_2``).
        prefix: Feature name prefix.
        extended_min_nodes: Minimum nodes required for small-world sigma.

    Returns:
        Dict[str, float]: Extended pairwise feature columns.
    """
    features = compute_extended_graph_metrics(
        nx_graph,
        prefix,
        extended_min_nodes=extended_min_nodes,
    )
    features.pop(f"{prefix}_degree_skewness", None)

    n_total = nx_graph.number_of_nodes()
    betweenness: Dict[str, float] = {}
    if n_total > 1 and nx_graph.number_of_edges() > 0:
        try:
            betweenness = nx.betweenness_centrality(nx_graph)
        except Exception:
            betweenness = {}

    for suffix, node_ids in (("_1", nodes_a), ("_2", nodes_b)):
        degrees = [int(nx_graph.degree(node_id)) for node_id in node_ids]
        skew_key = f"{prefix}_degree_skewness{suffix}"
        if len(degrees) >= 3:
            features[skew_key] = float(skew(degrees, bias=False))
        else:
            features[skew_key] = 0.0

        bc_values = [
            float(betweenness[node_id])
            for node_id in node_ids
            if node_id in betweenness
        ]
        bc_max = max(bc_values) if bc_values else 0.0
        features[f"{prefix}_betweenness_max{suffix}"] = bc_max
        # Full-graph n decides whether betweenness is defined; the value
        # itself is already NetworkX-normalized on that same full graph.
        features[f"{prefix}_betweenness_max{suffix}_norm"] = (
            _copy_normalized_betweenness(bc_max, n_total)
        )

    return features
