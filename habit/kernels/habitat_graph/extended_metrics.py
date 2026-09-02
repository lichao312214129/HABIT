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

from typing import Dict, List, Literal, Optional, Sequence, Tuple
import warnings

import networkx as nx
import numpy as np
from scipy.stats import skew

from habit.kernels.habitat_graph.null_ensemble import (
    GraphNullSampler,
    adjacency_from_undirected,
    ensemble_null_summaries,
    global_efficiency_value,
    local_efficiency_values,
)
from habit.kernels.habitat_graph.traversal import (
    betweenness_max_std,
    hop_metrics,
)

# Feature-level sampler: Humphries analytic ER is the default point
# estimate. ``config`` / ``rewire`` replace that one column with a
# degree-preserving ensemble (they are not extra columns).
SmallWorldNull = Literal["analytic", "config", "rewire"]

__all__ = [
    "compute_extended_graph_metrics",
    "compute_extended_pairwise_metrics",
]


def _finite_or_zero(value: float) -> float:
    """Return finite numeric values and replace NaN/Inf with zero."""
    return float(value) if np.isfinite(value) else 0.0


def _safe_skew(values: Sequence[float]) -> float:
    """
    Sample skewness, silently treating near-constant samples as 0.

    ``scipy.stats.skew`` emits ``RuntimeWarning: Precision loss occurred in
    moment calculation`` when every degree is (almost) identical -- common
    on regular habitat lattices. The warning is not actionable and floods
    the progress bar; the numeric result is already unreliable, so we
    return 0.0 instead of printing.

    Args:
        values: Degree (or other) samples. Skew is defined for ``n >= 3``.

    Returns:
        Finite skewness, or ``0.0`` when undefined / non-finite.
    """
    if len(values) < 3:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(skew(values, bias=False))
    return _finite_or_zero(value)


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


def _observed_clustering_and_path(
    nx_graph: nx.Graph,
    path_length: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """
    Transitivity and mean shortest-path length of a connected graph.

    Both small-world variants share this observed pair so they differ only
    in the null model (Humphries ER formulas vs degree-preserving rewires).
    Transitivity matches NetworkX ``smallworld.sigma``. When ``path_length``
    is already known from the same connected graph's hop sweep, skip the
    extra all-sources BFS.

    Args:
        nx_graph: Connected undirected graph.
        path_length: Optional ASPL already measured on ``nx_graph``.

    Returns:
        ``(C, L)`` or ``None`` when either quantity is undefined.
    """
    try:
        clustering = float(nx.transitivity(nx_graph))
        if path_length is None:
            path_length = float(nx.average_shortest_path_length(nx_graph))
        else:
            path_length = float(path_length)
    except Exception:
        return None
    if clustering <= 0.0 or path_length <= 0.0:
        return None
    if not np.isfinite(clustering) or not np.isfinite(path_length):
        return None
    return clustering, path_length


def _sigma_from_ratios(
    clustering: float,
    path_length: float,
    clustering_rand: float,
    path_rand: float,
) -> float:
    """Humphries sigma = (C / C_rand) / (L / L_rand)."""
    if clustering_rand <= 0.0 or path_rand <= 0.0:
        return 0.0
    if not np.isfinite(clustering_rand) or not np.isfinite(path_rand):
        return 0.0
    return _finite_or_zero(
        (clustering / clustering_rand) / (path_length / path_rand)
    )


def _small_world_sigma_er(
    nx_graph: nx.Graph,
    min_nodes: int,
    observed: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Humphries small-world-ness against an Erdős–Rényi null.

    Uses the analytic ER approximations from Humphries and Gurney
    (2008): ``C_rand ≈ ⟨k⟩ / n`` and ``L_rand ≈ ln(n) / ln(⟨k⟩)``.
    This is the point-estimate form of their *S* (they draw 1000 ER
    graphs only when testing borderline ``1 ≤ S ≤ 3``). It does **not**
    preserve the degree sequence.

    Args:
        nx_graph: Connected analysis subgraph.
        min_nodes: Return 0 below this node count.
        observed: Optional precomputed ``(C, L)`` of ``nx_graph``.

    Returns:
        Finite sigma, or ``0.0`` when undefined.
    """
    n_nodes = nx_graph.number_of_nodes()
    n_edges = nx_graph.number_of_edges()
    if n_nodes < min_nodes or n_edges < 1:
        return 0.0
    if not nx.is_connected(nx_graph):
        return 0.0
    pair = observed if observed is not None else _observed_clustering_and_path(
        nx_graph
    )
    if pair is None:
        return 0.0
    clustering, path_length = pair
    mean_degree = 2.0 * float(n_edges) / float(n_nodes)
    if mean_degree <= 1.0:
        return 0.0
    clustering_rand = mean_degree / float(n_nodes)
    path_rand = float(np.log(n_nodes) / np.log(mean_degree))
    return _sigma_from_ratios(
        clustering, path_length, clustering_rand, path_rand
    )


def _degree_preserving_summaries(
    nx_graph: nx.Graph,
    *,
    nrand: int,
    niter: int,
    sampler: GraphNullSampler,
    device: str,
    seed: int,
) -> Optional[Tuple[float, float, float]]:
    """
    Shared degree-preserving ensemble for sigma and rich-club.

    Args:
        nx_graph: Connected analysis subgraph.
        nrand: Requested null-graph count (default 100).
        niter: Swaps per edge when ``sampler='rewire'`` (default 100).
        sampler: ``config`` (fast default) or ``rewire``.
        device: Batched-metric device (``auto`` / ``cpu`` / ``cuda``).
        seed: Ensemble seed.

    Returns:
        ``(C_rand, L_rand, rich_club_mean)`` or ``None``.
    """
    adj, _ = adjacency_from_undirected(nx_graph)
    summary = ensemble_null_summaries(
        adj,
        nrand=nrand,
        sampler=sampler,
        niter=niter,
        device=device,
        seed=seed,
    )
    if summary is None:
        return None
    clustering_rand, path_rand, rich_club, _n_accepted = summary
    return clustering_rand, path_rand, rich_club


def _small_world_sigma_rewire(
    nx_graph: nx.Graph,
    min_nodes: int,
    *,
    nrand: int,
    niter: int,
    seed: int = 0,
    sampler: GraphNullSampler = "config",
    device: str = "auto",
    observed: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Small-world sigma against a degree-preserving null ensemble.

    Same Humphries ratio as :func:`_small_world_sigma_er`, but
    ``C_rand`` and ``L_rand`` are means over ``nrand`` graphs that
    keep the observed degree sequence. Default sampler is the
    configuration model. Pass ``sampler='rewire'`` for Maslov–Sneppen
    mixing (``niter`` swaps per edge). This is **not** the analytic
    ER *S*.

    Args:
        nx_graph: Connected analysis subgraph.
        min_nodes: Return 0 below this node count.
        nrand: Number of accepted null graphs in the ensemble.
        niter: Rewires per edge when ``sampler='rewire'``.
        seed: Ensemble seed.
        sampler: ``config`` (default) or ``rewire``.
        device: Batched-metric device.
        observed: Optional precomputed ``(C, L)`` of ``nx_graph``.

    Returns:
        Finite sigma, or ``0.0`` when fewer than one null graph is usable.
    """
    n_nodes = nx_graph.number_of_nodes()
    if n_nodes < min_nodes or nx_graph.number_of_edges() < 1:
        return 0.0
    if not nx.is_connected(nx_graph):
        return 0.0
    pair = observed if observed is not None else _observed_clustering_and_path(
        nx_graph
    )
    if pair is None:
        return 0.0
    clustering, path_length = pair
    summary = _degree_preserving_summaries(
        nx_graph,
        nrand=nrand,
        niter=niter,
        sampler=sampler,
        device=device,
        seed=seed,
    )
    if summary is None:
        return 0.0
    clustering_rand, path_rand, _rich_club = summary
    return _sigma_from_ratios(
        clustering, path_length, clustering_rand, path_rand
    )


def _append_betweenness_distribution(
    features: Dict[str, float],
    prefix: str,
    nx_graph: nx.Graph,
    betweenness: Optional[Dict[str, float]] = None,
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
        betweenness: Optional precomputed NetworkX-normalized betweenness
            on ``nx_graph``. When omitted, one Brandes sweep is run.
    """
    n_nodes = nx_graph.number_of_nodes()
    if n_nodes > 1 and nx_graph.number_of_edges() > 0:
        try:
            values = betweenness
            if values is None or any(node not in values for node in nx_graph):
                values = hop_metrics(nx_graph).betweenness
            bc_max, bc_std = betweenness_max_std(values, list(nx_graph.nodes()))
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
    small_world_nrand: int = 100,
    small_world_niter: int = 100,
    rich_club_q: int = 100,
    graph_null_sampler: SmallWorldNull = "analytic",
    graph_null_device: str = "auto",
    betweenness: Optional[Dict[str, float]] = None,
    avg_path_length: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute extended graph metrics for a single-habitat or full pairwise graph.

    Small-world is **one** column ``small_world_sigma``. The default
    ``graph_null_sampler='analytic'`` is Humphries' Erdős–Rényi *S*
    (analytic ``C_rand``, ``L_rand``). Pass ``config`` or ``rewire`` to
    replace that column with a degree-preserving ensemble -- do not
    emit both.

    Rich-club ``φ_rand`` uses one configuration-model graph under the
    analytic default (NetworkX / Milo point estimate). ``config`` /
    ``rewire`` reuse the same ensemble as sigma.

    Args:
        nx_graph: NetworkX graph converted from a habitat graph.
        prefix: Feature name prefix.
        extended_min_nodes: Minimum nodes required for small-world sigma.
        small_world_nrand: Ensemble size when sampler is ``config``/``rewire``.
        small_world_niter: Rewires per edge when sampler is ``rewire``.
        rich_club_q: Mixing floor for the ``rewire`` sampler.
        graph_null_sampler: ``analytic`` (default), ``config``, or ``rewire``.
        graph_null_device: Batched-metric device for an ensemble.
        betweenness: Optional Brandes values already computed on the
            analysis graph (same node set). Avoids a second sweep.
        avg_path_length: Optional mean hop distance on the same analysis
            graph. Reused for Humphries sigma so small-world does not
            run a second all-sources BFS.

    Returns:
        Dict[str, float]: Extended feature columns (dimensionless scalars).
    """
    features: Dict[str, float] = {}
    analysis_graph = _analysis_subgraph(nx_graph)
    n_analysis = analysis_graph.number_of_nodes()
    rewire_niter = max(int(small_world_niter), int(rich_club_q))
    sampler = str(graph_null_sampler)
    if sampler not in ("analytic", "config", "rewire"):
        raise ValueError(
            "graph_null_sampler must be 'analytic', 'config', or 'rewire'; "
            f"got {graph_null_sampler!r}."
        )

    if n_analysis >= 2 and analysis_graph.number_of_edges() > 0:
        analysis_adj, _analysis_nodes = adjacency_from_undirected(analysis_graph)
        try:
            features[f"{prefix}_global_efficiency"] = global_efficiency_value(
                analysis_adj
            )
        except Exception:
            try:
                features[f"{prefix}_global_efficiency"] = float(
                    nx.global_efficiency(analysis_graph)
                )
            except Exception:
                features[f"{prefix}_global_efficiency"] = 0.0
        try:
            local_eff_values = local_efficiency_values(analysis_adj)
        except Exception:
            local_eff_values = _node_local_efficiency_values(analysis_graph)
        features[f"{prefix}_local_efficiency"] = (
            float(np.mean(local_eff_values)) if local_eff_values else 0.0
        )
        reuse_path = (
            avg_path_length
            if avg_path_length is not None
            and betweenness is not None
            and n_analysis == len(betweenness)
            else None
        )
        observed = (
            _observed_clustering_and_path(analysis_graph, path_length=reuse_path)
            if n_analysis >= extended_min_nodes and nx.is_connected(analysis_graph)
            else None
        )
        use_ensemble = sampler in ("config", "rewire")
        null_summary = (
            _degree_preserving_summaries(
                analysis_graph,
                nrand=small_world_nrand if use_ensemble else 1,
                niter=rewire_niter,
                sampler="config" if sampler == "analytic" else sampler,
                device=graph_null_device,
                seed=0,
            )
            if n_analysis >= 4
            and nx.is_connected(analysis_graph)
            and analysis_graph.number_of_edges() >= 2
            else None
        )
        if sampler == "analytic":
            features[f"{prefix}_small_world_sigma"] = _small_world_sigma_er(
                analysis_graph,
                min_nodes=extended_min_nodes,
                observed=observed,
            )
        elif observed is not None and null_summary is not None:
            clustering, path_length = observed
            clustering_rand, path_rand, _rich_club = null_summary
            features[f"{prefix}_small_world_sigma"] = _sigma_from_ratios(
                clustering, path_length, clustering_rand, path_rand
            )
        else:
            features[f"{prefix}_small_world_sigma"] = 0.0
        features[f"{prefix}_rich_club_coefficient"] = (
            _finite_or_zero(null_summary[2]) if null_summary is not None else 0.0
        )
    else:
        features[f"{prefix}_global_efficiency"] = 0.0
        features[f"{prefix}_local_efficiency"] = 0.0
        features[f"{prefix}_small_world_sigma"] = 0.0
        features[f"{prefix}_rich_club_coefficient"] = 0.0
        local_eff_values = _node_local_efficiency_values(analysis_graph)

    analysis_betweenness = None
    if betweenness is not None and n_analysis > 0:
        analysis_ids = set(analysis_graph.nodes())
        if analysis_ids.issubset(betweenness) and len(analysis_ids) == len(betweenness):
            analysis_betweenness = betweenness
    _append_betweenness_distribution(
        features, prefix, analysis_graph, betweenness=analysis_betweenness
    )

    degrees = [int(degree) for _, degree in analysis_graph.degree()]
    features[f"{prefix}_degree_skewness"] = _safe_skew(degrees)

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
    small_world_nrand: int = 100,
    small_world_niter: int = 100,
    rich_club_q: int = 100,
    graph_null_sampler: SmallWorldNull = "analytic",
    graph_null_device: str = "auto",
    betweenness: Optional[Dict[str, float]] = None,
    avg_path_length: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute extended metrics for a pairwise habitat graph.

    Whole-graph efficiency/small-world/rich-club are computed on the full graph.
    Class-specific hub summaries use per-class betweenness maxima (full-graph
    paths) and degree skewness of *cross-class* degree.

    Args:
        nx_graph: Full pairwise NetworkX graph (intra + inter edges).
        nodes_a: Node ids belonging to habitat *a* (suffix ``_1``).
        nodes_b: Node ids belonging to habitat *b* (suffix ``_2``).
        prefix: Feature name prefix.
        extended_min_nodes: Minimum nodes required for either sigma.
        small_world_nrand: Degree-preserving ensemble size.
        small_world_niter: Rewires per edge when sampler is ``rewire``.
        rich_club_q: Mixing floor for the ``rewire`` sampler.
        graph_null_sampler: ``analytic`` (default), ``config``, or ``rewire``.
        graph_null_device: Batched-metric device.
        betweenness: Optional Brandes values on the full pairwise graph.
        avg_path_length: Optional ASPL on the same full graph. Only
            reuse it when that graph is connected (the analysis graph
            is then the full graph).

    Returns:
        Dict[str, float]: Extended pairwise feature columns.
    """
    features = compute_extended_graph_metrics(
        nx_graph,
        prefix,
        extended_min_nodes=extended_min_nodes,
        small_world_nrand=small_world_nrand,
        small_world_niter=small_world_niter,
        rich_club_q=rich_club_q,
        graph_null_sampler=graph_null_sampler,
        graph_null_device=graph_null_device,
        betweenness=betweenness,
        avg_path_length=avg_path_length,
    )
    features.pop(f"{prefix}_degree_skewness", None)

    n_total = nx_graph.number_of_nodes()
    if betweenness is None and n_total > 1 and nx_graph.number_of_edges() > 0:
        try:
            betweenness = hop_metrics(nx_graph).betweenness
        except Exception:
            betweenness = {}
    if betweenness is None:
        betweenness = {}

    for suffix, node_ids in (("_1", nodes_a), ("_2", nodes_b)):
        # Same interface basis as pair avg_h*_per_h* / degree_cv / entropy:
        # other-class neighbors only, not full (intra + inter) degree.
        cross_degrees = [
            sum(
                1
                for neighbor in nx_graph.neighbors(node_id)
                if int(nx_graph.nodes[neighbor]["habitat_label"])
                != int(nx_graph.nodes[node_id]["habitat_label"])
            )
            for node_id in node_ids
        ]
        features[f"{prefix}_degree_skewness{suffix}"] = _safe_skew(cross_degrees)

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
