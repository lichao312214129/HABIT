"""Opt-in degree-preserving null-model comparisons for habitat graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Union

import networkx as nx
import numpy as np

from habit.kernels.habitat_graph.models import HabitatGraph
from habit.kernels.habitat_graph.null_ensemble import (
    GraphNullSampler,
    adjacency_from_undirected,
    networkx_from_adjacency,
    sample_degree_preserving_adjacencies,
)

__all__ = [
    "GraphNullModelOptions",
    "GraphNullModelResult",
    "GraphStatistic",
    "compare_graph_to_degree_preserving_null",
]

GraphStatistic = Callable[[nx.Graph], float]


@dataclass(frozen=True)
class GraphNullModelOptions:
    """Reproducible degree-preserving null-model controls.

    Default sampler is the configuration model. Pass ``sampler='rewire'``
    for Maslov–Sneppen mixing (``swaps_per_edge`` follows NetworkX / Milo
    and defaults to 100).
    """

    n_random_graphs: int = 100
    swaps_per_edge: int = 100
    random_seed: int = 0
    sampler: GraphNullSampler = "config"

    def __post_init__(self) -> None:
        if self.n_random_graphs < 2:
            raise ValueError("n_random_graphs must be >= 2.")
        if self.swaps_per_edge < 1:
            raise ValueError("swaps_per_edge must be >= 1.")
        if self.sampler not in ("config", "rewire"):
            raise ValueError("sampler must be 'config' or 'rewire'.")


@dataclass(frozen=True)
class GraphNullModelResult:
    """Observed statistic and summary of degree-preserving random graphs."""

    observed: float
    null_mean: float
    null_std: float
    z_score: float
    empirical_two_sided_p: float
    n_requested: int
    n_successful: int
    is_valid: bool


def _as_networkx(graph: Union[HabitatGraph, nx.Graph]) -> nx.Graph:
    """Return a simple undirected NetworkX graph without mutating input."""
    if isinstance(graph, HabitatGraph):
        result = nx.Graph()
        result.add_nodes_from(graph.nodes)
        result.add_edges_from((edge.source, edge.target) for edge in graph.edges)
        return result
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("degree-preserving null models require a simple undirected graph.")
    return nx.Graph(graph)


def compare_graph_to_degree_preserving_null(
    graph: Union[HabitatGraph, nx.Graph],
    statistic: GraphStatistic,
    *,
    options: GraphNullModelOptions = GraphNullModelOptions(),
) -> GraphNullModelResult:
    """Compare a finite topology statistic with degree-preserving null graphs."""
    observed_graph = _as_networkx(graph)
    observed = float(statistic(observed_graph))
    if not np.isfinite(observed):
        raise ValueError("statistic must return a finite scalar.")
    n_edges = observed_graph.number_of_edges()
    if observed_graph.number_of_nodes() < 4 or n_edges < 2:
        return GraphNullModelResult(
            observed, 0.0, 0.0, 0.0, 0.0, options.n_random_graphs, 0, False
        )
    adj, nodes = adjacency_from_undirected(observed_graph)
    null_batch = sample_degree_preserving_adjacencies(
        adj,
        nrand=options.n_random_graphs,
        sampler=options.sampler,
        niter=options.swaps_per_edge,
        seed=options.random_seed,
    )
    samples: List[float] = []
    for null_adj in null_batch:
        random_graph = networkx_from_adjacency(
            null_adj, nodes, source_graph=observed_graph
        )
        try:
            value = float(statistic(random_graph))
        except (nx.NetworkXAlgorithmError, nx.NetworkXError, ValueError):
            continue
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return GraphNullModelResult(
            observed, 0.0, 0.0, 0.0, 0.0, options.n_random_graphs, 0, False
        )
    values = np.asarray(samples, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size >= 2 else 0.0
    valid = bool(values.size >= 2 and std > 1e-12)
    p_value = float(
        (1 + np.count_nonzero(np.abs(values - mean) >= abs(observed - mean)))
        / (values.size + 1)
    )
    return GraphNullModelResult(
        observed,
        mean,
        std,
        float((observed - mean) / std) if valid else 0.0,
        p_value,
        options.n_random_graphs,
        int(values.size),
        valid,
    )
