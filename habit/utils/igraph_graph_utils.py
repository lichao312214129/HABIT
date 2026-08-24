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
"""Optional igraph backend for habitat-graph hop metrics and modularity.

``python-igraph`` is GPL-2.0+. HABIT stays Apache-2.0: this module is an
opt-in extra (``pip install habitat-analysis[igraph]``), never a required
dependency. When the extra is absent, callers keep the NetworkX / numba path.

Hop-metric definitions are aligned to NetworkX:

* undirected betweenness is igraph's raw value times
  ``2 / ((n-1)*(n-2))`` (igraph counts each undirected pair once);
* closeness, mean path length, diameter, and mean local clustering match
  NetworkX on connected unweighted graphs.

Louvain modularity uses igraph's multilevel community detection. The
partition can differ from NetworkX ``seed=0``, so the scalar may move by
a small amount.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "igraph_is_available",
    "require_igraph",
    "igraph_from_undirected_edges",
    "hop_metrics_igraph",
    "average_clustering_igraph",
    "modularity_igraph",
]


def igraph_is_available() -> bool:
    """Return True when the optional ``igraph`` extra can be imported."""
    try:
        import igraph  # noqa: F401
    except Exception:
        return False
    return True


def require_igraph():
    """
    Import igraph or raise :class:`OptionalDependencyError`.

    Returns:
        The ``igraph`` module.
    """
    from habit.utils.optional_deps import require

    return require(
        "igraph",
        extra="igraph",
        purpose="the optional C graph-metric backend (betweenness, "
        "clustering, Louvain modularity)",
        alternatives=(
            "leave graph_metric_backend='auto' / 'networkx' to stay on "
            "NetworkX and compiled Brandes",
        ),
    )


def igraph_from_undirected_edges(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
    *,
    weights: Optional[Sequence[float]] = None,
):
    """
    Build an undirected simple igraph.Graph from integer endpoint pairs.

    Args:
        n_nodes: Vertex count.
        edges: Undirected pairs. Self-loops and duplicates are dropped.
        weights: Optional per-edge weights aligned with ``edges`` after
            dropping loops. Unused pairs are skipped together with the loop.

    Returns:
        ``igraph.Graph`` with ``n_nodes`` vertices.
    """
    igraph = require_igraph()
    clean: List[Tuple[int, int]] = []
    clean_w: List[float] = []
    seen = set()
    for slot, (raw_a, raw_b) in enumerate(edges):
        node_a = int(raw_a)
        node_b = int(raw_b)
        if node_a == node_b:
            continue
        key = (node_a, node_b) if node_a < node_b else (node_b, node_a)
        if key in seen:
            continue
        seen.add(key)
        clean.append(key)
        if weights is not None:
            clean_w.append(float(weights[slot]))
    graph = igraph.Graph(n=int(n_nodes), edges=clean, directed=False)
    if weights is not None and clean_w:
        graph.es["weight"] = clean_w
    return graph


def hop_metrics_igraph(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Brandes, closeness, ASPL, and diameter via igraph.

    Args:
        n_nodes: Vertex count.
        edges: Undirected integer pairs.

    Returns:
        ``(betweenness, closeness, avg_path_length, diameter)`` as in
        :func:`habit.utils.graph_brandes_utils.hop_metrics_csr`.
    """
    n_nodes = int(n_nodes)
    betweenness = np.zeros(n_nodes, dtype=np.float64)
    closeness = np.zeros(n_nodes, dtype=np.float64)
    if n_nodes <= 1:
        return betweenness, closeness, 0.0, 0.0
    graph = igraph_from_undirected_edges(n_nodes, edges)
    raw = np.asarray(graph.betweenness(directed=False), dtype=np.float64)
    if n_nodes > 2:
        betweenness = raw * (2.0 / float((n_nodes - 1) * (n_nodes - 2)))
    close = np.asarray(graph.closeness(normalized=True), dtype=np.float64)
    # Isolated vertices can be nan in igraph; HABIT stores 0.
    closeness = np.where(np.isfinite(close), close, 0.0)
    if graph.ecount() == 0:
        return betweenness, closeness, 0.0, 0.0
    avg_path = float(graph.average_path_length(directed=False))
    if not np.isfinite(avg_path):
        avg_path = 0.0
    try:
        diameter = float(graph.diameter(directed=False, unconn=True))
    except Exception:
        diameter = 0.0
    if not np.isfinite(diameter):
        diameter = 0.0
    return betweenness, closeness, avg_path, diameter


def average_clustering_igraph(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
) -> float:
    """Mean local clustering (NetworkX ``average_clustering``)."""
    if n_nodes <= 0:
        return 0.0
    graph = igraph_from_undirected_edges(n_nodes, edges)
    if graph.vcount() == 0:
        return 0.0
    value = float(graph.transitivity_avglocal_undirected(mode="zero"))
    return value if np.isfinite(value) else 0.0


def modularity_igraph(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
    *,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """
    Louvain modularity via igraph multilevel communities.

    Args:
        n_nodes: Vertex count.
        edges: Undirected pairs.
        weights: Optional edge weights (same order as ``edges``).

    Returns:
        float: Modularity of the multilevel partition, or 0 when undefined.
    """
    if n_nodes <= 0 or not edges:
        return 0.0
    graph = igraph_from_undirected_edges(n_nodes, edges, weights=weights)
    if graph.ecount() == 0:
        return 0.0
    weight_key = "weight" if weights is not None and "weight" in graph.es.attributes() else None
    try:
        partition = graph.community_multilevel(weights=weight_key)
        value = float(graph.modularity(partition, weights=weight_key))
    except Exception:
        return 0.0
    return value if np.isfinite(value) else 0.0
