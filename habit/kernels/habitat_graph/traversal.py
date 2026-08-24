# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""One unweighted BFS sweep: Brandes betweenness plus hop-distance summaries.

Habitat graphs are undirected and unweighted (hop counts). NetworkX used
to run Brandes, average shortest path, diameter, and closeness as four
separate all-sources searches. Those quantities all come from the same
BFS trees, so one sweep is enough and the numbers stay the NetworkX
definitions (normalized betweenness, connected-graph closeness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from habit.utils.graph_brandes_utils import csr_from_undirected_edges, hop_metrics_csr

__all__ = [
    "HopMetricResult",
    "component_summary",
    "hop_metrics",
]


@dataclass(frozen=True)
class HopMetricResult:
    """Hop-count summaries from one all-sources BFS on one graph.

    ``betweenness`` / ``closeness`` are keyed by the node ids of
    ``nx_graph``. Isolated nodes keep betweenness 0. Path length and
    diameter are 0 when the graph has fewer than two nodes or is
    disconnected (callers should pass a connected component).
    """

    n_nodes: int
    betweenness: Dict[str, float]
    closeness: Dict[str, float]
    avg_path_length: float
    diameter: float


def component_summary(
    nx_graph: nx.Graph,
) -> Tuple[int, nx.Graph]:
    """
    Count connected components once and return the largest as a subgraph.

    Args:
        nx_graph: Undirected NetworkX graph.

    Returns:
        ``(n_components, largest)``. ``largest`` is a subgraph view of
        the biggest component (by node count), or an empty copy when
        ``nx_graph`` has no nodes. ``n_components`` is 0 when empty.
    """
    n_nodes = nx_graph.number_of_nodes()
    if n_nodes == 0:
        return 0, nx_graph.copy()
    components = list(nx.connected_components(nx_graph))
    largest_nodes = max(components, key=len)
    return len(components), nx_graph.subgraph(largest_nodes)


def hop_metrics(
    nx_graph: nx.Graph,
    *,
    device: str = "auto",
    backend: str = "networkx",
) -> HopMetricResult:
    """
    Brandes betweenness, closeness, mean path length, and diameter.

    One BFS from every node. Betweenness matches NetworkX
    ``betweenness_centrality(..., normalized=True, weight=None,
    endpoints=False)``: raw Brandes accumulation over all sources,
    then multiply by ``1/((n-1)*(n-2))`` (``n<=2`` stays 0).
    Closeness matches connected-graph NetworkX
    ``(n-1) / sum_u d(v,u)``. Mean path length is the mean hop
    distance over ordered pairs ``s!=t``; diameter is the maximum
    finite hop distance.

    ``device='auto'`` uses CUDA batched BFS when the graph is large
    enough that the all-pairs distance buffers pay off, otherwise
    compiled CPU Brandes, otherwise the Python reference. Tiny float
    drift on the CUDA path is accepted.

    Args:
        nx_graph: Undirected unweighted graph. Path length and
            diameter are only meaningful when this graph is connected;
            callers pass the largest component for those features.
        device: ``auto``, ``cpu``, ``numba``, ``python``, or ``cuda``.
        backend: ``networkx`` (default) uses compiled Brandes in this
            process. ``igraph`` requires ``habitat-analysis[igraph]``.
            ``auto`` uses igraph when that extra is installed.

    Returns:
        HopMetricResult: Per-node centralities and two scalars.
    """
    nodes: List[str] = list(nx_graph.nodes())
    n_nodes = len(nodes)
    betweenness = {node_id: 0.0 for node_id in nodes}
    closeness = {node_id: 0.0 for node_id in nodes}
    if n_nodes == 0:
        return HopMetricResult(
            n_nodes=0,
            betweenness=betweenness,
            closeness=closeness,
            avg_path_length=0.0,
            diameter=0.0,
        )
    index = {node_id: slot for slot, node_id in enumerate(nodes)}
    edges = [
        (index[source], index[target])
        for source, target in nx_graph.edges()
    ]
    resolved = str(backend).strip().lower()
    use_igraph = False
    if resolved == "igraph":
        use_igraph = True
    elif resolved in {"auto", ""}:
        from habit.utils.igraph_graph_utils import igraph_is_available

        use_igraph = igraph_is_available()
    if use_igraph:
        from habit.utils.igraph_graph_utils import hop_metrics_igraph

        bc, cc, avg_path, diameter = hop_metrics_igraph(n_nodes, edges)
        for slot, node_id in enumerate(nodes):
            betweenness[node_id] = float(bc[slot])
            closeness[node_id] = float(cc[slot])
        return HopMetricResult(
            n_nodes=n_nodes,
            betweenness=betweenness,
            closeness=closeness,
            avg_path_length=float(avg_path),
            diameter=float(diameter) if n_nodes > 1 else 0.0,
        )
    indptr, indices = csr_from_undirected_edges(n_nodes, edges)
    arrays = hop_metrics_csr(indptr, indices, n_nodes, device=device)
    for slot, node_id in enumerate(nodes):
        betweenness[node_id] = float(arrays.betweenness[slot])
        closeness[node_id] = float(arrays.closeness[slot])
    return HopMetricResult(
        n_nodes=n_nodes,
        betweenness=betweenness,
        closeness=closeness,
        avg_path_length=float(arrays.avg_path_length),
        diameter=float(arrays.diameter) if n_nodes > 1 else 0.0,
    )


def mean_betweenness(
    betweenness: Dict[str, float],
    node_ids: Optional[Iterable[str]] = None,
) -> float:
    """Mean of selected betweenness values, or 0 when the selection is empty."""
    if node_ids is None:
        values = list(betweenness.values())
    else:
        values = [float(betweenness[node_id]) for node_id in node_ids if node_id in betweenness]
    if not values:
        return 0.0
    return float(np.mean(values))


def betweenness_max_std(
    betweenness: Dict[str, float],
    node_ids: Optional[Sequence[str]] = None,
) -> Tuple[float, float]:
    """Maximum and population std of selected betweenness values."""
    if node_ids is None:
        values = list(betweenness.values())
    else:
        values = [
            float(betweenness[node_id])
            for node_id in node_ids
            if node_id in betweenness
        ]
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=float)
    return float(array.max()), float(array.std())
