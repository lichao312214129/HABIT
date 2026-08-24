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
"""Fast degree-preserving null ensembles for habitat-graph metrics.

Two samplers, one scientific object (a simple graph with the observed
degree sequence):

* ``config`` (default) -- configuration-model stub matching. One draw
  is one exact degree-sequence graph. No ``niter`` mixing loop.
* ``rewire`` -- Maslov–Sneppen double-edge swaps on a copy of the
  observed graph (Milo / NetworkX mixing: about ``niter`` swaps per
  edge). This is the explicit random-rewiring option.

Clustering, mean path length, and rich-club curves are evaluated on a
stacked adjacency batch. NumPy is the default backend. A single PyTorch
CUDA launch is used only when the Floyd–Warshall work is large enough
that device transfer pays off -- never one kernel per graph.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.csgraph import connected_components

from habit.utils.torch_radiomics_utils import is_cuda_available, is_torch_available

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional accelerator
    njit = None
    _HAS_NUMBA = False

__all__ = [
    "GraphNullSampler",
    "adjacency_from_undirected",
    "networkx_from_adjacency",
    "sample_degree_preserving_adjacencies",
    "batched_transitivity",
    "batched_average_path_length",
    "rich_club_phi",
    "ensemble_null_summaries",
    "local_efficiency_values",
    "global_efficiency_value",
]

# ``analytic`` is handled in extended_metrics (Humphries ER formulas).
# This module only draws degree-preserving graphs.
GraphNullSampler = Literal["config", "rewire"]

# Unreachable marker for dense Floyd–Warshall. Habitat graphs are tiny
# (n ~ 100); a hop count this large cannot occur on a connected graph.
_UNREACHABLE: float = 1.0e6
# Use CUDA Floyd–Warshall only when B * n^3 exceeds this. Below it, the
# H2D cost dominates and NumPy is faster (same lesson as min-distance).
_CUDA_FW_MIN_OPS: int = 5_000_000


def adjacency_from_undirected(nx_graph) -> Tuple[np.ndarray, List]:
    """
    Dense 0/1 adjacency and a stable node order.

    Args:
        nx_graph: Simple undirected NetworkX graph.

    Returns:
        ``(adj, nodes)`` where ``adj`` is ``(n, n)`` float32 and
        ``nodes[i]`` is the original id of row/column ``i``.
    """
    nodes = list(nx_graph.nodes())
    n_nodes = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for source, target in nx_graph.edges():
        row = index[source]
        col = index[target]
        if row == col:
            continue
        adj[row, col] = 1.0
        adj[col, row] = 1.0
    return adj, nodes


def networkx_from_adjacency(
    adj: np.ndarray,
    nodes: Sequence,
    *,
    source_graph=None,
) -> "nx.Graph":
    """
    Rebuild a simple undirected graph, preserving the original node ids.

    Args:
        adj: Symmetric 0/1 adjacency, shape ``(n, n)``.
        nodes: Node ids in row order (length ``n``).
        source_graph: Optional graph whose node attributes are copied.

    Returns:
        A NetworkX graph with those nodes and the edges in ``adj``.
    """
    import networkx as nx

    graph = nx.Graph()
    if source_graph is not None:
        graph.add_nodes_from(
            (node, dict(source_graph.nodes[node])) for node in nodes
        )
    else:
        graph.add_nodes_from(nodes)
    rows, cols = np.nonzero(np.triu(adj, k=1))
    graph.add_edges_from(
        (nodes[int(row)], nodes[int(col)]) for row, col in zip(rows, cols)
    )
    return graph


def _is_connected(adj: np.ndarray) -> bool:
    """Return True when the undirected graph has one component."""
    n_nodes = int(adj.shape[0])
    if n_nodes <= 1:
        return True
    n_components, _ = connected_components(adj, directed=False)
    return int(n_components) == 1


def _try_stub_matching_numpy(
    degrees: np.ndarray,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    One configuration-model draw that stays a simple graph.

    Remaining stubs are a random stack. Each popped stub takes a
    uniformly chosen *legal* partner (not itself, not an existing edge)
    among whatever stubs are left -- one NumPy mask per pairing, not a
    Python trial loop. Degrees of a successful draw match ``degrees``
    exactly; a stuck pairing returns ``None``.

    Args:
        degrees: Target degree of each node, shape ``(n,)``, even sum.
        rng: NumPy generator for this draw.

    Returns:
        ``(n, n)`` uint8 adjacency, or ``None`` on failure.
    """
    n_nodes = int(degrees.size)
    stubs = np.repeat(
        np.arange(n_nodes, dtype=np.int32),
        degrees.astype(np.int32, copy=False),
    )
    rng.shuffle(stubs)
    n_stubs = int(stubs.size)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    failures = 0
    max_failures = max(n_nodes, 8)
    while n_stubs >= 2:
        n_stubs -= 1
        node_a = int(stubs[n_stubs])
        remaining = stubs[:n_stubs]
        legal = (remaining != node_a) & (adj[node_a, remaining] == 0)
        legal_idx = np.flatnonzero(legal)
        if legal_idx.size == 0:
            stubs[n_stubs] = np.int32(node_a)
            n_stubs += 1
            failures += 1
            if failures >= max_failures:
                return None
            rng.shuffle(stubs[:n_stubs])
            continue
        pick = int(legal_idx[int(rng.integers(legal_idx.size))])
        node_b = int(remaining[pick])
        adj[node_a, node_b] = 1
        adj[node_b, node_a] = 1
        n_stubs -= 1
        stubs[pick] = stubs[n_stubs]
    if n_stubs != 0:
        return None
    if not np.array_equal(
        adj.sum(axis=1).astype(np.int64), degrees.astype(np.int64)
    ):
        return None
    return adj


if _HAS_NUMBA:

    @njit(cache=True)
    def _try_stub_matching_numba(
        degrees: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, bool]:
        """Numba configuration-model pairing. Returns ``(adj, ok)``."""
        n_nodes = degrees.shape[0]
        n_stubs = 0
        for i in range(n_nodes):
            n_stubs += int(degrees[i])
        stubs = np.empty(n_stubs, dtype=np.int32)
        write = 0
        for i in range(n_nodes):
            for _ in range(int(degrees[i])):
                stubs[write] = i
                write += 1
        np.random.seed(seed)
        for i in range(n_stubs - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            tmp = stubs[i]
            stubs[i] = stubs[j]
            stubs[j] = tmp
        adj = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
        left = n_stubs
        failures = 0
        while left >= 2:
            left -= 1
            node_a = int(stubs[left])
            n_legal = 0
            for i in range(left):
                node_b = int(stubs[i])
                if node_a != node_b and adj[node_a, node_b] == 0:
                    n_legal += 1
            if n_legal == 0:
                stubs[left] = node_a
                left += 1
                failures += 1
                if failures >= n_nodes:
                    return adj, False
                for i in range(left - 1, 0, -1):
                    j = np.random.randint(0, i + 1)
                    tmp = stubs[i]
                    stubs[i] = stubs[j]
                    stubs[j] = tmp
                continue
            pick = np.random.randint(0, n_legal)
            seen = 0
            chosen = 0
            for i in range(left):
                node_b = int(stubs[i])
                if node_a != node_b and adj[node_a, node_b] == 0:
                    if seen == pick:
                        chosen = i
                        break
                    seen += 1
            node_b = int(stubs[chosen])
            adj[node_a, node_b] = 1
            adj[node_b, node_a] = 1
            left -= 1
            stubs[chosen] = stubs[left]
        if left != 0:
            return adj, False
        return adj, True

    @njit(cache=True)
    def _rewire_adjacency_numba(
        source: np.ndarray,
        nswap: int,
        seed: int,
    ) -> Tuple[np.ndarray, bool]:
        """Numba Maslov–Sneppen mixing. Returns ``(adj, accepted_enough)``."""
        n_nodes = source.shape[0]
        adj = source.copy()
        n_edges = 0
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj[i, j] != 0:
                    n_edges += 1
        if n_nodes < 4 or n_edges < 2 or nswap < 1:
            return adj, False
        sources = np.empty(n_edges, dtype=np.int32)
        targets = np.empty(n_edges, dtype=np.int32)
        write = 0
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj[i, j] != 0:
                    sources[write] = i
                    targets[write] = j
                    write += 1
        np.random.seed(seed)
        max_tries = nswap * 10
        if max_tries < 100:
            max_tries = 100
        accepted = 0
        for _ in range(max_tries):
            if accepted >= nswap:
                break
            edge_i = np.random.randint(0, n_edges)
            edge_j = np.random.randint(0, n_edges)
            if edge_i == edge_j:
                continue
            node_a = int(sources[edge_i])
            node_b = int(targets[edge_i])
            node_c = int(sources[edge_j])
            node_d = int(targets[edge_j])
            if (
                node_a == node_c
                or node_a == node_d
                or node_b == node_c
                or node_b == node_d
            ):
                continue
            if np.random.randint(0, 2) == 0:
                new_a, new_b, new_c, new_d = node_a, node_c, node_b, node_d
            else:
                new_a, new_b, new_c, new_d = node_a, node_d, node_b, node_c
            if new_a == new_b or new_c == new_d:
                continue
            if adj[new_a, new_b] != 0 or adj[new_c, new_d] != 0:
                continue
            adj[node_a, node_b] = 0
            adj[node_b, node_a] = 0
            adj[node_c, node_d] = 0
            adj[node_d, node_c] = 0
            adj[new_a, new_b] = 1
            adj[new_b, new_a] = 1
            adj[new_c, new_d] = 1
            adj[new_d, new_c] = 1
            if new_a > new_b:
                new_a, new_b = new_b, new_a
            if new_c > new_d:
                new_c, new_d = new_d, new_c
            sources[edge_i] = new_a
            targets[edge_i] = new_b
            sources[edge_j] = new_c
            targets[edge_j] = new_d
            accepted += 1
        return adj, accepted >= nswap

    @njit(cache=True)
    def _bfs_connected_numba(adj: np.ndarray) -> bool:
        """True when the undirected graph has one component."""
        n_nodes = adj.shape[0]
        if n_nodes <= 1:
            return True
        seen = np.zeros(n_nodes, dtype=np.uint8)
        queue = np.empty(n_nodes, dtype=np.int32)
        queue[0] = 0
        seen[0] = 1
        head = 0
        tail = 1
        count = 1
        while head < tail:
            node_u = int(queue[head])
            head += 1
            for node_v in range(n_nodes):
                if adj[node_u, node_v] != 0 and seen[node_v] == 0:
                    seen[node_v] = 1
                    queue[tail] = node_v
                    tail += 1
                    count += 1
        return count == n_nodes

    @njit(cache=True)
    def _sample_config_batch_numba(
        degrees: np.ndarray,
        seeds: np.ndarray,
        n_requested: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw up to ``n_requested`` connected configuration-model graphs.

        Args:
            degrees: Degree sequence, shape ``(n,)``.
            seeds: Candidate RNG seeds, shape ``(n_try,)``.
            n_requested: How many successes to keep.

        Returns:
            ``(stack, ok_mask)`` where ``stack`` is ``(n_try, n, n)`` and
            ``ok_mask[i]`` is 1 when draw ``i`` was kept. Only the first
            ``n_requested`` successes are marked.
        """
        n_try = seeds.shape[0]
        n_nodes = degrees.shape[0]
        stack = np.zeros((n_try, n_nodes, n_nodes), dtype=np.uint8)
        ok_mask = np.zeros(n_try, dtype=np.uint8)
        kept = 0
        for i in range(n_try):
            if kept >= n_requested:
                break
            adj, success = _try_stub_matching_numba(degrees, int(seeds[i]))
            if not success:
                continue
            if not _bfs_connected_numba(adj):
                continue
            stack[i] = adj
            ok_mask[i] = 1
            kept += 1
        return stack, ok_mask

    @njit(cache=True)
    def _floyd_warshall_batch_numba(adj_batch: np.ndarray) -> np.ndarray:
        """
        Dense hop-distance Floyd–Warshall on a stack of graphs.

        Args:
            adj_batch: ``(B, n, n)`` float32 0/1 adjacencies.

        Returns:
            ``(B, n, n)`` float32 distances; unreachable entries stay
            ``_UNREACHABLE``.
        """
        batch, n_nodes, _ = adj_batch.shape
        dist = np.empty((batch, n_nodes, n_nodes), dtype=np.float32)
        unreachable = np.float32(_UNREACHABLE)
        one = np.float32(1.0)
        zero = np.float32(0.0)
        for b in range(batch):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        dist[b, i, j] = zero
                    elif adj_batch[b, i, j] > 0.5:
                        dist[b, i, j] = one
                    else:
                        dist[b, i, j] = unreachable
            for k in range(n_nodes):
                for i in range(n_nodes):
                    dik = dist[b, i, k]
                    if dik >= unreachable * np.float32(0.5):
                        continue
                    for j in range(n_nodes):
                        via = dik + dist[b, k, j]
                        if via < dist[b, i, j]:
                            dist[b, i, j] = via
        return dist

    @njit(cache=True)
    def _local_efficiency_numba(adj: np.ndarray) -> np.ndarray:
        """Per-node local efficiency (Latora–Marchiori) on one adjacency."""
        n_nodes = adj.shape[0]
        values = np.zeros(n_nodes, dtype=np.float64)
        unreachable = np.float32(_UNREACHABLE)
        neighbors = np.empty(n_nodes, dtype=np.int32)
        for node in range(n_nodes):
            n_nbr = 0
            for other in range(n_nodes):
                if other != node and adj[node, other] > 0.5:
                    neighbors[n_nbr] = other
                    n_nbr += 1
            if n_nbr < 2:
                continue
            dist = np.empty((n_nbr, n_nbr), dtype=np.float32)
            for i in range(n_nbr):
                for j in range(n_nbr):
                    if i == j:
                        dist[i, j] = 0.0
                    elif adj[neighbors[i], neighbors[j]] > 0.5:
                        dist[i, j] = 1.0
                    else:
                        dist[i, j] = unreachable
            for k in range(n_nbr):
                for i in range(n_nbr):
                    dik = dist[i, k]
                    if dik >= unreachable * np.float32(0.5):
                        continue
                    for j in range(n_nbr):
                        via = dik + dist[k, j]
                        if via < dist[i, j]:
                            dist[i, j] = via
            inv_sum = 0.0
            for i in range(n_nbr):
                for j in range(n_nbr):
                    if i == j:
                        continue
                    if dist[i, j] < unreachable * np.float32(0.5) and dist[i, j] > 0.0:
                        inv_sum += 1.0 / float(dist[i, j])
            values[node] = inv_sum / float(n_nbr * (n_nbr - 1))
        return values

else:  # pragma: no cover - no numba in this environment
    _try_stub_matching_numba = None
    _rewire_adjacency_numba = None
    _bfs_connected_numba = None
    _sample_config_batch_numba = None
    _floyd_warshall_batch_numba = None
    _local_efficiency_numba = None


def _try_stub_matching(
    degrees: np.ndarray,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Dispatch configuration-model pairing to Numba when installed."""
    degree_i32 = np.asarray(degrees, dtype=np.int32)
    if _try_stub_matching_numba is not None:
        try:
            seed = int(rng.integers(0, 2**31 - 1))
            adj, ok = _try_stub_matching_numba(degree_i32, seed)
            if ok:
                return adj
            return None
        except Exception:
            return _try_stub_matching_numpy(degree_i32, rng)
    return _try_stub_matching_numpy(degree_i32, rng)


def _sample_configuration_adjacency(
    degrees: np.ndarray,
    rng: np.random.Generator,
    *,
    require_connected: bool,
    max_graph_tries: int = 48,
) -> Optional[np.ndarray]:
    """
    Sample a simple configuration-model graph with the given degrees.

    Args:
        degrees: Target degree sequence, shape ``(n,)``.
        rng: NumPy generator.
        require_connected: Reject disconnected realizations.
        max_graph_tries: Independent pairing attempts.

    Returns:
        ``(n, n)`` float32 adjacency, or ``None``.
    """
    n_nodes = int(degrees.size)
    n_stubs = int(degrees.sum())
    if n_nodes < 2 or n_stubs < 2 or n_stubs % 2 != 0:
        return None
    if int(degrees.max()) >= n_nodes:
        return None
    for _ in range(max_graph_tries):
        adj = _try_stub_matching(degrees, rng)
        if adj is None:
            continue
        if require_connected and not _is_connected(adj):
            continue
        return adj.astype(np.float32, copy=False)
    return None


def _rewire_adjacency_numpy(
    source: np.ndarray,
    nswap: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Maslov–Sneppen mixing without per-swap connectivity tests.

    The finished graph is rejected if it is disconnected (BCT
    ``randmio_und`` convention). Candidate endpoint pairs and pairing
    bits are drawn in one vectorized call; the accept/reject loop is
    then a tight Python pass over those pre-drawn indices.

    Args:
        source: Observed 0/1 adjacency, shape ``(n, n)``.
        nswap: Target number of accepted double-edge swaps.
        rng: NumPy generator for this realization.

    Returns:
        Mixed ``(n, n)`` float32 adjacency, or ``None``.
    """
    adj = source.astype(np.uint8, copy=True)
    n_nodes = int(adj.shape[0])
    rows, cols = np.nonzero(np.triu(adj, k=1))
    sources = rows.astype(np.int32, copy=True)
    targets = cols.astype(np.int32, copy=True)
    n_edges = int(sources.size)
    if n_nodes < 4 or n_edges < 2 or nswap < 1:
        return None
    max_tries = max(int(nswap) * 10, 100)
    rand_i = rng.integers(0, n_edges, size=max_tries, dtype=np.int32)
    rand_j = rng.integers(0, n_edges, size=max_tries, dtype=np.int32)
    rand_p = rng.integers(0, 2, size=max_tries, dtype=np.int32)
    accepted = 0
    for trial in range(max_tries):
        if accepted >= nswap:
            break
        edge_i = int(rand_i[trial])
        edge_j = int(rand_j[trial])
        if edge_i == edge_j:
            continue
        node_a = int(sources[edge_i])
        node_b = int(targets[edge_i])
        node_c = int(sources[edge_j])
        node_d = int(targets[edge_j])
        if (
            node_a == node_c
            or node_a == node_d
            or node_b == node_c
            or node_b == node_d
        ):
            continue
        if int(rand_p[trial]) == 0:
            new_a, new_b, new_c, new_d = node_a, node_c, node_b, node_d
        else:
            new_a, new_b, new_c, new_d = node_a, node_d, node_b, node_c
        if new_a == new_b or new_c == new_d:
            continue
        if adj[new_a, new_b] or adj[new_c, new_d]:
            continue
        adj[node_a, node_b] = 0
        adj[node_b, node_a] = 0
        adj[node_c, node_d] = 0
        adj[node_d, node_c] = 0
        adj[new_a, new_b] = 1
        adj[new_b, new_a] = 1
        adj[new_c, new_d] = 1
        adj[new_d, new_c] = 1
        if new_a > new_b:
            new_a, new_b = new_b, new_a
        if new_c > new_d:
            new_c, new_d = new_d, new_c
        sources[edge_i] = new_a
        targets[edge_i] = new_b
        sources[edge_j] = new_c
        targets[edge_j] = new_d
        accepted += 1
    if accepted < nswap:
        return None
    if not _is_connected(adj):
        return None
    return adj.astype(np.float32, copy=False)


def _rewire_adjacency(
    source: np.ndarray,
    nswap: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Dispatch Maslov–Sneppen mixing to Numba when installed."""
    observed = np.asarray(source, dtype=np.uint8)
    if _rewire_adjacency_numba is not None:
        try:
            seed = int(rng.integers(0, 2**31 - 1))
            mixed, ok = _rewire_adjacency_numba(observed, int(nswap), seed)
            if not ok or not _is_connected(mixed):
                return None
            return mixed.astype(np.float32, copy=False)
        except Exception:
            return _rewire_adjacency_numpy(observed, nswap, rng)
    return _rewire_adjacency_numpy(observed, nswap, rng)


def sample_degree_preserving_adjacencies(
    adj: np.ndarray,
    *,
    nrand: int,
    sampler: GraphNullSampler = "config",
    niter: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """
    Draw connected degree-preserving null adjacencies.

    ``config`` tries stub matching first. A draw that cannot be paired
    into a simple connected graph falls back to Maslov–Sneppen mixing
    of the observed graph so the ensemble size stays honest (we do not
    silently shrink ``nrand`` because the fast sampler failed).

    Args:
        adj: Observed 0/1 adjacency, shape ``(n, n)``.
        nrand: Requested number of accepted null graphs.
        sampler: ``config`` (default) or ``rewire``.
        niter: Target swaps per edge for ``rewire`` and for the
            configuration-model fallback. NetworkX / Milo default is 100.
        seed: Base seed for the ensemble.

    Returns:
        Float32 stack of shape ``(n_accepted, n, n)``. ``n_accepted``
        may be smaller than ``nrand`` when every attempt failed.
    """
    if sampler not in ("config", "rewire"):
        raise ValueError(
            f"sampler must be 'config' or 'rewire'; got {sampler!r}."
        )
    observed = np.asarray(adj, dtype=np.float32)
    if observed.ndim != 2 or observed.shape[0] != observed.shape[1]:
        raise ValueError("adj must be a square adjacency matrix.")
    degrees = observed.sum(axis=1).astype(np.int64)
    n_edges = int(degrees.sum() // 2)
    n_requested = max(1, int(nrand))
    nswap = max(1, int(niter) * max(n_edges, 1))
    rng = np.random.default_rng(int(seed))
    collected: List[np.ndarray] = []
    max_attempts = n_requested * 4
    if (
        sampler == "config"
        and _sample_config_batch_numba is not None
        and observed.shape[0] >= 2
    ):
        seeds = rng.integers(0, 2**31 - 1, size=max_attempts, dtype=np.int64)
        try:
            stack, ok_mask = _sample_config_batch_numba(
                degrees.astype(np.int32, copy=False),
                seeds,
                n_requested,
            )
            for i in range(int(ok_mask.size)):
                if ok_mask[i] != 0:
                    collected.append(stack[i].astype(np.float32, copy=False))
        except Exception:
            collected = []
    if len(collected) < n_requested:
        remaining = n_requested - len(collected)
        for _ in range(remaining * 4):
            if len(collected) >= n_requested:
                break
            child = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            drawn: Optional[np.ndarray] = None
            if sampler == "config":
                drawn = _sample_configuration_adjacency(
                    degrees, child, require_connected=True
                )
                if drawn is None:
                    drawn = _rewire_adjacency(observed, nswap, child)
            else:
                drawn = _rewire_adjacency(observed, nswap, child)
            if drawn is not None:
                collected.append(drawn)
    if not collected:
        return np.zeros((0, observed.shape[0], observed.shape[1]), dtype=np.float32)
    return np.stack(collected[:n_requested], axis=0)


def _resolve_metric_device(device: str, batch: int, n_nodes: int) -> Optional[str]:
    """
    Choose a torch device for batched Floyd–Warshall, or NumPy.

    Args:
        device: ``auto``, ``cpu``, ``cuda``, or ``cuda:N``.
        batch: Number of stacked graphs.
        n_nodes: Nodes per graph.

    Returns:
        A torch device string, or ``None`` to stay on NumPy.
    """
    requested = str(device).strip().lower()
    ops = int(batch) * int(n_nodes) ** 3
    if requested in {"cpu", ""}:
        return None
    wants_cuda = requested in {"auto", "cuda"} or requested.startswith("cuda:")
    if not wants_cuda:
        raise ValueError(
            "graph_null_device must be auto, cpu, cuda, or cuda:N; "
            f"got {device!r}."
        )
    if not is_torch_available() or not is_cuda_available():
        if requested != "auto":
            raise RuntimeError(
                f"graph_null_device={device!r} needs CUDA-enabled PyTorch."
            )
        return None
    if requested == "auto" and ops < _CUDA_FW_MIN_OPS:
        return None
    if requested == "auto" or requested == "cuda":
        return "cuda:0"
    return str(device)


def batched_transitivity(adj_batch: np.ndarray) -> np.ndarray:
    """
    NetworkX-style transitivity on a stack of simple undirected graphs.

    ``C = trace(A^3) / sum_i k_i (k_i - 1)``, which equals
    ``3 * triangles / connected triples``.

    Args:
        adj_batch: Adjacency stack, shape ``(B, n, n)`` or ``(n, n)``.

    Returns:
        Transitivity per graph, shape ``(B,)``. Zero when no triple exists.
    """
    stacked = np.asarray(adj_batch, dtype=np.float64)
    if stacked.ndim == 2:
        stacked = stacked[None, ...]
    degrees = stacked.sum(axis=-1)
    triples = (degrees * (degrees - 1.0)).sum(axis=-1)
    closed = (stacked @ stacked * stacked).sum(axis=(-1, -2))
    out = np.zeros(stacked.shape[0], dtype=np.float64)
    ok = triples > 0.0
    out[ok] = closed[ok] / triples[ok]
    return out


def _distances_to_aspl(dist: np.ndarray) -> np.ndarray:
    """Convert a distance stack to ASPL; NaN when a graph is disconnected."""
    n_nodes = int(dist.shape[1])
    reachable = dist < (0.5 * _UNREACHABLE)
    connected = reachable.all(axis=(-1, -2))
    off_diag = float(n_nodes * (n_nodes - 1))
    aspl = dist.sum(axis=(-1, -2)) / off_diag
    aspl = np.where(connected, aspl, np.nan)
    return aspl.astype(np.float64, copy=False)


def _aspl_numpy(adj_batch: np.ndarray) -> np.ndarray:
    """Dense Floyd–Warshall mean hop length; NaN if a graph is disconnected."""
    stacked = np.asarray(adj_batch, dtype=np.float32)
    if _floyd_warshall_batch_numba is not None:
        try:
            dist = _floyd_warshall_batch_numba(stacked)
            return _distances_to_aspl(dist)
        except Exception:
            pass
    batch, n_nodes, _ = stacked.shape
    dist = np.where(stacked > 0.5, np.float32(1.0), np.float32(_UNREACHABLE))
    index = np.arange(n_nodes)
    dist[:, index, index] = 0.0
    for k in range(n_nodes):
        dist = np.minimum(dist, dist[:, :, k : k + 1] + dist[:, k : k + 1, :])
    return _distances_to_aspl(dist)


def _clustering_and_aspl_torch(
    adj_batch: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One CUDA launch for transitivity and ASPL on the whole ensemble.

    Args:
        adj_batch: Adjacency stack, shape ``(B, n, n)``.
        device: Concrete torch device string.

    Returns:
        ``(clustering, aspl)`` as float64 host arrays, shape ``(B,)``.
        Disconnected graphs have NaN ASPL.
    """
    import torch

    adjacency = torch.as_tensor(adj_batch, device=device, dtype=torch.float32)
    batch, n_nodes, _ = adjacency.shape
    degrees = adjacency.sum(dim=-1)
    triples = (degrees * (degrees - 1.0)).sum(dim=-1)
    closed = torch.bmm(adjacency, adjacency)
    closed = (closed * adjacency).sum(dim=(-1, -2))
    clustering = torch.where(
        triples > 0.0, closed / triples, torch.zeros_like(triples)
    )

    inf = float(_UNREACHABLE)
    dist = torch.where(
        adjacency > 0.5,
        torch.ones_like(adjacency),
        torch.full_like(adjacency, inf),
    )
    index = torch.arange(n_nodes, device=device)
    dist[:, index, index] = 0.0
    for k in range(n_nodes):
        dist = torch.minimum(
            dist, dist[:, :, k : k + 1] + dist[:, k : k + 1, :]
        )
    connected = (dist < (0.5 * inf)).all(dim=(-1, -2))
    off_diag = float(n_nodes * (n_nodes - 1))
    aspl = dist.sum(dim=(-1, -2)) / off_diag
    aspl = torch.where(
        connected, aspl, torch.full_like(aspl, float("nan"))
    )
    return (
        clustering.detach().cpu().numpy().astype(np.float64, copy=False),
        aspl.detach().cpu().numpy().astype(np.float64, copy=False),
    )


def batched_average_path_length(
    adj_batch: np.ndarray,
    *,
    device: str = "auto",
) -> np.ndarray:
    """
    Unweighted mean shortest-path length on a stack of graphs.

    Args:
        adj_batch: Adjacency stack, shape ``(B, n, n)`` or ``(n, n)``.
        device: ``auto`` / ``cpu`` / ``cuda`` / ``cuda:N``.

    Returns:
        ASPL per graph, shape ``(B,)``. NaN when the graph is disconnected.
    """
    stacked = np.asarray(adj_batch, dtype=np.float32)
    if stacked.ndim == 2:
        stacked = stacked[None, ...]
    if stacked.shape[1] <= 1:
        return np.full(stacked.shape[0], np.nan, dtype=np.float64)
    resolved = _resolve_metric_device(
        device, batch=int(stacked.shape[0]), n_nodes=int(stacked.shape[1])
    )
    if resolved is None:
        return _aspl_numpy(stacked)
    _, aspl = _clustering_and_aspl_torch(stacked, resolved)
    return aspl


def _clustering_and_aspl(
    adj_batch: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transitivity and ASPL, sharing one CUDA transfer when used."""
    stacked = np.asarray(adj_batch, dtype=np.float32)
    if stacked.ndim == 2:
        stacked = stacked[None, ...]
    resolved = _resolve_metric_device(
        device, batch=int(stacked.shape[0]), n_nodes=int(stacked.shape[1])
    )
    if resolved is None:
        return batched_transitivity(stacked), _aspl_numpy(stacked)
    return _clustering_and_aspl_torch(stacked, resolved)


def global_efficiency_value(adj: np.ndarray) -> float:
    """
    Latora–Marchiori global efficiency of one unweighted graph.

    Unreachable pairs contribute 0 (same convention as NetworkX).

    Args:
        adj: 0/1 adjacency, shape ``(n, n)``.

    Returns:
        Efficiency in ``[0, 1]``, or ``0.0`` when ``n < 2``.
    """
    matrix = np.asarray(adj, dtype=np.float32)
    n_nodes = int(matrix.shape[0])
    if n_nodes < 2:
        return 0.0
    dist = None
    if _floyd_warshall_batch_numba is not None:
        try:
            dist = _floyd_warshall_batch_numba(matrix[None, ...])[0]
        except Exception:
            dist = None
    if dist is None:
        dist = np.where(matrix > 0.5, np.float32(1.0), np.float32(_UNREACHABLE))
        index = np.arange(n_nodes)
        dist[index, index] = 0.0
        for k in range(n_nodes):
            dist = np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
    inv = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    reachable = (dist > 0.0) & (dist < (0.5 * _UNREACHABLE))
    inv[reachable] = 1.0 / dist[reachable].astype(np.float64, copy=False)
    return float(inv.sum() / float(n_nodes * (n_nodes - 1)))


def local_efficiency_values(adj: np.ndarray) -> List[float]:
    """
    Per-node local efficiency of one unweighted graph.

    For each node, this is the global efficiency of the subgraph induced
    by its neighbours (NetworkX / Latora–Marchiori).

    Args:
        adj: 0/1 adjacency, shape ``(n, n)``.

    Returns:
        One value per node.
    """
    matrix = np.asarray(adj, dtype=np.float32)
    if matrix.shape[0] == 0:
        return []
    if _local_efficiency_numba is not None:
        try:
            return [float(v) for v in _local_efficiency_numba(matrix)]
        except Exception:
            pass
    # NumPy fallback: neighbour-induced subgraphs, one at a time.
    values: List[float] = []
    n_nodes = int(matrix.shape[0])
    for node in range(n_nodes):
        neighbors = np.flatnonzero(matrix[node] > 0.5)
        if neighbors.size < 2:
            values.append(0.0)
            continue
        values.append(global_efficiency_value(matrix[np.ix_(neighbors, neighbors)]))
    return values


def rich_club_phi(adj: np.ndarray, degrees: Optional[np.ndarray] = None) -> Dict[int, float]:
    """
    Unnormalized rich-club curve ``φ(k)`` (Colizza / NetworkX).

    ``φ(k) = 2 E_{>k} / (N_{>k} (N_{>k} - 1))`` for every ``k`` with
    at least two nodes of degree greater than ``k``.

    Args:
        adj: Single 0/1 adjacency, shape ``(n, n)``.
        degrees: Optional precomputed degrees, shape ``(n,)``.

    Returns:
        Mapping ``k -> φ(k)`` for defined levels.
    """
    matrix = np.asarray(adj, dtype=np.float64)
    if degrees is None:
        deg = matrix.sum(axis=1)
    else:
        deg = np.asarray(degrees, dtype=np.float64)
    if deg.size == 0:
        return {}
    max_degree = int(deg.max())
    curve: Dict[int, float] = {}
    for k in range(max_degree):
        hub = deg > k
        n_hub = int(hub.sum())
        if n_hub < 2:
            continue
        n_edges = float(matrix[np.ix_(hub, hub)].sum() * 0.5)
        curve[k] = 2.0 * n_edges / float(n_hub * (n_hub - 1))
    return curve


def _mean_rich_club_normalized(
    observed_adj: np.ndarray,
    null_batch: np.ndarray,
) -> float:
    """
    Mean of finite ``φ(k) / mean_φ_rand(k)`` over the ensemble.

    This is the connectomics point estimate, but ``φ_rand`` is the mean
    over ``nrand`` null graphs rather than NetworkX's single Q-mixed
    graph. That is strictly more stable, not a cheaper substitute.
    """
    observed = rich_club_phi(observed_adj)
    if not observed or null_batch.shape[0] == 0:
        return 0.0
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for graph in null_batch:
        for k, value in rich_club_phi(graph).items():
            if not np.isfinite(value):
                continue
            sums[k] = sums.get(k, 0.0) + float(value)
            counts[k] = counts.get(k, 0) + 1
    ratios: List[float] = []
    for k, phi_obs in observed.items():
        n_k = counts.get(k, 0)
        if n_k < 1:
            continue
        phi_rand = sums[k] / float(n_k)
        if phi_rand <= 0.0 or not np.isfinite(phi_obs) or not np.isfinite(phi_rand):
            continue
        ratios.append(float(phi_obs) / phi_rand)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def ensemble_null_summaries(
    adj: np.ndarray,
    *,
    nrand: int = 100,
    sampler: GraphNullSampler = "config",
    niter: int = 100,
    device: str = "auto",
    seed: int = 0,
) -> Optional[Tuple[float, float, float, int]]:
    """
    Degree-preserving ensemble means used by extended graph metrics.

    Args:
        adj: Observed 0/1 adjacency, shape ``(n, n)``.
        nrand: Requested ensemble size (default 100).
        sampler: ``config`` (default) or ``rewire``.
        niter: Swaps per edge when ``sampler='rewire'`` (default 100).
        device: Metric backend (``auto`` / ``cpu`` / ``cuda`` / ``cuda:N``).
        seed: Ensemble seed.

    Returns:
        ``(C_rand, L_rand, rich_club_mean, n_accepted)`` or ``None``
        when fewer than one usable null graph is available.
    """
    observed = np.asarray(adj, dtype=np.float32)
    if observed.shape[0] < 2:
        return None
    null_batch = sample_degree_preserving_adjacencies(
        observed,
        nrand=nrand,
        sampler=sampler,
        niter=niter,
        seed=seed,
    )
    if null_batch.shape[0] < 1:
        return None
    clustering, aspl = _clustering_and_aspl(null_batch, device)
    usable = np.isfinite(clustering) & np.isfinite(aspl) & (clustering > 0.0) & (
        aspl > 0.0
    )
    if not np.any(usable):
        return None
    kept = null_batch[usable]
    c_rand = float(np.mean(clustering[usable]))
    l_rand = float(np.mean(aspl[usable]))
    rich_club = _mean_rich_club_normalized(observed, kept)
    return c_rand, l_rand, rich_club, int(kept.shape[0])
