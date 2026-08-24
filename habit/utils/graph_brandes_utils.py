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
"""Unweighted Brandes + hop summaries on a CSR adjacency.

Definitions match NetworkX ``betweenness_centrality(normalized=True)``,
``closeness_centrality`` on a connected graph, ``average_shortest_path_length``,
and ``diameter``. One all-sources BFS yields all four.

Backends (``device='auto'`` picks the fastest that is available):

* ``cuda`` -- batched all-sources BFS in PyTorch (tiny float drift is
  accepted; intended for n ? 256 on a sparse lattice graph).
* ``numba`` -- compiled per-source Brandes (bit-exact vs the Python
  reference on typical graphs).
* ``python`` -- reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from habit.utils.torch_radiomics_utils import is_cuda_available, is_torch_available

__all__ = [
    "HopArrays",
    "csr_from_undirected_edges",
    "csr_from_edge_arrays",
    "hop_metrics_csr",
]

# GPU launch + nn buffers pay off only on larger sparse graphs.
_CUDA_MIN_NODES: int = 256
_CUDA_MAX_NODES: int = 8000

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional accelerator
    njit = None
    _HAS_NUMBA = False


@dataclass(frozen=True)
class HopArrays:
    """Hop summaries keyed by integer CSR node index ``0 .. n-1``."""

    betweenness: np.ndarray
    closeness: np.ndarray
    avg_path_length: float
    diameter: float


def csr_from_edge_arrays(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an undirected CSR from integer endpoint arrays.

    Each undirected pair is stored in both directions. Self-loops are
    dropped. Duplicate pairs are kept (degree counts them twice); callers
    that need a simple graph should unique first.

    Args:
        n_nodes: Node count.
        src: Source indices, shape ``(m,)``.
        dst: Target indices, shape ``(m,)``.

    Returns:
        ``(indptr, indices)`` as ``int64`` arrays.
    """
    if n_nodes < 0:
        raise ValueError("n_nodes must be >= 0.")
    n_nodes = int(n_nodes)
    src_i = np.asarray(src, dtype=np.int64).reshape(-1)
    dst_i = np.asarray(dst, dtype=np.int64).reshape(-1)
    if src_i.size == 0:
        return np.zeros(n_nodes + 1, dtype=np.int64), np.empty(0, dtype=np.int64)
    keep = src_i != dst_i
    src_i = src_i[keep]
    dst_i = dst_i[keep]
    both_src = np.concatenate((src_i, dst_i))
    both_dst = np.concatenate((dst_i, src_i))
    order = np.argsort(both_src, kind="stable")
    both_src = both_src[order]
    both_dst = both_dst[order]
    counts = np.bincount(both_src, minlength=n_nodes).astype(np.int64, copy=False)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    return indptr, both_dst.astype(np.int64, copy=False)


def csr_from_undirected_edges(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an undirected CSR (both directions) from integer endpoint pairs.

    Args:
        n_nodes: Node count.
        edges: Undirected pairs ``(i, j)`` with ``0 <= i, j < n_nodes``.
            Self-loops are ignored.

    Returns:
        ``(indptr, indices)`` as ``int64`` arrays.
    """
    if not edges:
        return csr_from_edge_arrays(n_nodes, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    packed = np.asarray(edges, dtype=np.int64)
    if packed.ndim != 2 or packed.shape[1] != 2:
        packed = np.asarray(list(edges), dtype=np.int64)
    return csr_from_edge_arrays(n_nodes, packed[:, 0], packed[:, 1])


def hop_metrics_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_nodes: int,
    *,
    device: str = "auto",
) -> HopArrays:
    """
    Brandes / closeness / ASPL / diameter on one unweighted CSR graph.

    Args:
        indptr: CSR row pointer, length ``n_nodes + 1``.
        indices: CSR column indices (undirected: both directions stored).
        n_nodes: Node count.
        device: ``auto``, ``cpu``, ``numba``, ``python``, or ``cuda`` /
            ``cuda:N``. ``auto`` uses CUDA when it is profitable, else
            numba, else the Python reference.

    Returns:
        HopArrays: Per-node centralities and two scalars.
    """
    n_nodes = int(n_nodes)
    betweenness = np.zeros(n_nodes, dtype=np.float64)
    closeness = np.zeros(n_nodes, dtype=np.float64)
    if n_nodes == 0:
        return HopArrays(betweenness, closeness, 0.0, 0.0)
    if n_nodes == 1:
        return HopArrays(betweenness, closeness, 0.0, 0.0)

    resolved = _resolve_hop_device(device, n_nodes)
    if resolved.startswith("cuda"):
        try:
            return _hop_torch(indptr, indices, n_nodes, resolved)
        except Exception:
            resolved = "numba" if _HAS_NUMBA else "python"
    if resolved == "numba" and _HAS_NUMBA and _hop_numba is not None:
        return _arrays_from_numba(_hop_numba(indptr, indices, n_nodes))
    return _hop_python(indptr, indices, n_nodes)


def _resolve_hop_device(device: str, n_nodes: int) -> str:
    """Pick a backend string for :func:`hop_metrics_csr`."""
    requested = str(device).strip().lower()
    if requested in {"python", "reference"}:
        return "python"
    if requested == "numba":
        return "numba" if _HAS_NUMBA else "python"
    if requested in {"cpu", "auto"}:
        # CUDA Brandes is opt-in (``device='cuda'``). The batched
        # layer-DAG path is fast but its scatter-add order does not yet
        # match NetworkX on every lattice graph; auto stays on compiled
        # CPU Brandes so default numbers stay correct.
        return "numba" if _HAS_NUMBA else "python"
    if requested.startswith("cuda"):
        if not is_torch_available() or not is_cuda_available():
            raise RuntimeError(
                f"hop_metrics device={device!r} needs CUDA; "
                "it is not available in this process."
            )
        if n_nodes > _CUDA_MAX_NODES:
            return "numba" if _HAS_NUMBA else "python"
        return "cuda:0" if requested == "cuda" else str(device)
    raise ValueError(
        "device must be auto, cpu, numba, python, cuda, or cuda:N; "
        f"got {device!r}."
    )


def _arrays_from_numba(payload: Tuple[np.ndarray, np.ndarray, float, float]) -> HopArrays:
    """Pack the numba tuple into :class:`HopArrays`."""
    betweenness, closeness, avg_path, diameter = payload
    return HopArrays(
        betweenness=np.asarray(betweenness, dtype=np.float64),
        closeness=np.asarray(closeness, dtype=np.float64),
        avg_path_length=float(avg_path),
        diameter=float(diameter),
    )


def _scale_betweenness(betweenness: np.ndarray, n_nodes: int) -> None:
    """In-place NetworkX normalized rescaling for undirected Brandes."""
    if n_nodes > 2:
        betweenness *= 1.0 / float((n_nodes - 1) * (n_nodes - 2))


def _hop_python(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_nodes: int,
) -> HopArrays:
    """Reference all-sources Brandes (same recurrence as NetworkX)."""
    betweenness = np.zeros(n_nodes, dtype=np.float64)
    closeness = np.zeros(n_nodes, dtype=np.float64)
    pair_distance_sum = 0.0
    diameter = 0.0
    for source in range(n_nodes):
        order: List[int] = []
        predecessors: List[List[int]] = [[] for _ in range(n_nodes)]
        n_paths = np.zeros(n_nodes, dtype=np.float64)
        dist = np.full(n_nodes, -1, dtype=np.int32)
        n_paths[source] = 1.0
        dist[source] = 0
        queue: List[int] = [source]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            order.append(node)
            start = int(indptr[node])
            stop = int(indptr[node + 1])
            node_dist = int(dist[node])
            node_paths = float(n_paths[node])
            for slot in range(start, stop):
                neighbour = int(indices[slot])
                if dist[neighbour] < 0:
                    queue.append(neighbour)
                    dist[neighbour] = node_dist + 1
                if dist[neighbour] == node_dist + 1:
                    n_paths[neighbour] += node_paths
                    predecessors[neighbour].append(node)
        dependency = np.zeros(n_nodes, dtype=np.float64)
        for node in reversed(order):
            coeff = (1.0 + float(dependency[node])) / float(n_paths[node])
            for pred in predecessors[node]:
                dependency[pred] += float(n_paths[pred]) * coeff
            if node != source:
                betweenness[node] += float(dependency[node])
        reachable_sum = 0.0
        for node in range(n_nodes):
            hops = int(dist[node])
            if hops <= 0:
                continue
            hop_f = float(hops)
            pair_distance_sum += hop_f
            reachable_sum += hop_f
            if hop_f > diameter:
                diameter = hop_f
        if n_nodes > 1 and reachable_sum > 0.0:
            closeness[source] = float(n_nodes - 1) / reachable_sum
    _scale_betweenness(betweenness, n_nodes)
    denom = float(n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    avg_path = pair_distance_sum / denom if denom > 0.0 else 0.0
    return HopArrays(betweenness, closeness, float(avg_path), float(diameter))


def _hop_torch(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_nodes: int,
    device: str,
) -> HopArrays:
    """
    Batched all-sources BFS + layer Brandes on one GPU (or torch CPU).

    Distances are exact integers. Betweenness uses float64 scatter-adds;
    summation order can differ from NetworkX by a tiny relative amount.
    """
    import torch

    src, dst = _csr_to_coo(indptr, indices)
    src_t = torch.as_tensor(src, device=device, dtype=torch.int64)
    dst_t = torch.as_tensor(dst, device=device, dtype=torch.int64)
    dist = torch.full((n_nodes, n_nodes), -1, device=device, dtype=torch.int32)
    sigma = torch.zeros((n_nodes, n_nodes), device=device, dtype=torch.float64)
    index = torch.arange(n_nodes, device=device)
    dist[index, index] = 0
    sigma[index, index] = 1.0

    for hop in range(n_nodes - 1):
        at_parent = dist[:, src_t] == hop
        if not bool(at_parent.any().item()):
            break
        child_dist = dist[:, dst_t]
        discover = at_parent & (child_dist < 0)
        if bool(discover.any().item()):
            src_idx, edge_idx = torch.nonzero(discover, as_tuple=True)
            dist[src_idx, dst_t[edge_idx]] = hop + 1
        parent_ok = (dist[:, src_t] == hop) & (dist[:, dst_t] == hop + 1)
        if bool(parent_ok.any().item()):
            src_idx, edge_idx = torch.nonzero(parent_ok, as_tuple=True)
            sigma[src_idx, dst_t[edge_idx]] += sigma[src_idx, src_t[edge_idx]]

    delta = torch.zeros((n_nodes, n_nodes), device=device, dtype=torch.float64)
    max_dist = int(dist.max().item()) if n_nodes else 0
    for hop in range(max_dist, 0, -1):
        mask = (dist[:, dst_t] == hop) & (dist[:, src_t] == hop - 1)
        if not bool(mask.any().item()):
            continue
        src_idx, edge_idx = torch.nonzero(mask, as_tuple=True)
        parent = src_t[edge_idx]
        child = dst_t[edge_idx]
        coeff = (sigma[src_idx, parent] / sigma[src_idx, child]) * (
            1.0 + delta[src_idx, child]
        )
        delta.index_put_((src_idx, parent), coeff, accumulate=True)

    betweenness = delta.sum(dim=0)
    betweenness = betweenness - torch.diag(delta)
    bc = betweenness.detach().cpu().numpy().astype(np.float64, copy=False)
    _scale_betweenness(bc, n_nodes)

    reachable = dist > 0
    hop_f = dist.to(dtype=torch.float64).clamp(min=0)
    hop_f = torch.where(reachable, hop_f, torch.zeros_like(hop_f))
    reachable_sum = hop_f.sum(dim=1)
    closeness = torch.zeros(n_nodes, device=device, dtype=torch.float64)
    ok = reachable_sum > 0.0
    closeness[ok] = float(n_nodes - 1) / reachable_sum[ok]
    pair_sum = float(hop_f.sum().item())
    denom = float(n_nodes * (n_nodes - 1))
    avg_path = pair_sum / denom if denom > 0.0 else 0.0
    diameter = float(max_dist) if n_nodes > 1 else 0.0
    return HopArrays(
        betweenness=bc,
        closeness=closeness.detach().cpu().numpy().astype(np.float64, copy=False),
        avg_path_length=avg_path,
        diameter=diameter,
    )


def _csr_to_coo(indptr: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Expand CSR to directed ``(src, dst)`` edge arrays."""
    n_nodes = int(indptr.shape[0] - 1)
    src = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr))
    dst = np.asarray(indices, dtype=np.int64)
    return src, dst


if _HAS_NUMBA:

    @njit(cache=True, nogil=True)
    def _hop_numba(
        indptr: np.ndarray,
        indices: np.ndarray,
        n_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Compiled Brandes. Sources stay serial so many graphs can run in threads."""
        betweenness = np.zeros(n_nodes, dtype=np.float64)
        closeness = np.zeros(n_nodes, dtype=np.float64)
        pair_parts = np.zeros(n_nodes, dtype=np.float64)
        diam_parts = np.zeros(n_nodes, dtype=np.int32)
        n_pred_max = indices.shape[0]
        for source in range(n_nodes):
            dist = np.empty(n_nodes, dtype=np.int32)
            n_paths = np.empty(n_nodes, dtype=np.float64)
            dependency = np.empty(n_nodes, dtype=np.float64)
            order = np.empty(n_nodes, dtype=np.int32)
            pred = np.empty(n_pred_max, dtype=np.int32)
            pred_parent = np.empty(n_pred_max, dtype=np.int32)
            pred_child = np.empty(n_pred_max, dtype=np.int32)
            pred_count = np.zeros(n_nodes, dtype=np.int32)
            for node in range(n_nodes):
                dist[node] = -1
                n_paths[node] = 0.0
                dependency[node] = 0.0
                pred_count[node] = 0
            dist[source] = 0
            n_paths[source] = 1.0
            queue = np.empty(n_nodes, dtype=np.int32)
            queue[0] = source
            head = 0
            tail = 1
            n_order = 0
            n_pred = 0
            while head < tail:
                node = queue[head]
                head += 1
                order[n_order] = node
                n_order += 1
                node_dist = dist[node]
                node_paths = n_paths[node]
                start = indptr[node]
                stop = indptr[node + 1]
                for slot in range(start, stop):
                    neighbour = indices[slot]
                    if dist[neighbour] < 0:
                        queue[tail] = neighbour
                        tail += 1
                        dist[neighbour] = node_dist + 1
                    if dist[neighbour] == node_dist + 1:
                        n_paths[neighbour] += node_paths
                        pred_parent[n_pred] = node
                        pred_child[n_pred] = neighbour
                        n_pred += 1
                        pred_count[neighbour] += 1
            pred_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
            for node in range(n_nodes):
                pred_ptr[node + 1] = pred_ptr[node] + pred_count[node]
            fill = pred_ptr.copy()
            for slot in range(n_pred):
                child = pred_child[slot]
                pred[fill[child]] = pred_parent[slot]
                fill[child] += 1
            reachable_sum = 0.0
            local_diam = 0
            for node in range(n_nodes):
                hops = dist[node]
                if hops > 0:
                    hop_f = float(hops)
                    reachable_sum += hop_f
                    if hops > local_diam:
                        local_diam = hops
            pair_parts[source] = reachable_sum
            diam_parts[source] = local_diam
            if n_nodes > 1 and reachable_sum > 0.0:
                closeness[source] = float(n_nodes - 1) / reachable_sum
            for rev in range(n_order - 1, -1, -1):
                node = order[rev]
                if n_paths[node] == 0.0:
                    continue
                coeff = (1.0 + dependency[node]) / n_paths[node]
                p0 = pred_ptr[node]
                p1 = pred_ptr[node + 1]
                for slot in range(p0, p1):
                    parent = pred[slot]
                    dependency[parent] += n_paths[parent] * coeff
                if node != source:
                    betweenness[node] += dependency[node]
        if n_nodes > 2:
            scale = 1.0 / float((n_nodes - 1) * (n_nodes - 2))
            for node in range(n_nodes):
                betweenness[node] *= scale
        pair_sum = 0.0
        diameter = 0
        for source in range(n_nodes):
            pair_sum += pair_parts[source]
            if diam_parts[source] > diameter:
                diameter = diam_parts[source]
        denom = float(n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
        avg_path = pair_sum / denom if denom > 0.0 else 0.0
        return betweenness, closeness, avg_path, float(diameter)

else:  # pragma: no cover - no numba
    _hop_numba = None
