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
"""Closest-voxel distances for habitat-graph ``min_distance`` edges.

The production path is ``scipy.spatial.cKDTree`` on CPU. Graph extracts
issue thousands of small cloud pairs (8-voxel cubes); CUDA ``cdist`` plus
per-call ``empty_cache`` is slower than the tree on that pattern. Torch
CUDA remains available only when the caller passes ``device="cuda"``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from habit.utils.torch_radiomics_utils import is_cuda_available, is_torch_available

# Product ``n_a * n_b`` above this uses the CPU tree even when CUDA is up.
# 8e6 float32 distances is about 32 MiB plus workspace -- safe on 8 GB GPUs.
_MAX_PAIRWISE_PRODUCT: int = 8_000_000
# Refuse to materialize a cloud larger than this on device (chunk the other
# side against it). Beyond this, the tree is the right algorithm.
_MAX_CLOUD_ON_DEVICE: int = 200_000


def min_voxel_distance(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    *,
    device: str = "auto",
) -> float:
    """
    Closest-point Euclidean distance between two voxel-index clouds.

    Args:
        coords_a: Coordinates of set A, shape ``(n_a, ndim)``.
        coords_b: Coordinates of set B, shape ``(n_b, ndim)``.
        device: ``auto`` / ``cpu`` (CPU kd-tree), or ``cuda`` / ``cuda:N``
            (require CUDA; fall back to the tree when the pairwise
            product is too large).

    Returns:
        Minimum pairwise Euclidean distance, or ``inf`` if either set
        is empty.
    """
    cloud_a = np.asarray(coords_a, dtype=np.float64)
    cloud_b = np.asarray(coords_b, dtype=np.float64)
    if cloud_a.size == 0 or cloud_b.size == 0:
        return float("inf")
    if cloud_a.ndim != 2 or cloud_b.ndim != 2:
        raise ValueError("Coordinate arrays must have shape (n_voxels, ndim).")

    resolved = _resolve_distance_device(
        device, n_a=int(cloud_a.shape[0]), n_b=int(cloud_b.shape[0])
    )
    if resolved is None:
        return _min_voxel_distance_tree(cloud_a, cloud_b)
    return _min_voxel_distance_torch(cloud_a, cloud_b, resolved)


def _resolve_distance_device(device: str, n_a: int, n_b: int) -> Optional[str]:
    """
    Choose a torch device string, or ``None`` to use the CPU tree.

    Args:
        device: User request (``auto``, ``cpu``, ``cuda``, ``cuda:N``).
        n_a: Number of points in the first cloud.
        n_b: Number of points in the second cloud.

    Returns:
        A torch device string, or ``None`` when the tree should run.

    Raises:
        RuntimeError: If CUDA is requested but unavailable.
    """
    product = int(n_a) * int(n_b)
    too_big = (
        product > _MAX_PAIRWISE_PRODUCT
        or max(int(n_a), int(n_b)) > _MAX_CLOUD_ON_DEVICE
    )
    requested = str(device).strip().lower()
    if requested in {"cpu", "off", "tree"}:
        return None
    if requested == "auto":
        # Default stays on the CPU tree. Per-pair CUDA cdist is slower
        # for the many small node clouds that min_distance actually emits.
        return None
    if requested.startswith("cuda"):
        if not is_torch_available() or not is_cuda_available():
            raise RuntimeError(
                f"min_voxel_distance device={device!r} needs CUDA; "
                "it is not available in this process."
            )
        if too_big:
            return None
        return "cuda:0" if requested == "cuda" else str(device)
    raise ValueError(
        f"device must be auto, cpu, cuda, or cuda:N; got {device!r}."
    )


def _min_voxel_distance_tree(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """CPU set-separation distance via a kd-tree of the larger cloud."""
    if coords_a.shape[0] <= coords_b.shape[0]:
        tree = cKDTree(coords_b)
        distances, _ = tree.query(coords_a, k=1)
    else:
        tree = cKDTree(coords_a)
        distances, _ = tree.query(coords_b, k=1)
    return float(np.min(distances))


def _min_voxel_distance_torch(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    device: str,
) -> float:
    """
    Chunked ``torch.cdist`` minimum, querying the smaller cloud in rows.

    Args:
        coords_a: First cloud, ``(n_a, ndim)``.
        coords_b: Second cloud, ``(n_b, ndim)``.
        device: Concrete torch device (already resolved).

    Returns:
        Minimum pairwise distance as a Python float.
    """
    import torch

    query = coords_a
    library = coords_b
    if query.shape[0] > library.shape[0]:
        query, library = library, query
    n_query = int(query.shape[0])
    n_library = int(library.shape[0])
    chunk = max(1, int(_MAX_PAIRWISE_PRODUCT // max(n_library, 1)))
    lib_t = torch.as_tensor(library, device=device, dtype=torch.float32)
    best = torch.tensor(float("inf"), device=device, dtype=torch.float32)
    try:
        for start in range(0, n_query, chunk):
            stop = min(start + chunk, n_query)
            query_t = torch.as_tensor(
                query[start:stop], device=device, dtype=torch.float32
            )
            best = torch.minimum(best, torch.cdist(query_t, lib_t).amin())
        return float(best.detach().cpu().item())
    finally:
        del lib_t
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
