"""
GPU slot helpers for parallel habitat / radiomics workers.

Spawn children receive a zero-based worker slot index via ``HABIT_GPU_SLOT_INDEX``.
Feature extractors map that slot to ``cuda:torchGpus[slot]`` when ``gpuSlotIndex`` is
not set explicitly in YAML.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from habit.utils.torch_radiomics_utils import (
    apply_torch_gpu_count,
    is_cuda_available,
    normalize_use_torch_radiomics,
    parse_torch_gpu_indices,
)

HABIT_GPU_SLOT_INDEX_ENV: str = "HABIT_GPU_SLOT_INDEX"

logger = logging.getLogger(__name__)


def read_worker_gpu_slot_index() -> Optional[int]:
    """
    Read the parallel worker GPU slot index from the current process environment.

    Returns:
        int | None: Slot index when set by :class:`~habit.utils.isolated_runner.IsolatedTaskRunner`;
        otherwise ``None``.
    """
    raw_value = os.environ.get(HABIT_GPU_SLOT_INDEX_ENV)
    if raw_value is None or raw_value == "":
        return None
    try:
        return int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; expected integer slot index.",
            HABIT_GPU_SLOT_INDEX_ENV,
            raw_value,
        )
        return None


def inject_worker_gpu_slot_index(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject ``gpuSlotIndex`` from the worker environment when YAML did not set it.

    Args:
        params: Extractor kwargs / step params (mutated only when injection applies).

    Returns:
        Dict[str, Any]: Same mapping or a shallow copy with ``gpuSlotIndex`` added.
    """
    if params.get("gpuSlotIndex") is not None:
        return params

    slot_index = read_worker_gpu_slot_index()
    if slot_index is None:
        return params

    updated = dict(params)
    updated["gpuSlotIndex"] = slot_index
    return updated


def resolve_habitat_torch_gpu_pool(config: Any) -> List[int]:
    """
    Resolve the effective Torch GPU index pool from a habitat analysis config.

    Reads ``FeatureConstruction.voxel_level.params`` for ``useTorchRadiomics``,
    ``torchGpus``, and ``torchGpuCount``. When torch mode is enabled (``true`` or
    ``auto`` with CUDA available) but ``torchGpus`` is omitted, returns ``[0]``.

    Args:
        config: Validated :class:`~habit.core.habitat_analysis.config_schemas.HabitatAnalysisConfig`
            or any object with ``FeatureConstruction.voxel_level.params``.

    Returns:
        List[int]: CUDA device indices used for process capping; empty when CPU-only.
    """
    feature_construction = getattr(config, "FeatureConstruction", None)
    if feature_construction is None:
        return []

    voxel_level = getattr(feature_construction, "voxel_level", None)
    if voxel_level is None:
        return []

    params = getattr(voxel_level, "params", None) or {}
    use_torch = normalize_use_torch_radiomics(params.get("useTorchRadiomics", "auto"))
    if use_torch == "false":
        return []

    gpu_indices = parse_torch_gpu_indices(params.get("torchGpus"))
    gpu_indices = apply_torch_gpu_count(gpu_indices, params.get("torchGpuCount"))

    if not gpu_indices and use_torch in ("true", "auto") and is_cuda_available():
        gpu_indices = [0]

    return gpu_indices


def apply_gpu_pool_process_cap(
    requested_processes: int,
    config: Any,
    *,
    log: Optional[logging.Logger] = None,
) -> int:
    """
    Apply optional GPU-pool capping to a configured parallel worker count.

    When ``config.cap_processes_to_gpu_pool`` is False, returns the requested count
    unchanged so CPU-heavy individual steps can use the full ``processes`` value while
    Torch radiomics workers share GPUs via ``gpu_slot_index % len(gpu_pool)``.

    Args:
        requested_processes: User-configured ``processes`` value (or equivalent).
        config: Habitat analysis config object.
        log: Optional logger for a one-line warning when capping occurs.

    Returns:
        int: Effective worker count (>= 1 when ``requested_processes`` >= 1).
    """
    requested = max(1, int(requested_processes))
    if config is None:
        return requested

    if not getattr(config, "cap_processes_to_gpu_pool", True):
        return requested

    gpu_pool = resolve_habitat_torch_gpu_pool(config)
    if not gpu_pool:
        return requested

    return cap_processes_to_gpu_pool(
        requested,
        len(gpu_pool),
        log=log,
        gpu_pool=gpu_pool,
    )


def cap_processes_to_gpu_pool(
    requested_processes: int,
    gpu_pool_size: int,
    *,
    log: Optional[logging.Logger] = None,
    gpu_pool: Optional[List[int]] = None,
) -> int:
    """
    Cap parallel worker count so it does not exceed the Torch GPU pool size.

    Args:
        requested_processes: User-configured ``processes`` value.
        gpu_pool_size: Number of GPUs in the active pool (0 skips capping).
        log: Optional logger for a one-line warning when capping occurs.
        gpu_pool: Optional GPU index list for log context.

    Returns:
        int: Effective worker count (>= 1 when ``requested_processes`` >= 1).
    """
    requested = max(1, int(requested_processes))
    if gpu_pool_size <= 0:
        return requested

    capped = min(requested, gpu_pool_size)
    if capped < requested and log is not None:
        pool_repr = gpu_pool if gpu_pool is not None else f"size={gpu_pool_size}"
        log.warning(
            "Capping parallel workers %s -> %s to match Torch GPU pool (%s). "
            "Each active worker binds to one GPU slot via gpuSlotIndex.",
            requested,
            capped,
            pool_repr,
        )
    return max(1, capped)
