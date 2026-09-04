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
"""
GPU slot helpers for parallel habitat / radiomics workers.

Spawn children receive a zero-based worker slot index via ``HABIT_GPU_SLOT_INDEX``.
Feature extractors map that slot to ``cuda:torch_gpus[slot]`` when ``gpu_slot_index`` is
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

#: When set to ``"1"`` in a worker, radiomics ``auto`` mode prefers CPU
#: PyRadiomics so extra workers on a single-GPU host do not thrash CUDA.
HABIT_FORCE_CPU_RADIOMICS_ENV: str = "HABIT_FORCE_CPU_RADIOMICS"

#: ``wrap`` keeps modulo assignment when workers > GPUs; anything else
#: (default) forces CPU for worker indices beyond the visible GPU pool
#: when only one GPU is visible.
HABIT_GPU_OVERSUBSCRIBE_ENV: str = "HABIT_GPU_OVERSUBSCRIBE"

logger = logging.getLogger(__name__)


def _unmasked_host_gpu_count() -> int:
    """Count host NVIDIA GPUs without importing torch.

    Prefers ``/dev/nvidiaN`` (Linux), then ``nvidia-smi -L``. Returns 0
    when neither is available so the caller can fall back to ``\"0\"``.

    Returns:
        int: Physical GPU count, or 0 when it cannot be determined.
    """
    n_dev = 0
    while os.path.exists(f"/dev/nvidia{n_dev}"):
        n_dev += 1
    if n_dev > 0:
        return n_dev
    try:
        import subprocess

        listing = subprocess.check_output(
            ["nvidia-smi", "-L"],
            text=True,
            timeout=5,
        )
        return sum(
            1 for line in listing.splitlines() if line.startswith("GPU ")
        )
    except Exception:
        return 0


def pin_worker_visible_cuda_device(worker_index: int) -> str:
    """Restrict this process to one GPU before torch initializes.

    Process-pool children previously kept every host GPU visible and
    selected ``cuda:N`` via ``HABIT_GPU_SLOT_INDEX``. Torch still builds a
    context on ``cuda:0``, so two workers can pile kernels onto GPU 0
    while GPU 1 stays idle. Masking ``CUDA_VISIBLE_DEVICES`` to a single
    id makes that GPU appear as ``cuda:0``; the slot index is then 0.

    On a **single-GPU** host with many CPU workers, workers beyond slot 0
    default to ``CUDA_VISIBLE_DEVICES=-1`` (CPU-preferred radiomics) so
    subject-level parallelism is not killed by CUDA context thrashing.
    Set ``HABIT_GPU_OVERSUBSCRIBE=wrap`` to restore modulo wrapping onto
    the same card. Multi-GPU pools still wrap across the visible list.

    Must run before ``import torch`` in the child.

    Args:
        worker_index: Zero-based worker slot.

    Returns:
        str: The single ``CUDA_VISIBLE_DEVICES`` value assigned (``"-1"``
        when the worker is forced onto CPU).
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw == "-1":
        os.environ.pop(HABIT_GPU_SLOT_INDEX_ENV, None)
        os.environ[HABIT_FORCE_CPU_RADIOMICS_ENV] = "1"
        logger.info(
            "Worker %s already has CUDA_VISIBLE_DEVICES=-1; forcing CPU radiomics",
            worker_index,
        )
        return "-1"
    if raw:
        pool = [part.strip() for part in raw.split(",") if part.strip() != ""]
    else:
        n_host = _unmasked_host_gpu_count()
        pool = [str(i) for i in range(n_host)] if n_host > 0 else ["0"]
    if not pool:
        pool = ["0"]

    slot = int(worker_index)
    oversubscribe = os.environ.get(HABIT_GPU_OVERSUBSCRIBE_ENV, "").strip().lower()
    # Single visible GPU + extra workers: prefer CPU unless the user asks
    # to wrap every worker onto that one card.
    if len(pool) == 1 and slot > 0 and oversubscribe != "wrap":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.pop(HABIT_GPU_SLOT_INDEX_ENV, None)
        os.environ[HABIT_FORCE_CPU_RADIOMICS_ENV] = "1"
        logger.info(
            "Pinned worker %s to CPU (CUDA_VISIBLE_DEVICES=-1); "
            "single-GPU host avoids CUDA thrashing "
            "(set HABIT_GPU_OVERSUBSCRIBE=wrap to share the card)",
            worker_index,
        )
        return "-1"

    chosen = pool[slot % len(pool)]
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    # Only one device is visible now; extractors must use cuda:0.
    os.environ[HABIT_GPU_SLOT_INDEX_ENV] = "0"
    os.environ.pop(HABIT_FORCE_CPU_RADIOMICS_ENV, None)
    logger.info(
        "Pinned worker %s to CUDA_VISIBLE_DEVICES=%s (HABIT_GPU_SLOT_INDEX=0)",
        worker_index,
        chosen,
    )
    return chosen


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
    Inject ``gpu_slot_index`` from the worker environment when YAML did not set it.

    Args:
        params: Extractor kwargs / step params (mutated only when injection applies).

    Returns:
        Dict[str, Any]: Same mapping or a shallow copy with ``gpu_slot_index`` added.
    """
    if params.get("gpu_slot_index") is not None:
        return params

    slot_index = read_worker_gpu_slot_index()
    if slot_index is None:
        return params

    updated = dict(params)
    updated["gpu_slot_index"] = slot_index
    return updated


def resolve_habitat_torch_gpu_pool(config: Any) -> List[int]:
    """
    Resolve the effective Torch GPU index pool from a habitat analysis config.

    Reads ``feature_construction.voxel_level.params`` for ``use_torch_radiomics``,
    ``torch_gpus``, and ``torch_gpu_count``. When torch mode is enabled (``true`` or
    ``auto`` with CUDA available) but ``torch_gpus`` is omitted, returns ``[0]``.

    Args:
        config: Validated :class:`~habit.core.habitat_analysis.config_schemas.HabitatAnalysisConfig`
            or any object with ``feature_construction.voxel_level.params``.

    Returns:
        List[int]: CUDA device indices used for process capping; empty when CPU-only.
    """
    feature_construction = getattr(config, "feature_construction", None)
    if feature_construction is None:
        return []

    voxel_level = getattr(feature_construction, "voxel_level", None)
    if voxel_level is None:
        return []

    params = getattr(voxel_level, "params", None) or {}
    use_torch = normalize_use_torch_radiomics(params.get("use_torch_radiomics", "auto"))
    if use_torch == "false":
        return []

    gpu_indices = parse_torch_gpu_indices(params.get("torch_gpus"))
    gpu_indices = apply_torch_gpu_count(gpu_indices, params.get("torch_gpu_count"))

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

    if not getattr(config, "cap_processes_to_gpu_pool", False):
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
        # Informational: this is the intended cap, not a failure. Warning
        # level made long Torch runs look like GPU contention errors.
        log.debug(
            "Capping parallel workers %s -> %s to match Torch GPU pool (%s). "
            "Each active worker binds to one GPU slot via gpu_slot_index.",
            requested,
            capped,
            pool_repr,
        )
    return max(1, capped)
