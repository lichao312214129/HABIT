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
"""Pick a voxel-radiomics batch from this machine, or honor an explicit size.

HABIT's documented default is 1000. That value is safe on an 8 GB laptop
GPU (RTX 3070 OOMs at 4000). Callers may set a larger batch on a 24 GB
card. ``\"auto\"`` / ``HABIT_VOXEL_BATCH=auto`` scales the pick with
VRAM; an explicit integer is never reduced to 1000.
"""

from __future__ import annotations

import os
from typing import Optional, Union

from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

# Documented HABIT default. Auto-select uses this on ~8 GB GPUs.
DEFAULT_VOXEL_BATCH: int = 1000
MIN_AUTO_VOXEL_BATCH: int = 64

_ENV_KEY: str = "HABIT_VOXEL_BATCH"

VoxelBatchArg = Optional[Union[int, str]]


def _cuda_total_gb(torch_device: str) -> Optional[float]:
    """Return total CUDA memory in GiB, or None when CUDA is unusable.

    Args:
        torch_device: ``auto``, ``cpu``, or a torch device string.

    Returns:
        Total device memory in GiB, or ``None``.
    """
    if str(torch_device).strip().lower() == "cpu":
        return None
    try:
        import torch

        if not torch.cuda.is_available() or int(torch.cuda.device_count()) < 1:
            return None
        index = 0
        spec = str(torch_device).strip().lower()
        if spec.startswith("cuda:") and spec.split(":", 1)[1].isdigit():
            index = int(spec.split(":", 1)[1])
        props = torch.cuda.get_device_properties(index)
        return float(props.total_memory) / float(1024**3)
    except Exception:
        return None


def _system_ram_gb() -> Optional[float]:
    """Return installed RAM in GiB when the OS exposes it.

    Returns:
        Physical RAM in GiB, or ``None`` if the probe fails.
    """
    try:
        if hasattr(os, "sysconf"):
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page > 0:
                return float(pages * page) / float(1024**3)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import ctypes

        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
            return float(status.ullTotalPhys) / float(1024**3)
    except Exception:
        return None
    return None


def _parse_positive_batch(voxel_batch: int) -> int:
    """Validate an explicit batch. ``-1`` (all voxels) is kept.

    Args:
        voxel_batch: Caller-requested batch, or ``-1`` for all ROI voxels.

    Returns:
        The same integer when it is ``-1`` or ``>= 1``.

    Raises:
        ValueError: If ``voxel_batch`` is ``0`` or another negative value.
    """
    raw = int(voxel_batch)
    if raw == -1:
        logger.warning(
            "voxel_batch=-1 processes every ROI voxel in one shot and "
            "often OOMs on clinical GTVs."
        )
        return -1
    if raw < 1:
        raise ValueError(f"voxel_batch must be positive or -1; got {raw}.")
    return raw


def recommend_voxel_batch(
    *,
    kernel_radius: int = 3,
    torch_device: str = "auto",
) -> int:
    """Pick a batch size from GPU VRAM (or CPU RAM).

    Calibration at kernel radius 3, ~90 features, TorchRadiomics GPU
    matrices: 8 GB (RTX 3070 laptop) OOMs at 4000 and finishes at 1000;
    12 GB can take 2000; 24 GB cloud cards can take 4000. Larger kernels
    scale the recommendation down by ``(3 / radius)^2``.

    This is a recommendation only. Explicit ``voxel_batch`` arguments
    are not forced down to 1000.

    Args:
        kernel_radius: Voxel neighbourhood radius (1 ? 3x3x3, 3 ? 7x7x7).
        torch_device: Preferred torch device; ``cpu`` skips the CUDA probe.

    Returns:
        A positive batch size (at least ``MIN_AUTO_VOXEL_BATCH``).
    """
    radius = max(1, int(kernel_radius))
    total_gb = _cuda_total_gb(str(torch_device))
    if total_gb is not None:
        if total_gb >= 20.0:
            base = 4000
        elif total_gb >= 12.0:
            base = 2000
        elif total_gb >= 7.0:
            base = DEFAULT_VOXEL_BATCH
        elif total_gb >= 5.0:
            base = 400
        else:
            base = 128
    else:
        ram_gb = _system_ram_gb()
        if ram_gb is not None and ram_gb < 8.0:
            base = 256
        else:
            base = DEFAULT_VOXEL_BATCH

    if radius > 3:
        scaled = int(base * (3.0 / float(radius)) ** 2)
        base = max(MIN_AUTO_VOXEL_BATCH, scaled)
    return max(MIN_AUTO_VOXEL_BATCH, int(base))


def resolve_voxel_batch(
    voxel_batch: VoxelBatchArg = DEFAULT_VOXEL_BATCH,
    *,
    kernel_radius: int = 3,
    torch_device: str = "auto",
) -> int:
    """Resolve ``auto`` / ``None`` / an integer.

    ``None`` and ``\"auto\"`` probe the machine. An integer (including
    4000 on a big GPU) is used as given. ``HABIT_VOXEL_BATCH`` overrides
    ``auto``/``None`` only; it does not override an explicit integer.

    Args:
        voxel_batch: ``\"auto\"``, ``None``, or an integer (``-1`` = all).
        kernel_radius: Forwarded to :func:`recommend_voxel_batch`.
        torch_device: Forwarded to :func:`recommend_voxel_batch`.

    Returns:
        The batch size that extractors should pass to PyRadiomics.
    """
    if voxel_batch is None or (
        isinstance(voxel_batch, str) and voxel_batch.strip().lower() in {"", "auto"}
    ):
        env_raw = os.environ.get(_ENV_KEY, "").strip()
        if env_raw and env_raw.lower() not in {"", "auto"}:
            return _parse_positive_batch(int(env_raw))
        return recommend_voxel_batch(
            kernel_radius=kernel_radius, torch_device=torch_device
        )
    if isinstance(voxel_batch, str):
        return _parse_positive_batch(int(voxel_batch.strip()))
    return _parse_positive_batch(int(voxel_batch))
