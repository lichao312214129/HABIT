"""
Optional TorchRadiomics backend helpers for voxel-based PyRadiomics extraction.

Torch is an optional dependency: these helpers lazy-import torch and fall back
to conventional CPU PyRadiomics when unavailable.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from typing import Any, Iterator, List, Literal, Optional, Sequence, Tuple, Union

from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

VoxelRadiomicsBackend = Literal["pyradiomics", "torch"]
UseTorchRadiomicsSetting = Union[bool, str]
TorchGpuSetting = Union[int, str, Sequence[Union[int, str]], None]

DEFAULT_USE_TORCH_RADIOMICS = "auto"
DEFAULT_TORCH_DEVICE = "auto"
DEFAULT_TORCH_DTYPE = "float64"

# Shown once per process when GPU TorchRadiomics is unavailable but would be used in auto mode.
_TORCH_GPU_INSTALL_HINT_LOGGED = False

TORCH_GPU_INSTALL_HINT = (
    "Install the CUDA-enabled PyTorch build for GPU-accelerated voxel_radiomics, e.g. "
    "pip install torch --index-url https://download.pytorch.org/whl/cu124 "
    "(pick the CUDA version matching your driver at https://pytorch.org/get-started/locally/). "
    "Or set useTorchRadiomics: false to keep CPU PyRadiomics without this message."
)


def reset_torch_gpu_install_hint_log() -> None:
    """Reset one-shot install hint logging (for tests only)."""
    global _TORCH_GPU_INSTALL_HINT_LOGGED
    _TORCH_GPU_INSTALL_HINT_LOGGED = False


def log_torch_gpu_install_hint(reason: str) -> None:
    """
    Log a one-time WARNING guiding users to install CUDA-enabled PyTorch.

    Args:
        reason: ``torch_not_installed`` or ``cuda_unavailable``.
    """
    global _TORCH_GPU_INSTALL_HINT_LOGGED
    if _TORCH_GPU_INSTALL_HINT_LOGGED:
        return
    _TORCH_GPU_INSTALL_HINT_LOGGED = True

    if reason == "torch_not_installed":
        logger.warning(
            "useTorchRadiomics=auto: torch is not installed; using CPU PyRadiomics. %s",
            TORCH_GPU_INSTALL_HINT,
        )
        return

    if reason == "cuda_unavailable":
        logger.warning(
            "useTorchRadiomics=auto: CUDA is unavailable (CPU-only PyTorch wheel or "
            "missing NVIDIA driver/GPU); using CPU PyRadiomics. %s",
            TORCH_GPU_INSTALL_HINT,
        )
        return

    logger.warning(
        "TorchRadiomics GPU acceleration unavailable (%s); using CPU PyRadiomics. %s",
        reason,
        TORCH_GPU_INSTALL_HINT,
    )


def is_torch_available() -> bool:
    """
    Check whether PyTorch can be imported in the current environment.

    Returns:
        bool: True when ``import torch`` succeeds.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def is_cuda_available() -> bool:
    """
    Check whether CUDA is available through an installed PyTorch build.

    Returns:
        bool: True when torch is installed and ``torch.cuda.is_available()``.
    """
    if not is_torch_available():
        return False
    import torch

    return bool(torch.cuda.is_available())


def normalize_use_torch_radiomics(value: UseTorchRadiomicsSetting) -> str:
    """
    Normalize user-facing ``useTorchRadiomics`` values to ``auto|true|false``.

    Args:
        value: Boolean or string setting from habitat config / kwargs.

    Returns:
        str: One of ``"auto"``, ``"true"``, or ``"false"``.

    Raises:
        ValueError: When the value is not recognized.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    normalized = str(value).strip().lower()
    if normalized in {"auto", "true", "false"}:
        return normalized
    raise ValueError(
        f"useTorchRadiomics must be auto, true, or false; got {value!r}"
    )


def resolve_torch_device(torch_device: str = DEFAULT_TORCH_DEVICE) -> str:
    """
    Resolve a torch device string from user config.

    Args:
        torch_device: ``auto``, ``cuda``, ``cuda:0``, ``cpu``, etc.

    Returns:
        str: Device string passed to TorchRadiomics settings.

    Raises:
        RuntimeError: When a CUDA device is requested but CUDA is unavailable.
    """
    if not is_torch_available():
        raise RuntimeError(
            "torch is not installed; install torch or set useTorchRadiomics to false/auto"
        )

    import torch

    normalized = str(torch_device).strip().lower()
    if normalized == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torchDevice requests CUDA but torch.cuda.is_available() is False"
            )
        return "cuda:0"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"torchDevice={torch_device!r} requests CUDA but CUDA is unavailable"
        )
    return str(torch_device)


def resolve_torch_dtype(dtype_name: str = DEFAULT_TORCH_DTYPE) -> Any:
    """
    Map a dtype name to a ``torch.dtype`` object.

    Args:
        dtype_name: ``float64`` or ``float32``.

    Returns:
        torch.dtype: Resolved dtype for TorchRadiomics settings.

    Raises:
        ValueError: When the dtype name is unsupported.
        RuntimeError: When torch is not installed.
    """
    if not is_torch_available():
        raise RuntimeError("torch is not installed")

    import torch

    mapping = {
        "float64": torch.float64,
        "float32": torch.float32,
    }
    normalized = str(dtype_name).strip().lower()
    if normalized not in mapping:
        raise ValueError(
            f"torchDtype must be one of {sorted(mapping.keys())}; got {dtype_name!r}"
        )
    return mapping[normalized]


def parse_torch_gpu_indices(value: TorchGpuSetting) -> List[int]:
    """
    Parse user ``torchGpus`` settings into a list of CUDA device indices.

    Accepts:
    - ``None`` or empty -> ``[]``
    - single int -> ``[int]``
    - ``"0,1,2"`` or ``"cuda:0,cuda:1"``
    - ``[0, 1, 2]`` or ``["cuda:0", "cuda:1"]``

    Args:
        value: Raw config value from habitat YAML / kwargs.

    Returns:
        List[int]: Sorted unique GPU indices in user order (duplicates removed).

    Raises:
        ValueError: When the value cannot be parsed.
    """
    if value is None:
        return []

    if isinstance(value, int):
        return [value]

    if isinstance(value, str):
        tokens = [part.strip() for part in value.split(",") if part.strip()]
        if not tokens:
            return []
        return [_parse_single_gpu_token(token) for token in tokens]

    if isinstance(value, Sequence):
        indices: List[int] = []
        for item in value:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str):
                indices.append(_parse_single_gpu_token(item.strip()))
            else:
                raise ValueError(f"Unsupported torchGpus entry: {item!r}")
        return _dedupe_preserve_order(indices)

    raise ValueError(f"torchGpus must be int, str, list, or null; got {value!r}")


def _parse_single_gpu_token(token: str) -> int:
    """Parse one GPU token such as ``0``, ``cuda:1``, or ``gpu2``."""
    normalized = token.strip().lower()
    if normalized.startswith("cuda:"):
        normalized = normalized.split(":", maxsplit=1)[1]
    if normalized.startswith("gpu"):
        normalized = normalized[3:]
    if not normalized.isdigit():
        raise ValueError(f"Invalid GPU token: {token!r}")
    return int(normalized)


def _dedupe_preserve_order(values: List[int]) -> List[int]:
    """Remove duplicate GPU indices while preserving first-seen order."""
    seen = set()
    unique: List[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def apply_torch_gpu_count(
    gpu_indices: List[int],
    torch_gpu_count: Optional[int] = None,
) -> List[int]:
    """
    Limit how many GPUs from ``torchGpus`` are actually used.

    Args:
        gpu_indices: Parsed GPU index list.
        torch_gpu_count: Maximum number of GPUs to use from the front of the list.

    Returns:
        List[int]: Possibly truncated GPU index list.
    """
    if torch_gpu_count is None:
        return gpu_indices
    if torch_gpu_count < 1:
        raise ValueError(f"torchGpuCount must be >= 1; got {torch_gpu_count}")
    return gpu_indices[:torch_gpu_count]


def stable_gpu_slot(key: str, modulo: int) -> int:
    """
    Map a stable string key (e.g. subject ID) to a GPU slot index.

    Args:
        key: Stable identifier such as a subject ID.
        modulo: Number of GPUs in the active pool.

    Returns:
        int: Slot index in ``[0, modulo)``.
    """
    if modulo < 1:
        raise ValueError("modulo must be >= 1")
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo


def select_torch_gpu_device(
    gpu_indices: List[int],
    *,
    subject: Optional[str] = None,
    gpu_slot_index: Optional[int] = None,
) -> str:
    """
    Pick one ``cuda:N`` device from an allowed GPU pool.

    Priority:
    1. Explicit ``gpu_slot_index`` (for future parallel worker wiring)
    2. Stable hash of ``subject`` when provided
    3. First GPU in the pool

    Args:
        gpu_indices: Allowed CUDA device indices.
        subject: Subject ID for stable multi-subject assignment.
        gpu_slot_index: Explicit slot index, e.g. worker id mod pool size.

    Returns:
        str: Device string such as ``cuda:1``.
    """
    if not gpu_indices:
        raise ValueError("gpu_indices must not be empty")

    if gpu_slot_index is not None:
        slot = gpu_slot_index % len(gpu_indices)
    elif subject is not None:
        slot = stable_gpu_slot(subject, len(gpu_indices))
    else:
        slot = 0
    return f"cuda:{gpu_indices[slot]}"


def validate_torch_gpu_indices(gpu_indices: List[int]) -> None:
    """
    Validate configured GPU indices against the current torch CUDA device count.

    Args:
        gpu_indices: Parsed GPU indices.

    Raises:
        RuntimeError: When torch/CUDA is unavailable or an index is out of range.
    """
    if not gpu_indices:
        return
    if not is_torch_available():
        raise RuntimeError("torch is not installed")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable but torchGpus was configured")
    device_count = torch.cuda.device_count()
    invalid = [idx for idx in gpu_indices if idx < 0 or idx >= device_count]
    if invalid:
        raise RuntimeError(
            f"Invalid torchGpus indices {invalid}; available CUDA devices: 0..{device_count - 1}"
        )


def resolve_voxel_radiomics_backend(
    use_torch_radiomics: UseTorchRadiomicsSetting = DEFAULT_USE_TORCH_RADIOMICS,
    torch_device: str = DEFAULT_TORCH_DEVICE,
    torch_gpus: TorchGpuSetting = None,
    torch_gpu_count: Optional[int] = None,
    subject: Optional[str] = None,
    gpu_slot_index: Optional[int] = None,
) -> Tuple[VoxelRadiomicsBackend, Optional[str]]:
    """
    Decide whether voxel extraction uses CPU PyRadiomics or injected TorchRadiomics.

    Policy for ``auto``:
    - torch missing -> CPU PyRadiomics
    - torch present but CUDA unavailable -> CPU PyRadiomics
    - torch + CUDA -> TorchRadiomics on selected GPU

    When ``torchGpus`` is set, it overrides ``torchDevice`` for CUDA device selection.
    ``torchGpuCount`` limits how many entries from ``torchGpus`` are used.
    With multiple GPUs and parallel subjects, subjects are mapped to GPUs via a
    stable hash of ``subject``.

    Args:
        use_torch_radiomics: ``auto``, ``true``, ``false``, or boolean equivalent.
        torch_device: Torch device string or ``auto`` when ``torchGpus`` is not set.
        torch_gpus: Allowed GPU indices, e.g. ``[0, 1]`` or ``"0,1"``.
        torch_gpu_count: Maximum number of GPUs to use from ``torchGpus``.
        subject: Subject ID for stable GPU assignment across parallel workers.
        gpu_slot_index: Explicit GPU slot index override.

    Returns:
        Tuple[VoxelRadiomicsBackend, Optional[str]]:
            Backend name and torch device when backend is ``torch``.

    Raises:
        ValueError: When ``useTorchRadiomics`` or GPU settings are invalid.
        RuntimeError: When ``useTorchRadiomics`` is ``true`` but torch is unavailable.
    """
    mode = normalize_use_torch_radiomics(use_torch_radiomics)
    parsed_gpus = apply_torch_gpu_count(
        parse_torch_gpu_indices(torch_gpus),
        torch_gpu_count=torch_gpu_count,
    )

    if mode == "false":
        return "pyradiomics", None

    torch_ok = is_torch_available()

    def _resolve_torch_device_string() -> str:
        if parsed_gpus:
            validate_torch_gpu_indices(parsed_gpus)
            device = select_torch_gpu_device(
                parsed_gpus,
                subject=subject,
                gpu_slot_index=gpu_slot_index,
            )
            logger.info(
                "voxel_radiomics torch device selected: %s from pool %s",
                device,
                parsed_gpus,
            )
            return device
        return resolve_torch_device(torch_device)

    if mode == "true":
        if not torch_ok:
            raise RuntimeError(
                "useTorchRadiomics=true but torch is not installed. "
                f"{TORCH_GPU_INSTALL_HINT}"
            )
        device = _resolve_torch_device_string()
        if not str(device).startswith("cuda"):
            log_torch_gpu_install_hint("cuda_unavailable")
        logger.info("voxel_radiomics backend: torch (%s)", device)
        return "torch", device

    # auto
    if not torch_ok:
        logger.info(
            "useTorchRadiomics=auto: torch not installed; using CPU PyRadiomics"
        )
        log_torch_gpu_install_hint("torch_not_installed")
        return "pyradiomics", None

    if parsed_gpus:
        if not is_cuda_available():
            logger.info(
                "useTorchRadiomics=auto: CUDA unavailable; using CPU PyRadiomics"
            )
            log_torch_gpu_install_hint("cuda_unavailable")
            return "pyradiomics", None
        try:
            device = _resolve_torch_device_string()
        except RuntimeError as exc:
            logger.info(
                "useTorchRadiomics=auto: %s; using CPU PyRadiomics",
                exc,
            )
            log_torch_gpu_install_hint("cuda_unavailable")
            return "pyradiomics", None
        logger.info("useTorchRadiomics=auto: using TorchRadiomics on %s", device)
        return "torch", device

    normalized_device = str(torch_device).strip().lower()
    if normalized_device == "auto":
        if not is_cuda_available():
            logger.info(
                "useTorchRadiomics=auto: CUDA unavailable; using CPU PyRadiomics"
            )
            log_torch_gpu_install_hint("cuda_unavailable")
            return "pyradiomics", None
        logger.info("useTorchRadiomics=auto: using TorchRadiomics on cuda:0")
        return "torch", "cuda:0"

    if normalized_device.startswith("cuda") and not is_cuda_available():
        logger.info(
            "useTorchRadiomics=auto: CUDA requested but unavailable; "
            "using CPU PyRadiomics"
        )
        log_torch_gpu_install_hint("cuda_unavailable")
        return "pyradiomics", None

    device = resolve_torch_device(torch_device)
    logger.info("useTorchRadiomics=auto: using TorchRadiomics on %s", device)
    return "torch", device


@contextmanager
def injected_torch_radiomics(enabled: bool) -> Iterator[None]:
    """
    Temporarily replace PyRadiomics feature classes with TorchRadiomics versions.

    Args:
        enabled: When False, this context manager is a no-op.

    Yields:
        None
    """
    if not enabled:
        yield
        return

    from habit.core.habitat_analysis.clustering_features.torchradiomics import (
        inject_torch_radiomics,
        restore_radiomics,
    )

    inject_torch_radiomics()
    try:
        yield
    finally:
        restore_radiomics()
