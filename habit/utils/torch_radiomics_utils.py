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
Optional TorchRadiomics backend helpers for voxel-based PyRadiomics extraction.

Torch is an optional dependency: these helpers lazy-import torch and fall back
to conventional CPU PyRadiomics when unavailable.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

VoxelRadiomicsBackend = Literal["pyradiomics", "torch"]
UseTorchRadiomicsSetting = Union[bool, str]
TorchGpuSetting = Union[int, str, Sequence[Union[int, str]], None]

DEFAULT_USE_TORCH_RADIOMICS = "auto"
DEFAULT_TORCH_DEVICE = "auto"
DEFAULT_TORCH_DTYPE = "float32"

# Shown once only when users explicitly select a GPU-oriented configuration.
_TORCH_GPU_INSTALL_HINT_LOGGED = False

TORCH_GPU_INSTALL_HINT = (
    "Install a PyTorch build compatible with your NVIDIA driver "
    "(pip extra: habitat-analysis[torch]). Set use_torch_radiomics: false "
    "to require CPU PyRadiomics."
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
            "use_torch_radiomics=auto: torch is not installed; using CPU PyRadiomics. %s",
            TORCH_GPU_INSTALL_HINT,
        )
        return

    if reason == "cuda_unavailable":
        logger.warning(
            "use_torch_radiomics=auto: CUDA is unavailable (CPU-only PyTorch wheel or "
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
    except Exception as exc:
        # A broken native PyTorch installation commonly raises OSError on Windows
        # (for example WinError 126 while loading fbgemm.dll), rather than ImportError.
        # Treat every regular import-time exception as an unavailable optional backend
        # so callers can safely fall back to CPU PyRadiomics.
        logger.debug("PyTorch import probe failed: %s", exc)
        return False
    return True


def is_cuda_available() -> bool:
    """
    Check whether CUDA is available through an installed PyTorch build.

    Returns:
        bool: True when torch is installed and ``torch.cuda.is_available()``.
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception as exc:
        # CUDA probing is optional and must not terminate the workflow when either
        # PyTorch native libraries or the CUDA runtime cannot be initialized.
        logger.debug("PyTorch CUDA availability probe failed: %s", exc)
        return False


def normalize_use_torch_radiomics(value: UseTorchRadiomicsSetting) -> str:
    """
    Normalize user-facing ``use_torch_radiomics`` values to ``auto|true|false``.

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
        f"use_torch_radiomics must be auto, true, or false; got {value!r}"
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
            "torch is not installed; install torch or set use_torch_radiomics to false/auto"
        )

    import torch

    normalized = str(torch_device).strip().lower()
    if normalized == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch_device requests CUDA but torch.cuda.is_available() is False"
            )
        return "cuda:0"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"torch_device={torch_device!r} requests CUDA but CUDA is unavailable"
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
            f"torch_dtype must be one of {sorted(mapping.keys())}; got {dtype_name!r}"
        )
    return mapping[normalized]


def parse_torch_gpu_indices(value: TorchGpuSetting) -> List[int]:
    """
    Parse user ``torch_gpus`` settings into a list of CUDA device indices.

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
                raise ValueError(f"Unsupported torch_gpus entry: {item!r}")
        return _dedupe_preserve_order(indices)

    raise ValueError(f"torch_gpus must be int, str, list, or null; got {value!r}")


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
    Limit how many GPUs from ``torch_gpus`` are actually used.

    Args:
        gpu_indices: Parsed GPU index list.
        torch_gpu_count: Maximum number of GPUs to use from the front of the list.

    Returns:
        List[int]: Possibly truncated GPU index list.
    """
    if torch_gpu_count is None:
        return gpu_indices
    if torch_gpu_count < 1:
        raise ValueError(f"torch_gpu_count must be >= 1; got {torch_gpu_count}")
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
        raise RuntimeError("CUDA is unavailable but torch_gpus was configured")
    device_count = torch.cuda.device_count()
    invalid = [idx for idx in gpu_indices if idx < 0 or idx >= device_count]
    if invalid:
        raise RuntimeError(
            f"Invalid torch_gpus indices {invalid}; available CUDA devices: 0..{device_count - 1}"
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

    When ``torch_gpus`` is set, it overrides ``torch_device`` for CUDA device selection.
    ``torch_gpu_count`` limits how many entries from ``torch_gpus`` are used.
    With multiple GPUs and parallel subjects, subjects are mapped to GPUs via a
    stable hash of ``subject``.

    Args:
        use_torch_radiomics: ``auto``, ``true``, ``false``, or boolean equivalent.
        torch_device: Torch device string or ``auto`` when ``torch_gpus`` is not set.
        torch_gpus: Allowed GPU indices, e.g. ``[0, 1]`` or ``"0,1"``.
        torch_gpu_count: Maximum number of GPUs to use from ``torch_gpus``.
        subject: Subject ID for stable GPU assignment across parallel workers.
        gpu_slot_index: Explicit GPU slot index override.

    Returns:
        Tuple[VoxelRadiomicsBackend, Optional[str]]:
            Backend name and torch device when backend is ``torch``.

    Raises:
        ValueError: When ``use_torch_radiomics`` or GPU settings are invalid.
        RuntimeError: When an explicitly requested Torch device is unavailable.
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
            logger.warning(
                "use_torch_radiomics=true was requested, but torch cannot be imported; "
                "falling back to CPU PyRadiomics. %s",
                TORCH_GPU_INSTALL_HINT,
            )
            return "pyradiomics", None
        device = _resolve_torch_device_string()
        if not str(device).startswith("cuda"):
            log_torch_gpu_install_hint("cuda_unavailable")
        logger.info("voxel_radiomics backend: torch (%s)", device)
        return "torch", device

    # auto
    if not torch_ok:
        logger.info(
            "use_torch_radiomics=auto: torch not installed; using CPU PyRadiomics"
        )
        return "pyradiomics", None

    if parsed_gpus:
        if not is_cuda_available():
            logger.info(
                "use_torch_radiomics=auto: CUDA unavailable; using CPU PyRadiomics"
            )
            return "pyradiomics", None
        try:
            device = _resolve_torch_device_string()
        except RuntimeError as exc:
            logger.info(
                "use_torch_radiomics=auto: %s; using CPU PyRadiomics",
                exc,
            )
            log_torch_gpu_install_hint("cuda_unavailable")
            return "pyradiomics", None
        logger.info("use_torch_radiomics=auto: using TorchRadiomics on %s", device)
        return "torch", device

    normalized_device = str(torch_device).strip().lower()
    if normalized_device == "auto":
        if not is_cuda_available():
            logger.info(
                "use_torch_radiomics=auto: CUDA unavailable; using CPU PyRadiomics"
            )
            log_torch_gpu_install_hint("cuda_unavailable")
            return "pyradiomics", None
        logger.info("use_torch_radiomics=auto: using TorchRadiomics on cuda:0")
        return "torch", "cuda:0"

    if normalized_device.startswith("cuda") and not is_cuda_available():
        logger.info(
            "use_torch_radiomics=auto: CUDA requested but unavailable; "
            "using CPU PyRadiomics"
        )
        log_torch_gpu_install_hint("cuda_unavailable")
        return "pyradiomics", None

    device = resolve_torch_device(torch_device)
    logger.info("use_torch_radiomics=auto: using TorchRadiomics on %s", device)
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

    from habit.kernels.radiomics.torchradiomics import (
        inject_torch_radiomics,
        restore_radiomics,
    )

    inject_torch_radiomics()
    try:
        yield
    finally:
        restore_radiomics()


def _habit_info_reaches_a_handler(log: logging.Logger) -> bool:
    """
    Return True when an INFO record from ``log`` will reach a configured handler.

    ``get_module_logger`` does not attach handlers. Without ``setup_logging`` /
    ``setup_logger``, INFO lines stay silent (the last-resort handler only
    shows WARNING+). Callers can then fall back to stderr via CustomTqdm.

    Args:
        log: Logger that will emit the INFO record.

    Returns:
        True when a handler exists on this logger or an ancestor that
        still propagates.
    """
    if not log.isEnabledFor(logging.INFO):
        return False
    current: Optional[logging.Logger] = log
    while current is not None:
        if current.handlers:
            return True
        if not current.propagate:
            return False
        current = current.parent
    return False


def _count_enabled_class_features(
    feature_class_name: str,
    feature_names: Optional[Sequence[str]],
    feature_classes: Mapping[str, Any],
) -> int:
    """
    Count how many features a PyRadiomics class will compute.

    An empty list or ``None`` means "all features of this class", matching
    ``RadiomicsFeatureExtractor.computeFeatures``.

    Args:
        feature_class_name: Class key such as ``"glcm"``.
        feature_names: Explicit names from ``extractor.enabledFeatures``, or
            ``None`` / empty to enable the full class.
        feature_classes: Mapping from ``radiomics.getFeatureClasses()``.

    Returns:
        Number of non-deprecated features that will run.
    """
    if feature_names:
        return len(list(feature_names))
    feature_class = feature_classes.get(feature_class_name)
    if feature_class is None:
        return 0
    names = feature_class.getFeatureNames()
    return sum(1 for deprecated in names.values() if not deprecated)


def _host_resource_snapshot() -> str:
    """
    Compact CPU / RAM / GPU line for per-class voxel-radiomics logs.

    Uses nvidia-smi when present (display GPU load and VRAM). Torch
    allocated/reserved memory is appended when CUDA is initialised so we
    can tell HABIT tensors from desktop compositor usage.

    Returns:
        str: One-line snapshot, e.g. ``cpu=12% ram=9.2/15.8GB gpu=71%
        vram=2140/8192MiB torch=1802/2400MiB``. Fields are omitted when
        the corresponding source is unavailable.
    """
    parts: List[str] = []
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        vm = psutil.virtual_memory()
        parts.append(f"cpu={proc.cpu_percent(interval=0.0):.0f}%")
        parts.append(f"ram={vm.used / (1024 ** 3):.1f}/{vm.total / (1024 ** 3):.1f}GB")
    except Exception:
        pass

    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            raw = subprocess.check_output(
                [
                    nvsmi,
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
            ).strip().split(",")
            gpu_util = raw[0].strip()
            vram_used = raw[1].strip()
            vram_total = raw[2].strip()
            parts.append(f"gpu={gpu_util}%")
            parts.append(f"vram={vram_used}/{vram_total}MiB")
        except Exception:
            pass

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            parts.append(f"torch={allocated:.0f}/{reserved:.0f}MiB")
    except Exception:
        pass
    return " ".join(parts) if parts else "resources=unavailable"


def release_cuda_cache() -> None:
    """
    Return unused CUDA blocks to the driver after a feature class finishes.

    PyTorch's caching allocator keeps reserved slabs (we measured ~3.3 GiB
    still reserved after GLCM even though live tensors dropped to a few
    MiB). On a laptop display GPU that leftover reservation is what made
    a second worker / the next tumour look like a leak and triggered TDR
    black screens. ``empty_cache`` does not change live tensors; it only
    gives unused cached blocks back to Windows / the display compositor.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _emit_voxel_class_progress(message: str) -> None:
    """
    Emit one per-class progress line to the habit logger and, if needed, stderr.

    When logging is configured, ``logger.info`` is enough (console + file).
    When it is not, print to stderr immediately so IDE Debug / non-TTY runs
    still see the line without calling ``setup_logging``.

    Args:
        message: English progress line, e.g. ``voxel_radiomics: glcm done in 84.2s``.
    """
    logger.info("%s", message)
    if not _habit_info_reaches_a_handler(logger):
        print(message, file=sys.stderr, flush=True)


def execute_voxel_based_with_class_progress(
    extractor: Any,
    image: Any,
    mask: Any,
    *,
    voxel_based: bool = True,
) -> Any:
    """
    Run voxel-based ``extractor.execute`` with per-feature-class progress.

    PyRadiomics already loops classes inside ``computeFeatures`` after a single
    image load / crop. This wraps that method so each class (firstorder, glcm,
    glrlm, glszm, gldm, ngtdm) reports start, elapsed seconds, and a CustomTqdm
    tick -- without re-reading the image or changing which features run.

    Opt-in only (``VoxelRadiomicsFeatures(class_progress=True)``). The default
    voxel path calls ``extractor.execute`` with no per-class lines.

    CustomTqdm writes to stderr with ``disable=False`` so the bar is visible in
    IDE Debug consoles that are not a TTY and without ``setup_logging``.

    Args:
        extractor: Configured ``RadiomicsFeatureExtractor``.
        image: SimpleITK image (or path) passed to ``execute``.
        mask: SimpleITK mask (or path) passed to ``execute``.
        voxel_based: Forwarded as ``voxelBased``; keep True for voxel maps.

    Returns:
        The same ``OrderedDict`` ``extractor.execute`` would have returned.
    """
    from radiomics import getFeatureClasses

    from habit.kernels.radiomics.voxel_maps import enabled_voxel_feature_classes
    from habit.utils.progress_utils import CustomTqdm

    feature_classes = getFeatureClasses()
    class_names = [
        name
        for name in enabled_voxel_feature_classes(extractor.enabledFeatures)
        if name in feature_classes
    ]
    n_image_types = max(1, len(getattr(extractor, "enabledImagetypes", {}) or {}))
    total_steps = len(class_names) * n_image_types
    # Restore the class method by deleting the instance attr unless the
    # extractor already overrode computeFeatures on the instance.
    had_instance_compute = "computeFeatures" in vars(extractor)
    original_compute = extractor.computeFeatures
    # Assigned in the CustomTqdm ``with`` block before execute() calls the hook.
    progress: Any = None

    def compute_features_with_class_progress(
        input_image: Any,
        input_mask: Any,
        image_type_name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Drop-in replacement for ``RadiomicsFeatureExtractor.computeFeatures``.

        Assigned on the extractor instance, so ``self.computeFeatures(...)``
        does not pass ``self``. The loop body matches PyRadiomics (constructor,
        optional ``enableFeatureByName``, then ``execute``) so numbers stay
        identical; only timing and progress are added.

        Args:
            input_image: Cropped (and optionally filtered) SimpleITK image.
            input_mask: Cropped SimpleITK mask.
            image_type_name: Filter name, typically ``"original"``.
            **kwargs: Settings forwarded to the feature-class constructor.

        Returns:
            Feature-name -> value (SimpleITK map when ``voxelBased``).
        """
        # Live lookup so TorchRadiomics injection is honoured, same as upstream.
        live_feature_classes = getFeatureClasses()
        feature_vector: Dict[str, Any] = OrderedDict()
        enabled_features = extractor.enabledFeatures

        for feature_class_name, names in enabled_features.items():
            if str(feature_class_name).startswith("shape"):
                continue
            if feature_class_name not in live_feature_classes:
                continue

            n_features = _count_enabled_class_features(
                feature_class_name, names, live_feature_classes
            )
            progress.set_description(
                f"voxel_radiomics {feature_class_name}",
                refresh=True,
            )
            before = _host_resource_snapshot()
            _emit_voxel_class_progress(
                f"voxel_radiomics: extracting {feature_class_name} "
                f"({n_features} features) ... [{before}]"
            )
            started = time.perf_counter()

            feature_class = live_feature_classes[feature_class_name](
                input_image, input_mask, **kwargs
            )
            if names is not None:
                for feature in names:
                    feature_class.enableFeatureByName(feature)
            for feature_name, feature_value in feature_class.execute().items():
                new_name = f"{image_type_name}_{feature_class_name}_{feature_name}"
                feature_vector[new_name] = feature_value

            elapsed = time.perf_counter() - started
            # Drop the feature-class graph (texture matrices live here)
            # before returning cached CUDA slabs to the driver.
            del feature_class
            release_cuda_cache()
            after = _host_resource_snapshot()
            progress.set_postfix_str(f"{elapsed:.1f}s", refresh=True)
            _emit_voxel_class_progress(
                f"voxel_radiomics: {feature_class_name} done in {elapsed:.1f}s [{after}]"
            )
            progress.update(1)

        return feature_vector

    extractor.computeFeatures = compute_features_with_class_progress
    try:
        # force-enable: IDE Debug / redirected stderr is often not a TTY, and
        # tqdm would otherwise disable itself so the user sees nothing.
        with CustomTqdm(
            total=total_steps,
            desc="voxel_radiomics classes",
            file=sys.stderr,
            disable=False,
            leave=True,
            mininterval=0.0,
        ) as progress:
            return extractor.execute(image, mask, voxelBased=voxel_based)
    finally:
        if had_instance_compute:
            extractor.computeFeatures = original_compute
        elif "computeFeatures" in vars(extractor):
            del extractor.computeFeatures
        release_cuda_cache()
