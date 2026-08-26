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
HABIT supervoxel radiomics C extension with PyRadiomics fallback.

Public helpers mirror ``supervoxel_cext._sv_cmatrices`` when compiled; otherwise
``supervoxel_cext._fallback`` loops labels through PyRadiomics ``cMatrices``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Mapping, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_BACKEND = "fallback"
_NATIVE = None
_FALLBACK_WARNED = False

_FALLBACK_MESSAGE = (
    "HABIT native radiomics C extension (_sv_cmatrices) is not loaded; "
    "using the slow per-label PyRadiomics cMatrices fallback. "
    "Rebuild the extension with: pip install -e ."
)


def _warn_fallback_once() -> None:
    """
    Emit a one-shot visible warning when the compiled extension is missing.

    Both ``warnings.warn`` and ``logger.warning`` fire so the message is
    visible even when the caller filters one channel. Subsequent matrix
    calls reuse the same flag and stay silent.
    """
    global _FALLBACK_WARNED
    if _BACKEND != "fallback" or _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    warnings.warn(_FALLBACK_MESSAGE, RuntimeWarning, stacklevel=2)
    logger.warning(_FALLBACK_MESSAGE)


try:
    from . import _sv_cmatrices as _native_module

    _NATIVE = _native_module
    _BACKEND = "native"
except ImportError:
    from . import _fallback as _native_module

    _warn_fallback_once()


def is_cext_available() -> bool:
    """Return True when the compiled ``_sv_cmatrices`` extension is importable."""
    return _BACKEND == "native"


def cext_backend() -> str:
    """Return ``native`` or ``fallback`` depending on the active backend."""
    return _BACKEND


def resolve_use_supervoxel_cext(settings: Mapping[str, object]) -> bool:
    """
    Resolve whether batched supervoxel extraction should use the C extension path.

    Args:
        settings: PyRadiomics / habit settings dict.

    Returns:
        bool: True when the C-extension batch path should be used.
    """
    flag = settings.get("use_supervoxel_cext", "auto")
    if flag is True or str(flag).lower() == "true":
        return True
    if flag is False or str(flag).lower() == "false":
        return False
    # auto: native C extension when built; otherwise prior Torch/PyRadiomics path
    return is_cext_available()


def supervoxel_cext_matrix_backend_label(settings: Mapping[str, object]) -> str:
    """
    Return a stable matrix-backend label for supervoxel texture logging.

    Args:
        settings: PyRadiomics / habit settings dict.

    Returns:
        str: One of ``habit_native_c``, ``habit_fallback_cmatrices``, or ``torch_cmatrices``.
    """
    if not resolve_use_supervoxel_cext(settings):
        return "torch_cmatrices"
    if is_cext_available():
        return "habit_native_c"
    return "habit_fallback_cmatrices"


def _validate_shared_inputs(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate that image and sv_map share shape and labels is 1D."""
    _warn_fallback_once()
    image_arr = np.asarray(image)
    sv_arr = np.asarray(sv_map)
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)

    if image_arr.shape != sv_arr.shape:
        raise ValueError(
            f"image shape {image_arr.shape} must match sv_map shape {sv_arr.shape}"
        )
    if labels_arr.size == 0:
        raise ValueError("labels must contain at least one supervoxel id")
    return image_arr, sv_arr, labels_arr


def calculate_glcm(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    Ng: int,
    force2D: int = 0,
    force2Ddimension: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch GLCM for multiple supervoxel labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(P_glcm, angles)``.
    """
    image_arr, sv_arr, labels_arr = _validate_shared_inputs(image, sv_map, labels)
    distances_arr = np.ascontiguousarray(np.asarray(distances, dtype=np.int32).reshape(-1))
    return _native_module.calculate_glcm(
        image_arr,
        sv_arr,
        labels_arr,
        distances_arr,
        int(Ng),
        int(force2D),
        int(force2Ddimension),
    )


def calculate_glrlm(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    Ng: int,
    Nr: int,
    force2D: int = 0,
    force2Ddimension: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch GLRLM for multiple supervoxel labels."""
    image_arr, sv_arr, labels_arr = _validate_shared_inputs(image, sv_map, labels)
    return _native_module.calculate_glrlm(
        image_arr,
        sv_arr,
        labels_arr,
        int(Ng),
        int(Nr),
        int(force2D),
        int(force2Ddimension),
    )


def calculate_glszm(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    Ng: int,
    force2D: int = 0,
    force2Ddimension: int = 0,
) -> np.ndarray:
    """Batch GLSZM for multiple supervoxel labels."""
    image_arr, sv_arr, labels_arr = _validate_shared_inputs(image, sv_map, labels)
    return _native_module.calculate_glszm(
        image_arr,
        sv_arr,
        labels_arr,
        int(Ng),
        int(force2D),
        int(force2Ddimension),
    )


def calculate_ngtdm(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    Ng: int,
    force2D: int = 0,
    force2Ddimension: int = 0,
) -> np.ndarray:
    """Batch NGTDM for multiple supervoxel labels."""
    image_arr, sv_arr, labels_arr = _validate_shared_inputs(image, sv_map, labels)
    distances_arr = np.ascontiguousarray(np.asarray(distances, dtype=np.int32).reshape(-1))
    return _native_module.calculate_ngtdm(
        image_arr,
        sv_arr,
        labels_arr,
        distances_arr,
        int(Ng),
        int(force2D),
        int(force2Ddimension),
    )


def calculate_gldm(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    Ng: int,
    alpha: int,
    force2D: int = 0,
    force2Ddimension: int = 0,
) -> np.ndarray:
    """Batch GLDM for multiple supervoxel labels."""
    image_arr, sv_arr, labels_arr = _validate_shared_inputs(image, sv_map, labels)
    distances_arr = np.ascontiguousarray(np.asarray(distances, dtype=np.int32).reshape(-1))
    return _native_module.calculate_gldm(
        image_arr,
        sv_arr,
        labels_arr,
        distances_arr,
        int(Ng),
        int(alpha),
        int(force2D),
        int(force2Ddimension),
    )


def glcm_formulas(
    p_counts: np.ndarray,
    gray_levels: np.ndarray,
    ng_full: np.ndarray,
    symmetrical: int = 1,
) -> np.ndarray:
    """
    Evaluate stacked GLCM formulas in OpenMP C (24 default features; MCC last).

    Args:
        p_counts: Raw GLCM counts ``[K, Ng, Ng, Na]``.
        gray_levels: 1-indexed gray-level values of length ``Ng``.
        ng_full: Per-label ``Ng`` used by Idn / Idmn.
        symmetrical: 1 to add the transpose before normalising.

    Returns:
        np.ndarray: ``[K, 24]`` in ``GLCM_FORMULA_COLUMNS`` order.
    """
    p_arr = np.ascontiguousarray(np.asarray(p_counts, dtype=np.float64))
    gray = np.ascontiguousarray(np.asarray(gray_levels, dtype=np.float64).reshape(-1))
    ng = np.ascontiguousarray(np.asarray(ng_full, dtype=np.float64).reshape(-1))
    return _native_module.glcm_formulas(p_arr, gray, ng, int(symmetrical))


def glcm_mcc(p_counts: np.ndarray, symmetrical: int = 1) -> np.ndarray:
    """
    Evaluate stacked GLCM MCC on the CPU C path.

    Args:
        p_counts: Raw GLCM counts ``[K, Ng, Ng, Na]``.
        symmetrical: 1 to add the transpose before normalising.

    Returns:
        np.ndarray: Shape ``[K]``.
    """
    p_arr = np.ascontiguousarray(np.asarray(p_counts, dtype=np.float64))
    return _native_module.glcm_mcc(p_arr, int(symmetrical))


def glrlm_formulas(p_counts: np.ndarray, gray_levels: np.ndarray) -> np.ndarray:
    """
    Evaluate stacked GLRLM formulas in OpenMP C.

    Args:
        p_counts: Raw GLRLM counts ``[K, Ng, Nr, Na]``.
        gray_levels: 1-indexed gray-level values of length ``Ng``.

    Returns:
        np.ndarray: ``[K, 16]`` in ``GLRLM_FORMULA_COLUMNS`` order.
    """
    p_arr = np.ascontiguousarray(np.asarray(p_counts, dtype=np.float64))
    gray = np.ascontiguousarray(np.asarray(gray_levels, dtype=np.float64).reshape(-1))
    return _native_module.glrlm_formulas(p_arr, gray)


def calculate_firstorder(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    Ng: int,
    binWidth: float,
    voxelArrayShift: float = 0.0,
    voxelVolume: float = 1.0,
) -> np.ndarray:
    """
    Batch first-order statistics for multiple supervoxel labels.

    Args:
        image: Raw intensity volume (float64), not the discretized bins.
        sv_map: Multi-label supervoxel map aligned with ``image``.
        labels: 1D label ids.
        Ng: Histogram bin count used for Entropy / Uniformity.
        binWidth: PyRadiomics ``binWidth`` (Entropy / Uniformity only).
        voxelArrayShift: Added to intensities before Energy / TotalEnergy / RMS.
        voxelVolume: ``prod(spacing)``; TotalEnergy = Energy * voxelVolume.

    Returns:
        np.ndarray: ``(n_labels, 17)`` in ``FIRSTORDER_CEXT_COLUMNS`` order.
    """
    image_arr, sv_arr, labels_arr = _validate_shared_inputs(image, sv_map, labels)
    return _native_module.calculate_firstorder(
        image_arr,
        sv_arr,
        labels_arr,
        int(Ng),
        float(binWidth),
        float(voxelArrayShift),
        float(voxelVolume),
    )


__all__ = [
    "calculate_firstorder",
    "calculate_gldm",
    "calculate_glcm",
    "calculate_glrlm",
    "calculate_glszm",
    "calculate_ngtdm",
    "cext_backend",
    "glcm_formulas",
    "glcm_mcc",
    "glrlm_formulas",
    "is_cext_available",
    "resolve_use_supervoxel_cext",
    "supervoxel_cext_matrix_backend_label",
]
