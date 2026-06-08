# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
HABIT supervoxel radiomics C extension with PyRadiomics fallback.

Public helpers mirror ``supervoxel_cext._sv_cmatrices`` when compiled; otherwise
``supervoxel_cext._fallback`` loops labels through PyRadiomics ``cMatrices``.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

_BACKEND = "fallback"
_NATIVE = None

try:
    from . import _sv_cmatrices as _native_module

    _NATIVE = _native_module
    _BACKEND = "native"
except ImportError:
    from . import _fallback as _native_module


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
    flag = settings.get("useSupervoxelCext", "auto")
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


def calculate_firstorder(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: np.ndarray,
    Ng: int,
    binWidth: float,
) -> np.ndarray:
    """Batch first-order statistics for multiple supervoxel labels."""
    image_arr, sv_arr, labels_arr = _validate_shared_inputs(image, sv_map, labels)
    return _native_module.calculate_firstorder(
        image_arr,
        sv_arr,
        labels_arr,
        int(Ng),
        float(binWidth),
    )


__all__ = [
    "calculate_firstorder",
    "calculate_gldm",
    "calculate_glcm",
    "calculate_glrlm",
    "calculate_glszm",
    "calculate_ngtdm",
    "cext_backend",
    "is_cext_available",
    "resolve_use_supervoxel_cext",
    "supervoxel_cext_matrix_backend_label",
]
