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
GPU texture-matrix calculation with PyRadiomics-compatible numerics.

This subpackage re-implements the matrix-building stage of PyRadiomics
texture feature classes (the ``radiomics.cMatrices`` C extension) as
vectorised torch operations running on GPU (or CPU torch for testing).
Feature formulas are untouched: HABIT's TorchRadiomics classes consume the
returned matrices exactly as they consume the C extension output.

Currently implemented: GLCM, GLDM, NGTDM, GLRLM, GLSZM.
"""

from __future__ import annotations

from typing import Mapping, Optional, Union

import torch

from .angles import build_angles, get_angle_count
from .glcm import calculate_glcm, calculate_glcm_coo
from .gldm import calculate_gldm
from .glrlm import calculate_glrlm, calculate_glrlm_coo
from .glszm import calculate_glszm, calculate_glszm_coo
from .ngtdm import calculate_ngtdm
from .sparse_coo import (
    GLCM_DENSE_ONLY_FEATURES,
    estimate_dense_working_bytes,
    estimate_sparse_working_bytes,
    glcm_features_from_coo,
    glrlm_features_from_coo,
    glszm_features_from_coo,
)


def is_available() -> bool:
    """Return True when a CUDA device is visible to torch."""
    return torch.cuda.is_available()


def resolve_use_gpu_matrices(
    settings: Mapping[str, object],
    device: Optional[Union[str, torch.device]] = None,
) -> bool:
    """
    Resolve whether texture matrices should be built on GPU.

    Follows the same ``auto``/``True``/``False`` convention as
    ``resolve_use_supervoxel_cext``: explicit ``True``/``False`` win;
    ``"auto"`` enables the GPU path when torch sees CUDA and the feature
    class itself targets a CUDA device.

    Args:
        settings: PyRadiomics / habit settings dict (reads
            ``use_gpu_matrices``, default ``"auto"``).
        device: Device the feature class will run on; the auto rule
            requires it to be a CUDA device.

    Returns:
        bool: True when the GPU matrix path should be used.
    """
    flag = settings.get("use_gpu_matrices", "auto")
    if flag is True or str(flag).lower() == "true":
        return True
    if flag is False or str(flag).lower() == "false":
        return False
    # auto: GPU matrices only when the torch path itself runs on CUDA
    if device is None:
        return is_available()
    return is_available() and str(device).startswith("cuda")


def resolve_use_sparse_matrices(
    settings: Mapping[str, object],
    device: Optional[Union[str, torch.device]] = None,
    voxel_based: bool = True,
) -> bool:
    """
    Resolve whether voxel GPU texture features use sparse COO.

    ``auto`` (default): on when the GPU matrix path is on, the compute
    dtype is float64, and MCC/Imc1/Imc2 are not enabled. float32 is
    opt-in for dense only -- sparse f32 GLCM ClusterShade failed 1e-3.

    Explicit ``False`` keeps the dense gold kernels (parity tests).

    Args:
        settings: PyRadiomics / habit settings (reads
            ``use_sparse_matrices``, ``dtype``, ``enabledFeatures``).
        device: Feature-class device; forwarded to the GPU-matrix rule.
        voxel_based: Sparse COO is the default for voxel kernels. Segment
            mode may still use COO for GLCM/GLRLM/GLSZM when auto.

    Returns:
        bool: True when the sparse COO feature path should run.
    """
    del voxel_based  # reserved: segment COO is also safe and is allowed.
    flag = settings.get("use_sparse_matrices", "auto")
    if flag is False or str(flag).lower() == "false":
        return False
    if not resolve_use_gpu_matrices(settings, device):
        return False
    dtype = settings.get("dtype", torch.float64)
    if dtype in (torch.float32, "float32"):
        return False
    enabled = settings.get("enabledFeatures") or {}
    if isinstance(enabled, Mapping):
        if any(enabled.get(name) for name in GLCM_DENSE_ONLY_FEATURES):
            return False
    weighting = settings.get("weightingNorm", None)
    if weighting not in (None, "no_weighting"):
        return False
    if flag is True or str(flag).lower() == "true":
        return True
    return True


__all__ = [
    "build_angles",
    "calculate_glcm",
    "calculate_glcm_coo",
    "calculate_gldm",
    "calculate_glrlm",
    "calculate_glrlm_coo",
    "calculate_glszm",
    "calculate_glszm_coo",
    "calculate_ngtdm",
    "estimate_dense_working_bytes",
    "estimate_sparse_working_bytes",
    "get_angle_count",
    "glcm_features_from_coo",
    "glrlm_features_from_coo",
    "glszm_features_from_coo",
    "is_available",
    "resolve_use_gpu_matrices",
    "resolve_use_sparse_matrices",
]
