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
Batched supervoxel ROI radiomics mirroring PyRadiomics voxelBased orchestration.

Flow (aligned with ``RadiomicsFeaturesBase._calculateVoxels``):

1. Discretize once on the union supervoxel mask via PyRadiomics ``_applyBinning``.
2. Crop image/mask/supervoxel map to the union-mask bounding box (+ ``padDistance``).
3. For each feature class, iterate supervoxel labels in batches (``supervoxelBatch``).
3. Per label in a batch: ``cMatrices`` via ``_calculateMatrix`` (PyRadiomics native ROI path).
4. Stack matrices to ``[B, …]`` and run Torch ``_calculateCoefficients`` + ``get*FeatureValue`` once
   per batch (Torch path). CPU PyRadiomics falls back to per-label scalar formulas.

Torch is lazy-imported so machines without PyTorch can still import this module.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk

from habit.utils.log_utils import get_module_logger
from habit.core.habitat_analysis.clustering_features.supervoxel_cext import (
    cext_backend,
    is_cext_available,
    resolve_use_supervoxel_cext,
    supervoxel_cext_matrix_backend_label,
)
from habit.core.habitat_analysis.clustering_features.supervoxel_cext.torch_batch import (
    extract_supervoxel_batch_via_cext,
)

logger = get_module_logger(__name__)

# Default batch size; controls matrix/formula batch width (like voxelBatch for voxel maps).
DEFAULT_SUPERVOXEL_BATCH = 64

# Default padding around union-mask bbox for ROI texture (not voxel kernelRadius).
DEFAULT_SUPERVOXEL_PAD_DISTANCE = 1

# Torch texture classes: matrix attribute on calculator after ``_calculateMatrix``.
_TEXTURE_MATRIX_ATTR: Dict[str, str] = {
    "glcm": "P_glcm",
    "glrlm": "P_glrlm",
    "glszm": "P_glszm",
    "ngtdm": "P_ngtdm",
}

# PyRadiomics feature names enabled by default when a whole class is requested.
DEFAULT_FEATURES_BY_CLASS: Dict[str, List[str]] = {
    "firstorder": [
        "Energy", "TotalEnergy", "Entropy", "Minimum", "10Percentile", "90Percentile",
        "Maximum", "Mean", "Median", "InterquartileRange", "Range",
        "MeanAbsoluteDeviation", "RobustMeanAbsoluteDeviation", "RootMeanSquared",
        "Skewness", "Kurtosis", "Variance", "Uniformity",
    ],
    "glcm": [
        "Autocorrelation", "JointAverage", "ClusterProminence", "ClusterShade",
        "ClusterTendency", "Contrast", "Correlation", "DifferenceAverage",
        "DifferenceEntropy", "DifferenceVariance", "JointEnergy", "JointEntropy",
        "Imc1", "Imc2", "Idm", "MCC", "Idmn", "Id", "Idn", "InverseVariance",
        "MaximumProbability", "SumAverage", "SumEntropy", "SumSquares",
    ],
    "glrlm": [
        "ShortRunEmphasis", "LongRunEmphasis", "GrayLevelNonUniformity",
        "GrayLevelNonUniformityNormalized", "RunLengthNonUniformity",
        "RunLengthNonUniformityNormalized", "RunPercentage", "GrayLevelVariance",
        "RunVariance", "RunEntropy", "LowGrayLevelRunEmphasis",
        "HighGrayLevelRunEmphasis", "ShortRunLowGrayLevelEmphasis",
        "ShortRunHighGrayLevelEmphasis", "LongRunLowGrayLevelEmphasis",
        "LongRunHighGrayLevelEmphasis",
    ],
    "glszm": [
        "SmallAreaEmphasis", "LargeAreaEmphasis", "GrayLevelNonUniformity",
        "GrayLevelNonUniformityNormalized", "SizeZoneNonUniformity",
        "SizeZoneNonUniformityNormalized", "ZonePercentage", "GrayLevelVariance",
        "ZoneVariance", "ZoneEntropy", "LowGrayLevelZoneEmphasis",
        "HighGrayLevelZoneEmphasis", "SmallAreaLowGrayLevelEmphasis",
        "SmallAreaHighGrayLevelEmphasis", "LargeAreaLowGrayLevelEmphasis",
        "LargeAreaHighGrayLevelEmphasis",
    ],
    "ngtdm": [
        "Coarseness", "Contrast", "Busyness", "Complexity", "Strength",
    ],
    "gldm": [
        "SmallDependenceEmphasis", "LargeDependenceEmphasis",
        "GrayLevelNonUniformity",
        "DependenceNonUniformity", "DependenceNonUniformityNormalized",
        "DependenceEntropy", "LowGrayLevelEmphasis", "HighGrayLevelEmphasis",
        "SmallDependenceLowGrayLevelEmphasis", "SmallDependenceHighGrayLevelEmphasis",
        "LargeDependenceLowGrayLevelEmphasis", "LargeDependenceHighGrayLevelEmphasis",
    ],
}

SUPPORTED_FEATURE_CLASSES: Tuple[str, ...] = (
    "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm",
)


def _resolve_enabled_features(
    enabled_features: Mapping[str, object],
) -> Dict[str, List[str]]:
    """
    Resolve enabled feature names per PyRadiomics class.

    Args:
        enabled_features: ``RadiomicsFeatureExtractor.enabledFeatures`` mapping.

    Returns:
        Dict[str, List[str]]: Feature class -> list of feature names.
    """
    resolved: Dict[str, List[str]] = {}
    for feature_class, names in enabled_features.items():
        class_name = str(feature_class)
        if class_name.startswith("shape"):
            continue
        if names is None:
            resolved[class_name] = list(DEFAULT_FEATURES_BY_CLASS.get(class_name, []))
        else:
            resolved[class_name] = [str(name) for name in names]
    return resolved


def _build_union_mask(
    supervoxel_map: sitk.Image,
    image: sitk.Image,
) -> Tuple[sitk.Image, np.ndarray]:
    """
    Build a binary union mask covering all supervoxel labels.

    Args:
        supervoxel_map: Multi-label supervoxel map.
        image: Source intensity image (geometry reference).

    Returns:
        Tuple[sitk.Image, np.ndarray]: Union mask image and label array (z, y, x).
    """
    label_array: np.ndarray = sitk.GetArrayFromImage(supervoxel_map)
    union_mask_array: np.ndarray = (label_array > 0).astype(np.uint8)
    union_mask: sitk.Image = sitk.GetImageFromArray(union_mask_array)
    union_mask.CopyInformation(image)
    return union_mask, label_array


def _resolve_supervoxel_pad_distance(settings: Mapping[str, object]) -> int:
    """
    Resolve padding for union-mask bbox crop.

    Supervoxel ROI extraction uses ``padDistance`` (PyRadiomics crop padding), not
    ``kernelRadius`` (voxel-based kernel only). ``supervoxelPadDistance`` overrides
    ``padDistance`` when present.

    Args:
        settings: PyRadiomics / habit radiomics settings dict.

    Returns:
        int: Non-negative pad distance in voxels.
    """
    if "supervoxelPadDistance" in settings:
        return max(0, int(settings["supervoxelPadDistance"]))
    if "padDistance" in settings:
        return max(0, int(settings["padDistance"]))
    return DEFAULT_SUPERVOXEL_PAD_DISTANCE


def _should_crop_union_bbox(settings: Mapping[str, object]) -> bool:
    """
    Return whether union-mask bbox cropping is enabled.

    Args:
        settings: PyRadiomics / habit radiomics settings dict.

    Returns:
        bool: True unless ``supervoxelUnionBboxCrop`` is explicitly False.
    """
    return bool(settings.get("supervoxelUnionBboxCrop", True))


def _crop_to_union_bounding_box(
    image: sitk.Image,
    union_mask: sitk.Image,
    supervoxel_map: sitk.Image,
    settings: Mapping[str, object],
) -> Tuple[sitk.Image, sitk.Image, sitk.Image, int]:
    """
    Crop volumes to the union supervoxel mask bounding box using PyRadiomics helpers.

    Mirrors ``RadiomicsFeatureExtractor.execute`` cropping semantics:
    ``checkMask`` for the bounding box, then ``cropToTumorMask`` with ``padDistance``.

    Args:
        image: Input intensity image.
        union_mask: Binary union mask (label 1 = foreground).
        supervoxel_map: Multi-label supervoxel map aligned with ``image``.
        settings: PyRadiomics settings (``label``, ``padDistance``, geometry keys).

    Returns:
        Tuple[sitk.Image, sitk.Image, sitk.Image, int]:
            Cropped image, cropped union mask, cropped supervoxel map, pad distance.
    """
    from radiomics import imageoperations

    pad_distance = _resolve_supervoxel_pad_distance(settings)
    crop_settings = dict(settings)
    crop_settings.setdefault("label", 1)

    bounding_box, corrected_mask = imageoperations.checkMask(
        image,
        union_mask,
        **crop_settings,
    )
    mask_for_crop = corrected_mask if corrected_mask is not None else union_mask

    cropped_image, cropped_union_mask = imageoperations.cropToTumorMask(
        image,
        mask_for_crop,
        bounding_box,
        padDistance=pad_distance,
    )
    # cropToTumorMask returns (croppedImage, croppedMask); pass supervoxel_map as
    # both nodes so the first output is the cropped label map (not intensity image).
    cropped_supervoxel_map, _ = imageoperations.cropToTumorMask(
        supervoxel_map,
        supervoxel_map,
        bounding_box,
        padDistance=pad_distance,
    )

    logger.debug(
        "Union bbox crop: padDistance=%d full_size=%s cropped_size=%s",
        pad_distance,
        image.GetSize(),
        cropped_image.GetSize(),
    )
    return cropped_image, cropped_union_mask, cropped_supervoxel_map, pad_distance


def _prepare_supervoxel_volumes(
    image: sitk.Image,
    supervoxel_map: sitk.Image,
    settings: Mapping[str, object],
) -> Tuple[sitk.Image, sitk.Image, sitk.Image, int, bool]:
    """
    Build union mask and optionally crop all volumes to its bounding box.

    Args:
        image: Input intensity image.
        supervoxel_map: Multi-label supervoxel map.
        settings: PyRadiomics settings dict.

    Returns:
        Tuple[sitk.Image, sitk.Image, sitk.Image, int, bool]:
            Image, union mask, supervoxel map (possibly cropped), pad distance,
            whether cropping was applied.
    """
    union_mask, _ = _build_union_mask(supervoxel_map, image)
    if not _should_crop_union_bbox(settings):
        return image, union_mask, supervoxel_map, 0, False

    cropped_image, cropped_union_mask, cropped_supervoxel_map, pad_distance = (
        _crop_to_union_bounding_box(image, union_mask, supervoxel_map, settings)
    )
    return cropped_image, cropped_union_mask, cropped_supervoxel_map, pad_distance, True


def _feature_column_name(
    feature_class: str,
    feature_name: str,
    image_name: str,
) -> str:
    """
    Build PyRadiomics-style column name with optional image suffix.

    Args:
        feature_class: Feature class name.
        feature_name: Feature name without class prefix.
        image_name: Optional modality suffix.

    Returns:
        str: Column name, e.g. ``original_glcm_Contrast-T2``.
    """
    col = f"original_{feature_class}_{feature_name}"
    if image_name:
        col = f"{col}-{image_name}"
    return col


def _extract_feature_values(
    calculator: object,
    feature_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Call get*FeatureValue methods and normalize outputs to 1D arrays.

    Args:
        calculator: PyRadiomics or TorchRadiomics feature class instance.
        feature_names: Feature names without class prefix.

    Returns:
        Dict[str, np.ndarray]: Feature values with shape [1] or [B].
    """
    values: Dict[str, np.ndarray] = {}
    for feature_name in feature_names:
        method_name = f"get{feature_name}FeatureValue"
        method = getattr(calculator, method_name)
        raw = method()
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        values[feature_name] = arr
    return values


def _scalar_feature_value(values: np.ndarray) -> float:
    """
    Reduce a feature value array to a single float for one supervoxel row.

    Args:
        values: Feature array from get*FeatureValue.

    Returns:
        float: Scalar feature value.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return float("nan")
    return float(flat[0])


def _assign_batch_feature_values(
    batch_row_maps: List[Dict[str, object]],
    feature_class: str,
    feature_names: Sequence[str],
    feature_values: Mapping[str, np.ndarray],
    image_name: str,
) -> None:
    """
    Write batched feature vectors into per-label row dicts.

    Args:
        batch_row_maps: One dict per supervoxel label in the batch.
        feature_class: PyRadiomics feature class name.
        feature_names: Enabled feature names without class prefix.
        feature_values: Feature name -> array of shape [B].
        image_name: Optional column suffix.
    """
    batch_len = len(batch_row_maps)
    for feature_name in feature_names:
        col = _feature_column_name(feature_class, feature_name, image_name)
        values = np.asarray(feature_values[feature_name], dtype=np.float64).reshape(-1)
        if values.size == 1 and batch_len > 1:
            # Some get* methods return a scalar when B=1; duplicate for safety.
            values = np.full(batch_len, values[0], dtype=np.float64)
        for row_idx, row in enumerate(batch_row_maps):
            row[col] = float(values[row_idx]) if row_idx < values.size else float("nan")


def _pad_and_stack_torch(
    tensors: Sequence[object],
    *,
    pad_dims: Sequence[int],
    fill: float = 0.0,
) -> object:
    """
    Pad tensors along selected dimensions and concatenate on batch dim 0.

    Each input tensor is expected to have leading batch dimension 1 (one ROI).

    Args:
        tensors: Sequence of torch tensors with shape ``(1, ...)``.
        pad_dims: Dimensions (excluding batch dim 0) to pad to the batch maximum.
        fill: Padding fill value.

    Returns:
        torch.Tensor: Stacked tensor with shape ``(B, ...)``.
    """
    import torch

    if not tensors:
        raise ValueError("Cannot stack an empty tensor sequence.")

    torch_tensors = [t if isinstance(t, torch.Tensor) else torch.as_tensor(t) for t in tensors]
    ndim = torch_tensors[0].ndim
    max_sizes = list(torch_tensors[0].shape)
    for tensor in torch_tensors[1:]:
        for dim_idx in range(ndim):
            max_sizes[dim_idx] = max(max_sizes[dim_idx], tensor.shape[dim_idx])

    padded: List[torch.Tensor] = []
    for tensor in torch_tensors:
        for dim_idx in pad_dims:
            current = tensor.shape[dim_idx]
            target = max_sizes[dim_idx]
            if current < target:
                pad_shape = list(tensor.shape)
                pad_shape[dim_idx] = target - current
                pad_tensor = torch.full(
                    pad_shape,
                    fill,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                tensor = torch.cat([tensor, pad_tensor], dim=dim_idx)
        padded.append(tensor)

    return torch.cat(padded, dim=0)


def _extract_torch_texture_batch(
    calculator: object,
    feature_class: str,
    feature_names: Sequence[str],
    *,
    label_array: np.ndarray,
    batch_labels: Sequence[int],
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
) -> None:
    """
    Stack ROI texture matrices for a label batch and evaluate Torch formulas once.

    GLRLM/GLSZM run ``_calculateCoefficients`` per label first because column
    pruning is ROI-specific; other texture classes stack raw matrices then batch
    the coefficient step.

    Args:
        calculator: TorchRadiomics texture calculator (GLCM/GLRLM/GLSZM/NGTDM).
        feature_class: Feature class name.
        feature_names: Enabled features for this class.
        label_array: Full supervoxel label map.
        batch_labels: Label ids in this batch.
        batch_row_maps: Mutable row dicts to fill with feature values.
        image_name: Optional column suffix.
    """
    if feature_class == "glrlm":
        _extract_torch_glrlm_batch(
            calculator,
            feature_names,
            label_array=label_array,
            batch_labels=batch_labels,
            batch_row_maps=batch_row_maps,
            image_name=image_name,
        )
        return
    if feature_class == "glszm":
        _extract_torch_glszm_batch(
            calculator,
            feature_names,
            label_array=label_array,
            batch_labels=batch_labels,
            batch_row_maps=batch_row_maps,
            image_name=image_name,
        )
        return

    import torch

    matrix_attr = _TEXTURE_MATRIX_ATTR[feature_class]
    pad_dim_by_class: Dict[str, Sequence[int]] = {
        "glcm": (2, 3),
        "ngtdm": (1,),
    }
    pad_dims = pad_dim_by_class[feature_class]

    matrices: List[torch.Tensor] = []
    for label in batch_labels:
        calculator.maskArray = (label_array == label)
        matrix = calculator._calculateMatrix(None)
        matrices.append(matrix)

    stacked = _pad_and_stack_torch(matrices, pad_dims=pad_dims, fill=0.0)
    setattr(calculator, matrix_attr, stacked)
    calculator._calculateCoefficients()
    feature_values = _extract_feature_values(calculator, feature_names)
    _assign_batch_feature_values(
        batch_row_maps,
        feature_class,
        feature_names,
        feature_values,
        image_name,
    )


def _extract_torch_glrlm_batch(
    calculator: object,
    feature_names: Sequence[str],
    *,
    label_array: np.ndarray,
    batch_labels: Sequence[int],
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
) -> None:
    """
    Batch GLRLM features after per-label matrix normalization.

    ``_calculateCoefficients`` removes empty run-length columns per ROI; doing
    that on a stacked batch would mix column sets across supervoxels.

    Args:
        calculator: TorchRadiomicsGLRLM instance.
        feature_names: Enabled GLRLM features.
        label_array: Full supervoxel label map.
        batch_labels: Label ids in this batch.
        batch_row_maps: Mutable row dicts to fill.
        image_name: Optional column suffix.
    """
    import torch

    per_label_states: List[Dict[str, torch.Tensor]] = []
    for label in batch_labels:
        calculator.maskArray = (label_array == label)
        calculator.P_glrlm = calculator._calculateMatrix(None)
        nr_tensor = calculator.coefficients["Nr"]
        calculator._calculateCoefficients()
        per_label_states.append(
            {
                "P_glrlm": calculator.P_glrlm,
                "pr": calculator.coefficients["pr"],
                "pg": calculator.coefficients["pg"],
                "jvector": calculator.coefficients["jvector"],
                "Nr": nr_tensor,
            }
        )

    calculator.P_glrlm = _pad_and_stack_torch(
        [state["P_glrlm"] for state in per_label_states],
        pad_dims=(2, 3),
        fill=0.0,
    )
    calculator.coefficients["pr"] = _pad_and_stack_torch(
        [state["pr"] for state in per_label_states],
        pad_dims=(1, 2),
        fill=0.0,
    )
    calculator.coefficients["pg"] = _pad_and_stack_torch(
        [state["pg"] for state in per_label_states],
        pad_dims=(1, 2),
        fill=0.0,
    )
    calculator.coefficients["Nr"] = _pad_and_stack_torch(
        [state["Nr"] for state in per_label_states],
        pad_dims=(1,),
        fill=float("nan"),
    )

    max_run_length = max(state["jvector"].shape[0] for state in per_label_states)
    calculator.coefficients["jvector"] = torch.arange(
        1,
        max_run_length + 1,
        dtype=calculator.P_glrlm.dtype,
        device=calculator.P_glrlm.device,
    )
    calculator.coefficients["ivector"] = calculator.tensor(
        calculator.coefficients["grayLevels"]
    )

    feature_values = _extract_feature_values(calculator, feature_names)
    _assign_batch_feature_values(
        batch_row_maps,
        "glrlm",
        feature_names,
        feature_values,
        image_name,
    )


def _extract_torch_glszm_batch(
    calculator: object,
    feature_names: Sequence[str],
    *,
    label_array: np.ndarray,
    batch_labels: Sequence[int],
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
) -> None:
    """
    Batch GLSZM features after per-label zone-size column pruning.

    Args:
        calculator: TorchRadiomicsGLSZM instance.
        feature_names: Enabled GLSZM features.
        label_array: Full supervoxel label map.
        batch_labels: Label ids in this batch.
        batch_row_maps: Mutable row dicts to fill.
        image_name: Optional column suffix.
    """
    import torch

    per_label_states: List[Dict[str, torch.Tensor]] = []
    for label in batch_labels:
        calculator.maskArray = (label_array == label)
        calculator.P_glszm = calculator._calculateMatrix(None)
        calculator._calculateCoefficients()
        per_label_states.append(
            {
                "P_glszm": calculator.P_glszm,
                "ps": calculator.coefficients["ps"],
                "pg": calculator.coefficients["pg"],
                "jvector": calculator.coefficients["jvector"],
                "Nz": calculator.coefficients["Nz"],
                "Np": calculator.coefficients["Np"],
            }
        )

    calculator.P_glszm = _pad_and_stack_torch(
        [state["P_glszm"] for state in per_label_states],
        pad_dims=(2,),
        fill=0.0,
    )
    calculator.coefficients["ps"] = _pad_and_stack_torch(
        [state["ps"] for state in per_label_states],
        pad_dims=(1,),
        fill=0.0,
    )
    calculator.coefficients["pg"] = _pad_and_stack_torch(
        [state["pg"] for state in per_label_states],
        pad_dims=(1,),
        fill=0.0,
    )
    calculator.coefficients["Nz"] = torch.cat(
        [
            state["Nz"].reshape(1) if state["Nz"].ndim == 0 else state["Nz"].reshape(1)
            for state in per_label_states
        ],
        dim=0,
    )
    calculator.coefficients["Np"] = torch.cat(
        [
            state["Np"].reshape(1) if state["Np"].ndim == 0 else state["Np"].reshape(1)
            for state in per_label_states
        ],
        dim=0,
    )

    max_zone_size = max(state["jvector"].shape[0] for state in per_label_states)
    calculator.coefficients["jvector"] = torch.arange(
        1,
        max_zone_size + 1,
        dtype=calculator.P_glszm.dtype,
        device=calculator.P_glszm.device,
    )
    calculator.coefficients["ivector"] = calculator.tensor(
        calculator.coefficients["grayLevels"]
    )

    feature_values = _extract_feature_values(calculator, feature_names)
    _assign_batch_feature_values(
        batch_row_maps,
        "glszm",
        feature_names,
        feature_values,
        image_name,
    )


def _extract_torch_gldm_batch(
    calculator: object,
    feature_names: Sequence[str],
    *,
    label_array: np.ndarray,
    batch_labels: Sequence[int],
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
) -> None:
    """
    Stack GLDM matrices and coefficient tensors for a label batch.

    GLDM sets ``pd``/``pg``/``Nz``/``jvector`` inside ``_calculateMatrix``; batch by
    padding the dependence-size dimension and stacking.

    Args:
        calculator: TorchRadiomicsGLDM instance.
        feature_names: Enabled GLDM features.
        label_array: Full supervoxel label map.
        batch_labels: Label ids in this batch.
        batch_row_maps: Mutable row dicts to fill.
        image_name: Optional column suffix.
    """
    import torch

    p_gldm_list: List[torch.Tensor] = []
    pd_list: List[torch.Tensor] = []
    pg_list: List[torch.Tensor] = []
    nz_list: List[torch.Tensor] = []

    for label in batch_labels:
        calculator.maskArray = (label_array == label)
        p_gldm = calculator._calculateMatrix(None)
        p_gldm_list.append(p_gldm)
        pd_list.append(calculator.coefficients["pd"])
        pg_list.append(calculator.coefficients["pg"])
        nz_list.append(calculator.coefficients["Nz"])

    stacked_p = _pad_and_stack_torch(p_gldm_list, pad_dims=(2,), fill=0.0)
    stacked_pd = _pad_and_stack_torch(pd_list, pad_dims=(1,), fill=0.0)
    stacked_pg = _pad_and_stack_torch(pg_list, pad_dims=(1,), fill=0.0)
    stacked_nz = torch.cat(
        [nz.reshape(1) if nz.ndim == 0 else nz.reshape(1) for nz in nz_list],
        dim=0,
    )

    max_nd = stacked_p.shape[2]
    jvector = torch.arange(1, max_nd + 1, dtype=stacked_p.dtype, device=stacked_p.device)

    calculator.P_gldm = stacked_p
    calculator.coefficients["pd"] = stacked_pd
    calculator.coefficients["pg"] = stacked_pg
    calculator.coefficients["Nz"] = stacked_nz
    calculator.coefficients["ivector"] = calculator.tensor(calculator.coefficients["grayLevels"])
    calculator.coefficients["jvector"] = jvector

    feature_values = _extract_feature_values(calculator, feature_names)
    _assign_batch_feature_values(
        batch_row_maps,
        "gldm",
        feature_names,
        feature_values,
        image_name,
    )


def _extract_torch_firstorder_batch(
    calculator: object,
    feature_names: Sequence[str],
    *,
    label_array: np.ndarray,
    batch_labels: Sequence[int],
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
) -> None:
    """
    Batch first-order statistics by padding ``targetVoxelArray`` per label.

    ``p_i`` histogram features still use per-label ``_initCalculation`` when any
    histogram-based feature is requested; otherwise voxel arrays are stacked.

    Args:
        calculator: TorchRadiomicsFirstOrder instance.
        feature_names: Enabled first-order features.
        label_array: Full supervoxel label map.
        batch_labels: Label ids in this batch.
        batch_row_maps: Mutable row dicts to fill.
        image_name: Optional column suffix.
    """
    histogram_features = {"Entropy", "Uniformity", "Energy", "TotalEnergy"}
    needs_histogram = any(name in histogram_features for name in feature_names)

    if needs_histogram or len(batch_labels) == 1:
        for label, row in zip(batch_labels, batch_row_maps):
            calculator.maskArray = (label_array == label)
            calculator._initCalculation(None)
            feature_values = _extract_feature_values(calculator, feature_names)
            for feature_name, values in feature_values.items():
                col = _feature_column_name("firstorder", feature_name, image_name)
                row[col] = _scalar_feature_value(values)
        return

    target_rows: List[np.ndarray] = []
    for label in batch_labels:
        calculator.maskArray = (label_array == label)
        calculator._initCalculation(None)
        target_rows.append(np.asarray(calculator.targetVoxelArray, dtype=np.float64))

    max_voxels = max(row.shape[1] for row in target_rows)
    target_batch = np.full((len(batch_labels), max_voxels), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(target_rows):
        target_batch[row_idx, : row.shape[1]] = row[0]

    calculator.targetVoxelArray = target_batch
    feature_values = _extract_feature_values(calculator, feature_names)
    _assign_batch_feature_values(
        batch_row_maps,
        "firstorder",
        feature_names,
        feature_values,
        image_name,
    )


def _extract_supervoxel_label_features_cext_batch(
    calculators: Mapping[str, object],
    resolved_features: Mapping[str, Sequence[str]],
    *,
    label_array: np.ndarray,
    batch_labels: Sequence[int],
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
) -> None:
    """
    Extract enabled features for a label batch via habit C-extension matrix batching.

    Args:
        calculators: Shared Torch calculators binned on the union mask.
        resolved_features: Enabled feature names per class.
        label_array: Full supervoxel label map array.
        batch_labels: Label ids in this batch.
        batch_row_maps: Pre-allocated row dicts (must include SupervoxelID).
        image_name: Optional column suffix.
    """
    extract_supervoxel_batch_via_cext(
        calculators,
        resolved_features,
        label_array=label_array,
        batch_labels=batch_labels,
        batch_row_maps=batch_row_maps,
        image_name=image_name,
        assign_batch_feature_values=_assign_batch_feature_values,
        extract_feature_values=_extract_feature_values,
        feature_column_name=_feature_column_name,
        pad_and_stack_torch=_pad_and_stack_torch,
    )


def _extract_supervoxel_label_features_torch_batch(
    calculators: Mapping[str, object],
    resolved_features: Mapping[str, Sequence[str]],
    *,
    label_array: np.ndarray,
    batch_labels: Sequence[int],
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
) -> None:
    """
    Extract enabled features for a supervoxel label batch using Torch formula batching.

    Outer loop is feature class; inner loop builds stacked matrices per batch.

    Args:
        calculators: Shared Torch calculators binned on the union mask.
        resolved_features: Enabled feature names per class.
        label_array: Full supervoxel label map array.
        batch_labels: Label ids in this batch.
        batch_row_maps: Pre-allocated row dicts (must include SupervoxelID).
        image_name: Optional column suffix.
    """
    for feature_class in sorted(calculators.keys()):
        feature_names = resolved_features.get(feature_class, [])
        if not feature_names:
            continue

        calculator = calculators[feature_class]
        try:
            if feature_class in _TEXTURE_MATRIX_ATTR:
                _extract_torch_texture_batch(
                    calculator,
                    feature_class,
                    feature_names,
                    label_array=label_array,
                    batch_labels=batch_labels,
                    batch_row_maps=batch_row_maps,
                    image_name=image_name,
                )
            elif feature_class == "gldm":
                _extract_torch_gldm_batch(
                    calculator,
                    feature_names,
                    label_array=label_array,
                    batch_labels=batch_labels,
                    batch_row_maps=batch_row_maps,
                    image_name=image_name,
                )
            elif feature_class == "firstorder":
                _extract_torch_firstorder_batch(
                    calculator,
                    feature_names,
                    label_array=label_array,
                    batch_labels=batch_labels,
                    batch_row_maps=batch_row_maps,
                    image_name=image_name,
                )
            else:
                raise ValueError(f"Unsupported Torch feature class: {feature_class}")
        except Exception as exc:
            logger.warning(
                "Failed Torch batched extraction for class %s labels %s: %s",
                feature_class,
                batch_labels,
                exc,
            )


def _create_torch_calculators(
    image: sitk.Image,
    union_mask: sitk.Image,
    feature_classes: Sequence[str],
    torch_settings: Dict[str, object],
) -> Dict[str, object]:
    """
    Instantiate TorchRadiomics calculators once with union-mask binning.

    Each calculator ``__init__`` calls PyRadiomics ``_applyBinning`` on the union
    mask, matching voxelBased single-mask discretization semantics.

    Args:
        image: Input SimpleITK image.
        union_mask: Binary mask of all supervoxel labels.
        feature_classes: Enabled feature class names.
        torch_settings: PyRadiomics/TorchRadiomics kwargs.

    Returns:
        Dict[str, object]: Feature class name to calculator instance.
    """
    from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsFirstOrder import (
        TorchRadiomicsFirstOrder,
    )
    from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsGLCM import (
        TorchRadiomicsGLCM,
    )
    from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsGLDM import (
        TorchRadiomicsGLDM,
    )
    from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsGLRLM import (
        TorchRadiomicsGLRLM,
    )
    from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsGLSZM import (
        TorchRadiomicsGLSZM,
    )
    from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsNGTDM import (
        TorchRadiomicsNGTDM,
    )

    calculator_types: Dict[str, Type[object]] = {
        "firstorder": TorchRadiomicsFirstOrder,
        "glcm": TorchRadiomicsGLCM,
        "glrlm": TorchRadiomicsGLRLM,
        "glszm": TorchRadiomicsGLSZM,
        "ngtdm": TorchRadiomicsNGTDM,
        "gldm": TorchRadiomicsGLDM,
    }

    calculators: Dict[str, object] = {}
    for feature_class in feature_classes:
        calculator_cls = calculator_types.get(feature_class)
        if calculator_cls is None:
            continue
        calculators[feature_class] = calculator_cls(image, union_mask, **torch_settings)
    return calculators


def _create_pyradiomics_calculators(
    image: sitk.Image,
    union_mask: sitk.Image,
    feature_classes: Sequence[str],
    settings: Dict[str, object],
) -> Dict[str, object]:
    """
    Instantiate native PyRadiomics calculators once with union-mask binning.

    Args:
        image: Input SimpleITK image.
        union_mask: Binary mask of all supervoxel labels.
        feature_classes: Enabled feature class names.
        settings: PyRadiomics settings dict.

    Returns:
        Dict[str, object]: Feature class name to calculator instance.
    """
    from radiomics import firstorder, glcm, gldm, glrlm, glszm, ngtdm

    calculator_types: Dict[str, Type[object]] = {
        "firstorder": firstorder.RadiomicsFirstOrder,
        "glcm": glcm.RadiomicsGLCM,
        "glrlm": glrlm.RadiomicsGLRLM,
        "glszm": glszm.RadiomicsGLSZM,
        "ngtdm": ngtdm.RadiomicsNGTDM,
        "gldm": gldm.RadiomicsGLDM,
    }
    calculators: Dict[str, object] = {}
    for feature_class in feature_classes:
        calculator_cls = calculator_types.get(feature_class)
        if calculator_cls is None:
            continue
        calculators[feature_class] = calculator_cls(image, union_mask, **settings)
    return calculators


def _extract_supervoxel_label_features(
    calculators: Mapping[str, object],
    resolved_features: Mapping[str, Sequence[str]],
    *,
    label_array: np.ndarray,
    label: int,
    image_name: str,
) -> Dict[str, float]:
    """
    Extract all enabled features for one supervoxel label.

    Args:
        calculators: Shared calculators binned on the union mask.
        resolved_features: Enabled feature names per class.
        label_array: Full supervoxel label map array.
        label: Supervoxel label id.
        image_name: Optional column suffix.

    Returns:
        Dict[str, float]: Feature column name to scalar value.
    """
    label_mask: np.ndarray = (label_array == label)
    row: Dict[str, float] = {}

    for feature_class, calculator in calculators.items():
        feature_names = resolved_features.get(feature_class, [])
        if not feature_names:
            continue

        calculator.maskArray = label_mask
        calculator._initCalculation(None)
        feature_values = _extract_feature_values(calculator, feature_names)
        for feature_name, values in feature_values.items():
            col = _feature_column_name(feature_class, feature_name, image_name)
            row[col] = _scalar_feature_value(values)

    return row


def _calculate_supervoxels(
    image: sitk.Image,
    supervoxel_map: sitk.Image,
    labels: np.ndarray,
    *,
    resolved_features: Mapping[str, Sequence[str]],
    calculators: Mapping[str, object],
    image_name: str = "",
    batch_size: int = DEFAULT_SUPERVOXEL_BATCH,
    progress_callback: Optional[Callable[[int], None]] = None,
    enable_torch_formula_batch: bool = False,
    use_cext_batch: bool = False,
) -> pd.DataFrame:
    """
    Extract supervoxel features using shared union-mask binning.

    When ``enable_torch_formula_batch`` is True, each feature class stacks ROI
    matrices for ``batch_size`` labels and runs Torch ``_calculateCoefficients``
    once per class per batch (voxelBased-style formula batching).

    Args:
        image: Input SimpleITK image.
        supervoxel_map: Multi-label supervoxel map aligned with image.
        labels: 1D supervoxel label ids to extract.
        resolved_features: Enabled feature names per class.
        calculators: Pre-initialized shared calculators (union mask binning).
        image_name: Optional feature column suffix.
        batch_size: Labels per batch group.
        progress_callback: Optional callback invoked once per processed label.
        enable_torch_formula_batch: Use stacked-matrix Torch formula batching.
        use_cext_batch: Use habit C-extension batched matrix path when True.

    Returns:
        pd.DataFrame: One row per supervoxel.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")

    _, label_array = _build_union_mask(supervoxel_map, image)
    rows: List[Dict[str, object]] = []

    for start in range(0, len(labels), batch_size):
        batch_labels_arr = labels[start:start + batch_size].astype(np.int64, copy=False)
        batch_labels = [int(label) for label in batch_labels_arr.tolist()]
        batch_row_maps: List[Dict[str, object]] = [
            {"SupervoxelID": label} for label in batch_labels
        ]

        if enable_torch_formula_batch:
            try:
                if use_cext_batch:
                    _extract_supervoxel_label_features_cext_batch(
                        calculators,
                        resolved_features,
                        label_array=label_array,
                        batch_labels=batch_labels,
                        batch_row_maps=batch_row_maps,
                        image_name=image_name,
                    )
                else:
                    _extract_supervoxel_label_features_torch_batch(
                        calculators,
                        resolved_features,
                        label_array=label_array,
                        batch_labels=batch_labels,
                        batch_row_maps=batch_row_maps,
                        image_name=image_name,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed Torch batched supervoxel extraction for labels %s: %s",
                    batch_labels,
                    exc,
                )
            rows.extend(batch_row_maps)
            if progress_callback is not None:
                progress_callback(len(batch_labels))
            continue

        for label, row in zip(batch_labels, batch_row_maps):
            try:
                feature_row = _extract_supervoxel_label_features(
                    calculators,
                    resolved_features,
                    label_array=label_array,
                    label=label,
                    image_name=image_name,
                )
                row.update(feature_row)
            except Exception as exc:
                logger.warning(
                    "Failed batched supervoxel extraction for label %s: %s",
                    label,
                    exc,
                )
            if progress_callback is not None:
                progress_callback(1)

        rows.extend(batch_row_maps)

    return pd.DataFrame(rows)


def extract_batched_supervoxel_features(
    image: sitk.Image,
    supervoxel_map: sitk.Image,
    labels: np.ndarray,
    *,
    enabled_features: Mapping[str, object],
    image_name: str = "",
    settings: Optional[Dict[str, object]] = None,
    device: str = "cuda:0",
    dtype_name: str = "float64",
    batch_size: int = DEFAULT_SUPERVOXEL_BATCH,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> pd.DataFrame:
    """
    Extract supervoxel ROI radiomics on GPU/CPU via TorchRadiomics.

    Discretization uses PyRadiomics ``_applyBinning`` once on the union supervoxel
    mask; per-label matrices use ``cMatrices`` inside ``_initCalculation``, or the
    habit C-extension batch path when ``useSupervoxelCext`` resolves to true (default
    ``auto`` when ``_sv_cmatrices`` is compiled).

    Args:
        image: Input SimpleITK image.
        supervoxel_map: Multi-label supervoxel map aligned with image.
        labels: 1D array of supervoxel label ids to extract.
        enabled_features: PyRadiomics enabledFeatures mapping.
        image_name: Optional suffix appended to feature column names.
        settings: PyRadiomics settings dict.
        device: Torch device, e.g. ``cuda:0`` or ``cpu``.
        dtype_name: ``float32`` or ``float64``.
        batch_size: Number of supervoxels per batch group.
        progress_callback: Optional callback invoked once per processed label.

    Returns:
        pd.DataFrame: One row per supervoxel with PyRadiomics-style column names.
    """
    import torch

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")

    settings = dict(settings or {})
    resolved = _resolve_enabled_features(enabled_features)
    if not resolved:
        raise ValueError("No enabled supervoxel radiomics feature classes.")

    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    label_array = sitk.GetArrayFromImage(supervoxel_map)
    if not np.any(label_array > 0):
        raise ValueError("Supervoxel map has no non-zero labels.")

    image, union_mask, supervoxel_map, pad_distance, union_bbox_crop = (
        _prepare_supervoxel_volumes(image, supervoxel_map, settings)
    )
    torch_settings = dict(settings)
    torch_settings["device"] = device
    torch_settings["dtype"] = dtype

    use_torch_setting = str(settings.get("useTorchRadiomics", "auto"))
    if str(device).startswith("cuda"):
        logger.info(
            "batched supervoxel_radiomics using TorchRadiomics GPU: "
            "useTorchRadiomics=%s device=%s dtype=%s labels=%d batch_size=%d "
            "union_bin=True union_bbox_crop=%s padDistance=%d cropped_size=%s",
            use_torch_setting,
            device,
            dtype_name,
            len(labels),
            batch_size,
            union_bbox_crop,
            pad_distance,
            image.GetSize(),
        )
    else:
        logger.info(
            "batched supervoxel_radiomics using TorchRadiomics CPU: "
            "useTorchRadiomics=%s device=%s labels=%d batch_size=%d "
            "union_bin=True union_bbox_crop=%s padDistance=%d cropped_size=%s",
            use_torch_setting,
            device,
            len(labels),
            batch_size,
            union_bbox_crop,
            pad_distance,
            image.GetSize(),
        )

    calculators = _create_torch_calculators(
        image,
        union_mask,
        sorted(resolved.keys()),
        torch_settings,
    )

    use_cext_batch = resolve_use_supervoxel_cext(settings)
    matrix_backend = supervoxel_cext_matrix_backend_label(settings)
    use_supervoxel_cext_flag = settings.get("useSupervoxelCext", "auto")
    if matrix_backend == "habit_native_c":
        logger.info(
            "supervoxel_radiomics using habit native C extension for texture matrices: "
            "useSupervoxelCext=%s module=supervoxel_cext._sv_cmatrices backend=%s "
            "labels=%d batch_size=%d",
            use_supervoxel_cext_flag,
            cext_backend(),
            len(labels),
            batch_size,
        )
    elif matrix_backend == "habit_fallback_cmatrices":
        logger.warning(
            "supervoxel_radiomics useSupervoxelCext=%s but native extension is not built; "
            "using PyRadiomics cMatrices fallback (labels=%d batch_size=%d). "
            "Run: pip install -e .",
            use_supervoxel_cext_flag,
            len(labels),
            batch_size,
        )
    elif use_supervoxel_cext_flag not in (False, "false", "False"):
        logger.info(
            "supervoxel_radiomics habit native C extension not selected: "
            "useSupervoxelCext=%s native_available=%s matrix_backend=%s labels=%d batch_size=%d",
            use_supervoxel_cext_flag,
            is_cext_available(),
            matrix_backend,
            len(labels),
            batch_size,
        )

    return _calculate_supervoxels(
        image,
        supervoxel_map,
        labels,
        resolved_features=resolved,
        calculators=calculators,
        image_name=image_name,
        batch_size=batch_size,
        progress_callback=progress_callback,
        enable_torch_formula_batch=True,
        use_cext_batch=use_cext_batch,
    )


def extract_supervoxel_features_pyradiomics(
    image: sitk.Image,
    supervoxel_map: sitk.Image,
    labels: np.ndarray,
    *,
    enabled_features: Mapping[str, object],
    image_name: str = "",
    settings: Optional[Dict[str, object]] = None,
    batch_size: int = DEFAULT_SUPERVOXEL_BATCH,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> pd.DataFrame:
    """
    Extract supervoxel ROI radiomics via native CPU PyRadiomics with union-mask bin.

    Args:
        image: Input SimpleITK image.
        supervoxel_map: Multi-label supervoxel map.
        labels: Supervoxel label ids.
        enabled_features: PyRadiomics enabledFeatures mapping.
        image_name: Optional feature column suffix.
        settings: PyRadiomics settings dict.
        batch_size: Labels per batch group.
        progress_callback: Optional callback invoked once per processed label.

    Returns:
        pd.DataFrame: One row per supervoxel.
    """
    settings = dict(settings or {})
    resolved = _resolve_enabled_features(enabled_features)
    if not resolved:
        raise ValueError("No enabled supervoxel radiomics feature classes.")

    label_array = sitk.GetArrayFromImage(supervoxel_map)
    if not np.any(label_array > 0):
        raise ValueError("Supervoxel map has no non-zero labels.")

    image, union_mask, supervoxel_map, pad_distance, union_bbox_crop = (
        _prepare_supervoxel_volumes(image, supervoxel_map, settings)
    )
    calculators = _create_pyradiomics_calculators(
        image,
        union_mask,
        sorted(resolved.keys()),
        settings,
    )

    logger.info(
        "supervoxel_radiomics using CPU PyRadiomics union_bin=True "
        "union_bbox_crop=%s padDistance=%d cropped_size=%s labels=%d batch_size=%d",
        union_bbox_crop,
        pad_distance,
        image.GetSize(),
        len(labels),
        batch_size,
    )

    return _calculate_supervoxels(
        image,
        supervoxel_map,
        labels,
        resolved_features=resolved,
        calculators=calculators,
        image_name=image_name,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )


def extract_batched_supervoxel_firstorder(
    image: sitk.Image,
    supervoxel_map: sitk.Image,
    labels: np.ndarray,
    *,
    image_name: str = "",
    enabled_features: Sequence[str] | None = None,
    settings: Dict[str, object] | None = None,
    device: str,
    dtype: object,
    batch_size: int = DEFAULT_SUPERVOXEL_BATCH,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper extracting first-order features only.

    Args:
        image: Input SimpleITK image.
        supervoxel_map: Supervoxel label map.
        labels: Supervoxel label ids.
        image_name: Optional feature-name suffix.
        enabled_features: First-order feature names; defaults to full class list.
        settings: PyRadiomics settings.
        device: Torch device string.
        dtype: ``torch.float32`` or ``torch.float64``.
        batch_size: Batch size.

    Returns:
        pd.DataFrame: First-order features per supervoxel.
    """
    import torch

    dtype_name = "float64" if dtype == torch.float64 else "float32"
    feature_list = (
        list(enabled_features)
        if enabled_features
        else DEFAULT_FEATURES_BY_CLASS["firstorder"]
    )
    enabled_map = {"firstorder": feature_list}
    return extract_batched_supervoxel_features(
        image,
        supervoxel_map,
        labels,
        enabled_features=enabled_map,
        image_name=image_name,
        settings=settings,
        device=device,
        dtype_name=dtype_name,
        batch_size=batch_size,
    )
