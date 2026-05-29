"""
Bridge raw batched C-extension matrices into TorchRadiomics coefficient evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

from habit.core.habitat_analysis.clustering_features.supervoxel_cext import (
    calculate_firstorder,
    calculate_gldm,
    calculate_glcm,
    calculate_glrlm,
    calculate_glszm,
    calculate_ngtdm,
    cext_backend,
)
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

# Column order returned by ``sv_calculate_firstorder`` / ``calculate_firstorder``.
FIRSTORDER_CEXT_COLUMNS: Sequence[str] = (
    "Energy",
    "TotalEnergy",
    "Entropy",
    "Minimum",
    "10Percentile",
    "90Percentile",
    "Maximum",
    "Mean",
    "Median",
    "InterquartileRange",
    "Range",
    "MeanAbsoluteDeviation",
    "RobustMeanAbsoluteDeviation",
    "RootMeanSquared",
    "Skewness",
    "Kurtosis",
    "Uniformity",
)


def _postprocess_glcm_single_roi(calculator: object, p_glcm: object, angles: object) -> object:
    """Apply TorchRadiomics GLCM matrix post-processing for one ROI batch item."""
    import torch

    gray_idx = calculator.gray_level_index(calculator.coefficients["grayLevels"], p_glcm)
    p_glcm = p_glcm[:, gray_idx, :, :]
    p_glcm = p_glcm[:, :, gray_idx, :]

    if getattr(calculator, "symmetricalGLCM", True):
        p_glcm = p_glcm + p_glcm.permute((0, 2, 1, 3))

    weighting_norm = getattr(calculator, "weightingNorm", None)
    if weighting_norm is not None:
        pixel_spacing = calculator.inputImage.GetSpacing()[::-1]
        weights = torch.empty(len(angles), dtype=calculator.dtype, device=calculator.device)
        for a_idx, angle in enumerate(angles):
            if weighting_norm == "infinity":
                weights[a_idx] = torch.exp(-torch.max(torch.abs(angle) * torch.tensor(pixel_spacing, device=calculator.device)) ** 2)
            elif weighting_norm == "euclidean":
                weights[a_idx] = torch.exp(-torch.sum((torch.abs(angle) * torch.tensor(pixel_spacing, device=calculator.device)) ** 2))
            elif weighting_norm == "manhattan":
                weights[a_idx] = torch.exp(-torch.sum(torch.abs(angle) * torch.tensor(pixel_spacing, device=calculator.device)) ** 2)
            else:
                weights[a_idx] = 1
        p_glcm = torch.sum(p_glcm * weights[None, None, None, :], 3, keepdims=True)

    sum_p = torch.sum(p_glcm, (1, 2))
    if p_glcm.shape[3] > 1:
        empty_angles = torch.where(torch.sum(sum_p, 0) == 0)
        if len(empty_angles[0]) > 0:
            p_glcm = calculator.delete(p_glcm, empty_angles, 3)
            sum_p = calculator.delete(sum_p, empty_angles, 1)

    sum_p[sum_p == 0] = torch.nan
    p_glcm = p_glcm / sum_p[:, None, None, :]
    return p_glcm


def _postprocess_glrlm_single_roi(calculator: object, p_glrlm: object, angles: object) -> object:
    """Apply TorchRadiomics GLRLM matrix post-processing for one ROI batch item."""
    import torch

    gray_idx = calculator.gray_level_index(calculator.coefficients["grayLevels"], p_glrlm)
    p_glrlm = p_glrlm[:, gray_idx]

    weighting_norm = getattr(calculator, "weightingNorm", None)
    if weighting_norm is not None:
        pixel_spacing = calculator.inputImage.GetSpacing()[::-1]
        weights = torch.empty(len(angles), dtype=calculator.dtype, device=calculator.device)
        for a_idx, angle in enumerate(angles):
            if weighting_norm == "infinity":
                weights[a_idx] = torch.max(torch.abs(angle) * torch.tensor(pixel_spacing, device=calculator.device))
            elif weighting_norm == "euclidean":
                weights[a_idx] = torch.sqrt(torch.sum((torch.abs(angle) * torch.tensor(pixel_spacing, device=calculator.device)) ** 2))
            elif weighting_norm == "manhattan":
                weights[a_idx] = torch.sum(torch.abs(angle) * torch.tensor(pixel_spacing, device=calculator.device))
            else:
                weights[a_idx] = 1
        p_glrlm = torch.sum(p_glrlm * weights[None, None, None, :], 3, keepdims=True)

    nr_tensor = torch.sum(p_glrlm, (1, 2))
    if p_glrlm.shape[3] > 1:
        empty_angles = torch.where(torch.sum(nr_tensor, 0) == 0)
        if len(empty_angles[0]) > 0:
            p_glrlm = calculator.delete(p_glrlm, empty_angles, 3)
            nr_tensor = calculator.delete(nr_tensor, empty_angles, 1)

    nr_tensor[nr_tensor == 0] = torch.nan
    calculator.coefficients["Nr"] = nr_tensor
    return p_glrlm


def _postprocess_glszm_single_roi(calculator: object, p_glszm: object) -> object:
    """Apply TorchRadiomics GLSZM gray-level pruning for one ROI batch item."""
    gray_idx = calculator.gray_level_index(calculator.coefficients["grayLevels"], p_glszm)
    return p_glszm[:, gray_idx, :]


def _postprocess_ngtdm_batch(calculator: object, p_ngtdm: object) -> object:
    """Apply TorchRadiomics NGTDM gray-level pruning for a label batch."""
    import torch

    empty_gray_levels = torch.where(torch.sum(p_ngtdm[:, :, 0], 0) == 0)
    return calculator.delete(p_ngtdm, empty_gray_levels, 1)


def _postprocess_gldm_single_roi(calculator: object, p_gldm: object) -> object:
    """Apply TorchRadiomics GLDM matrix post-processing for one ROI batch item."""
    import torch

    gray_idx = calculator.gray_level_index(calculator.coefficients["grayLevels"], p_gldm)
    p_gldm = p_gldm[:, gray_idx, :]

    jvector = torch.arange(1, p_gldm.shape[2] + 1, dtype=calculator.dtype, device=calculator.device)
    pd = torch.sum(p_gldm, 1)
    pg = torch.sum(p_gldm, 2)

    empty_sizes = torch.sum(pd, 0)
    p_gldm = calculator.delete(p_gldm, torch.where(empty_sizes == 0), 2)
    jvector = calculator.delete(jvector, torch.where(empty_sizes == 0), 0)
    pd = calculator.delete(pd, torch.where(empty_sizes == 0), 1)

    nz = torch.sum(pd, 1)
    nz[nz == 0] = 1

    calculator.coefficients["Nz"] = nz
    calculator.coefficients["pd"] = pd
    calculator.coefficients["pg"] = pg
    calculator.coefficients["ivector"] = calculator.tensor(calculator.coefficients["grayLevels"])
    calculator.coefficients["jvector"] = jvector
    return p_gldm


def extract_supervoxel_batch_via_cext(
    calculators: Mapping[str, object],
    resolved_features: Mapping[str, Sequence[str]],
    *,
    label_array: np.ndarray,
    batch_labels: Sequence[int],
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
    assign_batch_feature_values,
    extract_feature_values,
    feature_column_name,
    pad_and_stack_torch,
) -> None:
    """
    Fill ``batch_row_maps`` using batched C-extension matrices + Torch formulas.

    Args:
        calculators: Shared Torch calculators binned on the union mask.
        resolved_features: Enabled feature names per class.
        label_array: Full supervoxel label map.
        batch_labels: Label ids in this batch.
        batch_row_maps: Mutable row dicts to fill.
        image_name: Optional column suffix.
        assign_batch_feature_values: Callback from batched_supervoxel_radiomics.
        extract_feature_values: Callback from batched_supervoxel_radiomics.
        feature_column_name: Callback from batched_supervoxel_radiomics.
        pad_and_stack_torch: Callback from batched_supervoxel_radiomics.
    """
    import torch

    if not batch_labels:
        return

    labels_arr = np.asarray(batch_labels, dtype=np.int32)
    sv_map_i = np.ascontiguousarray(label_array.astype(np.int32))
    settings = calculators[next(iter(calculators))].settings
    force2d = int(bool(settings.get("force2D", False)))
    force2d_dim = int(settings.get("force2Ddimension", 0))
    distances = np.asarray(settings.get("distances", [1]), dtype=np.int32)

    logger.info(
        "habit cext batch extracting texture matrices: backend=%s labels=%d classes=%s",
        cext_backend(),
        len(batch_labels),
        sorted(calculators.keys()),
    )

    for feature_class in sorted(calculators.keys()):
        feature_names = list(resolved_features.get(feature_class, []))
        if not feature_names:
            continue

        calculator = calculators[feature_class]
        try:
            if feature_class == "glcm":
                _extract_glcm_cext_batch(
                    calculator,
                    feature_names,
                    sv_map_i=sv_map_i,
                    labels_arr=labels_arr,
                    batch_row_maps=batch_row_maps,
                    image_name=image_name,
                    distances=distances,
                    force2d=force2d,
                    force2d_dim=force2d_dim,
                    assign_batch_feature_values=assign_batch_feature_values,
                    extract_feature_values=extract_feature_values,
                    feature_column_name=feature_column_name,
                )
            elif feature_class == "glrlm":
                _extract_glrlm_cext_batch(
                    calculator,
                    feature_names,
                    sv_map_i=sv_map_i,
                    labels_arr=labels_arr,
                    batch_row_maps=batch_row_maps,
                    image_name=image_name,
                    force2d=force2d,
                    force2d_dim=force2d_dim,
                    assign_batch_feature_values=assign_batch_feature_values,
                    extract_feature_values=extract_feature_values,
                    pad_and_stack_torch=pad_and_stack_torch,
                )
            elif feature_class == "glszm":
                _extract_glszm_cext_batch(
                    calculator,
                    feature_names,
                    sv_map_i=sv_map_i,
                    labels_arr=labels_arr,
                    batch_row_maps=batch_row_maps,
                    image_name=image_name,
                    force2d=force2d,
                    force2d_dim=force2d_dim,
                    assign_batch_feature_values=assign_batch_feature_values,
                    extract_feature_values=extract_feature_values,
                    pad_and_stack_torch=pad_and_stack_torch,
                )
            elif feature_class == "ngtdm":
                _extract_ngtdm_cext_batch(
                    calculator,
                    feature_names,
                    sv_map_i=sv_map_i,
                    labels_arr=labels_arr,
                    batch_row_maps=batch_row_maps,
                    image_name=image_name,
                    distances=distances,
                    force2d=force2d,
                    force2d_dim=force2d_dim,
                    assign_batch_feature_values=assign_batch_feature_values,
                    extract_feature_values=extract_feature_values,
                    pad_and_stack_torch=pad_and_stack_torch,
                )
            elif feature_class == "gldm":
                _extract_gldm_cext_batch(
                    calculator,
                    feature_names,
                    sv_map_i=sv_map_i,
                    labels_arr=labels_arr,
                    batch_row_maps=batch_row_maps,
                    image_name=image_name,
                    distances=distances,
                    force2d=force2d,
                    force2d_dim=force2d_dim,
                    assign_batch_feature_values=assign_batch_feature_values,
                    extract_feature_values=extract_feature_values,
                    pad_and_stack_torch=pad_and_stack_torch,
                )
            elif feature_class == "firstorder":
                _extract_firstorder_cext_batch(
                    calculator,
                    feature_names,
                    sv_map_i=sv_map_i,
                    labels_arr=labels_arr,
                    batch_row_maps=batch_row_maps,
                    image_name=image_name,
                    feature_column_name=feature_column_name,
                )
            else:
                raise ValueError(f"Unsupported cext feature class: {feature_class}")
        except Exception as exc:
            logger.warning(
                "Failed cext batched extraction for class %s labels %s: %s",
                feature_class,
                batch_labels,
                exc,
            )


def _extract_glcm_cext_batch(
    calculator: object,
    feature_names: Sequence[str],
    *,
    sv_map_i: np.ndarray,
    labels_arr: np.ndarray,
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
    distances: np.ndarray,
    force2d: int,
    force2d_dim: int,
    assign_batch_feature_values,
    extract_feature_values,
    feature_column_name,
) -> None:
    """Extract GLCM features per label to preserve ROI-specific gray-level pruning."""
    del feature_column_name

    image_i = np.ascontiguousarray(calculator.imageArray.astype(np.int32))
    ng = int(calculator.coefficients["Ng"])
    p_raw, angles_np = calculate_glcm(
        image_i,
        sv_map_i,
        labels_arr,
        distances,
        ng,
        force2d,
        force2d_dim,
    )
    p_tensor = calculator.tensor(p_raw.astype(np.float64))
    angles = calculator.tensor(angles_np.astype(np.float64))

    for idx in range(p_tensor.shape[0]):
        one = _postprocess_glcm_single_roi(calculator, p_tensor[idx : idx + 1], angles)
        calculator.P_glcm = one
        calculator._calculateCoefficients()
        feature_values = extract_feature_values(calculator, feature_names)
        assign_batch_feature_values(
            [batch_row_maps[idx]],
            "glcm",
            feature_names,
            feature_values,
            image_name,
        )


def _extract_glrlm_cext_batch(
    calculator: object,
    feature_names: Sequence[str],
    *,
    sv_map_i: np.ndarray,
    labels_arr: np.ndarray,
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
    force2d: int,
    force2d_dim: int,
    assign_batch_feature_values,
    extract_feature_values,
    pad_and_stack_torch,
) -> None:
    """Extract GLRLM features per label to preserve pruned run-length vectors."""
    del pad_and_stack_torch

    image_i = np.ascontiguousarray(calculator.imageArray.astype(np.int32))
    ng = int(calculator.coefficients["Ng"])
    nr = int(np.max(calculator.imageArray.shape))
    p_raw, angles_np = calculate_glrlm(
        image_i,
        sv_map_i,
        labels_arr,
        ng,
        nr,
        force2d,
        force2d_dim,
    )
    angles = calculator.tensor(angles_np.astype(np.float64))

    for idx in range(p_raw.shape[0]):
        one = calculator.tensor(p_raw[idx : idx + 1].astype(np.float64))
        one = _postprocess_glrlm_single_roi(calculator, one, angles)
        calculator.P_glrlm = one
        calculator._calculateCoefficients()
        feature_values = extract_feature_values(calculator, feature_names)
        assign_batch_feature_values(
            [batch_row_maps[idx]],
            "glrlm",
            feature_names,
            feature_values,
            image_name,
        )


def _extract_glszm_cext_batch(
    calculator: object,
    feature_names: Sequence[str],
    *,
    sv_map_i: np.ndarray,
    labels_arr: np.ndarray,
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
    force2d: int,
    force2d_dim: int,
    assign_batch_feature_values,
    extract_feature_values,
    pad_and_stack_torch,
) -> None:
    """Extract GLSZM features per label to preserve pruned zone-size vectors."""
    del pad_and_stack_torch

    image_i = np.ascontiguousarray(calculator.imageArray.astype(np.int32))
    ng = int(calculator.coefficients["Ng"])
    p_raw = calculate_glszm(image_i, sv_map_i, labels_arr, ng, force2d, force2d_dim)

    for idx in range(p_raw.shape[0]):
        one = calculator.tensor(p_raw[idx : idx + 1].astype(np.float64))
        one = _postprocess_glszm_single_roi(calculator, one)
        calculator.P_glszm = one
        calculator._calculateCoefficients()
        feature_values = extract_feature_values(calculator, feature_names)
        assign_batch_feature_values(
            [batch_row_maps[idx]],
            "glszm",
            feature_names,
            feature_values,
            image_name,
        )


def _extract_ngtdm_cext_batch(
    calculator: object,
    feature_names: Sequence[str],
    *,
    sv_map_i: np.ndarray,
    labels_arr: np.ndarray,
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
    distances: np.ndarray,
    force2d: int,
    force2d_dim: int,
    assign_batch_feature_values,
    extract_feature_values,
    pad_and_stack_torch,
) -> None:
    image_i = np.ascontiguousarray(calculator.imageArray.astype(np.int32))
    ng = int(calculator.coefficients["Ng"])
    p_raw = calculate_ngtdm(
        image_i,
        sv_map_i,
        labels_arr,
        distances,
        ng,
        force2d,
        force2d_dim,
    )
    p_tensor = calculator.tensor(p_raw.astype(np.float64))
    p_tensor = _postprocess_ngtdm_batch(calculator, p_tensor)
    calculator.P_ngtdm = p_tensor
    calculator._calculateCoefficients()
    feature_values = extract_feature_values(calculator, feature_names)
    assign_batch_feature_values(
        batch_row_maps,
        "ngtdm",
        feature_names,
        feature_values,
        image_name,
    )


def _extract_gldm_cext_batch(
    calculator: object,
    feature_names: Sequence[str],
    *,
    sv_map_i: np.ndarray,
    labels_arr: np.ndarray,
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
    distances: np.ndarray,
    force2d: int,
    force2d_dim: int,
    assign_batch_feature_values,
    extract_feature_values,
    pad_and_stack_torch,
) -> None:
    """Extract GLDM features per label to preserve pruned dependence-size vectors."""
    del pad_and_stack_torch

    image_i = np.ascontiguousarray(calculator.imageArray.astype(np.int32))
    ng = int(calculator.coefficients["Ng"])
    alpha = int(getattr(calculator, "gldm_a", settings_gldm_alpha(calculator)))
    p_raw = calculate_gldm(
        image_i,
        sv_map_i,
        labels_arr,
        distances,
        ng,
        alpha,
        force2d,
        force2d_dim,
    )

    for idx in range(p_raw.shape[0]):
        one = calculator.tensor(p_raw[idx : idx + 1].astype(np.float64))
        one = _postprocess_gldm_single_roi(calculator, one)
        calculator.P_gldm = one
        feature_values = extract_feature_values(calculator, feature_names)
        assign_batch_feature_values(
            [batch_row_maps[idx]],
            "gldm",
            feature_names,
            feature_values,
            image_name,
        )


def settings_gldm_alpha(calculator: object) -> int:
    """Read GLDM alpha from calculator settings."""
    return int(calculator.settings.get("gldm_a", 0))


def _extract_firstorder_cext_batch(
    calculator: object,
    feature_names: Sequence[str],
    *,
    sv_map_i: np.ndarray,
    labels_arr: np.ndarray,
    batch_row_maps: List[Dict[str, object]],
    image_name: str,
    feature_column_name,
) -> None:
    image_f = np.ascontiguousarray(calculator.imageArray.astype(np.float64))
    ng = int(calculator.coefficients["Ng"])
    bin_width = float(calculator.settings.get("binWidth", 25))
    stats = calculate_firstorder(image_f, sv_map_i, labels_arr, ng, bin_width)

    name_to_idx = {name: idx for idx, name in enumerate(FIRSTORDER_CEXT_COLUMNS)}
    for row_idx, row in enumerate(batch_row_maps):
        for feature_name in feature_names:
            col = feature_column_name("firstorder", feature_name, image_name)
            stat_idx = name_to_idx.get(feature_name)
            if stat_idx is None:
                row[col] = float("nan")
            else:
                row[col] = float(stats[row_idx, stat_idx])
