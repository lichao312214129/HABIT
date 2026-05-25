"""
Batched supervoxel ROI radiomics mirroring PyRadiomics voxelBased orchestration.

Flow (aligned with ``RadiomicsFeaturesBase._calculateVoxels``):

1. Discretize once on the union supervoxel mask via PyRadiomics ``_applyBinning``.
2. Iterate supervoxels in batches (``supervoxelBatch``).
3. For each supervoxel label, swap ``maskArray`` and call ``_initCalculation`` so
   ``cMatrices`` builds ROI texture matrices (PyRadiomics native, no custom binning).
4. Run TorchRadiomics (or CPU PyRadiomics) ``get*FeatureValue`` for enabled features.

Torch is lazy-imported so machines without PyTorch can still import this module.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
import SimpleITK as sitk

from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

# Default batch size; groups supervoxel row assembly (like voxelBatch for voxel maps).
DEFAULT_SUPERVOXEL_BATCH = 64

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
        "LongRunHighGrayLevelRunEmphasis",
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
) -> pd.DataFrame:
    """
    Extract supervoxel features using shared union-mask binning.

    Mirrors PyRadiomics ``_calculateVoxels`` batching: group labels into batches,
    but compute each supervoxel ROI independently via ``_initCalculation``.

    Args:
        image: Input SimpleITK image.
        supervoxel_map: Multi-label supervoxel map aligned with image.
        labels: 1D supervoxel label ids to extract.
        resolved_features: Enabled feature names per class.
        calculators: Pre-initialized shared calculators (union mask binning).
        image_name: Optional feature column suffix.
        batch_size: Labels per batch group.
        progress_callback: Optional callback invoked once per processed label.

    Returns:
        pd.DataFrame: One row per supervoxel.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")

    _, label_array = _build_union_mask(supervoxel_map, image)
    rows: List[Dict[str, object]] = []

    for start in range(0, len(labels), batch_size):
        batch_labels = labels[start:start + batch_size].astype(np.int64, copy=False)
        for label in batch_labels.tolist():
            row: Dict[str, object] = {"SupervoxelID": int(label)}
            try:
                feature_row = _extract_supervoxel_label_features(
                    calculators,
                    resolved_features,
                    label_array=label_array,
                    label=int(label),
                    image_name=image_name,
                )
                row.update(feature_row)
            except Exception as exc:
                logger.warning(
                    "Failed batched supervoxel extraction for label %s: %s",
                    label,
                    exc,
                )
            rows.append(row)
            if progress_callback is not None:
                progress_callback(1)

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
    mask; per-label matrices use ``cMatrices`` inside ``_initCalculation``.

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

    union_mask, _ = _build_union_mask(supervoxel_map, image)
    torch_settings = dict(settings)
    torch_settings["device"] = device
    torch_settings["dtype"] = dtype

    use_torch_setting = str(settings.get("useTorchRadiomics", "auto"))
    if str(device).startswith("cuda"):
        logger.info(
            "batched supervoxel_radiomics using TorchRadiomics GPU: "
            "useTorchRadiomics=%s device=%s dtype=%s labels=%d batch_size=%d "
            "union_bin=True",
            use_torch_setting,
            device,
            dtype_name,
            len(labels),
            batch_size,
        )
    else:
        logger.info(
            "batched supervoxel_radiomics using TorchRadiomics CPU: "
            "useTorchRadiomics=%s device=%s labels=%d batch_size=%d union_bin=True",
            use_torch_setting,
            device,
            len(labels),
            batch_size,
        )

    calculators = _create_torch_calculators(
        image,
        union_mask,
        sorted(resolved.keys()),
        torch_settings,
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

    union_mask, _ = _build_union_mask(supervoxel_map, image)
    calculators = _create_pyradiomics_calculators(
        image,
        union_mask,
        sorted(resolved.keys()),
        settings,
    )

    logger.info(
        "supervoxel_radiomics using CPU PyRadiomics union_bin=True labels=%d batch_size=%d",
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
