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
Voxel-level radiomics feature extractor
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Union, List, Dict, Optional, Tuple, Any
from habit.utils.log_utils import (
    get_module_logger,
    radiomics_feature_class_logging,
    resolve_radiomics_logging_level,
)
from habit.utils.radiomics_params_utils import (
    create_radiomics_feature_extractor,
    configure_voxel_glcm_on_extractor,
)
from habit.utils.torch_radiomics_utils import (
    DEFAULT_TORCH_DTYPE,
    injected_torch_radiomics,
    resolve_torch_dtype,
    resolve_voxel_radiomics_backend,
)
from .base_extractor import BaseClusteringExtractor, FeatureExtractorRegistry
from .method_param_spec import MethodParamSpec
from habit.utils.radiomics_preset_utils import resolve_params_file

logger = get_module_logger(__name__)

# Habit default batch size; balances memory use on typical 8-16 GB machines vs speed.
# PyRadiomics accepts -1 for no batching (all ROI voxels at once).
DEFAULT_VOXEL_BATCH = 1000

# CT habitat voxel texture default (R3B12): 7×7×7 neighborhood at radius 3.
# Petersen et al., Radiol Artif Intell 2024;6(2):e230118.
DEFAULT_KERNEL_RADIUS = 3


def _enabled_voxel_feature_classes(enabled_features: Dict[str, Any]) -> List[str]:
    """
    Return sorted feature class names enabled for voxel extraction.

    Shape features are excluded because PyRadiomics does not compute them in
    voxel-based mode.

    Args:
        enabled_features: ``RadiomicsFeatureExtractor.enabledFeatures`` mapping.

    Returns:
        List[str]: Feature class names, e.g. ``["firstorder", "glcm"]``.
    """
    return sorted(
        feature_class
        for feature_class in enabled_features.keys()
        if not str(feature_class).startswith("shape")
    )


def _group_voxel_feature_keys_by_class(
    feature_keys: List[str],
    feature_classes: List[str],
) -> Dict[str, List[str]]:
    """
    Group PyRadiomics voxel result keys by feature class.

    Keys follow ``{imageType}_{featureClass}_{featureName}`` (see PyRadiomics
    ``computeFeatures``).

    Args:
        feature_keys: Non-diagnostic keys from ``execute(voxelBased=True)``.
        feature_classes: Enabled feature class names.

    Returns:
        Dict[str, List[str]]: Feature class name to matching result keys.
    """
    grouped: Dict[str, List[str]] = {feature_class: [] for feature_class in feature_classes}
    for key in feature_keys:
        for feature_class in feature_classes:
            if f"_{feature_class}_" in key:
                grouped[feature_class].append(key)
                break
    return grouped


def _log_voxel_feature_class_summary(
    feature_keys: List[str],
    feature_classes: List[str],
    *,
    subject: str,
    image_name: str,
) -> None:
    """
    Log how many voxel feature maps were produced per feature class.

    Args:
        feature_keys: Non-diagnostic keys from PyRadiomics voxel extraction.
        feature_classes: Enabled feature class names.
        subject: Subject identifier for log context.
        image_name: Image/modality name for log context.
    """
    grouped = _group_voxel_feature_keys_by_class(feature_keys, feature_classes)
    for feature_class in feature_classes:
        class_keys = grouped.get(feature_class, [])
        if not class_keys:
            continue
        logger.info(
            "voxel_radiomics feature class finished: subject=%s image=%s "
            "class=%s feature_maps=%d",
            subject,
            image_name,
            feature_class,
            len(class_keys),
        )


def _feature_values_in_mask(
    feature_array: np.ndarray,
    mask_array: np.ndarray,
) -> np.ndarray:
    """
    Select every voxel feature value inside the non-background mask.

    A feature value of zero or a negative value is scientifically valid for many
    radiomics maps.  Filtering on feature intensity silently dropped those values
    and yielded a different number of rows for different feature columns.  The
    segmentation mask, rather than a feature-value threshold, defines the voxel
    population for habitat clustering.

    Args:
        feature_array: One voxel feature map returned by PyRadiomics.
        mask_array: Spatially aligned segmentation labels, where zero is background.

    Returns:
        One-dimensional feature values for all nonzero mask voxels.

    Raises:
        ValueError: If the feature map and mask have different array shapes or the
            mask does not contain any foreground voxels.
    """
    if feature_array.shape != mask_array.shape:
        raise ValueError(
            "Voxel feature map shape does not match mask shape: "
            f"{feature_array.shape} != {mask_array.shape}."
        )
    roi = mask_array != 0
    if not np.any(roi):
        raise ValueError("Voxel radiomics mask does not contain any foreground voxels.")
    return feature_array[roi]


def _mask_array_for_feature_map(
    mask: sitk.Image,
    feature_map: sitk.Image,
    *,
    label: Optional[int] = None,
) -> np.ndarray:
    """
    Align a full-size segmentation mask to a PyRadiomics voxel feature map.

    PyRadiomics crops voxel-based feature maps to the physical bounding box of
    the requested ROI, with padding determined by ``kernelRadius``. Therefore,
    feature-map array dimensions are normally smaller than the source mask
    dimensions. Alignment must use the SimpleITK physical coordinate system;
    slicing only by array shape would lose the crop offset and would be wrong
    for non-unit spacing, non-zero origins, rotated directions, or radiomics
    resampling.

    Args:
        mask: Full-size segmentation mask supplied to PyRadiomics.
        feature_map: Cropped voxel feature map returned by PyRadiomics.
        label: Optional mask label selected by PyRadiomics. When omitted, every
            nonzero mask value is treated as foreground.

    Returns:
        np.ndarray: Binary mask on the exact array grid of ``feature_map``.

    Raises:
        ValueError: If dimensions differ, grids with equal sampling are not
            lattice-aligned, or the feature-map extent loses ROI voxels.
    """
    if mask.GetDimension() != feature_map.GetDimension():
        raise ValueError(
            "Voxel feature map and mask dimensions do not match: "
            f"{feature_map.GetDimension()} != {mask.GetDimension()}."
        )

    spacing_matches = np.allclose(
        mask.GetSpacing(),
        feature_map.GetSpacing(),
        rtol=0.0,
        atol=1e-6,
    )
    direction_matches = np.allclose(
        mask.GetDirection(),
        feature_map.GetDirection(),
        rtol=0.0,
        atol=1e-6,
    )
    if spacing_matches and direction_matches:
        # Cropped maps with unchanged sampling must start exactly on the source
        # mask lattice. Detecting a fractional index prevents nearest-neighbor
        # resampling from hiding an origin or direction metadata error.
        start_index = np.asarray(
            mask.TransformPhysicalPointToContinuousIndex(feature_map.GetOrigin()),
            dtype=np.float64,
        )
        if not np.allclose(start_index, np.rint(start_index), rtol=0.0, atol=1e-5):
            raise ValueError(
                "Voxel feature map is not aligned to the mask voxel lattice: "
                f"continuous_start_index={tuple(start_index)}."
            )

    aligned_mask = sitk.Resample(
        mask,
        feature_map,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID(),
    )
    aligned_array = sitk.GetArrayFromImage(aligned_mask)
    source_array = sitk.GetArrayViewFromImage(mask)

    if label is None:
        roi = aligned_array != 0
        source_roi_count = int(np.count_nonzero(source_array))
    else:
        roi = aligned_array == label
        source_roi_count = int(np.count_nonzero(source_array == label))

    # With unchanged sampling, PyRadiomics only crops the grid; it does not
    # change the ROI voxel population. A count difference means the physical
    # crop does not fully cover the requested label and must not be accepted.
    if spacing_matches and direction_matches:
        aligned_roi_count = int(np.count_nonzero(roi))
        if aligned_roi_count != source_roi_count:
            raise ValueError(
                "Voxel feature map physical extent does not contain the complete "
                f"mask ROI: aligned_voxels={aligned_roi_count}, "
                f"source_voxels={source_roi_count}, label={label}."
            )

    return roi.astype(np.uint8, copy=False)


@FeatureExtractorRegistry.register('voxel_radiomics')
class VoxelRadiomicsExtractor(BaseClusteringExtractor):
    """
    Extract voxel-level radiomics features from image within mask region
    using PyRadiomics' voxel-based extraction, optionally accelerated via
    in-tree TorchRadiomics injection when torch/CUDA are available.
    """

    # DSL contract: voxel_radiomics(<modality>, params_file, kernel_radius, ...).
    # ``params_file`` is optional and falls back to the bundled CT R3B12 voxel
    # preset; other knobs carry habit-level defaults so users may omit them.
    method_param_spec = MethodParamSpec(
        required=(),
        optional={
            "kernel_radius": DEFAULT_KERNEL_RADIUS,
            "voxel_batch": DEFAULT_VOXEL_BATCH,
            "use_torch_radiomics": "auto",
        },
        default_params_file_preset="voxel",
        takes_image=True,
    )

    def __init__(self, **kwargs):
        """
        Initialize voxel-level radiomics feature extractor.

        Resolve ``params_file`` explicitly: a user-provided path wins; when the
        value is missing (or an ``@preset:*`` reference), fall back to the
        bundled CT R3B12 voxel preset so ``params_file`` can be omitted.

        Args:
            **kwargs: Additional parameters. ``params_file`` is optional.
        """
        super().__init__(**kwargs)
        # User path (or @preset ref) wins; None/empty -> bundled CT voxel preset.
        self.params_file = resolve_params_file(
            kwargs.get('params_file'),
            preset=self.method_param_spec.default_params_file_preset or "voxel",
        )
        
    def extract_features(self, image_data: Union[str, sitk.Image],
                         mask_data: Union[str, sitk.Image],
                         **kwargs) -> pd.DataFrame:
        """
        Extract voxel-level radiomics features from image within mask region
        
        Args:
            image_data: Path to image file or SimpleITK image object
            mask_data: Path to mask file or SimpleITK mask object
            **kwargs: Optional keys — ``subj``, ``img_name``, ``kernel_radius``,
                ``voxel_batch``, ``use_torch_radiomics``, ``torch_device``, ``torch_gpus``,
                ``torch_gpu_count``, ``gpu_slot_index``, ``torch_dtype``, ``output_float32``.
                See ``habit/utils/torch_radiomics_utils.py`` for backend resolution.
            
        Returns:
            pd.DataFrame: Extracted voxel-level radiomics features
        """
        # Load image
        if isinstance(image_data, str):
            if os.path.exists(image_data):
                image = sitk.ReadImage(image_data)
            else:
                raise FileNotFoundError(f"Image file not found: {image_data}")
        else:
            image = image_data

        # Get image name
        image_name = kwargs.get('image', None)
        if image_name is None:
            image_name = os.path.basename(os.path.dirname(image_data))
            
        # Load mask
        if isinstance(mask_data, str):
            if os.path.exists(mask_data):
                mask = sitk.ReadImage(mask_data)
            else:
                raise FileNotFoundError(f"Mask file not found: {mask_data}")
        else:
            mask = mask_data

        # Ensure mask has the same geometric information as image
        # to avoid geometry mismatch errors in PyRadiomics
        mask.CopyInformation(image)

        # Check if mask has non-zero values
        mask_array = sitk.GetArrayFromImage(mask)
        if np.sum(mask_array > 0) == 0:
            raise ValueError("Mask has no non-zero values, cannot extract features")
        
        try:
            # Load params with explicit UTF-8/multi-encoding fallback (Windows GBK-safe).
            extractor = create_radiomics_feature_extractor(self.params_file)
            configure_voxel_glcm_on_extractor(extractor, logger=logger)

            # kernel_radius controls the size of the local neighborhood (in voxels) 
            # used for voxel-based feature extraction. A radius of 1 means a 3×3×3 cube
            # centered on each voxel, radius of 2 means 5×5×5, etc.
            kernel_radius = kwargs.get('kernel_radius', DEFAULT_KERNEL_RADIUS)
            voxel_batch = kwargs.get('voxel_batch', DEFAULT_VOXEL_BATCH)
            backend, torch_device = resolve_voxel_radiomics_backend(
                use_torch_radiomics=kwargs.get('use_torch_radiomics', 'auto'),
                torch_device=kwargs.get('torch_device', 'auto'),
                torch_gpus=kwargs.get('torch_gpus'),
                torch_gpu_count=kwargs.get('torch_gpu_count'),
                subject=kwargs.get('subject'),
                gpu_slot_index=kwargs.get('gpu_slot_index'),
            )
            settings_update: Dict[str, Any] = {
                'kernelRadius': kernel_radius,
                'voxelBatch': voxel_batch,
                'geometryTolerance': 1e-3  # Allow small geometric differences
            }
            if backend == "torch" and torch_device is not None:
                settings_update['device'] = torch_device
                settings_update['dtype'] = resolve_torch_dtype(
                    kwargs.get('torch_dtype', DEFAULT_TORCH_DTYPE)
                )
                if str(torch_device).startswith("cuda"):
                    logger.info(
                        "voxel_radiomics extraction using TorchRadiomics GPU: "
                        "subject=%s image=%s device=%s torch_gpus=%s torch_gpu_count=%s "
                        "kernel_radius=%s voxel_batch=%s dtype=%s",
                        kwargs.get("subject", "unknown"),
                        image_name,
                        torch_device,
                        kwargs.get("torch_gpus"),
                        kwargs.get("torch_gpu_count"),
                        kernel_radius,
                        voxel_batch,
                        kwargs.get("torch_dtype", DEFAULT_TORCH_DTYPE),
                    )
                else:
                    logger.info(
                        "voxel_radiomics extraction using TorchRadiomics CPU: "
                        "subject=%s image=%s device=%s kernel_radius=%s voxel_batch=%s",
                        kwargs.get("subject", "unknown"),
                        image_name,
                        torch_device,
                        kernel_radius,
                        voxel_batch,
                    )
            else:
                logger.info(
                    "voxel_radiomics extraction using CPU PyRadiomics: "
                    "subject=%s image=%s kernel_radius=%s voxel_batch=%s",
                    kwargs.get("subject", "unknown"),
                    image_name,
                    kernel_radius,
                    voxel_batch,
                )
            extractor.settings.update(settings_update)

            enabled_feature_classes = _enabled_voxel_feature_classes(
                extractor.enabledFeatures
            )
            configured_label = extractor.settings.get("label", 1)
            mask_label = 1 if configured_label is None else int(configured_label)
            subject_id = str(kwargs.get("subject", "unknown"))
            logger.info(
                "voxel_radiomics feature classes to extract: subject=%s image=%s classes=%s",
                subject_id,
                image_name,
                enabled_feature_classes,
            )

            # Extract voxel-based features; inject TorchRadiomics only when resolved.
            radiomics_log_level = resolve_radiomics_logging_level(
                bool(kwargs.get("debug", False))
            )
            with injected_torch_radiomics(enabled=(backend == "torch")):
                with radiomics_feature_class_logging(level=radiomics_log_level):
                    result = extractor.execute(image, mask, voxelBased=True)

            # Release extractor before materialising many per-feature arrays; peak RAM
            # inside execute() is unchanged, but we avoid holding extractor + all maps.
            del extractor

            # Pop each feature map from the result dict so we do not keep every
            # sitk.Image alive at once while building the feature matrix.
            keys = [
                k for k in result.keys()
                if not str(k).startswith('diagnostic')
            ]
            _log_voxel_feature_class_summary(
                keys,
                enabled_feature_classes,
                subject=subject_id,
                image_name=image_name,
            )
            feature_names: List[str] = []
            feature_matrix: List[np.ndarray] = []
            mask_arrays_by_geometry: Dict[Tuple[Any, ...], np.ndarray] = {}

            for key in keys:
                val = result.pop(key, None)
                if val is None:
                    continue
                if isinstance(val, sitk.Image):
                    feature_name = f"{key}-{image_name}" if image_name else key
                    feature_names.append(feature_name)
                    feature_array = sitk.GetArrayFromImage(val)
                    geometry_key: Tuple[Any, ...] = (
                        tuple(val.GetSize()),
                        tuple(val.GetSpacing()),
                        tuple(val.GetOrigin()),
                        tuple(val.GetDirection()),
                    )
                    if geometry_key not in mask_arrays_by_geometry:
                        mask_arrays_by_geometry[geometry_key] = _mask_array_for_feature_map(
                            mask,
                            val,
                            label=mask_label,
                        )
                    mask_array = mask_arrays_by_geometry[geometry_key]
                    values = _feature_values_in_mask(feature_array, mask_array)
                    feature_matrix.append(values)
                    del val, feature_array

            del result, mask_arrays_by_geometry

            voxel_counts = {values.shape[0] for values in feature_matrix}
            if len(voxel_counts) > 1:
                raise ValueError(
                    "Voxel radiomics feature maps produced inconsistent ROI row "
                    f"counts: {sorted(voxel_counts)}."
                )

            self.feature_names = feature_names
            
            # Create DataFrame with voxels as rows and features as columns
            feature_df = pd.DataFrame(feature_matrix)
            feature_df = feature_df.T
            feature_df.columns = feature_names

            if kwargs.get("output_float32", True):
                feature_df = feature_df.astype(np.float32)

            return feature_df
            
        except Exception as e:
            logger.error("Failed to extract voxel-based features: %s", str(e))
            raise
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names
