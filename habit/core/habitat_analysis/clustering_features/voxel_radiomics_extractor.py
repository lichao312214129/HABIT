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
from .base_extractor import BaseClusteringExtractor, register_feature_extractor

logger = get_module_logger(__name__)

# Habit default batch size; balances memory use on typical 8-16 GB machines vs speed.
# PyRadiomics accepts -1 for no batching (all ROI voxels at once).
DEFAULT_VOXEL_BATCH = 1000


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


@register_feature_extractor('voxel_radiomics')
class VoxelRadiomicsExtractor(BaseClusteringExtractor):
    """
    Extract voxel-level radiomics features from image within mask region
    using PyRadiomics' voxel-based extraction, optionally accelerated via
    in-tree TorchRadiomics injection when torch/CUDA are available.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize voxel-level radiomics feature extractor
        
        Args:
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        # self.params_file是kwargs中的path
        # 用os判断,如果没有file则报错
        for key, value in kwargs.items():
            if os.path.exists(value):
                self.params_file = value
                break
        if self.params_file is None:
            raise ValueError("params_file not found in kwargs")
        
    def extract_features(self, image_data: Union[str, sitk.Image],
                         mask_data: Union[str, sitk.Image],
                         **kwargs) -> pd.DataFrame:
        """
        Extract voxel-level radiomics features from image within mask region
        
        Args:
            image_data: Path to image file or SimpleITK image object
            mask_data: Path to mask file or SimpleITK mask object
            **kwargs: Additional parameters
                subj: subject name
                img_name: Name of the image to append to feature names
                kernelRadius: Neighborhood radius in voxels for voxel-based extraction.
                voxelBatch: Number of voxels per batch during voxel-based extraction.
                    Default is 1000. Use -1 to process all ROI voxels at once (PyRadiomics
                    native default). Lower values (e.g. 512) reduce peak memory on GPU or
                    large ROIs.
                useTorchRadiomics: ``auto`` (default), ``true``, or ``false``. ``auto`` uses
                    TorchRadiomics when torch and CUDA are available, otherwise CPU PyRadiomics.
                torchDevice: Single torch device when ``torchGpus`` is not set (``auto``, ``cuda:0``, ``cpu``).
                torchGpus: Allowed GPU indices, e.g. ``[0, 1, 2]`` or ``"0,1,2"``. Overrides ``torchDevice``.
                torchGpuCount: Use at most this many GPUs from the front of ``torchGpus``.
                gpuSlotIndex: Optional explicit GPU slot index for parallel workers.
                torchDtype: Torch dtype name for the torch backend (``float32`` or ``float64``;
                    default ``float32``).
                output_float32: If True, cast the returned DataFrame to float32 to halve
                    downstream memory (may affect numerical parity vs float64).
            
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

            # kernelRadius controls the size of the local neighborhood (in voxels) 
            # used for voxel-based feature extraction. A radius of 1 means a 3×3×3 cube
            # centered on each voxel, radius of 2 means 5×5×5, etc.
            kernelRadius = kwargs.get('kernelRadius', 1)
            voxelBatch = kwargs.get('voxelBatch', DEFAULT_VOXEL_BATCH)
            backend, torch_device = resolve_voxel_radiomics_backend(
                use_torch_radiomics=kwargs.get('useTorchRadiomics', 'auto'),
                torch_device=kwargs.get('torchDevice', 'auto'),
                torch_gpus=kwargs.get('torchGpus'),
                torch_gpu_count=kwargs.get('torchGpuCount'),
                subject=kwargs.get('subject'),
                gpu_slot_index=kwargs.get('gpuSlotIndex'),
            )
            settings_update: Dict[str, Any] = {
                'kernelRadius': kernelRadius,
                'voxelBatch': voxelBatch,
                'geometryTolerance': 1e-3  # Allow small geometric differences
            }
            if backend == "torch" and torch_device is not None:
                settings_update['device'] = torch_device
                settings_update['dtype'] = resolve_torch_dtype(
                    kwargs.get('torchDtype', DEFAULT_TORCH_DTYPE)
                )
                if str(torch_device).startswith("cuda"):
                    logger.info(
                        "voxel_radiomics extraction using TorchRadiomics GPU: "
                        "subject=%s image=%s device=%s torchGpus=%s torchGpuCount=%s "
                        "kernelRadius=%s voxelBatch=%s dtype=%s",
                        kwargs.get("subject", "unknown"),
                        image_name,
                        torch_device,
                        kwargs.get("torchGpus"),
                        kwargs.get("torchGpuCount"),
                        kernelRadius,
                        voxelBatch,
                        kwargs.get("torchDtype", DEFAULT_TORCH_DTYPE),
                    )
                else:
                    logger.info(
                        "voxel_radiomics extraction using TorchRadiomics CPU: "
                        "subject=%s image=%s device=%s kernelRadius=%s voxelBatch=%s",
                        kwargs.get("subject", "unknown"),
                        image_name,
                        torch_device,
                        kernelRadius,
                        voxelBatch,
                    )
            else:
                logger.info(
                    "voxel_radiomics extraction using CPU PyRadiomics: "
                    "subject=%s image=%s kernelRadius=%s voxelBatch=%s",
                    kwargs.get("subject", "unknown"),
                    image_name,
                    kernelRadius,
                    voxelBatch,
                )
            extractor.settings.update(settings_update)

            enabled_feature_classes = _enabled_voxel_feature_classes(
                extractor.enabledFeatures
            )
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

            for key in keys:
                val = result.pop(key, None)
                if val is None:
                    continue
                if isinstance(val, sitk.Image):
                    feature_name = f"{key}-{image_name}" if image_name else key
                    feature_names.append(feature_name)
                    feature_array = sitk.GetArrayFromImage(val)
                    values = feature_array[feature_array > 0]
                    feature_matrix.append(values)
                    del val, feature_array

            del result

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
