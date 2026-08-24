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
    execute_voxel_based_with_class_progress,
    injected_torch_radiomics,
    release_cuda_cache,
    resolve_torch_dtype,
    resolve_voxel_radiomics_backend,
)
from habit.kernels.radiomics.voxel_maps import (
    crop_to_roi_bounding_box,
    enabled_voxel_feature_classes,
    group_voxel_feature_keys_by_class,
    voxel_feature_frame,
)
from .base_extractor import BaseClusteringExtractor, FeatureExtractorRegistry
from .method_param_spec import MethodParamSpec
from habit.utils.radiomics_preset_utils import resolve_params_file

logger = get_module_logger(__name__)

# Habit default batch size; balances memory use on typical 8-16 GB machines vs speed.
# PyRadiomics accepts -1 for no batching (all ROI voxels at once).
DEFAULT_VOXEL_BATCH = 1000

# CT habitat voxel texture default (R3B12): 7×7×7 neighborhood at radius 3.
# Prior O, et al., Radiol Artif Intell 2024;6(2):e230118.
DEFAULT_KERNEL_RADIUS = 3


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
    grouped = group_voxel_feature_keys_by_class(feature_keys, feature_classes)
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
                'geometryTolerance': 1e-3,  # Allow small geometric differences
                # GPU texture-matrix building (gpumatrices); auto follows the
                # torch device. Counts are bit-identical to the C extension.
                'use_gpu_matrices': kwargs.get('use_gpu_matrices', 'auto'),
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

            enabled_feature_classes = enabled_voxel_feature_classes(
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

            # Pre-crop to the ROI bounding box (+ kernelRadius pad) by default:
            # execute() re-applies the identical crop internally, so feature
            # values are bit-identical, while the full-volume diagnostics
            # (sitk.Hash, whole-image statistics) and mask checks run on the
            # small volume instead. Opt out with crop_to_roi=False.
            if bool(kwargs.get('crop_to_roi', True)):
                image, mask = crop_to_roi_bounding_box(
                    image,
                    mask,
                    label=mask_label,
                    pad_distance=kernel_radius,
                )

            # Extract voxel-based features; inject TorchRadiomics only when resolved.
            radiomics_log_level = resolve_radiomics_logging_level(
                bool(kwargs.get("debug", False))
            )
            with injected_torch_radiomics(enabled=(backend == "torch")):
                with radiomics_feature_class_logging(level=radiomics_log_level):
                    # Default: one execute(), no per-class tqdm. Opt in with
                    # class_progress=True (debug of a slow GLCM/GLRLM class).
                    if bool(kwargs.get("class_progress", False)):
                        result = execute_voxel_based_with_class_progress(
                            extractor, image, mask, voxel_based=True
                        )
                    else:
                        result = extractor.execute(image, mask, voxelBased=True)
                        release_cuda_cache()

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
            feature_df = voxel_feature_frame(
                result,
                mask,
                image_name=image_name,
                mask_label=mask_label,
                output_float32=bool(kwargs.get("output_float32", True)),
            )
            del result
            self.feature_names = list(feature_df.columns)

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
