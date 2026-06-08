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
Supervoxel-level radiomics feature extractor
"""

import os
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Union, List, Dict, Optional
from habit.core.habitat_analysis.clustering_features.supervoxel_cext import (
    is_cext_available,
    supervoxel_cext_matrix_backend_label,
)
from habit.utils.log_utils import (
    get_module_logger,
    radiomics_feature_class_logging,
    resolve_radiomics_logging_level,
)
from habit.utils.radiomics_params_utils import create_radiomics_feature_extractor
from habit.utils.torch_radiomics_utils import (
    DEFAULT_TORCH_DTYPE,
    injected_torch_radiomics,
    resolve_torch_dtype,
    resolve_voxel_radiomics_backend,
)
from .batched_supervoxel_radiomics import (
    DEFAULT_SUPERVOXEL_BATCH,
    extract_batched_supervoxel_features,
    extract_supervoxel_features_pyradiomics,
)
from .base_extractor import BaseClusteringExtractor, register_feature_extractor
from .supervoxel_radiomics_settings import merge_supervoxel_settings

logger = get_module_logger(__name__)

def _enabled_supervoxel_feature_classes(enabled_features: Dict[str, object]) -> List[str]:
    """
    Return sorted PyRadiomics feature class names configured for extraction.

    Args:
        enabled_features: ``RadiomicsFeatureExtractor.enabledFeatures`` mapping.

    Returns:
        List[str]: Feature class names, e.g. ``["firstorder", "glcm"]``.
    """
    return sorted(str(feature_class) for feature_class in enabled_features.keys())


def _group_supervoxel_feature_names_by_class(
    feature_names: List[str],
    feature_classes: List[str],
) -> Dict[str, List[str]]:
    """
    Group supervoxel result column names by PyRadiomics feature class.

    Column names follow ``original_{class}_{name}`` or
    ``original_{class}_{name}-{image}``.

    Args:
        feature_names: Feature column names collected from the first supervoxel.
        feature_classes: Enabled feature class names.

    Returns:
        Dict[str, List[str]]: Feature class name to matching column names.
    """
    grouped: Dict[str, List[str]] = {feature_class: [] for feature_class in feature_classes}
    for name in feature_names:
        for feature_class in feature_classes:
            if f"_{feature_class}_" in name:
                grouped[feature_class].append(name)
                break
    return grouped


def _log_supervoxel_feature_class_summary(
    feature_names: List[str],
    feature_classes: List[str],
    *,
    subject: str,
    image_name: str,
    backend: str,
    matrix_backend: str,
) -> None:
    """
    Log how many supervoxel feature columns were produced per feature class.

    Args:
        feature_names: Feature column names from extraction.
        feature_classes: Enabled feature class names.
        subject: Subject identifier for log context.
        image_name: Image/modality name for log context.
        backend: Resolved backend name (``torch`` or ``pyradiomics``).
        matrix_backend: Texture matrix backend label (``habit_native_c``, etc.).
    """
    grouped = _group_supervoxel_feature_names_by_class(feature_names, feature_classes)
    for feature_class in feature_classes:
        class_names = grouped.get(feature_class, [])
        if not class_names:
            continue
        logger.info(
            "supervoxel_radiomics feature class finished: subject=%s image=%s "
            "backend=%s matrix_backend=%s class=%s features=%d",
            subject,
            image_name,
            backend,
            matrix_backend,
            feature_class,
            len(class_names),
        )


def _should_log_supervoxel_progress(current_index: int, total: int) -> bool:
    """
    Decide whether to emit a per-supervoxel INFO log for the current step.

    Logs every supervoxel when the count is small; for large maps, logs the
    first/last item and roughly every 5% in between to keep log files readable.

    Args:
        current_index: Zero-based index of the supervoxel currently processed.
        total: Total number of supervoxels to extract.

    Returns:
        bool: True when an INFO progress line should be written.
    """
    if total <= 0:
        return False
    if total <= 100:
        return True
    if current_index == 0 or current_index == total - 1:
        return True
    step = max(1, total // 20)
    return (current_index + 1) % step == 0


@register_feature_extractor('supervoxel_radiomics')
class SupervoxelRadiomicsExtractor(BaseClusteringExtractor):
    """
    Extract radiomics features for each supervoxel in the supervoxel map.

    Optionally accelerated via in-tree TorchRadiomics injection when torch/CUDA
    are available, using the same backend resolution as voxel_radiomics.
    """

    def __init__(self, params_file: str = None, **kwargs):
        """
        Initialize supervoxel radiomics feature extractor.

        Args:
            params_file: Path to PyRadiomics parameter file or YAML string containing parameters
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.params_file = params_file
        self.feature_names: List[str] = []

    def extract_features(
        self,
        image_data: Union[str, sitk.Image],
        supervoxel_map: Union[str, sitk.Image],
        config_file: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Extract radiomics features for each supervoxel in the supervoxel map.

        Args:
            image_data: Path to image file or SimpleITK image object
            supervoxel_map: Path to supervoxel map file or SimpleITK image object
            config_file: Path to PyRadiomics parameter file (overrides the one in constructor)
            **kwargs: Additional parameters
                subject: Subject identifier for logging and GPU assignment.
                image: Image/modality name appended to feature names.
                useTorchRadiomics: ``auto`` (default), ``true``, or ``false``. ``auto`` uses
                    TorchRadiomics when torch and CUDA are available, otherwise CPU PyRadiomics.
                torchDevice: Single torch device when ``torchGpus`` is not set.
                torchGpus: Allowed GPU indices, e.g. ``[0, 1]`` or ``"0,1"``.
                torchGpuCount: Use at most this many GPUs from the front of ``torchGpus``.
                gpuSlotIndex: Optional explicit GPU slot index for parallel workers.
                torchDtype: Torch dtype name for the torch backend (``float32`` or ``float64``).
                supervoxelBatch: Supervoxels per batch group (union-mask binning path).
                supervoxelUnionBboxCrop: Crop to union supervoxel bbox before extraction.
                useSupervoxelCext: ``auto`` (default), ``true``, or ``false``. ``auto`` uses
                    the habit C-extension batched matrix path when compiled; otherwise the
                    prior Torch/PyRadiomics stacked-matrix path.
                supervoxelPadDistance: Optional bbox pad override (voxels); else padDistance.
                output_float32: If True, cast numeric feature columns to float32.

        Returns:
            pd.DataFrame: DataFrame with radiomics features for each supervoxel
        """
        params_file = config_file or self.params_file

        img_name: str = kwargs.get('image', '')
        if not img_name and isinstance(image_data, str):
            img_name = os.path.basename(os.path.dirname(image_data))

        subject_id: str = str(kwargs.get("subject", "unknown"))

        if isinstance(image_data, str):
            if os.path.exists(image_data):
                image = sitk.ReadImage(image_data)
            else:
                raise FileNotFoundError(f"Image file not found: {image_data}")
        else:
            image = image_data

        if isinstance(supervoxel_map, str):
            if os.path.exists(supervoxel_map):
                sv_map = sitk.ReadImage(supervoxel_map)
            else:
                raise FileNotFoundError(f"Supervoxel map file not found: {supervoxel_map}")
        else:
            sv_map = supervoxel_map

        # Align supervoxel map geometry with the source image once; PyRadiomics
        # reads each label directly from this multi-label mask (no per-label binary mask).
        sv_map.CopyInformation(image)

        try:
            extractor = create_radiomics_feature_extractor(params_file)
        except Exception as exc:
            raise ValueError(
                f"Failed to initialize radiomics extractor with {params_file}: {exc}"
            ) from exc

        use_torch_setting: str = str(kwargs.get("useTorchRadiomics", "auto"))
        backend, torch_device = resolve_voxel_radiomics_backend(
            use_torch_radiomics=use_torch_setting,
            torch_device=kwargs.get('torchDevice', 'auto'),
            torch_gpus=kwargs.get('torchGpus'),
            torch_gpu_count=kwargs.get('torchGpuCount'),
            subject=kwargs.get('subject', subject_id),
            gpu_slot_index=kwargs.get('gpuSlotIndex'),
        )
        settings_update: Dict[str, object] = {
            'geometryTolerance': 1e-3,
        }
        if backend == "torch" and torch_device is not None:
            settings_update['device'] = torch_device
            settings_update['dtype'] = resolve_torch_dtype(
                kwargs.get('torchDtype', DEFAULT_TORCH_DTYPE)
            )
            if str(torch_device).startswith("cuda"):
                logger.info(
                    "supervoxel_radiomics extraction using TorchRadiomics GPU: "
                    "subject=%s image=%s useTorchRadiomics=%s device=%s "
                    "torchGpus=%s torchGpuCount=%s dtype=%s",
                    subject_id,
                    img_name,
                    use_torch_setting,
                    torch_device,
                    kwargs.get("torchGpus"),
                    kwargs.get("torchGpuCount"),
                    kwargs.get("torchDtype", DEFAULT_TORCH_DTYPE),
                )
            else:
                logger.info(
                    "supervoxel_radiomics extraction using TorchRadiomics CPU: "
                    "subject=%s image=%s useTorchRadiomics=%s device=%s",
                    subject_id,
                    img_name,
                    use_torch_setting,
                    torch_device,
                )
        else:
            logger.info(
                "supervoxel_radiomics extraction using CPU PyRadiomics: "
                "subject=%s image=%s useTorchRadiomics=%s",
                subject_id,
                img_name,
                use_torch_setting,
            )
        extractor.settings.update(settings_update)

        sv_array: np.ndarray = sitk.GetArrayFromImage(sv_map)
        sv_labels: np.ndarray = np.unique(sv_array)
        sv_labels = sv_labels[sv_labels > 0]

        if len(sv_labels) == 0:
            raise ValueError("Supervoxel map has no non-zero values, cannot extract features")

        n_supervoxels: int = len(sv_labels)
        enabled_feature_classes = _enabled_supervoxel_feature_classes(extractor.enabledFeatures)

        logger.info(
            "supervoxel_radiomics feature classes to extract: subject=%s image=%s "
            "classes=%s backend=%s",
            subject_id,
            img_name,
            enabled_feature_classes,
            backend,
        )

        logger.info(
            "supervoxel_radiomics start: subject=%s image=%s supervoxels=%d "
            "label_range=[%d..%d] backend=%s device=%s",
            subject_id,
            img_name,
            n_supervoxels,
            int(sv_labels.min()),
            int(sv_labels.max()),
            backend,
            torch_device if backend == "torch" else "cpu_pyradiomics",
        )

        if backend == "torch":
            logger.info(
                "supervoxel_radiomics TorchRadiomics injection enabled: "
                "subject=%s image=%s device=%s",
                subject_id,
                img_name,
                torch_device,
            )

        self.feature_names = []
        n_succeeded: int = 0
        n_failed: int = 0
        extraction_started_at = time.monotonic()
        supervoxel_batch: int = int(kwargs.get("supervoxelBatch", DEFAULT_SUPERVOXEL_BATCH))
        radiomics_settings = merge_supervoxel_settings(extractor.settings, kwargs)
        matrix_backend = supervoxel_cext_matrix_backend_label(radiomics_settings)
        use_supervoxel_cext_flag = radiomics_settings.get("useSupervoxelCext", "auto")

        if matrix_backend == "habit_native_c":
            logger.info(
                "supervoxel_radiomics habit native C extension ENABLED: subject=%s image=%s "
                "useSupervoxelCext=%s module=supervoxel_cext._sv_cmatrices",
                subject_id,
                img_name,
                use_supervoxel_cext_flag,
            )
        elif matrix_backend == "habit_fallback_cmatrices":
            logger.warning(
                "supervoxel_radiomics habit C extension requested but not built: "
                "subject=%s image=%s useSupervoxelCext=%s native_available=%s. "
                "Run: pip install -e .",
                subject_id,
                img_name,
                use_supervoxel_cext_flag,
                is_cext_available(),
            )

        logger.info(
            "supervoxel_radiomics union-mask binning enabled: subject=%s image=%s "
            "supervoxelBatch=%d union_bbox_crop=%s matrix_backend=%s",
            subject_id,
            img_name,
            supervoxel_batch,
            radiomics_settings.get("supervoxelUnionBboxCrop", True),
            matrix_backend,
        )

        radiomics_log_level = resolve_radiomics_logging_level(
            bool(kwargs.get("debug", False))
        )
        with injected_torch_radiomics(enabled=(backend == "torch")):
            with radiomics_feature_class_logging(level=radiomics_log_level):
                if backend == "torch":
                    feature_df = extract_batched_supervoxel_features(
                        image,
                        sv_map,
                        sv_labels,
                        enabled_features=extractor.enabledFeatures,
                        image_name=img_name,
                        settings=radiomics_settings,
                        device=str(torch_device),
                        dtype_name=str(
                            kwargs.get("torchDtype", DEFAULT_TORCH_DTYPE)
                        ),
                        batch_size=supervoxel_batch,
                    )
                else:
                    feature_df = extract_supervoxel_features_pyradiomics(
                        image,
                        sv_map,
                        sv_labels,
                        enabled_features=extractor.enabledFeatures,
                        image_name=img_name,
                        settings=radiomics_settings,
                        batch_size=supervoxel_batch,
                    )

        del extractor

        if not feature_df.empty:
            self.feature_names = [
                col for col in feature_df.columns if col != "SupervoxelID"
            ]
            numeric_cols = [col for col in self.feature_names]
            if numeric_cols:
                valid_rows = feature_df[numeric_cols].notna().any(axis=1)
                n_succeeded = int(valid_rows.sum())
                n_failed = int(len(feature_df) - n_succeeded)
            else:
                n_succeeded = len(feature_df)
        else:
            n_succeeded = 0
            n_failed = n_supervoxels

        for col in feature_df.columns:
            if col != 'SupervoxelID':
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

        if kwargs.get("output_float32", True):
            numeric_cols = [col for col in feature_df.columns if col != "SupervoxelID"]
            if numeric_cols:
                feature_df[numeric_cols] = feature_df[numeric_cols].astype(np.float32)

        _log_supervoxel_feature_class_summary(
            self.feature_names,
            enabled_feature_classes,
            subject=subject_id,
            image_name=img_name,
            backend=backend,
            matrix_backend=matrix_backend,
        )

        logger.info(
            "supervoxel_radiomics finished: subject=%s image=%s supervoxels=%d "
            "features=%d succeeded=%d failed=%d elapsed_sec=%.2f",
            subject_id,
            img_name,
            len(feature_df),
            max(len(self.feature_names), 0),
            n_succeeded,
            n_failed,
            time.monotonic() - extraction_started_at,
        )

        return feature_df

    def get_feature_names(self) -> List[str]:
        """
        Get feature names.

        Returns:
            List[str]: List of feature names
        """
        return self.feature_names
