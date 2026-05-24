"""
Supervoxel-level radiomics feature extractor
"""

import os
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Union, List, Dict, Optional
from habit.utils.log_utils import get_module_logger, radiomics_feature_class_logging
from habit.utils.progress_utils import CustomTqdm
from habit.utils.radiomics_params_utils import create_radiomics_feature_extractor
from habit.utils.torch_radiomics_utils import (
    DEFAULT_TORCH_DTYPE,
    injected_torch_radiomics,
    resolve_torch_dtype,
    resolve_voxel_radiomics_backend,
)
from .base_extractor import BaseClusteringExtractor, register_feature_extractor

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
                useTorchRadiomics: ``auto`` (default), ``true``, or ``false``.
                torchDevice: Single torch device when ``torchGpus`` is not set.
                torchGpus: Allowed GPU indices, e.g. ``[0, 1]`` or ``"0,1"``.
                torchGpuCount: Use at most this many GPUs from the front of ``torchGpus``.
                gpuSlotIndex: Optional explicit GPU slot index for parallel workers.
                torchDtype: Torch dtype name for the torch backend (``float32`` or ``float64``).
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

        backend, torch_device = resolve_voxel_radiomics_backend(
            use_torch_radiomics=kwargs.get('useTorchRadiomics', 'auto'),
            torch_device=kwargs.get('torchDevice', 'auto'),
            torch_gpus=kwargs.get('torchGpus'),
            torch_gpu_count=kwargs.get('torchGpuCount'),
            subject=kwargs.get('subject'),
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
                    "subject=%s image=%s device=%s torchGpus=%s torchGpuCount=%s dtype=%s",
                    subject_id,
                    img_name,
                    torch_device,
                    kwargs.get("torchGpus"),
                    kwargs.get("torchGpuCount"),
                    kwargs.get("torchDtype", DEFAULT_TORCH_DTYPE),
                )
            else:
                logger.info(
                    "supervoxel_radiomics extraction using TorchRadiomics CPU: "
                    "subject=%s image=%s device=%s",
                    subject_id,
                    img_name,
                    torch_device,
                )
        else:
            logger.info(
                "supervoxel_radiomics extraction using CPU PyRadiomics: "
                "subject=%s image=%s",
                subject_id,
                img_name,
            )
        extractor.settings.update(settings_update)

        sv_array: np.ndarray = sitk.GetArrayFromImage(sv_map)
        sv_labels: np.ndarray = np.unique(sv_array)
        sv_labels = sv_labels[sv_labels > 0]

        if len(sv_labels) == 0:
            raise ValueError("Supervoxel map has no non-zero values, cannot extract features")

        n_supervoxels: int = len(sv_labels)
        enabled_feature_classes = _enabled_supervoxel_feature_classes(extractor.enabledFeatures)
        progress_desc = f"supervoxel_radiomics {subject_id}"
        if img_name:
            progress_desc = f"{progress_desc} {img_name}"

        logger.info(
            "supervoxel_radiomics start: subject=%s image=%s supervoxels=%d "
            "label_range=[%d..%d] backend=%s device=%s feature_classes=%s",
            subject_id,
            img_name,
            n_supervoxels,
            int(sv_labels.min()),
            int(sv_labels.max()),
            backend,
            torch_device if backend == "torch" else "cpu_pyradiomics",
            enabled_feature_classes,
        )

        feature_data: List[Dict[str, object]] = []
        self.feature_names = []
        n_succeeded: int = 0
        n_failed: int = 0
        extraction_started_at = time.monotonic()

        with injected_torch_radiomics(enabled=(backend == "torch")):
            with radiomics_feature_class_logging():
                with CustomTqdm(total=n_supervoxels, desc=progress_desc) as progress_bar:
                    for sv_idx, sv_label in enumerate(sv_labels):
                        sv_voxel_count: int = int(np.sum(sv_array == sv_label))
                        if _should_log_supervoxel_progress(sv_idx, n_supervoxels):
                            logger.info(
                                "supervoxel_radiomics progress: subject=%s image=%s "
                                "step=%d/%d label=%d voxels=%d",
                                subject_id,
                                img_name,
                                sv_idx + 1,
                                n_supervoxels,
                                int(sv_label),
                                sv_voxel_count,
                            )

                        try:
                            label_started_at = time.monotonic()
                            features = extractor.execute(image, sv_map, label=int(sv_label))
                            label_elapsed_sec = time.monotonic() - label_started_at
                            feature_row: Dict[str, object] = {"SupervoxelID": int(sv_label)}

                            for feature_name, feature_value in features.items():
                                if feature_name.startswith('diagnostics_'):
                                    continue

                                new_feature_name = (
                                    f"{feature_name}-{img_name}" if img_name else feature_name
                                )

                                if sv_idx == 0 and new_feature_name not in self.feature_names:
                                    self.feature_names.append(new_feature_name)

                                if hasattr(feature_value, 'item'):
                                    feature_value = feature_value.item()
                                elif isinstance(feature_value, np.ndarray):
                                    feature_value = (
                                        float(feature_value.flat[0])
                                        if feature_value.size > 0
                                        else np.nan
                                    )

                                feature_row[new_feature_name] = feature_value

                            feature_data.append(feature_row)
                            n_succeeded += 1

                            if _should_log_supervoxel_progress(sv_idx, n_supervoxels):
                                logger.info(
                                    "supervoxel_radiomics label finished: subject=%s image=%s "
                                    "label=%d elapsed_sec=%.2f features=%d",
                                    subject_id,
                                    img_name,
                                    int(sv_label),
                                    label_elapsed_sec,
                                    len(self.feature_names),
                                )

                        except Exception as exc:
                            n_failed += 1
                            logger.warning(
                                "Failed to extract supervoxel radiomics for label %s: %s",
                                sv_label,
                                exc,
                            )
                            if self.feature_names:
                                feature_row = {"SupervoxelID": int(sv_label)}
                                for name in self.feature_names:
                                    feature_row[name] = np.nan
                                feature_data.append(feature_row)

                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            label=int(sv_label),
                            ok=n_succeeded,
                            fail=n_failed,
                            refresh=False,
                        )

        del extractor

        feature_df = pd.DataFrame(feature_data)

        for col in feature_df.columns:
            if col != 'SupervoxelID':
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

        if kwargs.get("output_float32", True):
            numeric_cols = [col for col in feature_df.columns if col != "SupervoxelID"]
            if numeric_cols:
                feature_df[numeric_cols] = feature_df[numeric_cols].astype(np.float32)

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
