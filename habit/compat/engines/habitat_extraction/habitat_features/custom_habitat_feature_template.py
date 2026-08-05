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
"""Template for a user-defined post-segmentation habitat feature.

Copy this file into an importable module, rename ``custom_foreground_volume``,
and replace ``CustomForegroundVolumeFeature`` with the desired feature
implementation. Import the module before creating ``HabitatMapAnalyzer`` so
the registration decorator is evaluated.

Current YAML usage::

    raw_img_folder: /path/to/raw_images
    habitats_map_folder: /path/to/habitat_maps
    out_dir: /path/to/feature_output
    feature_types:
      - custom_foreground_volume

The current YAML loader validates only HABIT's shared feature-extraction
fields. Therefore, keep feature-specific settings in ``plugin_configs`` until
generic YAML plugin-config sections are supported. The following code loads
the YAML file, imports the custom feature, and supplies those settings::

    import my_project.custom_habitat_feature  # Executes the registration decorator.

    from habit import load_feature_extraction_config, run_feature_extraction

    config, _ = load_feature_extraction_config("feature_extraction.yaml")
    run_feature_extraction(
        config,
        plugin_configs={"custom_foreground_volume": {"unit": "mm3"}},
    )

``plugin_configs`` is optional. When supplied, HABIT passes the matching
configuration object to the feature class constructor as ``config``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

from habit.compat.engines.habitat_extraction.feature_registry import (
    BaseHabitatFeature,
    BatchExportContext,
    HabitatFeatureFactory,
    SubjectExtractionContext,
)


@HabitatFeatureFactory.register("custom_foreground_volume")
class CustomForegroundVolumeFeature(BaseHabitatFeature):
    """Example feature that measures the non-background habitat-map volume.

    The implementation intentionally uses a simple, clinically interpretable
    quantity. Replace the extraction logic while retaining the factory
    registration, the per-subject result key, and the batch export contract.
    """

    subject_data_key = "custom_foreground_volume_features"
    output_csv_name = "custom_foreground_volume.csv"
    progress_desc = "Custom Foreground Volume"

    def __init__(self, config: Optional[Any] = None) -> None:
        """Store optional user configuration for this feature implementation.

        Args:
            config: Optional feature-specific configuration supplied through
                ``HabitatMapAnalyzer(plugin_configs=...)``.
        """
        super().__init__(config)

    @classmethod
    def feature_name(cls) -> str:
        """Return the factory name used in ``feature_types``."""
        return "custom_foreground_volume"

    def extract_subject(self, ctx: SubjectExtractionContext) -> Dict[str, Any]:
        """Calculate foreground voxel count and physical volume for one subject.

        Args:
            ctx: Subject paths and metadata. This example reads
                ``ctx.habitat_path`` only.

        Returns:
            Dictionary containing foreground voxel count and volume in mm³.
        """
        habitat_image: sitk.Image = sitk.ReadImage(ctx.habitat_path)
        habitat_array: np.ndarray = sitk.GetArrayViewFromImage(habitat_image)

        # Habitat label 0 is the background by HABIT convention. Every positive
        # label contributes to the total foreground volume regardless of label ID.
        foreground_voxel_count: int = int(np.count_nonzero(habitat_array > 0))
        spacing_mm: tuple[float, ...] = tuple(
            float(spacing) for spacing in habitat_image.GetSpacing()
        )
        voxel_volume_mm3: float = float(np.prod(spacing_mm))

        return {
            "foreground_voxel_count": foreground_voxel_count,
            "foreground_volume_mm3": foreground_voxel_count * voxel_volume_mm3,
        }

    def export_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        ctx: BatchExportContext,
    ) -> pd.DataFrame:
        """Collect the per-subject values and write one CSV file.

        Args:
            data: Mapping from subject ID to all extracted feature payloads.
            ctx: Batch output directory and logger provided by the analyzer.

        Returns:
            One-row-per-subject table written to ``output_csv_name``.
        """
        rows: Dict[str, Dict[str, Any]] = {}
        for subject_id, subject_data in data.items():
            feature_values: Dict[str, Any] = subject_data.get(
                self.subject_data_key, {}
            )
            if "error" in feature_values:
                ctx.logger.error(
                    "Custom foreground volume extraction failed for subject %s: %s",
                    subject_id,
                    feature_values["error"],
                )
            rows[subject_id] = feature_values

        result: pd.DataFrame = pd.DataFrame.from_dict(rows, orient="index")
        output_path: str = os.path.join(ctx.out_dir, self.output_csv_name)
        result.to_csv(output_path, index=True)
        ctx.logger.info("Custom foreground volume features saved to %s", output_path)
        return result
