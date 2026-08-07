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
Basic Habitat Features Extraction (non_radiomics).

1. Number of disconnected regions for each habitat
2. Volume percentage for each habitat

Uses L0 kernels :func:`habitat_region_stats` and
:func:`habitat_volume_fractions` instead of per-label SimpleITK
``BinaryThreshold`` + ``ConnectedComponent`` loops. The nested return
shape is unchanged so ``NonRadiomicsFeature.export_batch`` keeps working.
"""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import SimpleITK as sitk

from habit.kernels.habitat_metrics import (
    habitat_region_stats,
    habitat_volume_fractions,
)
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)


class BasicFeatureExtractor:
    """Extractor class for basic (non-radiomics) habitat features."""

    @staticmethod
    def get_non_radiomics_features(
        habitat_img: Union[str, sitk.Image],
    ) -> Dict[Any, Any]:
        """
        Calculate disconnected-region counts and volume ratios per habitat.

        Args:
            habitat_img: SimpleITK image or path to habitat map file.

        Returns:
            Nested dict::

                {
                    <habitat_id: int>: {
                        "num_regions": int,
                        "volume_ratio": float,
                    },
                    ...,
                    "num_habitats": int,
                }

            On failure returns ``{"error": ..., "num_habitats": 0}``.
        """
        try:
            if isinstance(habitat_img, str):
                habitat_img = sitk.ReadImage(habitat_img)
            elif not isinstance(habitat_img, sitk.Image):
                raise ValueError(
                    "habitat_img must be a SimpleITK image or a file path."
                )

            labels = np.asarray(sitk.GetArrayFromImage(habitat_img))
            if not np.issubdtype(labels.dtype, np.integer):
                labels = np.rint(labels).astype(np.int64)

            present_ids = sorted(
                int(v) for v in np.unique(labels) if int(v) != 0
            )
            region_stats = habitat_region_stats(labels)
            volume_fractions = habitat_volume_fractions(labels, present_ids)

            results: Dict[Any, Any] = {}
            for habitat_id in present_ids:
                num_regions, _largest = region_stats.get(habitat_id, (0, 0))
                results[habitat_id] = {
                    "num_regions": int(num_regions),
                    "volume_ratio": float(volume_fractions.get(habitat_id, 0.0)),
                }
            results["num_habitats"] = len(present_ids)
            return results
        except Exception as exc:  # noqa: BLE001 — keep CLI batch resilient
            logger.error("Error calculating basic habitat features: %s", exc)
            return {"error": str(exc), "num_habitats": 0}
