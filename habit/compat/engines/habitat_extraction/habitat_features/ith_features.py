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
Intratumoral Heterogeneity (ITH) Score Calculation.

Delegates to the L0 kernels :func:`habit.kernels.habitat_metrics.ith_score`
and :func:`habit.kernels.habitat_metrics.habitat_region_stats` (scipy
``ndimage.label`` with face connectivity), matching the historical
SimpleITK ``ConnectedComponent`` / ``SetFullyConnected(False)`` semantics
without per-habitat SimpleITK filter overhead.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import SimpleITK as sitk

from habit.kernels.habitat_metrics import habitat_region_stats, ith_score
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)


class ITHFeatureExtractor:
    """Extractor class for Intratumoral Heterogeneity (ITH) scores."""

    def extract_ith_features(
        self, habitat_img: Union[str, sitk.Image]
    ) -> Dict[str, Union[float, int, str]]:
        """
        Calculate ITH score and per-habitat fragmentation stats.

        Args:
            habitat_img: SimpleITK image or path to a habitat map file.

        Returns:
            Dict with ``ith_score``, ``num_habitats``, ``total_area``, and
            per-habitat ``habitat_{id}_regions`` /
            ``habitat_{id}_largest_area`` / ``habitat_{id}_area_ratio``
            (compat CSV column scheme). On failure returns
            ``{"error": ..., "ith_score": 0.0}``.
        """
        try:
            if isinstance(habitat_img, str):
                habitat_img = sitk.ReadImage(habitat_img)
            elif not isinstance(habitat_img, sitk.Image):
                raise ValueError(
                    "habitat_img must be a SimpleITK image or a file path."
                )

            labels = np.asarray(sitk.GetArrayFromImage(habitat_img))
            # Cast to integer labels; float habitat maps are rare but accepted
            # by the old SimpleITK path after an explicit Cast.
            if not np.issubdtype(labels.dtype, np.integer):
                labels = np.rint(labels).astype(np.int64)

            total_area = int(np.count_nonzero(labels))
            if total_area == 0:
                return {"ith_score": 0.0, "error": "Empty habitat map"}

            stats = habitat_region_stats(labels)
            if not stats:
                return {"ith_score": 0.0, "error": "No habitats found"}

            result: Dict[str, Union[float, int, str]] = {
                "ith_score": float(ith_score(labels)),
                "num_habitats": int(len(stats)),
                "total_area": total_area,
            }
            for habitat_id, (num_regions, largest_area) in stats.items():
                result[f"habitat_{habitat_id}_regions"] = int(num_regions)
                result[f"habitat_{habitat_id}_largest_area"] = int(largest_area)
                result[f"habitat_{habitat_id}_area_ratio"] = (
                    float(largest_area) / float(num_regions)
                    if num_regions > 0
                    else 0.0
                )
            return result
        except Exception as exc:  # noqa: BLE001 — keep CLI batch resilient
            logger.error("Error calculating ITH score from habitat image: %s", exc)
            return {"error": str(exc), "ith_score": 0.0}
