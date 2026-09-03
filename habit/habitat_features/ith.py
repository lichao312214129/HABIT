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
"""ITH score (topological fragmentation) habitat features."""

from __future__ import annotations

from typing import Dict

import numpy as np

from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.habitat_features._base import single_subject_table
from habit.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.kernels.habitat_metrics import habitat_region_stats, ith_score
from habit.spec.specs import Spec

__all__ = ["IthHabitatFeatures"]


@HabitatFeatureExtractorRegistry.register("ith_score")
class IthHabitatFeatures:
    """
    ITH score for one subject's habitat map.

    The score quantifies how fragmented the habitat partition is:
    ``1 - (1 / S_total) * sum_i(S_i,max / n_i)`` over connected
    components of each habitat. It is numerically identical to the v0.1
    ``ITHFeatureExtractor``.

    By default the table has a single column, ``ith_score``. Pass
    ``include_auxiliary=True`` (or ``Spec("ith_score",
    {"include_auxiliary": True})``) to also write the prefixed summaries
    ``ith_num_habitats`` / ``ith_total_area`` and the per-habitat
    ``habitat_{id}_regions`` / ``_largest_area`` / ``_area_ratio``
    columns. Those extras are emitted for every id the model can assign
    (zeros when the habitat is absent) so cohort tables stay rectangular.
    The prefixes avoid colliding with ``non_radiomics`` ``num_habitats``.
    """

    def __init__(self, include_auxiliary: bool = False) -> None:
        self.include_auxiliary = bool(include_auxiliary)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name="ith_score",
            params={"include_auxiliary": self.include_auxiliary},
        )

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the ITH feature family for one subject.

        Args:
            subject: Owning subject (labels suffice; intensities unused).
            habitat_map: Habitat labels for that subject.

        Returns:
            One-row table. Default is ``ith_score`` only.
        """
        labels = np.asarray(habitat_map.label_array)
        if not self.include_auxiliary:
            features: Dict[str, float] = {"ith_score": ith_score(labels)}
            return single_subject_table(
                subject_id=subject.subject_id,
                features=features,
                habitat_map=habitat_map,
                spec=self.spec,
            )

        stats = habitat_region_stats(labels)
        features = {
            "ith_score": ith_score(labels, region_stats=stats),
            "ith_num_habitats": float(len(stats)),
            "ith_total_area": float(np.count_nonzero(labels)),
        }
        for habitat_id in habitat_map.habitat_ids:
            num_regions, largest = stats.get(int(habitat_id), (0, 0))
            features[f"habitat_{habitat_id}_regions"] = float(num_regions)
            features[f"habitat_{habitat_id}_largest_area"] = float(largest)
            features[f"habitat_{habitat_id}_area_ratio"] = (
                float(largest) / num_regions if num_regions > 0 else 0.0
            )
        return single_subject_table(
            subject_id=subject.subject_id,
            features=features,
            habitat_map=habitat_map,
            spec=self.spec,
        )

