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
from pydantic import BaseModel

from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.domain.habitat_features._base import single_subject_table
from habit.domain.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.kernels.habitat_metrics import habitat_region_stats, ith_score
from habit.spec.specs import Spec

__all__ = ["IthHabitatFeatures", "IthHabitatFeaturesParams"]


class IthHabitatFeaturesParams(BaseModel):
    """Constructor parameters for :class:`IthHabitatFeatures` (none)."""


@HabitatFeatureExtractorRegistry.register("ith_score")
class IthHabitatFeatures:
    """
    ITH score and per-habitat fragmentation statistics for one subject.

    The ITH score quantifies how fragmented the habitat partition is:
    ``1 - (1 / S_total) * sum_i(S_i,max / n_i)`` over connected components
    of each habitat. The score is numerically identical to the v0.1
    ``ITHFeatureExtractor``; one deliberate improvement is that per-habitat
    columns are emitted for every id the model can assign (zeros when the
    habitat is absent), so cohort tables no longer have ragged columns.

    The auxiliary summary columns are prefixed (``ith_num_habitats`` /
    ``ith_total_area``) rather than the bare v0.1 names: the
    ``non_radiomics`` family legitimately reports its own ``num_habitats``
    and feature tables must join across families without duplicate columns.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name="ith_score", params={})

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the ITH feature family for one subject.

        Args:
            subject: Owning subject (labels suffice; intensities unused).
            habitat_map: Habitat labels for that subject.

        Returns:
            One-row table with the ITH score and per-habitat statistics.
        """
        labels = np.asarray(habitat_map.label_array)
        stats = habitat_region_stats(labels)
        features: Dict[str, float] = {
            "ith_score": ith_score(labels),
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


HabitatFeatureExtractorRegistry.register_params_model("ith_score", IthHabitatFeaturesParams)
