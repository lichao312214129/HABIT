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
"""Non-radiomics (basic spatial) habitat features."""

from __future__ import annotations

from typing import Dict

import numpy as np
from pydantic import BaseModel

from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.domain.habitat_features._base import single_subject_table
from habit.domain.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.kernels.habitat_metrics import habitat_region_stats, habitat_volume_fractions
from habit.spec.specs import Spec

__all__ = ["NonRadiomicsHabitatFeatures", "NonRadiomicsHabitatFeaturesParams"]


class NonRadiomicsHabitatFeaturesParams(BaseModel):
    """Constructor parameters for :class:`NonRadiomicsHabitatFeatures` (none)."""


@HabitatFeatureExtractorRegistry.register("non_radiomics")
class NonRadiomicsHabitatFeatures:
    """
    Basic spatial features of one subject's habitat map.

    Per habitat: the number of disconnected (face-connected) regions and the
    habitat's volume fraction of the whole ROI, computed by the L0 kernels
    :func:`~habit.kernels.habitat_metrics.habitat_region_stats` and
    :func:`~habit.kernels.habitat_metrics.habitat_volume_fractions`. These
    are numerically identical to the v0.1 ``BasicFeatureExtractor`` (whose
    SimpleITK ``ConnectedComponent`` ran with ``SetFullyConnected(False)``,
    i.e. the same face connectivity).

    Columns keep the v0.1 CSV scheme (``num_habitats``, ``{id}_num_regions``,
    ``{id}_volume_ratio``) and, like every v1 family, are emitted for every
    id the model can assign -- zeros when the habitat is absent from this
    subject -- so cohort tables never have ragged columns.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name="non_radiomics", params={})

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the non-radiomics feature family for one subject.

        Args:
            subject: Owning subject (labels suffice; intensities unused).
            habitat_map: Habitat labels for that subject.

        Returns:
            One-row table with region counts and volume ratios per habitat.
        """
        labels = np.asarray(habitat_map.label_array)
        region_stats = habitat_region_stats(labels)
        volume_fractions = habitat_volume_fractions(labels, habitat_map.habitat_ids)
        features: Dict[str, float] = {"num_habitats": float(len(region_stats))}
        for habitat_id in habitat_map.habitat_ids:
            num_regions, _ = region_stats.get(int(habitat_id), (0, 0))
            features[f"{habitat_id}_num_regions"] = float(num_regions)
            features[f"{habitat_id}_volume_ratio"] = float(volume_fractions[int(habitat_id)])
        return single_subject_table(
            subject_id=subject.subject_id,
            features=features,
            habitat_map=habitat_map,
            spec=self.spec,
        )


HabitatFeatureExtractorRegistry.register_params_model(
    "non_radiomics", NonRadiomicsHabitatFeaturesParams
)
