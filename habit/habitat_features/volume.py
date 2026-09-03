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
"""Habitat volume burden features."""

from __future__ import annotations

from typing import Dict

import numpy as np

from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.habitat_features._base import single_subject_table
from habit.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.kernels.habitat_metrics import habitat_volume_fractions
from habit.spec.specs import Spec

__all__ = ["HabitatVolumeFeatures"]
@HabitatFeatureExtractorRegistry.register("volume")
class HabitatVolumeFeatures:
    """
    Voxel counts and volume fractions of every habitat for one subject.

    The fraction of the ROI occupied by each habitat (the habitat burden)
    is the most widely used habitat-level descriptor in the literature.
    Counts are reported in voxels so the features stay spacing-independent
    and comparable with the v0.1 CSV exports.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name="volume", params={})

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the volume feature family for one subject.

        Args:
            subject: Owning subject (labels suffice; intensities unused).
            habitat_map: Habitat labels for that subject.

        Returns:
            One-row table of per-habitat voxel counts and fractions.
        """
        labels = np.asarray(habitat_map.label_array)
        fractions = habitat_volume_fractions(labels, habitat_map.habitat_ids)
        features: Dict[str, float] = {}
        for habitat_id in habitat_map.habitat_ids:
            count = int(np.count_nonzero(labels == habitat_id))
            features[f"habitat_{habitat_id}_voxel_count"] = float(count)
            features[f"habitat_{habitat_id}_volume_fraction"] = fractions[int(habitat_id)]
        return single_subject_table(
            subject_id=subject.subject_id,
            features=features,
            habitat_map=habitat_map,
            spec=self.spec,
        )

