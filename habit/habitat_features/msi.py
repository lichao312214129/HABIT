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
"""MSI (multiregional spatial interaction) habitat features."""

from __future__ import annotations

import numpy as np

from habit.exceptions import HABITAPIError
from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.habitat_features._base import single_subject_table
from habit.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.kernels.habitat_metrics import (
    msi_features_from_matrix,
    spatial_interaction_matrix,
)
from habit.spec.specs import Spec

__all__ = ["MsiHabitatFeatures"]
@HabitatFeatureExtractorRegistry.register("msi")
class MsiHabitatFeatures:
    """
    Multiregional spatial interaction features of one subject's habitat map.

    MSI counts face-connected neighbour pairs between habitat classes (the
    spatial interaction matrix) and derives first-order border volumes and
    normalised second-order texture statistics from it. The formulas are the
    L0 kernels :func:`~habit.kernels.habitat_metrics.spatial_interaction_matrix`
    and :func:`~habit.kernels.habitat_metrics.msi_features_from_matrix`,
    numerically identical to the established v0.1 implementation.

    The matrix size comes from the model's habitat ids (via the map's
    ``habitat_ids``), not from the labels present in this subject, so every
    subject of the same model yields the same feature columns.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name="msi", params={})

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the MSI feature family for one subject.

        Args:
            subject: Owning subject (labels suffice; intensities unused).
            habitat_map: Habitat labels for that subject.

        Returns:
            One-row table of MSI features keyed by subject id.

        Raises:
            HABITAPIError: If the map declares no habitat ids.
        """
        if not habitat_map.habitat_ids:
            raise HABITAPIError(
                f"Subject {subject.subject_id!r}: habitat map declares no "
                "habitat ids; MSI features require the model's id set."
            )
        labels = np.asarray(habitat_map.label_array)
        n_classes = max(int(v) for v in habitat_map.habitat_ids) + 1
        matrix = spatial_interaction_matrix(labels, n_classes)
        features = msi_features_from_matrix(matrix)
        return single_subject_table(
            subject_id=subject.subject_id,
            features=features,
            habitat_map=habitat_map,
            spec=self.spec,
        )

