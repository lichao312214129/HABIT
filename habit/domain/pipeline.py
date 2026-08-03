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
"""SubjectPipeline: the subject-level chain composed into one callable."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from habit.api.exceptions import HABITAPIError
from habit.contracts.habitat import HabitatMap, Supervoxelization, VoxelFeatureField
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.domain.protocols import (
    HabitatAssigner,
    HabitatFeatureExtractor,
    Supervoxelizer,
    VoxelFeatureExtractor,
)
from habit.spec.specs import Spec

__all__ = ["SubjectPipeline"]


def _voxel_units(field: VoxelFeatureField) -> Supervoxelization:
    """
    Wrap a voxel feature field as single-voxel clustering units.

    The one-step and direct-pooling designs cluster voxels directly, with no
    supervoxel step. Representing each voxel as a one-voxel
    ``Supervoxelization`` keeps the assigner contract uniform instead of
    giving assigners a second input type to handle.

    Args:
        field: Per-voxel features for one subject.

    Returns:
        A partition in which every ROI voxel is its own unit.
    """
    n_voxels = field.values.shape[0]
    labels = np.zeros(tuple(int(v) for v in field.geometry.shape), dtype=np.int32)
    unit_ids = np.arange(1, n_voxels + 1, dtype=np.int32)
    labels[tuple(field.voxel_index.T)] = unit_ids
    features = pd.DataFrame(field.values, columns=list(field.feature_names))
    features.index = pd.Index(unit_ids, name="supervoxel")
    provenance = field.provenance.derive(
        produced_by="pipeline.voxel_units",
        spec_fingerprint="",
    )
    return Supervoxelization(
        subject_id=field.subject_id,
        label_array=labels,
        features=features,
        geometry=field.geometry,
        provenance=provenance,
    )


class SubjectPipeline:
    """
    The subject-level chain composed into a single callable.

    HABIT's answer to ``monai.transforms.Compose``. A generic ``Compose``
    cannot be reused directly because HABIT's steps are heterogeneously typed
    -- ``Subject -> VoxelFeatureField -> Supervoxelization -> HabitatMap`` --
    and erasing those types would discard exactly the contracts that make
    the design checkable.

    A fitted :class:`~habit.contracts.habitat.HabitatModel` plus a
    ``SubjectPipeline`` is precisely the pair a study publishes for external
    validation: the definition, and the procedure that applies it.

    Args:
        voxel_feature_extractor: Step producing per-voxel features.
        supervoxelizer: Step producing supervoxels. ``None`` clusters voxels
            directly, which is what the one-step and direct-pooling
            designs do.
        habitat_assigner: Step assigning habitat labels, already bound to a
            fitted model.
    """

    def __init__(
        self,
        voxel_feature_extractor: VoxelFeatureExtractor,
        supervoxelizer: Optional[Supervoxelizer],
        habitat_assigner: HabitatAssigner,
    ) -> None:
        if voxel_feature_extractor is None or habitat_assigner is None:
            raise HABITAPIError(
                "SubjectPipeline requires a voxel feature extractor and a "
                "habitat assigner; only the supervoxelizer may be None."
            )
        self.voxel_feature_extractor = voxel_feature_extractor
        self.supervoxelizer = supervoxelizer
        self.habitat_assigner = habitat_assigner

    @property
    def spec(self) -> Spec:
        """Return the composed specification of every stage."""
        stage_specs: Dict[str, Any] = {
            "voxel_feature_extractor": self.voxel_feature_extractor.spec.to_dict(),
            "supervoxelizer": (
                self.supervoxelizer.spec.to_dict()
                if self.supervoxelizer is not None
                else None
            ),
            "habitat_assigner": self.habitat_assigner.spec.to_dict(),
        }
        return Spec(name="subject_pipeline", params=stage_specs)

    def __call__(self, subject: Subject) -> HabitatMap:
        """
        Run voxel features, supervoxelisation and assignment for one subject.

        Args:
            subject: The subject to label.

        Returns:
            The subject's habitat label image.
        """
        field = self.voxel_feature_extractor(subject)
        if self.supervoxelizer is None:
            units = _voxel_units(field)
        else:
            units = self.supervoxelizer(field)
        return self.habitat_assigner(units)

    def extract_features(
        self,
        subject: Subject,
        extractors: Sequence[HabitatFeatureExtractor],
    ) -> FeatureTable:
        """
        Run the pipeline and then the requested habitat feature families.

        Named ``extract_features`` (an action) rather than the bare noun
        ``features``, which would read as an attribute on a callable object.

        Args:
            subject: The subject to process.
            extractors: Habitat feature families to compute.

        Returns:
            One feature table for that subject, joined across families.

        Raises:
            HABITAPIError: If ``extractors`` is empty.
        """
        if not extractors:
            raise HABITAPIError(
                "SubjectPipeline.extract_features requires at least one "
                "habitat feature extractor."
            )
        habitat_map = self(subject)
        tables = [extractor(subject, habitat_map) for extractor in extractors]
        combined = tables[0]
        for table in tables[1:]:
            combined = combined.join(table)
        return combined
