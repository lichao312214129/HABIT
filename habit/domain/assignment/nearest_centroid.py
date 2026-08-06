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
"""Nearest-centroid habitat assignment: the reference HabitatAssigner."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from habit.exceptions import CompatibilityError
from habit.contracts.habitat import HabitatMap, HabitatModel, Supervoxelization
from habit.domain.assignment.registry import HabitatAssignerRegistry
from habit.spec.specs import Spec

__all__ = ["NearestCentroidAssigner", "NearestCentroidAssignerParams"]


class NearestCentroidAssignerParams(BaseModel):
    """Constructor parameters for :class:`NearestCentroidAssigner`."""

    model_config = ConfigDict(extra="forbid")
    model: Any = Field(
        description="Fitted HabitatModel the assigner projects onto subjects."
    )


@HabitatAssignerRegistry.register("nearest_centroid")
class NearestCentroidAssigner:
    """
    Assign each supervoxel to the habitat of its nearest centroid.

    The fitted model is bound at construction time -- the ordinary way to
    obtain this assigner is ``model.assigner()``. Prediction then has no way
    to re-learn anything, because everything it needs is inside the model;
    this is what enforces train/predict consistency structurally.

    Habitat ids are the centroid row indices plus one, so label ``0`` is
    reserved for background, matching the v0.1 label-image convention.

    Args:
        model: The fitted habitat definition to project.
    """

    def __init__(self, model: HabitatModel) -> None:
        if not isinstance(model, HabitatModel):
            raise CompatibilityError(
                "NearestCentroidAssigner requires a fitted HabitatModel; "
                f"got {type(model).__name__}."
            )
        self._model = model

    @property
    def model(self) -> HabitatModel:
        """Return the fitted habitat definition this assigner projects."""
        return self._model

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification, bound to the model id."""
        return Spec(
            name="nearest_centroid",
            params={"model_id": self._model.model_id},
        )

    def __call__(self, supervoxel_map: Supervoxelization) -> HabitatMap:
        """
        Project the fitted habitat definition onto one subject.

        Args:
            supervoxel_map: Supervoxelization of the subject to label.

        Returns:
            The subject's habitat label image, tagged with the model's id.

        Raises:
            CompatibilityError: If the supervoxel features lack a feature
                the model requires, or a label has no feature row.
        """
        frame = supervoxel_map.features
        missing = [
            name for name in self._model.feature_names if name not in frame.columns
        ]
        if missing:
            raise CompatibilityError(
                f"Subject {supervoxel_map.subject_id!r}: supervoxel features "
                f"lack the model-required features {missing}; the model "
                f"expects {list(self._model.feature_names)}."
            )
        # Column order must match the centroid matrix exactly; extra columns
        # are ignored so a richer feature frame stays assignable.
        matrix = frame[list(self._model.feature_names)].to_numpy(dtype=np.float64)

        unit_ids = np.asarray(frame.index, dtype=np.int64)
        labels = np.asarray(supervoxel_map.label_array)
        # one_step / direct_pooling use voxel_units: every ROI voxel is its
        # own id inside a full-volume label_array (often 10^6–10^7 voxels).
        # Prefer ``bincount`` coverage over ``np.unique`` + Python set
        # difference at that scale; behaviour is identical.
        if unit_ids.size == 0:
            if np.any(labels != 0):
                raise CompatibilityError(
                    f"Subject {supervoxel_map.subject_id!r}: supervoxel "
                    "labels are present but the feature table is empty."
                )
        else:
            max_unit = int(unit_ids.max())
            max_label = int(labels.max()) if labels.size else 0
            if max_label > max_unit:
                raise CompatibilityError(
                    f"Subject {supervoxel_map.subject_id!r}: supervoxel "
                    f"labels up to {max_label} have no feature rows "
                    f"(feature index max={max_unit})."
                )
            covered = np.zeros(max_unit + 1, dtype=bool)
            covered[unit_ids] = True
            counts = np.bincount(labels.ravel(), minlength=max_unit + 1)
            present = np.flatnonzero(counts)
            present = present[present != 0]
            missing_mask = ~covered[present]
            if np.any(missing_mask):
                unknown = [int(v) for v in present[missing_mask]]
                raise CompatibilityError(
                    f"Subject {supervoxel_map.subject_id!r}: supervoxel labels "
                    f"{unknown} have no feature rows."
                )

        # Euclidean nearest-centroid assignment; ids are row index + 1 so
        # that 0 stays available for background.
        distances = np.linalg.norm(
            matrix[:, None, :] - self._model.centroids[None, :, :], axis=2
        )
        assignments = np.argmin(distances, axis=1).astype(np.int64) + 1

        lookup = np.zeros(int(unit_ids.max()) + 1, dtype=np.int32)
        lookup[unit_ids.astype(np.int64)] = assignments.astype(np.int32)
        habitat_array = lookup[labels]

        provenance = supervoxel_map.provenance.derive(
            produced_by=f"habitat_assigner.{self.spec.name}",
            spec_fingerprint=self.spec.fingerprint(),
            random_seed=self._model.provenance.random_seed,
        )
        return HabitatMap(
            subject_id=supervoxel_map.subject_id,
            label_array=habitat_array,
            geometry=supervoxel_map.geometry,
            model_id=self._model.model_id,
            habitat_ids=tuple(range(1, self._model.n_habitats + 1)),
            provenance=provenance,
        )


HabitatAssignerRegistry.register_params_model(
    "nearest_centroid", NearestCentroidAssignerParams
)
