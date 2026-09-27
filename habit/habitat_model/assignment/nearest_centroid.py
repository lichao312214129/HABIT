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
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from habit.exceptions import CompatibilityError, HABITAPIError
from habit.contracts.habitat import HabitatMap, HabitatModel, Supervoxelization
from habit.habitat_model._predict import (
    assignment_rule,
    predict_labels,
    predict_proba_frame,
)
from habit.habitat_model.assignment.registry import HabitatAssignerRegistry
from habit.spec.specs import Spec

__all__ = ["NearestCentroidAssigner", "NearestCentroidAssignerParams"]




@HabitatAssignerRegistry.register("nearest_centroid")
class NearestCentroidAssigner:
    """
    Assign habitat labels with the decision rule stored on the fitted model.

    The fitted model is bound at construction time -- the ordinary way to
    obtain this assigner is ``model.assigner()``. Prediction then has no way
    to re-learn anything, because everything it needs is inside the model.

    The rule is ``estimator_state["rule"]``:

    * ``"kmeans"`` calls ``sklearn.cluster.KMeans.predict`` on the stored
      centres. There is no ``predict_proba``.
    * ``"gmm"`` rebuilds ``sklearn.mixture.GaussianMixture`` from the stored
      weights, means and covariances and calls ``predict`` and
      ``predict_proba``.
    * ``"nearest_centroid"`` labels each row by distance to the stored class
      centres. ``metric`` is ``euclidean`` (mean), ``manhattan`` (median),
      or ``cosine`` (mean of L2-normalised rows). There is no
      ``predict_proba``.

    A model saved before those parameters existed and whose fitter is not
    GMM is treated as ``"kmeans"`` (Euclidean ``KMeans.predict`` on the
    stored centres). A GMM archive without weights and covariances raises,
    because ``GaussianMixture.predict`` cannot be rebuilt.

    Habitat ids are the estimator's 0-based labels plus one, so label ``0``
    is reserved for background. The same id is painted onto every voxel of
    a supervoxel via ``label_array``.

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
            params={
                "model_id": self._model.model_id,
                "rule": assignment_rule(self._model),
            },
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

        # Library predict (or class-centre distance) returns 0-based labels.
        # Add one so 0 stays available for background.
        try:
            assignments = predict_labels(self._model, matrix) + 1
        except HABITAPIError:
            raise
        except Exception as exc:
            raise HABITAPIError(
                f"Subject {supervoxel_map.subject_id!r}: habitat assignment "
                f"failed under rule {assignment_rule(self._model)!r}."
            ) from exc

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

    def predict_proba(self, supervoxel_map: Supervoxelization) -> pd.DataFrame:
        """
        Return per-supervoxel habitat probabilities.

        Only a Gaussian mixture has this. The frame has one row per
        supervoxel, indexed by supervoxel id, and one column
        ``habitat_{id}`` per component in the mixture's own order. The hard
        label on the map is the column with the largest probability, plus
        the same background convention as :meth:`__call__`.

        Args:
            supervoxel_map: Supervoxelization of the subject to score.

        Returns:
            Probability frame aligned with ``supervoxel_map.features``.

        Raises:
            CompatibilityError: If a required feature column is missing.
            HABITAPIError: If the model's rule has no ``predict_proba``.
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
        matrix = frame[list(self._model.feature_names)].to_numpy(dtype=np.float64)
        return predict_proba_frame(self._model, matrix, frame.index)


