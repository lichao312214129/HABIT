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
"""K-means habitat model fitter (cohort level)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from habit.api.exceptions import HABITAPIError
from habit.contracts.habitat import HabitatModel, Supervoxelization
from habit.contracts.subject import Cohort
from habit.domain.habitat_model._base import build_habitat_model, pool_supervoxel_features
from habit.domain.habitat_model.registry import HabitatModelFitterRegistry
from habit.spec.specs import Spec

__all__ = ["KMeansHabitatModelFitter", "KMeansHabitatModelFitterParams"]

_VALIDATION_METHODS = (
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "elbow",
    "kneedle",
)

#: Selection criteria scored against cluster structure rather than the
#: inertia curve; these need labels from one fit per candidate count.
_SCORE_BASED_METHODS = ("silhouette", "calinski_harabasz", "davies_bouldin")


class KMeansHabitatModelFitterParams(BaseModel):
    """Constructor parameters for :class:`KMeansHabitatModelFitter`."""

    n_habitats: Optional[int] = Field(default=None, ge=2)
    min_habitats: int = Field(default=2, ge=2)
    max_habitats: int = Field(default=10, ge=3)
    validation: str = "silhouette"
    n_init: int = Field(default=50, gt=0)


@HabitatModelFitterRegistry.register("kmeans")
class KMeansHabitatModelFitter:
    """
    Learn population habitats by k-means over pooled supervoxel features.

    This is the cohort-level step: the ONLY place where information crosses
    subject boundaries. When ``n_habitats`` is omitted, the habitat count is
    selected by a validation score over ``min_habitats..max_habitats`` --
    the same model-selection behaviour the v0.1 clustering classes exposed
    through the configuration schema, here expressed as constructor params.

    The fitter is :class:`~habit.domain.protocols.Seedable`; the seed is
    applied to k-means initialisation at fit time.

    Args:
        n_habitats: Fixed habitat count, or ``None`` to select it by
            ``validation``.
        min_habitats: Smallest candidate count during selection.
        max_habitats: Largest candidate count during selection.
        validation: Selection criterion: ``"silhouette"`` (maximise),
            ``"calinski_harabasz"`` (maximise), ``"davies_bouldin"``
            (minimise), or ``"elbow"`` / ``"kneedle"`` (knee point of the
            inertia curve, the v0.1 default criterion).
        n_init: k-means restarts per candidate count.
    """

    def __init__(
        self,
        n_habitats: Optional[int] = None,
        min_habitats: int = 2,
        max_habitats: int = 10,
        validation: str = "silhouette",
        n_init: int = 50,
    ) -> None:
        if validation not in _VALIDATION_METHODS:
            raise HABITAPIError(
                f"validation must be one of {_VALIDATION_METHODS}; got {validation!r}."
            )
        if n_habitats is None and max_habitats <= min_habitats:
            raise HABITAPIError(
                "max_habitats must be greater than min_habitats when "
                "n_habitats is selected automatically."
            )
        self.n_habitats = n_habitats
        self.min_habitats = int(min_habitats)
        self.max_habitats = int(max_habitats)
        self.validation = validation
        self.n_init = int(n_init)
        self._seed = 0

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name="kmeans",
            params={
                "n_habitats": self.n_habitats,
                "min_habitats": self.min_habitats,
                "max_habitats": self.max_habitats,
                "validation": self.validation,
                "n_init": self.n_init,
            },
        )

    def set_random_state(self, seed: int) -> None:
        """Set the seed applied to k-means initialisation at fit time."""
        self._seed = int(seed)

    def _fit_kmeans(self, matrix: np.ndarray, n_clusters: int):
        """Fit one k-means model for a candidate cluster count."""
        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=n_clusters,
            random_state=self._seed,
            n_init=self.n_init,
        )
        model.fit(matrix)
        return model

    def _candidate_range(self, n_samples: int) -> range:
        """Return the candidate habitat counts, clamped to the sample size."""
        upper = min(self.max_habitats, n_samples - 1)
        if upper < self.min_habitats:
            raise HABITAPIError(
                f"Cannot search habitats in [{self.min_habitats}, "
                f"{self.max_habitats}] with only {n_samples} samples."
            )
        return range(self.min_habitats, upper + 1)

    @staticmethod
    def _knee_point(counts: np.ndarray, inertias: np.ndarray) -> int:
        """
        Select the knee of an inertia curve by maximum chord distance.

        The curve points are min-max normalised on both axes, then the point
        farthest from the chord joining the curve's endpoints wins -- the
        classical elbow heuristic that ``kneedle``-style locators formalise.
        A degenerate (flat or two-point) curve falls back to the smallest
        candidate count.

        Args:
            counts: Candidate cluster counts, ascending.
            inertias: Inertia achieved at each candidate count.

        Returns:
            The selected cluster count.
        """
        if counts.size <= 2 or float(np.ptp(inertias)) == 0.0:
            return int(counts[0])
        x = (counts - counts.min()) / float(np.ptp(counts))
        y = (inertias - inertias.min()) / float(np.ptp(inertias))
        start = np.array([x[0], y[0]])
        end = np.array([x[-1], y[-1]])
        chord = end - start
        chord_length = float(np.hypot(*chord))
        # Perpendicular distance of each curve point from the chord.
        distances = np.abs(
            chord[0] * (start[1] - y) - (start[0] - x) * chord[1]
        ) / chord_length
        return int(counts[int(np.argmax(distances))])

    def _select_n_habitats(self, matrix: np.ndarray) -> int:
        """Score every candidate count and keep the best validated one."""
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        candidates = self._candidate_range(matrix.shape[0])
        if self.validation in ("elbow", "kneedle"):
            inertias = {
                k: float(self._fit_kmeans(matrix, k).inertia_) for k in candidates
            }
            counts = np.asarray(sorted(inertias), dtype=np.float64)
            curve = np.asarray([inertias[int(k)] for k in counts], dtype=np.float64)
            return self._knee_point(counts, curve)

        scores = {}
        for k in candidates:
            labels = self._fit_kmeans(matrix, k).labels_
            if self.validation == "silhouette":
                scores[k] = silhouette_score(matrix, labels)
            elif self.validation == "calinski_harabasz":
                scores[k] = calinski_harabasz_score(matrix, labels)
            else:
                scores[k] = davies_bouldin_score(matrix, labels)
        if self.validation == "davies_bouldin":
            return min(scores, key=scores.get)
        return max(scores, key=scores.get)

    def fit(
        self,
        units: Sequence[Supervoxelization],
        *,
        cohort: Optional[Cohort] = None,
    ) -> HabitatModel:
        """
        Learn the shared habitat definition from all subjects.

        Args:
            units: Supervoxelizations in a defined, reproducible order.
            cohort: Cohort the units came from, fingerprinted into the model.

        Returns:
            A self-contained habitat model applicable to unseen subjects.
        """
        matrix, feature_names = pool_supervoxel_features(units)
        n_habitats = self.n_habitats
        if n_habitats is None:
            n_habitats = self._select_n_habitats(matrix)
        if matrix.shape[0] < n_habitats:
            raise HABITAPIError(
                f"Cannot fit {n_habitats} habitats on {matrix.shape[0]} "
                "pooled supervoxels."
            )
        model = self._fit_kmeans(matrix, n_habitats)
        return build_habitat_model(
            fitter_name="kmeans",
            spec=self.spec,
            centroids=np.asarray(model.cluster_centers_, dtype=np.float64),
            feature_names=feature_names,
            units=units,
            cohort=cohort,
            random_seed=self._seed,
            preprocessing_state={
                "validation": self.validation,
                "inertia": float(model.inertia_),
            },
        )


HabitatModelFitterRegistry.register_params_model(
    "kmeans", KMeansHabitatModelFitterParams
)
