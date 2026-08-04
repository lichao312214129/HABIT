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

from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field

from habit.exceptions import HABITAPIError
from habit.contracts.habitat import HabitatModel, Supervoxelization
from habit.contracts.subject import Cohort
from habit.domain.habitat_model._base import build_habitat_model, pool_supervoxel_features
from habit.domain.habitat_model._selection import (
    build_selection_report,
    normalize_validation,
    select_cluster_count,
)
from habit.domain.habitat_model.registry import HabitatModelFitterRegistry
from habit.kernels.cluster_selection import gap_statistic
from habit.spec.specs import Spec

__all__ = ["KMeansHabitatModelFitter", "KMeansHabitatModelFitterParams"]

#: Criteria this fitter can compute, matching the v0.1 k-means schema.
_VALIDATION_METHODS = (
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "gap",
    "inertia",
    "elbow",
    "kneedle",
)

#: Criteria read off the inertia curve; they share one number per candidate.
_INERTIA_METHODS = ("inertia", "elbow", "kneedle")


class KMeansHabitatModelFitterParams(BaseModel):
    """Constructor parameters for :class:`KMeansHabitatModelFitter`."""

    n_habitats: Optional[int] = Field(default=None, ge=2)
    min_habitats: int = Field(default=2, ge=2)
    max_habitats: int = Field(default=10, ge=3)
    validation: Union[str, List[str]] = "silhouette"
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
        validation: Selection criterion, or a list of criteria that each cast
            one vote: ``"silhouette"`` / ``"calinski_harabasz"`` / ``"gap"``
            (maximise), ``"davies_bouldin"`` (minimise), or ``"inertia"`` /
            ``"elbow"`` / ``"kneedle"`` (Kneedle knee of the inertia curve).
            Since v1.0 ``elbow`` is an alias of ``kneedle``; see
            :mod:`habit.kernels.cluster_selection`.
        n_init: k-means restarts per candidate count.
    """

    def __init__(
        self,
        n_habitats: Optional[int] = None,
        min_habitats: int = 2,
        max_habitats: int = 10,
        validation: Union[str, Sequence[str]] = "silhouette",
        n_init: int = 50,
    ) -> None:
        self._validation_methods = normalize_validation(
            validation, _VALIDATION_METHODS
        )
        if n_habitats is None and max_habitats <= min_habitats:
            raise HABITAPIError(
                "max_habitats must be greater than min_habitats when "
                "n_habitats is selected automatically."
            )
        self.n_habitats = n_habitats
        self.min_habitats = int(min_habitats)
        self.max_habitats = int(max_habitats)
        # Keep a single criterion scalar so specs (and their fingerprints)
        # stay identical to pre-voting runs.
        self.validation: Union[str, List[str]] = (
            self._validation_methods[0]
            if len(self._validation_methods) == 1
            else list(self._validation_methods)
        )
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

    def _score_candidate(
        self,
        matrix: np.ndarray,
        n_clusters: int,
        methods: Sequence[str],
    ) -> Mapping[str, float]:
        """
        Score one candidate habitat count against the requested criteria.

        Args:
            matrix: Pooled supervoxel features.
            n_clusters: Candidate habitat count.
            methods: Criteria to score; a single fit serves all of them.

        Returns:
            Criterion -> score for this candidate.
        """
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        model = self._fit_kmeans(matrix, n_clusters)
        labels = model.labels_
        scores: Dict[str, float] = {}
        for name in methods:
            if name in _INERTIA_METHODS:
                scores[name] = float(model.inertia_)
            elif name == "silhouette":
                scores[name] = float(silhouette_score(matrix, labels))
            elif name == "calinski_harabasz":
                scores[name] = float(calinski_harabasz_score(matrix, labels))
            elif name == "davies_bouldin":
                scores[name] = float(davies_bouldin_score(matrix, labels))
            elif name == "gap":
                scores[name] = float(gap_statistic(matrix, labels))
        return scores

    def _select_n_habitats(self, matrix: np.ndarray) -> tuple:
        """
        Score every candidate count and keep the best validated one.

        Args:
            matrix: Pooled supervoxel features.

        Returns:
            ``(n_habitats, selection_report)``; the report carries the scored
            curves so the choice stays auditable on the fitted model.
        """
        candidates = list(self._candidate_range(matrix.shape[0]))
        methods = self._validation_methods
        chosen, scores_by_method = select_cluster_count(
            candidates,
            methods,
            lambda count, names: self._score_candidate(matrix, count, names),
        )
        return chosen, build_selection_report(
            candidates, methods, scores_by_method, chosen
        )

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
        selection_report = None
        if n_habitats is None:
            n_habitats, selection_report = self._select_n_habitats(matrix)
        if matrix.shape[0] < n_habitats:
            raise HABITAPIError(
                f"Cannot fit {n_habitats} habitats on {matrix.shape[0]} "
                "pooled supervoxels."
            )
        model = self._fit_kmeans(matrix, n_habitats)
        preprocessing_state = {
            "validation": self.validation,
            "inertia": float(model.inertia_),
        }
        if selection_report is not None:
            preprocessing_state["selection_report"] = selection_report
        return build_habitat_model(
            fitter_name="kmeans",
            spec=self.spec,
            centroids=np.asarray(model.cluster_centers_, dtype=np.float64),
            feature_names=feature_names,
            units=units,
            cohort=cohort,
            random_seed=self._seed,
            preprocessing_state=preprocessing_state,
        )


HabitatModelFitterRegistry.register_params_model(
    "kmeans", KMeansHabitatModelFitterParams
)
