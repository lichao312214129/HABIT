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
"""GMM habitat model fitter (cohort level)."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

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

__all__ = ["GmmHabitatModelFitter", "GmmHabitatModelFitterParams"]

#: Criteria this fitter can compute. Beyond the v0.1 GMM schema (information
#: criteria plus silhouette / Calinski-Harabasz) the structure-based scores
#: available to every centroid model are also offered.
_VALIDATION_METHODS = (
    "bic",
    "aic",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "gap",
)
_COVARIANCE_TYPES = ("full", "tied", "diag", "spherical")


class GmmHabitatModelFitterParams(BaseModel):
    """Constructor parameters for :class:`GmmHabitatModelFitter`."""

    model_config = ConfigDict(extra="forbid")
    n_habitats: Optional[int] = Field(default=None, ge=2)
    min_habitats: int = Field(default=2, ge=2)
    max_habitats: int = Field(default=10, ge=3)
    validation: Union[str, List[str]] = "bic"
    covariance_type: str = "full"
    n_init: int = Field(default=50, gt=0)
    max_iter: int = Field(default=100, gt=0)


@HabitatModelFitterRegistry.register("gmm")
class GmmHabitatModelFitter:
    """
    Learn population habitats by a Gaussian mixture over pooled features.

    Probabilistic counterpart of the k-means fitter: habitat membership is a
    posterior distribution, and model selection uses an information
    criterion. The model stores the mixture means as centroids; soft
    assignment can be added by a dedicated assigner without changing the
    model artefact.

    The fitter is :class:`~habit.domain.protocols.Seedable`; the seed is
    applied to mixture initialisation at fit time.

    Args:
        n_habitats: Fixed habitat count, or ``None`` to select it by
            ``validation``.
        min_habitats: Smallest candidate count during selection.
        max_habitats: Largest candidate count during selection.
        validation: Selection criterion, or a list of criteria that each cast
            one vote: ``"bic"`` / ``"aic"`` / ``"davies_bouldin"`` (minimise),
            or ``"silhouette"`` / ``"calinski_harabasz"`` / ``"gap"``
            (maximise).
        covariance_type: GaussianMixture covariance structure.
        n_init: Number of mixture initialisations per candidate count; the
            best-likelihood run is kept (sklearn ``n_init``).
        max_iter: EM iteration limit per candidate count.
    """

    def __init__(
        self,
        n_habitats: Optional[int] = None,
        min_habitats: int = 2,
        max_habitats: int = 10,
        validation: Union[str, Sequence[str]] = "bic",
        covariance_type: str = "full",
        n_init: int = 50,
        max_iter: int = 100,
    ) -> None:
        self._validation_methods = normalize_validation(
            validation, _VALIDATION_METHODS
        )
        if covariance_type not in _COVARIANCE_TYPES:
            raise HABITAPIError(
                f"covariance_type must be one of {_COVARIANCE_TYPES}; "
                f"got {covariance_type!r}."
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
        self.covariance_type = covariance_type
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self._seed = 0

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name="gmm",
            params={
                "n_habitats": self.n_habitats,
                "min_habitats": self.min_habitats,
                "max_habitats": self.max_habitats,
                "validation": self.validation,
                "covariance_type": self.covariance_type,
                "n_init": self.n_init,
                "max_iter": self.max_iter,
            },
        )

    def set_random_state(self, seed: int) -> None:
        """Set the seed applied to mixture initialisation at fit time."""
        self._seed = int(seed)

    def _fit_gmm(self, matrix: np.ndarray, n_components: int):
        """Fit one Gaussian mixture for a candidate component count."""
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=n_components,
            random_state=self._seed,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )
        model.fit(matrix)
        return model

    def _score_candidate(
        self,
        matrix: np.ndarray,
        n_components: int,
        methods: Sequence[str],
    ) -> Mapping[str, float]:
        """
        Score one candidate habitat count against the requested criteria.

        Args:
            matrix: Pooled supervoxel features.
            n_components: Candidate habitat count.
            methods: Criteria to score; a single fit serves all of them.

        Returns:
            Criterion -> score for this candidate.
        """
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        model = self._fit_gmm(matrix, n_components)
        scores: Dict[str, float] = {}
        # Hard labels are only needed by the structure-based criteria.
        labels = (
            model.predict(matrix)
            if any(name not in ("bic", "aic") for name in methods)
            else None
        )
        for name in methods:
            if name == "bic":
                scores[name] = float(model.bic(matrix))
            elif name == "aic":
                scores[name] = float(model.aic(matrix))
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
        upper = min(self.max_habitats, matrix.shape[0] - 1)
        if upper < self.min_habitats:
            raise HABITAPIError(
                f"Cannot search habitats in [{self.min_habitats}, "
                f"{self.max_habitats}] with only {matrix.shape[0]} samples."
            )
        candidates = list(range(self.min_habitats, upper + 1))
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
        model = self._fit_gmm(matrix, n_habitats)
        preprocessing_state = {
            "validation": self.validation,
            "covariance_type": self.covariance_type,
        }
        if selection_report is not None:
            preprocessing_state["selection_report"] = selection_report
        return build_habitat_model(
            fitter_name="gmm",
            spec=self.spec,
            centroids=np.asarray(model.means_, dtype=np.float64),
            feature_names=feature_names,
            units=units,
            cohort=cohort,
            random_seed=self._seed,
            preprocessing_state=preprocessing_state,
        )


HabitatModelFitterRegistry.register_params_model("gmm", GmmHabitatModelFitterParams)
