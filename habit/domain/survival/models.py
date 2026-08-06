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
"""Built-in survival models (domain ``survival_model``).

Three estimators spanning the two survival back-ends:

- ``CoxPhSurvival`` wraps lifelines ``CoxPHFitter`` -- the semi-parametric
  Cox proportional-hazards model that clinical survival papers report.
- ``RandomSurvivalForest`` and ``GradientBoostingSurvival`` wrap
  scikit-survival -- the sklearn-compatible non-parametric alternatives.

lifelines and scikit-survival are OPTIONAL dependencies (the ``analysis``
extra); all their imports are lazy inside the backend hooks so importing
this module stays cheap and the rest of HABIT works without them. lifelines
is used for CoxPH because its partial-hazard API (``predict_partial_hazard``)
and baseline-survival handling are the reference implementation; scikit-
survival is used for the tree ensembles because it provides them with a
consistent sklearn interface plus the C-index/Brier metrics evaluation needs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from habit.exceptions import HABITAPIError
from habit.domain.survival._base import SurvivalModelBase
from habit.domain.survival.registry import SurvivalModelRegistry
from habit.spec.specs import Spec

__all__ = [
    "CoxPhSurvival",
    "CoxPhSurvivalParams",
    "RandomSurvivalForest",
    "RandomSurvivalForestParams",
    "GradientBoostingSurvival",
    "GradientBoostingSurvivalParams",
]


class _SpecParamsMixin:
    """Build ``spec.params`` from the constructor-stored parameter mapping."""

    _spec_name: str = ""
    _params: Dict[str, Any]

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name=self._spec_name, params=dict(self._params))


def _import_or_raise(module: str, extra: str, owner: str):
    """Import an optional survival back-end or raise with the fix attached."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise HABITAPIError(
            f"survival_model.{owner} needs the optional dependency "
            f"{module.split('.')[0]!r}; install it with the {extra!r} extra "
            f"(pip install \"habitat-analysis[{extra}]\")."
        ) from exc


# ---------------------------------------------------------------------------
# Cox proportional hazards (lifelines)
# ---------------------------------------------------------------------------


class CoxPhSurvivalParams(BaseModel):
    """Constructor parameters for :class:`CoxPhSurvival`."""

    model_config = ConfigDict(extra="forbid")
    #: L2 penaliser on the partial likelihood (lifelines ``penalizer``).
    penalizer: float = 0.0
    #: Elastic-net mixing; 1.0 is pure L2 (lifelines ``l1_ratio``).
    l1_ratio: float = 0.0


@SurvivalModelRegistry.register("CoxPH")
class CoxPhSurvival(_SpecParamsMixin, SurvivalModelBase):
    """
    Cox proportional-hazards model (lifelines ``CoxPHFitter``).

    The risk score is the partial hazard ``exp(beta'x)``, the natural
    prognostic index of the Cox model: monotone in the linear predictor, so
    the C-index is unchanged by the exponentiation, and interpretable as a
    hazard relative to the baseline. The survival function is read off the
    fitted baseline via ``predict_survival_function``.
    """

    _spec_name = "CoxPH"

    def __init__(self, penalizer: float = 0.0, l1_ratio: float = 0.0) -> None:
        super().__init__()
        self._params = {"penalizer": penalizer, "l1_ratio": l1_ratio}
        self._model: Optional[Any] = None

    def _fit_backend(self, X: pd.DataFrame, target: np.ndarray) -> None:
        cox = _import_or_raise("lifelines", "analysis", self._spec_name)
        from lifelines import CoxPHFitter  # type: ignore

        if self._outcome is None:
            raise HABITAPIError("survival_model.CoxPH has no fitted outcome.")
        frame = X.reset_index(drop=True).copy()
        frame["__habit_time__"] = target["time"]
        frame["__habit_event__"] = target["event"].astype(int)
        self._model = CoxPHFitter(**self._params)
        self._model.fit(
            frame,
            duration_col="__habit_time__",
            event_col="__habit_event__",
        )

    def _risk_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise HABITAPIError("survival_model.CoxPH is not fitted.")
        return np.asarray(
            self._model.predict_partial_hazard(X.reset_index(drop=True)).to_numpy(),
            dtype=np.float64,
        ).ravel()

    def _survival_probabilities(
        self, X: pd.DataFrame, times: np.ndarray
    ) -> np.ndarray:
        if self._model is None:
            raise HABITAPIError("survival_model.CoxPH is not fitted.")
        # lifelines returns (times x subjects); transpose to (subjects x times).
        frame = self._model.predict_survival_function(
            X.reset_index(drop=True), times=times
        )
        return np.asarray(frame.to_numpy().T, dtype=np.float64)


# ---------------------------------------------------------------------------
# Random survival forest (scikit-survival)
# ---------------------------------------------------------------------------


class RandomSurvivalForestParams(BaseModel):
    """Constructor parameters for :class:`RandomSurvivalForest`."""

    model_config = ConfigDict(extra="forbid")
    n_estimators: int = 100
    min_samples_split: int = 6
    min_samples_leaf: int = 3
    max_features: Optional[Any] = "sqrt"
    n_jobs: Optional[int] = None


@SurvivalModelRegistry.register("RandomSurvivalForest")
class RandomSurvivalForest(_SpecParamsMixin, SurvivalModelBase):
    """
    Random survival forest (scikit-survival ``RandomSurvivalForest``).

    The non-parametric tree-ensemble alternative to Cox. ``predict`` returns
    the ensemble risk score (higher means shorter survival); the survival
    function is the ensemble's per-terminal-node Nelson-Aalen average.
    """

    _spec_name = "RandomSurvivalForest"

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: Optional[Any] = "sqrt",
        n_jobs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._params = {
            "n_estimators": n_estimators,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "n_jobs": n_jobs,
        }
        self._model: Optional[Any] = None

    def _fit_backend(self, X: pd.DataFrame, target: np.ndarray) -> None:
        rsf = _import_or_raise("sksurv.ensemble", "analysis", self._spec_name)
        from sksurv.ensemble import RandomSurvivalForest as _RSF  # type: ignore

        params = {k: v for k, v in self._params.items() if v is not None}
        self._model = _RSF(random_state=self._seed, **params)
        self._model.fit(X.to_numpy(), target)

    def _risk_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise HABITAPIError("survival_model.RandomSurvivalForest is not fitted.")
        return np.asarray(self._model.predict(X.to_numpy()), dtype=np.float64)

    def _survival_probabilities(
        self, X: pd.DataFrame, times: np.ndarray
    ) -> np.ndarray:
        if self._model is None:
            raise HABITAPIError("survival_model.RandomSurvivalForest is not fitted.")
        curves = self._model.predict_survival_function(X.to_numpy())
        rows = np.empty((len(curves), times.size), dtype=np.float64)
        for i, curve in enumerate(curves):
            rows[i] = curve(times)
        return rows


# ---------------------------------------------------------------------------
# Gradient boosting on the Cox loss (scikit-survival)
# ---------------------------------------------------------------------------


class GradientBoostingSurvivalParams(BaseModel):
    """Constructor parameters for :class:`GradientBoostingSurvival`."""

    model_config = ConfigDict(extra="forbid")
    loss: str = "coxph"
    learning_rate: float = 0.1
    n_estimators: int = 100
    max_depth: int = 3
    subsample: float = 1.0


@SurvivalModelRegistry.register("GradientBoostingSurvival")
class GradientBoostingSurvival(_SpecParamsMixin, SurvivalModelBase):
    """
    Gradient boosting on the Cox partial-likelihood loss
    (scikit-survival ``GradientBoostingSurvivalAnalysis``).

    ``predict`` returns the risk score on the log-hazard scale. The survival
    function is available via ``predict_survival_function``, which
    scikit-survival computes from the boosted cumulative hazard.
    """

    _spec_name = "GradientBoostingSurvival"

    def __init__(
        self,
        loss: str = "coxph",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
        subsample: float = 1.0,
    ) -> None:
        super().__init__()
        self._params = {
            "loss": loss,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "subsample": subsample,
        }
        self._model: Optional[Any] = None

    def _fit_backend(self, X: pd.DataFrame, target: np.ndarray) -> None:
        _import_or_raise("sksurv.ensemble", "analysis", self._spec_name)
        from sksurv.ensemble import (
            GradientBoostingSurvivalAnalysis as _GBSA,  # type: ignore
        )

        self._model = _GBSA(random_state=self._seed, **self._params)
        self._model.fit(X.to_numpy(), target)

    def _risk_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise HABITAPIError(
                "survival_model.GradientBoostingSurvival is not fitted."
            )
        return np.asarray(self._model.predict(X.to_numpy()), dtype=np.float64)

    def _survival_probabilities(
        self, X: pd.DataFrame, times: np.ndarray
    ) -> np.ndarray:
        if self._model is None:
            raise HABITAPIError(
                "survival_model.GradientBoostingSurvival is not fitted."
            )
        curves = self._model.predict_survival_function(X.to_numpy())
        rows = np.empty((len(curves), times.size), dtype=np.float64)
        for i, curve in enumerate(curves):
            rows[i] = curve(times)
        return rows


# ---------------------------------------------------------------------------
# Parameter schemas (registered after the classes so names resolve)
# ---------------------------------------------------------------------------

SurvivalModelRegistry.register_params_model("CoxPH", CoxPhSurvivalParams)
SurvivalModelRegistry.register_params_model(
    "RandomSurvivalForest", RandomSurvivalForestParams
)
SurvivalModelRegistry.register_params_model(
    "GradientBoostingSurvival", GradientBoostingSurvivalParams
)
