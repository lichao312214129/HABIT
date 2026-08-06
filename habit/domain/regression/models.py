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
"""Built-in regressors (domain ``regressor``).

Six estimator wrappers over the sklearn regression family. Registered names
use the same spelling style as the classifier domain; stochastic estimators
receive the seed through the :class:`~habit.domain.protocols.Seedable`
contract rather than a constructor parameter (v1.0 naming decisions).

sklearn is imported lazily inside ``_build_estimator`` so importing this
module stays cheap (L3 layer rule).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict

from habit.domain.regression._base import SklearnRegressorBase
from habit.domain.regression.registry import RegressorRegistry
from habit.spec.specs import Spec

__all__ = [
    "RidgeRegressor",
    "RidgeRegressorParams",
    "LassoRegressor",
    "LassoRegressorParams",
    "ElasticNetRegressor",
    "ElasticNetRegressorParams",
    "SvrRegressor",
    "SvrRegressorParams",
    "RandomForestRegressor",
    "RandomForestRegressorParams",
    "GradientBoostingRegressor",
    "GradientBoostingRegressorParams",
]


class _SpecParamsMixin:
    """Build ``spec.params`` from the constructor-stored parameter mapping."""

    _spec_name: str = ""
    _params: Dict[str, Any]

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name=self._spec_name, params=dict(self._params))


# ---------------------------------------------------------------------------
# Linear family: Ridge / Lasso / ElasticNet (deterministic)
# ---------------------------------------------------------------------------


class RidgeRegressorParams(BaseModel):
    """Constructor parameters for :class:`RidgeRegressor`."""

    model_config = ConfigDict(extra="forbid")
    alpha: float = 1.0


@RegressorRegistry.register("Ridge")
class RidgeRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """L2-penalised linear regression (sklearn ``Ridge``; deterministic)."""

    _spec_name = "Ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self._params = {"alpha": alpha}

    def _build_estimator(self) -> Any:
        from sklearn.linear_model import Ridge

        return Ridge(**self._params)


class LassoRegressorParams(BaseModel):
    """Constructor parameters for :class:`LassoRegressor`."""

    model_config = ConfigDict(extra="forbid")
    alpha: float = 1.0
    max_iter: int = 1000


@RegressorRegistry.register("Lasso")
class LassoRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """L1-penalised linear regression (sklearn ``Lasso``; deterministic)."""

    _spec_name = "Lasso"

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000) -> None:
        super().__init__()
        self._params = {"alpha": alpha, "max_iter": max_iter}

    def _build_estimator(self) -> Any:
        from sklearn.linear_model import Lasso

        return Lasso(**self._params)


class ElasticNetRegressorParams(BaseModel):
    """Constructor parameters for :class:`ElasticNetRegressor`."""

    model_config = ConfigDict(extra="forbid")
    alpha: float = 1.0
    l1_ratio: float = 0.5
    max_iter: int = 1000


@RegressorRegistry.register("ElasticNet")
class ElasticNetRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """Combined L1/L2 linear regression (sklearn ``ElasticNet``)."""

    _spec_name = "ElasticNet"

    def __init__(
        self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000
    ) -> None:
        super().__init__()
        self._params = {"alpha": alpha, "l1_ratio": l1_ratio, "max_iter": max_iter}

    def _build_estimator(self) -> Any:
        from sklearn.linear_model import ElasticNet

        return ElasticNet(random_state=self._seed, **self._params)


# ---------------------------------------------------------------------------
# Support vector
# ---------------------------------------------------------------------------


class SvrRegressorParams(BaseModel):
    """Constructor parameters for :class:`SvrRegressor`."""

    model_config = ConfigDict(extra="forbid")
    C: float = 1.0
    kernel: str = "rbf"
    gamma: Union[str, float] = "scale"
    epsilon: float = 0.1


@RegressorRegistry.register("SVR")
class SvrRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """Epsilon support-vector regression (sklearn ``SVR``; deterministic)."""

    _spec_name = "SVR"

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        self._params = {"C": C, "kernel": kernel, "gamma": gamma, "epsilon": epsilon}

    def _build_estimator(self) -> Any:
        from sklearn.svm import SVR

        return SVR(**self._params)


# ---------------------------------------------------------------------------
# Ensembles: RandomForest / GradientBoosting (stochastic)
# ---------------------------------------------------------------------------


class RandomForestRegressorParams(BaseModel):
    """Constructor parameters for :class:`RandomForestRegressor`."""

    model_config = ConfigDict(extra="forbid")
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[Union[str, float, int]] = 1.0


@RegressorRegistry.register("RandomForest")
class RandomForestRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """Bagged decision-tree ensemble regression (sklearn ``RandomForestRegressor``)."""

    _spec_name = "RandomForest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, float, int]] = 1.0,
    ) -> None:
        super().__init__()
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }

    def _build_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor as _SkRF

        return _SkRF(random_state=self._seed, **self._params)


class GradientBoostingRegressorParams(BaseModel):
    """Constructor parameters for :class:`GradientBoostingRegressor`."""

    model_config = ConfigDict(extra="forbid")
    loss: str = "squared_error"
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1.0
    max_depth: int = 3


@RegressorRegistry.register("GradientBoosting")
class GradientBoostingRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """Stage-wise additive tree ensemble regression (sklearn ``GradientBoostingRegressor``)."""

    _spec_name = "GradientBoosting"

    def __init__(
        self,
        loss: str = "squared_error",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        max_depth: int = 3,
    ) -> None:
        super().__init__()
        self._params = {
            "loss": loss,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "max_depth": max_depth,
        }

    def _build_estimator(self) -> Any:
        from sklearn.ensemble import GradientBoostingRegressor as _SkGB

        return _SkGB(random_state=self._seed, **self._params)


# ---------------------------------------------------------------------------
# Parameter schemas (registered after the classes so names resolve)
# ---------------------------------------------------------------------------

RegressorRegistry.register_params_model("Ridge", RidgeRegressorParams)
RegressorRegistry.register_params_model("Lasso", LassoRegressorParams)
RegressorRegistry.register_params_model("ElasticNet", ElasticNetRegressorParams)
RegressorRegistry.register_params_model("SVR", SvrRegressorParams)
RegressorRegistry.register_params_model("RandomForest", RandomForestRegressorParams)
RegressorRegistry.register_params_model(
    "GradientBoosting", GradientBoostingRegressorParams
)
