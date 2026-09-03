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
receive the seed through the :class:`~habit._protocols.Seedable`
contract rather than a constructor parameter (v1.0 naming decisions).

sklearn is imported lazily inside ``_build_estimator`` so importing this
module stays cheap (L3 layer rule).
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from habit.regression._base import SklearnRegressorBase
from habit.regression.registry import RegressorRegistry
from habit.spec.specs import Spec

__all__ = [
    "RidgeRegressor",
    "LassoRegressor",
    "ElasticNetRegressor",
    "SvrRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
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


def _positive(name: str, value: float | int) -> None:
    """Reject non-positive numeric constructor parameters at the API boundary."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number; got {value!r}.")


@RegressorRegistry.register("Ridge")
class RidgeRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """L2-penalised linear regression (sklearn ``Ridge``; deterministic)."""

    _spec_name = "Ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize Ridge with positive L2 penalty ``alpha``."""
        super().__init__()
        _positive("alpha", alpha)
        self.alpha = alpha
        self._params = {"alpha": alpha}

    def _build_estimator(self) -> Any:
        from sklearn.linear_model import Ridge

        return Ridge(**self._params)


@RegressorRegistry.register("Lasso")
class LassoRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """L1-penalised linear regression (sklearn ``Lasso``; deterministic)."""

    _spec_name = "Lasso"

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000) -> None:
        """Initialize Lasso with positive penalty and iteration limit."""
        super().__init__()
        _positive("alpha", alpha)
        _positive("max_iter", max_iter)
        self.alpha = alpha
        self.max_iter = max_iter
        self._params = {"alpha": alpha, "max_iter": max_iter}

    def _build_estimator(self) -> Any:
        from sklearn.linear_model import Lasso

        return Lasso(**self._params)


@RegressorRegistry.register("ElasticNet")
class ElasticNetRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """Combined L1/L2 linear regression (sklearn ``ElasticNet``)."""

    _spec_name = "ElasticNet"

    def __init__(
        self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000
    ) -> None:
        """Initialize ElasticNet with ``l1_ratio`` constrained to [0, 1]."""
        super().__init__()
        _positive("alpha", alpha)
        _positive("max_iter", max_iter)
        if not isinstance(l1_ratio, (int, float)) or not 0 <= l1_ratio <= 1:
            raise ValueError(f"l1_ratio must lie in [0, 1]; got {l1_ratio!r}.")
        self.alpha, self.l1_ratio, self.max_iter = alpha, l1_ratio, max_iter
        self._params = {"alpha": alpha, "l1_ratio": l1_ratio, "max_iter": max_iter}

    def _build_estimator(self) -> Any:
        from sklearn.linear_model import ElasticNet

        return ElasticNet(random_state=self._seed, **self._params)


# ---------------------------------------------------------------------------
# Support vector
# ---------------------------------------------------------------------------


@RegressorRegistry.register("SVR")
class SvrRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """Epsilon support-vector regression (sklearn ``SVR``; deterministic)."""

    _spec_name = "SVR"

    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        gamma: Union[Literal["scale", "auto"], float] = "scale",
        epsilon: float = 0.1,
    ) -> None:
        """Initialize SVR with sklearn-supported kernel and gamma values."""
        super().__init__()
        _positive("C", C)
        _positive("epsilon", epsilon)
        if kernel not in {"linear", "poly", "rbf", "sigmoid", "precomputed"}:
            raise ValueError(f"Unsupported SVR kernel: {kernel!r}.")
        if not (gamma in {"scale", "auto"} or isinstance(gamma, (int, float))):
            raise ValueError("gamma must be 'scale', 'auto', or a number.")
        self.C, self.kernel, self.gamma, self.epsilon = C, kernel, gamma, epsilon
        self._params = {"C": C, "kernel": kernel, "gamma": gamma, "epsilon": epsilon}

    def _build_estimator(self) -> Any:
        from sklearn.svm import SVR

        return SVR(**self._params)


# ---------------------------------------------------------------------------
# Ensembles: RandomForest / GradientBoosting (stochastic)
# ---------------------------------------------------------------------------


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
        """Initialize a random forest with positive count-like parameters."""
        super().__init__()
        for name, value in (("n_estimators", n_estimators), ("min_samples_split", min_samples_split), ("min_samples_leaf", min_samples_leaf)):
            _positive(name, value)
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.min_samples_split, self.min_samples_leaf = min_samples_split, min_samples_leaf
        self.max_features = max_features
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


@RegressorRegistry.register("GradientBoosting")
class GradientBoostingRegressor(_SpecParamsMixin, SklearnRegressorBase):
    """Stage-wise additive tree ensemble regression (sklearn ``GradientBoostingRegressor``)."""

    _spec_name = "GradientBoosting"

    def __init__(
        self,
        loss: Literal["squared_error", "absolute_error", "huber", "quantile"] = "squared_error",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        max_depth: int = 3,
    ) -> None:
        """Initialize gradient boosting with valid loss and positive controls."""
        super().__init__()
        if loss not in {"squared_error", "absolute_error", "huber", "quantile"}:
            raise ValueError(f"Unsupported gradient-boosting loss: {loss!r}.")
        for name, value in (("learning_rate", learning_rate), ("n_estimators", n_estimators), ("subsample", subsample), ("max_depth", max_depth)):
            _positive(name, value)
        if subsample > 1:
            raise ValueError(f"subsample must not exceed 1; got {subsample!r}.")
        self.loss, self.learning_rate, self.n_estimators = loss, learning_rate, n_estimators
        self.subsample, self.max_depth = subsample, max_depth
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


