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
"""Built-in classifiers (domain ``classifier``).

Thirteen estimator wrappers, one per v0.1 registered model (the fourteenth,
``AutoGluonTabular``, lives in ``autogluon.py`` because of its optional
dependency). Registered names keep the v0.1 CamelCase spellings so existing
configurations keep working; constructor parameters are the documented v0.1
defaults, now explicit and schema-validated. The v0.1 hard-coded
``random_state: 42`` default is replaced by the
:class:`~habit.domain.protocols.Seedable` contract (v1.0 naming decisions).

sklearn and xgboost are imported lazily inside ``_build_estimator`` so
importing this module stays cheap (L3 layer rule).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from habit.domain.classification._base import SklearnClassifierBase
from habit.domain.classification.registry import ClassifierRegistry
from habit.spec.specs import Spec

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeClassifierParams",
    "KnnClassifier",
    "KnnClassifierParams",
    "SvmClassifier",
    "SvmClassifierParams",
    "SvcClassifier",
    "SvcClassifierParams",
    "MlpClassifier",
    "MlpClassifierParams",
    "LogisticRegressionClassifier",
    "LogisticRegressionClassifierParams",
    "RandomForestClassifier",
    "RandomForestClassifierParams",
    "GradientBoostingClassifier",
    "GradientBoostingClassifierParams",
    "XgboostClassifier",
    "XgboostClassifierParams",
    "AdaboostClassifier",
    "AdaboostClassifierParams",
    "GaussianNbClassifier",
    "GaussianNbClassifierParams",
    "MultinomialNbClassifier",
    "MultinomialNbClassifierParams",
    "BernoulliNbClassifier",
    "BernoulliNbClassifierParams",
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
# DecisionTree
# ---------------------------------------------------------------------------


class DecisionTreeClassifierParams(BaseModel):
    """Constructor parameters for :class:`DecisionTreeClassifier`."""

    criterion: str = "gini"
    splitter: str = "best"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[Union[str, float, int]] = None
    class_weight: Optional[Union[str, Dict[Any, float]]] = None


@ClassifierRegistry.register("DecisionTree")
class DecisionTreeClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """Single CART decision tree (sklearn ``DecisionTreeClassifier``)."""

    _spec_name = "DecisionTree"

    def __init__(
        self,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, float, int]] = None,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
    ) -> None:
        super().__init__()
        self._params = {
            "criterion": criterion,
            "splitter": splitter,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
        }

    def _build_estimator(self) -> Any:
        from sklearn.tree import DecisionTreeClassifier as _SkDecisionTree

        return _SkDecisionTree(random_state=self._seed, **self._params)


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------


class KnnClassifierParams(BaseModel):
    """Constructor parameters for :class:`KnnClassifier`."""

    n_neighbors: int = 5
    weights: str = "uniform"
    algorithm: str = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    n_jobs: int = -1


@ClassifierRegistry.register("KNN")
class KnnClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """k-nearest-neighbours classifier (sklearn ``KNeighborsClassifier``)."""

    _spec_name = "KNN"

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        n_jobs: int = -1,
    ) -> None:
        super().__init__()
        self._params = {
            "n_neighbors": n_neighbors,
            "weights": weights,
            "algorithm": algorithm,
            "leaf_size": leaf_size,
            "p": p,
            "metric": metric,
            "n_jobs": n_jobs,
        }

    def _build_estimator(self) -> Any:
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(**self._params)


# ---------------------------------------------------------------------------
# SVM (LinearSVC with decision-function probabilities) and SVC (kernel)
# ---------------------------------------------------------------------------


class SvmClassifierParams(BaseModel):
    """Constructor parameters for :class:`SvmClassifier`."""

    C: float = 1.0
    class_weight: Optional[Union[str, Dict[Any, float]]] = None
    max_iter: int = 1000


@ClassifierRegistry.register("SVM")
class SvmClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """
    Linear support-vector classifier (sklearn ``LinearSVC``).

    ``LinearSVC`` has no native probabilities; following the v0.1 wrapper,
    ``predict_proba`` maps the decision function through a sigmoid (binary)
    or softmax (multi-class), which is a monotone calibration sufficient for
    ranking metrics such as AUC.
    """

    _spec_name = "SVM"

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        max_iter: int = 1000,
    ) -> None:
        super().__init__()
        self._params = {"C": C, "class_weight": class_weight, "max_iter": max_iter}

    def _build_estimator(self) -> Any:
        from sklearn.svm import LinearSVC

        return LinearSVC(random_state=self._seed, **self._params)

    def _predict_proba_matrix(self, X: pd.DataFrame) -> np.ndarray:
        decision_values = self._estimator.decision_function(X)
        if len(self._classes) == 2:
            proba = 1.0 / (1.0 + np.exp(-decision_values))
            return np.vstack([1.0 - proba, proba]).T
        shifted = decision_values - np.max(decision_values, axis=1, keepdims=True)
        exp_decision = np.exp(shifted)
        return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)


class SvcClassifierParams(BaseModel):
    """Constructor parameters for :class:`SvcClassifier`."""

    C: float = 1.0
    kernel: str = "rbf"
    gamma: Union[str, float] = "scale"
    class_weight: Optional[Union[str, Dict[Any, float]]] = None
    probability: bool = True


@ClassifierRegistry.register("SVC")
class SvcClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """
    Kernel support-vector classifier (sklearn ``SVC``).

    ``probability=True`` stays the default because HABIT's ROC/AUC and
    calibration reporting relies on ``predict_proba``; sklearn then fits an
    internal Platt calibration, which makes training slower but probabilities
    meaningful.
    """

    _spec_name = "SVC"

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        probability: bool = True,
    ) -> None:
        super().__init__()
        self._params = {
            "C": C,
            "kernel": kernel,
            "gamma": gamma,
            "class_weight": class_weight,
            "probability": probability,
        }

    def _build_estimator(self) -> Any:
        from sklearn.svm import SVC as _SkSVC

        return _SkSVC(random_state=self._seed, **self._params)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MlpClassifierParams(BaseModel):
    """Constructor parameters for :class:`MlpClassifier`."""

    hidden_layer_sizes: Tuple[int, ...] = (100,)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: Union[str, int] = "auto"
    learning_rate: str = "constant"
    learning_rate_init: float = 0.001
    max_iter: int = 200
    shuffle: bool = True
    early_stopping: bool = False
    validation_fraction: float = 0.1


@ClassifierRegistry.register("MLP")
class MlpClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """Multi-layer perceptron classifier (sklearn ``MLPClassifier``)."""

    _spec_name = "MLP"

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: Union[str, int] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        shuffle: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self._params = {
            "hidden_layer_sizes": tuple(hidden_layer_sizes),
            "activation": activation,
            "solver": solver,
            "alpha": alpha,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "learning_rate_init": learning_rate_init,
            "max_iter": max_iter,
            "shuffle": shuffle,
            "early_stopping": early_stopping,
            "validation_fraction": validation_fraction,
        }

    def _build_estimator(self) -> Any:
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(random_state=self._seed, **self._params)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification (layer sizes serialised as list)."""
        params = dict(self._params)
        params["hidden_layer_sizes"] = list(self._params["hidden_layer_sizes"])
        return Spec(name=self._spec_name, params=params)


# ---------------------------------------------------------------------------
# LogisticRegression
# ---------------------------------------------------------------------------


class LogisticRegressionClassifierParams(BaseModel):
    """Constructor parameters for :class:`LogisticRegressionClassifier`."""

    C: float = 1.0
    penalty: str = "l2"
    solver: str = "liblinear"
    max_iter: int = 1000
    class_weight: Optional[Union[str, Dict[Any, float]]] = None


@ClassifierRegistry.register("LogisticRegression")
class LogisticRegressionClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """Penalised logistic regression (sklearn ``LogisticRegression``)."""

    _spec_name = "LogisticRegression"

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "liblinear",
        max_iter: int = 1000,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
    ) -> None:
        super().__init__()
        self._params = {
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "max_iter": max_iter,
            "class_weight": class_weight,
        }

    def _build_estimator(self) -> Any:
        from sklearn.linear_model import LogisticRegression as _SkLogReg

        return _SkLogReg(random_state=self._seed, **self._params)


# ---------------------------------------------------------------------------
# RandomForest / GradientBoosting / XGBoost / AdaBoost: ensembles
# ---------------------------------------------------------------------------


class RandomForestClassifierParams(BaseModel):
    """Constructor parameters for :class:`RandomForestClassifier`."""

    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[Union[str, float, int]] = "sqrt"
    bootstrap: bool = True
    class_weight: Optional[Union[str, Dict[Any, float]]] = None


@ClassifierRegistry.register("RandomForest")
class RandomForestClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """Bagged decision-tree ensemble (sklearn ``RandomForestClassifier``)."""

    _spec_name = "RandomForest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, float, int]] = "sqrt",
        bootstrap: bool = True,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
    ) -> None:
        super().__init__()
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "class_weight": class_weight,
        }

    def _build_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestClassifier as _SkRF

        return _SkRF(random_state=self._seed, **self._params)


class GradientBoostingClassifierParams(BaseModel):
    """Constructor parameters for :class:`GradientBoostingClassifier`."""

    loss: str = "log_loss"
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1.0
    criterion: str = "friedman_mse"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_depth: int = 3
    max_features: Optional[Union[str, float, int]] = None


@ClassifierRegistry.register("GradientBoosting")
class GradientBoostingClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """Stage-wise additive tree ensemble (sklearn ``GradientBoostingClassifier``)."""

    _spec_name = "GradientBoosting"

    def __init__(
        self,
        loss: str = "log_loss",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: str = "friedman_mse",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 3,
        max_features: Optional[Union[str, float, int]] = None,
    ) -> None:
        super().__init__()
        self._params = {
            "loss": loss,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "criterion": criterion,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
            "max_features": max_features,
        }

    def _build_estimator(self) -> Any:
        from sklearn.ensemble import GradientBoostingClassifier as _SkGB

        return _SkGB(random_state=self._seed, **self._params)


class XgboostClassifierParams(BaseModel):
    """Constructor parameters for :class:`XgboostClassifier`."""

    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"


@ClassifierRegistry.register("XGBoost")
class XgboostClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """
    Gradient-boosted trees from the xgboost library (``XGBClassifier``).

    Kept as a separate entry from sklearn's ``GradientBoosting`` because the
    xgboost regularisation and column-subsampling defaults differ; the import
    is lazy so environments without xgboost still import this module.
    """

    _spec_name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        objective: str = "binary:logistic",
        eval_metric: str = "logloss",
    ) -> None:
        super().__init__()
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": objective,
            "eval_metric": eval_metric,
        }

    def _build_estimator(self) -> Any:
        import xgboost as xgb

        return xgb.XGBClassifier(random_state=self._seed, **self._params)


class AdaboostClassifierParams(BaseModel):
    """Constructor parameters for :class:`AdaboostClassifier`."""

    n_estimators: int = 50
    learning_rate: float = 1.0


@ClassifierRegistry.register("AdaBoost")
class AdaboostClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """Adaptive boosting of decision stumps (sklearn ``AdaBoostClassifier``)."""

    _spec_name = "AdaBoost"

    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0) -> None:
        super().__init__()
        self._params = {"n_estimators": n_estimators, "learning_rate": learning_rate}

    def _build_estimator(self) -> Any:
        from sklearn.ensemble import AdaBoostClassifier as _SkAB

        return _SkAB(random_state=self._seed, **self._params)


# ---------------------------------------------------------------------------
# GaussianNB / MultinomialNB / BernoulliNB: naive Bayes family (deterministic)
# ---------------------------------------------------------------------------


class GaussianNbClassifierParams(BaseModel):
    """Constructor parameters for :class:`GaussianNbClassifier`."""

    priors: Optional[List[float]] = None
    var_smoothing: float = 1e-9


@ClassifierRegistry.register("GaussianNB")
class GaussianNbClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """Gaussian naive Bayes (sklearn ``GaussianNB``; deterministic)."""

    _spec_name = "GaussianNB"

    def __init__(
        self,
        priors: Optional[List[float]] = None,
        var_smoothing: float = 1e-9,
    ) -> None:
        super().__init__()
        self._params = {"priors": priors, "var_smoothing": var_smoothing}

    def _build_estimator(self) -> Any:
        from sklearn.naive_bayes import GaussianNB as _SkGNB

        return _SkGNB(**self._params)


class MultinomialNbClassifierParams(BaseModel):
    """Constructor parameters for :class:`MultinomialNbClassifier`."""

    alpha: float = 1.0
    fit_prior: bool = True
    class_prior: Optional[List[float]] = None


@ClassifierRegistry.register("MultinomialNB")
class MultinomialNbClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """
    Multinomial naive Bayes (sklearn ``MultinomialNB``; deterministic).

    Requires non-negative features (counts); it is typically chained after a
    ``binning`` or ``minmax`` table preprocessor, both of which produce
    non-negative values.
    """

    _spec_name = "MultinomialNB"

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self._params = {
            "alpha": alpha,
            "fit_prior": fit_prior,
            "class_prior": class_prior,
        }

    def _build_estimator(self) -> Any:
        from sklearn.naive_bayes import MultinomialNB as _SkMNB

        return _SkMNB(**self._params)


class BernoulliNbClassifierParams(BaseModel):
    """Constructor parameters for :class:`BernoulliNbClassifier`."""

    alpha: float = 1.0
    binarize: Optional[float] = 0.0
    fit_prior: bool = True
    class_prior: Optional[List[float]] = None


@ClassifierRegistry.register("BernoulliNB")
class BernoulliNbClassifier(_SpecParamsMixin, SklearnClassifierBase):
    """Bernoulli naive Bayes (sklearn ``BernoulliNB``; deterministic)."""

    _spec_name = "BernoulliNB"

    def __init__(
        self,
        alpha: float = 1.0,
        binarize: Optional[float] = 0.0,
        fit_prior: bool = True,
        class_prior: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self._params = {
            "alpha": alpha,
            "binarize": binarize,
            "fit_prior": fit_prior,
            "class_prior": class_prior,
        }

    def _build_estimator(self) -> Any:
        from sklearn.naive_bayes import BernoulliNB as _SkBNB

        return _SkBNB(**self._params)


# ---------------------------------------------------------------------------
# Parameter schemas (registered after the classes so names resolve)
# ---------------------------------------------------------------------------

ClassifierRegistry.register_params_model("DecisionTree", DecisionTreeClassifierParams)
ClassifierRegistry.register_params_model("KNN", KnnClassifierParams)
ClassifierRegistry.register_params_model("SVM", SvmClassifierParams)
ClassifierRegistry.register_params_model("SVC", SvcClassifierParams)
ClassifierRegistry.register_params_model("MLP", MlpClassifierParams)
ClassifierRegistry.register_params_model(
    "LogisticRegression", LogisticRegressionClassifierParams
)
ClassifierRegistry.register_params_model("RandomForest", RandomForestClassifierParams)
ClassifierRegistry.register_params_model(
    "GradientBoosting", GradientBoostingClassifierParams
)
ClassifierRegistry.register_params_model("XGBoost", XgboostClassifierParams)
ClassifierRegistry.register_params_model("AdaBoost", AdaboostClassifierParams)
ClassifierRegistry.register_params_model("GaussianNB", GaussianNbClassifierParams)
ClassifierRegistry.register_params_model("MultinomialNB", MultinomialNbClassifierParams)
ClassifierRegistry.register_params_model("BernoulliNB", BernoulliNbClassifierParams)
