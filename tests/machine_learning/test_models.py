"""
Unit tests for model factory and individual model classes.

Each model is tested: instantiation, fit, predict, predict_proba.
Optional-dependency models (XGBoost, AutoGluon) are guarded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_X_y(n: int = 80, n_features: int = 8, seed: int = 0):
    X_arr, y_arr = make_classification(
        n_samples=n, n_features=n_features, n_informative=4, random_state=seed
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(y_arr, name="label")
    return X, y


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------


class TestModelFactory:
    def test_get_available_models_returns_list(self) -> None:
        from habit.core.machine_learning.models.factory import ModelFactory

        models = ModelFactory.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_known_models_are_registered(self) -> None:
        from habit.core.machine_learning.models.factory import ModelFactory

        available = ModelFactory.get_available_models()
        expected = {"LogisticRegression", "RandomForest", "SVM", "DecisionTree"}
        # At least some expected models must be present
        assert len(expected & set(available)) > 0

    def test_create_logistic_regression(self) -> None:
        from habit.core.machine_learning.models.factory import ModelFactory

        model = ModelFactory.create_model("LogisticRegression", config={"max_iter": 100})
        assert model is not None

    def test_create_unknown_raises(self) -> None:
        from habit.core.machine_learning.models.factory import ModelFactory

        with pytest.raises(ValueError, match="not registered"):
            ModelFactory.create_model("NoSuchModel_xyz")


# ---------------------------------------------------------------------------
# BaseModel interface
# ---------------------------------------------------------------------------


class TestBaseModelContract:
    def test_base_model_is_abstract(self) -> None:
        import inspect

        from habit.core.machine_learning.models.base import BaseModel

        abstract = getattr(BaseModel, "__abstractmethods__", set())
        assert len(abstract) > 0


# ---------------------------------------------------------------------------
# Individual models — fit / predict / predict_proba
# ---------------------------------------------------------------------------


CORE_MODELS = [
    ("LogisticRegression", {"max_iter": 200}),
    ("RandomForest", {"n_estimators": 10, "random_state": 0}),
    ("SVM", {"kernel": "rbf", "probability": True}),
    ("DecisionTree", {"max_depth": 3, "random_state": 0}),
    ("KNN", {"n_neighbors": 3}),
    ("NaiveBayes", {}),
    ("GradientBoosting", {"n_estimators": 10, "random_state": 0}),
    ("AdaBoost", {"n_estimators": 10, "random_state": 0}),
    ("MLP", {"max_iter": 100, "random_state": 0}),
]


@pytest.mark.parametrize("model_name,params", CORE_MODELS)
class TestCoreModels:
    def test_model_fits_and_predicts(self, model_name: str, params: dict) -> None:
        from habit.core.machine_learning.models.factory import ModelFactory

        available = ModelFactory.get_available_models()
        if model_name not in available:
            pytest.skip(f"{model_name} not registered")

        X, y = _make_X_y()
        model = ModelFactory.create_model(model_name, config=params)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape[0] == len(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_returns_valid_probabilities(self, model_name: str, params: dict) -> None:
        from habit.core.machine_learning.models.factory import ModelFactory

        available = ModelFactory.get_available_models()
        if model_name not in available:
            pytest.skip(f"{model_name} not registered")

        X, y = _make_X_y()
        model = ModelFactory.create_model(model_name, config=params)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[0] == len(X)
        # Probabilities must be in [0, 1] and rows sum to ~1
        assert np.all(proba >= 0) and np.all(proba <= 1)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# XGBoost (optional)
# ---------------------------------------------------------------------------


class TestXGBoostModel:
    xgb = pytest.importorskip("xgboost", reason="xgboost not installed")

    def test_xgboost_fits(self) -> None:
        from habit.core.machine_learning.models.factory import ModelFactory

        available = ModelFactory.get_available_models()
        if "XGBoost" not in available:
            pytest.skip("XGBoost not registered")

        X, y = _make_X_y()
        model = ModelFactory.create_model("XGBoost", config={"n_estimators": 10})
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape[0] == len(X)


# ---------------------------------------------------------------------------
# HabitEnsembleModel
# ---------------------------------------------------------------------------


class TestHabitEnsembleModel:
    def test_soft_voting_ensemble(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from habit.core.machine_learning.models.ensemble import HabitEnsembleModel

        X, y = _make_X_y(80)

        # Train two separate pipelines as fold estimators
        estimators = []
        for seed in [0, 1]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=200, random_state=seed)),
            ])
            pipe.fit(X, y)
            estimators.append(pipe)

        ensemble = HabitEnsembleModel(estimators=estimators, voting="soft")
        proba = ensemble.predict_proba(X)
        assert proba.shape[0] == len(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_hard_voting_ensemble(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        from habit.core.machine_learning.models.ensemble import HabitEnsembleModel

        X, y = _make_X_y(80)
        estimators = [
            Pipeline([("model", LogisticRegression(max_iter=200, random_state=i))]).fit(X, y)
            for i in range(3)
        ]
        ensemble = HabitEnsembleModel(estimators=estimators, voting="hard")
        preds = ensemble.predict(X)
        assert preds.shape[0] == len(X)
