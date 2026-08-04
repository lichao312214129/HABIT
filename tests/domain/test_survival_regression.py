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
"""End-to-end tests for the survival and regression model + metric layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.api.exceptions import HABITAPIError
from habit.contracts import ContinuousOutcome, FeatureTable, SurvivalOutcome
from habit.domain.evaluation import (
    RegressionMetricRegistry,
    SurvivalMetricRegistry,
)
from habit.domain.regression import RegressorRegistry
from habit.domain.survival import SurvivalModelRegistry

pytestmark = pytest.mark.unit


def _regression_table(n: int = 50, seed: int = 0) -> FeatureTable:
    """Continuous outcome with a learnable linear signal."""
    rng = np.random.RandomState(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    y = 2.0 * f1 - f2 + rng.normal(0.0, 0.05, n)
    frame = pd.DataFrame(
        {"subject": [f"s{i}" for i in range(n)], "f1": f1, "f2": f2, "y": y}
    )
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f1", "f2"),
        outcome=ContinuousOutcome("y"),
    )


def _survival_table(n: int = 60, seed: int = 0) -> FeatureTable:
    """Survival endpoint whose hazard genuinely depends on f1."""
    rng = np.random.RandomState(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    # Higher f1 -> higher hazard -> shorter survival (so risk ~ f1 ranks well).
    baseline = 10.0
    time = baseline * np.exp(-0.8 * f1) * rng.uniform(0.5, 1.5, n)
    time = np.clip(time, 0.5, None)
    event = (rng.rand(n) < 0.75).astype(int)
    if not event.any():
        event[0] = 1
    frame = pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(n)],
            "f1": f1,
            "f2": f2,
            "t": time,
            "e": event,
        }
    )
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f1", "f2"),
        outcome=SurvivalOutcome(time_column="t", event_column="e"),
    )


# ---------------------------------------------------------------------------
# Regression models + metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["Ridge", "Lasso", "ElasticNet", "SVR", "RandomForest", "GradientBoosting"]
)
def test_regressors_fit_and_predict(name: str) -> None:
    """Every built-in regressor fits and returns aligned predictions."""
    table = _regression_table()
    model = RegressorRegistry.create(name)
    if hasattr(model, "set_random_state"):
        model.set_random_state(0)
    model.fit(table)
    prediction = model.predict(table)
    assert prediction.shape[0] == table.frame.shape[0]
    assert np.isfinite(prediction.to_numpy()).all()
    assert tuple(prediction.index) == tuple(table.feature_matrix().index)


def test_ridge_recovers_the_linear_signal() -> None:
    """Ridge on a clean linear signal reaches a high R-squared."""
    table = _regression_table()
    model = RegressorRegistry.create("Ridge").fit(table)
    r2 = RegressionMetricRegistry.create("r2")(
        table.frame["y"].to_numpy(), model.predict(table).to_numpy()
    )
    assert r2 > 0.95


@pytest.mark.parametrize("metric", ["r2", "mae", "mse", "rmse"])
def test_regression_metrics_are_finite_and_signed(metric: str) -> None:
    """Each regression metric returns a finite scalar of the right polarity."""
    table = _regression_table()
    model = RegressorRegistry.create("Ridge").fit(table)
    value = RegressionMetricRegistry.create(metric)(
        table.frame["y"].to_numpy(), model.predict(table).to_numpy()
    )
    assert np.isfinite(value)
    if metric != "r2":
        assert value >= 0.0


# ---------------------------------------------------------------------------
# Survival models + metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["CoxPH", "RandomSurvivalForest", "GradientBoostingSurvival"]
)
def test_survival_models_fit_predict_risk_and_function(name: str) -> None:
    """Every survival model returns risk scores and a survival function."""
    table = _survival_table()
    model = SurvivalModelRegistry.create(name)
    if hasattr(model, "set_random_state"):
        model.set_random_state(0)
    model.fit(table)
    risk = model.predict_risk(table)
    assert risk.shape[0] == table.frame.shape[0]
    assert np.isfinite(risk.to_numpy()).all()

    times = np.array([5.0, 10.0, 15.0])
    surv = model.predict_survival_function(table, times)
    assert surv.shape == (table.frame.shape[0], times.size)
    assert ((surv.to_numpy() >= 0.0) & (surv.to_numpy() <= 1.0)).all()


def test_survival_function_is_monotonically_non_increasing() -> None:
    """S(t|x) never rises with t, whatever the backend."""
    table = _survival_table()
    times = np.linspace(1.0, 20.0, 10)
    for name in ["CoxPH", "RandomSurvivalForest", "GradientBoostingSurvival"]:
        model = SurvivalModelRegistry.create(name)
        if hasattr(model, "set_random_state"):
            model.set_random_state(0)
        model.fit(table)
        surv = model.predict_survival_function(table, times).to_numpy()
        assert (np.diff(surv, axis=1) <= 1e-6).all(), name


def test_predict_survival_function_requires_ascending_times() -> None:
    """A descending or 2-D grid is rejected instead of silently misaligned."""
    table = _survival_table()
    model = SurvivalModelRegistry.create("CoxPH").fit(table)
    with pytest.raises(HABITAPIError, match="ascending"):
        model.predict_survival_function(table, np.array([10.0, 5.0]))


def test_cox_risk_ranks_the_signal() -> None:
    """Cox on a hazard driven by f1 ranks by f1, so C-index is high."""
    table = _survival_table()
    model = SurvivalModelRegistry.create("CoxPH").fit(table)
    risk = model.predict_risk(table).to_numpy()
    c_index = SurvivalMetricRegistry.create("c_index")(
        table.frame["t"].to_numpy(),
        table.frame["e"].to_numpy().astype(bool),
        risk,
    )
    assert c_index > 0.7


@pytest.mark.parametrize(
    "metric", ["c_index", "integrated_brier_score", "cumulative_dynamic_auc"]
)
def test_survival_metrics_are_finite(metric: str) -> None:
    """Each survival metric returns a finite scalar for a fitted Cox model."""
    table = _survival_table()
    model = SurvivalModelRegistry.create("CoxPH").fit(table)
    m = SurvivalMetricRegistry.create(metric)
    if m.needs_survival_function:
        table_frame = table.frame
        # Evaluation grid strictly inside the observed follow-up range.
        event_times = table_frame["t"].to_numpy()[table_frame["e"].to_numpy() == 1]
        grid = np.linspace(event_times.min(), table_frame["t"].max() - 1.0, 50)
        prediction = model.predict_survival_function(table, grid).to_numpy()
        value = m(
            table.frame["t"].to_numpy(),
            table.frame["e"].to_numpy().astype(bool),
            prediction,
            times=grid,
        )
    else:
        prediction = model.predict_risk(table).to_numpy()
        value = m(
            table.frame["t"].to_numpy(),
            table.frame["e"].to_numpy().astype(bool),
            prediction,
        )
    assert np.isfinite(value)


def test_integrated_brier_score_rejects_a_risk_vector() -> None:
    """IBS needs a probability matrix; a 1-D risk score is rejected by name."""
    table = _survival_table()
    model = SurvivalModelRegistry.create("CoxPH").fit(table)
    risk = model.predict_risk(table).to_numpy()
    with pytest.raises(HABITAPIError, match="survival-probability matrix"):
        SurvivalMetricRegistry.create("integrated_brier_score")(
            table.frame["t"].to_numpy(),
            table.frame["e"].to_numpy().astype(bool),
            risk,
        )


# ---------------------------------------------------------------------------
# Wrong-endpoint guards
# ---------------------------------------------------------------------------


def test_regressor_rejects_a_survival_table() -> None:
    """A regressor fitting a survival endpoint fails with a typed message."""
    model = RegressorRegistry.create("Ridge")
    with pytest.raises(HABITAPIError, match="'survival'"):
        model.fit(_survival_table())


def test_survival_model_rejects_a_continuous_table() -> None:
    """A survival model fitting a continuous endpoint fails by name."""
    model = SurvivalModelRegistry.create("CoxPH")
    with pytest.raises(HABITAPIError, match="'continuous'"):
        model.fit(_regression_table())
