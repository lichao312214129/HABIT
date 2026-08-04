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
"""End-to-end pipelines: preprocess -> select -> model -> evaluate -> figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.figure import Figure

from habit.api.exceptions import HABITAPIError
from habit.contracts import (
    ContinuousOutcome,
    FeatureTable,
    SurvivalOutcome,
    outcome_from_dict,
    outcome_to_dict,
)
from habit.domain.evaluation import (
    RegressionMetricRegistry,
    SurvivalMetricRegistry,
)
from habit.domain.feature_selection import FeatureSelectorRegistry
from habit.domain.pipeline import TablePipeline
from habit.domain.regression import RegressorRegistry
from habit.domain.split import stratify_labels, train_test_indices
from habit.domain.survival import SurvivalModelRegistry
from habit.domain.table_preprocessing import TablePreprocessorRegistry
from habit.viz import (
    plot_kaplan_meier,
    plot_predicted_vs_observed,
    plot_survival_calibration,
)

pytestmark = pytest.mark.unit


def _survival_table(n: int = 120, seed: int = 0) -> FeatureTable:
    """Survival table with a learnable f1-driven hazard and noise columns."""
    rng = np.random.RandomState(seed)
    f1 = rng.normal(size=n)
    time = np.clip(10.0 * np.exp(-0.9 * f1) * rng.uniform(0.5, 1.5, n), 0.5, None)
    event = (rng.rand(n) < 0.7).astype(int)
    frame = pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(n)],
            "f1": f1,
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
            "os_time": time,
            "os_event": event,
        }
    )
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f1", "f2", "f3"),
        outcome=SurvivalOutcome(time_column="os_time", event_column="os_event"),
    )


def _regression_table(n: int = 100, seed: int = 0) -> FeatureTable:
    """Continuous table with a clean linear signal."""
    rng = np.random.RandomState(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    y = 2.0 * f1 - f2 + rng.normal(0.0, 0.05, n)
    frame = pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(n)],
            "f1": f1,
            "f2": f2,
            "delta": y,
        }
    )
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f1", "f2"),
        outcome=ContinuousOutcome("delta"),
    )


def _split(table: FeatureTable, seed: int = 0):
    """Endpoint-aware 70/30 split of a FeatureTable."""
    labels = stratify_labels(table.outcome, table.frame)
    train_index, test_index = train_test_indices(
        len(table.frame), test_size=0.3, labels=labels, seed=seed
    )
    frame = table.frame
    train = FeatureTable(
        frame=frame.iloc[train_index].reset_index(drop=True),
        id_columns=table.id_columns,
        feature_columns=table.feature_columns,
        outcome=table.outcome,
    )
    test = FeatureTable(
        frame=frame.iloc[test_index].reset_index(drop=True),
        id_columns=table.id_columns,
        feature_columns=table.feature_columns,
        outcome=table.outcome,
    )
    return train, test


# ---------------------------------------------------------------------------
# Survival end-to-end
# ---------------------------------------------------------------------------


def test_survival_pipeline_end_to_end() -> None:
    """zscore -> univariate_cox -> CoxPH -> C-index + calibration figure."""
    table = _survival_table()
    train, test = _split(table)

    pipeline = TablePipeline(
        steps=[
            TablePreprocessorRegistry.create("zscore"),
            FeatureSelectorRegistry.create("univariate_cox", n_features_to_select=1),
        ],
        model=SurvivalModelRegistry.create("CoxPH"),
    )
    pipeline.set_random_state(0)
    pipeline.fit(train)

    results = pipeline.evaluate(
        test,
        [
            SurvivalMetricRegistry.create("c_index"),
            SurvivalMetricRegistry.create("cumulative_dynamic_auc"),
        ],
    )
    assert results["c_index"] > 0.6
    assert np.isfinite(results["cumulative_dynamic_auc"])

    # A KM figure stratified by the fitted model's risk, on the held-out set.
    risk = pipeline.predict(test).to_numpy()
    group = (risk > np.median(risk)).astype(int)
    event = test.frame["os_event"].to_numpy().astype(bool)
    time = test.frame["os_time"].to_numpy()
    fig = plot_kaplan_meier(time, event, group, group_names=["low", "high"])
    assert isinstance(fig, Figure)

    # Calibration at the median follow-up horizon.
    horizon = float(np.median(time))
    grid = np.array([horizon])
    predicted = pipeline.predict_survival_function(test, grid).to_numpy()[:, 0]
    calib = plot_survival_calibration(time, event, predicted, horizon=horizon, n_groups=3)
    assert isinstance(calib, Figure)


def test_survival_pipeline_save_load_roundtrip(tmp_path) -> None:
    """A fitted Cox pipeline persists and reloads with identical risk."""
    train, test = _split(_survival_table())
    pipeline = TablePipeline(
        steps=[TablePreprocessorRegistry.create("zscore")],
        model=SurvivalModelRegistry.create("CoxPH"),
    ).fit(train)
    destination = pipeline.save(tmp_path / "cox.habitpipeline")
    loaded = TablePipeline.load(destination)
    np.testing.assert_allclose(
        loaded.predict(test).to_numpy(), pipeline.predict(test).to_numpy()
    )


# ---------------------------------------------------------------------------
# Regression end-to-end
# ---------------------------------------------------------------------------


def test_regression_pipeline_end_to_end() -> None:
    """zscore -> Ridge -> R2/MAE + predicted-vs-observed figure."""
    table = _regression_table()
    train, test = _split(table)

    pipeline = TablePipeline(
        steps=[TablePreprocessorRegistry.create("zscore")],
        model=RegressorRegistry.create("Ridge"),
    )
    pipeline.fit(train)

    results = pipeline.evaluate(
        test,
        [
            RegressionMetricRegistry.create("r2"),
            RegressionMetricRegistry.create("mae"),
            RegressionMetricRegistry.create("rmse"),
        ],
    )
    assert results["r2"] > 0.9
    assert results["mae"] >= 0.0

    y_true = test.frame["delta"].to_numpy()
    y_pred = pipeline.predict(test).to_numpy()
    fig = plot_predicted_vs_observed(y_true, y_pred)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Family-mismatch guards on the generalized pipeline
# ---------------------------------------------------------------------------


def test_predict_proba_rejected_for_survival_model() -> None:
    """predict_proba on a survival pipeline fails with a typed message."""
    train, test = _split(_survival_table())
    pipeline = TablePipeline(
        steps=[], model=SurvivalModelRegistry.create("CoxPH")
    ).fit(train)
    with pytest.raises(HABITAPIError, match="predict_proba"):
        pipeline.predict_proba(test)


def test_evaluate_rejects_metric_family_mismatch() -> None:
    """A classification metric on a survival table raises (wrong family)."""
    from habit.domain.evaluation import MetricRegistry

    train, test = _split(_survival_table())
    pipeline = TablePipeline(
        steps=[], model=SurvivalModelRegistry.create("CoxPH")
    ).fit(train)
    # AUC expects class labels; feeding it (time, event) must not succeed.
    with pytest.raises((HABITAPIError, ValueError, TypeError)):
        pipeline.evaluate(test, [MetricRegistry.create("auc")])


# ---------------------------------------------------------------------------
# Outcome YAML round-trip
# ---------------------------------------------------------------------------


def test_outcome_yaml_round_trip() -> None:
    """Endpoint declarations survive a YAML document unchanged."""
    outcomes = [
        SurvivalOutcome(time_column="os_months", event_column="os_event", event_value=1),
        ContinuousOutcome("delta_volume"),
    ]
    for outcome in outcomes:
        document = yaml.safe_dump({"outcome": outcome_to_dict(outcome)})
        restored = outcome_from_dict(yaml.safe_load(document)["outcome"])
        assert restored == outcome
        assert restored.task == outcome.task


def test_outcome_from_dict_rejects_unknown_task() -> None:
    """A misspelled task names the four supported endpoint kinds."""
    with pytest.raises(HABITAPIError, match="survival"):
        outcome_from_dict({"task": "survial", "time_column": "t", "event_column": "e"})
