# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Real-data end-to-end verification for the survival + regression layer.
#
# NOT part of the package: a one-shot verification driver that builds a
# realistic synthetic cohort (genuine signal + noise + censoring, the shape a
# prognostic radiomics study actually has), runs the FULL v1 pipeline for both
# a survival and a regression endpoint, and writes the figures and a metric
# report to disk. Run it to prove the layer works end to end on data, not only
# in unit tests:
#
#   python developer/api_upgrade/_verify_survival_regression_e2e.py <out_dir>

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from habit.contracts import ContinuousOutcome, FeatureTable, SurvivalOutcome
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
    plot_bland_altman,
    plot_brier_curve,
    plot_kaplan_meier,
    plot_predicted_vs_observed,
    plot_survival_calibration,
    plot_time_dependent_auc,
    use_style,
)

RNG = np.random.RandomState(2026)


def make_cohort(n: int = 180) -> pd.DataFrame:
    """
    A realistic prognostic cohort: imaging features, some truly prognostic,
    plus a survival endpoint driven by a subset and a continuous endpoint
    driven by another. Mirrors a habitat-radiomics table (one row per patient).
    """
    # Correlated imaging features (as habitat proportions and texture tend to be).
    base = RNG.multivariate_normal(
        mean=[0.0] * 6,
        cov=(
            [[1, .5, .3, 0, 0, 0],
             [.5, 1, .2, 0, 0, 0],
             [.3, .2, 1, 0, 0, 0],
             [0, 0, 0, 1, .4, 0],
             [0, 0, 0, .4, 1, .3],
             [0, 0, 0, 0, .3, 1]]
        ),
        size=n,
    )
    cols = ["habitat_agg", "habitat_hypo", "texture_corr", "vol", "size_zone", "sphericity"]
    frame = pd.DataFrame(base, columns=cols)
    frame.insert(0, "patient_id", [f"P{i:03d}" for i in range(n)])

    # Survival: hazard driven by the aggressive/hypoxic habitats (f1, f2).
    log_hazard = 0.7 * frame["habitat_agg"] + 0.5 * frame["habitat_hypo"]
    true_time = 24.0 * np.exp(-log_hazard) * RNG.uniform(0.4, 1.6, n)
    # Administrative censoring at ~30 months, plus random loss to follow-up.
    censor_time = np.minimum(RNG.uniform(18, 36, n), 30.0)
    frame["os_months"] = np.clip(np.minimum(true_time, censor_time), 0.5, None)
    frame["os_event"] = (true_time <= censor_time).astype(int)

    # Continuous endpoint: tumour-volume change driven by vol + size_zone.
    frame["delta_volume"] = (
        1.5 * frame["vol"] + 1.0 * frame["size_zone"] + RNG.normal(0, 0.3, n)
    )
    return frame


def split_table(table: FeatureTable, seed: int = 0):
    labels = stratify_labels(table.outcome, table.frame)
    train_idx, test_idx = train_test_indices(
        len(table.frame), test_size=0.3, labels=labels, seed=seed
    )
    kw = dict(
        id_columns=table.id_columns,
        feature_columns=table.feature_columns,
        outcome=table.outcome,
    )
    train = FeatureTable(frame=table.frame.iloc[train_idx].reset_index(drop=True), **kw)
    test = FeatureTable(frame=table.frame.iloc[test_idx].reset_index(drop=True), **kw)
    return train, test


def run_survival(frame: pd.DataFrame, out: Path) -> dict:
    features = ["habitat_agg", "habitat_hypo", "texture_corr", "vol", "size_zone", "sphericity"]
    table = FeatureTable(
        frame=frame,
        id_columns=("patient_id",),
        feature_columns=tuple(features),
        outcome=SurvivalOutcome(time_column="os_months", event_column="os_event"),
    )
    train, test = split_table(table)

    pipeline = TablePipeline(
        steps=[
            TablePreprocessorRegistry.create("zscore"),
            FeatureSelectorRegistry.create("univariate_cox", n_features_to_select=3),
        ],
        model=SurvivalModelRegistry.create("CoxPH"),
    )
    pipeline.set_random_state(0)
    pipeline.fit(train)

    metrics = pipeline.evaluate(
        test,
        [
            SurvivalMetricRegistry.create("c_index"),
            SurvivalMetricRegistry.create("integrated_brier_score"),
            SurvivalMetricRegistry.create("cumulative_dynamic_auc"),
        ],
    )

    # Figures on the held-out cohort.
    time = test.frame["os_months"].to_numpy()
    event = test.frame["os_event"].to_numpy().astype(bool)
    risk = pipeline.predict(test).to_numpy()
    group = (risk > np.median(risk)).astype(int)
    figures = {}
    with use_style("radiology"):
        figures["km"] = plot_kaplan_meier(
            time, event, group, group_names=["low risk", "high risk"]
        )
        figures["time_auc"] = plot_time_dependent_auc(time, event, risk, time_label="Months")
        horizon = float(np.median(time))
        predicted = pipeline.predict_survival_function(test, np.array([horizon])).to_numpy()[:, 0]
        figures["calibration"] = plot_survival_calibration(
            time, event, predicted, horizon=horizon, n_groups=4, time_label="months"
        )
        grid = np.linspace(time[event].min(), time.max() - 1.0, 40)
        probability = pipeline.predict_survival_function(test, grid).to_numpy()
        figures["brier"] = plot_brier_curve(time, event, probability, grid, time_label="Months")

    out.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(out / f"survival_{name}.png", dpi=200)
    pipeline.save(out / "survival_pipeline.habitpipeline")
    return {
        "endpoint": "survival",
        "n_train": int(len(train.frame)),
        "n_test": int(len(test.frame)),
        "event_rate_test": float(event.mean()),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "selected_features": list(pipeline.steps[1].selected_columns_),
    }


def run_regression(frame: pd.DataFrame, out: Path) -> dict:
    features = ["vol", "size_zone", "sphericity", "habitat_agg", "habitat_hypo", "texture_corr"]
    table = FeatureTable(
        frame=frame,
        id_columns=("patient_id",),
        feature_columns=tuple(features),
        outcome=ContinuousOutcome("delta_volume"),
    )
    train, test = split_table(table)

    pipeline = TablePipeline(
        steps=[TablePreprocessorRegistry.create("zscore")],
        model=RegressorRegistry.create("Ridge"),
    )
    pipeline.fit(train)
    metrics = pipeline.evaluate(
        test,
        [
            RegressionMetricRegistry.create("r2"),
            RegressionMetricRegistry.create("mae"),
            RegressionMetricRegistry.create("rmse"),
        ],
    )

    y_true = test.frame["delta_volume"].to_numpy()
    y_pred = pipeline.predict(test).to_numpy()
    with use_style("nature"):
        fig_scatter = plot_predicted_vs_observed(y_true, y_pred)
        fig_ba = plot_bland_altman(y_true, y_pred)
    out.mkdir(parents=True, exist_ok=True)
    fig_scatter.savefig(out / "regression_pred_vs_obs.png", dpi=200)
    fig_ba.savefig(out / "regression_bland_altman.png", dpi=200)
    pipeline.save(out / "regression_pipeline.habitpipeline")
    return {
        "endpoint": "regression",
        "n_train": int(len(train.frame)),
        "n_test": int(len(test.frame)),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    }


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_data/results/_verify_survival_regression")
    frame = make_cohort()
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "cohort.csv", index=False)

    report = {
        "cohort_size": int(len(frame)),
        "event_rate": float(frame["os_event"].mean()),
        "survival": run_survival(frame, out),
        "regression": run_regression(frame, out),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nArtifacts written under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
