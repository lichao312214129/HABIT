#!/usr/bin/env python
"""
Tabular ML API: train / CV / predict / compare_models + .habitpipeline I/O.

* **Batch** — recipes on a full :class:`~habit.contracts.FeatureTable`.
* **Atomic** — predict on a one-row table or a held-out id slice.
* **compare_models** — requires ``prob_col`` (positive-class probability).

Accompanies ``docs/source/examples/tabular_ml_api.rst``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from habit import MLSpec, Spec, make_synthetic_feature_table
from habit.contracts.table import FeatureTable
from habit.domain.pipeline import TablePipeline
import habit.recipes as recipes

logging.basicConfig(level=logging.WARNING)

table = make_synthetic_feature_table(n_rows=80, n_features=8, rng=42)
# Variance on the raw table (pre_preprocessing_*); z-score afterwards.
spec = MLSpec(
    name="ml_api_demo",
    pre_preprocessing_feature_selectors=(
        Spec("variance", {"threshold": 0.01}),
    ),
    table_preprocessors=(Spec("zscore"),),
    classifier=Spec("LogisticRegression", {"max_iter": 500}),
    metrics=(Spec("accuracy"), Spec("auc")),
)

print("=== train_model (hold-out) ===")
fitted = recipes.train_model(table, spec, test_size=0.25, seed=42, stratify=True)
print(f"  test metrics: { {k: round(v, 3) for k, v in (fitted.test_metrics or {}).items()} }")

print("=== cross_validate ===")
cv = recipes.cross_validate(table, spec, n_splits=5, seed=42)
print(f"  mean metrics: { {k: round(v, 3) for k, v in cv.mean_metrics.items()} }")

print("=== predict_model + .habitpipeline ===")
with tempfile.TemporaryDirectory(prefix="habit_ml_api_") as tmp:
    archive = Path(tmp) / "demo.habitpipeline"
    fitted.pipeline.save(archive)
    reloaded = TablePipeline.load(archive)
    prediction = recipes.predict_model(reloaded, table)
    print(f"  predictions={len(prediction.predictions)}, "
          f"proba_cols={list(prediction.probabilities.columns)}")

    # Build two prediction CSVs for compare_models (needs prob_col).
    def _to_csv(path: Path, name_suffix: str) -> Path:
        proba = prediction.probabilities
        positive = proba["1"] if "1" in proba.columns else proba.iloc[:, -1]
        ids = prediction.predictions.index.astype(str)
        label_col = table.outcome_column
        assert label_col is not None
        labels = table.frame.set_index(table.id_columns[0]).loc[ids, label_col]
        # Perturb second model slightly so DeLong has two distinct curves.
        noise = 0.0 if name_suffix == "a" else 0.02
        pd.DataFrame(
            {
                "subject_id": ids,
                "label": labels.to_numpy(),
                "prediction": np.asarray(prediction.predictions),
                "probability": np.clip(positive.to_numpy() + noise, 0.0, 1.0),
            }
        ).to_csv(path, index=False)
        return path

    csv_a = _to_csv(Path(tmp) / "model_a.csv", "a")
    csv_b = _to_csv(Path(tmp) / "model_b.csv", "b")
    compare_dir = Path(tmp) / "comparison"
    comparison = recipes.compare_models(
        {
            "output_dir": str(compare_dir),
            "files_config": [
                {
                    "path": str(csv_a),
                    "model_name": "model_a",
                    "subject_id_col": "subject_id",
                    "label_col": "label",
                    "prob_col": "probability",
                    "pred_col": "prediction",
                },
                {
                    "path": str(csv_b),
                    "model_name": "model_b",
                    "subject_id_col": "subject_id",
                    "label_col": "label",
                    "prob_col": "probability",
                    "pred_col": "prediction",
                },
            ],
        }
    )
    print(f"=== compare_models -> {comparison.output_dir} ===")

print("=== Atomic: single-row predict ===")
one_row = table.frame.iloc[:1].reset_index(drop=True)
one_table = FeatureTable(
    frame=one_row,
    id_columns=table.id_columns,
    feature_columns=table.feature_columns,
    outcome=table.outcome,
)
one_pred = recipes.predict_model(fitted.pipeline, one_table)
print(f"  row id={one_pred.predictions.index[0]}, "
      f"pred={int(one_pred.predictions.iloc[0])}")
