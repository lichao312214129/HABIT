#!/usr/bin/env python
"""
Tabular machine learning: train, cross-validate, predict, TablePipeline I/O.

Demonstrates:

* **Batch** — ``train_model`` / ``cross_validate`` / ``predict_model`` on a
  :class:`~habit.contracts.FeatureTable`.
* **Pipeline artefact** — save/load a fitted :class:`~habit.domain.TablePipeline`
  (``.habitpipeline`` archive), mirroring ``demo_data/results/api/07_ml``.

This script accompanies ``docs/source/examples/tabular_ml.rst``.

Run from the repository root::

    python docs/source/examples/scripts/tabular_ml_quickstart.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from habit import MLSpec, Spec, make_synthetic_feature_table
from habit.domain.pipeline import TablePipeline
import habit.recipes as recipes

# 1. Feature table (replace with FeatureTable.from_csv for real data).
table = make_synthetic_feature_table(n_rows=80, n_features=8, rng=42)
print(f"Table: {table.frame.shape[0]} rows x "
      f"{len(table.feature_columns)} features, outcome={table.outcome.task}")

# Variance MUST run before z-score (raw scale). After z-scoring every
# feature has variance ~1, so putting variance in feature_selectors is a no-op.
# Post-preprocessing selectors (ANOVA, LASSO, ...) go in feature_selectors.
spec = MLSpec(
    name="demo",
    pre_preprocessing_feature_selectors=(
        Spec("variance", {"threshold": 0.01}),
    ),
    table_preprocessors=(Spec("zscore"),),
    classifier=Spec("LogisticRegression", {"max_iter": 500}),
    metrics=(Spec("accuracy"), Spec("auc")),
)

# 2. Hold-out evaluation (leak-free: pipeline fits on train rows only).
result = recipes.train_model(table, spec, test_size=0.25, seed=42)
print("\n--- Hold-out split (75% train / 25% test) ---")
print("Train metrics:", {k: round(v, 3) for k, v in result.train_metrics.items()})
print("Test metrics: ", {k: round(v, 3) for k, v in result.test_metrics.items()})

# 3. Cross-validation (fresh pipeline per fold).
cv = recipes.cross_validate(table, spec, n_splits=5, seed=42)
print("\n--- 5-fold cross-validation ---")
print("Mean metrics:", {k: round(v, 3) for k, v in cv.mean_metrics.items()})
print("Std metrics: ", {k: round(v, 3) for k, v in cv.std_metrics.items()})

# 4. TablePipeline save/load round-trip (publish-and-reuse for tabular ML).
with tempfile.TemporaryDirectory(prefix="habit_ml_pipeline_") as tmp:
    archive = Path(tmp) / "demo.habitpipeline"
    result.pipeline.save(archive)
    reloaded = TablePipeline.load(archive)
    print(f"\n--- TablePipeline round-trip ---")
    print(f"Saved {archive.name} ({archive.stat().st_size} bytes)")
    print(f"Reloaded classifier: {reloaded.classifier.__class__.__name__}")

    prediction = recipes.predict_model(reloaded, table)
    print(f"Predictions: {len(prediction.predictions)} rows")
    print("Class probability columns:", list(prediction.probabilities.columns))
    print(prediction.probabilities.head(3).round(3).to_string())

# 5. Same-table inference with the in-memory pipeline (no reload).
prediction_direct = recipes.predict_model(result.pipeline, table)
print(f"\nDirect predict_model (in-memory pipeline): {len(prediction_direct.predictions)} rows")
