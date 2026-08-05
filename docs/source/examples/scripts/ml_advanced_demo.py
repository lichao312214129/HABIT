#!/usr/bin/env python
"""
Advanced tabular ML: staged feature selection and model comparison.

* ``MLSpec.pre_preprocessing_feature_selectors`` — selection on the RAW table
  before z-scoring (v0.1 ``before_z_score: true``).
* ``MLSpec.feature_selectors`` — selection after preprocessing.
* :func:`~habit.recipes.compare_models` — ROC/AUC comparison across saved
  prediction CSVs (the programmatic twin of ``habit compare``).

This script accompanies ``docs/source/examples/ml_advanced.rst``.

Run from the repository root::

    python docs/source/examples/scripts/ml_advanced_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from habit import MLSpec, Spec, make_synthetic_feature_table
import habit.recipes as recipes

table = make_synthetic_feature_table(n_rows=60, n_features=10, rng=42)
print(f"Table: {table.frame.shape[0]} rows x {len(table.feature_columns)} features")

# Variance selection MUST run before z-score: after z-scoring every feature
# has unit variance and the selector becomes uninformative.
staged_spec = MLSpec(
    name="staged_selection",
    pre_preprocessing_feature_selectors=(
        Spec("variance", {"threshold": 0.05}),
    ),
    table_preprocessors=(Spec("zscore"),),
    feature_selectors=(Spec("anova", {"k": 3}),),
    classifier=Spec("LogisticRegression", {"max_iter": 500}),
    metrics=(Spec("accuracy"), Spec("auc")),
)

result = recipes.train_model(table, staged_spec, test_size=0.25, seed=42)
print("\n--- Staged pipeline (pre-variance -> zscore -> k-best -> LR) ---")
print("Train metrics:", {k: round(v, 3) for k, v in result.train_metrics.items()})
print("Test metrics: ", {k: round(v, 3) for k, v in result.test_metrics.items()})

# Build two synthetic prediction files for compare_models.
work_dir = Path(tempfile.mkdtemp(prefix="habit_compare_demo_"))
rows = table.frame[["subject", "label"]].copy()
rows["prediction"] = table.frame["label"]
rows["probability"] = table.frame["label"].astype(float)  # demo-only oracle scores
model_a = work_dir / "model_a_predictions.csv"
model_b = work_dir / "model_b_predictions.csv"
rows.to_csv(model_a, index=False)
noisy = rows.copy()
flip_mask = noisy.index % 5 == 0
noisy.loc[flip_mask, "prediction"] = 1 - noisy.loc[flip_mask, "prediction"]
noisy.loc[flip_mask, "probability"] = 1.0 - noisy.loc[flip_mask, "probability"]
noisy.to_csv(model_b, index=False)

compare_config: Dict[str, Any] = {
    "output_dir": str(work_dir / "comparison"),
    "files_config": [
        {
            "path": str(model_a),
            "model_name": "oracle",
            "subject_id_col": "subject",
            "label_col": "label",
            "prediction_col": "prediction",
            "prob_col": "probability",
        },
        {
            "path": str(model_b),
            "model_name": "noisy",
            "subject_id_col": "subject",
            "label_col": "label",
            "prediction_col": "prediction",
            "prob_col": "probability",
        },
    ],
}

compare_result = recipes.compare_models(compare_config)
print(f"\n--- compare_models output: {compare_result.output_dir} ---")
if compare_result.data:
    for model_name, metrics in compare_result.data.items():
        if isinstance(metrics, dict):
            print(f"  {model_name}: { {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()} }")

print("\nML evaluation figures (ROC, calibration) are written under output_dir/plots/")
