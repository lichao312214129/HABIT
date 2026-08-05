#!/usr/bin/env python
"""
Tabular machine learning on a feature table: train, cross-validate, predict.

This script accompanies docs/source/examples/tabular_ml.rst. It uses a
synthetic FeatureTable with one informative feature and pure-noise remainder,
so the correct answer is known and stable.

Run from the repository root:

    python docs/source/examples/scripts/tabular_ml_quickstart.py
"""

from habit import MLSpec, Spec, make_synthetic_feature_table
import habit.recipes as recipes

# 1. Feature table: 80 rows, one signal feature plus seven noise columns and
#    a binary outcome. Replace with FeatureTable.from_csv(...) for real data.
table = make_synthetic_feature_table(n_rows=80, n_features=8, rng=42)
print(f"Table: {table.frame.shape[0]} rows x "
      f"{len(table.feature_columns)} features, outcome={table.outcome.task}")

# 2. Modelling definition: z-score normalisation, a variance filter, and
#    logistic regression, scored with accuracy and AUC.
spec = MLSpec(
    name="demo",
    table_preprocessors=(Spec("zscore"),),
    feature_selectors=(Spec("variance", {"threshold": 0.01}),),
    classifier=Spec("LogisticRegression", {"max_iter": 500}),
    metrics=(Spec("accuracy"), Spec("auc")),
)

# 3. Hold-out evaluation: the pipeline sees the training rows ONLY, so
#    preprocessing statistics and feature selection never leak.
result = recipes.train_model(table, spec, test_size=0.25, seed=42)
print("\n--- Hold-out split (75% train / 25% test) ---")
print("Train metrics:", {k: round(v, 3) for k, v in result.train_metrics.items()})
print("Test metrics: ", {k: round(v, 3) for k, v in result.test_metrics.items()})

# 4. Cross-validation: every fold fits a FRESH pipeline on its own training
#    rows, again leak-free.
cv = recipes.cross_validate(table, spec, n_splits=5, seed=42)
print("\n--- 5-fold cross-validation ---")
print("Mean metrics:", {k: round(v, 3) for k, v in cv.mean_metrics.items()})
print("Std metrics: ", {k: round(v, 3) for k, v in cv.std_metrics.items()})

# 5. Inference: the fitted pipeline replays its TRAINING preprocessing and
#    selection state on new rows -- it never refits.
prediction = recipes.predict_model(result.pipeline, table)
print("\n--- Prediction with the fitted pipeline ---")
print(f"Predictions: {len(prediction.predictions)} rows")
print("Class probability columns:", list(prediction.probabilities.columns))
print(prediction.probabilities.head(3).round(3).to_string())
