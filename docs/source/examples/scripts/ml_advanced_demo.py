#!/usr/bin/env python
"""
Advanced tabular ML: ordered feature selection and model comparison.

* ``MLSpec.steps`` — one ordered list of table steps; the list order IS the
  execution order, so a selector can sit before, between or after
  preprocessors (older YAML ``before_z_score`` is just "put it first").
* :func:`~habit.recipes.compare_models` — ROC/AUC comparison across saved
  prediction CSVs (the programmatic twin of ``habit compare``).

This script accompanies ``docs/source/examples/ml_advanced.rst``.

Run from the repository root::

    python docs/source/examples/scripts/ml_advanced_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from habit import MLSpec, Spec
from habit.contracts import BinaryOutcome, FeatureTable
import habit.recipes as recipes

# Change DATA / ID_COL / LABEL_COL / FEATURES to your table
DATA = "demo_data/ml_data/breast_cancer_dataset.csv"
ID_COL = "subject_id"
LABEL_COL = "label"
FEATURES = (
    "compactness error",
    "concavity error",
    "symmetry error",
    "fractal dimension error",
    "worst texture",
    "worst smoothness",
)

frame = pd.read_csv(DATA, dtype={ID_COL: str})
table = FeatureTable(
    frame=frame[[ID_COL, LABEL_COL, *FEATURES]],
    id_columns=(ID_COL,),
    feature_columns=FEATURES,
    outcome=BinaryOutcome(column=LABEL_COL, positive_label=1),
)
print(f"Table: {table.frame.shape[0]} rows x {len(table.feature_columns)} features")

# Variance selection MUST run before z-score: after z-scoring every feature
# has unit variance and the selector becomes uninformative.
staged_spec = MLSpec(
    name="staged_selection",
    steps=(
        Spec("variance", {"threshold": 1e-8}),
        Spec("zscore"),
        Spec("anova", {"n_features_to_select": 3}),
    ),
    classifier=Spec("LogisticRegression", {"max_iter": 500}),
    metrics=(Spec("accuracy"), Spec("auc")),
)
forest_spec = MLSpec(
    name="rf_shallow",
    steps=(Spec("zscore"),),
    classifier=Spec("RandomForest", {"n_estimators": 80, "max_depth": 3}),
    metrics=(Spec("accuracy"), Spec("auc")),
)


def table_for_ids(source: FeatureTable, ids: tuple) -> FeatureTable:
    """Return a FeatureTable containing only the given row ids."""
    mask = source.frame[source.id_columns[0]].astype(str).isin(ids)
    return FeatureTable(
        frame=source.frame.loc[mask].reset_index(drop=True),
        id_columns=source.id_columns,
        feature_columns=source.feature_columns,
        outcome=source.outcome,
    )


def positive_scores(probabilities: pd.DataFrame) -> np.ndarray:
    """Positive-class probabilities from a predict_model probability frame."""
    if 1 in probabilities.columns:
        return probabilities[1].to_numpy()
    if "1" in probabilities.columns:
        return probabilities["1"].to_numpy()
    return probabilities.iloc[:, -1].to_numpy()


result = recipes.train_model(table, staged_spec, test_size=0.25, seed=42, stratify=True)
result_rf = recipes.train_model(table, forest_spec, test_size=0.25, seed=42, stratify=True)
print("\n--- Staged pipeline (variance -> zscore -> ANOVA k=3 -> LR) ---")
print("Train metrics:", {k: round(v, 3) for k, v in result.train_metrics.items()})
print("Test metrics: ", {k: round(v, 3) for k, v in (result.test_metrics or {}).items()})
print("--- Shallow random forest ---")
print("Test metrics: ", {k: round(v, 3) for k, v in (result_rf.test_metrics or {}).items()})

holdout = table_for_ids(table, result.test_row_ids)
pred_lr = recipes.predict_model(result.pipeline, holdout)
pred_rf = recipes.predict_model(result_rf.pipeline, holdout)
y_true = holdout.frame[LABEL_COL].to_numpy()
y_prob_lr = positive_scores(pred_lr.probabilities)
y_prob_rf = positive_scores(pred_rf.probabilities)
print(
    f"Hold-out AUC staged-LR={roc_auc_score(y_true, y_prob_lr):.3f} "
    f"RF={roc_auc_score(y_true, y_prob_rf):.3f}"
)

Path("out").mkdir(exist_ok=True)


def write_pred_csv(path: Path, ids: np.ndarray, labels: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> Path:
    """Write a compare_models input CSV (needs prob_col)."""
    pd.DataFrame(
        {
            "subject_id": ids,
            "label": labels,
            "prediction": pred,
            "probability": prob,
        }
    ).to_csv(path, index=False)
    return path


ids = holdout.frame[ID_COL].astype(str).to_numpy()
csv_a = write_pred_csv(Path("out") / "model_staged.csv", ids, y_true, np.asarray(pred_lr.predictions), y_prob_lr)
csv_b = write_pred_csv(Path("out") / "model_rf.csv", ids, y_true, np.asarray(pred_rf.predictions), y_prob_rf)
compare_result = recipes.compare_models(
    {
        "output_dir": "out/comparison",
        "files_config": [
            {
                "path": str(csv_a),
                "model_name": "staged LR",
                "subject_id_col": "subject_id",
                "label_col": "label",
                "pred_col": "prediction",
                "prob_col": "probability",
            },
            {
                "path": str(csv_b),
                "model_name": "shallow RF",
                "subject_id_col": "subject_id",
                "label_col": "label",
                "pred_col": "prediction",
                "prob_col": "probability",
            },
        ],
    }
)
print(f"\n--- compare_models output: {compare_result.output_dir} ---")
if compare_result.data:
    for model_name, metrics in compare_result.data.items():
        if isinstance(metrics, dict):
            printable = {
                k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()
            }
            print(f"  {model_name}: {printable}")
# END example

# BEGIN figures
# Paste after the Script block. Uses y_true, y_prob_lr, y_prob_rf, pred_lr.
from habit.viz import (
    plot_calibration,
    plot_confusion_matrix,
    plot_decision_curve,
    plot_precision_recall,
    plot_roc,
    use_style,
)

curves = {"staged LR": (y_true, y_prob_lr), "shallow RF": (y_true, y_prob_rf)}


def _save(fig: object, name: str) -> None:
    """Save a figure under out/ and close it."""
    fig.savefig(f"out/{name}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote out/{name}")


with use_style("radiology"):
    _save(plot_roc(curves=curves, title="Hold-out ROC"), "ml_advanced_roc.png")
    _save(plot_precision_recall(curves=curves, title="Hold-out PR"), "ml_advanced_pr.png")
    _save(plot_calibration(curves=curves, title="Hold-out calibration"), "ml_advanced_calibration.png")
    _save(plot_decision_curve(curves=curves, title="Hold-out DCA"), "ml_advanced_dca.png")
    _save(
        plot_confusion_matrix(
            y_true,
            np.asarray(pred_lr.predictions),
            title="Staged LR confusion",
            class_names=("0", "1"),
        ),
        "ml_advanced_confusion.png",
    )
# END figures

if __name__ == "__main__":
    gallery = Path("docs/source/_static/images/examples")
    gallery.mkdir(parents=True, exist_ok=True)
    mapping = {
        "out/ml_advanced_roc.png": "ml_advanced_roc.png",
        "out/ml_advanced_pr.png": "ml_advanced_pr.png",
        "out/ml_advanced_calibration.png": "ml_advanced_calibration.png",
        "out/ml_advanced_dca.png": "ml_advanced_dca.png",
        "out/ml_advanced_confusion.png": "ml_advanced_confusion.png",
    }
    for src, name in mapping.items():
        (gallery / name).write_bytes(Path(src).read_bytes())
    print("Copied gallery PNGs")
