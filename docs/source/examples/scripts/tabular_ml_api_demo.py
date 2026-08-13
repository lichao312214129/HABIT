#!/usr/bin/env python
"""
Tabular ML API: train / CV / predict / compare_models + .habitpipeline I/O.

* **Batch** — recipes on a full :class:`~habit.contracts.FeatureTable`.
* **Atomic** — predict on a one-row table or a held-out id slice.
* **compare_models** — requires ``prob_col`` (positive-class probability).

Accompanies ``docs/source/examples/tabular_ml_api.rst``.
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
from habit.domain.pipeline import TablePipeline
import habit.recipes as recipes
from habit.viz import (
    plot_calibration,
    plot_decision_curve,
    plot_precision_recall,
    plot_roc,
    use_style,
)

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
# Weaker subset for the second compare_models curve
FEATURES_B = (
    "compactness error",
    "symmetry error",
    "fractal dimension error",
    "worst texture",
)

frame = pd.read_csv(DATA, dtype={ID_COL: str})
table = FeatureTable(
    frame=frame[[ID_COL, LABEL_COL, *FEATURES]],
    id_columns=(ID_COL,),
    feature_columns=FEATURES,
    outcome=BinaryOutcome(column=LABEL_COL, positive_label=1),
)
table_b = FeatureTable(
    frame=frame[[ID_COL, LABEL_COL, *FEATURES_B]],
    id_columns=(ID_COL,),
    feature_columns=FEATURES_B,
    outcome=BinaryOutcome(column=LABEL_COL, positive_label=1),
)

spec = MLSpec(
    name="ml_api_demo",
    steps=(
        Spec("variance", {"threshold": 1e-8}),
        Spec("zscore"),
    ),
    classifier=Spec("LogisticRegression", {"max_iter": 500}),
    metrics=(Spec("accuracy"), Spec("auc")),
)
spec_b = MLSpec(
    name="ml_api_demo_b",
    steps=(Spec("zscore"),),
    classifier=Spec("LogisticRegression", {"max_iter": 500}),
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


print("=== train_model (hold-out) ===")
fitted = recipes.train_model(table, spec, test_size=0.25, seed=42, stratify=True)
fitted_b = recipes.train_model(table_b, spec_b, test_size=0.25, seed=42, stratify=True)
print(f"  model_a test metrics: { {k: round(v, 3) for k, v in (fitted.test_metrics or {}).items()} }")
print(f"  model_b test metrics: { {k: round(v, 3) for k, v in (fitted_b.test_metrics or {}).items()} }")

print("=== cross_validate ===")
cv = recipes.cross_validate(table, spec, n_splits=5, seed=42)
print(f"  mean metrics: { {k: round(v, 3) for k, v in cv.mean_metrics.items()} }")
print("  fold AUCs:   ", [round(float(fold["auc"]), 3) for fold in cv.fold_metrics])

Path("out").mkdir(exist_ok=True)
archive = Path("out") / "demo.habitpipeline"
fitted.pipeline.save(archive)
reloaded = TablePipeline.load(archive)
holdout = table_for_ids(table, fitted.test_row_ids)
holdout_b = table_for_ids(table_b, fitted_b.test_row_ids)
prediction = recipes.predict_model(reloaded, holdout)
prediction_b = recipes.predict_model(fitted_b.pipeline, holdout_b)
y_true = holdout.frame[LABEL_COL].to_numpy()
y_prob_a = positive_scores(prediction.probabilities)
y_prob_b = positive_scores(prediction_b.probabilities)
print(
    f"  hold-out AUC model_a={roc_auc_score(y_true, y_prob_a):.3f} "
    f"model_b={roc_auc_score(y_true, y_prob_b):.3f}"
)


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


csv_a = write_pred_csv(
    Path("out") / "model_a.csv",
    holdout.frame[ID_COL].astype(str).to_numpy(),
    y_true,
    np.asarray(prediction.predictions),
    y_prob_a,
)
csv_b = write_pred_csv(
    Path("out") / "model_b.csv",
    holdout_b.frame[ID_COL].astype(str).to_numpy(),
    holdout_b.frame[LABEL_COL].to_numpy(),
    np.asarray(prediction_b.predictions),
    y_prob_b,
)
comparison = recipes.compare_models(
    {
        "output_dir": "out/comparison",
        "files_config": [
            {
                "path": str(csv_a),
                "model_name": "LR (6 features)",
                "subject_id_col": "subject_id",
                "label_col": "label",
                "prob_col": "probability",
                "pred_col": "prediction",
            },
            {
                "path": str(csv_b),
                "model_name": "LR (4 features)",
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
print(f"  row id={one_pred.predictions.index[0]}, pred={int(one_pred.predictions.iloc[0])}")

curves = {
    "LR (6 features)": (y_true, y_prob_a),
    "LR (4 features)": (y_true, y_prob_b),
}


def _save(fig: object, name: str) -> None:
    """Save a figure under out/ and close it."""
    fig.savefig(f"out/{name}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote out/{name}")


with use_style("radiology"):
    _save(plot_roc(curves=curves, title="Hold-out ROC"), "tabular_ml_api_roc.png")
    _save(plot_precision_recall(curves=curves, title="Hold-out PR"), "tabular_ml_api_pr.png")
    _save(plot_calibration(curves=curves, title="Hold-out calibration"), "tabular_ml_api_calibration.png")
    _save(plot_decision_curve(curves=curves, title="Hold-out DCA"), "tabular_ml_api_dca.png")
# END example

if __name__ == "__main__":
    gallery = Path("docs/source/_static/images/examples")
    gallery.mkdir(parents=True, exist_ok=True)
    mapping = {
        "out/tabular_ml_api_roc.png": "tabular_ml_api_roc.png",
        "out/tabular_ml_api_pr.png": "tabular_ml_api_pr.png",
        "out/tabular_ml_api_calibration.png": "tabular_ml_api_calibration.png",
        "out/tabular_ml_api_dca.png": "tabular_ml_api_dca.png",
    }
    for src, name in mapping.items():
        (gallery / name).write_bytes(Path(src).read_bytes())
    print("Copied gallery PNGs")
