#!/usr/bin/env python
"""
Tabular machine learning: train, cross-validate, predict, TablePipeline I/O.

Demonstrates:

* **Batch** — ``train_model`` / ``cross_validate`` / ``predict_model`` on a
  :class:`~habit.contracts.FeatureTable`.
* **Hold-out figures** — ROC, PR, calibration, DCA, confusion, permutation
  importance (scored on held-out rows only).
* **Pipeline artefact** — save/load a fitted :class:`~habit.domain.TablePipeline`
  (``.habitpipeline`` archive).

This script accompanies ``docs/source/examples/tabular_ml.rst``.

Run from the repository root::

    python docs/source/examples/scripts/tabular_ml_quickstart.py
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

# Change DATA / ID_COL / LABEL_COL / FEATURES to your table
DATA = "demo_data/ml_data/breast_cancer_dataset.csv"
ID_COL = "subject_id"
LABEL_COL = "label"
# Moderately informative columns so the ROC looks like a typical imaging-ML
# paper (not a perfect split from tumour-size features such as worst area).
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
print(
    f"Table: {table.frame.shape[0]} rows x "
    f"{len(table.feature_columns)} features, outcome={table.outcome.task}"
)

# steps is ONE ordered list and the order is the execution order. Variance
# MUST come before zscore (raw scale). Use a tiny threshold here: the error
# features have small raw variance and would all be dropped at 0.01.
spec = MLSpec(
    name="demo",
    steps=(
        Spec("variance", {"threshold": 1e-8}),
        Spec("zscore"),
    ),
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


# Hold-out evaluation (leak-free: pipeline fits on train rows only).
result = recipes.train_model(table, spec, test_size=0.25, seed=42, stratify=True)
print("\n--- Hold-out split (75% train / 25% test) ---")
print("Train metrics:", {k: round(v, 3) for k, v in result.train_metrics.items()})
print("Test metrics: ", {k: round(v, 3) for k, v in (result.test_metrics or {}).items()})

# Cross-validation (fresh pipeline per fold).
cv = recipes.cross_validate(table, spec, n_splits=5, seed=42)
print("\n--- 5-fold cross-validation ---")
print("Mean metrics:", {k: round(v, 3) for k, v in cv.mean_metrics.items()})
print("Std metrics: ", {k: round(v, 3) for k, v in cv.std_metrics.items()})
print("Fold AUCs:   ", [round(float(fold["auc"]), 3) for fold in cv.fold_metrics])

# TablePipeline save/load round-trip (publish-and-reuse for tabular ML).
Path("out").mkdir(exist_ok=True)
archive = Path("out") / "demo.habitpipeline"
result.pipeline.save(archive)
reloaded = TablePipeline.load(archive)
print("\n--- TablePipeline round-trip ---")
print(f"Saved {archive.name} ({archive.stat().st_size} bytes)")
print(f"Reloaded classifier: {reloaded.classifier.__class__.__name__}")

# Score the held-out rows only (do not plot train+test together).
holdout_table = table_for_ids(table, result.test_row_ids)
holdout = recipes.predict_model(reloaded, holdout_table)
y_true = holdout_table.frame[LABEL_COL].to_numpy()
y_prob = positive_scores(holdout.probabilities)
y_pred = np.asarray(holdout.predictions)
print(f"Hold-out predictions: {len(y_true)} rows, AUC={roc_auc_score(y_true, y_prob):.3f}")

# Permutation importance: drop in hold-out AUC when one column is shuffled.
base_auc = float(roc_auc_score(y_true, y_prob))
rng = np.random.default_rng(42)
n_repeats = 8
imp_mean = np.zeros(len(FEATURES), dtype=np.float64)
imp_std = np.zeros(len(FEATURES), dtype=np.float64)
for index, column in enumerate(FEATURES):
    drops = np.empty(n_repeats, dtype=np.float64)
    for repeat in range(n_repeats):
        shuffled = holdout_table.frame.copy()
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
        perm_table = FeatureTable(
            frame=shuffled,
            id_columns=holdout_table.id_columns,
            feature_columns=holdout_table.feature_columns,
            outcome=holdout_table.outcome,
        )
        perm_pred = recipes.predict_model(reloaded, perm_table)
        perm_auc = float(roc_auc_score(y_true, positive_scores(perm_pred.probabilities)))
        drops[repeat] = base_auc - perm_auc
    imp_mean[index] = float(drops.mean())
    imp_std[index] = float(drops.std())
# END example

# BEGIN figures
# Paste after the Script block. Uses y_true, y_prob, y_pred, FEATURES, imp_*, cv.
from habit.viz import (
    plot_calibration,
    plot_confusion_matrix,
    plot_decision_curve,
    plot_permutation_importance,
    plot_precision_recall,
    plot_roc,
    use_style,
)


def _save(fig: object, name: str) -> None:
    """Save a figure under out/ and close it."""
    fig.savefig(f"out/{name}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote out/{name}")


with use_style("radiology"):
    _save(plot_roc(y_true, y_prob, model_name="LogisticRegression", title="Hold-out ROC"), "tabular_ml_roc.png")
    _save(plot_precision_recall(y_true, y_prob, model_name="LogisticRegression", title="Hold-out PR"), "tabular_ml_pr.png")
    _save(plot_calibration(y_true, y_prob, model_name="LogisticRegression", title="Hold-out calibration"), "tabular_ml_calibration.png")
    _save(plot_decision_curve(y_true, y_prob, model_name="LogisticRegression", title="Hold-out DCA"), "tabular_ml_dca.png")
    _save(
        plot_confusion_matrix(y_true, y_pred, title="Hold-out confusion", class_names=("0", "1")),
        "tabular_ml_confusion.png",
    )
    _save(
        plot_permutation_importance(
            FEATURES, imp_mean, importance_std=imp_std, title="Hold-out permutation importance"
        ),
        "tabular_ml_importance.png",
    )
    fig_cv, ax_cv = plt.subplots(figsize=(4.2, 4.0), facecolor="white")
    fold_aucs = [float(fold["auc"]) for fold in cv.fold_metrics]
    ax_cv.boxplot([fold_aucs], widths=0.35)
    ax_cv.set_xticklabels(["AUC"])
    ax_cv.scatter(np.ones(len(fold_aucs)), fold_aucs, color="#D55E00", zorder=3)
    ax_cv.set_ylim(0.50, 1.00)
    ax_cv.set_ylabel("ROC AUC")
    ax_cv.set_title("5-fold CV AUC")
    ax_cv.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig_cv.tight_layout()
    _save(fig_cv, "tabular_ml_cv_auc.png")
# END figures

if __name__ == "__main__":
    gallery = Path("docs/source/_static/images/examples")
    gallery.mkdir(parents=True, exist_ok=True)
    mapping = {
        "out/tabular_ml_roc.png": "tabular_ml_roc.png",
        "out/tabular_ml_pr.png": "tabular_ml_pr.png",
        "out/tabular_ml_calibration.png": "tabular_ml_calibration.png",
        "out/tabular_ml_dca.png": "tabular_ml_dca.png",
        "out/tabular_ml_confusion.png": "tabular_ml_confusion.png",
        "out/tabular_ml_importance.png": "tabular_ml_importance.png",
        "out/tabular_ml_cv_auc.png": "tabular_ml_cv_auc.png",
    }
    for src, name in mapping.items():
        (gallery / name).write_bytes(Path(src).read_bytes())
    print("Copied gallery PNGs")
