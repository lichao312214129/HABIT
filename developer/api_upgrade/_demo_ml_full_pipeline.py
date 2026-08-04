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
"""End-to-end tabular ML with the v1.0 domain API.

Flow: load CSV -> FeatureTable -> TablePipeline
      (correlation + variance selection + z-score + AutoGluon)
      -> metrics -> ROC plot -> save artefacts.

Image preprocessing (elastix etc.) is a separate branch; see the comment
block at the bottom of this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from habit.adapters.writers import DirectoryResultWriter
from habit.contracts.outcome import BinaryOutcome
from habit.contracts.table import FeatureTable
from habit.domain.classification import AutogluonTabularClassifier
from habit.domain.evaluation import AccuracyMetric, AucMetric
from habit.domain.feature_selection import CorrelationSelector, VarianceSelector
from habit.domain.pipeline import TablePipeline
from habit.domain.table_preprocessing import ZScorePreprocessor

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = REPO_ROOT / "demo_data" / "ml_data" / "breast_cancer_dataset.csv"
OUT_DIR = REPO_ROOT / "demo_data" / "results" / "_api_demo_ml_pipeline"


def load_feature_table_from_csv(
    csv_path: Path,
    *,
    subject_id_col: str = "subject_id",
    label_col: str = "label",
    feature_columns: Optional[Sequence[str]] = None,
) -> FeatureTable:
    """Load one CSV and wrap it as a typed FeatureTable.

    Args:
        csv_path: Path to the feature table CSV.
        subject_id_col: Column holding unique subject identifiers.
        label_col: Binary outcome column (0/1).
        feature_columns: Explicit feature list; ``None`` uses every column
            except the id and label columns.

    Returns:
        FeatureTable: Validated table with explicit column roles.
    """
    frame: pd.DataFrame = pd.read_csv(csv_path, dtype={subject_id_col: str})
    if feature_columns is None:
        feature_columns = tuple(
            c for c in frame.columns if c not in {subject_id_col, label_col}
        )
    return FeatureTable(
        frame=frame,
        id_columns=(subject_id_col,),
        feature_columns=tuple(feature_columns),
        outcome=BinaryOutcome(column=label_col, positive_label=1),
    )


def split_table(
    table: FeatureTable,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[FeatureTable, FeatureTable]:
    """Hold-out split that preserves FeatureTable semantics.

    Args:
        table: Full cohort table with an outcome column.
        test_size: Fraction held out for evaluation.
        random_state: RNG seed for reproducibility.
        stratify: Stratify on the outcome when possible.

    Returns:
        Tuple[FeatureTable, FeatureTable]: (train, test) tables.
    """

    def _subset(indices: np.ndarray) -> FeatureTable:
        sub = table.frame.iloc[indices].reset_index(drop=True)
        return FeatureTable(
            frame=sub,
            id_columns=table.id_columns,
            feature_columns=table.feature_columns,
            outcome=table.outcome,
            provenance=table.provenance,
        )

    y = table.frame[table.outcome_column]
    split_kwargs = {
        "test_size": test_size,
        "random_state": random_state,
    }
    if stratify and y.nunique() > 1:
        split_kwargs["stratify"] = y
    train_idx, test_idx = train_test_split(
        np.arange(len(table.frame)), **split_kwargs
    )
    return _subset(train_idx), _subset(test_idx)


def build_autogluon_pipeline(
    *,
    corr_threshold: float = 0.80,
    corr_method: str = "spearman",
    variance_top_k: int = 10,
    autogluon_predictor: Optional[dict] = None,
    autogluon_fit: Optional[dict] = None,
    random_state: int = 42,
) -> TablePipeline:
    """Assemble the preprocessing/selection/model chain.

    Step order mirrors ``config_machine_learning_kfold_demo.yaml``:
    correlation -> variance -> z-score -> AutoGluon.

    Args:
        corr_threshold: Absolute correlation cut-off.
        corr_method: ``spearman`` or ``pearson``.
        variance_top_k: Keep the top-k highest-variance features.
        autogluon_predictor: Kwargs forwarded to ``TabularPredictor(...)``.
        autogluon_fit: Kwargs forwarded to ``TabularPredictor.fit(...)``.
        random_state: Seed for stochastic components.

    Returns:
        TablePipeline: Unfitted pipeline ready for ``fit(table)``.
    """
    pipeline = TablePipeline(
        steps=[
            CorrelationSelector(threshold=corr_threshold, method=corr_method),
            VarianceSelector(top_k=variance_top_k),
            ZScorePreprocessor(),
        ],
        model=AutogluonTabularClassifier(
            predictor=autogluon_predictor or {"eval_metric": "roc_auc"},
            fit=autogluon_fit
            or {"presets": "medium_quality", "time_limit": 120, "verbosity": 1},
        ),
    )
    pipeline.set_random_state(random_state)
    return pipeline


def save_evaluation_plots(
    pipeline: TablePipeline,
    test_table: FeatureTable,
    out_dir: Path,
    model_name: str = "AutoGluonTabular",
) -> None:
    """Write ROC / calibration / DCA figures using the v0.1 plotter.

    Plotting has not migrated to ``habit.viz`` yet; the core ``Plotter`` is
    still the practical choice and accepts plain (y_true, y_score) arrays.

    Args:
        pipeline: Fitted TablePipeline.
        test_table: Held-out table with labels.
        out_dir: Directory for PDF figures.
        model_name: Legend label for the single model.
    """
    from habit.core.machine_learning.visualization.plotting import Plotter

    y_true: np.ndarray = test_table.frame[test_table.outcome_column].to_numpy()
    proba_frame: pd.DataFrame = pipeline.predict_proba(test_table)
    positive_col = "1" if "1" in proba_frame.columns else proba_frame.columns[-1]
    y_score: np.ndarray = proba_frame[positive_col].to_numpy(dtype=float)
    bundle = {model_name: (y_true, y_score)}

    plotter = Plotter(str(out_dir))
    plotter.plot_roc_v2(bundle, save_name="roc_curve.pdf", title="Test ROC")
    plotter.plot_calibration_v2(
        bundle, save_name="calibration_curve.pdf", title="Test calibration", n_bins=5
    )
    plotter.plot_dca_v2(bundle, save_name="decision_curve.pdf", title="Test DCA")


def main() -> None:
    """Run the full demo and persist artefacts under ``OUT_DIR``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    table = load_feature_table_from_csv(DATA_CSV)
    train_table, test_table = split_table(table, random_state=42)

    # ------------------------------------------------------------------
    # 2-4. Preprocess + feature selection + AutoGluon (one TablePipeline)
    # ------------------------------------------------------------------
    pipeline = build_autogluon_pipeline(
        corr_threshold=0.80,
        variance_top_k=10,
        autogluon_fit={"presets": "medium_quality", "time_limit": 90, "verbosity": 1},
    )
    pipeline.fit(train_table)

    # ------------------------------------------------------------------
    # 5. Metrics + predictions
    # ------------------------------------------------------------------
    metrics = pipeline.evaluate(
        test_table, [AccuracyMetric(), AucMetric()]
    )
    predictions = pipeline.predict(test_table)
    probabilities = pipeline.predict_proba(test_table)

    print("metrics:", metrics)
    print(probabilities.head())

    # ------------------------------------------------------------------
    # 6. Visualisation (core plotter until habit.viz grows ML plots)
    # ------------------------------------------------------------------
    save_evaluation_plots(pipeline, test_table, OUT_DIR / "figures")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    writer = DirectoryResultWriter(OUT_DIR)
    pipeline_path = pipeline.save(OUT_DIR / "model.habitpipeline")

    pred_frame = test_table.frame[list(test_table.id_columns)].copy()
    pred_frame["prediction"] = predictions.to_numpy()
    pred_frame = pred_frame.join(probabilities.reset_index(drop=True))
    pred_path = OUT_DIR / "test_predictions.csv"
    pred_frame.to_csv(pred_path, index=False)

    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    selected = list(pipeline.transform(train_table).feature_columns)
    (OUT_DIR / "selected_features.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )

    print(f"pipeline: {pipeline_path}")
    print(f"predictions: {pred_path}")
    print(f"figures: {OUT_DIR / 'figures'}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Image preprocessing (elastix) is NOT part of TablePipeline.
# It still goes through the workflow API today:
#
#   from habit.api.preprocessing import run_preprocess
#
#   run_preprocess({
#       "data_dir": "path/to/raw",
#       "out_dir": "path/to/preprocessed",
#       "preprocessing": {
#           "registration": {
#               "images": ["T1", "T2"],
#               "fixed_image": "T1",
#               "backend": "elastix",
#               "elastix_parameter_files": "par0001.txt",
#               "elastix_parameter_overrides": {
#                   "MaximumNumberOfIterations": 512,
#               },
#           },
#           "resample": {"target_spacing": [1.0, 1.0, 1.0]},
#       },
#   })
#
# The preprocessed images would then feed habitat analysis
# (habit.api.habitat / habit.recipes) to produce the feature CSV that
# ``load_feature_table_from_csv`` consumes above.
# ---------------------------------------------------------------------------
