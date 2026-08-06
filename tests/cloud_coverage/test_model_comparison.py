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
"""
Coverage matrix: model comparison on two tiny ML runs.

Two hold-out runs are fitted through the v1 ``habit.recipes.train_model``
recipe (run A on the synthetic radiomics table, run B on its noisy retest
twin). Their per-row probabilities are serialised into the v0.1
``all_prediction_results.csv`` layout -- the exact contract
``habit compare`` consumes -- and the comparison then runs through the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
from click.testing import CliRunner

from tests.cloud_coverage.conftest import RenderedConfig, run_cli
from tests.fixtures.synthetic_data import SyntheticTree

#: Feature columns of the synthetic radiomics table (label/id excluded).
def _feature_columns(frame: pd.DataFrame) -> Tuple[str, ...]:
    """Return the ``feature_*`` column names of a synthetic table."""
    return tuple(c for c in frame.columns if c.startswith("feature_"))


def _train_run(csv_path: Path, seed: int) -> pd.DataFrame:
    """
    Fit one hold-out run and return its prediction table.

    Args:
        csv_path: Synthetic feature table CSV (subject_id + features + label).
        seed: Spec random seed.

    Returns:
        DataFrame in the v0.1 ``all_prediction_results.csv`` layout:
        subject_id, label, dataset, LogisticRegression_prob,
        LogisticRegression_pred.
    """
    from habit.contracts.outcome import BinaryOutcome
    from habit.contracts.table import FeatureTable
    from habit.recipes.modeling import train_model
    from habit.spec.specs import MLSpec, Spec

    frame = pd.read_csv(csv_path)
    table = FeatureTable(
        frame=frame,
        id_columns=("subject_id",),
        feature_columns=_feature_columns(frame),
        outcome=BinaryOutcome(column="label", positive_label=1),
    )
    spec = MLSpec(
        name="comparison_ml",
        classifier=Spec(
            name="LogisticRegression",
            params={"max_iter": 1000},
        ),
        table_preprocessors=(Spec(name="zscore"),),
        metrics=(Spec(name="accuracy"), Spec(name="auc")),
        random_seed=seed,
    )
    result = train_model(table, spec, test_size=0.3, stratify=True)
    probabilities = result.pipeline.predict_proba(table)
    labels = result.pipeline.predict(table)
    # predict_proba returns (n_rows, n_classes); the positive class is the
    # last column for the binary endpoint, mirroring sklearn convention.
    import numpy as np

    probabilities = np.asarray(probabilities)
    positive_prob = probabilities[:, -1] if probabilities.ndim == 2 else probabilities
    dataset = pd.Series("train", index=frame.index, dtype=object)
    test_ids = set(result.test_row_ids)
    dataset.loc[frame["subject_id"].isin(test_ids)] = "test"
    return pd.DataFrame(
        {
            "subject_id": frame["subject_id"],
            "label": frame["label"],
            "dataset": dataset,
            "LogisticRegression_prob": positive_prob,
            "LogisticRegression_pred": np.asarray(labels),
        }
    )


@pytest.fixture(scope="module")
def compare_out(synthetic_tree: SyntheticTree, render_config, results_root: Path) -> Path:
    """
    Train both runs, serialise predictions, run ``habit compare`` once.

    Args:
        synthetic_tree: Session synthetic dataset.
        render_config: Session config renderer.
        results_root: Session results directory.

    Returns:
        The comparison output directory.
    """
    input_dir = results_root / "compare_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    pred_a = input_dir / "run_a_predictions.csv"
    pred_b = input_dir / "run_b_predictions.csv"
    _train_run(synthetic_tree.radiomics_csv, seed=42).to_csv(pred_a, index=False)
    _train_run(synthetic_tree.radiomics_retest_csv, seed=42).to_csv(pred_b, index=False)
    rendered: RenderedConfig = render_config(
        "model_comparison.yaml",
        "model_comparison",
        synthetic_tree,
        {"@PRED_A@": pred_a.as_posix(), "@PRED_B@": pred_b.as_posix()},
    )
    run_cli(CliRunner(), ["compare", "-c", str(rendered.path)])
    return rendered.out_dir


@pytest.mark.integration
def test_compare_writes_merged_table(compare_out: Path) -> None:
    """The comparison writes the combined prediction table for both models."""
    combined = compare_out / "combined_predictions.csv"
    assert combined.is_file(), f"missing {combined}"
    frame = pd.read_csv(combined)
    model_cols = [c for c in frame.columns if "prob" in c.lower()]
    assert len(model_cols) >= 2, f"expected two model prob columns, got {model_cols}"


@pytest.mark.integration
def test_compare_writes_figures(compare_out: Path) -> None:
    """ROC/DCA/calibration/PR comparison figures are produced."""
    figures = [
        p
        for p in compare_out.glob("**/*")
        if p.suffix.lower() in (".png", ".pdf", ".html", ".jpg")
    ]
    assert figures, (
        f"no comparison figures under {compare_out}: "
        f"{[p.name for p in compare_out.glob('**/*')]}"
    )


@pytest.mark.integration
def test_compare_input_tables_are_valid(compare_out: Path, results_root: Path) -> None:
    """The serialised run tables carry both splits and both classes."""
    for name in ("run_a_predictions.csv", "run_b_predictions.csv"):
        frame = pd.read_csv(results_root / "compare_inputs" / name)
        assert set(frame["dataset"]) == {"train", "test"}
        assert frame["label"].nunique() == 2
        assert frame["LogisticRegression_prob"].between(0, 1).all()
