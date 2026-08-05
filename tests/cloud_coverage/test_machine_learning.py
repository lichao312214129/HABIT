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
Coverage matrix: machine learning on the synthetic radiomics table.

- stratified 5-fold CV through ``habit cv`` (sklearn backends only --
  autogluon is intentionally not installed in this environment);
- two tiny holdout train runs through ``habit model`` whose prediction
  tables feed the model-comparison suite.

The synthetic table carries one signal feature correlated with the label,
so a sane classifier must beat chance by a wide margin.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from tests.cloud_coverage.conftest import RenderedConfig, run_cli
from tests.fixtures.synthetic_data import SyntheticTree


@pytest.fixture(scope="module")
def ml_train_out(
    synthetic_tree: SyntheticTree, render_config
) -> "tuple[Path, Path]":
    """
    Run both holdout trains once per module and return their output dirs.

    Args:
        synthetic_tree: Session synthetic dataset.
        render_config: Session config renderer.

    Returns:
        ``(out_a, out_b)`` output directories of runs A and B.
    """
    rendered_a: RenderedConfig = render_config(
        "ml_train_a.yaml",
        "ml_train_a",
        synthetic_tree,
        {"@RADIOMICS_CSV@": synthetic_tree.radiomics_csv.as_posix()},
    )
    run_cli(CliRunner(), ["model", "-c", str(rendered_a.path), "-m", "train"])
    rendered_b: RenderedConfig = render_config(
        "ml_train_b.yaml",
        "ml_train_b",
        synthetic_tree,
        {"@RADIOMICS_RETEST_CSV@": synthetic_tree.radiomics_retest_csv.as_posix()},
    )
    run_cli(CliRunner(), ["model", "-c", str(rendered_b.path), "-m", "train"])
    return rendered_a.out_dir, rendered_b.out_dir


@pytest.mark.integration
def test_kfold_cv_cli(synthetic_tree: SyntheticTree, render_config) -> None:
    """Stratified 5-fold CV (LogisticRegression) reports planted-signal AUC.

    Note: the v1 ``habit cv`` recipe fits one classifier per run (the v1
    MLSpec models a single classifier; v0 multi-model blocks warn and only
    fit the first entry), and it reports metrics as JSON rather than
    figures, so assertions target the JSON artefacts.
    """
    import json

    rendered: RenderedConfig = render_config(
        "ml_kfold.yaml",
        "ml_kfold",
        synthetic_tree,
        {"@RADIOMICS_CSV@": synthetic_tree.radiomics_csv.as_posix()},
    )
    run_cli(CliRunner(), ["cv", "-c", str(rendered.path)])
    out_dir = rendered.out_dir
    metrics_path = out_dir / "metrics.json"
    results_path = out_dir / "ml_kfold_results.json"
    assert metrics_path.is_file(), f"missing {metrics_path}"
    assert results_path.is_file(), f"missing {results_path}"
    assert (out_dir / "run_manifest.json").is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["mean"]["auc"] >= 0.7, (
        f"planted signal should give mean AUC >= 0.7, got {metrics['mean']['auc']:.3f}"
    )
    assert len(metrics["folds"]) == 5
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert "LogisticRegression" in results.get("aggregated", {})


@pytest.mark.integration
def test_kfold_cv_random_forest_cli(synthetic_tree: SyntheticTree, render_config) -> None:
    """Stratified 5-fold CV with the RandomForest backend completes."""
    import json

    rendered: RenderedConfig = render_config(
        "ml_kfold_rf.yaml",
        "ml_kfold_rf",
        synthetic_tree,
        {"@RADIOMICS_CSV@": synthetic_tree.radiomics_csv.as_posix()},
    )
    run_cli(CliRunner(), ["cv", "-c", str(rendered.path)])
    metrics = json.loads((rendered.out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mean"]["auc"] >= 0.7
    assert len(metrics["folds"]) == 5


@pytest.mark.integration
def test_model_train_writes_v1_artefacts(
    ml_train_out: "tuple[Path, Path]",
) -> None:
    """Each holdout run writes the v1 artefact set (model, metrics, manifest).

    Note: the v1 ``habit model`` recipe persists ``model.habitpipeline``
    plus JSON metrics; it does not write the v0.1
    ``all_prediction_results.csv`` (per-row probabilities stay in memory on
    the returned ``ModelResult``).
    """
    import json

    out_a, out_b = ml_train_out
    for out_dir in (out_a, out_b):
        assert (out_dir / "model.habitpipeline").is_file()
        assert (out_dir / "metrics.json").is_file()
        assert (out_dir / "run_manifest.json").is_file()
        assert (out_dir / "selected_features.json").is_file()
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert set(metrics) == {"train", "test"}
        assert "auc" in metrics["train"] and "auc" in metrics["test"]


@pytest.mark.integration
def test_model_train_signal_auc(ml_train_out: "tuple[Path, Path]") -> None:
    """The planted signal feature lets run A reach test AUC >= 0.8."""
    import json

    out_a, _ = ml_train_out
    metrics = json.loads((out_a / "metrics.json").read_text(encoding="utf-8"))
    auc = metrics["test"]["auc"]
    assert auc >= 0.8, f"expected AUC >= 0.8 on the planted signal, got {auc:.3f}"
