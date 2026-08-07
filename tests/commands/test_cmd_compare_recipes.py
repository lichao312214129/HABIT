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

"""Fast CLI wiring tests for the ``habit compare`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

import habit.commands.cmd_compare as cmd_compare
from habit.api.machine_learning import run_model_comparison
from habit.commands.cmd_compare import run_compare
from habit.schemas.workflows.ml import ModelComparisonConfig
from habit.recipes.comparison import compare_models, pairwise_delong_test


def _write_prediction_csv(
    path: Path,
    *,
    subject_ids: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    model_tag: str,
    datasets: np.ndarray | None = None,
) -> None:
    """Write one synthetic prediction CSV accepted by ModelComparisonConfig."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if datasets is None:
        datasets = np.array(["test"] * len(subject_ids))
    frame = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "label": labels,
            f"{model_tag}_prob": probs,
            f"{model_tag}_pred": (probs >= 0.5).astype(int),
            "dataset": datasets,
        }
    )
    frame.to_csv(path, index=False)


def _comparison_config_yaml(
    out_dir: Path,
    model_a_csv: Path,
    model_b_csv: Path,
    *,
    split_enabled: bool = False,
    enable_metrics: bool = False,
    enable_viz: bool = False,
) -> str:
    """Render a minimal model-comparison config for synthetic CSVs."""
    return f"""output_dir: "{out_dir.as_posix()}"
files_config:
  - path: "{model_a_csv.as_posix()}"
    model_name: model_a
    subject_id_col: subject_id
    label_col: label
    prob_col: model_a_prob
    pred_col: model_a_pred
    split_col: dataset
  - path: "{model_b_csv.as_posix()}"
    model_name: model_b
    subject_id_col: subject_id
    label_col: label
    prob_col: model_b_prob
    pred_col: model_b_pred
    split_col: dataset
merged_data:
  enabled: true
  save_name: combined_predictions.csv
split:
  enabled: {str(split_enabled).lower()}
visualization:
  roc:
    enabled: {str(enable_viz).lower()}
  dca:
    enabled: {str(enable_viz).lower()}
  calibration:
    enabled: {str(enable_viz).lower()}
  pr_curve:
    enabled: {str(enable_viz).lower()}
delong_test:
  enabled: true
  save_name: delong_results.json
metrics:
  basic_metrics:
    enabled: {str(enable_metrics).lower()}
  youden_metrics:
    enabled: {str(enable_metrics).lower()}
  target_metrics:
    enabled: {str(enable_metrics).lower()}
    targets:
      sensitivity: 0.5
      specificity: 0.5
"""


@pytest.fixture
def synthetic_predictions(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build two aligned synthetic prediction CSVs and return paths."""
    rng = np.random.default_rng(7)
    n_rows = 40
    labels = rng.integers(0, 2, size=n_rows)
    subject_ids = np.array([f"S{i:03d}" for i in range(n_rows)])
    scores_a = rng.random(n_rows)
    scores_b = np.clip(scores_a + rng.normal(0.0, 0.08, size=n_rows), 0.0, 1.0)

    model_a_csv = tmp_path / "pred_a.csv"
    model_b_csv = tmp_path / "pred_b.csv"
    _write_prediction_csv(
        model_a_csv,
        subject_ids=subject_ids,
        labels=labels,
        probs=scores_a,
        model_tag="model_a",
    )
    _write_prediction_csv(
        model_b_csv,
        subject_ids=subject_ids,
        labels=labels,
        probs=scores_b,
        model_tag="model_b",
    )
    return model_a_csv, model_b_csv, tmp_path / "out_compare"


def _write_config_file(root: Path, content: str) -> Path:
    """Write one YAML config under ``root`` and return its path."""
    path = root / "config_compare.yaml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.cli
def test_compare_cli_dispatches_to_recipe(
    synthetic_predictions: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compare loads YAML then calls the L4 recipe, not habit.core.run."""
    model_a_csv, model_b_csv, out_dir = synthetic_predictions
    config_path = out_dir.parent / "config_compare.yaml"
    config_path.write_text(
        _comparison_config_yaml(out_dir, model_a_csv, model_b_csv),
        encoding="utf-8",
    )

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> None:
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(cmd_compare, "compare_models", _spy)

    run_compare(str(config_path))

    assert len(calls) == 1
    config_arg = calls[0]["args"][0]
    assert isinstance(config_arg, ModelComparisonConfig)
    assert calls[0]["kwargs"]["output_dir"] == str(out_dir.resolve())


@pytest.mark.cli
def test_api_delegates_to_recipe(
    synthetic_predictions: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public API forwards to the v1 recipe (not the v0.1 engine)."""
    model_a_csv, model_b_csv, out_dir = synthetic_predictions
    config = ModelComparisonConfig.from_file(
        str(
            _write_config_file(
                out_dir.parent,
                _comparison_config_yaml(out_dir, model_a_csv, model_b_csv),
            )
        )
    )

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> object:
        calls.append({"args": args, "kwargs": kwargs})
        return object()

    monkeypatch.setattr("habit.recipes.comparison.compare_models", _spy)

    run_model_comparison(config, output_dir=str(out_dir))

    assert len(calls) == 1
    assert calls[0]["kwargs"]["output_dir"] == str(out_dir)


@pytest.mark.cli
def test_pairwise_delong_test_on_synthetic_scores() -> None:
    """pairwise_delong_test exposes the v1 DeLong kernel on aligned scores."""
    rng = np.random.default_rng(11)
    y_true = rng.integers(0, 2, size=50)
    scores_a = rng.random(50)
    scores_b = np.clip(scores_a + rng.normal(0.0, 0.05, size=50), 0.0, 1.0)

    result = pairwise_delong_test(y_true, scores_a, scores_b)

    assert 0.0 <= result.auc_a <= 1.0
    assert 0.0 <= result.auc_b <= 1.0
    assert 0.0 <= result.p_value <= 1.0


@pytest.mark.cli
def test_compare_models_writes_delong_artifact(
    synthetic_predictions: tuple[Path, Path, Path],
) -> None:
    """Synthetic CSVs run through compare_models and emit DeLong output."""
    model_a_csv, model_b_csv, out_dir = synthetic_predictions
    config_path = _write_config_file(
        out_dir.parent,
        _comparison_config_yaml(out_dir, model_a_csv, model_b_csv),
    )
    config = ModelComparisonConfig.from_file(str(config_path))

    compare_models(config, output_dir=str(out_dir))

    delong_path = out_dir / "delong_results.json"
    assert delong_path.is_file()
    assert (out_dir / "combined_predictions.csv").is_file()
    assert (out_dir / "habit_run_manifest.json").is_file()


@pytest.mark.cli
def test_compare_models_split_artifacts_checklist(tmp_path: Path) -> None:
    """Split-aware run writes the full acceptance artefact checklist."""
    rng = np.random.default_rng(21)
    n_rows = 60
    labels = rng.integers(0, 2, size=n_rows)
    # Ensure both classes in both splits.
    labels[:15] = 0
    labels[15:30] = 1
    labels[30:45] = 0
    labels[45:] = 1
    subject_ids = np.array([f"S{i:03d}" for i in range(n_rows)])
    datasets = np.array(["train"] * 30 + ["test"] * 30)
    scores_a = np.clip(labels + rng.normal(0.0, 0.25, size=n_rows), 0.0, 1.0)
    scores_b = np.clip(scores_a + rng.normal(0.0, 0.05, size=n_rows), 0.0, 1.0)

    model_a_csv = tmp_path / "pred_a.csv"
    model_b_csv = tmp_path / "pred_b.csv"
    out_dir = tmp_path / "out_split"
    _write_prediction_csv(
        model_a_csv,
        subject_ids=subject_ids,
        labels=labels,
        probs=scores_a,
        model_tag="model_a",
        datasets=datasets,
    )
    _write_prediction_csv(
        model_b_csv,
        subject_ids=subject_ids,
        labels=labels,
        probs=scores_b,
        model_tag="model_b",
        datasets=datasets,
    )
    config = ModelComparisonConfig.from_file(
        str(
            _write_config_file(
                tmp_path,
                _comparison_config_yaml(
                    out_dir,
                    model_a_csv,
                    model_b_csv,
                    split_enabled=True,
                    enable_metrics=True,
                    enable_viz=True,
                ),
            )
        )
    )

    compare_models(config, output_dir=str(out_dir))

    assert (out_dir / "combined_predictions.csv").is_file()
    assert (out_dir / "metrics" / "metrics.json").is_file()
    assert (out_dir / "habit_run_manifest.json").is_file()
    for split_name in ("train", "test"):
        split_dir = out_dir / split_name
        assert (split_dir / "roc_curves.pdf").is_file()
        assert (split_dir / "decision_curves.pdf").is_file()
        assert (split_dir / "calibration_curves.pdf").is_file()
        assert (split_dir / "precision_recall_curves.pdf").is_file()
        delong_path = split_dir / "delong_results.json"
        assert delong_path.is_file()
        payload = delong_path.read_text(encoding="utf-8")
        assert "p_value" in payload
        assert "significant_difference" in payload
        assert "conclusion" in payload

    import json

    metrics = json.loads((out_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    assert "train" in metrics and "test" in metrics
    train_a = metrics["train"]["model_a"]
    test_a = metrics["test"]["model_a"]
    assert "basic_metrics" in train_a
    assert "youden_metrics" in train_a and "youden_metrics" in test_a
    assert "target_metrics" in train_a and "target_metrics" in test_a
    assert "thresholds" in train_a and "youden" in train_a["thresholds"]
    # Youden / target thresholds are fixed on train and reused on test.
    assert train_a["thresholds"]["youden"] == test_a["thresholds"]["youden"]
    assert train_a["thresholds"]["target"] == test_a["thresholds"]["target"]


@pytest.mark.cli
def test_compare_recipe_does_not_import_v0_engine() -> None:
    """The comparison recipe source must stay free of the v0.1 ML engine."""
    import ast
    from pathlib import Path as _Path

    recipe_path = _Path(compare_models.__code__.co_filename)
    tree = ast.parse(recipe_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [
        name
        for name in imported
        if name.startswith("habit.compat.engines.machine_learning")
        or name.startswith("habit.compat.engines.machine_learning.")
    ]
    assert not offenders
