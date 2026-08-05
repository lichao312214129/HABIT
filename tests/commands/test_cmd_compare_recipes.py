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
from habit.commands.cmd_compare import run_compare
from habit.compat.engines.machine_learning.config_schemas import ModelComparisonConfig
from habit.recipes.comparison import compare_models, pairwise_delong_test


def _write_prediction_csv(
    path: Path,
    *,
    subject_ids: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    model_tag: str,
) -> None:
    """Write one synthetic prediction CSV accepted by ModelComparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "label": labels,
            f"{model_tag}_prob": probs,
            f"{model_tag}_pred": (probs >= 0.5).astype(int),
            "dataset": "test",
        }
    )
    frame.to_csv(path, index=False)


def _comparison_config_yaml(
    out_dir: Path,
    model_a_csv: Path,
    model_b_csv: Path,
) -> str:
    """Render a minimal v0.1 model-comparison config for synthetic CSVs."""
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
  enabled: false
visualization:
  roc:
    enabled: false
  dca:
    enabled: false
  calibration:
    enabled: false
  pr_curve:
    enabled: false
delong_test:
  enabled: true
  save_name: delong_results.json
metrics:
  basic_metrics:
    enabled: false
  youden_metrics:
    enabled: false
  target_metrics:
    enabled: false
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
def test_compare_models_recipe_delegates_to_api(
    synthetic_predictions: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The L4 recipe forwards to habit.api.machine_learning.run_model_comparison."""
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

    monkeypatch.setattr(
        "habit.api.machine_learning.run_model_comparison",
        _spy,
    )

    compare_models(config, output_dir=str(out_dir))

    assert len(calls) == 1
    assert calls[0]["kwargs"]["output_dir"] == str(out_dir)


def _write_config_file(root: Path, content: str) -> Path:
    """Write one YAML config under ``root`` and return its path."""
    path = root / "config_compare.yaml"
    path.write_text(content, encoding="utf-8")
    return path


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
