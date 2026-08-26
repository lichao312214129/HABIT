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

"""Phase-4b/5 CLI switch tests: ``habit model`` / ``habit cv`` run on v1 recipes.

These tests prove that ``habit.commands.cmd_ml`` wires the v0.1 YAML into the
v1 stack (LegacyConfigAdapter -> MLSpec -> FeatureTable -> recipe) instead
of the v0.1 engine for train/K-fold paths, including the hold-out validation
design (split_method / id files) and the precomputed-ICC selector. Predict
runs the v1 ``predict_model`` recipe for ``.habitpipeline`` artefacts and
keeps the v0.1 engine only for legacy ``*_final_pipeline.pkl`` pickles.
Everything here finishes in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Optional

import pytest

import habit.commands.cmd_ml as cmd_ml
from habit.commands.cmd_ml import run_kfold, run_ml
from habit.contracts.table import FeatureTable
from habit.datasets.synthetic import make_synthetic_feature_table
from habit.recipes.modeling import ModelResult, train_model as real_train_model
from habit.spec.specs import MLSpec


def _write_csv(table: FeatureTable, path: Path) -> None:
    """Write one synthetic table as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.frame.to_csv(path, index=False)


def _config_yaml(
    csv_path: Path,
    out_dir: Path,
    *,
    run_mode: str = "train",
    pipeline_path: Optional[Path] = None,
    n_splits: int = 3,
    split_block: str = "",
    feature_selection_block: str = "feature_selection_methods: []",
    extra_block: str = "",
) -> str:
    """
    Render a minimal v0.1 ML config for the synthetic CSV.

    Args:
        csv_path: Feature table CSV path.
        out_dir: Output directory.
        run_mode: ``train`` or ``predict``.
        pipeline_path: Saved pipeline path (predict mode only).
        n_splits: Fold count for K-fold configs.
        split_block: Extra split-design lines (``split_method`` etc.).
        feature_selection_block: Full ``feature_selection_methods`` section.
        extra_block: Extra top-level lines (``evaluate`` etc.).

    Returns:
        YAML text.
    """
    pipeline_line = ""
    if pipeline_path is not None:
        pipeline_line = f'pipeline_path: "{pipeline_path.as_posix()}"\n'
    return f"""run_mode: {run_mode}
{pipeline_line}input:
  - path: "{csv_path.as_posix()}"
    subject_id_col: subject
    label_col: label
output: "{out_dir.as_posix()}"
random_state: 0
n_splits: {n_splits}
{split_block}{extra_block}normalization:
  method: z_score
{feature_selection_block}
models:
  LogisticRegression:
    params:
      max_iter: 500
is_visualize: false
is_save_model: true
"""


def _write_config(root: Path, content: str, name: str = "config.yaml") -> Path:
    """Write one config file under ``root`` and return its path."""
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


class _RecipeSpy:
    """
    Wrapper recording recipe invocations while delegating to the real one.

    Attributes:
        calls: Keyword arguments of every invocation.
    """

    def __init__(self, recipe: Callable[..., Any]) -> None:
        self._recipe = recipe
        self.calls: List[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record the call and run the wrapped recipe."""
        self.calls.append({"args": args, "kwargs": kwargs})
        return self._recipe(*args, **kwargs)


@pytest.fixture
def synthetic_table(tmp_path: Path) -> tuple[FeatureTable, Path]:
    """Build a synthetic CSV and return the table plus its path."""
    table = make_synthetic_feature_table(n_rows=30, n_features=5, rng=11)
    csv_path = tmp_path / "features.csv"
    _write_csv(table, csv_path)
    return table, csv_path


@pytest.mark.cli
def test_run_model_dispatches_to_train_model_recipe(
    tmp_path: Path,
    synthetic_table: tuple[FeatureTable, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``habit model`` train path runs train_model, not the v0.1 engine."""
    _, csv_path = synthetic_table
    out_dir = tmp_path / "out_model"
    config_path = _write_config(
        tmp_path, _config_yaml(csv_path, out_dir), name="model.yaml"
    )
    spy = _RecipeSpy(cmd_ml.train_model)
    monkeypatch.setattr(cmd_ml, "train_model", spy)

    run_ml(str(config_path), mode=None)

    assert len(spy.calls) == 1
    table_arg = spy.calls[0]["args"][0]
    spec_arg = spy.calls[0]["args"][1]
    assert isinstance(table_arg, FeatureTable)
    assert len(table_arg.frame) == 30
    assert isinstance(spec_arg, MLSpec)
    assert spec_arg.classifier.name == "LogisticRegression"
    # The v0.1 hold-out workflow always split before fitting; the default
    # split_method=stratified reaches the recipe as a stratified test_size
    # hold-out.
    assert spy.calls[0]["kwargs"]["test_size"] == 0.3
    assert spy.calls[0]["kwargs"]["stratify"] is True

    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "run_manifest.json").is_file()
    assert (out_dir / cmd_ml._V1_PIPELINE_NAME).is_file()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "accuracy" in metrics["train"]
    assert "accuracy" in metrics["test"]
    legacy = json.loads(
        (out_dir / cmd_ml._LEGACY_HOLDOUT_RESULTS).read_text(encoding="utf-8")
    )
    assert set(legacy) == {"train", "test"}


@pytest.mark.cli
def test_run_model_random_split_passes_unstratified_holdout(
    tmp_path: Path,
    synthetic_table: tuple[FeatureTable, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``split_method: random`` reaches the recipe as an unstratified hold-out."""
    _, csv_path = synthetic_table
    out_dir = tmp_path / "out_random"
    config_path = _write_config(
        tmp_path,
        _config_yaml(
            csv_path,
            out_dir,
            split_block="split_method: random\ntest_size: 0.2\n",
        ),
        name="random.yaml",
    )
    spy = _RecipeSpy(cmd_ml.train_model)
    monkeypatch.setattr(cmd_ml, "train_model", spy)

    run_ml(str(config_path), mode=None)

    assert len(spy.calls) == 1
    assert spy.calls[0]["kwargs"]["test_size"] == pytest.approx(0.2)
    assert spy.calls[0]["kwargs"]["stratify"] is False
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"train", "test"}


@pytest.mark.cli
def test_run_model_custom_split_follows_id_files(
    tmp_path: Path,
    synthetic_table: tuple[FeatureTable, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``split_method: custom`` passes the id files' rows to the recipe.

    One id in each file names a row the table does not have: it is dropped
    with a warning (the v0.1 SplitStrategy rule), not an error.
    """
    table, csv_path = synthetic_table
    all_ids = [str(value) for value in table.frame["subject"]]
    train_ids, test_ids = all_ids[:20], all_ids[20:]
    train_file = tmp_path / "train_ids.txt"
    test_file = tmp_path / "test_ids.txt"
    train_file.write_text("\n".join(train_ids + ["ghost_train"]), encoding="utf-8")
    test_file.write_text(", ".join(test_ids + ["ghost_test"]), encoding="utf-8")
    out_dir = tmp_path / "out_custom"
    config_path = _write_config(
        tmp_path,
        _config_yaml(
            csv_path,
            out_dir,
            split_block=(
                "split_method: custom\n"
                f'train_ids_file: "{train_file.as_posix()}"\n'
                f'test_ids_file: "{test_file.as_posix()}"\n'
            ),
        ),
        name="custom.yaml",
    )
    spy = _RecipeSpy(cmd_ml.train_model)
    monkeypatch.setattr(cmd_ml, "train_model", spy)

    run_ml(str(config_path), mode=None)

    assert len(spy.calls) == 1
    assert spy.calls[0]["kwargs"]["train_ids"] == train_ids
    assert spy.calls[0]["kwargs"]["test_ids"] == test_ids
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"train", "test"}


@pytest.mark.cli
def test_run_model_precomputed_icc_selector_stays_on_v1(
    tmp_path: Path,
    synthetic_table: tuple[FeatureTable, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``icc`` block with ``icc_results`` runs on v1 as ``icc_precomputed``.

    The JSON keeps only ``signal`` above the threshold, so the fitted
    pipeline must select exactly that feature -- and the v0.1 engine must
    never be consulted.
    """
    _, csv_path = synthetic_table
    icc_path = tmp_path / "icc_results.json"
    icc_path.write_text(
        json.dumps(
            {
                "test_vs_retest": {
                    "signal": 0.95,
                    "noise0": 0.4,
                    "noise1": {"ICC3": {"value": 0.2}},
                }
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out_icc"
    config_path = _write_config(
        tmp_path,
        _config_yaml(
            csv_path,
            out_dir,
            feature_selection_block=(
                "feature_selection_methods:\n"
                "  - method: icc\n"
                "    params:\n"
                f'      icc_results: "{icc_path.as_posix()}"\n'
                "      groups: [test_vs_retest]\n"
                "      threshold: 0.75\n"
            ),
        ),
        name="icc.yaml",
    )
    spy = _RecipeSpy(cmd_ml.train_model)
    monkeypatch.setattr(cmd_ml, "train_model", spy)

    def _legacy_fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("v0.1 engine must not run for icc_results configs")

    monkeypatch.setattr("habit.api.machine_learning.run_ml", _legacy_fail)

    run_ml(str(config_path), mode=None)

    assert len(spy.calls) == 1
    spec_arg = spy.calls[0]["args"][1]
    assert [selector.name for selector in spec_arg.feature_selectors] == [
        "icc_precomputed"
    ]
    selected = json.loads(
        (out_dir / "selected_features.json").read_text(encoding="utf-8")
    )
    assert selected == ["signal"]


@pytest.mark.cli
def test_predict_with_v1_habitpipeline_runs_predict_model_recipe(
    tmp_path: Path,
    synthetic_table: tuple[FeatureTable, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predict on a ``.habitpipeline`` runs the v1 inference recipe.

    Train once to produce the artefact, then predict on the same CSV with
    ``evaluate: true``: predictions carry the configured output columns and
    the labelled input yields a metrics.json.
    """
    _, csv_path = synthetic_table
    train_dir = tmp_path / "out_train"
    train_config = _write_config(
        tmp_path, _config_yaml(csv_path, train_dir), name="train.yaml"
    )
    run_ml(str(train_config), mode=None)
    pipeline_path = train_dir / cmd_ml._V1_PIPELINE_NAME
    assert pipeline_path.is_file()

    predict_dir = tmp_path / "out_predict_v1"
    predict_config = _write_config(
        tmp_path,
        _config_yaml(
            csv_path,
            predict_dir,
            run_mode="predict",
            pipeline_path=pipeline_path,
            extra_block="evaluate: true\n",
        ),
        name="predict_v1.yaml",
    )
    spy = _RecipeSpy(cmd_ml.predict_model)
    monkeypatch.setattr(cmd_ml, "predict_model", spy)

    def _legacy_fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("v0.1 engine must not run for .habitpipeline")

    monkeypatch.setattr("habit.api.machine_learning.run_ml", _legacy_fail)

    run_ml(str(predict_config), mode=None)

    assert len(spy.calls) == 1
    predictions_path = predict_dir / "predictions.csv"
    assert predictions_path.is_file()
    import pandas as pd

    frame = pd.read_csv(predictions_path)
    assert "predicted_label" in frame.columns
    assert "predicted_probability" in frame.columns
    assert len(frame) == 30
    assert (predict_dir / "metrics.json").is_file()
    assert (predict_dir / "run_manifest.json").is_file()


@pytest.mark.cli
def test_predict_unlabelled_input_skips_evaluation(
    tmp_path: Path,
    synthetic_table: tuple[FeatureTable, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A predict CSV without the label column yields predictions only."""
    table, csv_path = synthetic_table
    unlabelled_path = tmp_path / "unlabelled.csv"
    table.frame.drop(columns=["label"]).to_csv(unlabelled_path, index=False)
    train_dir = tmp_path / "out_train_unlabelled"
    train_config = _write_config(
        tmp_path, _config_yaml(csv_path, train_dir), name="train_u.yaml"
    )
    run_ml(str(train_config), mode=None)

    predict_dir = tmp_path / "out_predict_unlabelled"
    predict_config = _write_config(
        tmp_path,
        _config_yaml(
            unlabelled_path,
            predict_dir,
            run_mode="predict",
            pipeline_path=train_dir / cmd_ml._V1_PIPELINE_NAME,
            extra_block="evaluate: true\n",
        ),
        name="predict_u.yaml",
    )
    monkeypatch.setattr(
        "habit.api.machine_learning.run_ml",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("v0.1 engine must not run for .habitpipeline")
        ),
    )

    run_ml(str(predict_config), mode=None)

    assert (predict_dir / "predictions.csv").is_file()
    assert not (predict_dir / "metrics.json").exists()


@pytest.mark.cli
def test_run_kfold_dispatches_to_cross_validate_recipe(
    tmp_path: Path,
    synthetic_table: tuple[FeatureTable, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``habit cv`` runs cross_validate with n_splits from the YAML."""
    _, csv_path = synthetic_table
    out_dir = tmp_path / "out_cv"
    config_path = _write_config(
        tmp_path,
        _config_yaml(csv_path, out_dir, n_splits=4),
        name="cv.yaml",
    )
    spy = _RecipeSpy(cmd_ml.cross_validate)
    monkeypatch.setattr(cmd_ml, "cross_validate", spy)

    run_kfold(str(config_path))

    assert len(spy.calls) == 1
    assert spy.calls[0]["kwargs"]["n_splits"] == 4
    result_spec = spy.calls[0]["args"][1]
    assert isinstance(result_spec, MLSpec)

    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "mean" in payload
    assert len(payload["folds"]) == 4
    assert (out_dir / "run_manifest.json").is_file()
    assert (out_dir / cmd_ml._LEGACY_KFOLD_RESULTS).is_file()


@pytest.mark.cli
def test_predict_delegates_to_legacy_api(
    tmp_path: Path,
    synthetic_table: tuple[FeatureTable, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predict mode keeps the v0.1 engine via habit.api.machine_learning.run_ml."""
    _, csv_path = synthetic_table
    fake_pipeline = tmp_path / "LogisticRegression_final_pipeline.pkl"
    fake_pipeline.write_bytes(b"legacy-pickle")
    out_dir = tmp_path / "out_predict"
    config_path = _write_config(
        tmp_path,
        _config_yaml(
            csv_path,
            out_dir,
            run_mode="predict",
            pipeline_path=fake_pipeline,
        ),
        name="predict.yaml",
    )

    calls: List[dict[str, Any]] = []

    def _spy(config: Any, **kwargs: Any) -> None:
        calls.append({"config": config, **kwargs})

    monkeypatch.setattr("habit.api.machine_learning.run_ml", _spy)

    run_ml(str(config_path), mode=None)

    assert len(calls) == 1
    assert calls[0]["config"].run_mode == "predict"


@pytest.mark.cli
def test_multi_model_config_fits_every_yaml_entry(
    tmp_path: Path,
    synthetic_table: tuple[FeatureTable, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-model YAML fits every entry, matching v0.1 CLI behaviour."""
    _, csv_path = synthetic_table
    out_dir = tmp_path / "out_multi"
    yaml_text = _config_yaml(csv_path, out_dir).replace(
        "      max_iter: 500",
        "      max_iter: 500\n  RandomForest:\n    params:\n      n_estimators: 10",
    )
    config_path = _write_config(tmp_path, yaml_text, name="multi.yaml")
    captured_specs: List[MLSpec] = []

    def _capture(table: FeatureTable, spec: MLSpec, **kwargs: Any) -> ModelResult:
        captured_specs.append(spec)
        return real_train_model(table, spec, **kwargs)

    monkeypatch.setattr(cmd_ml, "train_model", _capture)

    run_ml(str(config_path), mode=None)

    assert [spec.classifier.name for spec in captured_specs] == [
        "LogisticRegression",
        "RandomForest",
    ]
    pred = out_dir / "all_prediction_results.csv"
    assert pred.is_file()
    header = pred.read_text(encoding="utf-8").splitlines()[0]
    assert "LogisticRegression_prob" in header
    assert "RandomForest_prob" in header
