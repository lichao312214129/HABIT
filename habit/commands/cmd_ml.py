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
"""Machine learning command implementation.

L5 wiring only: the command parses the v0.1 YAML with the v0.1 schema,
translates it into the v1 document model through ``LegacyConfigAdapter``,
assembles a :class:`~habit.contracts.table.FeatureTable`, and hands the
work to the v1 tabular recipes. No algorithm lives here; the only
module-level ``habit.core`` dependency left is the v0.1 config *schema*,
which is the YAML parsing contract this command must honour.

Train mode routes to :func:`habit.recipes.modeling.train_model` (``habit
model``) or :func:`habit.recipes.modeling.cross_validate` (``habit cv``),
with the hold-out validation design (``split_method`` / ``test_size`` / id
files) stated at the call site from the v0.1 config. Predict mode applies a
v1 ``.habitpipeline`` through :func:`habit.recipes.modeling.predict_model`;
legacy ``*_final_pipeline.pkl`` pickles keep the v0.1 engine via
:func:`habit.api.machine_learning.run_ml` (they are opaque to the v1 loader,
mirroring habitat's legacy-pickle predict path).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import click
import pandas as pd

from habit.api.machine_learning import apply_ml_mode_override
from habit.commands.common import (
    echo_success,
    exit_with_error,
    load_config_or_exit,
)
from habit.contracts.outcome import BinaryOutcome
from habit.contracts.provenance import Provenance
from habit.contracts.table import FeatureTable
from habit.core.machine_learning.config_schemas import MLConfig
from habit.domain.evaluation import MetricRegistry
from habit.domain.pipeline import TablePipeline
from habit.recipes.modeling import (
    CVResult,
    ModelResult,
    PredictionResult,
    cross_validate,
    predict_model,
    train_model,
)
from habit.spec.legacy import LegacyConfigAdapter
from habit.spec.specs import MLSpec
from habit.utils.log_utils import setup_logger, stop_queue_listener

#: v1 pipeline artefact written under the output directory.
_V1_PIPELINE_NAME = "model.habitpipeline"

#: v0.1 hold-out workflow results JSON (minimal compatibility stub).
_LEGACY_HOLDOUT_RESULTS = "ml_standard_results.json"

#: v0.1 K-fold workflow results JSON (minimal compatibility stub).
_LEGACY_KFOLD_RESULTS = "ml_kfold_results.json"

#: File suffix of the versioned v1 pipeline artefact.
_V1_PIPELINE_SUFFIX = ".habitpipeline"


def run_ml(config_path: str, mode: Optional[str] = None) -> None:
    """
    Run the ML pipeline (training or prediction).

    Args:
        config_path: Path to configuration YAML file.
        mode: Optional CLI override for ``train`` or ``predict``. When
            ``None``, keep the YAML ``run_mode`` (schema default is train).
    """
    if mode is not None and mode not in ("train", "predict"):
        exit_with_error(
            f"Error: invalid --mode {mode!r} (expected 'train' or 'predict')."
        )

    config = load_config_or_exit(MLConfig, config_path)
    click.echo(f"Loaded configuration from: {config_path}")
    config = apply_ml_mode_override(config, mode)
    effective_mode: str = config.run_mode

    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_filename = (
        "processing.log" if effective_mode == "train" else "prediction.log"
    )
    logger_name = "cli.ml" if effective_mode == "train" else "cli.ml.predict"
    logger = setup_logger(
        name=logger_name,
        output_dir=output_dir,
        log_filename=log_filename,
        level=logging.INFO,
    )
    logger.info(
        "Starting machine learning pipeline (mode=%s) with config: %s",
        effective_mode,
        config_path,
    )
    logger.info("Full configuration: %s", config.model_dump())

    click.echo(
        f"Initialising machine learning pipeline (mode={effective_mode})..."
    )
    try:
        if effective_mode == "predict":
            _run_predict(config, logger)
        else:
            _run_train_model(config, logger)
    except ValueError as exc:
        logger.error("Workflow failed: %s", exc, exc_info=True)
        exit_with_error(f"Error during {effective_mode}: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Workflow failed: %s", exc, exc_info=True)
        exit_with_error(f"Error during {effective_mode}: {exc}")
    finally:
        stop_queue_listener()

    if effective_mode == "train":
        echo_success("Training completed successfully!")
    else:
        echo_success("Prediction completed successfully!")


def run_kfold(config_file: str) -> None:
    """
    Run K-fold cross-validation pipeline.

    Args:
        config_file: Path to configuration YAML file.
    """
    config = load_config_or_exit(MLConfig, config_file)
    click.echo(f"Loaded configuration from: {config_file}")

    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="cli.kfold",
        output_dir=output_dir,
        log_filename="kfold_cv.log",
        level=logging.INFO,
    )
    logger.info("Starting K-fold cross-validation with config: %s", config_file)
    logger.info("Full configuration: %s", config.model_dump())
    click.echo(f"Starting K-fold cross-validation with config: {config_file}")

    try:
        click.echo("Initialising machine learning pipeline...")
        _run_cross_validate(config, logger)
    except ValueError as exc:
        exit_with_error(f"Error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.error("K-fold failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")
    finally:
        stop_queue_listener()

    echo_success("K-fold cross-validation completed successfully!")


def _run_train_model(config: MLConfig, logger: logging.Logger) -> None:
    """
    Fit one pipeline through the v1 ``train_model`` recipe.

    The v0.1 hold-out workflow always split before fitting, so the split
    design is stated here from the config: ``custom`` follows the id files,
    ``stratified``/``random`` draw a ``test_size`` hold-out.

    Args:
        config: Validated v0.1 ML configuration (``run_mode='train'``).
        logger: Run logger.
    """
    document = _translate_document(config, workflow="model")
    _log_translation_warnings(document, logger)
    spec = _spec_from_document(document)
    table = _load_feature_table(config, logger)
    split_kwargs = _resolve_split_kwargs(config, table, logger)
    logger.info(
        "Running v1 recipe train_model (classifier=%s, rows=%d, features=%d, "
        "split=%s)",
        spec.classifier.name,
        len(table.frame),
        len(table.feature_columns),
        config.split_method,
    )
    result = train_model(table, spec, seed=config.random_state, **split_kwargs)
    _save_model_result(result, table, config, logger)


def _run_cross_validate(config: MLConfig, logger: logging.Logger) -> None:
    """
    Estimate generalisation through the v1 ``cross_validate`` recipe.

    Args:
        config: Validated v0.1 ML configuration (``run_mode='train'``).
        logger: Run logger.
    """
    document = _translate_document(config, workflow="cv")
    _log_translation_warnings(document, logger)
    spec = _spec_from_document(document)
    table = _load_feature_table(config, logger)
    n_splits = int(config.n_splits)
    logger.info(
        "Running v1 recipe cross_validate (classifier=%s, n_splits=%d, rows=%d)",
        spec.classifier.name,
        n_splits,
        len(table.frame),
    )
    result = cross_validate(table, spec, n_splits=n_splits, seed=config.random_state)
    _save_cv_result(result, config, logger)


def _run_predict(config: MLConfig, logger: logging.Logger) -> None:
    """
    Route predict mode by pipeline artefact format.

    A ``.habitpipeline`` is the versioned v1 artefact and runs through the
    v1 ``predict_model`` recipe; anything else (the v0.1
    ``*_final_pipeline.pkl`` pickles) stays on the v0.1 engine, which is the
    only loader that understands those opaque pickles.

    Args:
        config: Validated v0.1 ML configuration (``run_mode='predict'``).
        logger: Run logger.
    """
    pipeline_path = str(config.pipeline_path or "")
    if pipeline_path.endswith(_V1_PIPELINE_SUFFIX):
        _run_v1_predict(config, logger)
        return
    logger.info(
        "Predict mode delegates to the v0.1 engine via "
        "habit.api.machine_learning.run_ml (legacy pickle artefact)."
    )
    from habit.api.machine_learning import run_ml as api_run_ml

    api_run_ml(config, logger=logger, output_dir=str(config.output))


def _run_v1_predict(config: MLConfig, logger: logging.Logger) -> None:
    """
    Apply a v1 ``.habitpipeline`` through the ``predict_model`` recipe.

    Args:
        config: Validated v0.1 ML configuration (``run_mode='predict'``).
        logger: Run logger.

    Raises:
        ValueError: When ``pipeline_path`` does not name an existing file.
    """
    pipeline_path = Path(str(config.pipeline_path or ""))
    if not pipeline_path.is_file():
        raise ValueError(f"Saved pipeline not found: {pipeline_path}")
    pipeline = TablePipeline.load(pipeline_path)
    table = _load_predict_table(config, logger)
    logger.info(
        "Running v1 recipe predict_model (model=%s, rows=%d, features=%d)",
        pipeline.model.spec.name,
        len(table.frame),
        len(table.feature_columns),
    )
    result = predict_model(pipeline, table)
    _save_prediction_result(result, pipeline, table, config, logger)


def _resolve_split_kwargs(
    config: MLConfig, table: FeatureTable, logger: logging.Logger
) -> Dict[str, Any]:
    """
    Turn the v0.1 split settings into ``train_model`` keyword arguments.

    ``custom`` follows the id files exactly; ids the table does not have are
    dropped with a warning, mirroring the v0.1 ``SplitStrategy`` (which
    silently intersected the files with the data index). ``stratified`` and
    ``random`` both draw a ``test_size`` hold-out, stratified on the outcome
    only for ``stratified``.

    Args:
        config: Validated v0.1 ML configuration.
        table: The assembled feature table (id source for the custom split).
        logger: Run logger for the missing-id warnings.

    Returns:
        Keyword arguments for :func:`train_model`.

    Raises:
        ValueError: When ``split_method='custom'`` lacks an id file.
    """
    split_method = str(config.split_method)
    if split_method == "custom":
        if not config.train_ids_file or not config.test_ids_file:
            raise ValueError(
                "Custom split requires train_ids_file and test_ids_file"
            )
        row_ids = set(_table_row_ids(table))
        train_ids = _filter_split_ids(
            _read_split_ids(str(config.train_ids_file)), row_ids, "train", logger
        )
        test_ids = _filter_split_ids(
            _read_split_ids(str(config.test_ids_file)), row_ids, "test", logger
        )
        return {"train_ids": train_ids, "test_ids": test_ids}
    return {"test_size": float(config.test_size), "stratify": split_method == "stratified"}


def _table_row_ids(table: FeatureTable) -> List[str]:
    """Return one string id per row (single identifier column in cmd_ml)."""
    id_column = table.id_columns[0]
    return [str(value) for value in table.frame[id_column]]


def _read_split_ids(path: str) -> List[str]:
    """
    Read a v0.1 id file: JSON list, comma-separated, or one id per line.

    Args:
        path: Id file path (already resolved by the config loader).

    Returns:
        The ids in file order, as strings.
    """
    content = Path(path).read_text(encoding="utf-8").strip()
    if content.startswith("["):
        return [str(item) for item in json.loads(content)]
    if "," in content:
        return [item.strip() for item in content.split(",")]
    return [line.strip() for line in content.split("\n") if line.strip()]


def _filter_split_ids(
    ids: Sequence[str], row_ids: set, side: str, logger: logging.Logger
) -> List[str]:
    """
    Keep the ids present in the table, warning about the rest (v0.1 parity).

    Args:
        ids: Ids from the split file.
        row_ids: The table's row ids.
        side: ``"train"`` or ``"test"`` (for the warning text).
        logger: Run logger.

    Returns:
        The valid ids in file order.

    Raises:
        ValueError: When no id survives the filtering.
    """
    valid = [row_id for row_id in ids if row_id in row_ids]
    missing = [row_id for row_id in ids if row_id not in row_ids]
    if missing:
        logger.warning(
            "Custom split: %d %s IDs not found in data index. Sample: %s",
            len(missing),
            side,
            missing[:10],
        )
    if not valid:
        raise ValueError(
            f"Custom split: no {side} IDs from the id file are present in "
            "the input table."
        )
    return valid


def _load_predict_table(config: MLConfig, logger: logging.Logger) -> FeatureTable:
    """
    Assemble the predict-mode table; the outcome is optional.

    Predict uses the FIRST ``input`` entry (v0.1 inference reads one table).
    The configured ``label_col`` becomes the outcome only when the column
    actually exists -- a predict CSV is legitimately unlabelled.

    Args:
        config: Validated v0.1 ML configuration (``run_mode='predict'``).
        logger: Run logger.

    Returns:
        The prediction table, with a binary outcome when labels are present.

    Raises:
        ValueError: When ``input`` is empty or the file/columns are missing.
    """
    if not config.input:
        raise ValueError("ML config 'input' must contain at least one table.")
    entry = config.input[0]
    path = Path(entry.path)
    if not path.is_file():
        raise ValueError(f"Input table not found: {path}")
    subj_col = entry.subject_id_col
    frame = pd.read_csv(path, dtype={subj_col: str})
    if subj_col not in frame.columns:
        raise ValueError(f"Input table {path} is missing column {subj_col!r}.")
    frame[subj_col] = frame[subj_col].astype(str)

    label_col = entry.label_col if entry.label_col in frame.columns else None
    if label_col is None:
        logger.info(
            "Predict input has no label column %r; running unlabelled inference.",
            entry.label_col,
        )
    features = list(entry.features or [])
    available = [
        column for column in frame.columns if column not in {subj_col, label_col}
    ]
    selected = features if features else available
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"Input table {path} is missing feature columns {missing}.")
    if not selected:
        raise ValueError("No feature columns remain in the predict input table.")

    outcome = (
        BinaryOutcome(column=label_col, positive_label=1)
        if label_col is not None
        else None
    )
    return FeatureTable(
        frame=frame[[subj_col, *selected] + ([label_col] if label_col else [])],
        id_columns=(subj_col,),
        feature_columns=tuple(selected),
        outcome=outcome,
        provenance=Provenance.source("commands.cmd_ml"),
    )


def _save_prediction_result(
    result: PredictionResult,
    pipeline: TablePipeline,
    table: FeatureTable,
    config: MLConfig,
    logger: logging.Logger,
) -> None:
    """
    Persist a v1 prediction run with v0.1-friendly filenames.

    ``predictions.csv`` is the input frame plus the configured output
    columns; ``metrics.json`` appears only when ``evaluate=true`` AND the
    input carried ground-truth labels (the v0.1 inference rule).

    Args:
        result: Outcome of :func:`predict_model`.
        pipeline: The fitted pipeline (for optional evaluation).
        table: The predicted table (input frame source).
        config: Validated v0.1 ML configuration.
        logger: Run logger.
    """
    out_dir = Path(config.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = table.frame.copy()
    frame[config.output_label_col] = frame[table.id_columns[0]].map(
        result.predictions
    )
    if result.probabilities is not None:
        frame[config.output_prob_col] = _probability_output_column(
            result.probabilities, frame[table.id_columns[0]]
        )
    predictions_path = out_dir / "predictions.csv"
    frame.to_csv(predictions_path, index=False)
    logger.info("Predictions written to %s", predictions_path)
    result.manifest.to_json(out_dir / "run_manifest.json")

    if bool(config.evaluate) and table.outcome is not None:
        # The composed pipeline Spec carries no metric panel; the predict-time
        # evaluation panel is a CLI concern, built straight from the registry.
        metrics = tuple(
            MetricRegistry.create(name) for name in ("accuracy", "auc")
        )
        panel = pipeline.evaluate(table, metrics)
        (out_dir / "metrics.json").write_text(
            json.dumps(panel, indent=2), encoding="utf-8"
        )
        logger.info("Evaluation metrics written to %s", out_dir / "metrics.json")


def _probability_output_column(
    probabilities: pd.DataFrame, id_series: pd.Series
) -> List[Any]:
    """
    Reduce the per-class probability frame to one output cell per row.

    Binary classifiers report the positive-class probability (column ``"1"``
    for a 0/1 outcome, else the last class column, mirroring
    :meth:`TablePipeline.evaluate`); wider frames keep the full per-class
    list, the CSV-friendly multiclass form v0.1 used.

    Args:
        probabilities: Per-class probability frame indexed by row id; class
            columns are string labels.
        id_series: Row ids in output order.

    Returns:
        One scalar or list per row, in ``id_series`` order.
    """
    columns = list(probabilities.columns)
    indexed = probabilities.copy()
    indexed.index = [str(value) for value in indexed.index]
    ids = [str(value) for value in id_series]
    if len(columns) == 2:
        positive = "1" if "1" in columns else columns[-1]
        series = indexed[positive]
        return [series.get(row_id) for row_id in ids]
    return [
        [float(value) for value in indexed.loc[row_id].tolist()]
        if row_id in indexed.index
        else None
        for row_id in ids
    ]


def _translate_document(config: MLConfig, *, workflow: str) -> Dict[str, Any]:
    """
    Translate the validated v0.1 config into the v1 document model.

    Args:
        config: Validated v0.1 ML configuration.
        workflow: ``"model"`` for hold-out train or ``"cv"`` for K-fold.

    Returns:
        The translated document with ``spec``/``data``/``legacy`` sections.
    """
    translation = LegacyConfigAdapter().translate(config.model_dump(), workflow)
    return translation.document


def _log_translation_warnings(
    document: Mapping[str, Any], logger: logging.Logger
) -> None:
    """
    Surface adapter warnings so multi-model sweeps are visible in the log.

    Args:
        document: Translated v1 document (warnings are not stored on it).
        logger: Run logger.
    """
    legacy = document.get("legacy") or {}
    models_block = legacy.get("models")
    if isinstance(models_block, Mapping) and len(models_block) > 1:
        logger.warning(
            "v0 trained %d models in one run; the v1 MLSpec describes ONE "
            "classifier -- only the first entry is fitted. The full block "
            "is preserved under legacy.",
            len(models_block),
        )


def _spec_from_document(document: Mapping[str, Any]) -> MLSpec:
    """
    Parse the ``spec`` section of a translated document.

    Args:
        document: Translated v1 document.

    Returns:
        The modelling definition to pass to a recipe.

    Raises:
        ValueError: When the document carries no spec (train configs must).
    """
    spec_payload = document.get("spec")
    if spec_payload is None:
        raise ValueError(
            "The train config translated to no ML spec; train configs must "
            "carry a non-empty 'models' block."
        )
    return MLSpec.from_dict(spec_payload)


def _load_feature_table(config: MLConfig, logger: logging.Logger) -> FeatureTable:
    """
    Assemble one labelled :class:`FeatureTable` from the v0.1 ``input`` list.

    Paths are already resolved by :meth:`MLConfig.from_file`. Multi-table
    inputs are outer-joined on the subject-id column of the first table;
    collision columns are prefixed with each table's ``name`` when set.

    Args:
        config: Validated v0.1 ML configuration.
        logger: Run logger for skip warnings.

    Returns:
        The merged feature table with an inferred binary outcome.

    Raises:
        ValueError: When ``input`` is empty or required columns are missing.
    """
    if not config.input:
        raise ValueError("ML config 'input' must contain at least one table.")

    subject_id_col: Optional[str] = None
    label_col: Optional[str] = None
    merged: Optional[pd.DataFrame] = None

    for index, entry in enumerate(config.input):
        path = Path(entry.path)
        if not path.is_file():
            raise ValueError(f"Input table not found: {path}")

        subj_col = entry.subject_id_col
        lbl_col = entry.label_col
        if not subj_col or not lbl_col:
            raise ValueError(
                f"subject_id_col and label_col are required for input table {path}."
            )

        logger.info(
            "Reading %s (subject=%s, label=%s)", path, subj_col, lbl_col
        )
        frame = pd.read_csv(path, dtype={subj_col: str})
        if subj_col not in frame.columns or lbl_col not in frame.columns:
            raise ValueError(
                f"Input table {path} is missing required columns "
                f"{subj_col!r} and/or {lbl_col!r}."
            )
        frame[subj_col] = frame[subj_col].astype(str)

        if subject_id_col is None:
            subject_id_col = subj_col
            label_col = lbl_col

        features = list(entry.features or [])
        available = [column for column in frame.columns if column not in {subj_col, lbl_col}]
        selected = features if features else available
        prefix = str(entry.name or "").strip()
        rename = {
            column: f"{prefix}{column}"
            for column in selected
            if prefix and column in frame.columns
        }
        subset = frame.set_index(subj_col)[selected].rename(columns=rename)

        if merged is None:
            merged = subset
            label_series = frame.set_index(subj_col)[lbl_col]
        else:
            overlap = merged.columns.intersection(subset.columns)
            if len(overlap):
                safe_prefix = prefix if prefix.endswith("_") else f"{prefix or f'input{index}'}_"
                collision_map = {
                    column: f"{safe_prefix}{column}" for column in overlap
                }
                subset = subset.rename(columns=collision_map)
                logger.warning(
                    "Detected overlapping feature columns in %s; auto-renamed "
                    "%d columns.",
                    path,
                    len(collision_map),
                )
            merged = merged.join(subset, how="outer")
            label_series = label_series.combine_first(
                frame.set_index(subj_col)[lbl_col]
            )

    if merged is None or subject_id_col is None or label_col is None:
        raise ValueError("Failed to assemble a feature table from config.input.")

    common_index = merged.index.intersection(label_series.index)
    merged = merged.loc[common_index].reset_index()
    merged[label_col] = label_series.loc[common_index].to_numpy()
    merged = merged.rename(columns={subject_id_col: subject_id_col})

    feature_columns = tuple(
        column
        for column in merged.columns
        if column not in {subject_id_col, label_col}
    )
    if not feature_columns:
        raise ValueError("No feature columns remain after assembling input tables.")

    provenance = Provenance.source("commands.cmd_ml")
    return FeatureTable(
        frame=merged,
        id_columns=(subject_id_col,),
        feature_columns=feature_columns,
        outcome=BinaryOutcome(column=label_col, positive_label=1),
        provenance=provenance,
    )


def _save_model_result(
    result: ModelResult,
    table: FeatureTable,
    config: MLConfig,
    logger: logging.Logger,
) -> None:
    """
    Persist a hold-out train result with v0.1-friendly filenames.

    ``metrics.json`` is ``{"train": {...}, "test": {...}}`` when the recipe
    ran a hold-out split (the v0.1 workflow always did) and the flat train
    panel otherwise. The legacy ``ml_standard_results.json`` stub mirrors
    the same split, one section per side.

    Args:
        result: Outcome of :func:`train_model`.
        table: The table the recipe trained on (for selected-feature export).
        config: Validated v0.1 ML configuration.
        logger: Run logger.
    """
    out_dir = Path(config.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = result.pipeline.model.spec.name
    metrics_payload: Dict[str, Any]
    legacy_payload: Dict[str, Any]
    if result.test_metrics is not None:
        metrics_payload = {
            "train": dict(result.train_metrics),
            "test": dict(result.test_metrics),
        }
        legacy_payload = {
            "train": {model_name: {"metrics": dict(result.train_metrics)}},
            "test": {model_name: {"metrics": dict(result.test_metrics)}},
        }
    else:
        metrics_payload = dict(result.train_metrics)
        legacy_payload = {
            "train": {model_name: {"metrics": dict(result.train_metrics)}}
        }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    result.manifest.to_json(out_dir / "run_manifest.json")
    (out_dir / _LEGACY_HOLDOUT_RESULTS).write_text(
        json.dumps(legacy_payload, indent=2), encoding="utf-8"
    )

    selected = list(result.pipeline.transform(table).feature_columns)
    (out_dir / "selected_features.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )

    if config.is_save_model:
        pipeline_path = result.pipeline.save(out_dir / _V1_PIPELINE_NAME)
        logger.info("Saved fitted pipeline to %s", pipeline_path)

    logger.info("Metrics written to %s", metrics_path)


def _save_cv_result(
    result: CVResult, config: MLConfig, logger: logging.Logger
) -> None:
    """
    Persist a cross-validation result with v0.1-friendly filenames.

    Args:
        result: Outcome of :func:`cross_validate`.
        config: Validated v0.1 ML configuration.
        logger: Run logger.
    """
    out_dir = Path(config.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "mean": dict(result.mean_metrics),
        "std": dict(result.std_metrics),
        "folds": [dict(panel) for panel in result.fold_metrics],
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    result.manifest.to_json(out_dir / "run_manifest.json")

    model_name = result.manifest.spec_payload["classifier"]["name"]
    legacy_payload = {
        "aggregated": {model_name: dict(result.mean_metrics)},
        "folds": [
            {model_name: dict(panel)} for panel in result.fold_metrics
        ],
    }
    (out_dir / _LEGACY_KFOLD_RESULTS).write_text(
        json.dumps(legacy_payload, indent=2), encoding="utf-8"
    )

    if config.is_save_model and result.pipelines:
        model_dir = out_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for fold_index, pipeline in enumerate(result.pipelines):
            destination = model_dir / f"{model_name}_fold{fold_index}_pipeline.habitpipeline"
            pipeline.save(destination)
            logger.info("Saved fold %d pipeline to %s", fold_index, destination)

    logger.info("Cross-validation metrics written to %s", metrics_path)
