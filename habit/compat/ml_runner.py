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
"""Machine-learning workflow runners (L1 compat).

Hold-out train, K-fold, predict, and model comparison route through the L4
tabular recipes when the pipeline artefact is a v1 ``.habitpipeline`` archive.
Legacy v0.1 pickle pipelines and model-comparison orchestration still delegate
to the v0.1 engine through lazy imports in this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from habit.utils.log_utils import get_module_logger

__all__ = [
    "run_kfold_from_config",
    "run_ml_from_config",
    "run_model_comparison_from_config",
]

_LOG = get_module_logger(__name__)
_V1_PIPELINE_SUFFIX = ".habitpipeline"


@dataclass(frozen=True)
class _RecipeWorkflowResult:
    """Minimal stand-in for v0.1 workflow result objects at the API boundary."""

    metrics: Mapping[str, Any]


def run_ml_from_config(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> Any:
    """
    Run hold-out ML training or prediction.

    Args:
        config: Validated :class:`~habit.schemas.workflows.ml.MLConfig`.
        logger: Optional run logger.
        output_dir: Optional output directory override.

    Returns:
        v0.1 structured result for legacy paths, or a minimal shim for v1 recipes.
    """
    log = logger or _LOG
    if output_dir is not None:
        config.output = output_dir

    pipeline_path = str(getattr(config, "pipeline_path", "") or "")
    if str(config.run_mode) == "predict" and not pipeline_path.endswith(
        _V1_PIPELINE_SUFFIX
    ):
        return _run_legacy_ml(config, logger=log, output_dir=output_dir)

    if str(config.run_mode) == "predict":
        return _run_v1_predict(config, logger=log)

    return _run_v1_train(config, logger=log)


def run_kfold_from_config(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> Any:
    """
    Run K-fold cross-validation through the v1 ``cross_validate`` recipe.

    Args:
        config: Validated ML configuration (``run_mode`` must be ``train``).
        logger: Optional run logger.
        output_dir: Optional output directory override.

    Raises:
        ValueError: When ``run_mode`` is not ``train``.
    """
    if str(config.run_mode) != "train":
        raise ValueError("K-fold cross-validation requires run_mode='train'.")

    log = logger or _LOG
    if output_dir is not None:
        config.output = output_dir

    from habit.recipes.modeling import cross_validate
    from habit.recipes.yaml_runner import (
        _load_feature_table,
        _ml_spec_from_document,
        _save_cv_result,
    )
    from habit.spec.legacy import LegacyConfigAdapter

    document = LegacyConfigAdapter().translate(config.model_dump(), "cv").document
    spec = _ml_spec_from_document(document)
    table = _load_feature_table(config, logger=log)
    log.info(
        "Running v1 recipe cross_validate (classifier=%s, n_splits=%d)",
        spec.classifier.name,
        int(config.n_splits),
    )
    result = cross_validate(
        table,
        spec,
        n_splits=int(config.n_splits),
        seed=config.random_state,
    )
    _save_cv_result(result, config, logger=log)
    return _RecipeWorkflowResult(metrics=dict(result.mean_metrics))


def run_model_comparison_from_config(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> Any:
    """
    Run multi-model comparison (plots, metrics, DeLong tests).

    Thin legacy entry point: forwards to the v1
    :func:`habit.recipes.comparison.compare_models` recipe so callers that still
    import this helper do not resurrect the v0.1 ``ModelComparison`` engine.

    Args:
        config: Validated model-comparison configuration.
        logger: Optional run logger.
        output_dir: Optional output directory override.

    Returns:
        The metrics store mapping from the recipe ``WorkflowResult.data``.
    """
    from habit.recipes.comparison import compare_models

    result = compare_models(config, logger=logger, output_dir=output_dir)
    return result.data


def _run_v1_train(config: Any, *, logger: logging.Logger) -> _RecipeWorkflowResult:
    """Fit one hold-out pipeline through ``train_model`` and persist artefacts."""
    from habit.recipes.modeling import train_model
    from habit.recipes.yaml_runner import (
        _load_feature_table,
        _ml_spec_from_document,
        _resolve_ml_split_kwargs,
        _save_model_result,
    )
    from habit.spec.legacy import LegacyConfigAdapter

    document = LegacyConfigAdapter().translate(config.model_dump(), "model").document
    spec = _ml_spec_from_document(document)
    table = _load_feature_table(config, logger=logger)
    split_kwargs = _resolve_ml_split_kwargs(config, table, logger=logger)
    log = logger
    log.info(
        "Running v1 recipe train_model (classifier=%s, rows=%d, features=%d)",
        spec.classifier.name,
        len(table.frame),
        len(table.feature_columns),
    )
    result = train_model(table, spec, seed=config.random_state, **split_kwargs)
    _save_model_result(result, table, config, logger=log)
    metrics: Dict[str, Any] = dict(result.train_metrics)
    if result.test_metrics is not None:
        metrics = {"train": dict(result.train_metrics), "test": dict(result.test_metrics)}
    return _RecipeWorkflowResult(metrics=metrics)


def _run_v1_predict(config: Any, *, logger: logging.Logger) -> _RecipeWorkflowResult:
    """Apply a v1 ``.habitpipeline`` through ``predict_model``."""
    from habit.domain.pipeline import TablePipeline
    from habit.recipes.modeling import predict_model

    pipeline_path = Path(str(config.pipeline_path or ""))
    if not pipeline_path.is_file():
        raise ValueError(f"Saved pipeline not found: {pipeline_path}")
    pipeline = TablePipeline.load(pipeline_path)
    table = _load_predict_table(config, logger=logger)
    logger.info(
        "Running v1 recipe predict_model (model=%s, rows=%d)",
        pipeline.model.spec.name,
        len(table.frame),
    )
    result = predict_model(pipeline, table)
    _save_prediction_result(result, table, config, logger=logger)
    return _RecipeWorkflowResult(metrics={"predictions": len(result.predictions)})


def _load_predict_table(config: Any, logger: logging.Logger) -> Any:
    """
    Assemble the predict-mode table; the outcome column is optional.

    Predict uses the first ``input`` entry, matching the v0.1 inference rule.
    """
    import pandas as pd

    from habit.contracts.outcome import BinaryOutcome
    from habit.contracts.provenance import Provenance
    from habit.contracts.table import FeatureTable

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
    columns = [subj_col, *selected] + ([label_col] if label_col else [])
    return FeatureTable(
        frame=frame[columns],
        id_columns=(subj_col,),
        feature_columns=tuple(selected),
        outcome=outcome,
        provenance=Provenance.source("compat.ml_runner"),
    )


def _save_prediction_result(
    result: Any,
    table: Any,
    config: Any,
    *,
    logger: logging.Logger,
) -> None:
    """Write ``predictions.csv`` for a v1 predict run (v0.1-compatible name)."""
    out_dir = Path(str(config.output))
    out_dir.mkdir(parents=True, exist_ok=True)
    output = table.frame.copy()
    label_col = getattr(config, "output_label_col", None) or "prediction"
    prob_col = getattr(config, "output_prob_col", None) or "probability"
    output[label_col] = result.predictions.to_numpy()
    if result.probabilities is not None and not result.probabilities.empty:
        positive_index = int(getattr(config, "binary_positive_class_index", 1))
        if positive_index < result.probabilities.shape[1]:
            output[prob_col] = result.probabilities.iloc[:, positive_index].to_numpy()
    destination = out_dir / "predictions.csv"
    output.to_csv(destination, index=False)
    logger.info("Saved predictions to %s", destination)


def _run_legacy_ml(
    config: Any,
    *,
    logger: Optional[logging.Logger],
    output_dir: Optional[str],
) -> Any:
    """Reject legacy pickle pipelines with a migration hint."""
    from habit.utils.deprecation import build_deprecation_message
    import warnings

    from habit.utils.deprecation import HabitDeprecationWarning

    message = build_deprecation_message(
        "legacy pickle ML pipelines",
        "1.0.0",
        alternative=(
            "re-train and save a v1 `.habitpipeline` archive, then run predict "
            "with that path"
        ),
        removed_in="1.2.0",
    )
    warnings.warn(message, HabitDeprecationWarning, stacklevel=2)
    raise ValueError(
        f"Legacy pickle pipeline {config.pipeline_path!r} is no longer supported. "
        "Train a v1 `.habitpipeline` model and point `pipeline_path` to that file."
    )
