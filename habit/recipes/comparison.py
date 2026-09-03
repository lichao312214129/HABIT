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
"""L4 model-comparison recipe (``habit compare``).

Assembles domain merge / :func:`evaluate_comparison` with
:mod:`habit.recipes.comparison_reporting`. The v0.1 ML comparison
engine facade is not imported on this path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.provenance import create_run_manifest, write_run_manifest
from habit.evaluation.comparison import (
    PredictionSource,
    evaluate_comparison,
    merge_prediction_frames,
)
from habit.evaluation.statistics import DelongResult, delong_test
from habit.exceptions import HABITAPIError
from habit.recipes.comparison_reporting import write_comparison_artifacts
from habit.schemas.workflows.ml import ModelComparisonConfig

__all__ = ["compare_models", "pairwise_delong_test"]


def compare_models(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> WorkflowResult[Mapping[str, Any]]:
    """
    Compare multiple trained models from a validated comparison config.

    Args:
        config: Validated :class:`ModelComparisonConfig` or compatible mapping.
        logger: Optional run logger (CLI attaches ``processing.log``).
        output_dir: Optional output directory override.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with the metrics store in
        ``data`` and written artefact paths in ``artifacts``.
    """
    validated = coerce_config(config, ModelComparisonConfig)
    log = logger or logging.getLogger("habit.recipes.comparison")
    destination = Path(output_dir or validated.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    log.info("Loading prediction CSVs for model comparison")
    merged = _load_and_merge(validated, logger=log)

    result = evaluate_comparison(
        merged,
        split_enabled=bool(validated.split.enabled),
        basic_metrics=bool(validated.metrics.basic_metrics.enabled),
        youden_metrics=bool(validated.metrics.youden_metrics.enabled),
        target_metrics=bool(validated.metrics.target_metrics.enabled),
        targets=dict(validated.metrics.target_metrics.targets or {}),
        delong_test=bool(validated.delong_test.enabled),
    )
    log.info(
        "Evaluated %d model(s) across groups=%s (training_group=%s)",
        len(merged.model_names),
        [str(g) for g in result.groups.keys()],
        result.training_group,
    )

    artifacts = write_comparison_artifacts(
        result,
        destination,
        visualization=validated.visualization,
        merged_save_name=validated.merged_data.save_name or "combined_predictions.csv",
        write_merged=bool(validated.merged_data.enabled),
        delong_save_name=validated.delong_test.save_name or "delong_results.json",
        write_delong=bool(validated.delong_test.enabled),
        write_metrics=bool(result.metrics),
        split_enabled=bool(validated.split.enabled),
        logger=log,
    )

    manifest = create_run_manifest("model_comparison", validated)
    manifest_path = write_run_manifest(manifest, str(destination))
    artifacts["habit_run_manifest"] = Path(manifest_path)

    log.info("Model comparison completed; artefacts under %s", destination)
    return WorkflowResult(
        data=dict(result.metrics),
        output_dir=destination,
        artifacts={key: Path(path) for key, path in artifacts.items()},
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
            "n_models": len(merged.model_names),
            "groups": [str(g) for g in result.groups.keys()],
            "training_group": (
                None
                if result.training_group is None
                else str(result.training_group)
            ),
        },
        run_id=manifest.run_id,
        manifest_path=Path(manifest_path),
    )


def pairwise_delong_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> DelongResult:
    """
    Compare two models' ROC AUCs on the same subjects (paired DeLong test).

    Args:
        y_true: Binary ground-truth labels (0/1), both classes present.
        scores_a: Probability-of-class-1 scores of the first model.
        scores_b: Probability-of-class-1 scores of the second model, aligned
            to ``scores_a``.

    Returns:
        Frozen :class:`~habit.evaluation.statistics.DelongResult`.
    """
    return delong_test(y_true, scores_a, scores_b)


def _load_and_merge(
    config: ModelComparisonConfig,
    *,
    logger: logging.Logger,
):
    """Read each configured CSV and merge into one prediction table."""
    sources = []
    for file_cfg in config.files_config:
        path = Path(file_cfg.path)
        if not path.is_file():
            raise HABITAPIError(
                f"compare_models: prediction file not found: {path}"
            )
        frame = pd.read_csv(path)
        model_name = file_cfg.model_name or path.stem
        spec = PredictionSource(
            model_name=str(model_name),
            subject_id_col=file_cfg.subject_id_col,
            label_col=file_cfg.label_col,
            prob_col=file_cfg.prob_col,
            pred_col=file_cfg.pred_col,
            split_col=file_cfg.split_col,
        )
        logger.info("Loaded %s (%d rows) as model %s", path, len(frame), model_name)
        sources.append((spec, frame))
    return merge_prediction_frames(sources)
