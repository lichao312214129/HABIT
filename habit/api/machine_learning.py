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
"""Public machine-learning workflow API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.provenance import create_run_manifest, write_run_manifest

if TYPE_CHECKING:
    from habit.schemas.workflows.ml import (
        MLConfig,
        ModelComparisonConfig,
        TestRetestConfig,
    )

__all__ = [
    "MLConfig",
    "ModelComparisonConfig",
    "TestRetestConfig",
    "apply_ml_mode_override",
    "run_ml",
    "run_kfold",
    "run_model_comparison",
]


def __getattr__(name: str) -> Any:
    if name in {"MLConfig", "ModelComparisonConfig", "TestRetestConfig"}:
        from habit.schemas.workflows import ml as ml_schemas

        return getattr(ml_schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def apply_ml_mode_override(
    config: Union["MLConfig", Mapping[str, Any]],
    mode: Optional[str],
) -> "MLConfig":
    """
    Validate a config object or mapping before applying an ML mode override.

    Args:
        config: Validated ML config or a dictionary accepted by its schema.
        mode: Optional ``"train"`` or ``"predict"`` override.

    Returns:
        Validated ML configuration with the requested mode.
    """
    from habit.compat.legacy_core import apply_ml_mode_override_core
    from habit.schemas.workflows.ml import MLConfig

    return apply_ml_mode_override_core(coerce_config(config, MLConfig), mode)


def run_ml(
    config: Union["MLConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> WorkflowResult[Any]:
    """
    Run holdout ML train or predict workflow from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.schemas.workflows.ml.MLConfig`.
        logger: Optional logger passed to the core runner.
        output_dir: Optional output directory override.

    Returns:
        Structured workflow output in ``data`` and its output directory in
        ``artifacts``.
    """
    from habit.compat.legacy_core import run_ml_from_config
    from habit.schemas.workflows.ml import MLConfig

    validated_config = coerce_config(config, MLConfig)
    resolved_output_dir = output_dir or validated_config.output
    result = run_ml_from_config(
        validated_config,
        logger=logger,
        output_dir=resolved_output_dir,
    )
    metrics = getattr(result, "metrics", {})
    manifest = create_run_manifest(
        "machine_learning",
        validated_config,
        metadata={"run_mode": validated_config.run_mode},
    )
    manifest_path = write_run_manifest(manifest, resolved_output_dir)
    return WorkflowResult(
        data=result,
        output_dir=Path(resolved_output_dir),
        metrics=metrics,
        metadata={
            "run_mode": validated_config.run_mode,
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )


def run_kfold(
    config: Union["MLConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> WorkflowResult[Any]:
    """
    Run k-fold cross-validation from a validated train-mode config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.schemas.workflows.ml.MLConfig`
            (``run_mode`` must be ``train``).
        logger: Optional logger passed to the core runner.
        output_dir: Optional output directory override.

    Returns:
        Structured K-fold result in ``data`` and its output directory in
        ``artifacts``.
    """
    from habit.compat.legacy_core import run_kfold_from_config
    from habit.schemas.workflows.ml import MLConfig

    validated_config = coerce_config(config, MLConfig)
    resolved_output_dir = output_dir or validated_config.output
    result = run_kfold_from_config(
        validated_config,
        logger=logger,
        output_dir=resolved_output_dir,
    )
    manifest = create_run_manifest(
        "machine_learning_kfold",
        validated_config,
        metadata={"run_mode": validated_config.run_mode},
    )
    manifest_path = write_run_manifest(manifest, resolved_output_dir)
    return WorkflowResult(
        data=result,
        output_dir=Path(resolved_output_dir),
        metadata={
            "run_mode": validated_config.run_mode,
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )


def run_model_comparison(
    config: Union["ModelComparisonConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> WorkflowResult[Mapping[str, Any]]:
    """
    Compare multiple trained models from a validated config object.

    Delegates to :func:`habit.recipes.comparison.compare_models` (v1 domain +
    ``habit.viz``). The v0.1 ``ModelComparison`` engine is no longer invoked.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.schemas.workflows.ml.ModelComparisonConfig`.
        logger: Optional logger passed to the recipe.
        output_dir: Optional output directory override.

    Returns:
        Model-comparison metrics in ``data`` and written artefact paths in
        ``artifacts``.
    """
    from habit.recipes.comparison import compare_models

    return compare_models(config, logger=logger, output_dir=output_dir)
