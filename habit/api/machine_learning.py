# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Public machine-learning workflow API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from habit.api.contracts import WorkflowResult, coerce_config

if TYPE_CHECKING:
    from habit.core.machine_learning.config_schemas import (
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
        from habit.core.machine_learning import config_schemas

        return getattr(config_schemas, name)
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
    from habit.core.machine_learning.config_schemas import MLConfig
    from habit.core.machine_learning.run import (
        apply_ml_mode_override as _apply_ml_mode_override,
    )

    return _apply_ml_mode_override(coerce_config(config, MLConfig), mode)


def run_ml(
    config: Union["MLConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> WorkflowResult[Any]:
    """
    Run holdout ML train or predict workflow from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.core.machine_learning.config_schemas.MLConfig`.
        logger: Optional logger passed to the core runner.
        output_dir: Optional output directory override.

    Returns:
        Structured workflow output in ``data`` and its output directory in
        ``artifacts``.
    """
    from habit.core.machine_learning.config_schemas import MLConfig
    from habit.core.machine_learning.run import run_ml_from_config

    validated_config = coerce_config(config, MLConfig)
    resolved_output_dir = output_dir or validated_config.output
    result = run_ml_from_config(
        validated_config,
        logger=logger,
        output_dir=resolved_output_dir,
    )
    metrics = getattr(result, "metrics", {})
    return WorkflowResult(
        data=result,
        output_dir=Path(resolved_output_dir),
        metrics=metrics,
        metadata={"run_mode": validated_config.run_mode},
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
            :class:`~habit.core.machine_learning.config_schemas.MLConfig`
            (``run_mode`` must be ``train``).
        logger: Optional logger passed to the core runner.
        output_dir: Optional output directory override.

    Returns:
        Structured K-fold result in ``data`` and its output directory in
        ``artifacts``.
    """
    from habit.core.machine_learning.config_schemas import MLConfig
    from habit.core.machine_learning.run import run_kfold_from_config

    validated_config = coerce_config(config, MLConfig)
    resolved_output_dir = output_dir or validated_config.output
    result = run_kfold_from_config(
        validated_config,
        logger=logger,
        output_dir=resolved_output_dir,
    )
    return WorkflowResult(
        data=result,
        output_dir=Path(resolved_output_dir),
        metadata={"run_mode": validated_config.run_mode},
    )


def run_model_comparison(
    config: Union["ModelComparisonConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[Mapping[str, Any]]:
    """
    Compare multiple trained models from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.core.machine_learning.config_schemas.ModelComparisonConfig`.
        logger: Optional logger passed to the core runner.

    Returns:
        Model-comparison metrics in ``data`` and the output directory in
        ``artifacts``.
    """
    from habit.core.machine_learning.config_schemas import ModelComparisonConfig
    from habit.core.machine_learning.run import run_model_comparison_from_config

    validated_config = coerce_config(config, ModelComparisonConfig)
    result = run_model_comparison_from_config(validated_config, logger=logger)
    return WorkflowResult(data=result, output_dir=validated_config.output_dir)
