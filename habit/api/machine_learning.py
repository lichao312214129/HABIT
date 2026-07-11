# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Public machine-learning workflow API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

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
    if name == "apply_ml_mode_override":
        from habit.core.machine_learning.run import apply_ml_mode_override

        return apply_ml_mode_override
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_ml(
    config: "MLConfig",
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run holdout ML train or predict workflow from a validated config object.

    Args:
        config: Loaded ML configuration.
        logger: Optional logger passed to the core runner.
        output_dir: Optional output directory override.
    """
    from habit.core.machine_learning.run import run_ml_from_config

    run_ml_from_config(config, logger=logger, output_dir=output_dir)


def run_kfold(
    config: "MLConfig",
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run k-fold cross-validation from a validated train-mode config object.

    Args:
        config: Loaded ML configuration (``run_mode`` must be ``train``).
        logger: Optional logger passed to the core runner.
        output_dir: Optional output directory override.
    """
    from habit.core.machine_learning.run import run_kfold_from_config

    run_kfold_from_config(config, logger=logger, output_dir=output_dir)


def run_model_comparison(
    config: "ModelComparisonConfig",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Compare multiple trained models from a validated config object.

    Args:
        config: Loaded model-comparison configuration.
        logger: Optional logger passed to the core runner.
    """
    from habit.core.machine_learning.run import run_model_comparison_from_config

    run_model_comparison_from_config(config, logger=logger)
