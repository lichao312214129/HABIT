# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Public auxiliary analysis API (ICC and test-retest tools)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from habit.core.machine_learning.feature_selectors.icc.config import ICCConfig
    from habit.core.machine_learning.config_schemas import TestRetestConfig

__all__ = [
    "ICCConfig",
    "TestRetestConfig",
    "run_icc_analysis",
    "run_test_retest_analysis",
]


def __getattr__(name: str) -> Any:
    if name == "ICCConfig":
        from habit.core.machine_learning.feature_selectors.icc.config import ICCConfig

        return ICCConfig
    if name == "TestRetestConfig":
        from habit.core.machine_learning.config_schemas import TestRetestConfig

        return TestRetestConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_icc_analysis(config: "ICCConfig") -> None:
    """
    Run ICC analysis from a validated config object.

    Args:
        config: Loaded ICC configuration.
    """
    from habit.core.machine_learning.feature_selectors.icc.icc import (
        run_icc_analysis_from_config,
    )

    run_icc_analysis_from_config(config)


def run_test_retest_analysis(
    config: "TestRetestConfig",
    logger: Optional[logging.Logger] = None,
) -> Dict[int, int]:
    """
    Map retest habitat labels to test labels and write remapped habitat images.

    This is the programmatic equivalent of ``habit test-retest``.  It deliberately
    does not configure global logging or terminate the interpreter, so callers can
    safely compose it in notebooks, services, and larger Python workflows.

    Args:
        config: Validated test-retest workflow configuration.
        logger: Optional logger used to report the discovered label mapping.

    Returns:
        Mapping from each retest habitat label to its corresponding test label.
    """
    from habit.core.machine_learning.feature_selectors.icc.habitat_test_retest_mapper import (
        batch_process_files,
        find_habitat_mapping,
    )

    habitat_mapping: Dict[int, int] = find_habitat_mapping(
        config.test_habitat_table,
        config.retest_habitat_table,
        config.features,
        config.similarity_method,
    )
    if logger is not None:
        logger.info("Computed test-retest habitat mapping: %s", habitat_mapping)
    batch_process_files(
        config.input_dir,
        habitat_mapping,
        config.out_dir,
        config.processes,
    )
    return habitat_mapping
