# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Public auxiliary analysis API (ICC and test-retest tools)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union

from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.provenance import create_run_manifest, write_run_manifest

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


def run_icc_analysis(
    config: Union["ICCConfig", Mapping[str, Any]],
) -> WorkflowResult[None]:
    """
    Run ICC analysis from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.core.machine_learning.feature_selectors.icc.config.ICCConfig`.

    Returns:
        A result with the ICC output directory in ``artifacts``.
    """
    from habit.core.machine_learning.feature_selectors.icc.config import ICCConfig
    from habit.core.machine_learning.feature_selectors.icc.icc import (
        run_icc_analysis_from_config,
    )

    validated_config = coerce_config(config, ICCConfig)
    run_icc_analysis_from_config(validated_config)
    output_path = Path(validated_config.output.path)
    manifest = create_run_manifest("icc_analysis", validated_config)
    manifest_path = write_run_manifest(manifest, output_path.parent)
    return WorkflowResult(
        output_dir=output_path.parent,
        artifacts={"icc_result": output_path},
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )


def run_test_retest_analysis(
    config: Union["TestRetestConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[Dict[int, int]]:
    """
    Map retest habitat labels to test labels and write remapped habitat images.

    This is the programmatic equivalent of ``habit test-retest``.  It deliberately
    does not configure global logging or terminate the interpreter, so callers can
    safely compose it in notebooks, services, and larger Python workflows.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.core.machine_learning.config_schemas.TestRetestConfig`.
        logger: Optional logger used to report the discovered label mapping.

    Returns:
        Label mapping in ``data`` and the remapped-image directory in
        ``artifacts``.
    """
    from habit.core.machine_learning.config_schemas import TestRetestConfig
    from habit.core.machine_learning.feature_selectors.icc.habitat_test_retest_mapper import (
        batch_process_files,
        find_habitat_mapping,
    )

    validated_config = coerce_config(config, TestRetestConfig)
    habitat_mapping: Dict[int, int] = find_habitat_mapping(
        validated_config.test_habitat_table,
        validated_config.retest_habitat_table,
        validated_config.features,
        validated_config.similarity_method,
    )
    if logger is not None:
        logger.info("Computed test-retest habitat mapping: %s", habitat_mapping)
    batch_process_files(
        validated_config.input_dir,
        habitat_mapping,
        validated_config.out_dir,
        validated_config.processes,
    )
    manifest = create_run_manifest("test_retest_analysis", validated_config)
    manifest_path = write_run_manifest(manifest, validated_config.out_dir)
    return WorkflowResult(
        data=habitat_mapping,
        output_dir=validated_config.out_dir,
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )
