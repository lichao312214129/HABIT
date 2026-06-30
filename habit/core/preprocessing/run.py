# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Programmatic entry point for the preprocessing pipeline."""

from __future__ import annotations

import logging
from typing import Optional

from habit.core.preprocessing.config_schemas import PreprocessingConfig
from habit.core.preprocessing.configurator import PreprocessingConfigurator
from habit.utils.log_utils import get_module_logger

_LOG = get_module_logger(__name__)


def run_preprocess_from_config(
    config: PreprocessingConfig,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Run batch image preprocessing from a validated config object.

    Args:
        config: Validated preprocessing configuration (typically from
            ``PreprocessingConfig.from_file``).
        logger: Optional logger; when omitted a module logger is used.

    Raises:
        Exception: Propagates initialization or batch-processing failures.
    """
    log = logger or _LOG
    configurator = PreprocessingConfigurator(config=config, logger=log)
    processor = configurator.create_batch_processor()
    log.info("Starting batch preprocessing")
    processor.run()
    log.info("Batch preprocessing completed")
