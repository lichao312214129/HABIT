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
"""Programmatic entry point for the preprocessing pipeline."""

from __future__ import annotations

import logging
from typing import Optional

from habit.compat.engines.preprocessing.config_schemas import PreprocessingConfig
from habit.compat.engines.preprocessing.configurator import PreprocessingConfigurator
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
