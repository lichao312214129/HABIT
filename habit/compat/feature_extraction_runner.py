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
"""Habitat feature extraction and radiomics workflow runners (L1 compat)."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from habit.utils.log_utils import get_module_logger

__all__ = ["run_feature_extraction_from_config", "run_radiomics_from_config"]

_LOG = get_module_logger(__name__)


def run_feature_extraction_from_config(
    config: Any,
    *,
    plugin_configs: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Run habitat feature extraction from a validated config.

    Args:
        config: Validated feature-extraction configuration.
        plugin_configs: Optional private plugin configuration mapping.
        logger: Optional logger for the v0.1 configurator.
    """
    from habit.compat.engines.habitat_extraction.run import (
        run_feature_extraction_from_config as _run,
    )

    _run(
        config,
        plugin_configs=dict(plugin_configs) if plugin_configs is not None else None,
        logger=logger or _LOG,
    )


def run_radiomics_from_config(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run traditional radiomics extraction from a validated config.

    Args:
        config: Validated radiomics configuration.
        logger: Optional logger for the v0.1 configurator.
        output_dir: Optional output directory override.
    """
    from habit.compat.engines.habitat_extraction.run import (
        run_radiomics_from_config as _run,
    )

    _run(config, logger=logger or _LOG, output_dir=output_dir)
