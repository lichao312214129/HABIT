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
"""L4 feature-extraction recipes (thin assembly).

Stage-4 scope: wire the ``extract`` and ``radiomics`` CLI commands through
recipes instead of importing ``habit.core`` directly. A full domain-native
orchestrator -- cohort assembly from habitat maps, per-subject
:class:`~habit.domain.protocols.HabitatFeatureExtractor` dispatch, and
parallel execution through :class:`~habit.contracts.ops.ExecutionBackend`
-- is deferred until the v0.1 ``HabitatMapAnalyzer`` batch loop is
replaced. Until then these recipes delegate to the public
``habit.api.habitat`` workflow helpers (which still execute the v0.1
engine internally), keeping ``habit.recipes`` free of direct
``habit.core`` imports per the architecture gate.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

__all__ = ["extract_habitat_features", "traditional_radiomics"]


def extract_habitat_features(
    config: Any,
    *,
    plugin_configs: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Extract features from pre-computed habitat maps (``habit extract`` recipe).

    Args:
        config: Validated feature-extraction configuration (v0.1 schema object
            or mapping accepted by
            :func:`habit.api.habitat.build_feature_extraction_config`).
        plugin_configs: Optional plugin settings (e.g. ``graph``) returned
            alongside the shared config by
            :func:`habit.api.habitat.load_feature_extraction_config`.
        logger: Optional run logger forwarded to the workflow helper.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with output directory
        metadata and a run manifest path.
    """
    from habit.api.habitat import run_feature_extraction

    return run_feature_extraction(
        config,
        logger=logger,
        plugin_configs=dict(plugin_configs) if plugin_configs is not None else None,
    )


def traditional_radiomics(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Run standalone traditional radiomics extraction (``habit radiomics`` recipe).

    Args:
        config: Validated radiomics configuration (v0.1 schema object or
            mapping accepted by :class:`~habit.api.habitat.RadiomicsConfig`).
        logger: Optional run logger forwarded to the workflow helper.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with output directory
        metadata and a run manifest path.
    """
    from habit.api.habitat import run_radiomics

    return run_radiomics(config, logger=logger)
