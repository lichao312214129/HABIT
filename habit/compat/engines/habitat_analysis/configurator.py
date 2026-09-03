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
"""
HabitatConfigurator: factory for habitat-domain services.

Owns the assembly of:
    * :class:`HabitatAnalysis` and its three services
      (:class:`FeatureService` / :class:`ClusteringService` /
      :class:`HabitatImageWriter`),

All heavy imports are deferred to the factory methods so importing this
module does not pull in PyRadiomics / SimpleITK / scikit-learn until a
factory is actually called.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from habit.compat.configurator_base import BaseConfigurator


class HabitatConfigurator(BaseConfigurator):
    """Factory for habitat analysis, feature extraction and reproducibility."""

    logger_name = 'habitat_configurator'

    def __init__(
        self,
        config: Any,
        logger: Optional[Any] = None,
        output_dir: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config, logger=logger, output_dir=output_dir)
        self._plugin_configs: Dict[str, Any] = plugin_configs or {}

    def _get_habitat_config(self, config: Optional[Any] = None) -> Any:
        """
        Coerce a raw config to :class:`HabitatAnalysisConfig`.

        Args:
            config: optional override; falls back to ``self.config``.

        Returns:
            A validated ``HabitatAnalysisConfig``.
        """
        from habit.schemas.workflows.habitat import HabitatAnalysisConfig

        cfg = config if config is not None else self.config
        if isinstance(cfg, HabitatAnalysisConfig):
            return cfg
        if isinstance(cfg, dict):
            return HabitatAnalysisConfig.model_validate(cfg)
        return cfg

    def create_feature_service(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`FeatureService`."""
        from habit.compat.engines.habitat_analysis.services import FeatureService

        habitat_config = self._get_habitat_config(config)
        return FeatureService(habitat_config, self.logger)

    def create_clustering_service(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`ClusteringService`."""
        from habit.compat.engines.habitat_analysis.services import ClusteringService

        habitat_config = self._get_habitat_config(config)
        return ClusteringService(habitat_config, self.logger)

    def create_habitat_image_writer(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`HabitatImageWriter`."""
        from habit.compat.engines.habitat_analysis.services import HabitatImageWriter

        habitat_config = self._get_habitat_config(config)
        return HabitatImageWriter(habitat_config, self.logger)

    def create_habitat_analysis(self, config: Optional[Any] = None) -> Any:
        """
        Return a fully configured :class:`HabitatAnalysis`.

        Wires the three services and the logger; all real behaviour
        (build / fit / predict / run) lives inside ``HabitatAnalysis``.
        """
        from habit.compat.engines.habitat_analysis.habitat_analysis import HabitatAnalysis

        cfg = config if config is not None else self.config
        return HabitatAnalysis(
            config=cfg,
            feature_service=self.create_feature_service(cfg),
            clustering_service=self.create_clustering_service(cfg),
            habitat_image_writer=self.create_habitat_image_writer(cfg),
            logger=self.logger,
        )

