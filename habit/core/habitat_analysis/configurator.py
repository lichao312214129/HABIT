# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
HabitatConfigurator: factory for habitat-domain services.

Owns the assembly of:
    * :class:`HabitatAnalysis` and its three services
      (:class:`FeatureService` / :class:`ClusteringService` /
      :class:`HabitatImageWriter`),
    * :class:`HabitatMapAnalyzer` (post-clustering feature extraction),
    * :class:`TraditionalRadiomicsExtractor` (PyRadiomics wrapper),
    * :class:`TestRetestConfig` resolution for the test-retest analyser.

All heavy imports are deferred to the factory methods so importing this
module does not pull in PyRadiomics / SimpleITK / scikit-learn until a
factory is actually called.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from habit.core.common.configurators.base import BaseConfigurator


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
        from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

        cfg = config if config is not None else self.config
        if isinstance(cfg, HabitatAnalysisConfig):
            return cfg
        if isinstance(cfg, dict):
            return HabitatAnalysisConfig.model_validate(cfg)
        return cfg

    def create_feature_service(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`FeatureService`."""
        from habit.core.habitat_analysis.services import FeatureService

        habitat_config = self._get_habitat_config(config)
        return FeatureService(habitat_config, self.logger)

    def create_clustering_service(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`ClusteringService`."""
        from habit.core.habitat_analysis.services import ClusteringService

        habitat_config = self._get_habitat_config(config)
        return ClusteringService(habitat_config, self.logger)

    def create_habitat_image_writer(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`HabitatImageWriter`."""
        from habit.core.habitat_analysis.services import HabitatImageWriter

        habitat_config = self._get_habitat_config(config)
        return HabitatImageWriter(habitat_config, self.logger)

    def create_habitat_analysis(self, config: Optional[Any] = None) -> Any:
        """
        Return a fully configured :class:`HabitatAnalysis`.

        Wires the three services and the logger; all real behaviour
        (build / fit / predict / run) lives inside ``HabitatAnalysis``.
        """
        from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis

        cfg = config if config is not None else self.config
        return HabitatAnalysis(
            config=cfg,
            feature_service=self.create_feature_service(cfg),
            clustering_service=self.create_clustering_service(cfg),
            habitat_image_writer=self.create_habitat_image_writer(cfg),
            logger=self.logger,
        )

    def create_feature_extractor(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`HabitatMapAnalyzer`."""
        from habit.core.habitat_analysis.habitat_features.habitat_analyzer import HabitatMapAnalyzer
        from habit.core.habitat_analysis.config_schemas import FeatureExtractionConfig
        from habit.core.habitat_analysis.feature_extraction_loader import (
            parse_feature_extraction_config,
            plugin_configs_for_feature_types,
        )

        cfg = config if config is not None else self.config
        plugin_configs = dict(self._plugin_configs)
        if isinstance(cfg, dict):
            cfg, plugin_configs = parse_feature_extraction_config(cfg)
        elif not isinstance(cfg, FeatureExtractionConfig):
            try:
                cfg_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
                cfg, plugin_configs = parse_feature_extraction_config(cfg_dict)
            except Exception as exc:
                raise ValueError(
                    f"Invalid configuration for Feature Extraction: {exc}"
                ) from exc
        elif not plugin_configs:
            plugin_configs = plugin_configs_for_feature_types(cfg.feature_types)

        return HabitatMapAnalyzer(
            params_file_of_non_habitat=str(cfg.params_file_of_non_habitat),
            params_file_of_habitat=str(cfg.params_file_of_habitat),
            raw_img_folder=str(cfg.raw_img_folder),
            habitats_map_folder=str(cfg.habitats_map_folder),
            out_dir=str(cfg.out_dir),
            n_processes=cfg.n_processes,
            habitat_pattern=cfg.habitat_pattern,
            plugin_configs=plugin_configs,
        )

    def create_radiomics_extractor(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`TraditionalRadiomicsExtractor`."""
        from habit.core.habitat_analysis.habitat_features.traditional_radiomics_extractor import (
            TraditionalRadiomicsExtractor,
        )
        from habit.core.habitat_analysis.config_schemas import RadiomicsConfig

        cfg = config if config is not None else self.config
        if isinstance(cfg, dict):
            cfg = RadiomicsConfig.model_validate(cfg)
        elif not isinstance(cfg, RadiomicsConfig):
            try:
                cfg_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
                cfg = RadiomicsConfig.model_validate(cfg_dict)
            except Exception as exc:
                raise ValueError(
                    f"Invalid configuration for Radiomics Extraction: {exc}"
                ) from exc

        params_file = cfg.params_file or cfg.paths.params_file
        images_folder = cfg.images_folder or cfg.paths.images_folder
        out_dir = cfg.out_dir or cfg.paths.out_dir
        n_processes = cfg.n_processes or cfg.processing.n_processes

        extractor = TraditionalRadiomicsExtractor(
            params_file=params_file,
            images_folder=images_folder,
            out_dir=out_dir,
            n_processes=n_processes,
        )

        extractor.save_every_n_files = cfg.processing.save_every_n_files
        extractor.process_image_types = cfg.processing.process_image_types
        extractor.target_labels = cfg.processing.target_labels
        extractor.export_by_image_type = cfg.export.export_by_image_type
        extractor.export_combined = cfg.export.export_combined
        extractor.export_format = cfg.export.export_format
        extractor.add_timestamp = cfg.export.add_timestamp
        extractor.log_level = cfg.logging.level
        extractor.console_output = cfg.logging.console_output
        extractor.file_output = cfg.logging.file_output

        return extractor

    def create_test_retest_analyzer(self, config: Optional[Any] = None) -> Any:
        """
        Return a validated :class:`TestRetestConfig`.

        Test-retest analysis uses a functional API; the configurator only
        guarantees that the config is validated and ready to be passed into
        the downstream analysis function.
        """
        from habit.core.machine_learning.config_schemas import TestRetestConfig

        cfg = config if config is not None else self.config
        if isinstance(cfg, dict):
            cfg = TestRetestConfig.model_validate(cfg)
        elif not isinstance(cfg, TestRetestConfig):
            try:
                cfg_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
                cfg = TestRetestConfig.model_validate(cfg_dict)
            except Exception as exc:
                raise ValueError(
                    f"Invalid configuration for Test-Retest Analysis: {exc}"
                ) from exc
        return cfg
