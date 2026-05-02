"""
HabitatConfigurator: factory for habitat-domain services.

Owns the assembly of:
    * :class:`HabitatAnalysis` and its three services
      (:class:`FeatureService` / :class:`ClusteringService` /
      :class:`ResultWriter`),
    * :class:`HabitatMapAnalyzer` (post-clustering feature extraction),
    * :class:`TraditionalRadiomicsExtractor` (PyRadiomics wrapper),
    * :class:`TestRetestConfig` resolution for the test-retest analyser.

All heavy imports are deferred to the factory methods so importing this
module does not pull in PyRadiomics / SimpleITK / scikit-learn until a
factory is actually called.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import BaseConfigurator


class HabitatConfigurator(BaseConfigurator):
    """Factory for habitat analysis, feature extraction and reproducibility."""

    logger_name = 'habitat_configurator'

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # HabitatAnalysis
    # ------------------------------------------------------------------

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

    def create_result_writer(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`ResultWriter`."""
        from habit.core.habitat_analysis.services import ResultWriter

        habitat_config = self._get_habitat_config(config)
        return ResultWriter(habitat_config, self.logger)

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
            result_writer=self.create_result_writer(cfg),
            logger=self.logger,
        )

    # ------------------------------------------------------------------
    # HabitatMapAnalyzer (feature extraction from habitat maps)
    # ------------------------------------------------------------------

    def create_feature_extractor(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`HabitatMapAnalyzer`."""
        from habit.core.habitat_analysis.habitat_features.habitat_analyzer import HabitatMapAnalyzer
        from habit.core.habitat_analysis.config_schemas import FeatureExtractionConfig

        cfg = config if config is not None else self.config
        if isinstance(cfg, dict):
            cfg = FeatureExtractionConfig.model_validate(cfg)
        elif not isinstance(cfg, FeatureExtractionConfig):
            try:
                cfg_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
                cfg = FeatureExtractionConfig.model_validate(cfg_dict)
            except Exception as exc:
                raise ValueError(
                    f"Invalid configuration for Feature Extraction: {exc}"
                ) from exc

        return HabitatMapAnalyzer(
            params_file_of_non_habitat=str(cfg.params_file_of_non_habitat),
            params_file_of_habitat=str(cfg.params_file_of_habitat),
            raw_img_folder=str(cfg.raw_img_folder),
            habitats_map_folder=str(cfg.habitats_map_folder),
            out_dir=str(cfg.out_dir),
            n_processes=cfg.n_processes,
            habitat_pattern=cfg.habitat_pattern,
        )

    # ------------------------------------------------------------------
    # Traditional radiomics extractor
    # ------------------------------------------------------------------

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

        # Both nested and flat config forms are still valid (pydantic schema
        # exposes both shapes).
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

        # Apply additional config knobs after construction. Kept as direct
        # attribute writes for now to mirror the original ServiceConfigurator
        # behaviour; a follow-up could move these into the class __init__.
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

    # ------------------------------------------------------------------
    # Test-retest analysis
    # ------------------------------------------------------------------

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
