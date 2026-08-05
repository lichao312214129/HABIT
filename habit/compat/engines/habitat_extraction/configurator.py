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
"""Habitat feature-extraction configurator (compat engine slice)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from habit.compat.configurator_base import BaseConfigurator


class HabitatConfigurator(BaseConfigurator):
    """Factory for post-segmentation habitat feature and radiomics extractors."""

    logger_name = "habitat_configurator"

    def __init__(
        self,
        config: Any,
        logger: Optional[Any] = None,
        output_dir: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config, logger=logger, output_dir=output_dir)
        self._plugin_configs: Dict[str, Any] = plugin_configs or {}

    def create_feature_extractor(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`HabitatMapAnalyzer`."""
        from habit.compat.engines.habitat_extraction.habitat_features.habitat_analyzer import (
            HabitatMapAnalyzer,
        )
        from habit.compat.feature_extraction_loader import (
            parse_feature_extraction_config,
            plugin_configs_for_feature_types,
        )
        from habit.schemas.workflows.habitat import FeatureExtractionConfig
        from habit.utils.radiomics_preset_utils import resolve_params_file

        cfg = config if config is not None else self.config
        plugin_configs = dict(self._plugin_configs)
        if isinstance(cfg, dict):
            cfg, plugin_configs = parse_feature_extraction_config(cfg)
        elif not isinstance(cfg, FeatureExtractionConfig):
            try:
                cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
                cfg, plugin_configs = parse_feature_extraction_config(cfg_dict)
            except Exception as exc:
                raise ValueError(
                    f"Invalid configuration for Feature Extraction: {exc}"
                ) from exc
        elif not plugin_configs:
            plugin_configs = plugin_configs_for_feature_types(cfg.feature_types)

        return HabitatMapAnalyzer(
            params_file_of_non_habitat=resolve_params_file(
                cfg.params_file_of_non_habitat, preset="roi"
            ),
            params_file_of_habitat=resolve_params_file(
                cfg.params_file_of_habitat, preset="habitat"
            ),
            raw_img_folder=str(cfg.raw_img_folder),
            habitats_map_folder=str(cfg.habitats_map_folder),
            out_dir=str(cfg.out_dir),
            n_processes=cfg.n_processes,
            habitat_pattern=cfg.habitat_pattern,
            plugin_configs=plugin_configs,
        )

    def create_radiomics_extractor(self, config: Optional[Any] = None) -> Any:
        """Return a configured :class:`TraditionalRadiomicsExtractor`."""
        from habit.compat.engines.habitat_extraction.habitat_features.traditional_radiomics_extractor import (
            TraditionalRadiomicsExtractor,
        )
        from habit.schemas.workflows.habitat import RadiomicsConfig
        from habit.utils.radiomics_preset_utils import resolve_params_file

        cfg = config if config is not None else self.config
        if isinstance(cfg, dict):
            cfg = RadiomicsConfig.model_validate(cfg)
        elif not isinstance(cfg, RadiomicsConfig):
            try:
                cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
                cfg = RadiomicsConfig.model_validate(cfg_dict)
            except Exception as exc:
                raise ValueError(
                    f"Invalid configuration for Radiomics Extraction: {exc}"
                ) from exc

        params_file = resolve_params_file(
            cfg.params_file or cfg.paths.params_file, preset="roi"
        )
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
