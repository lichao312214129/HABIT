"""Feature extraction and radiomics orchestrators (compat engine)."""
from __future__ import annotations

import logging
from typing import Optional

from habit.schemas.workflows.habitat import FeatureExtractionConfig, RadiomicsConfig
from habit.compat.engines.habitat_extraction.configurator import HabitatConfigurator
from habit.utils.log_utils import get_module_logger

_LOG = get_module_logger(__name__)


def run_feature_extraction_from_config(
    config: FeatureExtractionConfig,
    *,
    plugin_configs: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Run habitat feature extraction from a validated config."""
    log = logger or _LOG
    configurator = HabitatConfigurator(
        config=config,
        logger=log,
        plugin_configs=plugin_configs,
    )
    extractor = configurator.create_feature_extractor()
    log.info("Executing feature extraction")
    extractor.run(
        feature_types=config.feature_types,
        n_habitats=config.n_habitats,
    )
    log.info("Feature extraction completed")


def run_radiomics_from_config(
    config: RadiomicsConfig,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Run traditional radiomics extraction from a validated config."""
    log = logger or _LOG
    out = output_dir or str(config.out_dir or config.paths.out_dir)
    configurator = HabitatConfigurator(config=config, logger=log, output_dir=out)
    extractor = configurator.create_radiomics_extractor()
    log.info("Executing radiomics extraction")
    extractor.run()
    log.info("Radiomics extraction completed")
