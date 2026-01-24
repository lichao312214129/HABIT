"""
Feature extraction command implementation
Extracts habitat features from clustered images
"""

import sys
import os
import logging
import click
from pathlib import Path

from habit.core.habitat_analysis.config_schemas import FeatureExtractionConfig
from habit.utils.log_utils import setup_logger

def run_extract_features(config_file: str) -> None:
    """
    Run habitat feature extraction pipeline
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    from habit.core.common.service_configurator import ServiceConfigurator
    
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(config_file):
        click.echo(f"Error: Configuration file not found: {config_file}", err=True)
        sys.exit(1)
    
    try:
        # 1. Load Config (Typed & Resolved)
        config = FeatureExtractionConfig.from_file(config_file)
        
        # 2. Setup Logging
        output_dir = Path(config.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_level = logging.DEBUG if config.debug else logging.INFO
        logger = setup_logger(
            name='cli.extract_features',
            output_dir=output_dir,
            log_filename='processing.log',
            level=log_level
        )
        
        logger.info(f"Starting habitat feature extraction with config: {config_file}")
        click.echo(f"Starting habitat feature extraction with config: {config_file}")
        
        # 3. Initialize Service
        configurator = ServiceConfigurator(config=config, logger=logger)
        extractor = configurator.create_feature_extractor()
        
        # 4. Run Extraction
        logger.info("Executing feature extraction...")
        extractor.run(
            feature_types=config.feature_types,
            n_habitats=config.n_habitats
        )
        
        logger.info("Feature extraction completed successfully")
        click.secho("✓ Feature extraction completed successfully!", fg='green')
        
    except Exception as e:
        # If logger is defined, use it
        if 'logger' in locals():
            logger.error(f"Error during feature extraction: {e}", exc_info=True)
        else:
            click.echo(f"Error: {e}", err=True)
            import traceback
            traceback.print_exc()
        sys.exit(1)

