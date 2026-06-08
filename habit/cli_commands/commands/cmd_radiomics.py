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
Traditional radiomics extraction command implementation
Extracts traditional radiomics features from medical images
"""

import sys
import os
import logging
import click
from pathlib import Path


def run_radiomics(config_file: str) -> None:
    """
    Run traditional radiomics feature extraction
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    from habit.utils.log_utils import setup_logger
    from habit.core.habitat_analysis.configurator import HabitatConfigurator
    from habit.core.habitat_analysis.config_schemas import RadiomicsConfig
    
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(config_file):
        click.echo(f"Error: Configuration file not found: {config_file}", err=True)
        sys.exit(1)
    
    try:
        # Load configuration using standard pattern
        config = RadiomicsConfig.from_file(config_file)
        
        # Get output directory
        output_dir = Path(config.out_dir or config.paths.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging at CLI entry point
        logger = setup_logger(
            name='cli.radiomics',
            output_dir=output_dir,
            log_filename='radiomics_extraction.log',
            level=logging.INFO
        )
        
        logger.info(f"Starting traditional radiomics extraction with config: {config_file}")
        click.echo(f"Starting traditional radiomics extraction with config: {config_file}")
        
        # Create service configurator and extractor
        configurator = HabitatConfigurator(config=config, logger=logger, output_dir=str(output_dir))
        extractor = configurator.create_radiomics_extractor()
        
        # Run extraction
        extractor.extract_features()
        
        logger.info("Radiomics extraction completed successfully")
        click.secho("✓ Radiomics extraction completed successfully!", fg='green')
        
    except Exception as e:
        logging.error(f"Error during radiomics extraction: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

