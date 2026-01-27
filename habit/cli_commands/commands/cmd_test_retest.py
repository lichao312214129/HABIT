"""
Test-retest reproducibility analysis command implementation
Analyzes test-retest reproducibility of habitat features
"""

import sys
import os
import logging
import click
from pathlib import Path


def run_test_retest(config_file: str) -> None:
    """
    Run test-retest reproducibility analysis
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    from habit.utils.log_utils import setup_logger
    from habit.core.common.service_configurator import ServiceConfigurator
    from habit.core.machine_learning.config_schemas import TestRetestConfig
    from habit.core.machine_learning.feature_selectors.icc.habitat_test_retest_mapper import (
        find_habitat_mapping, batch_process_files, setup_logger as setup_test_retest_logger
    )
    
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(config_file):
        click.echo(f"Error: Configuration file not found: {config_file}", err=True)
        sys.exit(1)
    
    try:
        # Load configuration using standard pattern
        config = TestRetestConfig.from_file(config_file)
        
        # Get output directory
        output_dir = Path(config.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging at CLI entry point
        logger = setup_logger(
            name='cli.test_retest',
            output_dir=output_dir,
            log_filename='test_retest.log',
            level=logging.DEBUG if config.debug else logging.INFO
        )
        
        logger.info(f"Starting test-retest analysis with config: {config_file}")
        click.echo(f"Starting test-retest analysis with config: {config_file}")
        
        # Create service configurator
        configurator = ServiceConfigurator(config=config, logger=logger, output_dir=str(output_dir))
        cfg = configurator.create_test_retest_analyzer()
        
        # Setup debug logging for test-retest mapper module
        setup_test_retest_logger(cfg.debug)
        
        # Find habitat mapping
        click.echo("Computing habitat mapping between test and retest data...")
        habitat_mapping = find_habitat_mapping(
            cfg.test_habitat_table,
            cfg.retest_habitat_table,
            cfg.features,
            cfg.similarity_method
        )
        
        # Print mapping
        click.echo("Habitat mapping:")
        for retest_label, test_label in habitat_mapping.items():
            click.echo(f"  Retest Habitat {retest_label} -> Test Habitat {test_label}")
        logger.info(f"Habitat mapping: {habitat_mapping}")
        
        # Process files
        click.echo(f"Processing files using {cfg.processes} processes...")
        batch_process_files(
            cfg.input_dir,
            habitat_mapping,
            cfg.out_dir,
            cfg.processes
        )
        
        logger.info("Test-retest analysis completed successfully")
        click.secho("✓ Test-retest analysis completed successfully!", fg='green')
        
    except Exception as e:
        logging.error(f"Error during test-retest analysis: {e}")
        click.echo(f"Error: {e}", err=True)
        if config.debug if 'config' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)

