"""
Legacy entry point for ICC analysis.

This script is a thin wrapper to maintain backward compatibility. It loads a
configuration file and delegates to centralized ICC handler.
The primary entry point for this functionality is via the main `habit` CLI:
`habit icc --config <path_to_config>`
"""

import argparse
import sys
import logging
from pathlib import Path

def main() -> None:
    """
    Main function to run ICC analysis from a configuration file.
    """
    parser = argparse.ArgumentParser(
        description="Calculate ICC values based on a configuration file."
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to YAML configuration file for ICC analysis.'
    )
    args = parser.parse_args()

    # Late import to avoid circular dependencies and keep startup fast
    from habit.core.common.service_configurator import ServiceConfigurator
    from habit.core.machine_learning.feature_selectors.icc.icc import run_icc_analysis_from_config
    from habit.utils.log_utils import setup_logger

    try:
        # Load config using ServiceConfigurator pattern
        configurator = ServiceConfigurator(config_path=args.config)
        config = configurator.config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or parsing configuration file: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logger based on config
    try:
        output_path = Path(config.get('output', {}).get('path', 'icc_analysis.json'))
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger = setup_logger(
            name='habit.icc',
            output_dir=output_dir,
            log_filename='icc_analysis.log',
            level=logging.DEBUG if config.get("debug") else logging.INFO
        )
    except Exception as e:
        print(f"Error setting up logger based on output path in config: {e}", file=sys.stderr)
        # Continue without file logging if logger setup fails
        logging.basicConfig(level=logging.INFO)

    # Run ICC analysis by delegating to centralized handler
    try:
        print(f"Starting ICC analysis with config: {args.config}")
        run_icc_analysis_from_config(config)
        print("ICC analysis completed successfully.")
    except Exception as e:
        logging.getLogger().error(f"An unexpected error occurred during ICC analysis: {e}", exc_info=True)
        print(f"An unexpected error occurred during analysis: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Example of how to run, assuming a default config exists
    if len(sys.argv) == 1:
        default_config = './config/config_icc_analysis.yaml'
        print(f"No config file provided. Using default: {default_config}")
        sys.argv.extend(['--config', default_config])
    
    main()
 