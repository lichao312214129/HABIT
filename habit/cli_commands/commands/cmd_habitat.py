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
Habitat analysis command implementation.
Generates habitat maps through clustering analysis.
"""

from typing import Optional


def run_habitat(
    config_file: str,
    debug_mode: bool,
    mode: Optional[str],
    pipeline_path: Optional[str],
    resume: bool = False,
) -> None:
    """
    Run habitat analysis pipeline in train or predict mode.

    Args:
        config_file (str): Path to configuration YAML file
        debug_mode (bool): Whether to enable debug mode
        mode (Optional[str]): Override run mode ("train" or "predict")
        pipeline_path (Optional[str]): Override pipeline path for prediction
    """
    import sys
    import logging
    import click
    from pathlib import Path

    from habit.utils.log_utils import setup_logger, stop_queue_listener
    from habit.core.habitat_analysis.configurator import HabitatConfigurator
    from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
    
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    try:
        # Use the static factory method to load, resolve paths, and validate schema
        config = HabitatAnalysisConfig.from_file(config_file)
        click.echo(f"Loaded configuration from: {config_file}")
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)

    # Override debug mode if specified via CLI
    if debug_mode:
        config.debug = True

    # Override run mode and pipeline path if specified via CLI
    if mode:
        config.run_mode = mode
    if pipeline_path:
        config.pipeline_path = pipeline_path
    if resume:
        config.resume = True

    output_path = Path(config.out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_level = logging.DEBUG if config.debug else logging.INFO
    logger = setup_logger(
        name='cli.habitat',
        output_dir=output_path,
        log_filename='habitat_analysis.log',
        level=log_level
    )
    
    logger.info("==== Starting Habitat Analysis ====")
    logger.info("Config file: %s", config_file)
    logger.info("Full configuration being used: %s", config.model_dump() if hasattr(config, 'model_dump') else config.dict())
    logger.info("=====================================")
    
    click.echo("Starting habitat analysis...")
    click.echo(f"  Mode: {config.run_mode}")
    click.echo(f"  Output directory: {config.out_dir}")
    if config.run_mode != "predict" and config.resume:
        click.echo("  Resume: enabled (skip checkpointed subjects)")
    if config.run_mode == "predict":
        click.echo(f"  Pipeline path: {config.pipeline_path or 'auto'}")
    click.echo(f"  Log file at: {output_path / 'habitat_analysis.log'}")

    try:
        configurator = HabitatConfigurator(config=config, logger=logger, output_dir=str(output_path))
        habitat_analysis = configurator.create_habitat_analysis()

        if config.run_mode == "predict":
            # Predict mode must load a trained pipeline to avoid retraining.
            if not config.pipeline_path:
                raise ValueError(
                    "Error: In 'predict' mode, you MUST provide a 'pipeline_path'. "
                    "Please specify it in your config.yaml or via the --pipeline argument."
                )
            resolved_pipeline = Path(config.pipeline_path)
            if not resolved_pipeline.exists():
                raise FileNotFoundError(
                    f"Pipeline file not found: {resolved_pipeline}. "
                    "Provide a valid pipeline_path in the YAML or via --pipeline."
                )

            habitat_analysis.predict(
                pipeline_path=str(resolved_pipeline),
                save_results_csv=config.save_results_csv,
            )
        else:
            habitat_analysis.fit(save_results_csv=config.save_results_csv)

        logger.info("Habitat analysis completed successfully")
        click.secho("Habitat analysis completed successfully!", fg='green')
    except Exception as e:
        logger.error("Error during habitat analysis: %s", str(e), exc_info=True)
        click.echo(f"An error occurred. See log file for details: {e}", err=True)
        sys.exit(1)
    finally:
        stop_queue_listener()

