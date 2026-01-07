"""
Habitat analysis command implementation
Generates habitat maps through clustering analysis
"""

import sys
import logging
import click
from datetime import datetime
from pathlib import Path


def run_habitat(config_file: str, debug_mode: bool) -> None:
    """
    Run habitat analysis pipeline
    
    Args:
        config_file (str): Path to configuration YAML file
        debug_mode (bool): Whether to enable debug mode
    """
    from habit.core.habitat_analysis import HabitatAnalysis
    from habit.utils.io_utils import load_config
    from habit.utils.log_utils import setup_logger
    
    # Check if config file is provided
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    # Load configuration first to get output directory
    try:
        config = load_config(config_file)
        click.echo(f"Loaded configuration from: {config_file}")
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)
    
    # Set basic parameters
    data_dir = config.get('data_dir')
    out_dir = config.get('out_dir')
    
    # Create output directory
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging at CLI entry point - all subsequent module logs go to this file
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger = setup_logger(
        name='cli.habitat',
        output_dir=output_path,
        log_filename='habitat_analysis.log',
        level=log_level
    )
    n_processes = config.get('processes', 4)
    plot_curves = config.get('plot_curves', True)
    random_state = config.get('random_state', 42)
    
    # Extract feature construction configuration
    feature_config = config.get('FeatureConstruction', {})
    
    # Extract habitat segmentation configuration
    habitats_config = config.get('HabitatsSegmention', {})
    clustering_strategy = habitats_config.get('clustering_strategy', 'two_step')
    
    # Extract supervoxel method configuration
    supervoxel_config = habitats_config.get('supervoxel', {})
    supervoxel_method = supervoxel_config.get('algorithm', 'kmeans')
    n_clusters_supervoxel = supervoxel_config.get('n_clusters', 50)

    # Extract habitat method configuration
    habitat_config = habitats_config.get('habitat', {})
    habitat_method = habitat_config.get('algorithm', 'kmeans')
    n_clusters_habitats_max = habitat_config.get('max_clusters', 10)
    n_clusters_habitats_min = habitat_config.get('min_clusters', 2)
    habitat_cluster_selection_method = habitat_config.get('habitat_cluster_selection_method', None)
    best_n_clusters = habitat_config.get('best_n_clusters', None)
    
    # Convert best_n_clusters to integer if it's not None and can be converted
    if best_n_clusters is not None and str(best_n_clusters).isdigit():
        best_n_clusters = int(best_n_clusters)
    
    # Get mode parameter - either 'training' or 'testing'
    mode = habitat_config.get('mode', 'training')
    
    # Extract one_step_settings if clustering_strategy is 'one_step'
    one_step_settings = None
    if clustering_strategy == 'one_step':
        one_step_settings = habitats_config.get('one_step_settings', None)
        if one_step_settings is None:
            # Try to get from supervoxel config for backward compatibility
            one_step_settings = supervoxel_config.get('one_step_settings', None)
    
    # Log parameters
    logger.info("==== Habitat Clustering Parameters ====")
    logger.info("Config file: %s", config_file)
    logger.info("Data directory: %s", data_dir)
    logger.info("Output folder: %s", out_dir)
    logger.info("Feature configuration: %s", feature_config)
    logger.info("Clustering strategy: %s", clustering_strategy)
    logger.info("Supervoxel method: %s", supervoxel_method)
    logger.info("Supervoxel clusters: %d", n_clusters_supervoxel)
    logger.info("Habitat method: %s", habitat_method)
    logger.info("Maximum habitat clusters: %d", n_clusters_habitats_max)
    logger.info("Habitat cluster selection method: %s", habitat_cluster_selection_method)
    logger.info("Best number of clusters (if specified): %s", best_n_clusters)
    logger.info("One-step settings: %s", one_step_settings)
    logger.info("Mode: %s", mode)
    logger.info("Number of processes: %d", n_processes)
    logger.info("Generate plots: %s", plot_curves)
    logger.info("Random seed: %d", random_state)
    logger.info("Debug mode: %s", debug_mode)
    logger.info("=========================")
    
    click.echo(f"Starting habitat analysis...")
    click.echo(f"  Data directory: {data_dir}")
    click.echo(f"  Output directory: {out_dir}")
    click.echo(f"  Mode: {mode}")
    
    # Create and run HabitatAnalysis
    habitat_analysis = HabitatAnalysis(
        root_folder=data_dir,
        out_folder=out_dir,
        feature_config=feature_config,
        clustering_strategy=clustering_strategy,
        supervoxel_clustering_method=supervoxel_method,
        n_clusters_supervoxel=n_clusters_supervoxel,
        habitat_clustering_method=habitat_method,
        n_clusters_habitats_max=n_clusters_habitats_max,
        n_clusters_habitats_min=n_clusters_habitats_min,
        habitat_cluster_selection_method=habitat_cluster_selection_method,
        best_n_clusters=best_n_clusters,
        one_step_settings=one_step_settings,
        mode=mode,
        n_processes=n_processes,
        plot_curves=plot_curves,
        random_state=random_state,
        config_file=config_file
    )
    
    try:
        habitat_analysis.run()
        logger.info("Habitat analysis completed successfully")
        click.secho("✓ Habitat analysis completed successfully!", fg='green')
    except Exception as e:
        logger.error("Error during habitat analysis: %s", str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

