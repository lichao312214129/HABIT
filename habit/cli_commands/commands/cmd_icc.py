"""
ICC analysis command implementation
Performs Intraclass Correlation Coefficient analysis
"""

import sys
import os
import logging
import click
import yaml
from pathlib import Path


def run_icc(config_file: str) -> None:
    """
    Run ICC (Intraclass Correlation Coefficient) analysis
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    from habit.utils.log_utils import setup_logger
    from habit.utils.config_utils import load_config
    
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(config_file):
        click.echo(f"Error: Configuration file not found: {config_file}", err=True)
        sys.exit(1)
    
    # Load config to get output directory
    try:
        config = load_config(config_file)
        output = Path(config.get('output').get('path'))
        output_dir = output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)
    
    # Setup logging at CLI entry point
    logger = setup_logger(
        name='cli.icc',
        output_dir=output_dir,
        log_filename='icc_analysis.log',
        level=logging.INFO
    )
    
    logger.info(f"Starting ICC analysis with config: {config_file}")
    click.echo(f"Starting ICC analysis with config: {config_file}")
    
    try:
        # Import and run the simplified ICC analysis
        from habit.core.machine_learning.feature_selectors.icc.simple_icc_analyzer import (
            analyze_features,
            save_results,
            print_summary
        )
        from habit.utils.icc_config import (
            parse_icc_config_files, parse_icc_config_directories,
            get_icc_config_output_path, get_icc_config_processes,
            get_icc_config_metrics, get_icc_config_selected_features
        )
        
        # Get output path
        output_path = get_icc_config_output_path(config)
        
        # Get debug mode
        debug_mode = config.get("debug", False)
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        
        # Get metrics
        metrics = get_icc_config_metrics(config)
        if not metrics:
            metrics = ['icc2', 'icc3']
        
        # Get processes
        processes = get_icc_config_processes(config)
        
        # Get selected features
        selected_features = get_icc_config_selected_features(config)
        
        # Parse file groups
        if config["input"]["type"] == "files":
            file_groups = parse_icc_config_files(config)
        else:
            dir_list = parse_icc_config_directories(config)
            file_groups = _parse_directories(dir_list)
        
        if not file_groups:
            logger.error("No valid file groups found")
            click.echo("Error: No valid file groups found", err=True)
            return
        
        logger.info(f"将分析 {len(file_groups)} 组文件")
        
        # Print file groups
        for i, group in enumerate(file_groups):
            logger.info(f"组 {i+1}: {', '.join(os.path.basename(f) for f in group)}")
        
        # Log metrics configuration
        logger.info(f"将计算以下指标: {', '.join(metrics)}")
        
        if selected_features:
            logger.info(f"将只分析以下特征: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
        
        # Analyze each file group
        all_results = {}
        
        for i, file_group in enumerate(file_groups):
            logger.info(f"\n组 {i+1}: {', '.join(os.path.basename(f) for f in file_group)}")
            
            try:
                # Analyze features
                group_results = analyze_features(
                    file_paths=file_group,
                    metrics=metrics,
                    selected_features=selected_features,
                    logger_instance=logger
                )
                
                # Merge results
                all_results.update(group_results)
                
            except Exception as e:
                logger.error(f"Error analyzing group {i+1}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        save_results(all_results, output_path, logger)
        
        # Print summary
        print_summary(all_results, logger)
        
        logger.info("ICC analysis completed successfully")
        click.secho("✓ ICC analysis completed successfully!", fg='green')
    except Exception as e:
        logger.error(f"Error during ICC analysis: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _parse_directories(dir_list: list) -> list:
    """
    Parse directory list and generate file groups.
    
    Args:
        dir_list: List of directory paths
        
    Returns:
        List of file groups
    """
    import pandas as pd
    from pathlib import Path
    
    # Get data files from each directory
    dir_files = {}
    for dir_path in dir_list:
        dir_path = Path(dir_path)
        if dir_path.exists():
            files = [f for f in dir_path.iterdir() 
                         if f.suffix.lower() in ['.csv', '.xlsx', '.xls']]
            dir_files[str(dir_path)] = files
    
    # Group files by filename (without extension)
    filename_groups = {}
    for dir_path, files in dir_files.items():
        for file_path in files:
            basename = file_path.stem
            if basename not in filename_groups:
                filename_groups[basename] = []
            filename_groups[basename].append(str(file_path))
    
    # Only keep file groups that exist in at least two directories
    file_groups = [files for basename, files in filename_groups.items() 
                   if len(files) >= 2]
    
    return file_groups

