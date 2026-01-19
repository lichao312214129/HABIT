"""
ICC Analysis Task Handler

This module acts as the primary downstream handler for the ICC command-line interface.
It takes a configuration, processes it, and uses the core `icc_analyzer` module
to run the analysis and report results.

Author: Li Chao
Email: lich356@mail.sysu.edu.cn
"""

import logging
from typing import Dict, Any, List
import os

from .icc_analyzer import (
    analyze_features,
    save_results,
    print_summary,
    print_statistics
)
from .config import (
    parse_icc_config_files,
    parse_icc_config_directories,
    get_icc_config_output_path,
    get_icc_config_metrics,
    get_icc_config_selected_features
)

logger = logging.getLogger(__name__)

def _parse_directories(dir_list: list) -> list:
    """
    Parse directory list and generate file groups based on matching filenames.
    """
    from pathlib import Path
    
    dir_files = {}
    for dir_path in dir_list:
        p = Path(dir_path)
        if p.is_dir():
            dir_files[dir_path] = {f.stem: str(f) for f in p.iterdir() if f.suffix.lower() in ['.csv', '.xlsx', '.xls']}

    if not dir_files:
        return []

    # Find common filenames (stems) across all directories
    common_stems = set.intersection(*[set(files.keys()) for files in dir_files.values()])

    # Create file groups
    file_groups = []
    for stem in sorted(list(common_stems)):
        group = [files[stem] for _, files in dir_files.items()]
        file_groups.append(group)
        
    return file_groups

def run_icc_analysis_from_config(config: Dict[str, Any]) -> None:
    """
    Orchestrates the ICC analysis based on a configuration dictionary.
    
    Args:
        config: The configuration dictionary, typically loaded from a YAML file.
    """
    # Get configuration settings
    output_path = get_icc_config_output_path(config)
    metrics = get_icc_config_metrics(config) or ['icc2', 'icc3']
    selected_features = get_icc_config_selected_features(config)
    
    # Parse file groups based on config type
    if config.get("input", {}).get("type") == "files":
        file_groups = parse_icc_config_files(config)
    else:
        dir_list = parse_icc_config_directories(config)
        file_groups = _parse_directories(dir_list)

    if not file_groups:
        logger.error("No valid file groups found in the configuration.")
        return

    logger.info(f"Found {len(file_groups)} file group(s) to analyze.")
    for i, group in enumerate(file_groups):
        logger.info(f"  Group {i+1}: {', '.join(os.path.basename(f) for f in group)}")
    
    logger.info(f"Metrics to be calculated: {', '.join(metrics)}")
    if selected_features:
        logger.info(f"Analyzing a subset of {len(selected_features)} features.")

    # Analyze each file group and aggregate results
    all_results = {}
    for group in file_groups:
        try:
            group_results = analyze_features(
                file_paths=group,
                metrics=metrics,
                selected_features=selected_features
            )
            all_results.update(group_results)
        except Exception as e:
            logger.error(f"An error occurred while processing group {group}: {e}", exc_info=True)

    # Save and report results
    if not all_results:
        logger.warning("Analysis generated no results.")
        return
        
    save_results(all_results, output_path)
    print_summary(all_results)
    print_statistics(all_results)
    
    logger.info("ICC analysis process finished.")
