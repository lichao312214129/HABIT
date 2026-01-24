"""
Simplified ICC Analysis Script

Command-line interface for calculating reliability metrics using the simplified architecture.
"""

import argparse
import sys
import os
import yaml
from pathlib import Path

from habit.core.machine_learning.feature_selectors.icc.icc_analyzer import (
    analyze_features,
    save_results,
)
from habit.utils.log_utils import setup_logger
from habit.core.common.config_loader import load_config


def run_analysis(config: dict) -> None:
    """
    Run ICC analysis with configuration dictionary.
    
    Args:
        config: Configuration dictionary loaded from YAML file
    """
    # Get output path
    output_path = config.get('output', {}).get('path', 'icc_results.json')
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(
        name='icc',
        output_dir=output_dir,
        log_filename='icc_analysis.log',
        level=logging.INFO
    )
    
    logger.info(f"Starting ICC analysis with config")
    
    # Get input configuration
    input_config = config.get('input', {})
    input_type = input_config.get('type', 'files')
    
    # Parse file groups
    file_groups = []
    
    if input_type == 'files':
        file_groups_config = input_config.get('file_groups', [])
        for group in file_groups_config:
            if len(group) >= 2:  # Need at least 2 files for reliability analysis
                file_groups.append(group)
    elif input_type == 'directories':
        dir_list = input_config.get('dir_list', [])
        if len(dir_list) >= 2:
            file_groups = _parse_directories(dir_list)
    
    if not file_groups:
        logger.error("No valid file groups found")
        print("Error: No valid file groups found")
        return
    
    logger.info(f"将分析 {len(file_groups)} 组文件")
    
    # Get metrics configuration
    metrics_config = config.get('metrics', ['icc2', 'icc3'])
    logger.info(f"将计算以下指标: {', '.join(metrics_config)}")
    
    # Get processes configuration
    processes = config.get('processes', None)
    
    # Analyze each file group
    all_results = {}
    
    for i, file_group in enumerate(file_groups):
        logger.info(f"\n组 {i+1}: {', '.join(os.path.basename(f) for f in file_group)}")
        
        try:
            # Analyze features
            group_results = analyze_features(
                file_paths=file_group,
                metrics=metrics_config,
            )
            
            # Merge results
            all_results.update(group_results)
            
        except Exception as e:
            logger.error(f"Error analyzing group {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    try:
        save_results(all_results, output_path, logger)
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        print(f"Error: Failed to save results: {e}")
        return
    
    # _print_statistics provides a more detailed summary
    _print_statistics(all_results, logger)
    
    logger.info("分析完成")
    print(f"\n分析完成，结果已保存到 {output_path}")


def _parse_directories(dir_list: list) -> list:
    """
    Parse directory list and generate file groups.
    
    Args:
        dir_list: List of directory paths
        
    Returns:
        List of file groups
    """
    import pandas as pd
    
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


def _print_statistics(results: dict, logger) -> None:
    """
    Print statistics of analysis results.
    
    Args:
        results: Results dictionary
        logger: Logger instance
    """
    total_features = 0
    total_values = 0
    good_features = 0  # value >= 0.75
    valid_features = 0  # any non-None value
    
    for group_name, features in results.items():
        for feature_name, ft_results in features.items():
            total_features += 1
            
            # Check if any metric has a valid value
            has_valid = any(v is not None for v in ft_results.values())
            if has_valid:
                valid_features += 1
            
            # Check for good ICC values (>= 0.75)
            for metric_name, value in ft_results.items():
                if value is not None and not _is_nan(value):
                    total_values += 1
                    if metric_name.startswith('icc') and value >= 0.75:
                        good_features += 1
    
    logger.info(f"\n统计信息:")
    logger.info(f"  总特征数: {total_features}")
    logger.info(f"  有效特征数: {valid_features}")
    logger.info(f"  计算值总数: {total_values}")
    logger.info(f"  良好指标数 (>= 0.75): {good_features}")
    
    if total_values > 0:
        good_pct = good_features / total_values * 100
        logger.info(f"  良好指标比例: {good_pct:.1f}%")


def _is_nan(value) -> bool:
    """
    Check if value is NaN.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is NaN
    """
    import math
    return isinstance(value, float) and math.isnan(value)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Calculate reliability metrics (ICC, Kappa, etc.) between test-retest data'
    )
    parser.add_argument(
        '-c', '--config',
        required=True,
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error: Failed to load configuration: {e}")
        sys.exit(1)
    
    # Run analysis
    try:
        run_analysis(config)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()