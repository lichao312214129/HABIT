"""
Command-line interface for calculating ICC values between test-retest features
This module provides functionality for assessing feature reliability through
intraclass correlation coefficient (ICC) analysis between test and retest data.
"""

import argparse
import sys
import os
import json
from habit.core.machine_learning.feature_selectors.icc.icc import (
    configure_logger, parse_files_groups, parse_directories, analyze_multiple_groups
)
from habit.utils.icc_config import (
    load_icc_config, parse_icc_config_files, parse_icc_config_directories,
    get_icc_config_output_path, get_icc_config_processes,
    get_icc_config_metrics, get_icc_config_full_results, get_icc_config_selected_features
)

def main() -> None:
    """
    Main function to run ICC analysis
    
    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Parses file groups or directories
    4. Analyzes ICC values between test and retest features
    5. Saves results to JSON file
    """
    parser = argparse.ArgumentParser(description="计算特征之间的ICC值以评估重复性")
    
    # Create a mutually exclusive group for input methods
    group = parser.add_mutually_exclusive_group(required=True)
    
    # Add config file option
    group.add_argument('--config', type=str, 
                    help='YAML配置文件路径')
    
    # Original options for backward compatibility
    group.add_argument('--files', type=str, 
                    help='文件列表，格式为 "file1.csv,file2.csv,file3.csv;file4.csv,file5.csv,file6.csv"')
    group.add_argument('--dirs', type=str, 
                    help='目录列表，格式为 "dir1,dir2,dir3"，将匹配目录中同名的数据文件')
    
    # Other parameters
    parser.add_argument('--processes', type=int, default=None, 
                    help='进程数，默认使用所有可用CPU')
    parser.add_argument('--output', type=str, default='icc_results.json',
                    help='输出结果的JSON文件路径')
    parser.add_argument('--debug', action='store_true', 
                    help='启用调试模式')
    
    args = parser.parse_args()
    
    # Initialize variables
    file_groups = []
    output_path = args.output
    processes = args.processes
    debug_mode = args.debug
    metrics = None  # Default metrics (icc3)
    full_results = False  # Default: simple values without CI
    selected_features = None  # Default: analyze all features
    
    # Handle config file if provided
    if args.config:
        try:
            config = load_icc_config(args.config)
            
            # Get output path from config
            output_path = get_icc_config_output_path(config)
            
            # Get processes from config if not specified in command line
            if processes is None:
                processes = get_icc_config_processes(config)
            
            # Get debug mode from config if not enabled in command line
            if not debug_mode and config.get("debug", False):
                debug_mode = True
            
            # Get metrics configuration
            metrics = get_icc_config_metrics(config)
            
            # Get full_results configuration
            full_results = get_icc_config_full_results(config)
            
            # Get selected_features configuration
            selected_features = get_icc_config_selected_features(config)
                
            # Parse input based on type
            if config["input"]["type"] == "files":
                file_groups = parse_icc_config_files(config)
            else:  # directories
                dir_list = parse_icc_config_directories(config)
                file_groups = parse_directories(",".join(dir_list))
                
        except Exception as e:
            print(f"配置文件解析错误: {str(e)}")
            sys.exit(1)
    else:
        # Use original methods for backward compatibility
        if args.files:
            file_groups = parse_files_groups(args.files)
        elif args.dirs:
            file_groups = parse_directories(args.dirs)
    
    # Configure logger
    logger = configure_logger(output_path)
    
    # Log level
    if debug_mode:
        import logging
        logger.setLevel(logging.DEBUG)
    
    try:
        if not file_groups:
            logger.error("没有有效的文件组可以分析")
            return
        
        logger.info(f"将分析 {len(file_groups)} 组文件")
        
        # Print file groups
        for i, group in enumerate(file_groups):
            logger.info(f"组 {i+1}: {', '.join(os.path.basename(f) for f in group)}")
        
        # Log metrics configuration
        if metrics:
            logger.info(f"将计算以下指标: {', '.join(metrics)}")
        else:
            logger.info("将计算默认指标: icc3")
        
        if full_results:
            logger.info("将返回完整结果（包含置信区间和p值）")
        
        if selected_features:
            logger.info(f"将只分析以下特征: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
        
        # Execute analysis with metrics parameters
        results = analyze_multiple_groups(
            file_groups, 
            processes, 
            logger,
            selected_features=selected_features,
            metrics=metrics,
            return_full_results=full_results
        )
        
        # Save results
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"分析完成，结果已保存到 {output_path}")
        
        # Print summary
        total_features = 0
        total_values = 0
        good_features = 0  # value >= 0.75
        
        for group_name, features in results.items():
            # Handle different result formats based on metrics and full_results settings
            feature_values = []
            for feature_name, feature_result in features.items():
                if isinstance(feature_result, (int, float)):
                    # Simple value format (single metric, full_results=False)
                    if not is_nan(feature_result):
                        feature_values.append(feature_result)
                elif isinstance(feature_result, dict):
                    # Dict format: either full_results or multiple metrics
                    if 'value' in feature_result:
                        # Full result format for single metric
                        val = feature_result['value']
                        if val is not None and not is_nan(val):
                            feature_values.append(val)
                    else:
                        # Multiple metrics format
                        for metric_name, metric_result in feature_result.items():
                            if isinstance(metric_result, dict) and 'value' in metric_result:
                                val = metric_result['value']
                                if val is not None and not is_nan(val):
                                    feature_values.append(val)
                            elif isinstance(metric_result, (int, float)) and not is_nan(metric_result):
                                feature_values.append(metric_result)
            
            if feature_values:
                avg_value = sum(feature_values) / len(feature_values)
                good_count = sum(1 for v in feature_values if v >= 0.75)
                
                logger.info(f"组 {group_name}: {len(features)} 特征, 平均值: {avg_value:.3f}, "
                           f"良好指标 (>= 0.75): {good_count} ({good_count/len(feature_values)*100:.1f}%)")
                
                total_features += len(features)
                total_values += sum(feature_values)
                good_features += good_count
        
        if total_features > 0 and total_values > 0:
            # Calculate average based on number of values, not features (for multiple metrics)
            n_values = sum(1 for g in results.values() for f in g.values() 
                          if isinstance(f, (int, float)) and not is_nan(f))
            if n_values == 0:
                n_values = total_features  # Fallback
            logger.info(f"分析完成: {total_features} 特征")
            
    except Exception as e:
        logger.error(f"ICC分析失败: {str(e)}")
        if debug_mode:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

def is_nan(value):
    """Check if a value is NaN"""
    return value != value

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(['--config', './config/config_icc_analysis.yaml'])
    main() 