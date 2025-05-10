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
    get_icc_config_output_path, get_icc_config_processes
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
        
        # Execute analysis
        results = analyze_multiple_groups(file_groups, processes, logger)
        
        # Save results
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"分析完成，结果已保存到 {output_path}")
        
        # Print summary
        total_features = 0
        total_icc = 0
        good_features = 0  # ICC >= 0.75
        
        for group_name, features in results.items():
            icc_values = [v for v in features.values() if isinstance(v, (int, float)) and not is_nan(v)]
            
            if icc_values:
                avg_icc = sum(icc_values) / len(icc_values)
                good_count = sum(1 for v in icc_values if v >= 0.75)
                
                logger.info(f"组 {group_name}: {len(icc_values)} 特征, 平均ICC: {avg_icc:.3f}, "
                           f"良好特征 (ICC >= 0.75): {good_count} ({good_count/len(icc_values)*100:.1f}%)")
                
                total_features += len(icc_values)
                total_icc += sum(icc_values)
                good_features += good_count
        
        if total_features > 0:
            logger.info(f"总体: {total_features} 特征, 平均ICC: {total_icc/total_features:.3f}, "
                       f"良好特征 (ICC >= 0.75): {good_features} ({good_features/total_features*100:.1f}%)")
            
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