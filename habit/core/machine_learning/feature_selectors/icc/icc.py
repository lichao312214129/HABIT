import pandas as pd
import numpy as np
import pingouin as pg
import json
import os
import multiprocessing
import logging
import argparse
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Any, Optional, Set

from habit.utils.log_utils import setup_logger, get_module_logger

# Module logger for use throughout the file
logger = get_module_logger(__name__)


def configure_logger(output_path: str) -> logging.Logger:
    """
    Configure the logger and save log files in the same directory as the output file.
    Uses the centralized logging system from habit.utils.log_utils.
    
    Args:
        output_path: Path to the output file
        
    Returns:
        Configured logger instance
    """
    output_dir = Path(output_path).parent.absolute()
    
    # Use the centralized logging system
    return setup_logger(
        name='icc',
        output_dir=output_dir,
        log_filename='icc_analysis.log',
        level=logging.INFO
    )

def read_file(file_path: str) -> pd.DataFrame:
    """
    Read CSV or Excel file based on file extension
    
    Args:
        file_path: Path to the file
        
    Returns:
        DataFrame object containing the file data
        
    Raises:
        ValueError: If the file format is not supported
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        return pd.read_csv(file_path, index_col=0)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .csv, .xlsx, .xls")

def calculate_icc(files_list: List[str], logger, selected_features: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Calculate ICC (Intraclass Correlation Coefficient) values between multiple files
    
    Args:
        files_list: List of file paths to analyze
        logger: Logger instance for recording progress and errors
        selected_features: Optional list of feature names to analyze (if None, all common features will be analyzed)
        
    Returns:
        Dictionary containing ICC values and 95% confidence intervals in the format 
        {feature_name: {"icc": icc_value, "ci95": [lower_bound, upper_bound]}}
    """
    # Get filenames without path and extension
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in files_list]
    group_name = "_vs_".join(file_names)
    
    # Read all files
    data_frames = []
    try:
        for file_path in files_list:
            data_frames.append(read_file(file_path))
    except Exception as e:
        logger.error(f"Error reading files: {e}")
        return {group_name: {}}
    
    # Find common indices across all dataframes
    common_index = set(data_frames[0].index)
    for df in data_frames[1:]:
        common_index = common_index.intersection(df.index)
    
    common_index = list(common_index)
    
    if len(common_index) == 0:
        logger.warning(f"{group_name} has no common patient IDs")
        return {group_name: {}}
    
    # Filter data using common indices
    for i in range(len(data_frames)):
        data_frames[i] = data_frames[i].loc[common_index]
    
    # Find common columns across all dataframes
    common_columns = set(data_frames[0].columns)
    for df in data_frames[1:]:
        common_columns = common_columns.intersection(df.columns)
    
    common_columns = list(common_columns)
    
    if len(common_columns) == 0:
        logger.warning(f"{group_name} has no common feature columns")
        return {group_name: {}}
    
    # Filter features if selected_features is provided
    if selected_features:
        # Find intersection between common_columns and selected_features
        target_columns = [col for col in common_columns if col in selected_features]
        if not target_columns:
            logger.warning(f"{group_name} has no selected features in common")
            return {group_name: {}}
        common_columns = target_columns
        logger.info(f"Analyzing {len(common_columns)} selected features")
    
    # Calculate ICC for each feature
    icc_values = {}
    total = len(common_columns)
    
    for i, ft in enumerate(common_columns):
        try:
            # Prepare data for ICC calculation
            data_list = []
            for reader_idx, df in enumerate(data_frames):
                df_feature = pd.DataFrame(df[ft])
                df_feature["reader"] = np.ones(df_feature.shape[0]) * (reader_idx + 1)
                df_feature["target"] = range(df_feature.shape[0])
                data_list.append(df_feature)
            
            data = pd.concat(data_list, axis=0)
            result = pg.intraclass_corr(
                data=data, 
                targets='target', 
                raters='reader', 
                ratings=ft, 
                nan_policy='omit'
            )
            icc = result.loc[2, "ICC"]
            
            # Get 95% confidence interval
            # ci95 = result.loc[2, "CI95%"]
            
            # Use feature name as key with ICC value and confidence interval
            # icc_values[f"{ft}"] = {
            #     "icc": icc,
            #     "ci95": [ci95[0], ci95[1]]
            # }
            icc_values[f"{ft}"] = icc
            
        except Exception as e:
            logger.error(f"Error calculating ICC for feature {ft}: {e}")
            # icc_values[f"{ft}"] = {
            #     "icc": np.nan,
            #     "ci95": [np.nan, np.nan]
            # }
            icc_values[f"{ft}"] = np.nan
    
    # Return results with group name
    return {group_name: icc_values}

def process_files_group(args: Tuple[List[str], logging.Logger, Optional[List[str]]]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Process a single group of files for multiprocessing
    
    Args:
        args: Tuple containing list of files, logger instance, and optional selected features
        
    Returns:
        Tuple containing group name and ICC values with confidence intervals
    """
    files_list, logger, selected_features = args
    result = calculate_icc(files_list, logger, selected_features)
    group_name = list(result.keys())[0]
    return group_name, result[group_name]

def analyze_multiple_groups(file_groups: List[List[str]], n_processes: int = None, logger=None, selected_features: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Analyze ICC for multiple groups of files using parallel processing
    
    Args:
        file_groups: List of file groups, where each group is a list of file paths
        n_processes: Number of processes to use (default: all available CPUs)
        logger: Logger instance for recording progress
        selected_features: Optional list of feature names to analyze (if None, all common features will be analyzed)
        
    Returns:
        Dictionary containing ICC values and 95% confidence intervals for all file groups
    """
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()
    
    n_processes = min(n_processes, len(file_groups))
    
    all_results = {}
    total = len(file_groups)
    
    # Prepare arguments for multiprocessing
    process_args = [(group, logger, selected_features) for group in file_groups]
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        # Use imap_unordered for faster results
        for i, (group_name, group_results) in enumerate(pool.imap_unordered(process_files_group, process_args)):
            # Merge results
            all_results[group_name] = group_results
            
            # Create progress bar
            progress = int((i + 1) / total * 50)  # 50 is the length of the progress bar
            bar = "█" * progress + "-" * (50 - progress)
            percent = (i + 1) / total * 100
            logger.info(f"\r[{bar}] {percent:.2f}% ({i+1}/{total})")
    
    return all_results

def parse_files_groups(files_input: str) -> List[List[str]]:
    """
    Parse user input for file groups
    
    Args:
        files_input: String containing file groups in format "file1.csv,file2.csv,file3.csv;file4.csv,file5.csv,file6.csv"
        
    Returns:
        List of file groups, where each group is a list of file paths
    """
    groups = []
    for group_str in files_input.split(';'):
        if ',' in group_str:
            files = [f.strip() for f in group_str.split(',')]
            if len(files) >= 2:  # Ensure at least two files for ICC calculation
                groups.append(files)
    return groups

def is_data_file(filename: str) -> bool:
    """
    Check if a file is in a supported data format
    
    Args:
        filename: Name of the file to check
        
    Returns:
        Boolean indicating if the file is in a supported format
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.csv', '.xlsx', '.xls']

def parse_directories(dirs_input: str) -> List[List[str]]:
    """
    Parse user input for directories and generate file groups
    
    Args:
        dirs_input: String containing directory paths in format "dir1,dir2,dir3"
        
    Returns:
        List of file groups, where each group contains files with the same name from different directories
    """
    all_groups = []
    
    # Parse directory list
    dirs = [d.strip() for d in dirs_input.split(',')]
    if len(dirs) < 2:
        return []  # At least two directories required for ICC calculation
    
    # Get data files from each directory
    dir_files = {}
    for dir_path in dirs:
        dir_files[dir_path] = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if is_data_file(f)]
    
    # Group files by name (ignoring extension)
    filename_groups = {}
    for dir_path, files in dir_files.items():
        for file_path in files:
            basename = os.path.splitext(os.path.basename(file_path))[0]
            if basename not in filename_groups:
                filename_groups[basename] = []
            filename_groups[basename].append(file_path)
    
    # Only keep groups with files in at least two directories
    for basename, files in filename_groups.items():
        if len(files) >= 2:
            all_groups.append(files)
    
    return all_groups

def parse_features(features_input: str) -> List[str]:
    """
    Parse user input for feature names
    
    Args:
        features_input: String containing feature names in format "feature1,feature2,feature3"
        
    Returns:
        List of feature names
    """
    if not features_input:
        return []
    return [f.strip() for f in features_input.split(',')]

def main():
    parser = argparse.ArgumentParser(description="对多个CSV/Excel文件计算ICC值")
    
    # 文件组或目录列表，二选一
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--files', type=str, help='文件列表，格式为 "file1.csv,file2.csv,file3.csv;file4.csv,file5.csv,file6.csv"')
    group.add_argument('--dirs', type=str, help='目录列表，格式为 "dir1,dir2,dir3"，将匹配目录中同名的数据文件')
    
    parser.add_argument('--features', type=str, help='要计算ICC的特征列名，格式为 "feature1,feature2,feature3"（不指定则计算所有共同特征）')
    parser.add_argument('--processes', type=int, default=None, help='进程数，默认使用所有可用CPU')
    parser.add_argument('--output', type=str, default='icc_results.json', help='输出结果的JSON文件路径')
    
    args = parser.parse_args()
    
    # 配置日志
    logger = configure_logger(args.output)
    
    # 解析特征列表
    selected_features = parse_features(args.features) if args.features else None
    if selected_features:
        logger.info(f"将只计算以下特征的ICC值: {', '.join(selected_features)}")
    
    # 解析文件组或目录列表
    if args.files:
        file_groups = parse_files_groups(args.files)
    else:
        file_groups = parse_directories(args.dirs)
    
    if not file_groups:
        logger.error("没有有效的文件组可以分析")
        return
    
    logger.info(f"将分析 {len(file_groups)} 组文件")
    
    # 执行分析
    results = analyze_multiple_groups(file_groups, args.processes, logger, selected_features)
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"分析完成，结果已保存到 {args.output}")

if __name__ == "__main__":
    main()
