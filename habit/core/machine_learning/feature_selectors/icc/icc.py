"""
Reliability Analysis Module

This module calculates various reliability metrics (ICC, Kappa, Fleiss' Kappa, etc.)
between multiple files (raters/measurements) for radiomics feature analysis.

Supports:
    - All 6 ICC types (ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k)
    - Cohen's Kappa (2 raters)
    - Fleiss' Kappa (multiple raters)
    - Krippendorff's Alpha
    - Custom user-defined metrics

Author: Li Chao
Email: lich356@mail.sysu.edu.cn
"""

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
from typing import Dict, List, Tuple, Any, Optional, Set, Union

from habit.utils.log_utils import setup_logger, get_module_logger
from .reliability_metrics import (
    ICCType,
    KappaType,
    MetricResult,
    BaseReliabilityMetric,
    ICCMetric,
    MultiICCMetric,
    CohenKappaMetric,
    FleissKappaMetric,
    KrippendorffAlphaMetric,
    create_metric,
    calculate_reliability,
    get_available_metrics
)

# Module logger for use throughout the file
logger = get_module_logger(__name__)


# Default metric configuration: ICC(3,1) for backward compatibility
DEFAULT_METRICS = ["icc3"]


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


def prepare_long_format_data(
    data_frames: List[pd.DataFrame],
    feature_name: str,
    common_index: List
) -> pd.DataFrame:
    """
    Prepare data in long format for reliability metric calculation.
    
    Args:
        data_frames: List of DataFrames from different raters/measurements
        feature_name: Name of the feature column to analyze
        common_index: List of common indices across all DataFrames
        
    Returns:
        Long-format DataFrame with columns: [feature_name, 'reader', 'target']
    """
    data_list = []
    for reader_idx, df in enumerate(data_frames):
        df_feature = pd.DataFrame(df.loc[common_index, feature_name])
        df_feature["reader"] = np.ones(df_feature.shape[0]) * (reader_idx + 1)
        df_feature["target"] = range(df_feature.shape[0])
        data_list.append(df_feature)
    
    return pd.concat(data_list, axis=0)


def calculate_reliability_metrics(
    files_list: List[str],
    logger: logging.Logger,
    selected_features: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    return_full_results: bool = False,
    **metric_kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate reliability metrics (ICC, Kappa, etc.) between multiple files.
    
    This is the main function for calculating reliability metrics. It supports
    multiple metric types and can return either simple values or full results
    with confidence intervals.
    
    Args:
        files_list: List of file paths to analyze (each file represents a rater/measurement)
        logger: Logger instance for recording progress and errors
        selected_features: Optional list of feature names to analyze 
                          (if None, all common features will be analyzed)
        metrics: List of metric types to calculate. Options include:
                - "icc1", "icc2", "icc3", "icc1k", "icc2k", "icc3k" (ICC types)
                - "multi_icc" (all ICC types at once)
                - "cohen_kappa", "fleiss_kappa" (Kappa coefficients)
                - "krippendorff" (Krippendorff's Alpha)
                Default: ["icc3"] for backward compatibility
        return_full_results: If True, return MetricResult objects with CI and p-value.
                            If False, return simple float values (backward compatible)
        **metric_kwargs: Additional parameters passed to metric calculators
                        (e.g., nan_policy, weights for Kappa)
        
    Returns:
        Dictionary in format:
        {group_name: {feature_name: value or MetricResult, ...}, ...}
        
        If multiple metrics are requested and return_full_results=True:
        {group_name: {feature_name: {metric_name: MetricResult, ...}, ...}, ...}
        
    Example:
        >>> # Calculate ICC(3,1) - default behavior for backward compatibility
        >>> results = calculate_reliability_metrics(
        ...     ['rater1.csv', 'rater2.csv'],
        ...     logger
        ... )
        
        >>> # Calculate multiple ICC types
        >>> results = calculate_reliability_metrics(
        ...     ['rater1.csv', 'rater2.csv'],
        ...     logger,
        ...     metrics=['icc2', 'icc3', 'icc2k', 'icc3k'],
        ...     return_full_results=True
        ... )
        
        >>> # Calculate Fleiss' Kappa for categorical features
        >>> results = calculate_reliability_metrics(
        ...     ['rater1.csv', 'rater2.csv', 'rater3.csv'],
        ...     logger,
        ...     metrics=['fleiss_kappa']
        ... )
    """
    # Use default metrics if not specified (backward compatibility)
    if metrics is None:
        metrics = DEFAULT_METRICS
    
    # Get filenames without path and extension for group naming
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
        target_columns = [col for col in common_columns if col in selected_features]
        if not target_columns:
            logger.warning(f"{group_name} has no selected features in common")
            return {group_name: {}}
        common_columns = target_columns
        logger.info(f"Analyzing {len(common_columns)} selected features")
    
    # Create metric instances
    metric_instances = []
    for metric_name in metrics:
        try:
            metric = create_metric(metric_name, **metric_kwargs)
            metric_instances.append((metric_name, metric))
        except Exception as e:
            logger.error(f"Error creating metric {metric_name}: {e}")
    
    if not metric_instances:
        logger.error("No valid metrics to calculate")
        return {group_name: {}}
    
    # Calculate metrics for each feature
    feature_results = {}
    total = len(common_columns)
    
    for i, ft in enumerate(common_columns):
        try:
            # Prepare long-format data for this feature
            data = prepare_long_format_data(data_frames, ft, common_index)
            
            # Calculate each metric
            if len(metrics) == 1 and not return_full_results:
                # Simple case: single metric, return just the value (backward compatible)
                metric_name, metric = metric_instances[0]
                result = metric.calculate(
                    data=data,
                    targets='target',
                    raters='reader',
                    ratings=ft
                )
                
                # Handle MultiICCMetric which returns a dict
                if isinstance(result, dict):
                    # For multi_icc, use the first specified type or ICC3
                    if hasattr(metric, 'icc_types') and metric.icc_types:
                        feature_results[ft] = result[metric.icc_types[0].name].value
                    else:
                        feature_results[ft] = list(result.values())[0].value
                else:
                    feature_results[ft] = result.value
            else:
                # Multiple metrics or full results requested
                ft_results = {}
                for metric_name, metric in metric_instances:
                    result = metric.calculate(
                        data=data,
                        targets='target',
                        raters='reader',
                        ratings=ft
                    )
                    result = result.to_dict()
                    ft_results[metric_name] = result
                
                feature_results[ft] = ft_results
                
        except Exception as e:
            logger.error(f"Error calculating metrics for feature {ft}: {e}")
            if len(metrics) == 1 and not return_full_results:
                feature_results[ft] = np.nan
            else:
                feature_results[ft] = {m: np.nan for m, _ in metric_instances}
    
    return {group_name: feature_results}


def calculate_icc(
    files_list: List[str],
    logger: logging.Logger,
    selected_features: Optional[List[str]] = None,
    icc_type: str = "icc3"
) -> Dict[str, Dict[str, float]]:
    """
    Calculate ICC (Intraclass Correlation Coefficient) values between multiple files.
    
    This function provides backward compatibility with the original ICC calculation.
    For more flexibility, use calculate_reliability_metrics() instead.
    
    Args:
        files_list: List of file paths to analyze
        logger: Logger instance for recording progress and errors
        selected_features: Optional list of feature names to analyze 
                          (if None, all common features will be analyzed)
        icc_type: Type of ICC to calculate. Options:
                 - "icc1": Single raters, absolute agreement
                 - "icc2": Single random raters, absolute agreement (default in some tools)
                 - "icc3": Single fixed raters, consistency (most commonly used)
                 - "icc1k": Average raters, absolute agreement
                 - "icc2k": Average random raters
                 - "icc3k": Average fixed raters
                 Default: "icc3" for backward compatibility
        
    Returns:
        Dictionary containing ICC values in the format 
        {group_name: {feature_name: icc_value, ...}}
        
    Example:
        >>> results = calculate_icc(['file1.csv', 'file2.csv'], logger)
        >>> results = calculate_icc(['file1.csv', 'file2.csv'], logger, icc_type="icc2")
    """
    return calculate_reliability_metrics(
        files_list=files_list,
        logger=logger,
        selected_features=selected_features,
        metrics=[icc_type],
        return_full_results=False
    )


def process_files_group(
    args: Tuple[List[str], logging.Logger, Optional[List[str]], Optional[List[str]], bool, Dict]
) -> Tuple[str, Dict[str, Any]]:
    """
    Process a single group of files for multiprocessing.
    
    Args:
        args: Tuple containing:
              - files_list: List of file paths
              - logger: Logger instance
              - selected_features: Optional list of features to analyze
              - metrics: List of metric types to calculate
              - return_full_results: Whether to return full MetricResult objects
              - metric_kwargs: Additional keyword arguments for metrics
        
    Returns:
        Tuple containing group name and metric values
    """
    files_list, logger, selected_features, metrics, return_full_results, metric_kwargs = args
    result = calculate_reliability_metrics(
        files_list, logger, selected_features, metrics, return_full_results, **metric_kwargs
    )
    group_name = list(result.keys())[0]
    return group_name, result[group_name]


def analyze_multiple_groups(
    file_groups: List[List[str]],
    n_processes: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    selected_features: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    return_full_results: bool = False,
    **metric_kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze reliability metrics for multiple groups of files using parallel processing.
    
    Args:
        file_groups: List of file groups, where each group is a list of file paths
        n_processes: Number of processes to use (default: all available CPUs)
        logger: Logger instance for recording progress
        selected_features: Optional list of feature names to analyze
        metrics: List of metric types to calculate (default: ["icc3"])
        return_full_results: Whether to return full MetricResult objects with CI
        **metric_kwargs: Additional parameters for metric calculators
        
    Returns:
        Dictionary containing reliability metrics for all file groups
        
    Example:
        >>> file_groups = [
        ...     ['group1_rater1.csv', 'group1_rater2.csv'],
        ...     ['group2_rater1.csv', 'group2_rater2.csv']
        ... ]
        >>> results = analyze_multiple_groups(
        ...     file_groups,
        ...     metrics=['icc2', 'icc3', 'fleiss_kappa'],
        ...     return_full_results=True
        ... )
    """
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()
    
    n_processes = min(n_processes, len(file_groups))
    
    # Use default metrics if not specified
    if metrics is None:
        metrics = DEFAULT_METRICS
    
    all_results = {}
    total = len(file_groups)
    
    # Prepare arguments for multiprocessing
    process_args = [
        (group, logger, selected_features, metrics, return_full_results, metric_kwargs) 
        for group in file_groups
    ]
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        for i, (group_name, group_results) in enumerate(pool.imap_unordered(process_files_group, process_args)):
            all_results[group_name] = group_results
            
            # Create progress bar
            progress = int((i + 1) / total * 50)
            bar = "█" * progress + "-" * (50 - progress)
            percent = (i + 1) / total * 100
            logger.info(f"\r[{bar}] {percent:.2f}% ({i+1}/{total})")
    
    return all_results


def parse_files_groups(files_input: str) -> List[List[str]]:
    """
    Parse user input for file groups.
    
    Args:
        files_input: String containing file groups in format 
                    "file1.csv,file2.csv,file3.csv;file4.csv,file5.csv,file6.csv"
        
    Returns:
        List of file groups, where each group is a list of file paths
    """
    groups = []
    for group_str in files_input.split(';'):
        if ',' in group_str:
            files = [f.strip() for f in group_str.split(',')]
            if len(files) >= 2:
                groups.append(files)
    return groups


def is_data_file(filename: str) -> bool:
    """
    Check if a file is in a supported data format.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        Boolean indicating if the file is in a supported format
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.csv', '.xlsx', '.xls']


def parse_directories(dirs_input: str) -> List[List[str]]:
    """
    Parse user input for directories and generate file groups.
    
    Args:
        dirs_input: String containing directory paths in format "dir1,dir2,dir3"
        
    Returns:
        List of file groups, where each group contains files with the same name 
        from different directories
    """
    all_groups = []
    
    dirs = [d.strip() for d in dirs_input.split(',')]
    if len(dirs) < 2:
        return []
    
    dir_files = {}
    for dir_path in dirs:
        dir_files[dir_path] = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if is_data_file(f)]
    
    filename_groups = {}
    for dir_path, files in dir_files.items():
        for file_path in files:
            basename = os.path.splitext(os.path.basename(file_path))[0]
            if basename not in filename_groups:
                filename_groups[basename] = []
            filename_groups[basename].append(file_path)
    
    for basename, files in filename_groups.items():
        if len(files) >= 2:
            all_groups.append(files)
    
    return all_groups


def parse_features(features_input: str) -> List[str]:
    """
    Parse user input for feature names.
    
    Args:
        features_input: String containing feature names in format "feature1,feature2,feature3"
        
    Returns:
        List of feature names
    """
    if not features_input:
        return []
    return [f.strip() for f in features_input.split(',')]


def parse_metrics(metrics_input: str) -> List[str]:
    """
    Parse user input for metric types.
    
    Args:
        metrics_input: String containing metric types in format "icc3,fleiss_kappa"
        
    Returns:
        List of metric type names
    """
    if not metrics_input:
        return DEFAULT_METRICS
    return [m.strip().lower() for m in metrics_input.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Calculate reliability metrics (ICC, Kappa, etc.) for multiple CSV/Excel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Metrics:
    ICC Types (Intraclass Correlation Coefficient):
        icc1    - Single raters, absolute agreement, one-way random model
        icc2    - Single random raters, absolute agreement, two-way random model
        icc3    - Single fixed raters, consistency, two-way mixed model (default)
        icc1k   - Average raters, absolute agreement
        icc2k   - Average random raters, absolute agreement
        icc3k   - Average fixed raters, consistency
        multi_icc - Calculate all 6 ICC types at once
    
    Kappa Coefficients:
        cohen_kappa  - Cohen's Kappa for 2 raters
        fleiss_kappa - Fleiss' Kappa for multiple raters
    
    Other:
        krippendorff - Krippendorff's Alpha

Examples:
    # Calculate ICC(3,1) - default behavior
    python icc.py --files "rater1.csv,rater2.csv" --output results.json
    
    # Calculate multiple metrics
    python icc.py --files "rater1.csv,rater2.csv" --metrics "icc2,icc3,fleiss_kappa" --output results.json
    
    # Calculate all ICC types with full results (including CI)
    python icc.py --files "rater1.csv,rater2.csv" --metrics "multi_icc" --full --output results.json
        """
    )
    
    # File input options (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--files', type=str, 
        help='File list in format "file1.csv,file2.csv;file3.csv,file4.csv"'
    )
    group.add_argument(
        '--dirs', type=str, 
        help='Directory list in format "dir1,dir2", matches same-named files'
    )
    
    # Feature selection
    parser.add_argument(
        '--features', type=str, 
        help='Feature columns to analyze in format "feature1,feature2" (default: all common features)'
    )
    
    # Metric selection
    parser.add_argument(
        '--metrics', type=str, 
        default='icc3',
        help='Metrics to calculate in format "icc3,fleiss_kappa" (default: icc3)'
    )
    
    # Output options
    parser.add_argument(
        '--full', action='store_true',
        help='Return full results with confidence intervals and p-values'
    )
    parser.add_argument(
        '--processes', type=int, default=None, 
        help='Number of processes (default: all CPUs)'
    )
    parser.add_argument(
        '--output', type=str, default='reliability_results.json', 
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    # Configure logger
    logger = configure_logger(args.output)
    
    # Parse features
    selected_features = parse_features(args.features) if args.features else None
    if selected_features:
        logger.info(f"Will analyze features: {', '.join(selected_features)}")
    
    # Parse metrics
    metrics = parse_metrics(args.metrics)
    logger.info(f"Will calculate metrics: {', '.join(metrics)}")
    
    # Parse file groups
    if args.files:
        file_groups = parse_files_groups(args.files)
    else:
        file_groups = parse_directories(args.dirs)
    
    if not file_groups:
        logger.error("No valid file groups to analyze")
        return
    
    logger.info(f"Will analyze {len(file_groups)} file groups")
    
    # Execute analysis
    results = analyze_multiple_groups(
        file_groups, 
        args.processes, 
        logger, 
        selected_features,
        metrics=metrics,
        return_full_results=args.full
    )
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    results = convert_to_serializable(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Analysis complete, results saved to {args.output}")


if __name__ == "__main__":
    main()
