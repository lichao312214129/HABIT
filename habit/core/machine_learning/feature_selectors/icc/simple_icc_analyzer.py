"""
Simple ICC Analyzer Module

Main analysis module that provides a simple interface for calculating
all reliability metrics across multiple files.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .simple_icc_calculator import (
    calculate_all_icc,
    calculate_cohen_kappa,
    calculate_fleiss_kappa,
    calculate_krippendorff_alpha
)
from .data_processor import (
    load_and_merge_data,
    find_common_indices,
    find_common_columns,
    prepare_long_format,
    validate_data_for_metric
)

logger = logging.getLogger(__name__)


def analyze_features(
    file_paths: List[str],
    metrics: Optional[List[str]] = None,
    selected_features: Optional[List[str]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze reliability metrics for all features across multiple files.
    
    Args:
        file_paths: List of file paths to analyze (each file = one rater/measurement)
        metrics: List of metric types to calculate. Options:
                 - 'icc2', 'icc3' (individual ICC types)
                 - 'all_icc' (all 6 ICC types)
                 - 'cohen' (Cohen's Kappa)
                 - 'fleiss' (Fleiss' Kappa)
                 - 'krippendorff' (Krippendorff's Alpha)
                 Default: ['icc2', 'icc3']
        selected_features: Optional list of feature names to analyze
                          (if None, all common features will be analyzed)
        logger_instance: Optional logger instance for logging progress
        
    Returns:
        Dictionary in format:
        {
            'group_name': {
                'feature_name': {
                    'metric_name': value or None,
                    ...
                },
                ...
            }
        }
        
    Example:
        >>> results = analyze_features(
        ...     ['file1.csv', 'file2.csv'],
        ...     metrics=['icc2', 'icc3', 'cohen']
        ... )
        >>> # Save results
        >>> with open('results.json', 'w') as f:
        ...     json.dump(results, f, indent=4)
    """
    # Use provided logger or create new one
    if logger_instance is None:
        logger_instance = logger
    
    # Default metrics
    if metrics is None:
        metrics = ['icc2', 'icc3']
    
    # Load data
    logger_instance.info(f"Loading {len(file_paths)} files...")
    data_frames, file_names = load_and_merge_data(file_paths)
    
    # Create group name
    group_name = "_vs_".join(file_names)
    logger_instance.info(f"Analyzing group: {group_name}")
    
    # Find common indices and columns
    common_index = find_common_indices(data_frames)
    if len(common_index) == 0:
        logger_instance.error("No common indices found, cannot proceed")
        return {group_name: {}}
    
    common_columns = find_common_columns(data_frames)
    if len(common_columns) == 0:
        logger_instance.error("No common columns found, cannot proceed")
        return {group_name: {}}
    
    # Filter features if selected
    if selected_features:
        target_features = [col for col in common_columns if col in selected_features]
        if not target_features:
            logger_instance.error("No selected features found in common columns")
            return {group_name: {}}
        common_columns = target_features
        logger_instance.info(f"Analyzing {len(common_columns)} selected features")
    else:
        logger_instance.info(f"Analyzing {len(common_columns)} common features")
    
    # Filter dataframes with common indices
    for i in range(len(data_frames)):
        data_frames[i] = data_frames[i].loc[common_index]
    
    # Analyze each feature
    feature_results = {}
    total = len(common_columns)
    
    for i, feature_name in enumerate(common_columns):
        try:
            # Prepare long-format data
            data = prepare_long_format(data_frames, feature_name, common_index)
            
            # Calculate requested metrics
            ft_results = {}
            
            if 'all_icc' in metrics:
                # Calculate all ICC types
                icc_results = calculate_all_icc(data)
                ft_results.update(icc_results)
            else:
                # Calculate individual ICC types
                for metric in metrics:
                    if metric.startswith('icc'):
                        icc_results = calculate_all_icc(data)
                        if metric in icc_results:
                            ft_results[metric] = icc_results[metric]
            
            if 'cohen' in metrics:
                kappa_results = calculate_cohen_kappa(data)
                ft_results.update(kappa_results)
            
            if 'fleiss' in metrics:
                fleiss_results = calculate_fleiss_kappa(data)
                ft_results.update(fleiss_results)
            
            if 'krippendorff' in metrics:
                krippendorff_results = calculate_krippendorff_alpha(data)
                ft_results.update(krippendorff_results)
            
            feature_results[feature_name] = ft_results
            
            # Progress logging
            if (i + 1) % 10 == 0 or (i + 1) == total:
                progress = (i + 1) / total * 100
                logger_instance.info(f"Progress: {progress:.1f}% ({i+1}/{total})")
                
        except Exception as e:
            logger_instance.error(f"Error analyzing feature {feature_name}: {e}")
            # Return None for all requested metrics
            ft_results = {}
            if 'all_icc' in metrics:
                for icc_type in ['icc1', 'icc2', 'icc3', 'icc1k', 'icc2k', 'icc3k']:
                    ft_results[icc_type] = None
            else:
                for metric in metrics:
                    ft_results[metric] = None
            feature_results[feature_name] = ft_results
    
    logger_instance.info(f"Analysis complete: {len(feature_results)} features")
    return {group_name: feature_results}


def save_results(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Save analysis results to JSON file.
    
    Args:
        results: Results dictionary from analyze_features()
        output_path: Path to output JSON file
        logger_instance: Optional logger instance
    """
    if logger_instance is None:
        logger_instance = logger
    
    try:
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger_instance.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger_instance.error(f"Error saving results: {e}")
        raise


def print_summary(results: Dict[str, Dict[str, Any]], logger_instance: Optional[logging.Logger] = None) -> None:
    """
    Print summary of analysis results.
    
    Args:
        results: Results dictionary from analyze_features()
        logger_instance: Optional logger instance
    """
    if logger_instance is None:
        logger_instance = logger
    
    for group_name, group_results in results.items():
        logger_instance.info(f"\nGroup: {group_name}")
        
        # Count features with valid results
        total_features = len(group_results)
        valid_features = sum(
            1 for ft_results in group_results.values()
            if any(v is not None for v in ft_results.values())
        )
        
        logger_instance.info(f"Total features: {total_features}")
        logger_instance.info(f"Features with valid results: {valid_features}")
        
        # Print sample results
        for feature_name, ft_results in list(group_results.items())[:3]:
            logger_instance.info(f"  {feature_name}: {ft_results}")
        
        if total_features > 3:
            logger_instance.info(f"  ... and {total_features - 3} more features")