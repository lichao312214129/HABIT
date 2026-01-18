"""
Data Processor Module

Handles data loading, merging, and preparation for ICC analysis.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_and_merge_data(file_paths: List[str], index_col: int = 0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load multiple CSV/Excel files and merge them.
    
    Args:
        file_paths: List of file paths to load
        index_col: Column to use as index (default: first column)
        
    Returns:
        Tuple of (merged DataFrame, list of file names)
        
    Raises:
        ValueError: If files cannot be loaded or have no common indices
    """
    data_frames = []
    file_names = []
    
    for file_path in file_paths:
        try:
            df = _read_file(file_path, index_col)
            data_frames.append(df)
            file_names.append(os.path.splitext(os.path.basename(file_path))[0])
            logger.info(f"Loaded {file_path}: shape {df.shape}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise ValueError(f"Failed to load file {file_path}: {e}")
    
    return data_frames, file_names


def find_common_indices(data_frames: List[pd.DataFrame]) -> List:
    """
    Find common indices across all DataFrames.
    
    Args:
        data_frames: List of DataFrames
        
    Returns:
        List of common indices
    """
    if not data_frames:
        return []
    
    common_index = set(data_frames[0].index)
    for df in data_frames[1:]:
        common_index = common_index.intersection(df.index)
    
    common_index = list(common_index)
    
    if len(common_index) == 0:
        logger.warning("No common indices found across all DataFrames")
    else:
        logger.info(f"Found {len(common_index)} common indices")
    
    return common_index


def find_common_columns(data_frames: List[pd.DataFrame]) -> List[str]:
    """
    Find common columns across all DataFrames.
    
    Args:
        data_frames: List of DataFrames
        
    Returns:
        List of common column names
    """
    if not data_frames:
        return []
    
    common_columns = set(data_frames[0].columns)
    for df in data_frames[1:]:
        common_columns = common_columns.intersection(df.columns)
    
    common_columns = list(common_columns)
    
    if len(common_columns) == 0:
        logger.warning("No common columns found across all DataFrames")
    else:
        logger.info(f"Found {len(common_columns)} common columns")
    
    return common_columns


def prepare_long_format(
    data_frames: List[pd.DataFrame],
    feature_name: str,
    common_index: List
) -> pd.DataFrame:
    """
    Prepare data in long format for reliability metric calculation.
    
    Args:
        data_frames: List of DataFrames from different raters/measurements
        feature_name: Name of feature column to analyze
        common_index: List of common indices across all DataFrames
        
    Returns:
        Long-format DataFrame with columns: ['value', 'reader', 'target']
    """
    data_list = []
    
    for reader_idx, df in enumerate(data_frames):
        # Extract feature values for common indices
        df_feature = pd.DataFrame(df.loc[common_index, feature_name])
        df_feature.columns = ['value']
        
        # Add reader and target columns
        df_feature['reader'] = reader_idx + 1
        df_feature['target'] = range(len(common_index))
        
        data_list.append(df_feature)
    
    # Concatenate all data
    result = pd.concat(data_list, axis=0, ignore_index=True)
    return result


def validate_data_for_metric(
    data: pd.DataFrame,
    metric_type: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate data for a specific metric type.
    
    Args:
        data: Long-format DataFrame with columns: 'target', 'reader', 'value'
        metric_type: Type of metric ('icc', 'cohen', 'fleiss', 'krippendorff')
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required columns
    required_cols = ['target', 'reader', 'value']
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        return False, f"Missing required columns: {missing}"
    
    # Check minimum requirements
    n_targets = data['target'].nunique()
    n_raters = data['reader'].nunique()
    
    if n_targets < 2:
        return False, f"Need at least 2 targets, got {n_targets}"
    
    if n_raters < 2:
        return False, f"Need at least 2 raters, got {n_raters}"
    
    # Metric-specific validation
    if metric_type == 'cohen' and n_raters != 2:
        return False, f"Cohen's Kappa requires exactly 2 raters, got {n_raters}"
    
    return True, None


def _read_file(file_path: str, index_col: int) -> pd.DataFrame:
    """
    Read CSV or Excel file based on file extension.
    
    Args:
        file_path: Path to the file
        index_col: Column to use as index
        
    Returns:
        DataFrame
        
    Raises:
        ValueError: If file format is not supported
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        return pd.read_csv(file_path, index_col=index_col)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path, index_col=index_col)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .csv, .xlsx, .xls")