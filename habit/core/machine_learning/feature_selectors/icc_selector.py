"""
ICC (Intraclass Correlation Coefficient) Feature Selector

Uses ICC to evaluate feature reproducibility and selects features with high ICC values
"""
import pandas as pd
import numpy as np
import pingouin as pg
import json
import os
import multiprocessing
import logging
import argparse
from itertools import product
from typing import Dict, List, Tuple, Any, Union
from .selector_registry import register_selector

# Logger configuration will be set in the main function to use the output path

def configure_logger(output_path: str):
    """
    Configure logger to save logs in the same directory as the output file
    
    Args:
        output_path: Output file path
    """
    from habit.utils.log_utils import setup_logger
    
    output_dir = os.path.dirname(os.path.abspath(output_path))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return setup_logger(
        name='icc_selector',
        output_dir=output_dir,
        log_filename='icc_analysis.log',
        level=logging.INFO
    )

def read_file(file_path: str, index_col=0) -> pd.DataFrame:
    """
    Read CSV or Excel file based on file extension
    
    Args:
        file_path: Path to the file
        index_col: Column to use as index, can be column name or index, default is first column (0)
        
    Returns:
        DataFrame object
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        return pd.read_csv(file_path, index_col=index_col)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path, index_col=index_col)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .csv, .xlsx, .xls")


def calculate_icc(files_list: List[Union[str, pd.DataFrame]], logger, 
                 column_selection=None, index_col=0) -> Dict[str, Dict[str, float]]:
    """
    Calculate ICC values between multiple files or DataFrames
    
    Args:
        files_list: List containing multiple file paths or DataFrames
        logger: Logger instance
        column_selection: Column selection configuration, only valid in file mode
        index_col: Column to use as index, can be column name or index
        
    Returns:
        Dictionary containing ICC values, format {feature_name: icc_value}
    """
    # Get file names (without path and extension)
    file_names = []
    for f in files_list:
        if isinstance(f, str):
            file_names.append(os.path.splitext(os.path.basename(f))[0])
        else:
            file_names.append(f"df_{len(file_names)}")
    group_name = "_vs_".join(file_names)
    
    # Read all files or use DataFrames
    data_frames = []
    try:
        for file_path in files_list:
            if isinstance(file_path, str):
                df = read_file(file_path, index_col=index_col)
                
                # Apply column selection
                if column_selection:
                    selected_columns = []
                    for sel in column_selection:
                        if isinstance(sel, slice):
                            # Handle slices by applying them to DataFrame column indices
                            selected_columns.extend(df.columns[sel])
                        else:
                            # Handle column names
                            if sel in df.columns:
                                selected_columns.append(sel)
                            else:
                                logger.warning(f"Column name '{sel}' does not exist in file {file_path}, ignored")
                    
                    if selected_columns:
                        df = df[selected_columns]
                    else:
                        logger.warning(f"No matching columns found in file {file_path}, will use all columns")
                
                data_frames.append(df)
            else:
                data_frames.append(file_path)
    except Exception as e:
        logger.error(f"Error reading file or processing DataFrame: {e}")
        return {group_name: {}}
    
    # Find common indices across all DataFrames
    common_index = set(data_frames[0].index)
    for df in data_frames[1:]:
        common_index = common_index.intersection(df.index)
    
    common_index = list(common_index)
    
    if len(common_index) == 0:
        logger.warning(f"{group_name} has no common patient IDs")
        return {group_name: {}}
    
    # Filter data with common indices
    for i in range(len(data_frames)):
        data_frames[i] = data_frames[i].loc[common_index]
    
    # Find common columns across all DataFrames
    common_columns = set(data_frames[0].columns)
    for df in data_frames[1:]:
        common_columns = common_columns.intersection(df.columns)
    
    common_columns = list(common_columns)
    
    if len(common_columns) == 0:
        logger.warning(f"{group_name} has no common feature columns")
        return {group_name: {}}
    
    # Calculate ICC for each feature
    icc_values = {}
    total = len(common_columns)
    
    for i, ft in enumerate(common_columns):
        try:
            # Prepare data
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
            
            # Use feature name as key
            icc_values[f"{ft}"] = icc
            
        except Exception as e:
            logger.error(f"Error calculating ICC for feature {ft}: {e}")
            icc_values[f"{ft}"] = np.nan
    
    # Return result with file group name
    return {group_name: icc_values}

def process_files_group(args: Tuple[List[str], logging.Logger, Any, Any]) -> Tuple[str, Dict[str, float]]:
    """
    Function to process a single file group, used for multiprocessing
    
    Args:
        args: Tuple containing file list, logger, column selection configuration, and index column
        
    Returns:
        Tuple containing file group name and ICC values
    """
    files_list, logger, column_selection, index_col = args
    result = calculate_icc(files_list, logger, column_selection, index_col)
    group_name = list(result.keys())[0]
    return group_name, result[group_name]

def analyze_multiple_groups(file_groups: List[List[str]], n_processes: int = None, 
                          logger=None, column_selection=None, index_col=0) -> Dict[str, Dict[str, float]]:
    """
    Analyze ICC for multiple file groups
    
    Args:
        file_groups: List of file groups, each element is a list of file paths
        n_processes: Number of processes, default is None (use all available CPUs)
        logger: Logger instance
        column_selection: Column selection configuration, only valid in file mode
        index_col: Column to use as index, can be column name or index
        
    Returns:
        Dictionary containing ICC values for all file groups
    """
    if n_processes is None:
        n_processes = 4
    
    n_processes = min(n_processes, len(file_groups))
    
    all_results = {}
    total = len(file_groups)
    
    # Prepare multiprocessing arguments
    process_args = [(group, logger, column_selection, index_col) for group in file_groups]
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        # Use imap_unordered to get results
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
    Parse user input file groups
    
    Args:
        files_input: User input file group string, format "file1.csv,file2.csv,file3.csv;file4.csv,file5.csv,file6.csv"
        
    Returns:
        List of file groups
    """
    groups = []
    for group_str in files_input.split(';'):
        if ',' in group_str:
            files = [f.strip() for f in group_str.split(',')]
            if len(files) >= 2:  # Ensure at least two files are present to calculate ICC
                groups.append(files)
    return groups

def is_data_file(filename: str) -> bool:
    """
    Check if file is a supported data file format
    
    Args:
        filename: File name
        
    Returns:
        Whether the file is a supported data file
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.csv', '.xlsx', '.xls']

def parse_directories(dirs_input: str) -> List[List[str]]:
    """
    Parse user input directory list and generate file groups
    
    Args:
        dirs_input: User input directory list string, format "dir1,dir2,dir3"
        
    Returns:
        List of file groups
    """
    all_groups = []
    
    # Parse directory list
    dirs = [d.strip() for d in dirs_input.split(',')]
    if len(dirs) < 2:
        return []  # At least two directories are needed to calculate ICC
    
    # Get data files from each directory
    dir_files = {}
    for dir_path in dirs:
        dir_files[dir_path] = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if is_data_file(f)]
    
    # Group files by filename (without extension)
    filename_groups = {}
    for dir_path, files in dir_files.items():
        for file_path in files:
            basename = os.path.splitext(os.path.basename(file_path))[0]
            if basename not in filename_groups:
                filename_groups[basename] = []
            filename_groups[basename].append(file_path)
    
    # Only keep file groups that exist in at least two directories
    for basename, files in filename_groups.items():
        if len(files) >= 2:
            all_groups.append(files)
    
    return all_groups

@register_selector('icc')
def icc_selector(icc_results: Dict[str, Dict[str, float]], keys: List[str], threshold: float = 0.75) -> Dict[str, List[str]]:
    """
    Filter features based on ICC threshold
    
    Args:
        icc_results: ICC values of features in different groups, Dict[str, Dict[str, float]]
        keys: which groups to select features from, List[str]
        threshold: ICC threshold, float
        
    Returns:
        Dictionary containing selected features, format {group_name: [selected_features]}
    """
    # if icc results is a file, read it
    if isinstance(icc_results, str):
        with open(icc_results, 'r') as f:
            icc_results = json.load(f)

    selected_features = {}
    
    for group_name, features in icc_results.items():
        selected = [feature for feature, icc in features.items() if icc >= threshold]
        selected_features[group_name] = selected
    
    results = {key: selected_features[key] for key in keys}
    results = list(results.values())
    # flatten the list
    results = [item for sublist in results for item in sublist]
    return results

def main():
    parser = argparse.ArgumentParser(description="Calculate ICC for multiple CSV/Excel files")
    
    # File group or directory list, choose one
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--files', type=str, help='File list, format "file1.csv,file2.csv,file3.csv;file4.csv,file5.csv,file6.csv"')
    group.add_argument('--dirs', type=str, help='Directory list, format "dir1,dir2,dir3", will match data files with the same name in directories')
    
    parser.add_argument('--processes', type=int, default=None, help='Number of processes, default uses all available CPUs')
    parser.add_argument('--output', type=str, default='icc_results.json', help='Path to the output JSON file')
    parser.add_argument('--columns', type=str, help='Column selection, only valid in file mode. Supported formats:\n'
                      '1. Number range: 2:10 means select columns 2 to 10\n'
                      '2. Start position: 2: means from column 2\n'
                      '3. Column name list: col1,col2,col3\n'
                      '4. Multiple ranges: 2:5,7:10,col1,col2')
    parser.add_argument('--index-col', type=str, default='0', help='Column to use as index (uid), can be column name or numeric index, default is 0 (first column)')
    
    args = parser.parse_args()
    
    # Configure logger
    logger = configure_logger(args.output)
    
    # Parse index column parameter
    index_col = args.index_col
    if index_col.isdigit():
        index_col = int(index_col)
    logger.info(f"Using column '{index_col}' as index (uid)")
    
    # Parse column selection, only valid in file mode
    columns = None
    if args.files and args.columns:
        columns = parse_column_selection(args.columns)
        logger.info(f"Column selection set: {args.columns}")
    elif args.columns and not args.files:
        logger.warning("Column selection only valid in file mode, --columns parameter ignored")
    
    # Parse file group or directory list
    if args.files:
        file_groups = parse_files_groups(args.files)
    else:
        file_groups = parse_directories(args.dirs)
    
    if not file_groups:
        logger.error("No valid file groups to analyze")
        return
    
    logger.info(f"Analyzing {len(file_groups)} file groups")
    
    # Execute analysis
    results = analyze_multiple_groups(file_groups, args.processes, logger, columns, index_col)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Analysis completed, results saved to {args.output}")

def parse_column_selection(column_str: str) -> List[Union[slice, str]]:
    """
    Parse column selection string
    
    Args:
        column_str: Column selection string, e.g. "2:10,col1,col2,5:"
        
    Returns:
        List of column selections, containing slice objects or column names
    """
    selections = []
    for part in column_str.split(','):
        part = part.strip()
        if ':' in part:
            # Handle slice format
            parts = part.split(':')
            if len(parts) == 2:
                start, end = parts
                start = int(start) if start.strip() else None
                end = int(end) if end.strip() else None
                selections.append(slice(start, end))
        elif part and not part.isdigit():  # Ensure not a number, i.e., column name
            # Handle column name
            selections.append(part)
    return selections

def filter_columns(df: pd.DataFrame, column_selection: List[Union[int, str, slice]]) -> pd.DataFrame:
    """
    Filter DataFrame based on column selection
    
    Args:
        df: Original DataFrame
        column_selection: List of column selections
        
    Returns:
        Filtered DataFrame
    """
    selected_columns = []
    for sel in column_selection:
        if isinstance(sel, slice):
            # Handle slice
            selected_columns.extend(df.columns[sel])
        elif isinstance(sel, int):
            # Handle integer index
            selected_columns.append(df.columns[sel])
        else:
            # Handle column name
            if sel in df.columns:
                selected_columns.append(sel)
    return df[selected_columns]

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:  # If no command line arguments are provided
        sys.argv.extend([
            '--files', '../demo_data/breast_cancer_dataset.csv, ../demo_data/breast_cancer_dataset.csv',
            '--columns', '1:',
            '--index-col', 'subjID',  # Use column named 'id' as index
            '--processes', '6',
            '--output', '../demo_data/results/icc_results.json',
        ])
    main() 
