"""
Univariate Logistic Regression Feature Selector

Performs feature selection using univariate logistic regression
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import statsmodels.formula.api as smf
from tqdm import tqdm
from pathlib import Path

from .selector_registry import register_selector

def detect_file_type(input_path: str) -> Optional[str]:
    """
    Automatically detect file type
    
    Args:
        input_path: str, Input file path
        
    Returns:
        Optional[str]: Detected file type, returns None if cannot detect
    """
    file_ext = Path(input_path).suffix.lower()
    file_types = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.parquet': 'parquet',
        '.json': 'json',
        '.pkl': 'pickle',
        '.pickle': 'pickle'
    }
    
    if file_ext in file_types:
        return file_types[file_ext]
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if ',' in first_line and len(first_line.split(',')) > 1:
                return 'csv'
            elif first_line.startswith('{') or first_line.startswith('['):
                return 'json'
    except:
        pass
    
    return None

def load_data(input_data: Union[str, pd.DataFrame], 
              target_column: Optional[str] = None,
              file_type: Optional[str] = None,
              columns: Optional[Union[str, List[str]]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from various formats
    
    Args:
        input_data: Union[str, pd.DataFrame], Input data path or DataFrame object
        target_column: Optional[str], Target variable column name
        file_type: Optional[str], File type
        columns: Optional[Union[str, List[str]]], Feature column selection
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature data and target variable
    """
    if isinstance(input_data, pd.DataFrame):
        data = input_data
    else:
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"Error: File {input_data} does not exist")
        
        if file_type is None:
            file_type = detect_file_type(input_data)
            if file_type is None:
                raise ValueError(f"Cannot detect file type: {input_data}")
            print(f"Automatically detected file type: {file_type}")
        
        loaders = {
            'csv': pd.read_csv,
            'excel': pd.read_excel,
            'parquet': pd.read_parquet,
            'json': pd.read_json,
            'pickle': pd.read_pickle
        }
        
        if file_type.lower() not in loaders:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        try:
            data = loaders[file_type.lower()](input_data)
            if data.empty:
                raise ValueError(f"Loaded data is empty: {input_data}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    # Handle column selection
    if target_column is None:
        raise ValueError("Target column name must be specified")
    
    if columns is not None:
        if isinstance(columns, str):
            if ':' in columns:
                start, end = columns.split(':')
                start = int(start) if start else 0
                end = int(end) if end else None
                # Get all column names
                all_cols = data.columns.tolist()
                # Target column index
                target_idx = all_cols.index(target_column)
                # Feature columns (excluding target column)
                X_cols = all_cols[start:end]
                if target_column in X_cols:
                    X_cols.remove(target_column)
                X = data[X_cols]
            else:
                columns_list = [col.strip() for col in columns.split(',')]
                X = data[columns_list]
        elif isinstance(columns, list):
            X = data[columns]
        else:
            raise ValueError("columns parameter must be a list of column names or a column range string")
    else:
        # Use all columns except target column as features
        X = data.drop(columns=[target_column])
    
    y = data[target_column]
    
    return X, y

@register_selector('univariate_logistic')
def univariate_logistic_selector(X: pd.DataFrame, 
                               y: pd.Series,
                               outdir: str = None,
                               alpha: float = 0.05) -> Dict:
    """
    Perform univariate logistic regression feature selection
    
    Args:
        X: pd.DataFrame, Feature matrix
        y: pd.Series, Target variable
        alpha: float, Significance level for feature selection
        
    Returns:
        Dict: Dictionary containing selection results including:
            - selected_features: List of selected feature names
            - results: DataFrame with detailed statistics for each feature
    """
    results = []
    header = X.columns
    
    # Perform univariate logistic regression for each feature
    for i in tqdm(range(X.shape[1])):
        x_ = X.iloc[:, i]
        model = smf.logit(formula="event ~ x", data=pd.DataFrame({'event': y, 'x': x_}))
        result = model.fit(verbose=0)
        
        # Extract statistics
        p = result.pvalues[1]
        OR = np.exp(result.params[1])
        low, high = np.exp(result.conf_int().iloc[1, :])
        
        # Store results
        results.append(pd.DataFrame({
            'p_value': p,
            'odds_ratio': OR,
            'ci_lower': low,
            'ci_upper': high
        }, index=[header[i]]))
    
    # Combine all results
    results_df = pd.concat(results, axis=0)
    
    # Select features based on p-value threshold
    selected_features = np.array(header)[results_df['p_value'] < alpha].tolist()
    results_dict = {
        'selected_features': selected_features,
        'results': results_df.to_dict(),
        'alpha': alpha
    } 

    # save results_df to csv
    if outdir is not None:
        results_df.to_csv(os.path.join(outdir, 'univariate_logistic_results.csv'), index=True)

    return selected_features, results_dict

