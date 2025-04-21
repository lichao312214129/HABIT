"""
Stepwise Regression Feature Selector

Uses forward, backward, or stepwise (bidirectional) logistic regression for feature selection
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union
from tqdm import tqdm
from pathlib import Path

from . import register_selector

def _set_r_environment(Rhome: Optional[str]) -> None:
    """
    Set up R environment
    
    Args:
        Rhome: R installation path
        
    Raises:
        EnvironmentError: If R installation path is not found
    """
    if Rhome:
        os.environ['R_HOME'] = Rhome
    else:
        raise EnvironmentError("R installation path not found, please specify Rhome in config file")
    
    # import rpy2 related packages after setting R_HOME
    global ro, r, pandas2ri, importr
    import rpy2.robjects as ro
    from rpy2.robjects import r, pandas2ri
    from rpy2.robjects.packages import importr
        
def detect_file_type(input_path: str) -> Optional[str]:
    """
    Automatically detect file type
    
    Args:
        input_path: Input file path
        
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
        input_data: Input data path or DataFrame object
        target_column: Target variable column name
        file_type: File type
        columns: Feature column selection
        
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

@register_selector('stepwise')
def stepwise_selector(X: pd.DataFrame, 
                     y: pd.Series,
                     direction: str = 'backward',
                     outdir: str = None,
                     Rhome: str = None) -> List[str]:
    """
    Run stepwise regression feature selection using R's stepAIC
    
    Args:
        X: pd.DataFrame, Feature matrix
        y: pd.Series, Target variable
        direction: str, Direction of stepwise selection ('forward', 'backward', or 'both')
        Rhome: str, R installation path
        
    Returns:
        List[str]: List of selected feature names
    """
    _set_r_environment(Rhome)

    # Initialize R environment
    pandas2ri.activate()
    MASS = importr('MASS')
    stargazer = importr('stargazer')
    
    # Prepare data for R
    X['event'] = y
    X_r = pandas2ri.py2rpy(X)
    r.assign('x_train', X_r)
    
    # Run logistic regression and stepwise AIC in R
    ro.r('''
        model <- glm(event ~ ., data = x_train, family = binomial)
        step_model <- stepAIC(model, trace = FALSE, direction = "{}")
    '''.format(direction))
    
    # Calculate OR and 95% CI
    ro.r('''
        exp_coef <- exp(coef(step_model))
        conf_int <- exp(confint(step_model))
        p_values <- summary(step_model)$coefficients[, 4]
        results <- data.frame(OR = exp_coef, `2.5% CI` = conf_int[, 1], `97.5% CI` = conf_int[, 2], p_value = p_values)
    ''')
    
    # Get selected features
    selected_features = ro.r('names(step_model$coefficients)')
    # Remove intercept term
    selected_features = selected_features[1:]
    # Clean feature names, remove backticks
    selected_features = [feature.strip('`') for feature in selected_features]
    
    # Extract results from R and convert to pandas DataFrame
    try:
        results_r = ro.r('results')
        header_name = ro.r('names(results)')
        feature_name = ro.r('rownames(results)')
        results_df = pd.DataFrame(results_r)
        results_df.columns = feature_name
        results_df.index = header_name
        results_df.to_csv(os.path.join(outdir, 'stepwise_results.csv'), index=True)
    except:
        results_r = ro.r('results')
        results_df = pd.DataFrame(results_r)
        results_df.to_csv(os.path.join(outdir, 'stepwise_results.csv'), index=True)

    return selected_features

