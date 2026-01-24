"""
Python Stepwise Regression Feature Selector

Uses forward, backward, or stepwise (bidirectional) logistic regression for feature selection,
implemented purely in Python without R dependencies.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import warnings

from habit.utils.progress_utils import CustomTqdm
from .selector_registry import register_selector

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

def calculate_odds_ratio_and_ci(model: "statsmodels.discrete.discrete_model.LogitResults") -> pd.DataFrame:
    """
    Calculate odds ratio and 95% confidence interval for logistic regression model
    
    Args:
        model: Fitted statsmodels logistic regression model
        
    Returns:
        pd.DataFrame: DataFrame with odds ratio and confidence intervals
    """
    params = model.params
    conf_int = model.conf_int()
    
    # Calculate odds ratio and confidence intervals
    odds_ratio = np.exp(params)
    conf_int_exponentiated = np.exp(conf_int)
    
    # Create results dataframe
    results = pd.DataFrame({
        'OR': odds_ratio,
        '2.5% CI': conf_int_exponentiated.iloc[:, 0],
        '97.5% CI': conf_int_exponentiated.iloc[:, 1],
        'p_value': model.pvalues
    })
    
    return results

def forward_selection(X: pd.DataFrame, 
                     y: pd.Series, 
                     threshold_in: float = 0.05,
                     verbose: bool = False,
                     criterion: str = 'aic') -> List[str]:
    """
    Forward stepwise feature selection
    
    Args:
        X: pd.DataFrame, Feature matrix
        y: pd.Series, Target variable
        threshold_in: float, p-value threshold for inclusion (only used when criterion='pvalue')
        verbose: bool, Whether to print progress
        criterion: str, Criterion for selection ('aic', 'bic', or 'pvalue')
        
    Returns:
        List[str]: List of selected feature names
    """
    initial_features = []
    remaining_features = list(X.columns)
    
    best_criterion = np.inf if criterion in ['aic', 'bic'] else 0
    while remaining_features:
        best_new_criterion = np.inf if criterion in ['aic', 'bic'] else 0
        best_feature = None
        
        progress_bar = None
        if verbose:
            progress_bar = CustomTqdm(total=len(remaining_features), desc="Forward selection")
        
        for feature in remaining_features:
            model_features = initial_features + [feature]
            X_subset = X[model_features]
            
            # Add constant (intercept)
            X_with_const = sm.add_constant(X_subset)
            
            # Handle perfect separation
            try:
                model = Logit(y, X_with_const).fit(disp=0)
                
                if criterion == 'aic':
                    current_criterion = model.aic
                    # For AIC, lower is better
                    if current_criterion < best_new_criterion:
                        best_new_criterion = current_criterion
                        best_feature = feature
                elif criterion == 'bic':
                    current_criterion = model.bic
                    # For BIC, lower is better
                    if current_criterion < best_new_criterion:
                        best_new_criterion = current_criterion
                        best_feature = feature
                else:  # pvalue
                    # For p-value criterion, we add the feature with lowest p-value
                    pvalue = model.pvalues[feature]
                    # Only consider if p-value is below threshold
                    if pvalue < threshold_in and pvalue > best_new_criterion:
                        best_new_criterion = pvalue
                        best_feature = feature
                        
            except Exception as e:
                if verbose:
                    print(f"Error fitting model with feature {feature}: {str(e)}")
            finally:
                if progress_bar:
                    progress_bar.update(1)
        
        # If using AIC/BIC, we add features that improve criterion
        if criterion in ['aic', 'bic']:
            if best_feature is not None and best_new_criterion < best_criterion:
                best_criterion = best_new_criterion
                initial_features.append(best_feature)
                remaining_features.remove(best_feature)
                
                if verbose:
                    print(f"Added {best_feature} ({criterion.upper()}: {best_criterion})")
            else:
                # No improvement, stop
                break
        # If using p-value, we add the most significant feature below threshold
        else:  # pvalue
            if best_feature is not None:
                initial_features.append(best_feature)
                remaining_features.remove(best_feature)
                
                if verbose:
                    print(f"Added {best_feature} (p-value: {best_new_criterion})")
            else:
                # No feature with p-value below threshold, stop
                break
            
    return initial_features

def backward_elimination(X: pd.DataFrame, 
                        y: pd.Series, 
                        threshold_out: float = 0.05,
                        verbose: bool = False,
                        criterion: str = 'aic') -> List[str]:
    """
    Backward stepwise feature elimination
    
    Args:
        X: pd.DataFrame, Feature matrix
        y: pd.Series, Target variable
        threshold_out: float, p-value threshold for removal (only used when criterion='pvalue')
        verbose: bool, Whether to print progress
        criterion: str, Criterion for selection ('aic', 'bic', or 'pvalue')
        
    Returns:
        List[str]: List of selected feature names
    """
    initial_features = list(X.columns)
    
    # Add constant (intercept)
    X_with_const = sm.add_constant(X)
    
    # Try to fit full model first
    try:
        full_model = Logit(y, X_with_const).fit(disp=0)
        if criterion == 'aic':
            best_criterion = full_model.aic
        elif criterion == 'bic':
            best_criterion = full_model.bic
        else:  # pvalue
            best_criterion = 0  # Not used for pvalue criterion
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix error in full model. Using forward selection instead.")
        return forward_selection(X, y, threshold_out, verbose, criterion)
    except Exception as e:
        warnings.warn(f"Error fitting full model: {str(e)}. Using forward selection instead.")
        return forward_selection(X, y, threshold_out, verbose, criterion)
    
    while len(initial_features) > 0:
        if criterion in ['aic', 'bic']:
            best_new_criterion = np.inf
            worst_feature = None
            
            # Try removing each feature
            desc = f"Backward elimination ({criterion.upper()})"
            progress_bar = None
            if verbose:
                progress_bar = CustomTqdm(total=len(initial_features), desc=desc)
            
            for feature in initial_features:
                model_features = [f for f in initial_features if f != feature]
                
                # Skip if empty
                if not model_features:
                    if progress_bar:
                        progress_bar.update(1)
                    continue
                    
                X_subset = X[model_features]
                X_with_const = sm.add_constant(X_subset)
                
                try:
                    model = Logit(y, X_with_const).fit(disp=0)
                    if criterion == 'aic':
                        current_criterion = model.aic
                    else:  # bic
                        current_criterion = model.bic
                    
                    # If removing improves criterion
                    if current_criterion < best_new_criterion:
                        best_new_criterion = current_criterion
                        worst_feature = feature
                except:
                    pass
                finally:
                    if progress_bar:
                        progress_bar.update(1)
            
            # If removing a feature improves criterion
            if worst_feature is not None and best_new_criterion < best_criterion:
                best_criterion = best_new_criterion
                initial_features.remove(worst_feature)
                if verbose:
                    print(f"Removed {worst_feature} ({criterion.upper()}: {best_criterion})")
            else:
                # No improvement, stop
                break
                
        else:  # pvalue criterion
            # Get full model results
            model = Logit(y, sm.add_constant(X[initial_features])).fit(disp=0)
            
            # Find feature with highest p-value
            best_pvalue = 0
            worst_feature = None
            
            progress_bar = None
            if verbose:
                progress_bar = CustomTqdm(total=len(initial_features), desc="Backward elimination (p-value)")
            
            for feature in initial_features:
                pvalue = model.pvalues.get(feature, 0)
                if pvalue > best_pvalue:
                    best_pvalue = pvalue
                    worst_feature = feature
                if progress_bar:
                    progress_bar.update(1)
            
            # If the worst feature's p-value is above the threshold, remove it
            if worst_feature is not None and best_pvalue > threshold_out:
                initial_features.remove(worst_feature)
                if verbose:
                    print(f"Removed {worst_feature} (p-value: {best_pvalue})")
            else:
                # No feature with p-value above threshold, stop
                break
            
    return initial_features

def stepwise_selection(X: pd.DataFrame, 
                      y: pd.Series, 
                      threshold_in: float = 0.05,
                      threshold_out: float = 0.05,
                      verbose: bool = False,
                      criterion: str = 'aic') -> List[str]:
    """
    Bidirectional stepwise feature selection (both forward and backward)
    
    Args:
        X: pd.DataFrame, Feature matrix
        y: pd.Series, Target variable
        threshold_in: float, p-value threshold for inclusion (only used when criterion='pvalue')
        threshold_out: float, p-value threshold for removal (only used when criterion='pvalue')
        verbose: bool, Whether to print progress
        criterion: str, Criterion for selection ('aic', 'bic', or 'pvalue')
        
    Returns:
        List[str]: List of selected feature names
    """
    initial_features = []
    remaining_features = list(X.columns)
    
    # Initialize
    best_criterion = np.inf if criterion in ['aic', 'bic'] else 0  # Not used for p-value
    
    progress_bar = None
    if verbose:
        desc = f"Feature selection" if criterion == 'pvalue' else f"Feature selection ({criterion.upper()})"
        progress_bar = CustomTqdm(total=len(remaining_features), desc=desc)
    
    while True:
        changed = False
        
        # Forward step
        if criterion in ['aic', 'bic']:
            best_new_criterion = np.inf
            best_feature_to_add = None
            
            for feature in remaining_features:
                model_features = initial_features + [feature]
                X_subset = X[model_features]
                
                # Add constant (intercept)
                X_with_const = sm.add_constant(X_subset)
                
                try:
                    model = Logit(y, X_with_const).fit(disp=0)
                    if criterion == 'aic':
                        current_criterion = model.aic
                    else:  # bic
                        current_criterion = model.bic
                    
                    if current_criterion < best_new_criterion:
                        best_new_criterion = current_criterion
                        best_feature_to_add = feature
                except:
                    pass
                finally:
                    if progress_bar:
                        progress_bar.update(1)
            
            # Add the best feature if it improves criterion
            if best_feature_to_add is not None and best_new_criterion < best_criterion:
                best_criterion = best_new_criterion
                initial_features.append(best_feature_to_add)
                remaining_features.remove(best_feature_to_add)
                changed = True
                
                if verbose:
                    print(f"Added {best_feature_to_add} ({criterion.upper()}: {best_criterion})")
        
        else:  # pvalue criterion
            best_pvalue = threshold_in  # Initialize with threshold
            best_feature_to_add = None
            
            # Get model with current features 
            if initial_features:
                X_current = X[initial_features]
                X_current = sm.add_constant(X_current)
                current_model = Logit(y, X_current).fit(disp=0)
            
            for feature in remaining_features:
                model_features = initial_features + [feature]
                X_subset = X[model_features]
                
                # Add constant (intercept)
                X_with_const = sm.add_constant(X_subset)
                
                try:
                    model = Logit(y, X_with_const).fit(disp=0)
                    pvalue = model.pvalues[feature]
                    
                    # If p-value is below threshold and better than current best
                    if pvalue < best_pvalue:
                        best_pvalue = pvalue
                        best_feature_to_add = feature
                except:
                    pass
                finally:
                    if progress_bar:
                        progress_bar.update(1)
            
            # Add the best feature if it's significant
            if best_feature_to_add is not None and best_pvalue < threshold_in:
                initial_features.append(best_feature_to_add)
                remaining_features.remove(best_feature_to_add)
                changed = True
                
                if verbose:
                    print(f"Added {best_feature_to_add} (p-value: {best_pvalue})")
        
        # Backward step (if we have features to remove)
        if len(initial_features) > 0:
            if criterion in ['aic', 'bic']:
                worst_criterion = np.inf
                worst_feature = None
                
                # Try without each feature
                for feature in initial_features:
                    model_features = [f for f in initial_features if f != feature]
                    
                    # Skip if empty
                    if not model_features:
                        if progress_bar:
                            progress_bar.update(1)
                        continue
                        
                    X_subset = X[model_features]
                    X_with_const = sm.add_constant(X_subset)
                    
                    try:
                        model = Logit(y, X_with_const).fit(disp=0)
                        if criterion == 'aic':
                            current_criterion = model.aic
                        else:  # bic
                            current_criterion = model.bic
                        
                        # If removing improves criterion
                        if current_criterion < worst_criterion:
                            worst_criterion = current_criterion
                            worst_feature = feature
                    except:
                        pass
                    finally:
                        if progress_bar:
                            progress_bar.update(1)
                
                # If removing a feature improves criterion
                if worst_feature is not None and worst_criterion < best_criterion:
                    best_criterion = worst_criterion
                    initial_features.remove(worst_feature)
                    remaining_features.append(worst_feature)
                    changed = True
                    
                    if verbose:
                        print(f"Removed {worst_feature} ({criterion.upper()}: {best_criterion})")
            
            else:  # pvalue criterion
                # Get full model results
                X_current = X[initial_features]
                X_current = sm.add_constant(X_current)
                try:
                    model = Logit(y, X_current).fit(disp=0)
                    
                    # Find feature with highest p-value
                    worst_pvalue = 0
                    worst_feature = None
                    
                    for feature in initial_features:
                        pvalue = model.pvalues.get(feature, 0)
                        if pvalue > worst_pvalue:
                            worst_pvalue = pvalue
                            worst_feature = feature
                        if progress_bar:
                            progress_bar.update(1)
                    
                    # If the worst feature's p-value is above the threshold, remove it
                    if worst_feature is not None and worst_pvalue > threshold_out:
                        initial_features.remove(worst_feature)
                        remaining_features.append(worst_feature)
                        changed = True
                        
                        if verbose:
                            print(f"Removed {worst_feature} (p-value: {worst_pvalue})")
                except:
                    # If model fitting fails, don't remove any features
                    pass
        
        # If no changes were made in this iteration, stop
        if not changed:
            break
            
    return initial_features

@register_selector('stepwise')
def python_stepwise_selector(X: pd.DataFrame, 
                           y: pd.Series,
                           direction: str = 'backward',
                           threshold_in: float = 0.05,
                           threshold_out: float = 0.05,
                           criterion: str = 'aic',
                           verbose: bool = False,
                           outdir: str = None) -> List[str]:
    """
    Run stepwise regression feature selection using pure Python
    
    Args:
        X: pd.DataFrame, Feature matrix
        y: pd.Series, Target variable
        direction: str, Direction of stepwise selection ('forward', 'backward', or 'both')
        threshold_in: float, p-value threshold for inclusion (only used when criterion='pvalue')
        threshold_out: float, p-value threshold for removal (only used when criterion='pvalue')
        criterion: str, Criterion for feature selection ('aic', 'bic', or 'pvalue')
        verbose: bool, Whether to print progress
        outdir: str, Output directory for results
        
    Returns:
        List[str]: List of selected feature names
    """
    # Check direction
    if direction not in ['forward', 'backward', 'both']:
        raise ValueError("direction must be one of: 'forward', 'backward', 'both'")
    
    # Check criterion
    if criterion not in ['aic', 'bic', 'pvalue']:
        raise ValueError("criterion must be one of: 'aic', 'bic', 'pvalue'")
    
    # Create output directory if it doesn't exist
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
    
    # Run appropriate selection method
    if direction == 'forward':
        selected_features = forward_selection(X, y, threshold_in, verbose, criterion)
    elif direction == 'backward':
        selected_features = backward_elimination(X, y, threshold_out, verbose, criterion)
    else:  # 'both'
        selected_features = stepwise_selection(X, y, threshold_in, threshold_out, verbose, criterion)
    
    # If we have features, calculate and save the final model results
    if selected_features and outdir:
        try:
            X_selected = X[selected_features]
            X_with_const = sm.add_constant(X_selected)
            final_model = Logit(y, X_with_const).fit(disp=0)
            
            # Calculate odds ratio and confidence intervals
            results_df = calculate_odds_ratio_and_ci(final_model)
            
            # Save results
            results_df.to_csv(os.path.join(outdir, 'stepwise_results.csv'))
            
            # Plot coefficients with error bars if there are features
            if len(selected_features) > 0:
                plt.figure(figsize=(10, 8))
                results_df_plot = results_df.drop('Intercept', errors='ignore')
                
                # Sort by odds ratio for better visualization
                results_df_plot = results_df_plot.sort_values(by='OR')
                
                features = results_df_plot.index
                ors = results_df_plot['OR']
                lower_ci = results_df_plot['2.5% CI']
                upper_ci = results_df_plot['97.5% CI']
                
                plt.errorbar(
                    ors, range(len(features)),
                    xerr=[ors - lower_ci, upper_ci - ors],
                    fmt='o', capsize=5, elinewidth=1, markeredgewidth=1
                )
                # line width
                plt.gca().lines[0].set_linewidth(1.5)
                
                plt.axvline(x=1.0, color='r', linestyle='--', label='OR = 1.0')
                plt.yticks(range(len(features)), features)
                plt.xscale('log')
                plt.xlabel('Odds Ratio (log scale)')
                plt.ylabel('Features')
                plt.title('Odds Ratios with 95% Confidence Intervals')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plt.savefig(os.path.join(outdir, f'stepwise_{criterion}_odds_ratios.png'), dpi=600)
                plt.close()
        except Exception as e:
            warnings.warn(f"Error generating final results: {str(e)}")
    
    return selected_features 