"""
RFECV Feature Selector

Implementation of Recursive Feature Elimination with Cross-Validation (RFECV) for feature selection
"""

import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional, Dict, Any
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold

from .selector_registry import register_selector
from habit.utils.progress_utils import CustomTqdm

@register_selector("rfecv")
def rfecv_selector(X: pd.DataFrame, 
                  y: pd.Series, 
                  selected_features: Optional[List[str]] = None,
                  estimator: str = "random_forest",
                  task_type: str = "classification",
                  cv: int = 5,
                  step: Union[int, float] = 1,
                  min_features_to_select: int = 1,
                  scoring: Optional[str] = None,
                  n_jobs: int = -1,
                  verbose: int = 0,
                  **estimator_params) -> List[str]:
    """
    Recursive Feature Elimination with Cross-Validation (RFECV) for feature selection.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        selected_features (List[str], optional): List of features to consider. If None, all features are used.
        estimator (str, optional): Estimator to use. Options: "random_forest", "linear_regression", "logistic_regression". Default: "random_forest"
        task_type (str, optional): Task type, either "classification" or "regression". Default: "classification"
        cv (int, optional): Number of cross-validation folds. Default: 5
        step (Union[int, float], optional): If int, number of features to remove at each iteration. 
                                           If float, percentage of features to remove at each iteration. Default: 1
        min_features_to_select (int, optional): Minimum number of features to be selected. Default: 1
        scoring (str, optional): Scoring method for cross-validation. Default: None (uses estimator's default scorer)
        n_jobs (int, optional): Number of parallel jobs. Default: -1 (all processors)
        verbose (int, optional): Verbosity level. Default: 0
        **estimator_params: Additional parameters to pass to the estimator
        
    Returns:
        List[str]: List of selected features
    """
    if selected_features is None:
        selected_features = X.columns.tolist()
    
    X_work = X[selected_features].copy()
    
    # Set up cross-validation strategy
    if task_type == "classification":
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create estimator
    if estimator == "random_forest":
        if task_type == "classification":
            base_estimator = RandomForestClassifier(random_state=42, **estimator_params)
        else:
            base_estimator = RandomForestRegressor(random_state=42, **estimator_params)
    elif estimator == "linear_regression":
        if task_type == "classification":
            base_estimator = LogisticRegression(random_state=42, **estimator_params)
        else:
            base_estimator = LinearRegression(**estimator_params)
    else:
        raise ValueError(f"Unsupported estimator: {estimator}")
    
    # Create progress bar if verbose
    progress_bar = None
    if verbose > 0:
        progress_bar = CustomTqdm(desc="RFECV Feature Selection")
    
    # Create and fit selector
    selector = RFECV(
        estimator=base_estimator,
        step=step,
        min_features_to_select=min_features_to_select,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    selector.fit(X_work, y)
    
    # Get selected features
    selected_mask = selector.support_
    selected_feature_names = X_work.columns[selected_mask].tolist()
    
    if verbose > 0:
        # Print ranking and optimal number of features
        print(f"Optimal number of features: {selector.n_features_}")
        
        # Get feature importance ranking
        ranking_with_names = list(zip(X_work.columns, selector.ranking_))
        ranking_with_names.sort(key=lambda x: x[1])
        
        print("Feature ranking (lower is better):")
        for feature, rank in ranking_with_names:
            selection_status = "Selected" if feature in selected_feature_names else "Rejected"
            print(f"  {feature}: {rank} ({selection_status})")
    
    return selected_feature_names 