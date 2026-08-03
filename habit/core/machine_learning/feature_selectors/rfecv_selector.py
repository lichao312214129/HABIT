# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
RFECV (Recursive Feature Elimination with Cross-Validation) Feature Selector

Uses scikit-learn's RFECV to select features based on model performance with cross-validation.
Supports both classification and regression tasks with various estimators.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb
import matplotlib.pyplot as plt

from habit.utils.font_config import show_or_close_figure
from habit.utils.log_utils import get_module_logger
logger = get_module_logger(__name__)

from .selector_registry import SelectorRegistry

# Dictionary mapping estimator names to their classes
ESTIMATOR_MAP = {
    # Classification estimators
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier': RandomForestClassifier,
    'SVC': SVC,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'XGBClassifier': xgb.XGBClassifier,
    
    # Regression estimators
    'LinearRegression': LinearRegression,
    'RandomForestRegressor': RandomForestRegressor,
    'SVR': SVR,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'XGBRegressor': xgb.XGBRegressor,
}

@SelectorRegistry.register('rfecv')
def rfecv_selector(data: pd.DataFrame,
                  target: Union[str, pd.Series],
                  estimator: str = 'RandomForestClassifier',
                  step: int = 1,
                  cv: int = 5,
                  scoring: str = 'roc_auc',
                  min_features_to_select: int = 1,
                  n_jobs: int = -1,
                  random_state: Optional[int] = None,
                  visualize: bool = False,
                  outdir: Optional[str] = None,
                  selected_features: Optional[List[str]] = None,
                  **estimator_params) -> List[str]:
    """
    Select features using Recursive Feature Elimination with Cross-Validation (RFECV)
    
    Args:
        data: Feature data as pandas DataFrame
        target: Target variable, can be either a column name (str) or a pandas Series
        estimator: Name of the estimator to use (must be one of the supported estimators)
        step: Number of features to remove at each iteration
        cv: Number of cross-validation folds
        scoring: Scoring metric to use
        min_features_to_select: Minimum number of features to select
        n_jobs: Number of jobs to run in parallel
        random_state: Random state for reproducibility
        visualize: Whether to generate visualization plots
        outdir: Output directory for visualization plots
        selected_features: List of already selected features, if None use all columns of data
        **estimator_params: Additional parameters to pass to the estimator
        
    Returns:
        List[str]: List of selected features
        
    Raises:
        ValueError: If estimator name is not supported
        TypeError: If input data is not a pandas DataFrame
        ValueError: If target column is not found in data (when target is str)
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")
    
    if estimator not in ESTIMATOR_MAP:
        raise ValueError(f"Unsupported estimator: {estimator}. Must be one of {list(ESTIMATOR_MAP.keys())}")
    
    # Prepare data
    if selected_features is None:
        if isinstance(target, str):
            selected_features = [col for col in data.columns if col != target]
        else:
            selected_features = data.columns.tolist()
    
    X = data[selected_features]
    y = data[target] if isinstance(target, str) else target
    
    # Initialize estimator
    estimator_class = ESTIMATOR_MAP[estimator]
    estimator_instance = estimator_class(random_state=random_state, **estimator_params)
    
    # Initialize RFECV
    rfecv = RFECV(
        estimator=estimator_instance,
        step=step,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=n_jobs
    )
    
    try:
        # Fit RFECV
        rfecv.fit(X, y)
        
        # Get selected features
        selected = [selected_features[i] for i in range(len(selected_features)) if rfecv.support_[i]]
        
        # Generate visualization if requested
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
            plt.xlabel('Number of Features Selected')
            plt.ylabel('Cross-validation Score')
            plt.title('RFECV Feature Selection')
            plt.grid(True)
            
            if outdir:
                import os
                os.makedirs(outdir, exist_ok=True)
                plt.savefig(os.path.join(outdir, 'rfecv_selection.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                show_or_close_figure()
        
        logger.info(f"RFECV selection: Selected {len(selected)} features from {len(selected_features)} features")
        return selected
        
    except Exception as e:
        logger.error(f"Error in RFECV selection: {e}")
        return [] 