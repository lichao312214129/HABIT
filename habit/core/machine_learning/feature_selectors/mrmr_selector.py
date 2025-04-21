"""
mRMR Feature Selector

Implementation of Minimum Redundancy Maximum Relevance algorithm for feature selection.
REF: "Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy"
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict, Optional, Union
import os
import json
try:
    from skfeature.function.information_theoretical_based import MRMR
    SKFEATURE_AVAILABLE = True
except ImportError:
    SKFEATURE_AVAILABLE = False
    print("Warning: scikit-feature package not available. mRMR selection will not be available.")
    print("You can install scikit-feature from GitHub: https://github.com/jundongl/scikit-feature")

from . import register_selector

@register_selector('mrmr')
def mrmr_selector(
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        n_features_to_select: int = 20,
        outdir: Optional[str] = None,
        **kwargs
    ) -> List[str]:
    """
    Apply mRMR algorithm to select the most relevant features
    
    Args:
        X: Feature data
        y: Target variable
        selected_features: List of features to select from
        n_features_to_select: Number of features to select
        outdir: Output directory for results
        **kwargs: Additional arguments
        
    Returns:
        List[str]: Selected feature names
        
    Raises:
        ImportError: If scikit-feature package is not available
    """
    if not SKFEATURE_AVAILABLE:
        raise ImportError("scikit-feature package is required for mRMR selection")
    
    # Ensure n_features_to_select is valid
    n_features_to_select = min(n_features_to_select, len(selected_features))
    
    # Convert data to numpy arrays
    X_data = X[selected_features].values
    y_data = y.values
    
    # Apply the mRMR algorithm to select features
    selected_indices = MRMR.mrmr(X_data, y_data, n_selected_features=n_features_to_select)
    
    # Get the selected feature names
    selected_features_result = [selected_features[i] for i in selected_indices]

    print(f"mRMR selection: Selected {n_features_to_select} features from {len(selected_features)} features")
    
    # Save results if output directory specified
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        
        # Save selected features
        result_file = os.path.join(outdir, 'mrmr_selected_features.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'selected_features': selected_features_result,
                'selected_indices': selected_indices.tolist(),
                'n_features_selected': len(selected_features_result),
                'original_feature_count': len(selected_features)
            }, f, ensure_ascii=False, indent=4)
            
        # Save feature ranking
        ranking_data = pd.DataFrame({
            'feature': [selected_features[i] for i in selected_indices],
            'rank': range(1, len(selected_indices) + 1)
        })
        ranking_file = os.path.join(outdir, 'mrmr_feature_ranking.csv')
        ranking_data.to_csv(ranking_file, index=False)
        
    return selected_features_result[:n_features_to_select]