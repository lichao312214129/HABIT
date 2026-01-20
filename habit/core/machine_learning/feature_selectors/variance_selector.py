"""
Variance Feature Selector

Implementation of variance based feature selection for removing features with low variance.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict, Optional, Union
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

from .selector_registry import register_selector

@register_selector('variance', display_name='Variance Threshold', default_before_z_score=True)
def variance_selector(
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        threshold: float = 0.0,
        plot_variances: bool = True,
        top_k: Optional[int] = None,
        top_percent: Optional[float] = None,
        outdir: Optional[str] = None,
        **kwargs
    ) -> List[str]:
    """
    Apply variance threshold based feature selection
    
    Args:
        X: Feature data
        y: Target variable (not used for variance selection)
        selected_features: List of features to select from
        threshold: Variance threshold for feature selection (default: 0.0)
        plot_variances: Whether to plot feature variances
        top_k: Select top k features with highest variance (overrides threshold if provided)
        top_percent: Select top percentage of features with highest variance (0-100, overrides threshold if provided)
        outdir: Output directory for results
        **kwargs: Additional arguments
        
    Returns:
        List[str]: Selected feature names
    """
    # Get feature subset
    X_subset = X[selected_features]
    
    # Calculate variances
    variances = X_subset.var().sort_values(ascending=False)
    
    # Create feature ranking
    feature_ranking = pd.DataFrame({
        'feature': variances.index,
        'variance': variances.values
    })
    
    selected_features_result = []
    selection_method = "threshold"
    selection_value = threshold
    
    # Apply feature selection based on the provided parameters
    if top_k is not None and top_k > 0:
        # Select top k features with highest variance
        top_k = min(top_k, len(selected_features))  # Ensure we don't try to select more features than available
        selected_features_result = list(variances.index[:top_k])
        selection_method = "top_k"
        selection_value = top_k
        print(f"Selected top {top_k} features from {len(selected_features)} features based on variance")
    
    elif top_percent is not None and 0 < top_percent <= 100:
        # Select top percentage of features with highest variance
        k = int(np.ceil(len(selected_features) * top_percent / 100))
        selected_features_result = list(variances.index[:k])
        selection_method = "top_percent"
        selection_value = top_percent
        print(f"Selected top {top_percent}% ({k} out of {len(selected_features)}) features based on variance")
    
    else:
        # Apply traditional variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_subset)
        
        # Get selected feature mask
        selected_mask = selector.get_support()
        
        # Get selected feature names
        selected_features_result = [selected_features[i] for i in range(len(selected_features)) if selected_mask[i]]
        print(f"Selected {len(selected_features_result)} features from {len(selected_features)} features "
              f"using variance threshold > {threshold}")
    
    # Add selection status to the ranking
    feature_ranking['selected'] = feature_ranking['feature'].isin(selected_features_result)
    
    # Save results if output directory specified
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        
        # Save selected features
        result_file = os.path.join(outdir, 'variance_selected_features.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'selected_features': selected_features_result,
                'n_features_selected': len(selected_features_result),
                'original_feature_count': len(selected_features),
                'selection_method': selection_method,
                'selection_value': float(selection_value) if isinstance(selection_value, (int, float)) else selection_value
            }, f, ensure_ascii=False, indent=4)
            
        # Save feature ranking
        ranking_file = os.path.join(outdir, 'variance_feature_ranking.csv')
        feature_ranking.to_csv(ranking_file, index=False)
        
        # Plot feature variances
        if plot_variances:
            plt.figure(figsize=(12, 8))
            
            # Plot top 20 features (or all if less than 20)
            top_features = feature_ranking.head(min(20, len(selected_features)))
            
            # Create bar plot with selected features highlighted
            ax = sns.barplot(x='variance', y='feature', data=top_features)
            
            # Highlight selected features
            for i, feature in enumerate(top_features['feature']):
                if feature in selected_features_result:
                    ax.patches[i].set_facecolor('green')
                else:
                    ax.patches[i].set_facecolor('lightgray')
            
            # Add selection method to title
            if selection_method == "threshold":
                plt.title(f'Feature Variances (threshold: {threshold})')
                plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
            elif selection_method == "top_k":
                plt.title(f'Feature Variances (top {top_k} selected)')
            else:  # top_percent
                plt.title(f'Feature Variances (top {top_percent}% selected)')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'variance_feature_importance.pdf'))
            plt.close()
        
    return selected_features_result 