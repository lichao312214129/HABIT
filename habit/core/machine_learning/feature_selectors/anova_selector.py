"""
ANOVA Feature Selector

Implementation of ANOVA F-value based feature selection for classification problems.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict, Optional, Union
import os
import json
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import seaborn as sns

from .selector_registry import register_selector

@register_selector('anova')
def anova_selector(
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        p_threshold: float = 0.05,
        n_features_to_select: Optional[int] = None,
        plot_importance: bool = True,
        outdir: Optional[str] = None,
        **kwargs
    ) -> List[str]:
    """
    Apply ANOVA F-value based feature selection using p-value threshold
    
    Args:
        X: Feature data
        y: Target variable
        selected_features: List of features to select from
        p_threshold: P-value threshold for feature selection (default: 0.05)
        n_features_to_select: Optional number of features to select (if specified, overrides p_threshold)
        plot_importance: Whether to plot feature importance
        outdir: Output directory for results
        **kwargs: Additional arguments
        
    Returns:
        List[str]: Selected feature names
    """
    # Get feature subset
    X_subset = X[selected_features]
    
    # Apply ANOVA F-test
    F_values, p_values = f_classif(X_subset, y)
    
    # Create feature ranking
    feature_ranking = pd.DataFrame({
        'feature': selected_features,
        'score': F_values,
        'pvalue': p_values
    })
    
    # Sort by score
    feature_ranking = feature_ranking.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Select features based on p-value threshold or n_features_to_select
    if n_features_to_select is not None:
        # If n_features_to_select is specified, use it (limited by available features)
        n_features_to_select = min(n_features_to_select, len(selected_features))
        selected_mask = feature_ranking.index < n_features_to_select
        selection_method = f"top {n_features_to_select} features"
    else:
        # Otherwise use p-value threshold
        selected_mask = feature_ranking['pvalue'] < p_threshold
        selection_method = f"p-value < {p_threshold}"
    
    feature_ranking['selected'] = selected_mask
    
    # Get selected feature names
    selected_features_result = feature_ranking.loc[selected_mask, 'feature'].tolist()
    
    print(f"Selected {len(selected_features_result)} features from {len(selected_features)} features using {selection_method}")
    
    # Save results if output directory specified
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        
        # Save selected features
        result_file = os.path.join(outdir, 'anova_selected_features.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'selected_features': selected_features_result,
                'n_features_selected': len(selected_features_result),
                'original_feature_count': len(selected_features),
                'selection_method': selection_method
            }, f, ensure_ascii=False, indent=4)
            
        # Save feature ranking
        ranking_file = os.path.join(outdir, 'anova_feature_ranking.csv')
        feature_ranking.to_csv(ranking_file, index=False)
        
        # Plot feature importance
        if plot_importance:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='score', y='feature', data=feature_ranking.head(min(20, len(selected_features))))
            plt.title('ANOVA F-Score Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'anova_feature_importance.pdf'))
            plt.close()
            
            # Plot feature importance with p-values
            plt.figure(figsize=(12, 8))
            top_features = feature_ranking.head(min(20, len(selected_features)))
            ax = sns.barplot(x='score', y='feature', data=top_features)
            
            # Add p-value annotations
            for i, feature in enumerate(top_features['feature']):
                p_value = top_features.loc[top_features['feature'] == feature, 'pvalue'].values[0]
                stars = ''
                if p_value < 0.001:
                    stars = '***'
                elif p_value < 0.01:
                    stars = '**'
                elif p_value < 0.05:
                    stars = '*'
                ax.text(top_features.loc[top_features['feature'] == feature, 'score'].values[0] + 0.5, 
                        i, stars, ha='left', va='center')
            
            plt.title('ANOVA F-Score Feature Importance with P-values (* p<0.05, ** p<0.01, *** p<0.001)')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'anova_feature_importance_with_pvalues.pdf'))
            plt.close()
        
    return selected_features_result 