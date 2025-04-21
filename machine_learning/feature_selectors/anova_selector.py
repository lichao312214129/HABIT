"""
ANOVA Feature Selector

Implementation of ANOVA F-value based feature selection for classification problems.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict, Optional, Union
import os
import json
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

from . import register_selector

@register_selector('anova')
def anova_selector(
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        n_features_to_select: int = 20,
        plot_importance: bool = True,
        outdir: Optional[str] = None,
        **kwargs
    ) -> List[str]:
    """
    Apply ANOVA F-value based feature selection
    
    Args:
        X: Feature data
        y: Target variable
        selected_features: List of features to select from
        n_features_to_select: Number of features to select
        plot_importance: Whether to plot feature importance
        outdir: Output directory for results
        **kwargs: Additional arguments
        
    Returns:
        List[str]: Selected feature names
    """
    # Ensure n_features_to_select is valid
    n_features_to_select = min(n_features_to_select, len(selected_features))
    
    # Get feature subset
    X_subset = X[selected_features]
    
    # Apply ANOVA F-test
    selector = SelectKBest(f_classif, k=n_features_to_select)
    selector.fit(X_subset, y)
    
    # Get selected feature indices and scores
    selected_indices = selector.get_support(indices=True)
    scores = selector.scores_
    
    # Create feature ranking
    feature_ranking = pd.DataFrame({
        'feature': selected_features,
        'score': scores,
        'pvalue': selector.pvalues_,
        'selected': selector.get_support()
    })
    
    # Sort by score
    feature_ranking = feature_ranking.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Get selected feature names
    selected_features_result = [selected_features[i] for i in selected_indices]
    
    # Save results if output directory specified
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        
        # Save selected features
        result_file = os.path.join(outdir, 'anova_selected_features.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'selected_features': selected_features_result,
                'n_features_selected': len(selected_features_result),
                'original_feature_count': len(selected_features)
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