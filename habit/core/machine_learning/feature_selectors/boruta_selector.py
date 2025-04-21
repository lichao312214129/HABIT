"""
Boruta Feature Selector

Implementation of Boruta algorithm for feature selection.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict, Optional, Union
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    print("Warning: Boruta package not available. Boruta selection will not be available.")
    print("You can install Boruta with: pip install Boruta")

from . import register_selector

@register_selector('boruta')
def boruta_selector(
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        n_estimators: int = 100,
        max_iter: int = 100,
        perc: int = 100,
        alpha: float = 0.05,
        random_state: int = 42,
        plot_importance: bool = True,
        outdir: Optional[str] = None,
        **kwargs
    ) -> List[str]:
    """
    Apply Boruta algorithm for feature selection
    
    Args:
        X: Feature data
        y: Target variable
        selected_features: List of features to select from
        n_estimators: Number of trees in the random forest
        max_iter: Maximum number of iterations
        perc: Percentage of shadow features to keep
        alpha: Level of significance
        random_state: Random state for reproducibility
        plot_importance: Whether to plot feature importance
        outdir: Output directory for results
        **kwargs: Additional arguments
        
    Returns:
        List[str]: Selected feature names
        
    Raises:
        ImportError: If Boruta package is not available
    """
    if not BORUTA_AVAILABLE:
        raise ImportError("Boruta package is required for Boruta selection")
    
    # Get feature subset
    X_subset = X[selected_features]
    
    # Prepare base classifier
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    # Initialize Boruta
    boruta_selector = BorutaPy(
        rf, 
        n_estimators=n_estimators,
        max_iter=max_iter,
        perc=perc,
        alpha=alpha,
        random_state=random_state,
        verbose=2
    )
    
    # Fit Boruta
    boruta_selector.fit(X_subset.values, y.values)
    
    # Get selected feature mask and ranking
    selected_mask = boruta_selector.support_
    ranking = boruta_selector.ranking_
    
    # Create feature ranking dataframe
    feature_ranking = pd.DataFrame({
        'feature': selected_features,
        'ranking': ranking,
        'selected': selected_mask,
        'tentative': boruta_selector.support_weak_
    })
    
    # Sort by ranking
    feature_ranking = feature_ranking.sort_values('ranking').reset_index(drop=True)
    
    # Get selected feature names
    selected_features_result = list(X_subset.columns[selected_mask])
    
    # If we also want to include tentative features
    if kwargs.get('include_tentative', True):
        tentative_mask = boruta_selector.support_weak_
        tentative_features = list(X_subset.columns[tentative_mask & ~selected_mask])
        all_selected = selected_features_result + tentative_features
    else:
        all_selected = selected_features_result
        
    # Save results if output directory specified
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        
        # Save selected features
        result_file = os.path.join(outdir, 'boruta_selected_features.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'confirmed_features': selected_features_result,
                'tentative_features': list(X_subset.columns[boruta_selector.support_weak_ & ~selected_mask]),
                'rejected_features': list(X_subset.columns[~boruta_selector.support_ & ~boruta_selector.support_weak_]),
                'n_features_selected': len(selected_features_result),
                'n_tentative_features': sum(boruta_selector.support_weak_ & ~selected_mask),
                'n_rejected_features': sum(~boruta_selector.support_ & ~boruta_selector.support_weak_),
                'original_feature_count': len(selected_features)
            }, f, ensure_ascii=False, indent=4)
            
        # Save feature ranking
        ranking_file = os.path.join(outdir, 'boruta_feature_ranking.csv')
        feature_ranking.to_csv(ranking_file, index=False)
        
        # Plot feature importance
        if plot_importance and hasattr(boruta_selector, 'importance_history_'):
            # Plot feature importances
            plt.figure(figsize=(12, 10))
            x = range(len(selected_features))
            y = boruta_selector.importance_history_.mean(axis=0)
            y_shadow = boruta_selector.shadow_importance_history_.mean(axis=0).mean()
            
            colors = ['green' if selected else 'blue' if tentative else 'red' 
                     for selected, tentative in zip(selected_mask, boruta_selector.support_weak_)]
            
            # Plot feature importances
            plt.bar(x, y, color=colors)
            plt.axhline(y=y_shadow, color='r', linestyle='-', label='Shadow Max')
            
            plt.xticks(x, selected_features, rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Boruta Feature Importance')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Confirmed'),
                Patch(facecolor='blue', label='Tentative'),
                Patch(facecolor='red', label='Rejected')
            ]
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'boruta_feature_importance.pdf'))
            plt.close()
        
    return all_selected 