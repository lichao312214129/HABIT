"""
Correlation Feature Selector

Removes highly correlated redundant features and retains the most informative feature subset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import List, Optional, Tuple, Dict, Union
from . import register_selector

@register_selector('correlation')
def correlation_selector(data: pd.DataFrame, 
                        threshold: float = 0.8, 
                        method: str = 'spearman',
                        visualize: bool = False,
                        outdir: Optional[str] = None,
                        selected_features: Optional[List[str]] = None) -> List[str]:
    """
    Remove highly correlated features
    
    Args:
        data: Feature data
        threshold: Correlation threshold, features with correlation above this value will be removed
        method: Correlation coefficient calculation method, options: 'pearson', 'spearman', 'kendall'
        visualize: Whether to generate visualization plots
        outdir: Output directory, required if visualize is True
        selected_features: List of already selected features, if None use all columns of data
        
    Returns:
        List[str]: List of selected features after correlation filtering
    """
    if selected_features is None:
        selected_features = data.columns.tolist()
    
    # Only use selected features
    data = data[selected_features]
    
    # Initialize feature set
    features = data.columns.tolist()
    
    # Calculate full correlation matrix (can be used for visualization)
    full_corr = data.corr(method=method)
    
    # Iteratively process features
    i = 0
    removed_features = []
    while i < len(features):
        current_feature = features[i]
        to_remove = []
        for j in range(i + 1, len(features)):
            corr = abs(full_corr.loc[current_feature, features[j]])
            if corr > threshold:
                to_remove.append(features[j])
        
        # Add to removed features list
        removed_features.extend(to_remove)
        
        # Remove from features list
        features = [f for f in features if f not in to_remove]
        
        # Move to next feature
        i += 1
    
    # Output removed features
    print(f"Correlation selection: Selected {len(features)} features from {len(selected_features)} features (removed {len(removed_features)} features)")
    
    # Visualize correlation matrix
    if visualize:
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            
            # Create figure
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            
            # Plot correlation heatmap before feature removal
            sns.heatmap(full_corr, annot=False, cmap="coolwarm", ax=ax[0])
            ax[0].set_title("Correlation Matrix (Before Feature Removal)")
            
            # Plot correlation heatmap after feature removal
            sns.heatmap(full_corr.loc[features, features], annot=False, cmap="coolwarm", ax=ax[1])
            ax[1].set_title("Correlation Matrix (After Feature Removal)")
            
            # Set rotation
            for axis in ax:
                axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
                axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save correlation heatmap
            plt.savefig(os.path.join(outdir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
            # plt.close()
            print(f"Correlation heatmap saved to {os.path.join(outdir, 'correlation_analysis.png')}")
            
            # Save removed features list
            results_dict = {
                'removed_features': removed_features,
                'selected_features': features,
                'threshold': threshold,
                'method': method
            }
            
            with open(os.path.join(outdir, 'correlation_results.json'), 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=4)
        else:
            print("Warning: outdir not specified, cannot save visualization results")
    
    return features 