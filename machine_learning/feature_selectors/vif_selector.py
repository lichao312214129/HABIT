"""
VIF (Variance Inflation Factor) Feature Selector

Used to detect and remove multicollinear features
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Optional, Tuple, Dict, Union
from statsmodels.stats.outliers_influence import variance_inflation_factor
from . import register_selector

@register_selector('vif')
def vif_selector(X: pd.DataFrame, 
                max_vif: float = 10.0, 
                visualize: bool = False, 
                outdir: Optional[str] = None,
                selected_features: Optional[List[str]] = None) -> Union[List[str], Tuple[List[str], pd.DataFrame, Dict[str, float]]]:
    """
    Select features based on VIF (Variance Inflation Factor)
    
    Args:
        X: Feature data
        max_vif: Maximum allowed VIF value, features with VIF above this will be removed
        visualize: Whether to generate visualization plots
        outdir: Output directory, required if visualize is True
        selected_features: List of already selected features, if None use all columns of X
        
    Returns:
        Union[List[str], Tuple[List[str], pd.DataFrame, Dict[str, float]]]: 
            - If detailed_output is False, returns only the list of selected features
            - If detailed_output is True, returns (selected features list, VIF DataFrame, excluded features dictionary)
    """
    if selected_features is None:
        selected_features = X.columns.tolist()
    
    # Only use selected features
    data = X[selected_features].copy()
    
    # Calculate initial VIF
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif["feature"] = data.columns
    
    # Sort
    vif = vif.sort_values(by="VIF", ascending=False)
    
    # Save initial VIF
    initial_vif = vif.copy()
    
    # Save excluded features and their VIF values
    excluded_features = {}
    
    # Iteratively remove features with VIF above threshold
    while (vif["VIF"] > max_vif).any():
        # Get feature with highest VIF
        max_vif_feature = vif.iloc[0]["feature"]
        max_vif_value = vif.iloc[0]["VIF"]
        
        # Add to excluded list
        excluded_features[max_vif_feature] = max_vif_value
        
        # Remove feature from data
        data = data.drop(max_vif_feature, axis=1)
        
        # Stop if too few features remain
        if data.shape[1] < 2:
            print(f"Warning: Too few features remaining, stopping VIF selection")
            break
        
        # Recalculate VIF
        vif = pd.DataFrame()
        vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        vif["feature"] = data.columns
        
        # Sort
        vif = vif.sort_values(by="VIF", ascending=False)
    
    # Get retained features
    selected = data.columns.tolist()
    
    # Output results
    print(f"VIF selection: Selected {len(selected)} features from {len(selected_features)} features, removed {len(excluded_features)} features")
    print(f"Maximum VIF value: {vif['VIF'].max() if not vif.empty else 0}")
    
    # Visualization
    if visualize and outdir:
        os.makedirs(outdir, exist_ok=True)
        
        # Create VIF bar plot
        plt.figure(figsize=(12, 6))
        
        # Sort by VIF value
        initial_vif = initial_vif.sort_values(by="VIF", ascending=True)
        
        # Set colors
        colors = ['red' if feature in excluded_features else 'blue' for feature in initial_vif['feature']]
        
        # Draw bar plot
        bars = plt.barh(initial_vif['feature'], initial_vif['VIF'], color=colors)
        
        # Add threshold line
        plt.axvline(x=max_vif, color='red', linestyle='--', label=f'VIF threshold = {max_vif}')
        
        # Add labels and title
        plt.xlabel('VIF Value')
        plt.ylabel('Feature')
        plt.title('Feature VIF Values (Red indicates removed features)')
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(outdir, 'vif_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save VIF values
        initial_vif.to_csv(os.path.join(outdir, 'vif_values.csv'), index=False)
        
        # Save excluded features
        excluded_df = pd.DataFrame({
            'feature': list(excluded_features.keys()),
            'VIF': list(excluded_features.values())
        })
        excluded_df.to_csv(os.path.join(outdir, 'excluded_features.csv'), index=False)
    
    return selected, vif, excluded_features 