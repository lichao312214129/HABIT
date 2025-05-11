"""
Lasso Feature Selector

Uses L1 regularization for feature selection, automatically identifying important features
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.preprocessing import StandardScaler
from .selector_registry import register_selector

@register_selector('lasso')
def lasso_selector(X: pd.DataFrame, 
                  y: pd.Series,
                  cv: int = 10, 
                  n_alphas: int = 100,
                  alphas: Optional[List[float]] = None,
                  random_state: int = 42,
                  visualize: bool = False,
                  outdir: Optional[str] = None,
                  selected_features: Optional[List[str]] = None) -> Union[List[str], Tuple[List[str], float, np.ndarray, np.ndarray]]:
    """
    Feature selection based on Lasso regression
    
    Args:
        X: Feature data
        y: Target variable
        cv: Number of cross-validation folds
        n_alphas: Number of alpha parameters (if alphas is None)
        alphas: List of specified alpha parameters
        random_state: Random seed
        visualize: Whether to generate visualization results
        outdir: Output directory, required if visualize is True
        selected_features: List of already selected features, if None use all columns of X
        
    Returns:
        Union[List[str], Tuple[List[str], float, np.ndarray, np.ndarray]]:
            - If detailed_output is False, returns only the list of selected features
            - If detailed_output is True, returns (selected features list, optimal alpha, alpha path, coefficient path)
    """
    if selected_features is None:
        selected_features = X.columns.tolist()
    
    # Only use selected features
    X_selected = X[selected_features]
    
    # Create LassoCV object
    lasso_cv = LassoCV(
        cv=cv, 
        n_alphas=n_alphas,
        alphas=alphas,
        random_state=random_state,
        n_jobs=-1  # Use all available CPU cores
    )
    
    # Fit the model
    lasso_cv.fit(X, y)
    
    # Get optimal alpha value
    best_alpha = lasso_cv.alpha_
    print(f"Lasso selection: Optimal alpha value is {best_alpha:.6f}")
    
    # Get feature coefficients
    coefs = lasso_cv.coef_
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'coefficient': coefs
    })
    
    # Select features with non-zero coefficients
    selected = feature_importance[feature_importance['coefficient'] != 0]['feature'].tolist()
    
    # Output results
    print(f"Lasso selection: Selected {len(selected)} features from {len(selected_features)} features")
    
    # Calculate Lasso path (for visualization)
    alphas_path, coefs_path, _ = lasso_path(X, y, alphas=alphas)
    
    # Visualization
    if visualize and outdir:
        os.makedirs(outdir, exist_ok=True)
        
        # Create Lasso path plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # First subplot: MSE path
        alphas = lasso_cv.alphas_
        mse_path = lasso_cv.mse_path_.mean(axis=1)

        ax1.plot(alphas, mse_path, '-', label='Mean Squared Error')
        ax1.axvline(best_alpha, color='r', linestyle='--', label=f'Optimal alpha = {best_alpha:.6f}')
        ax1.set_xscale('log')
        ax1.set_xlabel('alpha')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('Cross-validation MSE vs alpha')
        ax1.legend()
        ax1.grid(True)
        
        # Second subplot: Coefficient path
        for i, feature in enumerate(selected_features):
            ax2.plot(alphas_path, coefs_path[i], label=feature if feature in selected else None)
        
        # Highlight selected features
        for i, feature in enumerate(selected_features):
            if feature in selected:
                ax2.plot(alphas_path, coefs_path[i], linewidth=2)
        
        ax2.axvline(best_alpha, color='r', linestyle='--')
        ax2.set_xscale('log')
        ax2.set_xlabel('alpha')
        ax2.set_ylabel('Coefficient')
        ax2.set_title('Lasso Coefficient Path')
        ax2.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(outdir, 'lasso_path.pdf'), bbox_inches='tight')
        plt.close()
        
        # Save feature importance
        feature_importance = feature_importance.sort_values(by='coefficient', key=abs, ascending=False)
        feature_importance.to_csv(os.path.join(outdir, 'lasso_feature_importance.csv'), index=False)
        
        # Create feature importance bar plot
        plt.figure(figsize=(12, 8))
        
        # Only show features with non-zero coefficients
        nonzero_features = feature_importance[feature_importance['coefficient'] != 0]
        
        # Sort by absolute value
        nonzero_features = nonzero_features.reindex(nonzero_features['coefficient'].abs().sort_values(ascending=True).index)
        
        # Set colors
        colors = ['red' if c < 0 else 'blue' for c in nonzero_features['coefficient']]
        
        # Draw bar plot
        plt.barh(nonzero_features['feature'], nonzero_features['coefficient'], color=colors)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Coefficient')
        plt.ylabel('Feature')
        plt.title('Lasso Feature Importance (Red: negative coefficients, Blue: positive coefficients)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'lasso_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return selected, best_alpha, alphas_path, coefs_path 