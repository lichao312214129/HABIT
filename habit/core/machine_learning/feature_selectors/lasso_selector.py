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
        
        # Import and setup publication font
        from ....utils.font_config import setup_publication_font
        setup_publication_font()
        
        # Set larger font sizes for all plots with Arial font
        plt.rcParams.update({'font.size': 14, 
                             'axes.titlesize': 16, 
                             'axes.labelsize': 14,
                             'xtick.labelsize': 12,
                             'ytick.labelsize': 12,
                             'legend.fontsize': 12})
        
        # Create Lasso path plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # First subplot: MSE path
        alphas = lasso_cv.alphas_
        mse_path = lasso_cv.mse_path_.mean(axis=1)

        ax1.plot(alphas, mse_path, '-', linewidth=2, label='Mean Squared Error')
        ax1.axvline(best_alpha, color='r', linestyle='--', linewidth=2, label=f'Optimal alpha = {best_alpha:.6f}')
        ax1.set_xscale('log')
        ax1.set_xlabel('Alpha', fontsize=14, fontfamily='Arial')
        ax1.set_ylabel('Mean Squared Error', fontsize=14, fontfamily='Arial')
        ax1.set_title('Cross-validation MSE vs Alpha', fontsize=16, fontfamily='Arial')
        ax1.legend(fontsize=12)
        ax1.grid(True)
        
        # Second subplot: Coefficient path
        # Plot all features but with thin lines and no labels first
        for i, feature in enumerate(selected_features):
            ax2.plot(alphas_path, coefs_path[i], linewidth=0.7, alpha=0.4, color='gray')
        
        # Calculate feature importance at optimal alpha point
        # Find the closest alpha in the path to our optimal alpha
        optimal_idx = np.argmin(np.abs(alphas_path - best_alpha))
        coefs_at_optimal = coefs_path[:, optimal_idx]
        
        # Create a dataframe with features and their importance
        feature_importance_at_alpha = pd.DataFrame({
            'feature': selected_features,
            'coefficient': coefs_at_optimal
        })
        
        # Sort by absolute coefficient value
        feature_importance_at_alpha = feature_importance_at_alpha.reindex(
            feature_importance_at_alpha['coefficient'].abs().sort_values(ascending=False).index
        )
        
        # Get top N features to label (adjust this number based on your preference)
        num_features_to_label = min(10, len(selected))  # Show at most 10 features
        top_features = feature_importance_at_alpha['feature'].head(num_features_to_label).tolist()
        
        # Highlight and label selected top features
        for i, feature in enumerate(selected_features):
            if feature in top_features:
                line, = ax2.plot(alphas_path, coefs_path[i], linewidth=2, label=feature)
        
        # Add a legend for the number of features shown vs total
        if len(selected) > num_features_to_label:
            ax2.text(
                0.02, 0.02, 
                f'Showing top {num_features_to_label} of {len(selected)} selected features', 
                transform=ax2.transAxes, 
                fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        ax2.axvline(best_alpha, color='r', linestyle='--', linewidth=2)
        ax2.set_xscale('log')
        ax2.set_xlabel('Alpha', fontsize=14, fontfamily='Arial')
        ax2.set_ylabel('Coefficient', fontsize=14, fontfamily='Arial')
        ax2.set_title('Lasso Coefficient Path', fontsize=16, fontfamily='Arial')
        ax2.grid(True)
        
        # Optimize legend layout
        # Calculate number of columns based on number of features
        n_cols = min(5, len(top_features))  # Maximum 5 columns
        n_rows = (len(top_features) + n_cols - 1) // n_cols  # Ceiling division
        
        # Adjust figure size to accommodate legend
        fig.set_size_inches(12, 12 + n_rows * 0.5)  # Increase height based on number of rows
        
        # Place legend at the bottom of the plot
        ax2.legend(
            bbox_to_anchor=(0.5, -0.15 - n_rows * 0.05),  # Position below the plot
            loc='upper center',
            borderaxespad=0.,
            ncol=n_cols,
            fontsize=10,
            frameon=True,
            framealpha=0.8
        )
        
        # Adjust layout to make room for legend
        plt.tight_layout(rect=[0, 0.1 + n_rows * 0.05, 1, 1])  # Leave space at bottom for legend
        
        # Save plot
        plt.savefig(os.path.join(outdir, 'lasso_path.pdf'), bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save feature importance
        feature_importance = feature_importance.sort_values(by='coefficient', key=abs, ascending=False)
        feature_importance.to_csv(os.path.join(outdir, 'lasso_feature_importance.csv'), index=False)
        
        # Create feature importance bar plot
        plt.figure(figsize=(14, 10))
        
        # Only show features with non-zero coefficients
        nonzero_features = feature_importance[feature_importance['coefficient'] != 0]
        
        # Sort by absolute value
        nonzero_features = nonzero_features.reindex(nonzero_features['coefficient'].abs().sort_values(ascending=True).index)
        
        # Set colors
        colors = ['red' if c < 0 else 'blue' for c in nonzero_features['coefficient']]
        
        # Draw bar plot
        bars = plt.barh(nonzero_features['feature'], nonzero_features['coefficient'], color=colors, height=0.7)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Coefficient', fontsize=14, fontfamily='Arial')
        plt.ylabel('Feature', fontsize=14, fontfamily='Arial')
        plt.title('Lasso Feature Importance', fontsize=16, fontfamily='Arial')
        
        # Add a note about colors
        plt.figtext(0.5, 0.01, 'Red: negative coefficients, Blue: positive coefficients', 
                   ha='center', fontsize=12, fontfamily='Arial')
        
        # Adjust layout to ensure all feature names are visible
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save with high resolution
        plt.savefig(os.path.join(outdir, 'lasso_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create an additional interactive HTML plot for coefficient paths when there are many features
        if len(selected) > 20:  # For datasets with many features
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Create plotly figure
                fig = make_subplots(rows=1, cols=1)
                
                # Add traces for each feature
                for i, feature in enumerate(selected_features):
                    if feature in selected:  # Only plot selected features
                        fig.add_trace(
                            go.Scatter(
                                x=alphas_path, 
                                y=coefs_path[i], 
                                name=feature,
                                mode='lines',
                                line=dict(width=2) if feature in top_features else dict(width=1, dash='dot')
                            )
                        )
                
                # Add vertical line for optimal alpha
                fig.add_vline(x=best_alpha, line_dash="dash", line_color="red", 
                             annotation_text=f"Optimal alpha = {best_alpha:.6f}",
                             annotation_position="top right")
                
                # Update layout
                fig.update_layout(
                    title='Interactive Lasso Coefficient Path',
                    xaxis_title='Alpha',
                    yaxis_title='Coefficient',
                    xaxis_type='log',
                    legend_title='Features',
                    hovermode='closest',
                    width=1000,
                    height=600
                )
                
                # Save to HTML
                fig.write_html(os.path.join(outdir, 'lasso_path_interactive.html'))
                print(f"Created interactive coefficient path plot: {os.path.join(outdir, 'lasso_path_interactive.html')}")
            except ImportError:
                print("Plotly not installed. Skipping interactive plot creation.")
    
    return selected, best_alpha, alphas_path, coefs_path 