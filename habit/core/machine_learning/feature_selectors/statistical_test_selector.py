"""
Statistical Test Feature Selector

Implementation of feature selection based on t-test or Mann-Whitney U test,
automatically choosing the appropriate test based on data normality.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict, Optional, Union
import os
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .selector_registry import register_selector

@register_selector('statistical_test')
def statistical_test_selector(
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        p_threshold: float = 0.05,
        n_features_to_select: Optional[int] = None,
        normality_test_threshold: float = 0.05,
        plot_importance: bool = True,
        outdir: Optional[str] = None,
        force_test: Optional[str] = None,  # 'ttest' or 'mannwhitney'
        **kwargs
    ) -> List[str]:
    """
    Apply statistical test (t-test or Mann-Whitney U test) based feature selection
    
    Args:
        X: Feature data
        y: Target variable (binary)
        selected_features: List of features to select from
        p_threshold: P-value threshold for feature selection (default: 0.05)
        n_features_to_select: Optional number of features to select (if specified, overrides p_threshold)
        normality_test_threshold: Threshold for Shapiro-Wilk normality test (default: 0.05)
        plot_importance: Whether to plot feature importance
        outdir: Output directory for results
        force_test: Force a specific test ('ttest' or 'mannwhitney'), otherwise auto-select
        **kwargs: Additional arguments
        
    Returns:
        List[str]: Selected feature names
    """
    # Get feature subset
    X_subset = X[selected_features]
    
    # Check if target is binary
    unique_values = y.unique()
    if len(unique_values) != 2:
        raise ValueError("Target variable must be binary for t-test or Mann-Whitney U test")
    
    # Get indices for each class
    class_0_indices = y == unique_values[0]
    class_1_indices = y == unique_values[1]
    
    # Initialize results
    selected_test_type = []
    p_values = []
    test_stats = []
    
    # Perform statistical tests for each feature
    for feature in selected_features:
        feature_values = X_subset[feature].values
        group0_values = feature_values[class_0_indices]
        group1_values = feature_values[class_1_indices]
        
        # Check if the forced test is specified
        if force_test is not None:
            test_type = force_test
        else:
            # Test for normality using Shapiro-Wilk test
            _, p_val_0 = stats.shapiro(group0_values) if len(group0_values) < 5000 else (0, 0)
            _, p_val_1 = stats.shapiro(group1_values) if len(group1_values) < 5000 else (0, 0)
            
            # If both groups are normally distributed, use t-test, otherwise use Mann-Whitney U test
            test_type = "ttest" if (p_val_0 > normality_test_threshold and p_val_1 > normality_test_threshold) else "mannwhitney"
        
        # Apply appropriate test
        if test_type == "ttest":
            stat, p_value = stats.ttest_ind(group0_values, group1_values, equal_var=False)  # Welch's t-test
        else:  # "mannwhitney"
            stat, p_value = stats.mannwhitneyu(group0_values, group1_values)
        
        # Store results
        test_stats.append(abs(stat))  # Use absolute value for ranking
        p_values.append(p_value)
        selected_test_type.append(test_type)
    
    # Create feature ranking DataFrame
    feature_ranking = pd.DataFrame({
        'feature': selected_features,
        'test_statistic': test_stats,
        'pvalue': p_values,
        'test_type': selected_test_type
    })
    
    # Sort by test statistic (larger absolute values are more significant)
    feature_ranking = feature_ranking.sort_values('test_statistic', ascending=False).reset_index(drop=True)
    
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
    
    # Count tests by type
    test_counts = feature_ranking['test_type'].value_counts()
    test_summary = ", ".join([f"{count} {test}" for test, count in test_counts.items()])
    
    print(f"Selected {len(selected_features_result)} features from {len(selected_features)} features using {selection_method}")
    print(f"Tests used: {test_summary}")
    
    # Save results if output directory specified
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        
        # Save selected features
        result_file = os.path.join(outdir, 'statistical_test_selected_features.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'selected_features': selected_features_result,
                'n_features_selected': len(selected_features_result),
                'original_feature_count': len(selected_features),
                'selection_method': selection_method,
                'tests_used': test_counts.to_dict()
            }, f, ensure_ascii=False, indent=4)
            
        # Save feature ranking
        ranking_file = os.path.join(outdir, 'statistical_test_feature_ranking.csv')
        feature_ranking.to_csv(ranking_file, index=False)
        
        # Plot feature importance
        if plot_importance:
            plt.figure(figsize=(12, 8))
            top_features = feature_ranking.head(min(20, len(selected_features)))
            
            # Color by test type
            palette = {"ttest": "blue", "mannwhitney": "green"}
            ax = sns.barplot(x='test_statistic', y='feature', hue='test_type', data=top_features, palette=palette)
            
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
                ax.text(top_features.loc[top_features['feature'] == feature, 'test_statistic'].values[0] + 0.5, 
                        i, stars, ha='left', va='center')
            
            plt.title('Feature Importance by Statistical Test')
            plt.legend(title='Test Type')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'statistical_test_feature_importance.pdf'))
            plt.close()
            
            # Plot p-value distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(feature_ranking['pvalue'], kde=True, bins=20)
            plt.axvline(x=p_threshold, color='red', linestyle='--', label=f'P-value threshold: {p_threshold}')
            plt.title('Distribution of P-values')
            plt.xlabel('P-value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'statistical_test_pvalue_distribution.pdf'))
            plt.close()
            
            # Violin plot comparing tests
            plt.figure(figsize=(10, 6))
            sns.violinplot(x='test_type', y='pvalue', data=feature_ranking)
            plt.axhline(y=p_threshold, color='red', linestyle='--', label=f'P-value threshold: {p_threshold}')
            plt.title('P-values by Test Type')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'statistical_test_comparison.pdf'))
            plt.close()
        
    return selected_features_result 