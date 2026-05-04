"""
Univariate Logistic Regression Feature Selector

Performs feature selection using univariate logistic regression
"""
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import statsmodels.formula.api as smf
from pathlib import Path

from habit.utils.progress_utils import CustomTqdm
from .selector_registry import register_selector, SelectorContext
from ._io import detect_file_type, load_data  # noqa: F401 – re-exported for backward compat

@register_selector('univariate_logistic')
def univariate_logistic_selector(
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    outdir: str = None,
    alpha: float = 0.05,
    context: Optional[SelectorContext] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Perform univariate logistic regression feature selection
    
    Args:
        X: pd.DataFrame, Feature matrix
        y: pd.Series, Target variable
        alpha: float, Significance level for feature selection
        
    Returns:
        Tuple[List[str], Dict[str, Any]]:
            - Selected feature list.
            - Metadata dict containing per-feature statistics.
    """
    if context is not None:
        X = context.X
        y = context.y
        outdir = context.outdir
    if X is None or y is None:
        raise ValueError("univariate_logistic_selector requires X and y inputs.")

    results = []
    header = X.columns
    
    # Perform univariate logistic regression for each feature
    progress = CustomTqdm(total=X.shape[1], desc="Univariate logistic")
    for i in range(X.shape[1]):
        x_ = X.iloc[:, i]
        model = smf.logit(formula="event ~ x", data=pd.DataFrame({'event': y, 'x': x_}))
        result = model.fit(verbose=0)
        
        # Extract statistics
        p = result.pvalues[1]
        OR = np.exp(result.params[1])
        low, high = np.exp(result.conf_int().iloc[1, :])
        
        # Store results
        results.append(pd.DataFrame({
            'p_value': p,
            'odds_ratio': OR,
            'ci_lower': low,
            'ci_upper': high
        }, index=[header[i]]))
        progress.update(1)
    if hasattr(progress, "close"):
        progress.close()
    
    # Combine all results
    results_df = pd.concat(results, axis=0)
    
    # Select features based on p-value threshold
    selected_features = np.array(header)[results_df['p_value'] < alpha].tolist()
    results_dict = {
        'selected_features': selected_features,
        'results': results_df.to_dict(),
        'alpha': alpha
    } 

    # save results_df to csv
    if outdir is not None:
        results_df.to_csv(os.path.join(outdir, 'univariate_logistic_results.csv'), index=True)

    return selected_features, results_dict

