"""
Stepwise Regression Feature Selector

Uses forward, backward, or stepwise (bidirectional) logistic regression for feature selection
"""
import os
import pandas as pd
from typing import List, Optional

from .selector_registry import register_selector, SelectorContext
from ._io import detect_file_type, load_data  # noqa: F401 – re-exported for backward compat

def _set_r_environment(Rhome: Optional[str]) -> None:
    """
    Set up R environment
    
    Args:
        Rhome: R installation path
        
    Raises:
        EnvironmentError: If R installation path is not found
    """
    if Rhome:
        os.environ['R_HOME'] = Rhome
    else:
        raise EnvironmentError("R installation path not found, please specify Rhome in config file")
    
    # import rpy2 related packages after setting R_HOME
    global ro, r, pandas2ri, importr
    import rpy2.robjects as ro
    from rpy2.robjects import r, pandas2ri
    from rpy2.robjects.packages import importr
        
@register_selector('stepwise_r')
def stepwise_selector(
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    direction: str = 'backward',
    outdir: str = None,
    Rhome: str = None,
    context: Optional[SelectorContext] = None,
) -> List[str]:
    """
    Run stepwise regression feature selection using R's stepAIC
    
    Args:
        X: pd.DataFrame, Feature matrix
        y: pd.Series, Target variable
        direction: str, Direction of stepwise selection ('forward', 'backward', or 'both')
        Rhome: str, R installation path
        
    Returns:
        List[str]: List of selected feature names
    """
    if context is not None:
        X = context.X
        y = context.y
        outdir = context.outdir
    if X is None or y is None:
        raise ValueError("stepwise_selector requires X and y inputs.")

    _set_r_environment(Rhome)

    # Initialize R environment
    pandas2ri.activate()
    MASS = importr('MASS')
    stargazer = importr('stargazer')
    
    # Prepare data for R
    X['event'] = y
    X_r = pandas2ri.py2rpy(X)
    r.assign('x_train', X_r)
    
    # Run logistic regression and stepwise AIC in R
    ro.r('''
        model <- glm(event ~ ., data = x_train, family = binomial)
        step_model <- stepAIC(model, trace = FALSE, direction = "{}")
    '''.format(direction))
    
    # Calculate OR and 95% CI
    ro.r('''
        exp_coef <- exp(coef(step_model))
        conf_int <- exp(confint(step_model))
        p_values <- summary(step_model)$coefficients[, 4]
        results <- data.frame(OR = exp_coef, `2.5% CI` = conf_int[, 1], `97.5% CI` = conf_int[, 2], p_value = p_values)
    ''')
    
    # Get selected features
    selected_features = ro.r('names(step_model$coefficients)')
    # Remove intercept term
    selected_features = selected_features[1:]
    # Clean feature names, remove backticks
    selected_features = [feature.strip('`') for feature in selected_features]
    
    # Extract results from R and convert to pandas DataFrame
    try:
        results_r = ro.r('results')
        header_name = ro.r('names(results)')
        feature_name = ro.r('rownames(results)')
        results_df = pd.DataFrame(results_r)
        results_df.columns = feature_name
        results_df.index = header_name
        if outdir:
            results_df.to_csv(os.path.join(outdir, 'stepwise_results.csv'), index=True)
    except:
        results_r = ro.r('results')
        results_df = pd.DataFrame(results_r)
        if outdir:
            results_df.to_csv(os.path.join(outdir, 'stepwise_results.csv'), index=True)

    return selected_features

