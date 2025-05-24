"""
Hosmer-Lemeshow Goodness-of-Fit Test

The Hosmer-Lemeshow test is a statistical test for goodness of fit for binary logistic regression models.
It is widely used to assess the calibration of prediction models.

The test groups subjects into deciles based on predicted probability, then computes a chi-square statistic
comparing observed and expected frequencies in each group.

Reference: 
- Hosmer, D.W., Lemeshow, S. (2000). Applied Logistic Regression, 2nd Edition.
- https://en.wikipedia.org/wiki/Hosmer%E2%80%93Lemeshow_test
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2
from typing import Tuple, Optional


# def hosmer_lemeshow_test(
#     y_true: np.ndarray, 
#     y_pred_proba: np.ndarray, 
#     n_bins: int = 5,
#     method: str = 'quantile'
# ) -> Tuple[float, float, pd.DataFrame]:
#     """
#     Perform Hosmer-Lemeshow goodness-of-fit test for binary classification model calibration
    
#     Args:
#         y_true (np.ndarray): True binary labels (0 or 1)
#         y_pred_proba (np.ndarray): Predicted probabilities
#         n_bins (int): Number of bins to group the data (default: 10)
#         method (str): Binning method, 'quantile' or 'uniform' (default: 'quantile')
        
#     Returns:
#         Tuple[float, float, pd.DataFrame]: 
#             - chi2_statistic: Chi-square test statistic
#             - p_value: P-value of the test
#             - contingency_table: Detailed breakdown by bins
#     """
#     # Input validation
#     if len(y_true) != len(y_pred_proba):
#         raise ValueError("y_true and y_pred_proba must have the same length")
    
#     if not all(np.isin(y_true, [0, 1])):
#         raise ValueError("y_true must contain only 0 and 1")
    
#     if not all((y_pred_proba >= 0) & (y_pred_proba <= 1)):
#         raise ValueError("y_pred_proba must be between 0 and 1")
    
#     # Remove any potential NaN values
#     mask = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
#     y_true_clean = y_true[mask]
#     y_pred_proba_clean = y_pred_proba[mask]
    
#     n_samples = len(y_true_clean)
    
#     # Create bins based on predicted probabilities
#     if method == 'quantile':
#         # Create approximately equal-sized bins based on quantiles
#         bin_edges = np.quantile(y_pred_proba_clean, np.linspace(0, 1, n_bins + 1))
#         # Handle edge case where multiple values are the same
#         bin_edges = np.unique(bin_edges)
#         if len(bin_edges) < n_bins + 1:
#             # Fall back to uniform binning if quantiles don't provide enough unique edges
#             bin_edges = np.linspace(0, 1, n_bins + 1)
#     else:  # uniform
#         bin_edges = np.linspace(0, 1, n_bins + 1)
    
#     # Assign samples to bins
#     bin_indices = np.digitize(y_pred_proba_clean, bin_edges) - 1
#     # Handle edge case for maximum value
#     bin_indices[bin_indices == len(bin_edges) - 1] = len(bin_edges) - 2
    
#     # Initialize arrays for statistics
#     observed_pos = np.zeros(len(bin_edges) - 1)
#     observed_neg = np.zeros(len(bin_edges) - 1)
#     expected_pos = np.zeros(len(bin_edges) - 1)
#     expected_neg = np.zeros(len(bin_edges) - 1)
#     bin_counts = np.zeros(len(bin_edges) - 1)
#     mean_pred_proba = np.zeros(len(bin_edges) - 1)
    
#     # Calculate observed and expected frequencies for each bin
#     for i in range(len(bin_edges) - 1):
#         bin_mask = bin_indices == i
#         bin_samples = y_true_clean[bin_mask]
#         bin_proba = y_pred_proba_clean[bin_mask]
        
#         if len(bin_samples) > 0:
#             bin_counts[i] = len(bin_samples)
#             observed_pos[i] = np.sum(bin_samples == 1)
#             observed_neg[i] = np.sum(bin_samples == 0)
#             expected_pos[i] = np.sum(bin_proba)
#             expected_neg[i] = len(bin_samples) - expected_pos[i]
#             mean_pred_proba[i] = np.mean(bin_proba)
    
#     # Remove empty bins
#     non_empty_bins = bin_counts > 0
#     observed_pos = observed_pos[non_empty_bins]
#     observed_neg = observed_neg[non_empty_bins]
#     expected_pos = expected_pos[non_empty_bins]
#     expected_neg = expected_neg[non_empty_bins]
#     bin_counts = bin_counts[non_empty_bins]
#     mean_pred_proba = mean_pred_proba[non_empty_bins]
    
#     # Calculate chi-square statistic
#     # H-L statistic: sum of (observed - expected)^2 / expected for both positive and negative cases
#     chi2_statistic = np.sum(
#         (observed_pos - expected_pos) ** 2 / expected_pos +
#         (observed_neg - expected_neg) ** 2 / expected_neg
#     )
    
#     # Degrees of freedom = number of groups - 2
#     df = len(observed_pos) - 2
    
#     # Calculate p-value
#     p_value = 1 - stats.chi2.cdf(chi2_statistic, df)
    
#     # Create detailed contingency table
#     contingency_table = pd.DataFrame({
#         'Bin': range(1, len(observed_pos) + 1),
#         'N': bin_counts.astype(int),
#         'Observed_Positive': observed_pos.astype(int),
#         'Expected_Positive': expected_pos,
#         'Observed_Negative': observed_neg.astype(int),
#         'Expected_Negative': expected_neg,
#         'Mean_Predicted_Probability': mean_pred_proba,
#         'Observed_Rate': observed_pos / bin_counts,
#     })
    
#     return chi2_statistic, p_value, contingency_table


def hosmer_lemeshow_test(data, Q=10):
    '''
    data: dataframe format, with ground_truth label name is y,
                                 prediction value column name is y_pred_proba
    '''
    data = data.sort_values('y_pred_proba')
    data['Q_group'] = pd.qcut(data['y_pred_proba'], Q)
    
    y_p = data['y_true'].groupby(data.Q_group).sum()
    y_total = data['y_true'].groupby(data.Q_group).count()
    y_n = y_total - y_p
    
    y_pred_proba_p = data['y_pred_proba'].groupby(data.Q_group).sum()
    y_pred_proba_total = data['y_pred_proba'].groupby(data.Q_group).count()
    y_pred_proba_n = y_pred_proba_total - y_pred_proba_p
    
    hltest = (((y_p - y_pred_proba_p)**2 / y_pred_proba_p) + ((y_n - y_pred_proba_n)**2 / y_pred_proba_n)).sum()
    pval = 1 - chi2.cdf(hltest, Q-2)
    
    return hltest, pval

def interpret_hl_test(chi2_statistic: float, p_value: float, alpha: float = 0.05) -> str:
    """
    Interpret Hosmer-Lemeshow test results
    
    Args:
        chi2_statistic (float): Chi-square test statistic
        p_value (float): P-value of the test
        alpha (float): Significance level (default: 0.05)
        
    Returns:
        str: Interpretation of the test results
    """
    if p_value < alpha:
        return (f"Reject null hypothesis (p = {p_value:.4f} < {alpha}). "
                f"The model shows poor calibration/goodness-of-fit.")
    else:
        return (f"Fail to reject null hypothesis (p = {p_value:.4f} >= {alpha}). "
                f"The model shows acceptable calibration/goodness-of-fit.")


if __name__ == "__main__":
    # Example usage for Hosmer-Lemeshow test demonstration
    print("Hosmer-Lemeshow Test Example:")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    
    # Create a well-calibrated model example
    print("\nExample 1: Well-calibrated model")
    print("-" * 30)
    
    # Generate true probabilities and outcomes
    y_true_calibrated = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    y_pred_proba_calibrated = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    
    # Perform HL test
    chi2_stat, p_val, table = Hosmer_Lemeshow_test(y_true_calibrated, y_pred_proba_calibrated)
    
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    print(f"Interpretation: {interpret_hl_test(chi2_stat, p_val)}")
    print("\nContingency Table:")
    print(table.round(4))
    
    # Create a poorly-calibrated model example
    print("\n\nExample 2: Poorly-calibrated model")
    print("-" * 30)
    
    # Generate systematically biased predictions
    y_true_poor = np.random.binomial(1, 0.3, n_samples)  # True rate around 30%
    y_pred_proba_poor = np.random.beta(5, 2, n_samples)  # Biased towards higher probabilities
    
    # Perform HL test
    chi2_stat_poor, p_val_poor, table_poor = hosmer_lemeshow_test(y_true_poor, y_pred_proba_poor)
    
    print(f"Chi-square statistic: {chi2_stat_poor:.4f}")
    print(f"P-value: {p_val_poor:.4f}")
    print(f"Interpretation: {interpret_hl_test(chi2_stat_poor, p_val_poor)}")
    print("\nContingency Table:")
    print(table_poor.round(4)) 