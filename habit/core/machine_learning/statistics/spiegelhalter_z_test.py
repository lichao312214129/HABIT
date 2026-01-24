"""
Spiegelhalter Z-test

The Spiegelhalter Z-test is a statistical test used to compare the performance of a binary classification model.
It is named after the British statistician David Spiegelhalter.

REF:https://stats.stackexchange.com/questions/635843/very-low-hosmer-and-lemeshow-goodness-of-fit-in-logistic-regression

"""

import numpy as np
import scipy.stats as stats

def spiegelhalter_z_test(y_true, y_pred):
    n = len(y_true)
    o_minus_e = y_true - y_pred
    var = y_pred * (1 - y_pred)
    z = np.sum(o_minus_e) / np.sqrt(np.sum(var))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


if __name__ == "__main__":
    # Example data for Spiegelhalter Z-test demonstration
    # Simulating true labels and predicted probabilities
    np.random.seed(42)  # Ensure reproducibility
    
    # Generate sample data
    n_samples = 10
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # Binary true labels
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Predicted probabilities
    
    # Perform Spiegelhalter Z-test
    z_statistic, p_value = spiegelhalter_z_test(y_true, y_pred)
    
    # Print results
    print("Spiegelhalter Z-Test Example:")
    print(f"Z-statistic: {z_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    