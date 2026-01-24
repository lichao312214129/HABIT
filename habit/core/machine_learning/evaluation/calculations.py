"""
Shared Calculation Functions

This module contains shared calculation functions used across the machine learning module.
Extracting these functions helps avoid circular dependencies and improves code organization.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Union, Optional


def calculate_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate confusion matrix-based metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing tp, tn, fp, fn
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def calculate_sensitivity(tp: int, fn: int) -> float:
    """
    Calculate sensitivity (recall, true positive rate).
    
    Args:
        tp: True positives
        fn: False negatives
        
    Returns:
        Sensitivity value
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def calculate_specificity(tn: int, fp: int) -> float:
    """
    Calculate specificity (true negative rate).
    
    Args:
        tn: True negatives
        fp: False positives
        
    Returns:
        Specificity value
    """
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def calculate_ppv(tp: int, fp: int) -> float:
    """
    Calculate positive predictive value (precision).
    
    Args:
        tp: True positives
        fp: False positives
        
    Returns:
        PPV value
    """
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def calculate_npv(tn: int, fn: int) -> float:
    """
    Calculate negative predictive value.
    
    Args:
        tn: True negatives
        fn: False negatives
        
    Returns:
        NPV value
    """
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score
    """
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def calculate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calculate accuracy from confusion matrix components.
    
    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives
        
    Returns:
        Accuracy value
    """
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def calculate_youden_index(sensitivity: float, specificity: float) -> float:
    """
    Calculate Youden index.
    
    Args:
        sensitivity: Sensitivity value
        specificity: Specificity value
        
    Returns:
        Youden index
    """
    return sensitivity + specificity - 1


def apply_threshold_to_probabilities(
    y_pred_proba: np.ndarray, 
    threshold: float
) -> np.ndarray:
    """
    Apply threshold to predicted probabilities to get binary predictions.
    
    Args:
        y_pred_proba: Predicted probabilities
        threshold: Threshold value
        
    Returns:
        Binary predictions
    """
    return (np.array(y_pred_proba) >= threshold).astype(int)


def find_optimal_threshold(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    method: str = 'youden'
) -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        method: Method to use ('youden', 'balanced_accuracy', 'f1')
        
    Returns:
        Tuple of (optimal_threshold, optimal_value)
    """
    from sklearn.metrics import roc_curve
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    if method == 'youden':
        youden_indices = tpr + (1 - fpr) - 1
        optimal_idx = np.argmax(youden_indices)
        optimal_threshold = thresholds[optimal_idx]
        optimal_value = youden_indices[optimal_idx]
    elif method == 'balanced_accuracy':
        balanced_accuracy = (tpr + (1 - fpr)) / 2
        optimal_idx = np.argmax(balanced_accuracy)
        optimal_threshold = thresholds[optimal_idx]
        optimal_value = balanced_accuracy[optimal_idx]
    elif method == 'f1':
        f1_scores = []
        for threshold in thresholds:
            y_pred = apply_threshold_to_probabilities(y_pred_proba, threshold)
            cm = calculate_confusion_matrix_metrics(y_true, y_pred)
            precision = calculate_ppv(cm['tp'], cm['fp'])
            recall = calculate_sensitivity(cm['tp'], cm['fn'])
            f1 = calculate_f1_score(precision, recall)
            f1_scores.append(f1)
        f1_scores = np.array(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_value = f1_scores[optimal_idx]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(optimal_threshold), float(optimal_value)


def calculate_net_benefit(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    threshold: float
) -> float:
    """
    Calculate net benefit for decision curve analysis.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Threshold value
        
    Returns:
        Net benefit value
    """
    if isinstance(threshold, list):
        threshold = threshold[0]
    if threshold >= 0.999:
        return 0.0
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = apply_threshold_to_probabilities(y_pred_proba, threshold)
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    
    benefit = (true_positives / n) - (false_positives / n) * (threshold / (1 - threshold))
    return benefit if np.isfinite(benefit) else 0.0


def calculate_confidence_interval(
    value: float, 
    std_error: float, 
    alpha: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a given value and standard error.
    
    Args:
        value: Point estimate
        std_error: Standard error
        alpha: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    z_score = stats.norm.ppf(1 - (1 - alpha) / 2)
    lower = value - z_score * std_error
    upper = value + z_score * std_error
    
    # Clamp to valid range if needed
    if 0 <= value <= 1:
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    
    return float(lower), float(upper)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric_func: callable,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None
) -> Tuple[float, float, np.ndarray]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric_func: Function to calculate the metric
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (metric_value, std_error, bootstrap_samples)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(y_true)
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]
        
        try:
            value = metric_func(y_true_boot, y_pred_proba_boot)
            bootstrap_values.append(value)
        except:
            pass
    
    bootstrap_values = np.array(bootstrap_values)
    metric_value = np.mean(bootstrap_values)
    std_error = np.std(bootstrap_values, ddof=1)
    
    return float(metric_value), float(std_error), bootstrap_values


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate calibration metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration assessment
        
    Returns:
        Dictionary containing calibration metrics
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges) - 1
    bin_indices[bin_indices == len(bin_edges) - 1] = len(bin_edges) - 2
    
    observed_fractions = []
    predicted_fractions = []
    bin_counts = []
    
    for i in range(len(bin_edges) - 1):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            observed_fractions.append(np.mean(y_true[bin_mask]))
            predicted_fractions.append(np.mean(y_pred_proba[bin_mask]))
            bin_counts.append(np.sum(bin_mask))
    
    observed_fractions = np.array(observed_fractions)
    predicted_fractions = np.array(predicted_fractions)
    bin_counts = np.array(bin_counts)
    
    # Calculate calibration error
    calibration_error = np.mean(np.abs(observed_fractions - predicted_fractions))
    
    # Calculate expected calibration error (weighted by bin size)
    weights = bin_counts / np.sum(bin_counts)
    expected_calibration_error = np.sum(weights * np.abs(observed_fractions - predicted_fractions))
    
    return {
        'calibration_error': float(calibration_error),
        'expected_calibration_error': float(expected_calibration_error),
        'observed_fractions': observed_fractions,
        'predicted_fractions': predicted_fractions,
        'bin_counts': bin_counts
    }


def clean_data_arrays(*arrays: np.ndarray) -> List[np.ndarray]:
    """
    Remove NaN values from multiple arrays simultaneously.
    
    Args:
        *arrays: Variable number of numpy arrays
        
    Returns:
        List of arrays with NaN rows removed
    """
    if not arrays:
        return []
    
    mask = ~np.isnan(arrays[0])
    for arr in arrays[1:]:
        if arr.ndim > 1:
            mask &= ~np.any(np.isnan(arr), axis=1)
        else:
            mask &= ~np.isnan(arr)
    
    return [arr[mask] for arr in arrays]


def validate_binary_classification_data(
    y_true: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Validate and clean binary classification data.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (optional)
        y_pred: Predicted labels (optional)
        
    Returns:
        Tuple of cleaned (y_true, y_pred_proba, y_pred)
    """
    y_true = np.array(y_true)
    
    # Check for valid labels
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")
    
    # Clean y_pred_proba if provided
    if y_pred_proba is not None:
        y_pred_proba = np.array(y_pred_proba)
        if not np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)):
            raise ValueError("y_pred_proba must be between 0 and 1")
    
    # Clean y_pred if provided
    if y_pred is not None:
        y_pred = np.array(y_pred)
        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("y_pred must contain only 0 and 1")
    
    # Remove NaN values
    arrays_to_clean = [y_true]
    if y_pred_proba is not None:
        arrays_to_clean.append(y_pred_proba)
    if y_pred is not None:
        arrays_to_clean.append(y_pred)
    
    cleaned_arrays = clean_data_arrays(*arrays_to_clean)
    
    y_true_cleaned = cleaned_arrays[0]
    y_pred_proba_cleaned = cleaned_arrays[1] if y_pred_proba is not None else None
    y_pred_cleaned = cleaned_arrays[2] if y_pred is not None else None
    
    return y_true_cleaned, y_pred_proba_cleaned, y_pred_cleaned
