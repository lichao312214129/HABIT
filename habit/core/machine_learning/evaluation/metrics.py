"""
Metrics Calculation Module
Provides functions for calculating various evaluation metrics
"""

import numpy as np
from typing import Dict, Tuple, Union, List
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from scipy import stats
from habit.core.machine_learning.statistics.delong_test import delong_roc_variance, delong_roc_test

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate model evaluation metrics
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels or predicted probabilities (if same as y_pred_proba)
        y_pred_proba (np.ndarray): Predicted probabilities
        
    Returns:
        Dict[str, float]: Dictionary containing various metrics
    """
    # Check if y_pred is the same as y_pred_proba (probability values)
    if np.array_equal(y_pred, y_pred_proba):
        # Convert probabilities to binary predictions using a threshold of 0.5
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
        cm = metrics.confusion_matrix(y_true, y_pred_binary)
    else:
        # Use y_pred as is (already binary)
        cm = metrics.confusion_matrix(y_true, y_pred)
    
    metrics_dict = {
        'accuracy': accuracy_score(y_true, y_pred if not np.array_equal(y_pred, y_pred_proba) else (y_pred_proba >= 0.5).astype(int)),
        'sensitivity': cm[1, 1] / (cm[1, 1] + cm[1, 0]),
        'specificity': cm[0, 0] / (cm[0, 0] + cm[0, 1]),
        'ppv': cm[1, 1] / (cm[1, 1] + cm[0, 1]),
        'npv': cm[0, 0] / (cm[0, 0] + cm[1, 0]),
        'auc': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics_dict

def apply_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Apply a given threshold to predicted probabilities and calculate metrics
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        threshold (float): Threshold to apply
        
    Returns:
        Dict[str, float]: Dictionary containing various metrics at the given threshold
    """
    # Make binary predictions using the specified threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Compile metrics
    metrics_dict = {
        'threshold': float(threshold),
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1_score,
        'auc': auc
    }
    
    return metrics_dict

def calculate_metrics_youden(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate metrics based on the optimal Youden index (sensitivity + specificity - 1)
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        
    Returns:
        Dict: Dictionary containing:
            - 'threshold': Optimal threshold based on Youden index
            - 'youden_index': Maximum Youden index value
            - 'metrics': Dictionary of metrics at the optimal threshold
            - 'all_thresholds': Array of all tested thresholds
            - 'all_sensitivities': Array of sensitivities at all thresholds
            - 'all_specificities': Array of specificities at all thresholds
            - 'all_youdens': Array of Youden indices at all thresholds
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate sensitivities and specificities
    sensitivities = tpr
    specificities = 1 - fpr
    
    # Calculate Youden index for each threshold
    youden_indices = sensitivities + specificities - 1
    
    # Find threshold with maximum Youden index
    optimal_idx = np.argmax(youden_indices)
    optimal_threshold = thresholds[optimal_idx]
    max_youden = youden_indices[optimal_idx]
    
    # Calculate metrics at optimal threshold
    metrics_dict = apply_threshold(y_true, y_pred_proba, optimal_threshold)
    
    # Return results
    return {
        'threshold': float(optimal_threshold),
        'youden_index': float(max_youden),
        'metrics': metrics_dict,
        'all_thresholds': thresholds.tolist(),
        'all_sensitivities': sensitivities.tolist(),
        'all_specificities': specificities.tolist(),
        'all_youdens': youden_indices.tolist()
    }

def apply_youden_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Apply a pre-determined Youden threshold to evaluate predictions
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        threshold (float): Threshold determined previously (usually from training data)
        
    Returns:
        Dict: Dictionary containing metrics at the provided threshold
    """
    # Calculate metrics at the given threshold
    metrics_dict = apply_threshold(y_true, y_pred_proba, threshold)
    
    # Calculate ROC AUC for reference
    auc = roc_auc_score(y_true, y_pred_proba)
    metrics_dict['auc'] = auc
    
    # Calculate sensitivity and specificity for Youden index
    sensitivity = metrics_dict['sensitivity']
    specificity = metrics_dict['specificity']
    youden_index = sensitivity + specificity - 1
    
    # Return results
    return {
        'threshold': float(threshold),
        'youden_index': float(youden_index),
        'metrics': metrics_dict
    }

def calculate_metrics_at_target(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               target_metrics: Dict[str, float]) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate metrics at thresholds that achieve target metric values
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        target_metrics (Dict[str, float]): Dictionary of target metrics and their values
                                          Supported metrics: 'sensitivity', 'specificity', 'ppv', 'npv'
                                          Example: {'sensitivity': 0.9, 'specificity': 0.8}
    
    Returns:
        Dict: Dictionary containing results for each threshold that satisfies at least one target metric:
            - 'thresholds': Dictionary mapping metric name to threshold that achieves the target
            - 'metrics_at_thresholds': Dictionary mapping metric name to all metrics at that threshold
            - 'combined_results': Results for thresholds that satisfy multiple target metrics (if applicable)
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Add special threshold points to ensure we cover the entire range
    if len(thresholds) > 0:
        if thresholds[0] != 1.0:
            thresholds = np.append(thresholds, 1.0)
            fpr = np.append(fpr, 0.0)
            tpr = np.append(tpr, 0.0)
        if thresholds[-1] != 0.0:
            thresholds = np.append(thresholds, 0.0)
            fpr = np.append(fpr, 1.0)
            tpr = np.append(tpr, 1.0)
    
    # Sort thresholds from highest to lowest
    sorted_indices = np.argsort(thresholds)[::-1]
    thresholds = thresholds[sorted_indices]
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    
    # Calculate all metrics for all thresholds
    all_metrics = {}
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = metrics.confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        all_metrics[threshold] = {
            'threshold': threshold,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'f1_score': f1_score
        }
    
    # Find thresholds that achieve target metrics
    target_thresholds = {}
    metrics_at_thresholds = {}
    
    for metric_name, target_value in target_metrics.items():
        # Skip unsupported metrics
        if metric_name not in ['sensitivity', 'specificity', 'ppv', 'npv']:
            continue
        
        # Find threshold that achieves the target metric value
        best_threshold = None
        
        # Different search strategies for different metrics
        if metric_name in ['specificity', 'ppv']:
            # For specificity and PPV: as threshold increases, these metrics increase
            # Start from lowest threshold and find the first threshold that meets the target
            for threshold in reversed(thresholds):  # Search from low to high
                metric_value = all_metrics[threshold][metric_name]
                if metric_value >= target_value:
                    best_threshold = threshold
                    break
        else:  # For sensitivity and NPV
            # For sensitivity and NPV: as threshold decreases, these metrics increase
            # Start from highest threshold and find the first threshold that meets the target
            for threshold in thresholds:  # Search from high to low (thresholds are already sorted high to low)
                metric_value = all_metrics[threshold][metric_name]
                if metric_value >= target_value:
                    best_threshold = threshold
                    break
        
        # Store the best threshold and its metrics
        if best_threshold is not None:
            target_thresholds[metric_name] = best_threshold
            metrics_at_thresholds[metric_name] = all_metrics[best_threshold]
    
    # Calculate combined results for multiple target metrics
    combined_results = {}
    
    if len(target_metrics) > 1:
        # Check all thresholds to find ones that satisfy multiple metrics
        satisfied_metrics_by_threshold = {}
        
        for threshold, metrics_dict in all_metrics.items():
            satisfied_metrics = []
            
            for metric_name, target_value in target_metrics.items():
                if metric_name in ['sensitivity', 'specificity', 'ppv', 'npv']:
                    metric_value = metrics_dict[metric_name]
                    if metric_value >= target_value:
                        satisfied_metrics.append(metric_name)
            
            if len(satisfied_metrics) > 1:
                satisfied_metrics_by_threshold[threshold] = satisfied_metrics
        
        # For each combination of metrics, save all thresholds that satisfy the combination
        from itertools import combinations
        for r in range(2, len(target_metrics) + 1):
            for combo in combinations(target_metrics.keys(), r):
                combo_name = ' & '.join(combo)
                valid_thresholds = {}
                
                # Check each threshold that satisfied multiple metrics
                for threshold, satisfied in satisfied_metrics_by_threshold.items():
                    if all(metric in satisfied for metric in combo):
                        valid_thresholds[str(threshold)] = all_metrics[threshold]
                
                if valid_thresholds:
                    combined_results[combo_name] = valid_thresholds
    
    # Return results
    return {
        'thresholds': target_thresholds,
        'metrics_at_thresholds': metrics_at_thresholds,
        'combined_results': combined_results
    }

def apply_target_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                           threshold: float) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Apply a pre-determined target threshold to evaluate predictions
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        threshold (float): Threshold determined previously (usually from training data)
        
    Returns:
        Dict: Dictionary containing metrics at the provided threshold
    """
    # Calculate metrics at the given threshold
    metrics_dict = apply_threshold(y_true, y_pred_proba, threshold)
    
    # Return results with the threshold included
    return {
        'threshold': float(threshold),
        'metrics': metrics_dict
    }

def calculate_net_benefit(y_true, y_pred_proba, threshold):
    # Ensure threshold is a scalar value
    if isinstance(threshold, list):
        threshold = threshold[0]  # Or use np.mean(threshold)
    
    # Prevent division by zero error with extreme threshold values
    if threshold >= 0.999:  # If threshold is too close to 1, it can cause division by zero
        return 0.0
    
    # Convert to numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Get positive and negative predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate true positives and false positives
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    
    n = len(y_true)
    
    # Calculate net benefit
    benefit = (true_positives / n) - (false_positives / n) * (threshold / (1 - threshold))
    
    # Ensure result is a finite value
    if not np.isfinite(benefit):
        return 0.0
        
    return benefit

def spiegelhalter_z_test(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
    """
    Perform Spiegelhalter Z-test for calibration
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        
    Returns:
        Tuple[float, float]: (z-score, p-value)
    """
    n = len(y_true)
    o_minus_e = y_true - y_pred_proba
    var = y_pred_proba * (1 - y_pred_proba)
    z = np.sum(o_minus_e) / np.sqrt(np.sum(var))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

def delong_roc_ci(y_true: np.ndarray, y_pred_proba: np.ndarray, alpha: float = 0.95) -> Tuple[float, np.ndarray]:
    """
    Calculate DeLong confidence intervals for ROC curve
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        alpha (float): Confidence level
        
    Returns:
        Tuple[float, np.ndarray]: (AUC value, confidence interval)
    """
    aucs, auc_cov = delong_roc_variance(y_true, y_pred_proba)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=aucs, scale=auc_std)
    ci[ci > 1] = 1
    return aucs, ci 