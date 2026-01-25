"""
Metrics Calculation Module
Provides a registry-based system for calculating model evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List, Callable, Any, Optional
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, f1_score as sklearn_f1_score
from scipy import stats
from ..statistics.delong_test import delong_roc_variance, delong_roc_test
from ..statistics.hosmer_lemeshow_test import hosmer_lemeshow_test
from ..statistics.spiegelhalter_z_test import spiegelhalter_z_test

# --- Metric Registry System ---

METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_metric(name: str, display_name: str, category: str = 'basic'):
    """
    Decorator to register a metric function.
    
    Args:
        name: Internal unique key for the metric
        display_name: Pretty name for reports and plots
        category: 'basic', 'statistical', or 'clinical'
    """
    def decorator(func: Callable):
        METRIC_REGISTRY[name] = {
            'func': func,
            'display_name': display_name,
            'category': category
        }
        return func
    return decorator

# --- Basic Metrics Implementation ---

@register_metric('accuracy', 'Accuracy')
def calc_accuracy(y_true, y_pred, y_prob):
    return accuracy_score(y_true, y_pred)

@register_metric('sensitivity', 'Sensitivity')
def calc_sensitivity(y_true, y_pred, y_prob):
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0

@register_metric('specificity', 'Specificity')
def calc_specificity(y_true, y_pred, y_prob):
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

@register_metric('ppv', 'PPV')
def calc_ppv(y_true, y_pred, y_prob):
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0

@register_metric('npv', 'NPV')
def calc_npv(y_true, y_pred, y_prob):
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0

@register_metric('f1_score', 'F1-score')
def calc_f1(y_true, y_pred, y_prob):
    precision = calc_ppv(y_true, y_pred, y_prob)
    recall = calc_sensitivity(y_true, y_pred, y_prob)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

@register_metric('auc', 'AUC')
def calc_auc(y_true, y_pred, y_prob):
    # Support for multi-class AUC will be handled here if y_prob is 2D
    if isinstance(y_prob, np.ndarray) and y_prob.ndim > 1:
        return roc_auc_score(y_true, y_prob, multi_class='ovr')
    return roc_auc_score(y_true, y_prob)

# --- Statistical Metrics Implementation ---

@register_metric('hosmer_lemeshow_p_value', 'H-L P-value', category='statistical')
def calc_hl_p(y_true, y_pred, y_prob):
    try:
        # H-L test usually only for binary
        if y_prob.ndim > 1 and y_prob.shape[1] > 1: return np.nan
        df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_prob})
        _, p_value = hosmer_lemeshow_test(df)
        return p_value
    except:
        return np.nan

@register_metric('spiegelhalter_z_p_value', 'Spiegelhalter P-value', category='statistical')
def calc_spiegelhalter_p(y_true, y_pred, y_prob):
    try:
        if y_prob.ndim > 1 and y_prob.shape[1] > 1: return np.nan
        _, p_value = spiegelhalter_z_test(y_true, y_prob)
        return p_value
    except:
        return np.nan

from .prediction_container import PredictionContainer

# --- Main Interface ---

def calculate_metrics(y_true: Union[np.ndarray, PredictionContainer], 
                      y_pred: Optional[np.ndarray] = None, 
                      y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate all registered metrics using PredictionContainer.
    """
    if isinstance(y_true, PredictionContainer):
        container = y_true
    else:
        container = PredictionContainer(y_true, y_pred_proba, y_pred)

    metrics_dict = {}
    for name, info in METRIC_REGISTRY.items():
        try:
            # Inject container data based on what the metric function expects
            # Most of our metrics currently expect (y_true, y_pred, y_prob)
            metrics_dict[name] = info['func'](
                container.y_true, 
                container.y_pred, 
                container.get_eval_probs()
            )
        except Exception as e:
            metrics_dict[name] = np.nan
    
    return metrics_dict

def apply_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Apply a given threshold to predicted probabilities and calculate metrics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return calculate_metrics(y_true, y_pred, y_pred_proba)

def calculate_metrics_youden(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate metrics based on the optimal Youden index.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    youden_indices = tpr + (1 - fpr) - 1
    optimal_idx = np.argmax(youden_indices)
    optimal_threshold = thresholds[optimal_idx]
    
    metrics_at_threshold = apply_threshold(y_true, y_pred_proba, optimal_threshold)
    
    return {
        'threshold': float(optimal_threshold),
        'youden_index': float(youden_indices[optimal_idx]),
        'metrics': metrics_at_threshold
    }

def apply_youden_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Apply a pre-determined Youden threshold.
    """
    metrics_at_threshold = apply_threshold(y_true, y_pred_proba, threshold)
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba) # For youden index calculation if needed
    
    # Simple youden index at this specific threshold
    sens = metrics_at_threshold['sensitivity']
    spec = metrics_at_threshold['specificity']
    youden_index = sens + spec - 1
    
    return {
        'threshold': float(threshold),
        'youden_index': float(youden_index),
        'metrics': metrics_at_threshold
    }

def calculate_metrics_at_target(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               target_metrics: Dict[str, float]) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate metrics at thresholds that achieve target metric values (sensitivity, specificity, etc.)
    
    This function:
    1. Finds individual thresholds that meet each target separately
    2. Searches for combined thresholds that meet all targets simultaneously
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Sort thresholds from highest to lowest
    sorted_indices = np.argsort(thresholds)[::-1]
    thresholds = thresholds[sorted_indices]
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    
    target_thresholds = {}
    metrics_at_thresholds = {}
    
    # Find individual thresholds for each target
    for metric_name, target_value in target_metrics.items():
        best_threshold = None
        if metric_name == 'sensitivity':
            # Sensitivity increases as threshold decreases
            for i, val in enumerate(tpr):
                if val >= target_value:
                    best_threshold = thresholds[i]
                    break
        elif metric_name == 'specificity':
            # Specificity increases as threshold increases
            for i in range(len(fpr)-1, -1, -1):
                if (1 - fpr[i]) >= target_value:
                    best_threshold = thresholds[i]
                    break
        
        if best_threshold is not None:
            target_thresholds[metric_name] = best_threshold
            metrics_at_thresholds[metric_name] = apply_threshold(y_true, y_pred_proba, best_threshold)
    
    # Search for combined thresholds that meet all targets simultaneously
    combined_results = {}
    
    if len(target_metrics) > 0:
        # Iterate through all thresholds to find ones meeting all criteria
        for i, thresh in enumerate(thresholds):
            meets_all = True
            current_metrics = {}
            
            # Check sensitivity target if specified
            if 'sensitivity' in target_metrics:
                current_sensitivity = tpr[i]
                current_metrics['sensitivity'] = float(current_sensitivity)
                if current_sensitivity < target_metrics['sensitivity']:
                    meets_all = False
            
            # Check specificity target if specified
            if 'specificity' in target_metrics:
                current_specificity = 1 - fpr[i]
                current_metrics['specificity'] = float(current_specificity)
                if current_specificity < target_metrics['specificity']:
                    meets_all = False
            
            # If this threshold meets all targets, add to results
            if meets_all:
                # Calculate full metrics at this threshold
                full_metrics = apply_threshold(y_true, y_pred_proba, thresh)
                combined_key = ' & '.join(target_metrics.keys())
                
                # Store under the combined key
                if combined_key not in combined_results:
                    combined_results[combined_key] = {}
                
                combined_results[combined_key][float(thresh)] = full_metrics
    
    return {
        'thresholds': target_thresholds,
        'metrics_at_thresholds': metrics_at_thresholds,
        'combined_results': combined_results
    }

def apply_target_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                           threshold: float) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Apply a pre-determined target threshold.
    """
    metrics_at_threshold = apply_threshold(y_true, y_pred_proba, threshold)
    return {
        'threshold': float(threshold),
        'metrics': metrics_at_threshold
    }

# --- Utility Functions (Legacy Support) ---

def delong_roc_ci(y_true: np.ndarray, y_pred_proba: np.ndarray, alpha: float = 0.95) -> Tuple[float, np.ndarray]:
    """Calculate DeLong confidence intervals for ROC curve."""
    aucs, auc_cov = delong_roc_variance(y_true, y_pred_proba)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=aucs, scale=auc_std)
    ci[ci > 1] = 1
    return aucs, ci

def calculate_net_benefit(y_true, y_pred_proba, threshold):
    """Used for DCA plotting."""
    if isinstance(threshold, list): threshold = threshold[0]
    if threshold >= 0.999: return 0.0
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    
    benefit = (true_positives / n) - (false_positives / n) * (threshold / (1 - threshold))
    return benefit if np.isfinite(benefit) else 0.0

# Add other functions like calculate_metrics_youden, calculate_metrics_at_target...
# For brevity, these are assumed to remain or be refactored similarly.
